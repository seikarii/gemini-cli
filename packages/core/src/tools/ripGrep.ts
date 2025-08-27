/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';
import { EOL } from 'os';
import { spawn } from 'child_process';
import { rgPath } from '@lvce-editor/ripgrep';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolResult,
} from './tools.js';
import { SchemaValidator } from '../utils/schemaValidator.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { getErrorMessage, isNodeError } from '../utils/errors.js';
import { Config } from '../config/config.js';
import { FileExclusions } from '../utils/ignorePatterns.js';
import { ToolErrorType } from './tool-error.js';

const DEFAULT_TOTAL_MAX_MATCHES = 20000;

/**
 * Parameters for the GrepTool
 */
export interface RipGrepToolParams {
  /**
   * The regular expression pattern to search for in file contents
   */
  pattern: string;

  /**
   * The directory to search in (optional, defaults to current directory relative to root)
   */
  path?: string;

  /**
   * File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")
   */
  include?: string;

  /**
   * Maximum number of matches to return (optional, defaults to 20000)
   */
  max_matches?: number;

  /**
   * Whether to show dependency check warnings (optional, defaults to false)
   */
  show_dependency_warnings?: boolean;
}

/**
 * Result object for a single grep match
 */
interface GrepMatch {
  filePath: string;
  lineNumber: number;
  line: string;
}

class GrepToolInvocation extends BaseToolInvocation<
  RipGrepToolParams,
  ToolResult
> {
  private readonly fileExclusions: FileExclusions;

  constructor(
    private readonly config: Config,
    params: RipGrepToolParams,
  ) {
    super(params);
    this.fileExclusions = new FileExclusions(config);
  }

  /**
   * Checks if a path is within the root directory and resolves it.
   * @param relativePath Path relative to the root directory (or undefined for root).
   * @returns The absolute path if valid and exists, or null if no path specified (to search all directories).
   * @throws {Error} If path is outside root, doesn't exist, or isn't a directory.
   */
  private async resolveAndValidatePath(relativePath?: string): Promise<string | null> {
    // If no path specified, return null to indicate searching all workspace directories
    if (!relativePath) {
      return null;
    }

    const targetPath = path.resolve(this.config.getTargetDir(), relativePath);

    // Security Check: Ensure the resolved path is within workspace boundaries
    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(targetPath)) {
      const directories = workspaceContext.getDirectories();
      throw new Error(
        `Path validation failed: Attempted path "${relativePath}" resolves outside the allowed workspace directories: ${directories.join(', ')}`,
      );
    }

    // Check existence and type after resolving
    try {
      const fileInfo = await this.config.getFileSystemService().getFileInfo(targetPath);
      if (!fileInfo.success || !fileInfo.data?.isDirectory) {
        throw new Error(`Path is not a directory: ${targetPath}`);
      }
    } catch (error: unknown) {
      if (isNodeError(error) && error.code !== 'ENOENT') {
        throw new Error(`Path does not exist: ${targetPath}`);
      }
      throw new Error(
        `Failed to access path stats for ${targetPath}: ${error}`,
      );
    }

    return targetPath;
  }

  /**
   * Checks if ripgrep dependency is available and executable
   */
  private async checkRipgrepDependency(): Promise<{ available: boolean; message: string }> {
    try {
      // Check if rgPath is defined
      if (!rgPath) {
        return {
          available: false,
          message: 'ripgrep executable path is not available. Please ensure @lvce-editor/ripgrep is properly installed.',
        };
      }

      // Check if the executable exists and is accessible
      const fs = await import('fs/promises');
      try {
        await fs.access(rgPath);
      } catch (_error) {
        return {
          available: false,
          message: `ripgrep executable not found at ${rgPath}. Please reinstall @lvce-editor/ripgrep.`,
        };
      }

      // Try to execute ripgrep with a simple command to verify it's working
      const { spawn } = await import('child_process');
      return new Promise((resolve) => {
        const child = spawn(rgPath, ['--version'], {
          stdio: 'ignore',
          windowsHide: true,
        });

        child.on('close', (code) => {
          if (code === 0) {
            resolve({ available: true, message: 'ripgrep is available and working' });
          } else {
            resolve({
              available: false,
              message: `ripgrep executable exists but returned exit code ${code}. Please check the installation.`,
            });
          }
        });

        child.on('error', (err) => {
          resolve({
            available: false,
            message: `Failed to execute ripgrep: ${err.message}. Please check the installation.`,
          });
        });
      });
    } catch (error) {
      return {
        available: false,
        message: `Error checking ripgrep dependency: ${getErrorMessage(error)}`,
      };
    }
  }

  async execute(signal: AbortSignal): Promise<ToolResult> {
    try {
      // Check ripgrep dependency proactively
      const dependencyCheck = await this.checkRipgrepDependency();
      if (!dependencyCheck.available) {
        if (this.params.show_dependency_warnings) {
          console.warn(`Ripgrep dependency warning: ${dependencyCheck.message}`);
        }
        return {
          llmContent: `Error: ${dependencyCheck.message}`,
          returnDisplay: `Dependency Error`,
          error: {
            message: dependencyCheck.message,
            type: ToolErrorType.LS_EXECUTION_ERROR,
          },
        };
      }

      const workspaceContext = this.config.getWorkspaceContext();
      const searchDirAbs = await this.resolveAndValidatePath(this.params.path);
      const searchDirDisplay = this.params.path || '.';

      // Determine which directories to search
      let searchDirectories: readonly string[];
      if (searchDirAbs === null) {
        // No path specified - search all workspace directories
        searchDirectories = workspaceContext.getDirectories();
      } else {
        // Specific path provided - search only that directory
        searchDirectories = [searchDirAbs];
      }

      let allMatches: GrepMatch[] = [];
      const totalMaxMatches = this.params.max_matches ?? DEFAULT_TOTAL_MAX_MATCHES;

      if (this.config.getDebugMode()) {
        console.log(`[RipGrepTool] Total result limit: ${totalMaxMatches}`);
      }

      for (const searchDir of searchDirectories) {
        const searchResult = await this.performRipgrepSearch({
          pattern: this.params.pattern,
          path: searchDir,
          include: this.params.include,
          signal,
          maxMatches: totalMaxMatches - allMatches.length,
        });

        if (searchDirectories.length > 1) {
          const dirName = path.basename(searchDir);
          searchResult.forEach((match) => {
            match.filePath = path.join(dirName, match.filePath);
          });
        }

        allMatches = allMatches.concat(searchResult);

        if (allMatches.length >= totalMaxMatches) {
          allMatches = allMatches.slice(0, totalMaxMatches);
          break;
        }
      }

      let searchLocationDescription: string;
      if (searchDirAbs === null) {
        const numDirs = workspaceContext.getDirectories().length;
        searchLocationDescription =
          numDirs > 1
            ? `across ${numDirs} workspace directories`
            : `in the workspace directory`;
      } else {
        searchLocationDescription = `in path "${searchDirDisplay}"`;
      }

      if (allMatches.length === 0) {
        const noMatchMsg = `No matches found for pattern "${this.params.pattern}" ${searchLocationDescription}${this.params.include ? ` (filter: "${this.params.include}")` : ''}.`;
        return { llmContent: noMatchMsg, returnDisplay: `No matches found` };
      }

      const wasTruncated = allMatches.length >= totalMaxMatches;

      const matchesByFile = allMatches.reduce(
        (acc, match) => {
          const fileKey = match.filePath;
          if (!acc[fileKey]) {
            acc[fileKey] = [];
          }
          acc[fileKey].push(match);
          acc[fileKey].sort((a, b) => a.lineNumber - b.lineNumber);
          return acc;
        },
        {} as Record<string, GrepMatch[]>,
      );

      const matchCount = allMatches.length;
      const matchTerm = matchCount === 1 ? 'match' : 'matches';

      let llmContent = `Found ${matchCount} ${matchTerm} for pattern "${this.params.pattern}" ${searchLocationDescription}${this.params.include ? ` (filter: "${this.params.include}")` : ''}`;

      if (wasTruncated) {
        llmContent += ` (results limited to ${totalMaxMatches} matches for performance)`;
      }

      llmContent += `:\n---\n`;

      for (const filePath in matchesByFile) {
        llmContent += `File: ${filePath}\n`;
        matchesByFile[filePath].forEach((match) => {
          const trimmedLine = match.line.trim();
          llmContent += `L${match.lineNumber}: ${trimmedLine}\n`;
        });
        llmContent += '---\n';
      }

      let displayMessage = `Found ${matchCount} ${matchTerm}`;
      if (wasTruncated) {
        displayMessage += ` (limited)`;
      }

      return {
        llmContent: llmContent.trim(),
        returnDisplay: displayMessage,
      };
    } catch (error) {
      console.error(`Error during GrepLogic execution: ${error}`);
      const errorMessage = getErrorMessage(error);
      return {
        llmContent: `Error during grep search operation: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
      };
    }
  }

  private parseRipgrepOutput(output: string, basePath: string): GrepMatch[] {
    const results: GrepMatch[] = [];
    if (!output) return results;

    const lines = output.split(EOL);

    for (const line of lines) {
      if (!line.trim()) continue;

      // Try multiple parsing strategies for robustness

      // Strategy 1: Standard format - filePath:lineNumber:lineContent
      let match = this.parseStandardRipgrepFormat(line, basePath);
      if (match) {
        results.push(match);
        continue;
      }

      // Strategy 2: Handle file paths with colons (escaped or quoted)
      match = this.parseComplexPathRipgrepFormat(line, basePath);
      if (match) {
        results.push(match);
        continue;
      }

      // Strategy 3: Fallback - try to extract any line number and content
      match = this.parseFallbackRipgrepFormat(line, basePath);
      if (match) {
        results.push(match);
      }
    }
    return results;
  }

  /**
   * Parses standard ripgrep format: filePath:lineNumber:lineContent
   */
  private parseStandardRipgrepFormat(line: string, basePath: string): GrepMatch | null {
    // Find the index of the first colon.
    const firstColonIndex = line.indexOf(':');
    if (firstColonIndex === -1) return null;

    // Find the index of the second colon, searching *after* the first one.
    const secondColonIndex = line.indexOf(':', firstColonIndex + 1);
    if (secondColonIndex === -1) return null;

    // Extract parts based on the found colon indices
    const filePathRaw = line.substring(0, firstColonIndex);
    const lineNumberStr = line.substring(firstColonIndex + 1, secondColonIndex);
    const lineContent = line.substring(secondColonIndex + 1);

    const lineNumber = parseInt(lineNumberStr, 10);
    if (isNaN(lineNumber)) return null;

    const absoluteFilePath = path.resolve(basePath, filePathRaw);
    const relativeFilePath = path.relative(basePath, absoluteFilePath);

    return {
      filePath: relativeFilePath || path.basename(absoluteFilePath),
      lineNumber,
      line: lineContent,
    };
  }

  /**
   * Parses complex path formats with escaped colons or quotes
   */
  private parseComplexPathRipgrepFormat(line: string, basePath: string): GrepMatch | null {
    // Handle quoted file paths
    const quotedMatch = line.match(/^"([^"]+)":(\d+):(.*)$/);
    if (quotedMatch) {
      const [, filePathRaw, lineNumberStr, lineContent] = quotedMatch;
      const lineNumber = parseInt(lineNumberStr, 10);
      if (isNaN(lineNumber)) return null;

      const absoluteFilePath = path.resolve(basePath, filePathRaw);
      const relativeFilePath = path.relative(basePath, absoluteFilePath);

      return {
        filePath: relativeFilePath || path.basename(absoluteFilePath),
        lineNumber,
        line: lineContent,
      };
    }

    // Handle escaped colons in file paths
    const parts = line.split(':');
    if (parts.length < 3) return null;

    // Try to find the line number by looking for a numeric part
    for (let i = 1; i < parts.length - 1; i++) {
      const potentialLineNumber = parseInt(parts[i], 10);
      if (!isNaN(potentialLineNumber)) {
        const filePathRaw = parts.slice(0, i).join(':');
        const lineContent = parts.slice(i + 1).join(':');

        const absoluteFilePath = path.resolve(basePath, filePathRaw);
        const relativeFilePath = path.relative(basePath, absoluteFilePath);

        return {
          filePath: relativeFilePath || path.basename(absoluteFilePath),
          lineNumber: potentialLineNumber,
          line: lineContent,
        };
      }
    }

    return null;
  }

  /**
   * Fallback parser for unrecognized formats
   */
  private parseFallbackRipgrepFormat(line: string, basePath: string): GrepMatch | null {
    // Look for any pattern that might contain a line number
    const lineNumberMatch = line.match(/:(\d+):/);
    if (!lineNumberMatch) return null;

    const lineNumber = parseInt(lineNumberMatch[1], 10);
    if (isNaN(lineNumber)) return null;

    // Everything before the line number is the file path
    const filePathEnd = lineNumberMatch.index!;
    const filePathRaw = line.substring(0, filePathEnd);
    const lineContent = line.substring(filePathEnd + lineNumberMatch[0].length);

    const absoluteFilePath = path.resolve(basePath, filePathRaw);
    const relativeFilePath = path.relative(basePath, absoluteFilePath);

    return {
      filePath: relativeFilePath || path.basename(absoluteFilePath),
      lineNumber,
      line: lineContent,
    };
  }

  private async performRipgrepSearch(options: {
    pattern: string;
    path: string;
    include?: string;
    signal: AbortSignal;
    maxMatches?: number;
  }): Promise<GrepMatch[]> {
    const { pattern, path: absolutePath, include, maxMatches } = options;

    const rgArgs = [
      '--line-number',
      '--no-heading',
      '--with-filename',
      '--ignore-case',
      '--regexp',
      pattern,
    ];

    if (maxMatches) {
      rgArgs.push('--max-count', maxMatches.toString());
    }

    if (include) {
      rgArgs.push('--glob', include);
    }

    // Use FileExclusions for proper exclusion patterns
    const excludePatterns = this.fileExclusions.getGlobExcludes();
    excludePatterns.forEach((exclude) => {
      rgArgs.push('--glob', `!${exclude}`);
    });

    // Dynamic threading based on CPU cores
    const os = await import('os');
    const cpuCount = os.cpus().length;
    const threadCount = Math.max(1, Math.min(cpuCount, 8)); // Cap at 8 threads
    rgArgs.push('--threads', threadCount.toString());
    rgArgs.push(absolutePath);

    try {
      const output = await new Promise<string>((resolve, reject) => {
        const child = spawn(rgPath, rgArgs, {
          windowsHide: true,
        });

        const stdoutChunks: Buffer[] = [];
        const stderrChunks: Buffer[] = [];

        const cleanup = () => {
          if (options.signal.aborted) {
            child.kill();
          }
        };

        options.signal.addEventListener('abort', cleanup, { once: true });

        child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
        child.stderr.on('data', (chunk) => stderrChunks.push(chunk));

        child.on('error', (err) => {
          options.signal.removeEventListener('abort', cleanup);
          reject(
            new Error(
              `Failed to start ripgrep: ${err.message}. Please ensure @lvce-editor/ripgrep is properly installed.`,
            ),
          );
        });

        child.on('close', (code) => {
          options.signal.removeEventListener('abort', cleanup);
          const stdoutData = Buffer.concat(stdoutChunks).toString('utf8');
          const stderrData = Buffer.concat(stderrChunks).toString('utf8');

          if (code === 0) {
            resolve(stdoutData);
          } else if (code === 1) {
            resolve(''); // No matches found
          } else {
            reject(
              new Error(`ripgrep exited with code ${code}: ${stderrData}`),
            );
          }
        });
      });

      return this.parseRipgrepOutput(output, absolutePath);
    } catch (error: unknown) {
      console.error(`GrepLogic: ripgrep failed: ${getErrorMessage(error)}`);
      throw error;
    }
  }

  /**
   * Gets a description of the grep operation
   * @param params Parameters for the grep operation
   * @returns A string describing the grep
   */
  getDescription(): string {
    let description = `'${this.params.pattern}'`;
    if (this.params.include) {
      description += ` in ${this.params.include}`;
    }
    if (this.params.path) {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.path,
      );
      if (
        resolvedPath === this.config.getTargetDir() ||
        this.params.path === '.'
      ) {
        description += ` within ./`;
      } else {
        const relativePath = makeRelative(
          resolvedPath,
          this.config.getTargetDir(),
        );
        description += ` within ${shortenPath(relativePath)}`;
      }
    } else {
      // When no path is specified, indicate searching all workspace directories
      const workspaceContext = this.config.getWorkspaceContext();
      const directories = workspaceContext.getDirectories();
      if (directories.length > 1) {
        description += ` across all workspace directories`;
      }
    }
    return description;
  }
}

/**
 * Implementation of the Grep tool logic (moved from CLI)
 */
export class RipGrepTool extends BaseDeclarativeTool<
  RipGrepToolParams,
  ToolResult
> {
  static readonly Name = 'search_file_content';

  constructor(private readonly config: Config) {
    super(
      RipGrepTool.Name,
      'SearchText',
      'Searches for a regular expression pattern within the content of files in a specified directory (or current working directory). Can filter files by a glob pattern. Returns the lines containing matches, along with their file paths and line numbers. Uses ripgrep for high-performance searching with configurable limits and dynamic threading.',
      Kind.Search,
      {
        properties: {
          pattern: {
            description:
              "The regular expression (regex) pattern to search for within file contents (e.g., 'function\\s+myFunction', 'import\\s+\\{.*\\}\\s+from\\s+.*').",
            type: 'string',
          },
          path: {
            description:
              'Optional: The absolute path to the directory to search within. If omitted, searches the current working directory.',
            type: 'string',
          },
          include: {
            description:
              "Optional: A glob pattern to filter which files are searched (e.g., '*.js', '*.{ts,tsx}', 'src/**'). If omitted, searches all files (respecting potential global ignores).",
            type: 'string',
          },
          max_matches: {
            description:
              'Optional: Maximum number of matches to return. Defaults to 20000.',
            type: 'number',
          },
          show_dependency_warnings: {
            description:
              'Optional: Whether to show warnings about ripgrep dependency availability. Defaults to false.',
            type: 'boolean',
          },
        },
        required: ['pattern'],
        type: 'object',
      },
    );
  }

  /**
   * Validates the parameters for the tool
   * @param params Parameters to validate
   * @returns An error message string if invalid, null otherwise
   */
  override validateToolParams(params: RipGrepToolParams): string | null {
    const errors = SchemaValidator.validate(
      this.schema.parameters,
      params,
    );
    if (errors) {
      return errors;
    }

    // Path validation is done in the invocation's execute method
    return null; // Parameters are valid
  }

  protected createInvocation(
    params: RipGrepToolParams,
  ): ToolInvocation<RipGrepToolParams, ToolResult> {
    return new GrepToolInvocation(this.config, params);
  }
}
