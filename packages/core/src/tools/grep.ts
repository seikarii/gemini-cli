/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fsPromises from 'fs/promises';
import path from 'path';
import { EOL } from 'os';
import os from 'os';
import { spawn } from 'child_process';
import { globStream } from 'glob';
import { rgPath } from '@lvce-editor/ripgrep';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolResult,
} from './tools.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { getErrorMessage, isNodeError } from '../utils/errors.js';
import { isGitRepository } from '../utils/gitUtils.js';
import { Config } from '../config/config.js';
import { FileExclusions } from '../utils/ignorePatterns.js';
import { ToolErrorType } from './tool-error.js';

const DEFAULT_TOTAL_MAX_MATCHES = 20000;

// --- Interfaces ---

/**
 * Parameters for the GrepTool
 */
export interface GrepToolParams {
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
   * Whether to show which search strategy was used (optional, defaults to false)
   */
  show_strategy?: boolean;

  /**
   * Maximum number of matches to return (optional, defaults to 20000)
   */
  max_matches?: number;
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
  GrepToolParams,
  ToolResult
> {
  private readonly fileExclusions: FileExclusions;

  constructor(
    private readonly config: Config,
    params: GrepToolParams,
  ) {
    super(params);
    this.fileExclusions = config.getFileExclusions();
  }

  /**
   * Checks if a path is within the root directory and resolves it.
   * @param relativePath Path relative to the root directory (or undefined for root).
   * @returns The absolute path if valid and exists, or null if no path specified (to search all directories).
   * @throws {Error} If path is outside root, doesn't exist, or isn't a directory.
   */
  private async resolveAndValidatePath(
    relativePath?: string,
  ): Promise<string | null> {
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
      const fileInfo = await this.config
        .getFileSystemService()
        .getFileInfo(targetPath);
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

  async execute(signal: AbortSignal): Promise<ToolResult> {
    try {
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

      // Collect matches from all search directories
      let allMatches: GrepMatch[] = [];
      const totalMaxMatches =
        this.params.max_matches ?? DEFAULT_TOTAL_MAX_MATCHES;

      if (this.config.getDebugMode()) {
        console.log(`[GrepTool] Total result limit: ${totalMaxMatches}`);
      }

      for (const searchDir of searchDirectories) {
        const matches = await this.performGrepSearch({
          pattern: this.params.pattern,
          path: searchDir,
          include: this.params.include,
          signal,
          maxMatches: totalMaxMatches - allMatches.length,
        });

        // Add directory prefix if searching multiple directories
        if (searchDirectories.length > 1) {
          const dirName = path.basename(searchDir);
          matches.forEach((match) => {
            match.filePath = path.join(dirName, match.filePath);
          });
        }

        allMatches = allMatches.concat(matches);

        if (allMatches.length >= totalMaxMatches) {
          allMatches = allMatches.slice(0, totalMaxMatches);
          break;
        }
      }

      const wasTruncated = allMatches.length >= totalMaxMatches;

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

      // Group matches by file
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

      llmContent += `:
---
`;

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
        error: {
          message: errorMessage,
          type: ToolErrorType.GREP_EXECUTION_ERROR,
        },
      };
    }
  }

  /**
   * Checks if a command is available in the system's PATH.
   * @param {string} command The command name (e.g., 'git', 'grep').
   * @returns {Promise<boolean>} True if the command is available, false otherwise.
   */
  private isCommandAvailable(command: string): Promise<boolean> {
    return new Promise((resolve) => {
      const checkCommand = process.platform === 'win32' ? 'where' : 'command';
      const checkArgs =
        process.platform === 'win32' ? [command] : ['-v', command];
      try {
        const child = spawn(checkCommand, checkArgs, {
          stdio: 'ignore',
          shell: process.platform === 'win32',
        });
        child.on('close', (code) => resolve(code === 0));
        child.on('error', () => resolve(false));
      } catch {
        resolve(false);
      }
    });
  }

  /**
   * Parses the standard output of grep-like commands (ripgrep, git grep, system grep).
   * Expects format: filePath:lineNumber:lineContent
   * Handles colons within file paths and line content correctly.
   * Supports multiple output formats for robustness.
   * @param {string} output The raw stdout string.
   * @param {string} basePath The absolute directory the search was run from, for relative paths.
   * @returns {GrepMatch[]} Array of match objects.
   */
  private parseGrepOutput(output: string, basePath: string): GrepMatch[] {
    const results: GrepMatch[] = [];
    if (!output) return results;

    const lines = output.split(EOL); // Use OS-specific end-of-line

    for (const line of lines) {
      if (!line.trim()) continue;

      // Try multiple parsing strategies for robustness

      // Strategy 1: Standard format - filePath:lineNumber:lineContent
      let match = this.parseStandardGrepFormat(line, basePath);
      if (match) {
        results.push(match);
        continue;
      }

      // Strategy 2: Handle file paths with colons (escaped or quoted)
      match = this.parseComplexPathFormat(line, basePath);
      if (match) {
        results.push(match);
        continue;
      }

      // Strategy 3: Fallback - try to extract any line number and content
      match = this.parseFallbackFormat(line, basePath);
      if (match) {
        results.push(match);
      }
    }
    return results;
  }

  /**
   * Parses standard grep format: filePath:lineNumber:lineContent
   */
  private parseStandardGrepFormat(
    line: string,
    basePath: string,
  ): GrepMatch | null {
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
  private parseComplexPathFormat(
    line: string,
    basePath: string,
  ): GrepMatch | null {
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
  private parseFallbackFormat(
    line: string,
    basePath: string,
  ): GrepMatch | null {
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

  /**
   * Gets a description of the grep operation
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

  /**
   * Performs the actual search using the prioritized strategies.
   * @param options Search options including pattern, absolute path, and include glob.
   * @returns A promise resolving to an array of match objects.
   */
  private async performGrepSearch(options: {
    pattern: string;
    path: string; // Expects absolute path
    include?: string;
    signal: AbortSignal;
    maxMatches?: number;
  }): Promise<GrepMatch[]> {
    const { pattern, path: absolutePath, include, maxMatches } = options;
    let strategyUsed = 'none';

    try {
      // --- Strategy 1: ripgrep ---
      const ripgrepAvailable = (await this.isCommandAvailable('rg')) || rgPath;
      if (ripgrepAvailable) {
        strategyUsed = 'ripgrep';
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
        const cpuCount = os.cpus().length;
        const threadCount = Math.max(1, Math.min(cpuCount, 8)); // Cap at 8 threads
        rgArgs.push('--threads', threadCount.toString());
        rgArgs.push(absolutePath);

        try {
          const output = await new Promise<string>((resolve, reject) => {
            const child = spawn(rgPath || 'rg', rgArgs, {
              windowsHide: true,
            });
            const stdoutChunks: Buffer[] = [];
            const stderrChunks: Buffer[] = [];

            child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
            child.stderr.on('data', (chunk) => stderrChunks.push(chunk));
            child.on('error', (err) =>
              reject(new Error(`Failed to start ripgrep: ${err.message}`)),
            );
            child.on('close', (code) => {
              const stdoutData = Buffer.concat(stdoutChunks).toString('utf8');
              const stderrData = Buffer.concat(stderrChunks).toString('utf8');
              if (code === 0) resolve(stdoutData);
              else if (code === 1)
                resolve(''); // No matches
              else
                reject(
                  new Error(`ripgrep exited with code ${code}: ${stderrData}`),
                );
            });
          });
          return this.parseGrepOutput(output, absolutePath);
        } catch (ripgrepError: unknown) {
          console.debug(
            `GrepLogic: ripgrep failed: ${getErrorMessage(
              ripgrepError,
            )}. Falling back...`,
          );
        }
      }

      // --- Strategy 2: git grep ---
      const isGit = isGitRepository(absolutePath);
      const gitAvailable = isGit && (await this.isCommandAvailable('git'));

      if (gitAvailable) {
        strategyUsed = 'git grep';
        const gitArgs = [
          'grep',
          '--untracked',
          '-n',
          '-E',
          '--ignore-case',
          pattern,
        ];
        if (include) {
          gitArgs.push('--', include);
        }

        try {
          const output = await new Promise<string>((resolve, reject) => {
            const child = spawn('git', gitArgs, {
              cwd: absolutePath,
              windowsHide: true,
            });
            const stdoutChunks: Buffer[] = [];
            const stderrChunks: Buffer[] = [];

            child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
            child.stderr.on('data', (chunk) => stderrChunks.push(chunk));
            child.on('error', (err) =>
              reject(new Error(`Failed to start git grep: ${err.message}`)),
            );
            child.on('close', (code) => {
              const stdoutData = Buffer.concat(stdoutChunks).toString('utf8');
              const stderrData = Buffer.concat(stderrChunks).toString('utf8');
              if (code === 0) resolve(stdoutData);
              else if (code === 1)
                resolve(''); // No matches
              else
                reject(
                  new Error(`git grep exited with code ${code}: ${stderrData}`),
                );
            });
          });
          return this.parseGrepOutput(output, absolutePath);
        } catch (gitError: unknown) {
          console.debug(
            `GrepLogic: git grep failed: ${getErrorMessage(
              gitError,
            )}. Falling back...`,
          );
        }
      }

      // --- Strategy 3: System grep ---
      const grepAvailable = await this.isCommandAvailable('grep');
      if (grepAvailable) {
        strategyUsed = 'system grep';
        const grepArgs = ['-r', '-n', '-H', '-E'];
        // Extract directory names from exclusion patterns for grep --exclude-dir
        const globExcludes = this.fileExclusions.getGlobExcludes();
        const commonExcludes = globExcludes
          .map((pattern) => {
            let dir = pattern;
            if (dir.startsWith('**/')) {
              dir = dir.substring(3);
            }
            if (dir.endsWith('/**')) {
              dir = dir.slice(0, -3);
            } else if (dir.endsWith('/')) {
              dir = dir.slice(0, -1);
            }

            // Only consider patterns that are likely directories. This filters out file patterns.
            if (dir && !dir.includes('/') && !dir.includes('*')) {
              return dir;
            }
            return null;
          })
          .filter((dir): dir is string => !!dir);
        commonExcludes.forEach((dir) => grepArgs.push(`--exclude-dir=${dir}`));
        if (include) {
          grepArgs.push(`--include=${include}`);
        }
        grepArgs.push(pattern);
        grepArgs.push('.');

        try {
          const output = await new Promise<string>((resolve, reject) => {
            const child = spawn('grep', grepArgs, {
              cwd: absolutePath,
              windowsHide: true,
            });
            const stdoutChunks: Buffer[] = [];
            const stderrChunks: Buffer[] = [];

            const onData = (chunk: Buffer) => stdoutChunks.push(chunk);
            const onStderr = (chunk: Buffer) => {
              const stderrStr = chunk.toString();
              // Suppress common harmless stderr messages
              if (
                !stderrStr.includes('Permission denied') &&
                !/grep:.*: Is a directory/i.test(stderrStr)
              ) {
                stderrChunks.push(chunk);
              }
            };
            const onError = (err: Error) => {
              cleanup();
              reject(new Error(`Failed to start system grep: ${err.message}`));
            };
            const onClose = (code: number | null) => {
              const stdoutData = Buffer.concat(stdoutChunks).toString('utf8');
              const stderrData = Buffer.concat(stderrChunks)
                .toString('utf8')
                .trim();
              cleanup();
              if (code === 0) resolve(stdoutData);
              else if (code === 1)
                resolve(''); // No matches
              else {
                if (stderrData)
                  reject(
                    new Error(
                      `System grep exited with code ${code}: ${stderrData}`,
                    ),
                  );
                else resolve(''); // Exit code > 1 but no stderr, likely just suppressed errors
              }
            };

            const cleanup = () => {
              child.stdout.removeListener('data', onData);
              child.stderr.removeListener('data', onStderr);
              child.removeListener('error', onError);
              child.removeListener('close', onClose);
              if (child.connected) {
                child.disconnect();
              }
            };

            child.stdout.on('data', onData);
            child.stderr.on('data', onStderr);
            child.on('error', onError);
            child.on('close', onClose);
          });
          return this.parseGrepOutput(output, absolutePath);
        } catch (grepError: unknown) {
          console.debug(
            `GrepLogic: System grep failed: ${getErrorMessage(
              grepError,
            )}. Falling back...`,
          );
        }
      }

      // --- Strategy 4: Pure JavaScript Fallback ---
      console.debug(
        'GrepLogic: Falling back to JavaScript grep implementation.',
      );
      strategyUsed = 'javascript fallback';
      const globPattern = include ? include : '**/*';
      const ignorePatterns = this.fileExclusions.getGlobExcludes();

      const filesStream = globStream(globPattern, {
        cwd: absolutePath,
        dot: true,
        ignore: ignorePatterns,
        absolute: true,
        nodir: true,
        signal: options.signal,
      });

      const regex = new RegExp(pattern, 'i');
      const allMatches: GrepMatch[] = [];

      for await (const filePath of filesStream) {
        const fileAbsolutePath = filePath as string;
        try {
          const content = await fsPromises.readFile(fileAbsolutePath, 'utf8');
          const lines = content.split(/\r?\n/);
          lines.forEach((line, index) => {
            if (regex.test(line)) {
              allMatches.push({
                filePath:
                  path.relative(absolutePath, fileAbsolutePath) ||
                  path.basename(fileAbsolutePath),
                lineNumber: index + 1,
                line,
              });
            }
          });
        } catch (readError: unknown) {
          // Ignore errors like permission denied or file gone during read
          if (!isNodeError(readError) || readError.code !== 'ENOENT') {
            console.debug(
              `GrepLogic: Could not read/process ${fileAbsolutePath}: ${getErrorMessage(
                readError,
              )}`,
            );
          }
        }
      }

      return allMatches;
    } catch (error: unknown) {
      console.error(
        `GrepLogic: Error in performGrepSearch (Strategy: ${strategyUsed}): ${getErrorMessage(
          error,
        )}`,
      );
      throw error; // Re-throw
    }
  }
}

// --- GrepLogic Class ---

/**
 * Implementation of the Grep tool logic (moved from CLI)
 */
export class GrepTool extends BaseDeclarativeTool<GrepToolParams, ToolResult> {
  static readonly Name = 'search_file_content'; // Keep static name

  constructor(private readonly config: Config) {
    super(
      GrepTool.Name,
      'SearchText',
      'Searches for a regular expression pattern within the content of files in a specified directory (or current working directory). Can filter files by a glob pattern. Returns the lines containing matches, along with their file paths and line numbers. Supports multiple search strategies (ripgrep, git grep, system grep, JavaScript fallback) with automatic fallback and strategy reporting.',
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
          show_strategy: {
            description:
              'Optional: Whether to show which search strategy (ripgrep, git grep, system grep, or JavaScript fallback) was used. Defaults to false.',
            type: 'boolean',
          },
          max_matches: {
            description:
              'Optional: Maximum number of matches to return. Defaults to 20000.',
            type: 'number',
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
  protected override validateToolParamValues(
    params: GrepToolParams,
  ): string | null {
    try {
      new RegExp(params.pattern);
    } catch (error) {
      return `Invalid regular expression pattern provided: ${params.pattern}. Error: ${getErrorMessage(error)}`;
    }

    // Only validate path if one is provided
    if (params.path) {
      // Basic path validation - check if it's a reasonable path string
      // Full validation will happen during execution where we can use async operations
      if (typeof params.path !== 'string' || params.path.trim() === '') {
        return 'Path must be a non-empty string';
      }
      // Check for obviously invalid path characters
      if (params.path.includes('\0')) {
        return 'Path contains invalid null characters';
      }
    }

    return null; // Parameters are valid
  }

  protected createInvocation(
    params: GrepToolParams,
  ): ToolInvocation<GrepToolParams, ToolResult> {
    return new GrepToolInvocation(this.config, params);
  }
}
