/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolResult,
} from './tools.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { getErrorMessage } from '../utils/errors.js';
import { Config } from '../config/config.js';
import { ToolErrorType } from './tool-error.js';

// Subset of semantic search result structure
interface SemanticSearchResult {
  file_path: string;
  line_start: number;
  line_end: number;
  content: string;
  type: string;
  name: string;
  context: string;
  similarity: number;
}

/**
 * Parameters for the SemanticSearchTool
 */
export interface SemanticSearchToolParams {
  /**
   * The semantic search query
   */
  query: string;

  /**
   * The directory to search in (optional, defaults to current directory)
   */
  path?: string;

  /**
   * Type of search to perform
   */
  action?:
    | 'search_semantic'
    | 'search_functions'
    | 'search_classes'
    | 'analyze_structure';

  /**
   * Maximum number of results to return (optional, defaults to 10)
   */
  max_results?: number;

  /**
   * Minimum similarity threshold (optional, defaults to 0.1)
   */
  min_similarity?: number;
}

class SemanticSearchToolInvocation extends BaseToolInvocation<
  SemanticSearchToolParams,
  ToolResult
> {
  constructor(
    private config: Config,
    params: SemanticSearchToolParams,
  ) {
    super(params);
  }

  /**
   * Checks if a path is within the root directory and resolves it.
   */
  private async resolveAndValidatePath(
    relativePath?: string,
  ): Promise<string | null> {
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

    // Check existence and type
    try {
      const fileInfo = await this.config
        .getFileSystemService()
        .getFileInfo(targetPath);
      if (!fileInfo.success || !fileInfo.data?.isDirectory) {
        throw new Error(`Path is not a directory: ${targetPath}`);
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.message.includes('ENOENT')) {
        throw new Error(`Path does not exist: ${targetPath}`);
      }
      throw new Error(
        `Failed to access path stats for ${targetPath}: ${error}`,
      );
    }

    return targetPath;
  }

  /**
   * Detects if the given path contains a Python project
   */
  private async isPythonProject(searchPath: string): Promise<boolean> {
    try {
      // Check for common Python project indicators
      const pythonIndicators = [
        'requirements.txt',
        'pyproject.toml',
        'setup.py',
        'Pipfile',
        'poetry.lock',
        '__init__.py',
      ];

      for (const indicator of pythonIndicators) {
        const indicatorPath = path.join(searchPath, indicator);
        try {
          await fs.promises.access(indicatorPath);
          return true;
        } catch {
          // Continue checking other indicators
        }
      }

      // Check if there are any .py files in the directory
      const files = await fs.promises.readdir(searchPath);
      const hasPythonFiles = files.some((file) => file.endsWith('.py'));

      return hasPythonFiles;
    } catch (error) {
      console.warn(`Error detecting Python project: ${getErrorMessage(error)}`);
      return false;
    }
  }

  /**
   * Executes the semantic search using the Python tool
   */
  private async executeSemanticSearch(
    searchPath: string,
    params: SemanticSearchToolParams,
  ): Promise<SemanticSearchResult[]> {
    return new Promise((resolve, reject) => {
      const pythonPath = process.env['PYTHONPATH'] || 'python3';
      const scriptPath = path.join(
        this.config.getTargetDir(),
        'crisalida_lib',
        'ASTRAL_TOOLS',
        'semantic_search.py',
      );

      // Prepare arguments for the Python script
      const args = [
        scriptPath,
        '--action',
        params.action || 'search_semantic',
        '--query',
        params.query,
        '--path',
        searchPath,
        '--max-results',
        (params.max_results || 10).toString(),
        '--min-similarity',
        (params.min_similarity || 0.1).toString(),
      ];

      const child = spawn(pythonPath, args, {
        cwd: this.config.getTargetDir(),
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONPATH: path.join(this.config.getTargetDir(), 'crisalida_lib'),
        },
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        if (code === 0) {
          try {
            // Parse JSON output from Python script
            const results = JSON.parse(stdout.trim());
            resolve(results);
          } catch (parseError) {
            reject(
              new Error(
                `Failed to parse semantic search results: ${parseError}`,
              ),
            );
          }
        } else {
          reject(
            new Error(`Semantic search failed with code ${code}: ${stderr}`),
          );
        }
      });

      child.on('error', (error) => {
        reject(new Error(`Failed to start semantic search: ${error.message}`));
      });
    });
  }

  async execute(_signal: AbortSignal): Promise<ToolResult> {
    try {
      const workspaceContext = this.config.getWorkspaceContext();
      const searchDirAbs = await this.resolveAndValidatePath(this.params.path);

      // Determine which directories to search
      let searchDirectories: readonly string[];
      if (searchDirAbs === null) {
        // No path specified - search all workspace directories
        searchDirectories = workspaceContext.getDirectories();
      } else {
        // Specific path provided - search only that directory
        searchDirectories = [searchDirAbs];
      }

      // Check if any of the search directories contain Python projects
      let hasPythonProject = false;
      for (const searchDir of searchDirectories) {
        if (await this.isPythonProject(searchDir)) {
          hasPythonProject = true;
          break;
        }
      }

      if (!hasPythonProject) {
        return {
          llmContent: `Semantic search is only available for Python projects. No Python files or project indicators found in the search path.`,
          returnDisplay: `Not a Python project`,
        };
      }

      // Perform semantic search across all Python projects found
      let allResults: SemanticSearchResult[] = [];

      for (const searchDir of searchDirectories) {
        if (!(await this.isPythonProject(searchDir))) {
          continue; // Skip non-Python directories
        }

        try {
          const results = await this.executeSemanticSearch(
            searchDir,
            this.params,
          );
          allResults = allResults.concat(results);
        } catch (error) {
          console.warn(
            `Semantic search failed for ${searchDir}: ${getErrorMessage(error)}`,
          );
          // Continue with other directories
        }
      }

      if (allResults.length === 0) {
        const noMatchMsg = `No semantic matches found for query "${this.params.query}" in Python files`;
        return { llmContent: noMatchMsg, returnDisplay: `No matches found` };
      }

      // Sort by similarity (highest first)
      allResults.sort((a, b) => b.similarity - a.similarity);

      // Apply max results limit
      const maxResults = this.params.max_results || 10;
      if (allResults.length > maxResults) {
        allResults = allResults.slice(0, maxResults);
      }

      // Format results for display
      let llmContent = `Found ${allResults.length} semantic matches for "${this.params.query}" in Python code:\n\n`;

      for (const result of allResults) {
        llmContent += `**${result.type}: ${result.name}** (similarity: ${(result.similarity * 100).toFixed(1)}%)\n`;
        llmContent += `File: ${makeRelative(result.file_path, this.config.getTargetDir())}\n`;
        llmContent += `Lines ${result.line_start}-${result.line_end}:\n`;
        llmContent += `\`\`\`python\n${result.content}\n\`\`\`\n`;

        if (result.context && result.context !== result.content) {
          llmContent += `Context:\n\`\`\`python\n${result.context}\n\`\`\`\n`;
        }

        llmContent += `---\n\n`;
      }

      let displayMessage = `Found ${allResults.length} semantic matches`;
      if (allResults.length >= maxResults) {
        displayMessage += ` (showing top ${maxResults})`;
      }

      return {
        llmContent: llmContent.trim(),
        returnDisplay: displayMessage,
      };
    } catch (error) {
      console.error(`Error during semantic search: ${error}`);
      const errorMessage = getErrorMessage(error);
      return {
        llmContent: `Error during semantic search: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.LS_EXECUTION_ERROR,
        },
      };
    }
  }

  getDescription(): string {
    let description = `'${this.params.query}'`;
    if (this.params.action && this.params.action !== 'search_semantic') {
      description += ` (${this.params.action})`;
    }
    if (this.params.path) {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.path,
      );
      const relativePath = makeRelative(
        resolvedPath,
        this.config.getTargetDir(),
      );
      description += ` within ${shortenPath(relativePath)}`;
    } else {
      description += ` across Python files in workspace`;
    }
    return description;
  }
}

/**
 * Implementation of the SemanticSearch tool
 */
export class SemanticSearchTool extends BaseDeclarativeTool<
  SemanticSearchToolParams,
  ToolResult
> {
  static readonly Name = 'semantic_search';

  constructor(private readonly config: Config) {
    super(
      SemanticSearchTool.Name,
      'SearchSemantic',
      'Performs semantic search and analysis in Python codebases using advanced NLP techniques. Can find functions, classes, and code patterns by meaning rather than exact text matches. Only works with Python projects.',
      Kind.Search,
      {
        properties: {
          query: {
            description:
              "The semantic search query (e.g., 'function that handles user authentication', 'class for database connection').",
            type: 'string',
          },
          path: {
            description:
              'Optional: The absolute path to the directory to search within. If omitted, searches the current working directory.',
            type: 'string',
          },
          action: {
            description:
              'Optional: Type of search - "search_semantic" (general), "search_functions", "search_classes", or "analyze_structure". Defaults to "search_semantic".',
            type: 'string',
            enum: [
              'search_semantic',
              'search_functions',
              'search_classes',
              'analyze_structure',
            ],
          },
          max_results: {
            description:
              'Optional: Maximum number of results to return. Defaults to 10.',
            type: 'number',
          },
          min_similarity: {
            description:
              'Optional: Minimum similarity threshold (0-1). Defaults to 0.1.',
            type: 'number',
          },
        },
        required: ['query'],
        type: 'object',
      },
    );
  }

  /**
   * Validates the parameters for the tool
   */
  protected override validateToolParamValues(
    params: SemanticSearchToolParams,
  ): string | null {
    if (
      !params.query ||
      typeof params.query !== 'string' ||
      params.query.trim() === ''
    ) {
      return "The 'query' parameter cannot be empty.";
    }

    if (
      params.max_results &&
      (params.max_results < 1 || params.max_results > 100)
    ) {
      return 'max_results must be between 1 and 100.';
    }

    if (
      params.min_similarity &&
      (params.min_similarity < 0 || params.min_similarity > 1)
    ) {
      return 'min_similarity must be between 0 and 1.';
    }

    if (
      params.action &&
      ![
        'search_semantic',
        'search_functions',
        'search_classes',
        'analyze_structure',
      ].includes(params.action)
    ) {
      return 'action must be one of: search_semantic, search_functions, search_classes, analyze_structure.';
    }

    // Path validation is done in the invocation's execute method
    return null;
  }

  protected createInvocation(
    params: SemanticSearchToolParams,
  ): ToolInvocation<SemanticSearchToolParams, ToolResult> {
    return new SemanticSearchToolInvocation(this.config, params);
  }
}
