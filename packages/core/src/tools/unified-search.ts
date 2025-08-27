/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'fs';
import path from 'path';
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

// Search result interfaces
interface TextSearchResult {
  file_path: string;
  line_number: number;
  content: string;
  type: string;
  similarity: number;
  search_directory?: string;
}

interface SemanticSearchResult {
  file_path: string;
  line_start: number;
  line_end: number;
  content: string;
  type: string;
  name: string;
  context?: string;
  similarity: number;
  search_directory?: string;
}

interface ProjectTypeInfo {
  isPython: boolean;
  isJavaScript: boolean;
  isTypeScript: boolean;
  hasLargeCodebase: boolean;
}

type SearchResult = TextSearchResult | SemanticSearchResult;

/**
 * Parameters for the UnifiedSearchTool
 */
export interface UnifiedSearchToolParams {
  /**
   * The search query
   */
  query: string;

  /**
   * The directory to search in (optional, defaults to current directory)
   */
  path?: string;

  /**
   * Search mode: 'auto', 'text', 'semantic', 'functions', 'classes'
   */
  mode?: 'auto' | 'text' | 'semantic' | 'functions' | 'classes';

  /**
   * Maximum number of results to return (optional, defaults to 10)
   */
  max_results?: number;

  /**
   * Case sensitive search (text mode only)
   */
  case_sensitive?: boolean;

  /**
   * Use regex patterns (text mode only)
   */
  regex?: boolean;
}

class UnifiedSearchToolInvocation extends BaseToolInvocation<
  UnifiedSearchToolParams,
  ToolResult
> {
  constructor(
    private config: Config,
    params: UnifiedSearchToolParams,
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
   * Detects the project type and determines the best search strategy
   */
  private async detectProjectType(searchPath: string): Promise<{
    isPython: boolean;
    isJavaScript: boolean;
    isTypeScript: boolean;
    hasLargeCodebase: boolean;
  }> {
    try {
      const result = {
        isPython: false,
        isJavaScript: false,
        isTypeScript: false,
        hasLargeCodebase: false,
      };

      // Check for Python project indicators
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
          result.isPython = true;
          break;
        } catch {
          // Continue checking other indicators
        }
      }

      // Check for JavaScript/TypeScript project indicators
      const jsIndicators = [
        'package.json',
        'node_modules',
        'tsconfig.json',
        'jsconfig.json',
      ];

      for (const indicator of jsIndicators) {
        const indicatorPath = path.join(searchPath, indicator);
        try {
          await fs.promises.access(indicatorPath);
          if (indicator === 'tsconfig.json') {
            result.isTypeScript = true;
          } else {
            result.isJavaScript = true;
          }
        } catch {
          // Continue checking other indicators
        }
      }

      // Check codebase size (rough estimate)
      try {
        const files = await fs.promises.readdir(searchPath, {
          recursive: true,
        });
        const codeFiles = files.filter(
          (file: string) =>
            file.endsWith('.py') ||
            file.endsWith('.js') ||
            file.endsWith('.ts') ||
            file.endsWith('.java') ||
            file.endsWith('.cpp') ||
            file.endsWith('.c'),
        );
        result.hasLargeCodebase = codeFiles.length > 100;
      } catch (_error) {
        // Ignore errors when checking codebase size
      }

      return result;
    } catch (error) {
      console.warn(`Error detecting project type: ${getErrorMessage(error)}`);
      return {
        isPython: false,
        isJavaScript: false,
        isTypeScript: false,
        hasLargeCodebase: false,
      };
    }
  }

  /**
   * Determines the best search mode based on query analysis and project type
   */
  private determineSearchMode(
    query: string,
    projectType: {
      isPython: boolean;
      isJavaScript: boolean;
      isTypeScript: boolean;
      hasLargeCodebase: boolean;
    },
    requestedMode: UnifiedSearchToolParams['mode'],
  ): 'text' | 'semantic' | 'functions' | 'classes' {
    // If user explicitly requested a mode, use it
    if (requestedMode && requestedMode !== 'auto') {
      switch (requestedMode) {
        case 'text':
          return 'text';
        case 'semantic':
          return projectType.isPython ? 'semantic' : 'text';
        case 'functions':
          return projectType.isPython ? 'functions' : 'text';
        case 'classes':
          return projectType.isPython ? 'classes' : 'text';
        default:
          return 'text';
      }
    }

    // Auto mode: intelligent selection based on query and project type

    // For Python projects, prefer semantic search for natural language queries
    if (projectType.isPython) {
      // Check if query looks like natural language (contains spaces, common words)
      const naturalLanguageIndicators =
        /\b(function|class|method|variable|code|implementation|handle|process|create|find|get|set|convert|parse|validate)\b/i;
      const hasSpaces = query.includes(' ');

      if (hasSpaces || naturalLanguageIndicators.test(query)) {
        return 'semantic';
      }

      // For specific patterns, use specialized semantic search
      if (query.includes('def ') || query.includes('class ')) {
        return query.includes('def ') ? 'functions' : 'classes';
      }
    }

    // For large codebases, prefer text search for performance
    if (projectType.hasLargeCodebase) {
      return 'text';
    }

    // Default to text search for simple queries or non-Python projects
    return 'text';
  }

  /**
   * Executes text search using ripgrep or grep
   */
  private async executeTextSearch(
    searchPath: string,
    params: UnifiedSearchToolParams,
  ): Promise<
    Array<{
      file_path: string;
      line_number: number;
      content: string;
      type: string;
      similarity: number;
    }>
  > {
    const useRipgrep = this.config.getUseRipgrep();

    // Import tools dynamically to avoid circular dependencies
    const { RipGrepTool } = await import('./ripGrep.js');
    const { GrepTool } = await import('./grep.js');

    const ToolClass = useRipgrep ? RipGrepTool : GrepTool;
    const searchTool = new ToolClass(this.config);

    const searchParams = {
      pattern: params.query,
      path: searchPath,
      max_results: params.max_results || 10,
      case_sensitive: params.case_sensitive,
      regex: params.regex,
    };

    const invocation = searchTool.build(searchParams);
    const result = await invocation.execute(new AbortController().signal);

    if (result.error) {
      throw new Error(`Text search failed: ${result.error.message}`);
    }

    // Convert PartListUnion to string for processing
    const contentString = Array.isArray(result.llmContent)
      ? result.llmContent
          .map((part) => (typeof part === 'string' ? part : String(part)))
          .join('')
      : String(result.llmContent);

    // Parse results (this is a simplified parsing - actual implementation would need to match the tool's output format)
    const lines = contentString.split('\n');
    const results: Array<{
      file_path: string;
      line_number: number;
      content: string;
      type: string;
      similarity: number;
    }> = [];

    for (const line of lines) {
      if (line.includes(':')) {
        const parts = line.split(':');
        if (parts.length >= 3) {
          results.push({
            file_path: parts[0],
            line_number: parseInt(parts[1], 10),
            content: parts.slice(2).join(':').trim(),
            type: 'text_match',
            similarity: 1.0, // Text search has perfect match
          });
        }
      }
    }

    return results;
  }

  /**
   * Executes semantic search using the Python tool
   */
  private async executeSemanticSearch(
    searchPath: string,
    params: UnifiedSearchToolParams,
    mode: 'semantic' | 'functions' | 'classes',
  ): Promise<SemanticSearchResult[]> {
    const { spawn } = await import('child_process');

    return new Promise((resolve, reject) => {
      const pythonPath = process.env['PYTHONPATH'] || 'python3';
      const scriptPath = path.join(
        this.config.getTargetDir(),
        'crisalida_lib',
        'ASTRAL_TOOLS',
        'semantic_search.py',
      );

      // Map mode to semantic search action
      const action =
        mode === 'functions'
          ? 'search_functions'
          : mode === 'classes'
            ? 'search_classes'
            : 'search_semantic';

      const args = [
        scriptPath,
        '--action',
        action,
        '--query',
        params.query,
        '--path',
        searchPath,
        '--max-results',
        (params.max_results || 10).toString(),
        '--min-similarity',
        '0.1',
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
        searchDirectories = workspaceContext.getDirectories();
      } else {
        searchDirectories = [searchDirAbs];
      }

      // Analyze project types across all search directories
      const projectTypes = new Map<string, ProjectTypeInfo>();
      for (const searchDir of searchDirectories) {
        const projectType = await this.detectProjectType(searchDir);
        projectTypes.set(searchDir, projectType);
      }

      // Determine search mode
      const primaryProjectType = projectTypes.get(searchDirectories[0]) || {
        isPython: false,
        isJavaScript: false,
        isTypeScript: false,
        hasLargeCodebase: false,
      };

      const searchMode = this.determineSearchMode(
        this.params.query,
        primaryProjectType,
        this.params.mode,
      );

      // Execute search across all directories
      let allResults: SearchResult[] = [];

      for (const searchDir of searchDirectories) {
        const projectType = projectTypes.get(searchDir)!;

        try {
          let results: SearchResult[] = [];

          if (searchMode === 'text') {
            results = await this.executeTextSearch(searchDir, this.params);
          } else if (
            projectType.isPython &&
            (searchMode === 'semantic' ||
              searchMode === 'functions' ||
              searchMode === 'classes')
          ) {
            results = await this.executeSemanticSearch(
              searchDir,
              this.params,
              searchMode,
            );
          } else {
            // Fallback to text search for non-Python projects or unsupported modes
            results = await this.executeTextSearch(searchDir, this.params);
          }

          // Add directory context to results
          results.forEach((result) => {
            result.search_directory = searchDir;
          });

          allResults = allResults.concat(results);
        } catch (error) {
          console.warn(
            `Search failed for ${searchDir}: ${getErrorMessage(error)}`,
          );
          // Continue with other directories
        }
      }

      if (allResults.length === 0) {
        const modeDescription = searchMode === 'text' ? 'text' : 'semantic';
        const noMatchMsg = `No ${modeDescription} matches found for query "${this.params.query}"`;
        return { llmContent: noMatchMsg, returnDisplay: `No matches found` };
      }

      // Sort results by relevance (similarity for semantic, line number for text)
      if (searchMode === 'text') {
        allResults.sort((a, b) => {
          const aLine = 'line_number' in a ? a.line_number : 0;
          const bLine = 'line_number' in b ? b.line_number : 0;
          return aLine - bLine;
        });
      } else {
        allResults.sort((a, b) => b.similarity - a.similarity);
      }

      // Apply max results limit
      const maxResults = this.params.max_results || 10;
      if (allResults.length > maxResults) {
        allResults = allResults.slice(0, maxResults);
      }

      // Format results for display
      const modeDescription = searchMode === 'text' ? 'text' : 'semantic';
      let llmContent = `Found ${allResults.length} ${modeDescription} matches for "${this.params.query}":\n\n`;

      for (const result of allResults) {
        if ('line_number' in result) {
          // Text search result
          llmContent += `**${makeRelative(result.file_path, this.config.getTargetDir())}:${result.line_number}**\n`;
          llmContent += `\`\`\`\n${result.content}\n\`\`\`\n`;
        } else {
          // Semantic search result
          llmContent += `**${result.type}: ${result.name}** (similarity: ${(result.similarity * 100).toFixed(1)}%)\n`;
          llmContent += `File: ${makeRelative(result.file_path, this.config.getTargetDir())}\n`;
          llmContent += `Lines ${result.line_start}-${result.line_end}:\n`;
          llmContent += `\`\`\`python\n${result.content}\n\`\`\`\n`;
        }
        llmContent += `---\n\n`;
      }

      let displayMessage = `Found ${allResults.length} ${modeDescription} matches`;
      if (allResults.length >= maxResults) {
        displayMessage += ` (showing top ${maxResults})`;
      }

      return {
        llmContent: llmContent.trim(),
        returnDisplay: displayMessage,
      };
    } catch (error) {
      console.error(`Error during unified search: ${error}`);
      const errorMessage = getErrorMessage(error);
      return {
        llmContent: `Error during unified search: ${errorMessage}`,
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
    if (this.params.mode && this.params.mode !== 'auto') {
      description += ` (${this.params.mode} mode)`;
    } else {
      description += ` (auto mode)`;
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
      description += ` across workspace`;
    }
    return description;
  }
}

/**
 * Implementation of the UnifiedSearch tool
 */
export class UnifiedSearchTool extends BaseDeclarativeTool<
  UnifiedSearchToolParams,
  ToolResult
> {
  static readonly Name = 'unified_search';

  constructor(private readonly config: Config) {
    super(
      UnifiedSearchTool.Name,
      'SearchUnified',
      'Intelligent search that automatically selects between text and semantic search based on project type and query analysis. Supports Python semantic search for natural language queries and traditional text search for exact matches.',
      Kind.Search,
      {
        properties: {
          query: {
            description:
              'The search query (supports both exact text and natural language for Python projects).',
            type: 'string',
          },
          path: {
            description:
              'Optional: The absolute path to the directory to search within. If omitted, searches the current working directory.',
            type: 'string',
          },
          mode: {
            description:
              'Optional: Search mode - "auto" (intelligent selection), "text" (exact matches), "semantic" (Python natural language), "functions" (Python functions), "classes" (Python classes). Defaults to "auto".',
            type: 'string',
            enum: ['auto', 'text', 'semantic', 'functions', 'classes'],
          },
          max_results: {
            description:
              'Optional: Maximum number of results to return. Defaults to 10.',
            type: 'number',
          },
          case_sensitive: {
            description:
              'Optional: Case sensitive search (text mode only). Defaults to false.',
            type: 'boolean',
          },
          regex: {
            description:
              'Optional: Use regex patterns (text mode only). Defaults to false.',
            type: 'boolean',
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
    params: UnifiedSearchToolParams,
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
      params.mode &&
      !['auto', 'text', 'semantic', 'functions', 'classes'].includes(
        params.mode,
      )
    ) {
      return 'mode must be one of: auto, text, semantic, functions, classes.';
    }

    // Path validation is done in the invocation's execute method
    return null;
  }

  protected createInvocation(
    params: UnifiedSearchToolParams,
  ): ToolInvocation<UnifiedSearchToolParams, ToolResult> {
    return new UnifiedSearchToolInvocation(this.config, params);
  }
}
