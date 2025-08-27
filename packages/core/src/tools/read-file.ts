/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';
// import fs from 'fs/promises';
import { makeRelative, shortenPath } from '../utils/paths.js';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolLocation,
  ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';

import { PartListUnion, PartUnion } from '@google/genai';
import {
  processSingleFileContent,
  getSpecificMimeType,
} from '../utils/fileUtils.js';
import { Config } from '../config/config.js';
import { FileOperation } from '../telemetry/metrics.js';
import { getProgrammingLanguage } from '../telemetry/telemetry-utils.js';
import { logFileOperation } from '../telemetry/loggers.js';
import { FileOperationEvent } from '../telemetry/types.js';

import fetch from 'node-fetch';

// Import AST analysis tools
import {
  readAndParseFile,
  type ParseResult,
  type Intentions,
} from '../ast/parser.js';
import { findNodes } from '../ast/finder.js';
import { SourceFile, SyntaxKind } from 'ts-morph';

/**
 * Parameters for the ReadFile tool
 */
export interface ReadFileToolParams {
  /**
   * The absolute path to the file to read
   */
  absolute_path: string;

  /**
   * The line number to start reading from (optional)
   */
  offset?: number;

  /**
   * The number of lines to read (optional)
   */
  limit?: number;

  /**
   * Whether to include AST analysis for supported file types (default: true)
   */
  include_ast?: boolean;

  /**
   * AST query to find specific nodes (XPath-like syntax or custom query)
   */
  ast_query?: string;

  /**
   * Whether to show detailed AST tree structure (default: true)
   */
  show_ast_tree?: boolean;
}

interface MewServerInfo {
  port: number;
  lastChecked: number;
  isAvailable: boolean;
}

interface ASTTreeNode {
  kind: string;
  name?: string;
  text?: string;
  line?: number;
  children?: ASTTreeNode[];
}

class ReadFileToolInvocation extends BaseToolInvocation<
  ReadFileToolParams,
  ToolResult
> {
  private static mewServerCache: MewServerInfo | null = null;
  private static readonly CACHE_DURATION = 30000; // 30 seconds
  private static readonly DEFAULT_PORTS = [3000, 3001, 3002, 3003, 8080, 8081];
  private static readonly EXTENDED_PORTS = Array.from(
    { length: 7000 },
    (_, i) => i + 3000,
  ); // 3000-9999
  private static readonly FAST_TIMEOUT = 1000; // 1 second for initial checks
  private static readonly SLOW_TIMEOUT = 3000; // 3 seconds for comprehensive checks
  private static readonly MAX_CONCURRENT_CHECKS = 10; // Check 10 ports simultaneously
  private static readonly MAX_AST_CONTENT_SIZE = 200 * 1024; // 200KB limit for AST content to prevent API issues

  constructor(
    private config: Config,
    params: ReadFileToolParams,
  ) {
    super(params);
  }

  getDescription(): string {
    const relativePath = makeRelative(
      this.params.absolute_path,
      this.config.getTargetDir(),
    );
    return shortenPath(relativePath);
  }

  override toolLocations(): ToolLocation[] {
    return [{ path: this.params.absolute_path, line: this.params.offset }];
  }

  private async findMewServerPort(): Promise<number | null> {
    const now = Date.now();

    // Check cache first
    if (
      ReadFileToolInvocation.mewServerCache &&
      now - ReadFileToolInvocation.mewServerCache.lastChecked <
        ReadFileToolInvocation.CACHE_DURATION &&
      ReadFileToolInvocation.mewServerCache.isAvailable
    ) {
      return ReadFileToolInvocation.mewServerCache.port;
    }

    // Try to read port from file first (most reliable method)
    try {
      const portFilePath = path.join(
        this.config.getTargetDir(),
        '.gemini',
        'mew_port.txt',
      );
      const portResult = await this.config
        .getFileSystemService()
        .readTextFile(portFilePath);
      if (!portResult.success || typeof portResult.data !== 'string')
        throw new Error(portResult.error || 'Error reading port file');
      const portStr = portResult.data;
      const parsedPort = parseInt(portStr.trim(), 10);
      if (!isNaN(parsedPort) && parsedPort > 0 && parsedPort < 65536) {
        const isAvailable = await this.checkPortAvailability(
          parsedPort,
          ReadFileToolInvocation.FAST_TIMEOUT,
        );
        if (isAvailable) {
          ReadFileToolInvocation.mewServerCache = {
            port: parsedPort,
            lastChecked: now,
            isAvailable: true,
          };
          console.log(
            `[read_file] Found Mew server on port from file: ${parsedPort}`,
          );
          return parsedPort;
        }
      }
    } catch (_error) {
      // File doesn't exist or couldn't be read, continue with port scanning
    }

    // Fast scan of default ports first
    console.log('[read_file] Scanning default ports...');
    for (const port of ReadFileToolInvocation.DEFAULT_PORTS) {
      const isAvailable = await this.checkPortAvailability(
        port,
        ReadFileToolInvocation.FAST_TIMEOUT,
      );
      if (isAvailable) {
        ReadFileToolInvocation.mewServerCache = {
          port,
          lastChecked: now,
          isAvailable: true,
        };
        console.log(`[read_file] Found Mew server on default port: ${port}`);
        return port;
      }
    }

    // Comprehensive scan of extended port range with parallel checking
    console.log('[read_file] Performing comprehensive port scan...');
    const port = await this.scanPortsComprehensively();
    if (port) {
      ReadFileToolInvocation.mewServerCache = {
        port,
        lastChecked: now,
        isAvailable: true,
      };
      console.log(`[read_file] Found Mew server on scanned port: ${port}`);
      return port;
    }

    // Update cache to indicate no server found
    ReadFileToolInvocation.mewServerCache = {
      port: 0,
      lastChecked: now,
      isAvailable: false,
    };

    console.log('[read_file] No Mew server found on any scanned ports');
    return null;
  }

  private async checkPortAvailability(
    port: number,
    timeout: number = ReadFileToolInvocation.SLOW_TIMEOUT,
  ): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(`http://localhost:${port}/api/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: { 'User-Agent': 'gemini-cli-read-file' },
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (_error) {
      return false;
    }
  }

  private async scanPortsComprehensively(): Promise<number | null> {
    // Check ports in batches to avoid overwhelming the system
    const ports = ReadFileToolInvocation.EXTENDED_PORTS;

    for (
      let i = 0;
      i < ports.length;
      i += ReadFileToolInvocation.MAX_CONCURRENT_CHECKS
    ) {
      const batch = ports.slice(
        i,
        i + ReadFileToolInvocation.MAX_CONCURRENT_CHECKS,
      );

      // Check all ports in this batch concurrently
      const promises = batch.map(async (port: number) => {
        const isAvailable = await this.checkPortAvailability(
          port,
          ReadFileToolInvocation.SLOW_TIMEOUT,
        );
        return isAvailable ? port : null;
      });

      try {
        const results = await Promise.all(promises);
        const foundPort = results.find((port: number | null) => port !== null);

        if (foundPort) {
          return foundPort;
        }
      } catch (error) {
        // Continue with next batch if this batch fails
        console.log(
          `[read_file] Error checking port batch ${i / ReadFileToolInvocation.MAX_CONCURRENT_CHECKS + 1}:`,
          error,
        );
      }

      // Small delay between batches to avoid overwhelming
      if (i + ReadFileToolInvocation.MAX_CONCURRENT_CHECKS < ports.length) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }

    return null;
  }

  private async updateMewWindow(filePath: string): Promise<void> {
    try {
      const port = await this.findMewServerPort();
      if (!port) {
        console.log(
          '[read_file] No Mew server available, skipping window update',
        );
        return;
      }

      const url = `http://localhost:${port}/api/mew/set-active-file`;
      const body = JSON.stringify({
        filePath,
        timestamp: Date.now(),
        source: 'gemini-cli-read-file',
      });

      console.log(`[read_file] Updating Mew window with file: ${filePath}`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'gemini-cli-read-file',
        },
        body,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(
          `[read_file] Mew server error (${response.status}): ${errorText}`,
        );
        // Invalidate cache on error
        ReadFileToolInvocation.mewServerCache = null;
      } else {
        console.log(
          `[read_file] Successfully updated Mew window for: ${filePath}`,
        );
      }
    } catch (error) {
      console.error('[read_file] Failed to update Mew Window:', error);
      // Invalidate cache on error
      ReadFileToolInvocation.mewServerCache = null;
    }
  }

  private isCodeFile(filePath: string): boolean {
    const codeExtensions = [
      '.ts',
      '.tsx',
      '.js',
      '.jsx',
      '.mjs',
      '.cjs',
      '.py',
      '.java',
      '.cpp',
      '.c',
      '.h',
      '.cs',
      '.php',
      '.rb',
      '.go',
      '.rs',
      '.swift',
      '.kt',
      '.scala',
      '.sh',
      '.bash',
      '.ps1',
      '.vue',
      '.svelte',
      '.astro',
      '.json',
      '.yaml',
      '.yml',
      '.xml',
      '.html',
      '.htm',
      '.css',
      '.scss',
      '.sass',
      '.less',
      '.sql',
      '.graphql',
      '.gql',
    ];
    const ext = path.extname(filePath).toLowerCase();
    return codeExtensions.includes(ext);
  }

  private buildASTTree(
    sourceFile: SourceFile,
    maxDepth: number = 3,
  ): ASTTreeNode {
    const buildNode = (node: unknown, depth: number): ASTTreeNode => {
      const astNode: ASTTreeNode = {
        kind:
          SyntaxKind[(node as { getKind(): number }).getKind()] ||
          (node as { getKind(): number }).getKind().toString(),
        line:
          (
            node as { getStartLineNumber?: () => number | undefined }
          ).getStartLineNumber?.() || undefined,
      };

      // Add name if available
      if (typeof (node as { getName?: () => string }).getName === 'function') {
        try {
          astNode.name = (node as { getName: () => string }).getName();
        } catch (_error) {
          // Ignore errors getting name
        }
      }

      // Add short text preview for leaf nodes or small nodes
      if (
        depth >= maxDepth ||
        (node as { getChildren(): unknown[] }).getChildren().length === 0
      ) {
        const text = (node as { getText(): string }).getText();
        if (text && text.length < 100) {
          astNode.text = text.replace(/\s+/g, ' ').trim();
        } else if (text && text.length >= 100) {
          astNode.text =
            text.substring(0, 97).replace(/\s+/g, ' ').trim() + '...';
        }
      }

      // Add children recursively
      if (depth < maxDepth) {
        const children = (node as { getChildren(): unknown[] }).getChildren();
        if (children.length > 0) {
          astNode.children = children
            .slice(0, 20) // Limit to first 20 children to avoid overwhelming output
            .map((child: unknown) => buildNode(child, depth + 1));
        }
      }

      return astNode;
    };

    return buildNode(sourceFile, 0);
  }

  private formatASTTree(
    node: ASTTreeNode,
    indent: string = '',
    isLast: boolean = true,
  ): string {
    const connector = isLast ? '└── ' : '├── ';
    const nextIndent = indent + (isLast ? '    ' : '│   ');

    let result = indent + connector;
    result += `${node.kind}`;

    if (node.name) {
      result += ` "${node.name}"`;
    }

    if (node.line) {
      result += ` (line ${node.line})`;
    }

    if (node.text && !node.children) {
      result += ` → ${node.text}`;
    }

    result += '\n';

    if (node.children && node.children.length > 0) {
      node.children.forEach((child, index) => {
        const isChildLast = index === node.children!.length - 1;
        result += this.formatASTTree(child, nextIndent, isChildLast);
      });
    }

    return result;
  }

  private async performASTAnalysis(filePath: string): Promise<{
    astContent: string;
    parseResult?: ParseResult;
  }> {
    try {
      // Parse the file using the AST parser
      const parseResult = await readAndParseFile(filePath);

      if (parseResult.parseError) {
        return {
          astContent: `⚠️ AST Parse Error: ${parseResult.parseError}\n\nFile could not be fully parsed into AST.\n`,
        };
      }

      let astContent = '🌳 **AST ANALYSIS**\n\n';

      // Add file info
      astContent += `📊 **File Information:**\n`;
      astContent += `- Size: ${parseResult.fileInfo.sizeBytes} bytes (${parseResult.fileInfo.lineCount} lines)\n`;
      astContent += `- Processing time: ${parseResult.fileInfo.processingTimeMs}ms\n\n`;

      // Add intentions summary
      if (parseResult.intentions) {
        const intentions = parseResult.intentions as Intentions;
        astContent += `📋 **Code Structure Summary:**\n`;
        astContent += `- Functions: ${intentions.functions?.length || 0}\n`;
        astContent += `- Classes: ${intentions.classes?.length || 0}\n`;
        astContent += `- Imports: ${intentions.imports?.length || 0}\n`;
        astContent += `- Constants: ${intentions.constants?.length || 0}\n`;

        if (intentions.parsingErrors && intentions.parsingErrors.length > 0) {
          astContent += `- Parsing errors: ${intentions.parsingErrors.length}\n`;
        }
        astContent += '\n';

        // Add complexity metrics if available
        if (intentions.complexity) {
          const complexity = intentions.complexity;
          astContent += `📊 **Code Quality Metrics:**\n`;
          astContent += `- Cyclomatic Complexity: ${complexity.cyclomaticComplexity}\n`;
          astContent += `- Cognitive Complexity: ${complexity.cognitiveComplexity}\n`;
          astContent += `- Lines of Code: ${complexity.linesOfCode}\n`;
          astContent += `- Maintainability Index: ${complexity.maintainabilityIndex.toFixed(2)}\n`;

          if (complexity.halsteadMetrics) {
            const halstead = complexity.halsteadMetrics;
            astContent += `\n🧮 **Halstead Metrics:**\n`;
            astContent += `- Vocabulary Size: ${halstead.vocabularySize}\n`;
            astContent += `- Program Length: ${halstead.programLength}\n`;
            astContent += `- Difficulty: ${halstead.difficulty.toFixed(2)}\n`;
            astContent += `- Effort: ${halstead.effort.toFixed(2)}\n`;
            astContent += `- Time Required: ${halstead.timeRequired.toFixed(2)} seconds\n`;
            astContent += `- Estimated Bugs: ${halstead.bugsDelivered.toFixed(3)}\n`;
          }
          astContent += '\n';
        }

        // Add detailed function information
        if (intentions.functions && intentions.functions.length > 0) {
          astContent += `🔧 **Functions:**\n`;
          intentions.functions.slice(0, 10).forEach((func) => {
            astContent += `- ${func.name || '<anonymous>'}`;
            if (func.isAsync) astContent += ' (async)';
            if (func.startLine) astContent += ` [line ${func.startLine}]`;
            if (func.params && func.params.length > 0) {
              const paramNames = func.params
                .map((p) => p.name || '?')
                .join(', ');
              astContent += ` (${paramNames})`;
            }
            astContent += '\n';
          });
          if (intentions.functions.length > 10) {
            astContent += `... and ${intentions.functions.length - 10} more functions\n`;
          }
          astContent += '\n';
        }

        // Add detailed class information
        if (intentions.classes && intentions.classes.length > 0) {
          astContent += `🏛️ **Classes:**\n`;
          intentions.classes.slice(0, 5).forEach((cls) => {
            astContent += `- ${cls.name || '<anonymous>'}`;
            if (cls.isExported) astContent += ' (exported)';
            if (cls.startLine) astContent += ` [line ${cls.startLine}]`;
            if (cls.methods && cls.methods.length > 0) {
              astContent += ` - Methods: ${cls.methods.map((m) => m.name).join(', ')}`;
            }
            astContent += '\n';
          });
          if (intentions.classes.length > 5) {
            astContent += `... and ${intentions.classes.length - 5} more classes\n`;
          }
          astContent += '\n';
        }

        // Add imports information
        if (intentions.imports && intentions.imports.length > 0) {
          astContent += `📦 **Imports:**\n`;
          intentions.imports.slice(0, 8).forEach((imp) => {
            astContent += `- from "${imp.moduleSpecifier}"`;
            if (imp.defaultImport)
              astContent += ` default: ${imp.defaultImport}`;
            if (imp.namedImports && imp.namedImports.length > 0) {
              const named = imp.namedImports
                .map((n) => (n.alias ? `${n.name} as ${n.alias}` : n.name))
                .join(', ');
              astContent += ` named: {${named}}`;
            }
            if (imp.namespaceImport)
              astContent += ` namespace: ${imp.namespaceImport}`;
            astContent += '\n';
          });
          if (intentions.imports.length > 8) {
            astContent += `... and ${intentions.imports.length - 8} more imports\n`;
          }
          astContent += '\n';
        }
      }

      // Add AST tree structure if requested and sourceFile is available
      // Skip for large files to prevent API issues
      if (this.params.show_ast_tree !== false && parseResult.sourceFile && parseResult.fileInfo.sizeBytes < 500 * 1024) {
        astContent += `🌲 **AST Tree Structure:**\n`;
        const astTree = this.buildASTTree(parseResult.sourceFile);
        astContent += '```\n';
        astContent += this.formatASTTree(astTree);
        astContent += '```\n\n';
      } else if (this.params.show_ast_tree !== false && parseResult.sourceFile && parseResult.fileInfo.sizeBytes >= 500 * 1024) {
        astContent += `🌲 **AST Tree Structure:** Skipped for large file (>500KB) to prevent API issues.\n\n`;
      }

      // Perform AST query if specified
      if (this.params.ast_query && parseResult.sourceFile) {
        astContent += `🔍 **AST Query Results** (query: "${this.params.ast_query}"):\n`;
        try {
          const queryResults = findNodes(
            parseResult.sourceFile,
            this.params.ast_query,
          );
          if (queryResults.length > 0) {
            astContent += `Found ${queryResults.length} matching nodes:\n`;
            queryResults.slice(0, 10).forEach((node, index) => {
              const kind =
                SyntaxKind[node.getKind()] || node.getKind().toString();
              const line = node.getStartLineNumber?.() || '?';
              const text = node
                .getText()
                .substring(0, 100)
                .replace(/\s+/g, ' ')
                .trim();
              astContent += `${index + 1}. ${kind} [line ${line}]: ${text}${text.length === 100 ? '...' : ''}\n`;
            });
            if (queryResults.length > 10) {
              astContent += `... and ${queryResults.length - 10} more results\n`;
            }
          } else {
            astContent += 'No nodes matched the query.\n';
          }
        } catch (error) {
          astContent += `Query error: ${error}\n`;
        }
        astContent += '\n';
      }

      // Add comments and documentation
      if (parseResult.comments && parseResult.comments.length > 0) {
        astContent += `💬 **Comments** (${parseResult.comments.length} found):\n`;
        parseResult.comments.slice(0, 5).forEach((comment, index) => {
          const preview =
            comment.length > 100 ? comment.substring(0, 97) + '...' : comment;
          astContent += `${index + 1}. ${preview}\n`;
        });
        if (parseResult.comments.length > 5) {
          astContent += `... and ${parseResult.comments.length - 5} more comments\n`;
        }
        astContent += '\n';
      }

      if (parseResult.jsdocs && parseResult.jsdocs.length > 0) {
        astContent += `📖 **JSDoc Documentation** (${parseResult.jsdocs.length} found):\n`;
        parseResult.jsdocs.slice(0, 3).forEach((jsdoc, index) => {
          const preview =
            jsdoc.length > 150 ? jsdoc.substring(0, 147) + '...' : jsdoc;
          astContent += `${index + 1}. ${preview}\n`;
        });
        if (parseResult.jsdocs.length > 3) {
          astContent += `... and ${parseResult.jsdocs.length - 3} more JSDoc blocks\n`;
        }
        astContent += '\n';
      }

      astContent += '---\n\n';

      // Limit AST content size to prevent API issues
      if (astContent.length > ReadFileToolInvocation.MAX_AST_CONTENT_SIZE) {
        const truncatedLength = ReadFileToolInvocation.MAX_AST_CONTENT_SIZE - 100;
        astContent = astContent.substring(0, truncatedLength) + '\n\n⚠️ **AST content truncated** to prevent API issues.\n\n---\n\n';
      }

      return { astContent, parseResult };
    } catch (error) {
      return {
        astContent: `❌ **AST Analysis Failed:** ${error}\n\nContinuing with basic file content...\n\n---\n\n`,
      };
    }
  }

  async execute(): Promise<ToolResult> {
    // Always try to update the Mew window (fire and forget)
    this.updateMewWindow(this.params.absolute_path).catch(() => {
      // Silently ignore errors - this is a nice-to-have feature
    });

    const result = await processSingleFileContent(
      this.params.absolute_path,
      this.config.getTargetDir(),
      this.config.getFileSystemService(),
      this.params.offset,
      this.params.limit,
    );

    if (result.error) {
      return {
        llmContent:
          result.llmContent || `❌ **Error reading file:** ${result.error}`,
        returnDisplay: result.returnDisplay || 'Error reading file',
        error: {
          message: result.error,
          type: result.errorType,
        },
      };
    }

    let finalContent = '';
    let astAnalysis: { astContent: string; parseResult?: ParseResult } | null =
      null;

    // Perform AST analysis for code files (unless explicitly disabled)
    if (
      this.params.include_ast !== false &&
      this.isCodeFile(this.params.absolute_path)
    ) {
      astAnalysis = await this.performASTAnalysis(this.params.absolute_path);
      finalContent += astAnalysis.astContent;
    }

    const MAX_LLM_CONTENT_SIZE = 512 * 1024; // 512KB limit for LLM content

    if (result.isTruncated) {
      const [start, end] = result.linesShown!;
      const total = result.originalLineCount!;
      const nextOffset = this.params.offset
        ? this.params.offset + end - start + 1
        : end;

      finalContent += `
IMPORTANT: The file content has been truncated.
📊 Status: Showing lines ${start}-${end} of ${total} total lines.
🔧 Action: To read more of the file, use the 'offset' and 'limit' parameters in a subsequent 'read_file' call.
📍 Next section: Use offset: ${nextOffset}

--- FILE CONTENT (truncated) ---
${result.llmContent}`;
    } else {
      // Only add header if there's AST content or for better formatting
      const content = result.llmContent;
      if (astAnalysis && astAnalysis.astContent && typeof content === 'string') {
        finalContent = `📄 **FILE CONTENT:**\n\n${content}`;
      } else {
        finalContent = typeof content === 'string' ? content : JSON.stringify(content);
      }
    }

    const llmContent: PartListUnion = [{ text: finalContent }];

    // Allow llmContent to be either a string (text) or an object with expected parts
    const mimetype = getSpecificMimeType(this.params.absolute_path);
    const programming_language = getProgrammingLanguage({
      absolute_path: this.params.absolute_path,
    });

    // llmContent is now always a PartListUnion array, so we don't need the format validation
    // The validation was moved earlier in the process

    // Since llmContent is now always a PartListUnion, we check the original finalContent for size
    if (finalContent.length > MAX_LLM_CONTENT_SIZE) {
      const errorMsg = `File content exceeds maximum allowed size for LLM (${MAX_LLM_CONTENT_SIZE} bytes). Actual size: ${finalContent.length} bytes.`;
      return {
        llmContent: errorMsg,
        returnDisplay: `❌ Error: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_TOO_LARGE,
        },
      };
    }

    const lines = typeof result.llmContent === 'string'
      ? result.llmContent.split('\n').length
      : undefined;
    logFileOperation(
      this.config,
      new FileOperationEvent(
        ReadFileTool.Name,
        FileOperation.READ,
        lines,
        mimetype,
        path.extname(this.params.absolute_path),
        undefined,
        programming_language,
      ),
    );

    // Enhanced return display with AST info
    let returnDisplay = result.returnDisplay || '';
    if (astAnalysis?.parseResult) {
      const intentions = astAnalysis.parseResult.intentions as Intentions;
      if (intentions) {
        returnDisplay += `\n🌳 AST: ${intentions.functions?.length || 0} functions, ${intentions.classes?.length || 0} classes, ${intentions.imports?.length || 0} imports`;
      }
    }

    // For backward compatibility, return string for simple text content
    // and PartListUnion only for complex content like images/PDFs
    let finalLlmContent: string | PartListUnion = finalContent;
    
    // For inlineData objects (images/PDFs), return the object directly for backward compatibility
    if (result.llmContent && typeof result.llmContent === 'object' && 'inlineData' in result.llmContent) {
      finalLlmContent = result.llmContent as PartUnion;
    }
    // For objects with text property (like error messages), extract the text value
    else if (result.llmContent && typeof result.llmContent === 'object' && 'text' in result.llmContent) {
      finalLlmContent = (result.llmContent as { text: string }).text;
    }
    // For other objects, stringify them
    else if (result.llmContent && typeof result.llmContent === 'object') {
      finalLlmContent = JSON.stringify(result.llmContent);
    }

    return {
      llmContent: finalLlmContent,
      returnDisplay,
    };
  }
}

/**
 * Implementation of the ReadFile tool logic
 */
export class ReadFileTool extends BaseDeclarativeTool<
  ReadFileToolParams,
  ToolResult
> {
  static readonly Name: string = 'read_file';

  constructor(private config: Config) {
    super(
      ReadFileTool.Name,
      'ReadFile',
      `Advanced file reading tool with comprehensive AST analysis capabilities. Reads and returns the content of a specified file with automatic AST (Abstract Syntax Tree) analysis for code files. 

Features:
- 🌳 **Automatic AST Analysis**: For code files, shows structure, functions, classes, imports, and constants
- 🔍 **AST Querying**: Query specific AST nodes using XPath-like syntax
- 📊 **Code Structure**: Detailed breakdown of functions, classes, methods, and imports
- 💬 **Documentation**: Extracts comments and JSDoc documentation
- 🌲 **Visual Tree**: Shows hierarchical AST tree structure
- 📱 **MewApp Integration**: Automatically updates MewApp window with current file
- 📄 **Smart Truncation**: Handles large files with pagination support

Supports text, images (PNG, JPG, GIF, WEBP, SVG, BMP), PDF files, and comprehensive analysis for code files (JS, TS, Python, Java, C++, etc.).`,
      Kind.Read,
      {
        properties: {
          absolute_path: {
            description:
              "The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path.",
            type: 'string',
          },
          offset: {
            description:
              "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.",
            type: 'number',
          },
          limit: {
            description:
              "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit).",
            type: 'number',
          },
          include_ast: {
            description:
              'Optional: Whether to include AST analysis for supported code files (default: true). Set to false to skip AST analysis and get only raw file content.',
            type: 'boolean',
          },
          ast_query: {
            description:
              "Optional: XPath-like query to find specific AST nodes (e.g., '//FunctionDeclaration', '//ClassDeclaration[@name=\"MyClass\"]', '//ImportDeclaration'). Only works when include_ast is true.",
            type: 'string',
          },
          show_ast_tree: {
            description:
              'Optional: Whether to show the visual AST tree structure (default: true). Set to false to get only structural analysis without the tree visualization.',
            type: 'boolean',
          },
        },
        required: ['absolute_path'],
        type: 'object',
      },
    );
  }

  protected override validateToolParamValues(
    params: ReadFileToolParams,
  ): string | null {
    const filePath = params.absolute_path;
    if (params.absolute_path.trim() === '') {
      return "The 'absolute_path' parameter must be non-empty.";
    }

    if (!path.isAbsolute(filePath)) {
      return `File path must be absolute, but was relative: ${filePath}. You must provide an absolute path.`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(filePath)) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(', ')}`;
    }
    if (params.offset !== undefined && params.offset < 0) {
      return 'Offset must be a non-negative number';
    }
    if (params.limit !== undefined && params.limit <= 0) {
      return 'Limit must be a positive number';
    }

    const fileService = this.config.getFileService();
    if (fileService.shouldGeminiIgnoreFile(params.absolute_path)) {
      return `File path '${filePath}' is ignored by .geminiignore pattern(s).`;
    }

    return null;
  }

  protected createInvocation(
    params: ReadFileToolParams,
  ): ToolInvocation<ReadFileToolParams, ToolResult> {
    return new ReadFileToolInvocation(this.config, params);
  }
}
