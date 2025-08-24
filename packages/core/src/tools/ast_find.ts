/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Project } from 'ts-morph';
import {
  BaseDeclarativeTool,
  ToolInvocation,
  ToolResult,
  Kind,
} from './tools.js';
import { BaseToolInvocation } from './tools.js';
import * as fs from 'fs';

/**
 * Parameters for the AstFind tool.
 */
export interface AstFindToolParams {
  /**
   * The absolute path to the file to search within.
   */
  file_path: string;

  /**
   * A query object to specify what to find.
   */
  query: {
    /**
     * The type of AST node to find (e.g., 'FunctionDeclaration', 'ClassDeclaration', 'VariableDeclaration').
     * If omitted, searches for all common top-level declarations.
     */
    type?: string;
    /**
     * The name of the node to find (e.g., function name, class name, variable name).
     * If omitted, matches any name of the specified type.
     */
    name?: string;
    // More complex query properties can be added here later (e.g., decorators, modifiers, parent)
  };

  /**
   * Whether to include the full source code of the found node(s) in the result.
   * Defaults to false.
   */
  include_content?: boolean;
}

interface FoundNodeInfo {
  type: string;
  name?: string;
  startLine: number;
  endLine: number;
  content?: string;
}

class AstFindToolInvocation extends BaseToolInvocation<
  AstFindToolParams,
  ToolResult
> {
  constructor(params: AstFindToolParams) {
    super(params);
  }

  getDescription(): string {
    const queryParts: string[] = [];
    if (this.params.query.type) queryParts.push(`type: ${this.params.query.type}`);
    if (this.params.query.name) queryParts.push(`name: ${this.params.query.name}`);
    const queryStr = queryParts.length > 0 ? ` (${queryParts.join(', ')})` : '';
    return `Finding AST nodes in ${this.params.file_path}${queryStr}`;
  }

  async execute(abortSignal: AbortSignal): Promise<ToolResult> {
    try {
      if (!fs.existsSync(this.params.file_path)) {
        return {
          llmContent: `File not found: ${this.params.file_path}`,
          returnDisplay: `❌ File not found: ${this.params.file_path}`,
        };
      }

      const project = new Project({
        useInMemoryFileSystem: true,
      });
      const sourceFile = project.addSourceFileAtPath(this.params.file_path);

      const foundNodes: FoundNodeInfo[] = [];
      const queryType = this.params.query.type;
      const queryName = this.params.query.name;
      const includeContent = this.params.include_content ?? false;

      // Search for functions
      if (!queryType || queryType === 'FunctionDeclaration') {
        sourceFile.getFunctions().forEach(func => {
          if (!queryName || func.getName() === queryName) {
            foundNodes.push({
              type: 'FunctionDeclaration',
              name: func.getName(),
              startLine: func.getStartLineNumber(),
              endLine: func.getEndLineNumber(),
              content: includeContent ? func.getText() : undefined,
            });
          }
        });
      }

      // Search for classes
      if (!queryType || queryType === 'ClassDeclaration') {
        sourceFile.getClasses().forEach(cls => {
          if (!queryName || cls.getName() === queryName) {
            foundNodes.push({
              type: 'ClassDeclaration',
              name: cls.getName(),
              startLine: cls.getStartLineNumber(),
              endLine: cls.getEndLineNumber(),
              content: includeContent ? cls.getText() : undefined,
            });
          }
        });
      }

      // Search for variable declarations (top-level)
      if (!queryType || queryType === 'VariableDeclaration') {
        sourceFile.getVariableStatements().forEach(stmt => {
          stmt.getDeclarations().forEach(decl => {
            if (!queryName || decl.getName() === queryName) {
              foundNodes.push({
                type: 'VariableDeclaration',
                name: decl.getName(),
                startLine: decl.getStartLineNumber(),
                endLine: decl.getEndLineNumber(),
                content: includeContent ? stmt.getText() : undefined, // Get text of the whole statement
              });
            }
          });
        });
      }

      const outputMessage = `Found ${foundNodes.length} nodes matching your query.`;
      return {
        llmContent: outputMessage,
        returnDisplay: `✅ ${outputMessage}`,
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error executing ast_find: ${errorMessage}`,
        returnDisplay: `❌ Error executing ast_find: ${errorMessage}`,
      };
    }
  }
}

/**
 * Implementation of the AstFind tool.
 */
export class AstFindTool extends BaseDeclarativeTool<
  AstFindToolParams,
  ToolResult
> {
  static readonly Name = 'ast_find';

  constructor() {
    super(
      AstFindTool.Name,
      'AstFind',
      'Searches for specific nodes (functions, classes, variables, etc.) in the Abstract Syntax Tree (AST) of a TypeScript/JavaScript file. Allows querying by node type and name.',
      Kind.Search,
      {
        properties: {
          file_path: {
            description: "The absolute path to the file to search within. Must start with '/'.",
            type: 'string',
          },
          query: {
            description: 'A query object to specify what to find. Supports \'type\' (e.g., FunctionDeclaration, ClassDeclaration) and \'name\'.',
            type: 'object',
            properties: {
              type: { type: 'string' },
              name: { type: 'string' },
            },
            required: [],
          },
          include_content: {
            description: 'Whether to include the full source code of the found node(s) in the result. Defaults to false.',
            type: 'boolean',
          },
        },
        required: ['file_path', 'query'],
        type: 'object',
      }
    );
  }

  protected createInvocation(
    params: AstFindToolParams
  ): ToolInvocation<AstFindToolParams, ToolResult> {
    return new AstFindToolInvocation(params);
  }
}
