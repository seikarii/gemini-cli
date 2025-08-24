/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolResult,
} from './tools';
import { Config } from '../config/config';
import { getSourceFile, extractIntentionMap } from '../ast/parser';
import { ASTQuery, DictionaryQuery, ASTQuery as _ASTQuery } from '../ast/models';

// Modifier imports
import { ASTModifier } from '../ast/modifier';
import { ModificationSpec, ModificationOperation } from '../ast/models';
import { findNodes } from '../ast/finder';
import { Node } from 'ts-morph';

/**
 * Very small, dependency-free line-oriented unified-ish diff generator.
 * Not perfect, but returns a readable diff for preview mode without adding deps.
 */
function generateSimpleLineDiff(oldText: string, newText: string, filePath: string) {
  const oldLines = oldText.split(/\r\n|\r|\n/);
  const newLines = newText.split(/\r\n|\r|\n/);
  const max = Math.max(oldLines.length, newLines.length);
  const out: string[] = [];
  out.push(`--- ${filePath}`);
  out.push(`+++ ${filePath} (modified)`);
  out.push('');
  for (let i = 0; i < max; i++) {
    const o = oldLines[i];
    const n = newLines[i];
    if (o === n) {
      // show a small context window only for readability
      if (i === 0 || i % 200 === 0) {
        out.push(' ' + (o ?? ''));
      }
    } else {
      if (o !== undefined) out.push('-' + o);
      if (n !== undefined) out.push('+' + n);
    }
  }
  return out.join('\n');
}

//=================================================================================================
// AST Read Tool
//=================================================================================================

export interface ASTReadToolParams {
  file_path: string;
}

class ASTReadToolInvocation extends BaseToolInvocation<ASTReadToolParams, ToolResult> {
  constructor(private readonly config: Config, params: ASTReadToolParams) {
    super(params);
  }

  getDescription(): string {
    return `Reading AST and intentions from ${this.params.file_path}`;
  }

  async execute(): Promise<ToolResult> {
    const { file_path } = this.params;
    const fs = this.config.getFileSystemService();

    const readResult = await fs.readTextFile(file_path);
    if (!readResult.success) {
      return {
        llmContent: `Error: Could not read file: ${readResult.error}`,
      };
    }

    const sourceFile = getSourceFile(file_path, readResult.data!);
    const intentionMap = extractIntentionMap(sourceFile);

    return {
      llmContent: JSON.stringify(intentionMap, null, 2),
    };
  }
}

export class ASTReadTool extends BaseDeclarativeTool<ASTReadToolParams, ToolResult> {
  static readonly Name = 'ast_read';

  constructor(private readonly config: Config) {
    super(
      ASTReadTool.Name,
      'ASTRead',
      'Reads a TypeScript/JavaScript file and returns a structured map of its intentions (functions, classes, imports, etc.).',
      Kind.Read,
      {
        properties: {
          file_path: {
            description: "The absolute path to the file to analyze.",
            type: 'string',
          },
        },
        required: ['file_path'],
        type: 'object',
      }
    );
  }

  protected createInvocation(params: ASTReadToolParams): ToolInvocation<ASTReadToolParams, ToolResult> {
    return new ASTReadToolInvocation(this.config, params);
  }
}

//=================================================================================================
// AST Find Tool
//=================================================================================================

export interface ASTFindToolParams {
  file_path: string;
  query: ASTQuery | DictionaryQuery | string;
}

class ASTFindToolInvocation extends BaseToolInvocation<ASTFindToolParams, ToolResult> {
  constructor(private readonly config: Config, params: ASTFindToolParams) {
    super(params);
  }

  getDescription(): string {
    return `Finding AST nodes in ${this.params.file_path}`;
  }

  async execute(): Promise<ToolResult> {
    const { file_path, query } = this.params;
    const fs = this.config.getFileSystemService();

    const readResult = await fs.readTextFile(file_path);
    if (!readResult.success) {
      return {
        llmContent: `Error: Could not read file: ${readResult.error}`,
      };
    }

    const sourceFile = getSourceFile(file_path, readResult.data!);
    const foundNodes = findNodes(sourceFile, query as any);

    if (foundNodes.length === 0) {
      return {
        llmContent: 'No nodes found matching the query.',
      };
    }

    const results = foundNodes.map((node: Node) => {
      return {
        kind: node.getKindName(),
        text: node.getText().substring(0, 100) + (node.getText().length > 100 ? '...' : ''),
        startLine: node.getStartLineNumber(),
        endLine: node.getEndLineNumber(),
      };
    });

    return {
      llmContent: JSON.stringify(results, null, 2),
    };
  }
}

export class ASTFindTool extends BaseDeclarativeTool<ASTFindToolParams, ToolResult> {
  static readonly Name = 'ast_find';

  constructor(private readonly config: Config) {
    super(
      ASTFindTool.Name,
      'ASTFind',
      'Finds and returns information about nodes in a TypeScript/JavaScript file that match a given query.',
      Kind.Read,
      {
        properties: {
          file_path: {
            description: "The absolute path to the file to search.",
            type: 'string',
          },
          query: {
            description: "A query to find nodes based on their properties. Accepts dictionary query, ComplexQuery, or XPath-like string.",
            type: 'object',
          },
        },
        required: ['file_path', 'query'],
        type: 'object',
      }
    );
  }

  protected createInvocation(params: ASTFindToolParams): ToolInvocation<ASTFindToolParams, ToolResult> {
    return new ASTFindToolInvocation(this.config, params);
  }
}

//=================================================================================================
// AST Edit Tool
//=================================================================================================

export interface ASTEditToolParams {
  file_path: string;
  query: ASTQuery | DictionaryQuery | string;
  new_text: string;
  // Optional flags:
  preview?: boolean; // if true, do not write file, return modified content
  create_backup?: boolean; // if false, skip creating backup (default true)
}

class ASTEditToolInvocation extends BaseToolInvocation<ASTEditToolParams, ToolResult> {
  constructor(private readonly config: Config, params: ASTEditToolParams) {
    super(params);
  }

  getDescription(): string {
    return `Editing AST node in ${this.params.file_path}`;
  }

  async execute(): Promise<ToolResult> {
    const { file_path, query, new_text } = this.params;
    const preview = !!(this.params as any).preview;
    const createBackup = (this.params as any).create_backup !== false; // default true
    const fs = this.config.getFileSystemService();

    const readResult = await fs.readTextFile(file_path);
    if (!readResult.success) {
      return {
        llmContent: `Error: Could not read file: ${readResult.error}`,
      };
    }

    const sourceFile = getSourceFile(file_path, readResult.data!);
    const foundNodes = findNodes(sourceFile, query as any);

    if (foundNodes.length === 0) {
      return {
        llmContent: 'Error: No nodes found matching the query. No changes made.',
      };
    }

    if (foundNodes.length > 1) {
      return {
        llmContent: `Error: Query is not specific enough. Found ${foundNodes.length} nodes. No changes made.`,
      };
    }

    // Use ASTModifier to perform the replacement in a robust, consistent way
    const modifier = new ASTModifier();
    const mods: ModificationSpec[] = [
      {
        operation: ModificationOperation.REPLACE,
        targetQuery: query as any,
        newCode: new_text,
      },
    ];

    // Ask modifier to format; it will always create an internal backup id.
    const modResult = await modifier.applyModifications(readResult.data!, mods, { filePath: file_path, format: true });

    if (!modResult.success) {
      return {
        llmContent: `Error applying modification: ${modResult.error ?? 'unknown error'}`,
      };
    }

    const newContent = modResult.modifiedText ?? readResult.data!;

    if (preview) {
      // Return a compact diff instead of the full modified file
      const diff = generateSimpleLineDiff(readResult.data!, newContent, file_path);
      return {
        llmContent: `Preview diff (no file written). Backup id: ${modResult.backupId ?? 'n/a'}\n\n${diff}`,
      };
    }

    // If createBackup is false, we still used modifier backup; user opted out of persistent backup on write.
    const writeResult = await fs.writeTextFile(file_path, newContent);
    if (!writeResult.success) {
      return {
        llmContent: `Error writing file: ${writeResult.error}`,
      };
    }

    return {
      llmContent: `Successfully edited node in ${file_path} (backup: ${createBackup ? modResult.backupId ?? 'n/a' : 'skipped'})`,
    };
  }
}

export class ASTEditTool extends BaseDeclarativeTool<ASTEditToolParams, ToolResult> {
  static readonly Name = 'ast_edit';

  constructor(private readonly config: Config) {
    super(
      ASTEditTool.Name,
      'ASTEdit',
      'Finds a single AST node using a query and replaces it with new text.',
      Kind.Edit,
      {
        properties: {
          file_path: {
            description: "The absolute path to the file to edit.",
            type: 'string',
          },
          query: {
            description: "The query to locate the single node to edit (DictionaryQuery, ComplexQuery, or XPath-like string).",
            type: 'object',
          },
          new_text: {
            description: "The new text to replace the found node with.",
            type: 'string',
          },
          preview: {
            description: "If true, return the modified content without writing the file.",
            type: 'boolean',
          },
          create_backup: {
            description: "If false, skip creating a persistent backup when writing the file (default true).",
            type: 'boolean',
          },
        },
        required: ['file_path', 'query', 'new_text'],
        type: 'object',
      }
    );
  }

  protected createInvocation(params: ASTEditToolParams): ToolInvocation<ASTEditToolParams, ToolResult> {
    return new ASTEditToolInvocation(this.config, params);
  }
}
