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
} from './tools.js';
import { generateUnifiedDiff } from './diffPreview.js';
import { Config } from '../config/config.js';
import { parseSourceToSourceFile } from '../ast/parser.js';
import { ASTQuery, DictionaryQuery } from '../ast/models.js';
import { findNodes } from '../ast/finder.js';
import { ASTModifier } from '../ast/modifier.js';
import { ModificationOperation } from '../ast/models.js';
import type { Node } from 'ts-morph';

//=================================================================================================
// AST Find Tool
//=================================================================================================

export interface ASTFindToolParams {
  file_path: string;
  query: ASTQuery | DictionaryQuery | string;
}

class ASTFindToolInvocation extends BaseToolInvocation<
  ASTFindToolParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    private readonly toolParams: ASTFindToolParams,
  ) {
    super(toolParams);
  }

  getDescription(): string {
    return `Find AST nodes in ${this.toolParams.file_path}`;
  }

  async execute(): Promise<ToolResult> {
    const { file_path, query } = this.toolParams;
    const fs = this.config.getFileSystemService();
    const readResult = await fs.readTextFile(file_path);
    if (!readResult.success) {
      return {
        llmContent: `Error: Could not read file: ${readResult.error}`,
        returnDisplay: `Error: Could not read file: ${readResult.error}`,
      };
    }
    const parseResult = parseSourceToSourceFile(readResult.data!, file_path);
    const sourceFile = parseResult?.sourceFile;
    if (!sourceFile) {
      return {
        llmContent: `Error: Could not parse file: ${file_path}`,
        returnDisplay: `Error: Could not parse file: ${file_path}`,
      };
    }
    const foundNodes = findNodes(sourceFile, query as DictionaryQuery | string);
    if (!foundNodes || foundNodes.length === 0) {
      return {
        llmContent: 'No nodes found matching the query.',
        returnDisplay: 'No nodes found matching the query.',
      };
    }
    // Return basic info for each node
    const results = foundNodes.map((node) => {
      const n = node as Node;
      return {
        kind: n.getKindName ? n.getKindName() : n.getKind(),
        text: n.getText
          ? n.getText().substring(0, 100) +
            (n.getText().length > 100 ? '...' : '')
          : '',
        startLine: n.getStartLineNumber ? n.getStartLineNumber() : undefined,
        endLine: n.getEndLineNumber ? n.getEndLineNumber() : undefined,
      };
    });
    return {
      llmContent: JSON.stringify(results, null, 2),
      returnDisplay: JSON.stringify(results, null, 2),
    };
  }
}

export class ASTFindTool extends BaseDeclarativeTool<
  ASTFindToolParams,
  ToolResult
> {
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
            description: 'The absolute path to the file to search.',
            type: 'string',
          },
          query: {
            description:
              'A query to find nodes based on their properties. Accepts dictionary query, ComplexQuery, or XPath-like string.',
            type: 'object',
          },
        },
        required: ['file_path', 'query'],
        type: 'object',
      },
    );
  }

  protected createInvocation(
    params: ASTFindToolParams,
  ): ToolInvocation<ASTFindToolParams, ToolResult> {
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
  preview?: boolean;
  create_backup?: boolean;
}

class ASTEditToolInvocation extends BaseToolInvocation<
  ASTEditToolParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    private readonly toolParams: ASTEditToolParams,
  ) {
    super(toolParams);
  }

  getDescription(): string {
    return `Edit AST node in ${this.toolParams.file_path}`;
  }

  async execute(): Promise<ToolResult> {
    const { file_path, query, new_text, preview, create_backup } =
      this.toolParams;
    const fs = this.config.getFileSystemService();
    const readResult = await fs.readTextFile(file_path);
    if (!readResult.success) {
      return {
        llmContent: `Error: Could not read file: ${readResult.error}`,
        returnDisplay: `Error: Could not read file: ${readResult.error}`,
      };
    }
    const parseResult = parseSourceToSourceFile(readResult.data!, file_path);
    const sourceFile = parseResult?.sourceFile;
    if (!sourceFile) {
      return {
        llmContent: `Error: Could not parse file: ${file_path}`,
        returnDisplay: `Error: Could not parse file: ${file_path}`,
      };
    }
    // Use ASTModifier for robust editing
    const modifier = new ASTModifier();
    const mods = [
      {
        operation: ModificationOperation.REPLACE,
        targetQuery: query as DictionaryQuery | string,
        newCode: new_text,
      },
    ];
    const modResult = await modifier.applyModifications(
      readResult.data!,
      mods,
      { filePath: file_path, format: true },
    );
    if (!modResult.success) {
      return {
        llmContent: `Error applying modification: ${modResult.error ?? 'unknown error'}`,
        returnDisplay: `Error applying modification: ${modResult.error ?? 'unknown error'}`,
      };
    }
    const newContent = modResult.modifiedText ?? readResult.data!;
    if (preview) {
      const diff = generateUnifiedDiff(readResult.data!, newContent, file_path);
      return {
        llmContent: `Preview diff (no file written). Backup id: ${modResult.backupId ?? 'n/a'}\n\n${diff}`,
        returnDisplay: `Preview diff (no file written). Backup id: ${modResult.backupId ?? 'n/a'}\n\n${diff}`,
      };
    }
    const writeResult = await fs.writeTextFile(file_path, newContent);
    if (!writeResult.success) {
      return {
        llmContent: `Error writing file: ${writeResult.error}`,
        returnDisplay: `Error writing file: ${writeResult.error}`,
      };
    }
    return {
      llmContent: `Successfully edited node in ${file_path} (backup: ${create_backup ? (modResult.backupId ?? 'n/a') : 'skipped'})`,
      returnDisplay: `Successfully edited node in ${file_path} (backup: ${create_backup ? (modResult.backupId ?? 'n/a') : 'skipped'})`,
    };
  }
}

export class ASTEditTool extends BaseDeclarativeTool<
  ASTEditToolParams,
  ToolResult
> {
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
            description: 'The absolute path to the file to edit.',
            type: 'string',
          },
          query: {
            description:
              'The query to locate the single node to edit (DictionaryQuery, ComplexQuery, or XPath-like string).',
            type: 'object',
          },
          new_text: {
            description: 'The new text to replace the found node with.',
            type: 'string',
          },
          preview: {
            description:
              'If true, return the modified content without writing the file.',
            type: 'boolean',
          },
          create_backup: {
            description:
              'If false, skip creating a persistent backup when writing the file (default true).',
            type: 'boolean',
          },
        },
        required: ['file_path', 'query', 'new_text'],
        type: 'object',
      },
    );
  }

  protected createInvocation(
    params: ASTEditToolParams,
  ): ToolInvocation<ASTEditToolParams, ToolResult> {
    return new ASTEditToolInvocation(this.config, params);
  }
}
