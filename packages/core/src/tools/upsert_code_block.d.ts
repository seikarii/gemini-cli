/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
export interface UpsertCodeBlockToolParams {
    /**
     * The absolute path to the file to modify.
     */
    file_path: string;
    /**
     * The name of the block (e.g., function, class, const) to find.
     */
    block_name: string;
    /**
     * The new content for the block.
     */
    content: string;
    /**
     * The type of block to upsert (function, class, variable, interface, type).
     * If not specified, will auto-detect from content.
     */
    block_type?: 'function' | 'class' | 'variable' | 'interface' | 'type' | 'auto';
    /**
     * Position preference for new blocks: 'top', 'bottom', 'after_imports', 'before_exports'.
     * Default: 'bottom'
     */
    insert_position?: 'top' | 'bottom' | 'after_imports' | 'before_exports';
    /**
     * Whether to preserve existing formatting and comments around the block.
     * Default: true
     */
    preserve_formatting?: boolean;
}
/**
 * Enhanced implementation of the UpsertCodeBlock tool with robust AST parsing,
 * multi-language support, and intelligent block detection.
 *
 * Features:
 * - Multi-language support (Python, TypeScript/JavaScript, plain text)
 * - Intelligent block detection and replacement
 * - Configurable insertion positions
 * - Preservation of formatting and comments
 * - Comprehensive error handling and validation
 * - Defensive programming patterns following Crisalida conventions
 */
export declare class UpsertCodeBlockTool extends BaseDeclarativeTool<UpsertCodeBlockToolParams, ToolResult> {
    private config;
    static readonly Name = "upsert_code_block";
    constructor(config: Config);
    protected createInvocation(params: UpsertCodeBlockToolParams): ToolInvocation<UpsertCodeBlockToolParams, ToolResult>;
}
