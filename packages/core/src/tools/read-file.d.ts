/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
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
/**
 * Implementation of the ReadFile tool logic
 */
export declare class ReadFileTool extends BaseDeclarativeTool<ReadFileToolParams, ToolResult> {
    private config;
    static readonly Name: string;
    constructor(config: Config);
    protected validateToolParamValues(params: ReadFileToolParams): string | null;
    protected createInvocation(params: ReadFileToolParams): ToolInvocation<ReadFileToolParams, ToolResult>;
}
