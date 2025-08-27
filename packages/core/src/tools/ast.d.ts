/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
import { ASTQuery, DictionaryQuery } from '../ast/models.js';
export interface ASTFindToolParams {
    file_path: string;
    query: ASTQuery | DictionaryQuery | string;
}
export declare class ASTFindTool extends BaseDeclarativeTool<ASTFindToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "ast_find";
    constructor(config: Config);
    protected createInvocation(params: ASTFindToolParams): ToolInvocation<ASTFindToolParams, ToolResult>;
}
export interface ASTEditToolParams {
    file_path: string;
    query: ASTQuery | DictionaryQuery | string;
    new_text: string;
    preview?: boolean;
    create_backup?: boolean;
}
export declare class ASTEditTool extends BaseDeclarativeTool<ASTEditToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "ast_edit";
    constructor(config: Config);
    protected createInvocation(params: ASTEditToolParams): ToolInvocation<ASTEditToolParams, ToolResult>;
}
