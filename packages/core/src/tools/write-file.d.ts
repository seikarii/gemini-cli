/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Config } from '../config/config.js';
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { ModifiableDeclarativeTool, ModifyContext } from './modifiable-tool.js';
/**
 * Parameters for the WriteFile tool
 */
export interface WriteFileToolParams {
    /**
     * The absolute path to the file to write to
     */
    file_path: string;
    /**
     * The content to write to the file
     */
    content: string;
    /**
     * The mode of writing. 'overwrite' will replace the entire file, 'append' will add to the end.
     * Defaults to 'overwrite'.
     */
    mode?: 'overwrite' | 'append';
    /**
     * If true, bypasses the content correction logic (ensureCorrectEdit).
     * Use with caution, primarily for simple appends to config files.
     */
    skip_correction?: boolean;
    /**
     * Whether the proposed content was modified by the user.
     */
    modified_by_user?: boolean;
    /**
     * Initially proposed content.
     */
    ai_proposed_content?: string;
}
interface GetCorrectedFileContentResult {
    originalContent: string;
    correctedContent: string;
    fileExists: boolean;
    error?: {
        message: string;
        code?: string;
    };
}
export declare function getCorrectedFileContent(config: Config, filePath: string, proposedContent: string, abortSignal: AbortSignal): Promise<GetCorrectedFileContentResult>;
/**
 * Implementation of the WriteFile tool logic
 */
export declare class WriteFileTool extends BaseDeclarativeTool<WriteFileToolParams, ToolResult> implements ModifiableDeclarativeTool<WriteFileToolParams> {
    private readonly config;
    static readonly Name: string;
    constructor(config: Config);
    protected validateToolParamValues(params: WriteFileToolParams): string | null;
    protected createInvocation(params: WriteFileToolParams): ToolInvocation<WriteFileToolParams, ToolResult>;
    getModifyContext(abortSignal: AbortSignal): ModifyContext<WriteFileToolParams>;
}
export {};
