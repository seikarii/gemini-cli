/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
import { ModifiableDeclarativeTool, ModifyContext } from './modifiable-tool.js';
export declare function applyReplacement(currentContent: string | null, oldString: string, newString: string, isNewFile: boolean, targetOccurrence?: number | 'first' | 'last' | 'all'): string;
/**
 * Applies a range-based edit to content using line/column coordinates
 */
export declare function applyRangeEdit(currentContent: string, startLine: number, startColumn: number, endLine: number, endColumn: number, newContent: string): string;
/**
 * Parameters for the Edit tool
 */
export interface EditToolParams {
    /**
     * The absolute path to the file to modify
     */
    file_path: string;
    /**
     * The text to replace (for string-based editing)
     */
    old_string?: string;
    /**
     * The text to replace it with (for string-based editing)
     */
    new_string?: string;
    /**
     * Number of replacements expected. Defaults to 1 if not specified.
     * Use when you want to replace multiple occurrences.
     */
    expected_replacements?: number;
    /**
     * Controls which occurrence(s) to replace:
     * - number (1-based): replace that specific occurrence
     * - 'first': replace the first occurrence (default)
     * - 'last': replace the last occurrence
     * - 'all': replace all occurrences
     */
    target_occurrence?: number | 'first' | 'last' | 'all';
    /**
     * Start line for range-based editing (0-indexed)
     */
    start_line?: number;
    /**
     * Start column for range-based editing (0-indexed)
     */
    start_column?: number;
    /**
     * End line for range-based editing (0-indexed)
     */
    end_line?: number;
    /**
     * End column for range-based editing (0-indexed)
     */
    end_column?: number;
    /**
     * New content to insert (for range-based editing)
     */
    new_content?: string;
    /**
     * Whether to perform a dry run validation before executing the edit
     */
    dry_run?: boolean;
    /**
     * Whether the edit was modified manually by the user.
     */
    modified_by_user?: boolean;
    /**
     * Initially proposed string.
     */
    ai_proposed_string?: string;
}
/**
 * Implementation of the Edit tool logic
 */
export declare class EditTool extends BaseDeclarativeTool<EditToolParams, ToolResult> implements ModifiableDeclarativeTool<EditToolParams> {
    private readonly config;
    static readonly Name = "replace";
    constructor(config: Config);
    /**
     * Validates the parameters for the Edit tool
     */
    protected validateToolParamValues(params: EditToolParams): string | null;
    getModifyContext(_abortSignal: AbortSignal): ModifyContext<EditToolParams>;
    protected createInvocation(params: EditToolParams): ToolInvocation<EditToolParams, ToolResult>;
}
