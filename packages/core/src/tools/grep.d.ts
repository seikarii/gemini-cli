/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
/**
 * Parameters for the GrepTool
 */
export interface GrepToolParams {
    /**
     * The regular expression pattern to search for in file contents
     */
    pattern: string;
    /**
     * The directory to search in (optional, defaults to current directory relative to root)
     */
    path?: string;
    /**
     * File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")
     */
    include?: string;
    /**
     * Whether to show which search strategy was used (optional, defaults to false)
     */
    show_strategy?: boolean;
    /**
     * Maximum number of matches to return (optional, defaults to 20000)
     */
    max_matches?: number;
}
/**
 * Implementation of the Grep tool logic (moved from CLI)
 */
export declare class GrepTool extends BaseDeclarativeTool<GrepToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "search_file_content";
    constructor(config: Config);
    /**
     * Validates the parameters for the tool
     * @param params Parameters to validate
     * @returns An error message string if invalid, null otherwise
     */
    protected validateToolParamValues(params: GrepToolParams): string | null;
    protected createInvocation(params: GrepToolParams): ToolInvocation<GrepToolParams, ToolResult>;
}
