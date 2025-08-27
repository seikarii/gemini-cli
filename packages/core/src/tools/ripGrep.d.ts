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
export interface RipGrepToolParams {
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
     * Maximum number of matches to return (optional, defaults to 20000)
     */
    max_matches?: number;
    /**
     * Whether to show dependency check warnings (optional, defaults to false)
     */
    show_dependency_warnings?: boolean;
}
/**
 * Implementation of the Grep tool logic (moved from CLI)
 */
export declare class RipGrepTool extends BaseDeclarativeTool<RipGrepToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "search_file_content";
    constructor(config: Config);
    /**
     * Validates the parameters for the tool
     * @param params Parameters to validate
     * @returns An error message string if invalid, null otherwise
     */
    validateToolParams(params: RipGrepToolParams): string | null;
    protected createInvocation(params: RipGrepToolParams): ToolInvocation<RipGrepToolParams, ToolResult>;
}
