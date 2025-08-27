/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
/**
 * Parameters for the UnifiedSearchTool
 */
export interface UnifiedSearchToolParams {
    /**
     * The search query
     */
    query: string;
    /**
     * The directory to search in (optional, defaults to current directory)
     */
    path?: string;
    /**
     * Search mode: 'auto', 'text', 'semantic', 'functions', 'classes'
     */
    mode?: 'auto' | 'text' | 'semantic' | 'functions' | 'classes';
    /**
     * Maximum number of results to return (optional, defaults to 10)
     */
    max_results?: number;
    /**
     * Case sensitive search (text mode only)
     */
    case_sensitive?: boolean;
    /**
     * Use regex patterns (text mode only)
     */
    regex?: boolean;
}
/**
 * Implementation of the UnifiedSearch tool
 */
export declare class UnifiedSearchTool extends BaseDeclarativeTool<UnifiedSearchToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "unified_search";
    constructor(config: Config);
    /**
     * Validates the parameters for the tool
     */
    protected validateToolParamValues(params: UnifiedSearchToolParams): string | null;
    protected createInvocation(params: UnifiedSearchToolParams): ToolInvocation<UnifiedSearchToolParams, ToolResult>;
}
