/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { BaseDeclarativeTool, ToolInvocation, ToolResult } from './tools.js';
import { Config } from '../config/config.js';
/**
 * Parameters for the SemanticSearchTool
 */
export interface SemanticSearchToolParams {
    /**
     * The semantic search query
     */
    query: string;
    /**
     * The directory to search in (optional, defaults to current directory)
     */
    path?: string;
    /**
     * Type of search to perform
     */
    action?: 'search_semantic' | 'search_functions' | 'search_classes' | 'analyze_structure';
    /**
     * Maximum number of results to return (optional, defaults to 10)
     */
    max_results?: number;
    /**
     * Minimum similarity threshold (optional, defaults to 0.1)
     */
    min_similarity?: number;
}
/**
 * Implementation of the SemanticSearch tool
 */
export declare class SemanticSearchTool extends BaseDeclarativeTool<SemanticSearchToolParams, ToolResult> {
    private readonly config;
    static readonly Name = "semantic_search";
    constructor(config: Config);
    /**
     * Validates the parameters for the tool
     */
    protected validateToolParamValues(params: SemanticSearchToolParams): string | null;
    protected createInvocation(params: SemanticSearchToolParams): ToolInvocation<SemanticSearchToolParams, ToolResult>;
}
