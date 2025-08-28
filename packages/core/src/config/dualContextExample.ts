/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GenerateContentConfig } from '@google/genai';

/**
 * URGENT: Dual Context Token Management Strategy
 * 
 * This module implements the revised token management approach for Gemini CLI:
 * 
 * 1. **Long-Term Memory (Main Prompts)**: Up to 1M tokens with RAG processing
 * 2. **Short-Term Memory (Tool Execution)**: ~28K tokens, highly focused
 * 
 * The key distinction is between:
 * - PROMPT = Long-term memory (large context, RAG-processed)
 * - EXECUTION/TOOL_INTERACTION = Short-term memory (small context, current and focused)
 */

export enum ContextType {
  /** Long-term memory for reasoning, analysis, and general understanding */
  LONG_TERM_MEMORY = 'long_term_memory',
  /** Short-term memory for tool execution and immediate tasks */
  SHORT_TERM_MEMORY = 'short_term_memory'
}

export interface DualContextConfig {
  /** Enable dual-context token management */
  enableDualContext: boolean;

  /** Long-term memory configuration (for prompts and reasoning) */
  longTermMemory: {
    /** Maximum tokens for long-term context (up to 1M with Gemini 2.5 Pro) */
    maxTokens: number;
    /** Model optimized for large context processing */
    model: string;
    /** Enable RAG processing for long-term memory */
    enableRAG: boolean;
    /** Maximum RAG chunks for long-term context */
    maxRAGChunks: number;
    /** RAG relevance threshold for long-term memory */
    ragRelevanceThreshold: number;
  };

  /** Short-term memory configuration (for tool execution) */
  shortTermMemory: {
    /** Maximum tokens for short-term context (optimized for tool execution) */
    maxTokens: number;
    /** Model optimized for fast tool execution */
    model: string;
    /** Include only most recent conversation for context */
    maxRecentMessages: number;
    /** Maximum execution history to maintain */
    maxExecutionHistory: number;
  };

  /** Compression and optimization settings */
  compression: {
    /** Compress long-term context when reaching this percentage */
    longTermThreshold: number;
    /** Compress short-term context when reaching this percentage */
    shortTermThreshold: number;
    /** Enable intelligent context switching */
    enableSmartSwitching: boolean;
  };
}

export const DUAL_CONTEXT_CONFIG_EXAMPLE: DualContextConfig = {
  // Enable dual-context token management
  enableDualContext: true,

  // Long-term memory: Large context for reasoning and analysis (1M tokens with Gemini 2.5 Pro)
  longTermMemory: {
    maxTokens: 1000000,
    model: 'gemini-2.5-pro',
    enableRAG: true,
    maxRAGChunks: 50, // More chunks for comprehensive understanding
    ragRelevanceThreshold: 0.6,
  },

  // Short-term memory: Small context for tool execution (28K tokens with Gemini 2.5 Flash-Lite)
  shortTermMemory: {
    maxTokens: 28000,
    model: 'gemini-2.5-flash-lite',
    maxRecentMessages: 10, // Only recent conversation
    maxExecutionHistory: 5, // Limited execution history
  },

  // Compression settings
  compression: {
    longTermThreshold: 80, // Compress when long-term context reaches 80% of limit
    shortTermThreshold: 70, // Compress when short-term context reaches 70% of limit
    enableSmartSwitching: true, // Automatically switch between contexts
  },
};

/**
 * Context Manager for Dual Memory Strategy
 */
export class DualContextManager {
  private config: DualContextConfig;
  private currentContextType: ContextType = ContextType.LONG_TERM_MEMORY;

  constructor(config: DualContextConfig = DUAL_CONTEXT_CONFIG_EXAMPLE) {
    this.config = config;
  }

  /**
   * Determines which context type to use based on the operation
   */
  determineContextType(operation: string, isToolExecution: boolean = false): ContextType {
    // Tool execution always uses short-term memory
    if (isToolExecution) {
      return ContextType.SHORT_TERM_MEMORY;
    }

    // Analysis and reasoning operations use long-term memory
    const longTermOperations = [
      'analyze', 'understand', 'explain', 'reason', 'plan', 'design',
      'review', 'evaluate', 'compare', 'summarize', 'research'
    ];

    if (longTermOperations.some(op => operation.toLowerCase().includes(op))) {
      return ContextType.LONG_TERM_MEMORY;
    }

    // Default to long-term memory for general prompts
    return ContextType.LONG_TERM_MEMORY;
  }

  /**
   * Gets the appropriate configuration for the given context type
   */
  getContextConfig(contextType: ContextType): DualContextConfig['longTermMemory'] | DualContextConfig['shortTermMemory'] {
    return contextType === ContextType.LONG_TERM_MEMORY
      ? this.config.longTermMemory
      : this.config.shortTermMemory;
  }

  /**
   * Gets the appropriate GenerateContentConfig for the context type
   */
  getGenerationConfig(contextType: ContextType): Partial<GenerateContentConfig> {
    const isShortTerm = contextType === ContextType.SHORT_TERM_MEMORY;
    
    return {
      // Note: model selection will be handled by the calling code
      // maxTokens property might not be directly supported in GenerateContentConfig
      ...(isShortTerm 
        ? { /* short-term specific config */ }
        : { /* long-term specific config */ })
    };
  }

  /**
   * Switches context type and returns the new configuration
   */
  switchContext(newType: ContextType): DualContextConfig['longTermMemory'] | DualContextConfig['shortTermMemory'] {
    this.currentContextType = newType;
    return this.getContextConfig(newType);
  }

  /**
   * Checks if context compression is needed
   */
  needsCompression(currentTokens: number, contextType: ContextType): boolean {
    const config = this.getContextConfig(contextType);
    const threshold = contextType === ContextType.LONG_TERM_MEMORY
      ? this.config.compression.longTermThreshold
      : this.config.compression.shortTermThreshold;
    
    return (currentTokens / config.maxTokens) * 100 >= threshold;
  }

  /**
   * Gets current context type
   */
  getCurrentContextType(): ContextType {
    return this.currentContextType;
  }
}

/**
 * Environment variables for dual-context configuration:
 *
 * GEMINI_ENABLE_DUAL_CONTEXT=true
 * GEMINI_LONG_TERM_TOKENS=1000000
 * GEMINI_SHORT_TERM_TOKENS=28000
 * GEMINI_LONG_TERM_MODEL=gemini-2.5-pro
 * GEMINI_SHORT_TERM_MODEL=gemini-2.5-flash-lite
 * GEMINI_ENABLE_SMART_SWITCHING=true
 */

export const ENVIRONMENT_MAPPING = {
  enableDualContext: 'GEMINI_ENABLE_DUAL_CONTEXT',
  longTermTokens: 'GEMINI_LONG_TERM_TOKENS',
  shortTermTokens: 'GEMINI_SHORT_TERM_TOKENS',
  longTermModel: 'GEMINI_LONG_TERM_MODEL',
  shortTermModel: 'GEMINI_SHORT_TERM_MODEL',
  enableSmartSwitching: 'GEMINI_ENABLE_SMART_SWITCHING',
};
