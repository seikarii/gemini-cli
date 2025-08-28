/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TokenEstimator } from '../services/chatRecordingService.js';

/**
 * Context type enumeration for dual-context system
 */
export enum ContextType {
  PROMPT = 'prompt', // Large context for prompts (1M tokens)
  TOOL_EXECUTION = 'tool_execution', // Small context for tool execution (28K tokens)
}

/**
 * Configuration for dual-context token management
 */
export interface DualContextConfig {
  promptContextTokens: number; // Tokens for prompt context (default: 1M)
  toolContextTokens: number; // Tokens for tool execution context (default: 28K)
  enableDualContext: boolean; // Whether dual-context is enabled
  promptModel: string; // Model for prompts (default: gemini-2.5-pro)
  toolModel: string; // Model for tool execution (default: gemini-2.5-flash-lite)
}

/**
 * Context usage statistics
 */
export interface ContextUsageStats {
  contextType: ContextType;
  currentTokens: number;
  maxTokens: number;
  compressionRatio: number;
  messageCount: number;
}

/**
 * Intelligent context manager for dual-context token management
 */
export class ContextManager {
  private config: DualContextConfig;
  private tokenEstimator: TokenEstimator;
  private usageStats: Map<ContextType, ContextUsageStats>;

  constructor(config: DualContextConfig, tokenEstimator: TokenEstimator) {
    this.config = config;
    this.tokenEstimator = tokenEstimator;
    this.usageStats = new Map();

    // Initialize usage stats
    this.initializeUsageStats();
  }

  /**
   * Initialize usage statistics for both contexts
   */
  private initializeUsageStats(): void {
    this.usageStats.set(ContextType.PROMPT, {
      contextType: ContextType.PROMPT,
      currentTokens: 0,
      maxTokens: this.config.promptContextTokens,
      compressionRatio: 1.0,
      messageCount: 0,
    });

    this.usageStats.set(ContextType.TOOL_EXECUTION, {
      contextType: ContextType.TOOL_EXECUTION,
      currentTokens: 0,
      maxTokens: this.config.toolContextTokens,
      compressionRatio: 1.0,
      messageCount: 0,
    });
  }

  /**
   * Determine the appropriate context type for a given operation
   */
  determineContextType(operation: string, content: string): ContextType {
    if (!this.config.enableDualContext) {
      return ContextType.PROMPT; // Fallback to single context
    }

    // Tool execution patterns
    const toolPatterns = [
      /execute.*tool/i,
      /run.*command/i,
      /call.*function/i,
      /invoke.*method/i,
      /tool.*call/i,
      /function.*response/i,
    ];

    // If content matches tool patterns, use tool execution context
    if (toolPatterns.some((pattern) => pattern.test(content))) {
      return ContextType.TOOL_EXECUTION;
    }

    // For prompts, analysis, or general conversation, use prompt context
    return ContextType.PROMPT;
  }

  /**
   * Get the appropriate model for a context type
   */
  getModelForContext(contextType: ContextType): string {
    switch (contextType) {
      case ContextType.PROMPT:
        return this.config.promptModel;
      case ContextType.TOOL_EXECUTION:
        return this.config.toolModel;
      default:
        return this.config.promptModel;
    }
  }

  /**
   * Check if content fits within the specified context limits
   */
  async canFitInContext(
    content: string,
    contextType: ContextType,
  ): Promise<boolean> {
    const tokens = await this.tokenEstimator.estimateTokens(content);
    const maxTokens = this.getMaxTokensForContext(contextType);
    return tokens <= maxTokens;
  }

  /**
   * Get maximum tokens for a context type
   */
  getMaxTokensForContext(contextType: ContextType): number {
    switch (contextType) {
      case ContextType.PROMPT:
        return this.config.promptContextTokens;
      case ContextType.TOOL_EXECUTION:
        return this.config.toolContextTokens;
      default:
        return this.config.promptContextTokens;
    }
  }

  /**
   * Update usage statistics for a context
   */
  async updateUsageStats(
    contextType: ContextType,
    content: string,
    messageCount: number,
  ): Promise<void> {
    const tokens = await this.tokenEstimator.estimateTokens(content);
    const stats = this.usageStats.get(contextType)!;

    stats.currentTokens = tokens;
    stats.messageCount = messageCount;
    stats.compressionRatio = tokens > 0 ? tokens / stats.maxTokens : 0;

    this.usageStats.set(contextType, stats);
  }

  /**
   * Get usage statistics for a context type
   */
  getUsageStats(contextType: ContextType): ContextUsageStats | undefined {
    return this.usageStats.get(contextType);
  }

  /**
   * Get all usage statistics
   */
  getAllUsageStats(): ContextUsageStats[] {
    return Array.from(this.usageStats.values());
  }

  /**
   * Check if a context needs compression
   */
  needsCompression(contextType: ContextType, currentTokens: number): boolean {
    const maxTokens = this.getMaxTokensForContext(contextType);
    return currentTokens > maxTokens * 0.8; // 80% threshold
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<DualContextConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.initializeUsageStats(); // Reinitialize with new limits
  }
}
