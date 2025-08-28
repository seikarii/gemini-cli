/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { DualContextConfig } from './dualContextExample.js';
import { tokenLimit } from '../core/tokenLimits.js';

/**
 * Environment-based configuration for dual-context token management strategy.
 * This service loads configuration from environment variables and provides
 * optimized defaults for leveraging Gemini's full token capacity.
 */
export class DualContextEnvironmentConfig {
  private static config: DualContextConfig | null = null;

  /**
   * Gets the dual-context configuration from environment variables or defaults
   */
  static getConfig(): DualContextConfig {
    if (this.config) {
      return this.config;
    }

    // Load from environment variables with intelligent defaults
    const enableDualContext =
      process.env.GEMINI_ENABLE_DUAL_CONTEXT === 'true' || true; // Default enabled

    // Long-term memory configuration (for analysis, reasoning, comprehensive understanding)
    const longTermModel =
      process.env.GEMINI_LONG_TERM_MODEL || 'gemini-2.5-pro';
    const longTermMaxTokens = parseInt(
      process.env.GEMINI_LONG_TERM_TOKENS || '1000000',
      10,
    );
    const longTermRAGChunks = parseInt(
      process.env.GEMINI_LONG_TERM_RAG_CHUNKS || '20',
      10,
    );

    // Short-term memory configuration (for tool execution, immediate responses)
    const shortTermModel =
      process.env.GEMINI_SHORT_TERM_MODEL || 'gemini-2.5-flash';
    const shortTermMaxTokens = parseInt(
      process.env.GEMINI_SHORT_TERM_TOKENS || '28000',
      10,
    );
    const shortTermMessages = parseInt(
      process.env.GEMINI_SHORT_TERM_MESSAGES || '10',
      10,
    );

    // Compression and switching configuration
    const longTermThreshold = parseInt(
      process.env.GEMINI_LONG_TERM_THRESHOLD || '80',
      10,
    );
    const shortTermThreshold = parseInt(
      process.env.GEMINI_SHORT_TERM_THRESHOLD || '70',
      10,
    );
    const enableSmartSwitching =
      process.env.GEMINI_ENABLE_SMART_SWITCHING !== 'false'; // Default enabled

    // Validate token limits against model capabilities
    const longTermModelLimit = tokenLimit(longTermModel);
    const shortTermModelLimit = tokenLimit(shortTermModel);

    // Ensure token limits don't exceed model capabilities
    const validatedLongTermTokens = Math.min(
      longTermMaxTokens,
      longTermModelLimit,
    );
    const validatedShortTermTokens = Math.min(
      shortTermMaxTokens,
      shortTermModelLimit,
    );

    this.config = {
      enableDualContext,
      longTermMemory: {
        maxTokens: validatedLongTermTokens,
        model: longTermModel,
        enableRAG: true,
        maxRAGChunks: longTermRAGChunks,
        ragRelevanceThreshold: 0.5, // Lower threshold for broader knowledge
      },
      shortTermMemory: {
        maxTokens: validatedShortTermTokens,
        model: shortTermModel,
        maxRecentMessages: shortTermMessages,
        maxExecutionHistory: 5, // Keep execution history minimal
      },
      compression: {
        longTermThreshold,
        shortTermThreshold,
        enableSmartSwitching,
      },
    };

    console.log('Dual-context configuration loaded', {
      enableDualContext,
      longTerm: {
        model: longTermModel,
        maxTokens: validatedLongTermTokens,
        modelLimit: longTermModelLimit,
      },
      shortTerm: {
        model: shortTermModel,
        maxTokens: validatedShortTermTokens,
        modelLimit: shortTermModelLimit,
      },
      smartSwitching: enableSmartSwitching,
    });

    return this.config;
  }

  /**
   * Resets the configuration cache (useful for testing)
   */
  static resetConfig(): void {
    this.config = null;
  }

  /**
   * Gets the recommended configuration for production environments
   */
  static getProductionConfig(): DualContextConfig {
    return {
      enableDualContext: true,
      longTermMemory: {
        maxTokens: 1_000_000, // Full 1M capacity for Gemini 2.5 Pro
        model: 'gemini-2.5-pro',
        enableRAG: true,
        maxRAGChunks: 25,
        ragRelevanceThreshold: 0.4, // Broader knowledge retrieval
      },
      shortTermMemory: {
        maxTokens: 28_000, // Conservative for tool execution
        model: 'gemini-2.5-flash', // Fast execution
        maxRecentMessages: 8,
        maxExecutionHistory: 3,
      },
      compression: {
        longTermThreshold: 85, // Compress at 85% for production
        shortTermThreshold: 75, // Compress at 75% for tools
        enableSmartSwitching: true,
      },
    };
  }

  /**
   * Gets the recommended configuration for development environments
   */
  static getDevelopmentConfig(): DualContextConfig {
    return {
      enableDualContext: true,
      longTermMemory: {
        maxTokens: 500_000, // Reduced for development
        model: 'gemini-2.5-pro',
        enableRAG: true,
        maxRAGChunks: 15,
        ragRelevanceThreshold: 0.6,
      },
      shortTermMemory: {
        maxTokens: 20_000, // Smaller for development
        model: 'gemini-2.5-flash-lite', // Fastest for development
        maxRecentMessages: 6,
        maxExecutionHistory: 2,
      },
      compression: {
        longTermThreshold: 70, // Earlier compression in dev
        shortTermThreshold: 60,
        enableSmartSwitching: true,
      },
    };
  }

  /**
   * Validates that the current configuration is optimal for the detected models
   */
  static validateConfiguration(config: DualContextConfig): {
    isValid: boolean;
    warnings: string[];
    recommendations: string[];
  } {
    const warnings: string[] = [];
    const recommendations: string[] = [];

    // Check long-term memory configuration
    const longTermLimit = tokenLimit(config.longTermMemory.model);
    if (config.longTermMemory.maxTokens > longTermLimit) {
      warnings.push(
        `Long-term memory tokens (${config.longTermMemory.maxTokens}) exceed model limit (${longTermLimit})`,
      );
    }

    if (config.longTermMemory.maxTokens < 500_000) {
      recommendations.push(
        'Consider increasing long-term memory tokens to at least 500K for better analysis capability',
      );
    }

    // Check short-term memory configuration
    const shortTermLimit = tokenLimit(config.shortTermMemory.model);
    if (config.shortTermMemory.maxTokens > shortTermLimit) {
      warnings.push(
        `Short-term memory tokens (${config.shortTermMemory.maxTokens}) exceed model limit (${shortTermLimit})`,
      );
    }

    if (config.shortTermMemory.maxTokens > 50_000) {
      recommendations.push(
        'Consider reducing short-term memory tokens for faster tool execution',
      );
    }

    // Check model selection
    if (config.longTermMemory.model === config.shortTermMemory.model) {
      recommendations.push(
        'Consider using different models for long-term (Pro) and short-term (Flash) memory for optimal performance',
      );
    }

    return {
      isValid: warnings.length === 0,
      warnings,
      recommendations,
    };
  }
}

/**
 * Convenient function to get the current dual-context configuration
 */
export function getDualContextConfig(): DualContextConfig {
  return DualContextEnvironmentConfig.getConfig();
}

/**
 * Environment variables documentation:
 *
 * GEMINI_ENABLE_DUAL_CONTEXT=true          # Enable dual-context strategy
 * GEMINI_LONG_TERM_MODEL=gemini-2.5-pro    # Model for analysis and reasoning
 * GEMINI_LONG_TERM_TOKENS=1000000          # Max tokens for long-term memory
 * GEMINI_LONG_TERM_RAG_CHUNKS=20           # RAG chunks for comprehensive analysis
 * GEMINI_SHORT_TERM_MODEL=gemini-2.5-flash # Model for tool execution
 * GEMINI_SHORT_TERM_TOKENS=28000           # Max tokens for tool execution
 * GEMINI_SHORT_TERM_MESSAGES=10            # Recent messages for tool context
 * GEMINI_LONG_TERM_THRESHOLD=80            # Compression threshold for long-term
 * GEMINI_SHORT_TERM_THRESHOLD=70           # Compression threshold for short-term
 * GEMINI_ENABLE_SMART_SWITCHING=true       # Enable intelligent context switching
 */
