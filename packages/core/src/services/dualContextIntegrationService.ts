/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { DualContextManager, ContextType } from '../config/dualContextExample.js';
import { getDualContextConfig } from '../config/dualContextEnvironmentConfig.js';
import { PromptContextManager } from './promptContextManager.js';
import { RAGChatIntegrationService } from './ragChatIntegrationService.js';
import { GenerateContentConfig } from '@google/genai';
import { Config } from '../config/config.js';

/**
 * Service that integrates the dual-context token management strategy
 * with existing prompt management and RAG services.
 * 
 * This service implements the urgent strategy to leverage Gemini's full 1M token capacity
 * by intelligently separating long-term memory (knowledge base, conversations)
 * from short-term memory (immediate tool execution context).
 */
export class DualContextIntegrationService {
  private dualContextManager: DualContextManager;
  private shortTermContextManager?: PromptContextManager;
  private longTermContextManager?: PromptContextManager;

  constructor(
    private readonly config: Config,
    private readonly ragChatIntegration: RAGChatIntegrationService,
  ) {
    // Initialize dual-context configuration from environment or defaults
    const dualContextConfig = getDualContextConfig();

    this.dualContextManager = new DualContextManager(dualContextConfig);

    console.log('DualContextIntegrationService initialized with environment-based configuration', {
      enableDualContext: dualContextConfig.enableDualContext,
      longTermModel: dualContextConfig.longTermMemory.model,
      longTermTokens: dualContextConfig.longTermMemory.maxTokens,
      shortTermModel: dualContextConfig.shortTermMemory.model,
      shortTermTokens: dualContextConfig.shortTermMemory.maxTokens,
      smartSwitching: dualContextConfig.compression.enableSmartSwitching,
    });
  }

  /**
   * Determines whether to use long-term or short-term context based on the operation type
   */
  async selectOptimalContext(
    operation: string,
    userMessage: string,
    requiresTools: boolean = false
  ): Promise<ContextType> {
    // Use dual-context manager's intelligent detection
    return this.dualContextManager.determineContextType(operation, requiresTools);
  }

  /**
   * Gets the appropriate prompt context manager for the determined context type
   */
  async getContextManager(contextType: ContextType): Promise<PromptContextManager> {
    if (contextType === ContextType.SHORT_TERM_MEMORY) {
      if (!this.shortTermContextManager) {
        // Create short-term context manager with tool execution limits
        this.shortTermContextManager = new PromptContextManager(
          this.ragChatIntegration['ragService'],
          this.ragChatIntegration['chatRecordingService'],
          this.config,
          {
            maxTotalTokens: 28000, // Short-term limit for tool execution
            maxRAGChunks: 4, // Reduced chunks for focused tool context
            ragRelevanceThreshold: 0.75, // Higher threshold for precise tool context
            ragWeight: 0.2, // Lower RAG weight for immediate tool needs
            prioritizeRecentConversation: true,
            useConversationalContext: true,
          }
        );
      }
      return this.shortTermContextManager;
    } else {
      if (!this.longTermContextManager) {
        // Create long-term context manager with full capacity
        this.longTermContextManager = new PromptContextManager(
          this.ragChatIntegration['ragService'],
          this.ragChatIntegration['chatRecordingService'],
          this.config,
          {
            maxTotalTokens: 1000000, // Full 1M token capacity for comprehensive analysis
            maxRAGChunks: 20, // More chunks for comprehensive knowledge
            ragRelevanceThreshold: 0.5, // Lower threshold for broader knowledge
            ragWeight: 0.6, // Higher RAG weight for knowledge-based tasks
            prioritizeRecentConversation: false, // Consider full conversation history
            useConversationalContext: true,
          }
        );
      }
      return this.longTermContextManager;
    }
  }

  /**
   * Gets the optimal generation configuration for the context type
   */
  getGenerationConfig(contextType: ContextType): Partial<GenerateContentConfig> {
    return this.dualContextManager.getGenerationConfig(contextType);
  }

  /**
   * Orchestrates the intelligent context switching based on operation complexity
   * This is the main method that external services should use
   */
  async processWithOptimalContext(
    operation: string,
    userMessage: string,
    requiresTools: boolean = false
  ): Promise<{
    contextType: ContextType;
    contextManager: PromptContextManager;
    generationConfig: Partial<GenerateContentConfig>;
    maxTokens: number;
  }> {
    // Determine optimal context type
    const contextType = await this.selectOptimalContext(operation, userMessage, requiresTools);
    
    // Get appropriate context manager
    const contextManager = await this.getContextManager(contextType);
    
    // Get generation configuration
    const generationConfig = this.getGenerationConfig(contextType);
    
    // Get token limit for this context
    const contextConfig = this.dualContextManager.getContextConfig(contextType);
    const maxTokens = contextConfig.maxTokens;

    console.log(`Dual-context strategy selected: ${contextType}`, {
      operation,
      requiresTools,
      maxTokens,
      messageLength: userMessage.length
    });

    return {
      contextType,
      contextManager,
      generationConfig,
      maxTokens
    };
  }

  /**
   * Provides metrics about context usage for monitoring and optimization
   */
  getContextMetrics(): {
    currentContextType: ContextType;
    isCompressionNeeded: boolean;
    smartSwitchingEnabled: boolean;
  } {
    const currentType = this.dualContextManager.getCurrentContextType();
    return {
      currentContextType: currentType,
      isCompressionNeeded: false, // TODO: Implement compression detection
      smartSwitchingEnabled: this.dualContextManager['config'].compression.enableSmartSwitching
    };
  }

  /**
   * Resets context managers to free memory when needed
   */
  async resetContextManagers(): Promise<void> {
    this.shortTermContextManager = undefined;
    this.longTermContextManager = undefined;
    console.log('Dual-context managers reset for memory optimization');
  }
}
