/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content } from '@google/genai';
import { RAGService } from '../rag/ragService.js';
import { ChatRecordingService } from './chatRecordingService.js';
import { Config } from '../config/config.js';
import {
  RAGQuery,
  EnhancedQueryResult,
  RAGChunk,
  QueryEnhancementOptions,
  QueryType,
} from '../rag/types.js';

/**
 * Configuration options for the PromptContextManager
 */
export interface PromptContextConfig {
  /** Maximum total tokens allowed in the combined context */
  maxTotalTokens: number;
  /** Maximum number of RAG chunks to include */
  maxRAGChunks: number;
  /** Minimum relevance score for RAG chunks */
  ragRelevanceThreshold: number;
  /** Weight given to RAG content vs conversational history (0-1) */
  ragWeight: number;
  /** Whether to prioritize recent conversation over older RAG content */
  prioritizeRecentConversation: boolean;
  /** Whether to include conversation context in RAG queries */
  useConversationalContext: boolean;
}

/**
 * Context assembly result containing the optimized prompt components
 */
export interface AssembledContext {
  /** Complete content array ready for LLM consumption */
  contents: Content[];
  /** Estimated total token count */
  estimatedTokens: number;
  /** Number of RAG chunks included */
  ragChunksIncluded: number;
  /** Number of conversation messages included */
  conversationMessagesIncluded: number;
  /** RAG context summary for debugging */
  ragContextSummary?: string;
  /** Conversation compression applied */
  compressionLevel?:
    | 'none'
    | 'minimal'
    | 'moderate'
    | 'aggressive'
    | 'intelligent';
}

/**
 * Central orchestrator for building optimized prompts that combine
 * RAG-retrieved knowledge with conversational history.
 *
 * This service addresses the integration gap between the RAG system
 * and chat components by intelligently combining external knowledge
 * with conversation context while respecting token limits.
 */
export class PromptContextManager {
  private readonly config: PromptContextConfig;

  constructor(
    private readonly ragService: RAGService,
    private readonly chatRecordingService: ChatRecordingService,
    private readonly systemConfig: Config,
    config?: Partial<PromptContextConfig>,
  ) {
    // Default configuration with smart defaults
    this.config = {
      maxTotalTokens: 30000, // Conservative limit for most models
      maxRAGChunks: 8,
      ragRelevanceThreshold: 0.6,
      ragWeight: 0.4, // 40% RAG, 60% conversation
      prioritizeRecentConversation: true,
      useConversationalContext: true,
      ...config,
    };

    console.log('PromptContextManager initialized', {
      config: this.config,
    });
  }

  /**
   * Assembles an optimized context that combines RAG knowledge with
   * conversational history, respecting token limits and relevance.
   */
  async assembleContext(
    userMessage: string,
    conversationHistory: Content[],
    _promptId?: string,
  ): Promise<AssembledContext> {
    const startTime = performance.now();

    try {
      // Step 1: Extract conversational context for RAG query enhancement
      const conversationalContext = this.config.useConversationalContext
        ? this.extractConversationalContext(conversationHistory)
        : '';

      // Step 2: Create RAG query object
      const ragQuery: RAGQuery = {
        text: this.config.useConversationalContext
          ? `${conversationalContext}\n\nCurrent query: ${userMessage}`
          : userMessage,
        type: QueryType.GENERAL_QUESTION,
        maxResults: this.config.maxRAGChunks,
        filters: {},
      };

      // Step 3: Create enhancement options
      const enhancementOptions: QueryEnhancementOptions = {
        maxTokens: this.config.maxTotalTokens,
        includeDocumentation: true,
      };

      // Step 4: Retrieve relevant RAG chunks
      const ragResults = await this.ragService.enhanceQuery(
        ragQuery,
        enhancementOptions,
      );

      // Step 5: Get compressed conversation history
      const compressedHistory = await this.getOptimizedConversationHistory(
        conversationHistory,
        ragResults.sourceChunks.length,
      );

      // Step 6: Combine and optimize final context
      const assembledContext = this.combineContexts(
        userMessage,
        ragResults,
        compressedHistory,
      );

      const duration = performance.now() - startTime;
      console.log('Context assembly completed', {
        duration: `${duration.toFixed(2)}ms`,
        ragChunks: assembledContext.ragChunksIncluded,
        conversationMessages: assembledContext.conversationMessagesIncluded,
        estimatedTokens: assembledContext.estimatedTokens,
      });

      return assembledContext;
    } catch (error) {
      console.error('Context assembly failed', { error });

      // Fallback to conversation-only context
      return this.createFallbackContext(userMessage, conversationHistory);
    }
  }

  /**
   * Extracts relevant context from recent conversation for RAG query enhancement
   */
  private extractConversationalContext(history: Content[]): string {
    if (!history || history.length === 0) return '';

    // Take last 3-5 messages for context
    const recentMessages = history.slice(-5);
    const contextParts: string[] = [];

    for (const content of recentMessages) {
      if (content.role === 'user') {
        const text = this.extractTextFromContent(content);
        if (text && text.length > 10) {
          contextParts.push(`User: ${text.substring(0, 200)}`);
        }
      } else if (content.role === 'model') {
        const text = this.extractTextFromContent(content);
        if (text && text.length > 10) {
          contextParts.push(`Assistant: ${text.substring(0, 150)}`);
        }
      }
    }

    return contextParts.length > 0
      ? `Recent conversation context:\n${contextParts.join('\n')}`
      : '';
  }

  /**
   * Extracts text content from a Content object
   */
  private extractTextFromContent(content: Content): string {
    if (!content.parts) return '';

    return content.parts
      .filter((part) => part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
  }

  /**
   * Gets optimized conversation history based on available token budget
   */
  private async getOptimizedConversationHistory(
    history: Content[],
    ragChunksCount: number,
  ): Promise<{ content: Content[]; compressionLevel: string }> {
    // Calculate token budget for conversation
    const ragTokenBudget = ragChunksCount * 500; // Estimate 500 tokens per chunk
    const conversationTokenBudget = Math.max(
      this.config.maxTotalTokens - ragTokenBudget - 1000, // Reserve for system prompt
      5000, // Minimum conversation budget
    );

    // Use ChatRecordingService's advanced compression with flexible token budget
    try {
      const optimizedResult =
        await this.chatRecordingService.getOptimizedHistoryForPrompt(
          conversationTokenBudget,
          false, // Don't include system info for performance
        );

      // Determine compression level based on metadata
      let compressionLevel = 'none';
      if (optimizedResult.metaInfo.compressionApplied) {
        const reductionRatio =
          1 -
          optimizedResult.metaInfo.finalMessageCount /
            optimizedResult.metaInfo.originalMessageCount;
        if (reductionRatio < 0.3) {
          compressionLevel = 'minimal';
        } else if (reductionRatio < 0.6) {
          compressionLevel = 'moderate';
        } else {
          compressionLevel = 'aggressive';
        }
      }

      return {
        content: optimizedResult.history,
        compressionLevel,
      };
    } catch (error) {
      // Fallback to basic compression if ChatRecordingService fails
      console.warn(
        'Failed to get optimized history from ChatRecordingService, using fallback:',
        error,
      );

      if (history.length <= 10) {
        return { content: history, compressionLevel: 'none' };
      }

      if (history.length <= 20) {
        return { content: history.slice(-15), compressionLevel: 'minimal' };
      }

      if (history.length <= 50) {
        return { content: history.slice(-25), compressionLevel: 'moderate' };
      }

      // For very long histories, use aggressive compression
      return { content: history.slice(-20), compressionLevel: 'aggressive' };
    }
  }

  /**
   * Combines RAG chunks and conversation history into optimized context
   */
  private combineContexts(
    userMessage: string,
    ragResults: EnhancedQueryResult,
    conversationData: { content: Content[]; compressionLevel: string },
  ): AssembledContext {
    const contents: Content[] = [];
    let estimatedTokens = 0;

    // Add system context with RAG information first
    if (ragResults.sourceChunks && ragResults.sourceChunks.length > 0) {
      const ragContext = this.formatRAGContext(ragResults);
      contents.push({
        role: 'user',
        parts: [{ text: ragContext }],
      });
      estimatedTokens += this.estimateTokens(ragContext);
    }

    // Add conversation history
    for (const content of conversationData.content) {
      contents.push(content);
      estimatedTokens += this.estimateTokens(
        this.extractTextFromContent(content),
      );
    }

    // Current user message will be added by the calling code

    return {
      contents,
      estimatedTokens,
      ragChunksIncluded: ragResults.sourceChunks?.length || 0,
      conversationMessagesIncluded: conversationData.content.length,
      ragContextSummary: ragResults.content,
      compressionLevel: conversationData.compressionLevel as
        | 'none'
        | 'minimal'
        | 'moderate'
        | 'aggressive'
        | 'intelligent',
    };
  }

  /**
   * Formats RAG chunks into a coherent context for the LLM
   */
  private formatRAGContext(ragResults: EnhancedQueryResult): string {
    if (!ragResults.sourceChunks || ragResults.sourceChunks.length === 0) {
      return '';
    }

    const contextParts = [
      '## Relevant Knowledge Base Information',
      '',
      'The following information has been retrieved from the codebase and documentation to help answer your question:',
      '',
    ];

    ragResults.sourceChunks.forEach((chunk: RAGChunk, index: number) => {
      contextParts.push(
        `### Context ${index + 1}: ${chunk.metadata?.file?.path || 'Code Snippet'}`,
      );
      contextParts.push(
        `**Source:** ${chunk.metadata?.file?.path || 'Unknown'}`,
      );
      contextParts.push(
        `**Relevance:** ${((chunk as RAGChunk & { score?: number }).score || 0.8 * 100).toFixed(1)}%`,
      );
      contextParts.push('');
      contextParts.push('```');
      contextParts.push(chunk.content);
      contextParts.push('```');
      contextParts.push('');
    });

    contextParts.push('## End of Knowledge Base Information');
    contextParts.push('');

    return contextParts.join('\n');
  }

  /**
   * Simple token estimation (should be replaced with proper tokenizer)
   */
  private estimateTokens(text: string): number {
    // Very rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  /**
   * Creates a fallback context when RAG fails
   */
  private createFallbackContext(
    userMessage: string,
    conversationHistory: Content[],
  ): AssembledContext {
    console.warn('Using fallback context due to RAG failure');

    return {
      contents: conversationHistory.slice(-10), // Last 10 messages
      estimatedTokens: this.estimateTokens(
        conversationHistory
          .slice(-10)
          .map((c) => this.extractTextFromContent(c))
          .join(' '),
      ),
      ragChunksIncluded: 0,
      conversationMessagesIncluded: Math.min(conversationHistory.length, 10),
      compressionLevel: 'moderate',
    };
  }

  /**
   * Updates configuration at runtime
   */
  updateConfig(newConfig: Partial<PromptContextConfig>): void {
    Object.assign(this.config, newConfig);
    console.log('Configuration updated', { config: this.config });
  }

  /**
   * Gets current configuration
   */
  getConfig(): PromptContextConfig {
    return { ...this.config };
  }
}
