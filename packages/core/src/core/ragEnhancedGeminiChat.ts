/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  GenerateContentResponse,
  SendMessageParameters,
} from '@google/genai';
import { GeminiChat } from './geminiChat.js';
import { PromptContextManager } from '../services/promptContextManager.js';
import { DualContextIntegrationService } from '../services/dualContextIntegrationService.js';
import { ContextType } from '../config/dualContextExample.js';
import { RAGService } from '../rag/ragService.js';
import { partListUnionToString } from './geminiRequest.js';
import { ChunkType } from '../rag/types.js';
import { ChatRecordingService } from '../services/chatRecordingService.js';
import { Config } from '../config/config.js';

/**
 * Enhanced GeminiChat with RAG integration and dual-context token management.
 *
 * This service wraps the original GeminiChat and enhances it with:
 * - RAG-powered context assembly for external knowledge integration
 * - Dual-context strategy leveraging Gemini's full 1M token capacity
 * - Intelligent switching between long-term memory (analysis) and short-term memory (tool execution)
 * - Eliminates manual history management with automated context optimization
 */
export class RAGEnhancedGeminiChat {
  private readonly dualContextService: DualContextIntegrationService;
  private currentContextType: ContextType = ContextType.LONG_TERM_MEMORY;

  constructor(
    private readonly geminiChat: GeminiChat,
    private readonly ragService: RAGService,
    private readonly chatRecordingService: ChatRecordingService,
    private readonly config: Config,
  ) {
    // Create a minimal RAGChatIntegrationService-compatible wrapper
    const ragChatIntegrationWrapper = {
      ragService: this.ragService,
      chatRecordingService: this.chatRecordingService,
    };

    this.dualContextService = new DualContextIntegrationService(
      config,
      ragChatIntegrationWrapper as any, // Type assertion for compatibility
    );

    console.log(
      'RAGEnhancedGeminiChat initialized with dual-context token management strategy',
      {
        longTermCapacity: '1M tokens',
        shortTermCapacity: '28K tokens',
        smartSwitching: true,
      },
    );
  }

  /**
   * Enhanced sendMessage with dual-context RAG integration
   */
  async sendMessage(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<GenerateContentResponse> {
    const startTime = performance.now();

    try {
      const userMessage = partListUnionToString(params.message);

      // Step 1: Determine optimal context strategy based on message content
      const contextResult =
        await this.dualContextService.processWithOptimalContext(
          'general_query',
          userMessage,
          false, // Not a tool execution
        );

      this.currentContextType = contextResult.contextType;

      // Step 2: Get current conversation history (curated)
      const conversationHistory = this.geminiChat.getHistory(true);

      // Step 3: Assemble optimized context using the appropriate context manager
      const assembledContext =
        await contextResult.contextManager.assembleContext(
          userMessage,
          conversationHistory,
          prompt_id,
        );

      // Step 4: Store original history and apply enhanced context
      const originalHistory = this.getOriginalHistory();
      let response: GenerateContentResponse;

      try {
        // Step 5: Temporarily replace the chat's history with enhanced context
        this.setTemporaryHistory([...assembledContext.contents]);

        // Step 6: Send message with enhanced context
        response = await this.geminiChat.sendMessage(params, prompt_id);
      } finally {
        // Step 7: Always restore original history and add the new exchange
        // This ensures history consistency even if errors occur
        try {
          if (response!) {
            this.restoreHistory(originalHistory, userMessage, response);
          } else {
            // If no response, just restore original history
            this.geminiChat.setHistory(originalHistory);
          }
        } catch (restoreError) {
          console.error('Critical: Failed to restore chat history', {
            restoreError,
            promptId: prompt_id,
            originalHistoryLength: originalHistory.length,
          });
          // Attempt emergency history restoration
          this.geminiChat.setHistory(originalHistory);
        }
      }

      const duration = performance.now() - startTime;
      console.log('Dual-context RAG-enhanced message sent successfully', {
        duration: `${duration.toFixed(2)}ms`,
        contextType: this.currentContextType,
        maxTokens: contextResult.maxTokens,
        ragChunks: assembledContext.ragChunksIncluded,
        conversationMessages: assembledContext.conversationMessagesIncluded,
        estimatedTokens: assembledContext.estimatedTokens,
        compressionLevel: assembledContext.compressionLevel,
        promptId: prompt_id,
      });

      return response;
    } catch (error) {
      console.error(
        'Dual-context RAG-enhanced message failed, falling back to basic chat',
        {
          error,
          promptId: prompt_id,
          contextType: this.currentContextType,
        },
      );

      // Fallback to original behavior
      return this.geminiChat.sendMessage(params, prompt_id);
    }
  }

  /**
   * Enhanced sendMessageStream with dual-context RAG integration
   */
  async sendMessageStream(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<AsyncGenerator<GenerateContentResponse, void, unknown>> {
    try {
      const userMessage = partListUnionToString(params.message);

      // Apply dual-context RAG enhancement for streaming
      const contextResult =
        await this.dualContextService.processWithOptimalContext(
          'streaming_query',
          userMessage,
          false, // Not a tool execution
        );

      this.currentContextType = contextResult.contextType;

      const conversationHistory = this.geminiChat.getHistory(true);
      const assembledContext =
        await contextResult.contextManager.assembleContext(
          userMessage,
          conversationHistory,
          prompt_id,
        );

      // Store original history for restoration
      const originalHistory = this.getOriginalHistory();

      // Temporarily enhance history with better error handling
      try {
        this.setTemporaryHistory([...assembledContext.contents]);

        // Stream with enhanced context
        const streamGenerator = await this.geminiChat.sendMessageStream(
          params,
          prompt_id,
        );

        console.log('Dual-context streaming initiated', {
          contextType: this.currentContextType,
          maxTokens: contextResult.maxTokens,
          ragChunks: assembledContext.ragChunksIncluded,
          promptId: prompt_id,
        });

        // Return enhanced stream generator with proper history restoration
        return this.wrapStreamWithHistoryRestore(
          streamGenerator,
          originalHistory,
          userMessage,
        );
      } catch (streamError) {
        // Ensure history is restored even if streaming setup fails
        console.error('Failed to setup enhanced streaming, restoring history', {
          streamError,
          promptId: prompt_id,
        });

        try {
          this.geminiChat.setHistory(originalHistory);
        } catch (restoreError) {
          console.error(
            'Critical: Failed to restore history after stream error',
            {
              restoreError,
              originalError: streamError,
            },
          );
        }

        throw streamError;
      }
    } catch (error) {
      console.error(
        'Dual-context RAG-enhanced streaming failed, falling back',
        {
          error,
          contextType: this.currentContextType,
        },
      );
      return this.geminiChat.sendMessageStream(params, prompt_id);
    }
  }

  /**
   * Wraps the stream generator to restore history after completion
   */
  private async *wrapStreamWithHistoryRestore(
    streamGenerator: AsyncGenerator<GenerateContentResponse, void, unknown>,
    originalHistory: Content[],
    userMessage: string,
  ): AsyncGenerator<GenerateContentResponse, void, unknown> {
    let lastResponse: GenerateContentResponse | undefined;

    try {
      for await (const response of streamGenerator) {
        lastResponse = response;
        yield response;
      }
    } catch (streamError) {
      console.error(
        'Error during streaming, will still attempt history restoration',
        {
          streamError,
        },
      );
      throw streamError;
    } finally {
      // Always attempt to restore history, even if streaming failed
      try {
        if (lastResponse) {
          this.restoreHistory(originalHistory, userMessage, lastResponse);
        } else {
          // If no response received, just restore original history
          this.geminiChat.setHistory(originalHistory);
        }
      } catch (restoreError) {
        console.error('Critical: Failed to restore history after streaming', {
          restoreError,
          originalHistoryLength: originalHistory.length,
        });
        // Emergency restoration attempt
        try {
          this.geminiChat.setHistory(originalHistory);
        } catch (emergencyError) {
          console.error('Emergency history restoration failed', {
            emergencyError,
          });
        }
      }
    }
  }

  /**
   * Gets the current original history from GeminiChat (with caching optimization)
   */
  private getOriginalHistory(): Content[] {
    // Use uncurated history for better performance and accuracy
    const history = this.geminiChat.getHistory(false);

    // For large histories, we don't need to clone since we're only reading
    // The clone will happen in setHistory() when needed
    return history;
  }

  /**
   * Temporarily sets the history in GeminiChat for enhanced context
   * (Optimized to reduce unnecessary cloning)
   */
  private setTemporaryHistory(enhancedHistory: Content[]): void {
    // Use the public API - GeminiChat will handle the cloning internally
    // No need to clone here since we're passing ownership
    this.geminiChat.setHistory(enhancedHistory);
  }

  /**
   * Restores the original history and properly adds the new exchange
   * (Optimized for better performance)
   */
  private restoreHistory(
    originalHistory: Content[],
    userMessage: string,
    response: GenerateContentResponse,
  ): void {
    // Performance optimization: pre-allocate array size
    const updatedHistory = new Array(originalHistory.length + 2);

    // Copy original history efficiently
    for (let i = 0; i < originalHistory.length; i++) {
      updatedHistory[i] = originalHistory[i];
    }

    // Add the user message to history
    updatedHistory[originalHistory.length] = {
      role: 'user',
      parts: [{ text: userMessage }],
    };

    // Add the assistant response to history if valid
    if (response.candidates && response.candidates[0]?.content) {
      updatedHistory[originalHistory.length + 1] =
        response.candidates[0].content;
    } else {
      // If no valid response, trim the array to exclude the empty slot
      updatedHistory.length = originalHistory.length + 1;
    }

    // Restore history using public API
    this.geminiChat.setHistory(updatedHistory);
  }

  /**
   * Provides access to the original GeminiChat for compatibility
   */
  getOriginalChat(): GeminiChat {
    return this.geminiChat;
  }

  /**
   * Gets history - delegates to original chat
   */
  getHistory(curated: boolean = false): Content[] {
    return this.geminiChat.getHistory(curated);
  }

  /**
   * Updates the dual-context RAG configuration
   */
  async updateRAGConfig(
    contextType: ContextType,
    config: Parameters<PromptContextManager['updateConfig']>[0],
  ): Promise<void> {
    try {
      // Get the appropriate context manager and update its configuration
      const contextManager =
        await this.dualContextService.getContextManager(contextType);
      await contextManager.updateConfig(config);

      console.log('RAG config updated successfully', {
        contextType,
        configKeys: Object.keys(config),
      });
    } catch (error) {
      console.error('Failed to update RAG config', {
        error,
        contextType,
        configKeys: config ? Object.keys(config) : 'undefined',
      });
      throw error; // Re-throw to allow caller to handle
    }
  }

  /**
   * Gets current dual-context RAG configuration
   */
  async getRAGConfig(
    contextType: ContextType = this.currentContextType,
  ): Promise<ReturnType<PromptContextManager['getConfig']>> {
    try {
      const contextManager =
        await this.dualContextService.getContextManager(contextType);
      return contextManager.getConfig();
    } catch (error) {
      console.error('Failed to get RAG config', { error, contextType });
      throw error;
    }
  }

  /**
   * Specialized method for tool execution with short-term context optimization
   */
  async sendMessageForToolExecution(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<GenerateContentResponse> {
    const startTime = performance.now();

    try {
      const userMessage = partListUnionToString(params.message);

      // Force short-term context for tool execution
      const contextResult =
        await this.dualContextService.processWithOptimalContext(
          'tool_execution',
          userMessage,
          true, // This IS a tool execution
        );

      this.currentContextType = ContextType.SHORT_TERM_MEMORY; // Force short-term for tools

      // Get minimal conversation history for tool context
      const conversationHistory = this.geminiChat.getHistory(true).slice(-5); // Only last 5 exchanges

      // Use short-term context manager for optimized tool execution
      const assembledContext =
        await contextResult.contextManager.assembleContext(
          userMessage,
          conversationHistory,
          prompt_id,
        );

      // Store original history and apply enhanced context with proper error handling
      const originalHistory = this.getOriginalHistory();
      let response: GenerateContentResponse;

      try {
        this.setTemporaryHistory([...assembledContext.contents]);

        // Execute with optimized short-term context
        response = await this.geminiChat.sendMessage(params, prompt_id);
      } finally {
        // Always restore history, even if tool execution fails
        try {
          if (response!) {
            this.restoreHistory(originalHistory, userMessage, response);
          } else {
            this.geminiChat.setHistory(originalHistory);
          }
        } catch (restoreError) {
          console.error(
            'Critical: Failed to restore history after tool execution',
            {
              restoreError,
              promptId: prompt_id,
              toolExecution: true,
            },
          );
          // Emergency restoration
          this.geminiChat.setHistory(originalHistory);
        }
      }

      const duration = performance.now() - startTime;
      console.log(
        'Tool execution completed with short-term context optimization',
        {
          duration: `${duration.toFixed(2)}ms`,
          contextType: this.currentContextType,
          maxTokens: contextResult.maxTokens,
          ragChunks: assembledContext.ragChunksIncluded,
          toolOptimized: true,
          promptId: prompt_id,
        },
      );

      return response;
    } catch (error) {
      console.error('Tool execution with dual-context failed, falling back', {
        error,
        promptId: prompt_id,
        toolExecution: true,
      });

      return this.geminiChat.sendMessage(params, prompt_id);
    }
  }

  /**
   * Gets dual-context metrics for monitoring
   */
  getDualContextMetrics(): ReturnType<
    DualContextIntegrationService['getContextMetrics']
  > {
    return this.dualContextService.getContextMetrics();
  }

  /**
   * Gets current context type being used
   */
  getCurrentContextType(): ContextType {
    return this.currentContextType;
  }

  /**
   * Forces RAG indexing of current conversation for future retrieval
   * (Enhanced with better error handling and performance monitoring)
   */
  async indexCurrentConversation(): Promise<void> {
    const startTime = performance.now();

    try {
      const history = this.getHistory(false);

      if (!history || history.length === 0) {
        console.log('No conversation history to index');
        return;
      }

      // Extract meaningful exchanges for indexing
      const conversationChunks = this.extractConversationChunks(history);

      if (conversationChunks.length === 0) {
        console.log('No meaningful conversation chunks found for indexing');
        return;
      }

      // Index each meaningful chunk with progress tracking
      let indexedCount = 0;
      const errors: Error[] = [];

      for (const chunk of conversationChunks) {
        try {
          await this.ragService.indexContent([
            {
              id: `conversation_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              content: chunk.content,
              type: ChunkType.CONVERSATION,
              metadata: {
                ...chunk.metadata,
                indexedAt: new Date().toISOString(),
                conversationId: `conv_${Date.now()}`,
              },
            },
          ]);
          indexedCount++;
        } catch (chunkError) {
          console.error('Failed to index conversation chunk', {
            chunkError,
            chunkPreview: chunk.content.substring(0, 100),
          });
          errors.push(chunkError as Error);
        }
      }

      const duration = performance.now() - startTime;
      console.log(
        `Conversation indexing completed: ${indexedCount}/${conversationChunks.length} chunks indexed`,
        {
          duration: `${duration.toFixed(2)}ms`,
          successRate: `${((indexedCount / conversationChunks.length) * 100).toFixed(1)}%`,
          errors: errors.length,
        },
      );

      if (errors.length > 0 && indexedCount === 0) {
        throw new Error(
          `Failed to index any conversation chunks. ${errors.length} errors encountered.`,
        );
      }
    } catch (error) {
      console.error('Failed to index conversation', {
        error,
        duration: `${(performance.now() - startTime).toFixed(2)}ms`,
      });
      throw error;
    }
  }

  /**
   * Extracts meaningful chunks from conversation history for indexing
   * (Enhanced with better filtering and metadata)
   */
  private extractConversationChunks(history: Content[]): Array<{
    content: string;
    metadata: Record<string, unknown>;
  }> {
    const chunks: Array<{
      content: string;
      metadata: Record<string, unknown>;
    }> = [];

    if (!history || history.length < 2) {
      return chunks;
    }

    // Process exchanges in pairs (user + assistant)
    for (let i = 0; i < history.length - 1; i += 2) {
      const userMsg = history[i];
      const assistantMsg = history[i + 1];

      // Validate message structure
      if (!userMsg?.role || !assistantMsg?.role) {
        continue;
      }

      if (userMsg.role === 'user' && assistantMsg.role === 'model') {
        const userText = this.extractTextFromContent(userMsg);
        const assistantText = this.extractTextFromContent(assistantMsg);

        // Enhanced filtering: check for meaningful content
        if (
          userText.length > 20 &&
          assistantText.length > 50 &&
          !this.isLowQualityContent(userText, assistantText)
        ) {
          chunks.push({
            content: `User Question: ${userText}\n\nAssistant Response: ${assistantText}`,
            metadata: {
              type: 'conversation_exchange',
              timestamp: new Date().toISOString(),
              userQueryLength: userText.length,
              responseLength: assistantText.length,
              exchangeIndex: Math.floor(i / 2),
              quality: this.assessContentQuality(userText, assistantText),
            },
          });
        }
      }
    }

    return chunks;
  }

  /**
   * Checks if content is low quality and shouldn't be indexed
   */
  private isLowQualityContent(
    userText: string,
    assistantText: string,
  ): boolean {
    // Filter out common low-quality patterns
    const lowQualityPatterns = [
      /^(ok|yes|no|thanks?|sure|great)$/i,
      /^(error|failed|undefined)$/i,
      /^\s*$/,
    ];

    return lowQualityPatterns.some(
      (pattern) => pattern.test(userText) || pattern.test(assistantText),
    );
  }

  /**
   * Assesses content quality for metadata
   */
  private assessContentQuality(
    userText: string,
    assistantText: string,
  ): string {
    const totalLength = userText.length + assistantText.length;
    const hasCodeBlocks = /```/.test(assistantText);
    const hasStructuredContent = /\n\s*[-*]\s/.test(assistantText);

    if (totalLength > 1000 && (hasCodeBlocks || hasStructuredContent)) {
      return 'high';
    } else if (totalLength > 500) {
      return 'medium';
    } else {
      return 'low';
    }
  }

  /**
   * Extracts text from Content objects (with enhanced safety)
   */
  private extractTextFromContent(content: Content): string {
    if (!content || !content.parts || !Array.isArray(content.parts)) {
      return '';
    }

    return content.parts
      .filter((part) => part && typeof part === 'object' && part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
  }
}
