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
      }
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
      const contextResult = await this.dualContextService.processWithOptimalContext(
        'general_query',
        userMessage,
        false // Not a tool execution
      );
      
      this.currentContextType = contextResult.contextType;

      // Step 2: Get current conversation history (curated)
      const conversationHistory = this.geminiChat.getHistory(true);

      // Step 3: Assemble optimized context using the appropriate context manager
      const assembledContext = await contextResult.contextManager.assembleContext(
        userMessage,
        conversationHistory,
        prompt_id,
      );

      // Step 4: Replace the naive history concatenation with smart context
      // Instead of: this.getHistory(true).concat(userContent)
      // We use: assembledContext.contents.concat(userContent)

      // Create a temporary enhanced history
      const enhancedHistory = [...assembledContext.contents];

      // Step 5: Temporarily replace the chat's history with enhanced context
      // We'll use a technique to inject the context without permanently modifying history
      const originalHistory = this.getOriginalHistory();
      this.setTemporaryHistory(enhancedHistory);

      // Step 6: Send message with enhanced context
      const response = await this.geminiChat.sendMessage(params, prompt_id);

      // Step 7: Restore original history and add the new exchange
      this.restoreHistory(
        originalHistory,
        userMessage,
        response,
      );

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
      console.error('Dual-context RAG-enhanced message failed, falling back to basic chat', {
        error,
        promptId: prompt_id,
        contextType: this.currentContextType,
      });

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
      const contextResult = await this.dualContextService.processWithOptimalContext(
        'streaming_query',
        userMessage,
        false // Not a tool execution
      );
      
      this.currentContextType = contextResult.contextType;

      const conversationHistory = this.geminiChat.getHistory(true);
      const assembledContext = await contextResult.contextManager.assembleContext(
        userMessage,
        conversationHistory,
        prompt_id,
      );

      // Temporarily enhance history
      const originalHistory = this.getOriginalHistory();
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

      // Note: We'll restore history after streaming completes
      // This is handled by the async generator wrapper

      return this.wrapStreamWithHistoryRestore(
        streamGenerator,
        originalHistory,
        userMessage,
      );
    } catch (error) {
      console.error('Dual-context RAG-enhanced streaming failed, falling back', { 
        error,
        contextType: this.currentContextType,
      });
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
    } finally {
      // Restore history and add the exchange
      if (lastResponse) {
        this.restoreHistory(originalHistory, userMessage, lastResponse);
      }
    }
  }

  /**
   * Gets the current original history from GeminiChat
   */
  private getOriginalHistory(): Content[] {
    return this.geminiChat.getHistory(false); // Get uncurated full history
  }

  /**
   * Temporarily sets the history in GeminiChat for enhanced context
   */
  private setTemporaryHistory(enhancedHistory: Content[]): void {
    // Access the private history property - this is a POC approach
    // In production, we would modify GeminiChat to accept context directly
    (this.geminiChat as unknown as { history: Content[] }).history = [
      ...enhancedHistory,
    ];
  }

  /**
   * Restores the original history and properly adds the new exchange
   */
  private restoreHistory(
    originalHistory: Content[],
    userMessage: string,
    response: GenerateContentResponse,
  ): void {
    // Restore original history
    (this.geminiChat as unknown as { history: Content[] }).history = [
      ...originalHistory,
    ];

    // Add the user message to history
    (this.geminiChat as unknown as { history: Content[] }).history.push({
      role: 'user',
      parts: [{ text: userMessage }],
    });

    // Add the assistant response to history if valid
    if (response.candidates && response.candidates[0]?.content) {
      (this.geminiChat as unknown as { history: Content[] }).history.push(
        response.candidates[0].content,
      );
    }
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
  updateRAGConfig(
    contextType: ContextType,
    config: Parameters<PromptContextManager['updateConfig']>[0],
  ): void {
    // Get the appropriate context manager and update its configuration
    this.dualContextService.getContextManager(contextType).then(contextManager => {
      contextManager.updateConfig(config);
    }).catch(error => {
      console.error('Failed to update RAG config', { error, contextType });
    });
  }

  /**
   * Gets current dual-context RAG configuration
   */
  async getRAGConfig(contextType: ContextType = this.currentContextType): Promise<ReturnType<PromptContextManager['getConfig']>> {
    try {
      const contextManager = await this.dualContextService.getContextManager(contextType);
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
      const contextResult = await this.dualContextService.processWithOptimalContext(
        'tool_execution',
        userMessage,
        true // This IS a tool execution
      );
      
      this.currentContextType = ContextType.SHORT_TERM_MEMORY; // Force short-term for tools

      // Get minimal conversation history for tool context
      const conversationHistory = this.geminiChat.getHistory(true).slice(-5); // Only last 5 exchanges

      // Use short-term context manager for optimized tool execution
      const assembledContext = await contextResult.contextManager.assembleContext(
        userMessage,
        conversationHistory,
        prompt_id,
      );

      // Apply enhanced context temporarily
      const originalHistory = this.getOriginalHistory();
      this.setTemporaryHistory([...assembledContext.contents]);

      // Execute with optimized short-term context
      const response = await this.geminiChat.sendMessage(params, prompt_id);

      // Restore history
      this.restoreHistory(originalHistory, userMessage, response);

      const duration = performance.now() - startTime;
      console.log('Tool execution completed with short-term context optimization', {
        duration: `${duration.toFixed(2)}ms`,
        contextType: this.currentContextType,
        maxTokens: contextResult.maxTokens,
        ragChunks: assembledContext.ragChunksIncluded,
        toolOptimized: true,
        promptId: prompt_id,
      });

      return response;
    } catch (error) {
      console.error('Tool execution with dual-context failed, falling back', {
        error,
        promptId: prompt_id,
      });

      return this.geminiChat.sendMessage(params, prompt_id);
    }
  }

  /**
   * Gets dual-context metrics for monitoring
   */
  getDualContextMetrics(): ReturnType<DualContextIntegrationService['getContextMetrics']> {
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
   */
  async indexCurrentConversation(): Promise<void> {
    try {
      const history = this.getHistory(false);

      // Extract meaningful exchanges for indexing
      const conversationChunks = this.extractConversationChunks(history);

      // Index each meaningful chunk
      for (const chunk of conversationChunks) {
        await this.ragService.indexContent([
          {
            id: `conversation_${Date.now()}_${Math.random()}`,
            content: chunk.content,
            type: ChunkType.CONVERSATION,
            metadata: chunk.metadata,
          },
        ]);
      }

      console.log(
        `Indexed ${conversationChunks.length} conversation chunks for future retrieval`,
      );
    } catch (error) {
      console.error('Failed to index conversation', { error });
    }
  }

  /**
   * Extracts meaningful chunks from conversation history for indexing
   */
  private extractConversationChunks(history: Content[]): Array<{
    content: string;
    metadata: Record<string, unknown>;
  }> {
    const chunks: Array<{
      content: string;
      metadata: Record<string, unknown>;
    }> = [];

    for (let i = 0; i < history.length - 1; i += 2) {
      const userMsg = history[i];
      const assistantMsg = history[i + 1];

      if (userMsg?.role === 'user' && assistantMsg?.role === 'model') {
        const userText = this.extractTextFromContent(userMsg);
        const assistantText = this.extractTextFromContent(assistantMsg);

        if (userText.length > 20 && assistantText.length > 50) {
          chunks.push({
            content: `User Question: ${userText}\n\nAssistant Response: ${assistantText}`,
            metadata: {
              type: 'conversation_exchange',
              timestamp: new Date().toISOString(),
              userQueryLength: userText.length,
              responseLength: assistantText.length,
            },
          });
        }
      }
    }

    return chunks;
  }

  /**
   * Extracts text from Content objects
   */
  private extractTextFromContent(content: Content): string {
    if (!content.parts) return '';

    return content.parts
      .filter((part) => part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
  }
}
