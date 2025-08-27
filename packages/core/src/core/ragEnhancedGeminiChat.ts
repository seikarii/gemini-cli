/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, GenerateContentResponse, SendMessageParameters } from '@google/genai';
import { GeminiChat } from './geminiChat.js';
import { PromptContextManager } from '../services/promptContextManager.js';
import { RAGService } from '../rag/ragService.js';
import { partListUnionToString } from './geminiRequest.js';
import { ChunkType } from '../rag/types.js';
import { ChatRecordingService } from '../services/chatRecordingService.js';
import { Config } from '../config/config.js';

/**
 * Enhanced GeminiChat with RAG integration capabilities.
 * 
 * This service wraps the original GeminiChat and enhances it with
 * RAG-powered context assembly, eliminating the need for manual
 * history management and providing intelligent external knowledge integration.
 */
export class RAGEnhancedGeminiChat {
  private readonly promptContextManager: PromptContextManager;

  constructor(
    private readonly geminiChat: GeminiChat,
    private readonly ragService: RAGService,
    private readonly chatRecordingService: ChatRecordingService,
    private readonly config: Config
  ) {
    // Initialize the PromptContextManager with smart defaults
    this.promptContextManager = new PromptContextManager(
      ragService,
      chatRecordingService,
      config,
      {
        maxTotalTokens: 42000, // Conservative for most Gemini models
        maxRAGChunks: 6,
        ragRelevanceThreshold: 0.65,
        ragWeight: 0.35, // 35% RAG, 65% conversation
        prioritizeRecentConversation: true,
        useConversationalContext: true
      }
    );

    console.log('RAGEnhancedGeminiChat initialized with intelligent context management');
  }

  /**
   * Enhanced sendMessage that automatically integrates RAG context
   * with conversational history before sending to the model.
   */
  async sendMessage(
    params: SendMessageParameters,
    prompt_id: string
  ): Promise<GenerateContentResponse> {
    const startTime = performance.now();

    try {
      // Step 1: Get current conversation history (curated)
      const conversationHistory = this.geminiChat.getHistory(true);

      // Step 2: Assemble optimized context using RAG + conversation
      const assembledContext = await this.promptContextManager.assembleContext(
        partListUnionToString(params.message),
        conversationHistory,
        prompt_id
      );

      // Step 3: Replace the naive history concatenation with smart context
      // Instead of: this.getHistory(true).concat(userContent)
      // We use: assembledContext.contents.concat(userContent)
      
      // Create a temporary enhanced history
      const enhancedHistory = [...assembledContext.contents];

      // Step 4: Temporarily replace the chat's history with enhanced context
      // We'll use a technique to inject the context without permanently modifying history
      const originalHistory = this.getOriginalHistory();
      this.setTemporaryHistory(enhancedHistory);

      // Step 5: Send message with enhanced context
      const response = await this.geminiChat.sendMessage(params, prompt_id);

      // Step 6: Restore original history and add the new exchange
      this.restoreHistory(originalHistory, partListUnionToString(params.message), response);

      const duration = performance.now() - startTime;
      console.log('RAG-enhanced message sent successfully', {
        duration: `${duration.toFixed(2)}ms`,
        ragChunks: assembledContext.ragChunksIncluded,
        conversationMessages: assembledContext.conversationMessagesIncluded,
        estimatedTokens: assembledContext.estimatedTokens,
        compressionLevel: assembledContext.compressionLevel,
        promptId: prompt_id
      });

      return response;

    } catch (error) {
      console.error('RAG-enhanced message failed, falling back to basic chat', { 
        error, 
        promptId: prompt_id 
      });
      
      // Fallback to original behavior
      return this.geminiChat.sendMessage(params, prompt_id);
    }
  }

  /**
   * Enhanced sendMessageStream with RAG integration
   */
  async sendMessageStream(
    params: SendMessageParameters,
    prompt_id: string
  ): Promise<AsyncGenerator<GenerateContentResponse, void, unknown>> {
    
    try {
      // Apply same RAG enhancement for streaming
      const conversationHistory = this.geminiChat.getHistory(true);
      const assembledContext = await this.promptContextManager.assembleContext(
        partListUnionToString(params.message),
        conversationHistory,
        prompt_id
      );

      // Temporarily enhance history
      const originalHistory = this.getOriginalHistory();
      this.setTemporaryHistory([...assembledContext.contents]);

      // Stream with enhanced context
      const streamGenerator = await this.geminiChat.sendMessageStream(params, prompt_id);

      // Note: We'll restore history after streaming completes
      // This is handled by the async generator wrapper

      return this.wrapStreamWithHistoryRestore(
        streamGenerator, 
        originalHistory, 
        partListUnionToString(params.message)
      );

    } catch (error) {
      console.error('RAG-enhanced streaming failed, falling back', { error });
      return this.geminiChat.sendMessageStream(params, prompt_id);
    }
  }

  /**
   * Wraps the stream generator to restore history after completion
   */
  private async* wrapStreamWithHistoryRestore(
    streamGenerator: AsyncGenerator<GenerateContentResponse, void, unknown>,
    originalHistory: Content[],
    userMessage: string
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
    (this.geminiChat as unknown as { history: Content[] }).history = [...enhancedHistory];
  }

  /**
   * Restores the original history and properly adds the new exchange
   */
  private restoreHistory(
    originalHistory: Content[], 
    userMessage: string, 
    response: GenerateContentResponse
  ): void {
    
    // Restore original history
    (this.geminiChat as unknown as { history: Content[] }).history = [...originalHistory];

    // Add the user message to history
    (this.geminiChat as unknown as { history: Content[] }).history.push({
      role: 'user',
      parts: [{ text: userMessage }]
    });

    // Add the assistant response to history if valid
    if (response.candidates && response.candidates[0]?.content) {
      (this.geminiChat as unknown as { history: Content[] }).history.push(response.candidates[0].content);
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
   * Updates the RAG context configuration
   */
  updateRAGConfig(config: Parameters<PromptContextManager['updateConfig']>[0]): void {
    this.promptContextManager.updateConfig(config);
  }

  /**
   * Gets current RAG configuration
   */
  getRAGConfig(): ReturnType<PromptContextManager['getConfig']> {
    return this.promptContextManager.getConfig();
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
        await this.ragService.indexContent([{
          id: `conversation_${Date.now()}_${Math.random()}`,
          content: chunk.content,
          type: ChunkType.CONVERSATION,
          metadata: chunk.metadata
        }]);
      }

      console.log(`Indexed ${conversationChunks.length} conversation chunks for future retrieval`);
      
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
    
    const chunks: Array<{ content: string; metadata: Record<string, unknown> }> = [];
    
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
              responseLength: assistantText.length
            }
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
      .filter(part => part.text)
      .map(part => part.text)
      .join(' ')
      .trim();
  }
}
