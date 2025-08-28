/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  GenerateContentResponse,
  SendMessageParameters,
  GenerateContentConfig,
} from '@google/genai';
import { GeminiChat } from '../core/geminiChat.js';
import {
  PromptContextManager,
  AssembledContext,
} from './promptContextManager.js';
import { RAGService } from '../rag/ragService.js';
import { ChatRecordingService } from './chatRecordingService.js';
import { Config } from '../config/config.js';
import { ContentGenerator } from '../core/contentGenerator.js';
import { partListUnionToString } from '../core/geminiRequest.js';

/**
 * Factory service for creating RAG-enhanced chat instances and managing
 * the integration between RAG system and conversational AI.
 *
 * This service provides a clean integration point that eliminates the need
 * for the "CRITICAL FIX" manual history reversal in GeminiChat by using
 * intelligent context assembly from RAG + conversation.
 */
export class RAGChatIntegrationService {
  private promptContextManager?: PromptContextManager;
  private isRAGEnabled: boolean = false;

  constructor(
    private readonly ragService: RAGService,
    private readonly chatRecordingService: ChatRecordingService,
    private readonly config: Config,
  ) {
    this.initializeRAGIntegration();
  }

  /**
   * Initializes RAG integration if available
   */
  private async initializeRAGIntegration(): Promise<void> {
    try {
      await this.ragService.initialize();

      this.promptContextManager = new PromptContextManager(
        this.ragService,
        this.chatRecordingService,
        this.config,
        {
          maxTotalTokens: 28000,
          maxRAGChunks: 6,
          ragRelevanceThreshold: 0.65,
          ragWeight: 0.35,
          prioritizeRecentConversation: true,
          useConversationalContext: true,
        },
      );

      this.isRAGEnabled = true;
      console.log('RAG integration initialized successfully');
    } catch (error) {
      console.warn('RAG integration failed to initialize, using basic chat', {
        error,
      });
      this.isRAGEnabled = false;
    }
  }

  /**
   * Creates an enhanced GeminiChat with intelligent context management.
   * This replaces the manual history management in the original GeminiChat.
   */
  createEnhancedChat(
    contentGenerator: ContentGenerator,
    generationConfig: GenerateContentConfig = {},
    initialHistory: Content[] = [],
  ): EnhancedGeminiChatProxy {
    return new EnhancedGeminiChatProxy(
      this.config,
      contentGenerator,
      generationConfig,
      initialHistory,
      this.promptContextManager,
      this.isRAGEnabled,
    );
  }

  /**
   * Updates RAG configuration for all future chats
   */
  updateRAGConfig(
    config: Parameters<PromptContextManager['updateConfig']>[0],
  ): void {
    if (this.promptContextManager) {
      this.promptContextManager.updateConfig(config);
    }
  }

  /**
   * Gets current RAG status and configuration
   */
  getRAGStatus(): {
    enabled: boolean;
    config?: ReturnType<PromptContextManager['getConfig']>;
  } {
    return {
      enabled: this.isRAGEnabled,
      config: this.promptContextManager?.getConfig(),
    };
  }
}

/**
 * Enhanced GeminiChat proxy that integrates RAG context assembly
 * while maintaining full compatibility with the original GeminiChat interface.
 */
export class EnhancedGeminiChatProxy extends GeminiChat {
  constructor(
    config: Config,
    contentGenerator: ContentGenerator,
    generationConfig: GenerateContentConfig,
    initialHistory: Content[],
    private readonly promptContextManager?: PromptContextManager,
    private readonly ragEnabled: boolean = false,
  ) {
    super(config, contentGenerator, generationConfig, initialHistory);
  }

  /**
   * Enhanced sendMessage with intelligent RAG context assembly.
   * This eliminates the need for the manual "CRITICAL FIX" history reversal.
   */
  async sendMessage(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<GenerateContentResponse> {
    if (!this.ragEnabled || !this.promptContextManager) {
      // Fallback to original behavior
      return super.sendMessage(params, prompt_id);
    }

    const startTime = performance.now();

    try {
      // Step 1: Get current conversation history
      const conversationHistory = this.getHistory(true);

      // Step 2: Use PromptContextManager to create optimized context
      const assembledContext = await this.promptContextManager.assembleContext(
        partListUnionToString(params.message),
        conversationHistory,
        prompt_id,
      );

      // Step 3: Create enhanced sendMessage parameters with RAG context
      const enhancedParams = this.createEnhancedParameters(
        params,
        assembledContext,
      );

      // Step 4: Call original sendMessage with enhanced context
      const response = await super.sendMessage(enhancedParams, prompt_id);

      const duration = performance.now() - startTime;
      console.log('RAG-enhanced sendMessage completed', {
        duration: `${duration.toFixed(2)}ms`,
        ragChunks: assembledContext.ragChunksIncluded,
        conversationMessages: assembledContext.conversationMessagesIncluded,
        estimatedTokens: assembledContext.estimatedTokens,
        promptId: prompt_id,
      });

      return response;
    } catch (error) {
      console.error('RAG enhancement failed, falling back to basic chat', {
        error,
        promptId: prompt_id,
      });

      // Fallback to original implementation
      return super.sendMessage(params, prompt_id);
    }
  }

  /**
   * Enhanced sendMessageStream with RAG integration
   */
  async sendMessageStream(
    params: SendMessageParameters,
    prompt_id: string,
  ): Promise<AsyncGenerator<GenerateContentResponse, void, unknown>> {
    if (!this.ragEnabled || !this.promptContextManager) {
      return super.sendMessageStream(params, prompt_id);
    }

    try {
      const conversationHistory = this.getHistory(true);
      const assembledContext = await this.promptContextManager.assembleContext(
        partListUnionToString(params.message),
        conversationHistory,
        prompt_id,
      );

      const enhancedParams = this.createEnhancedParameters(
        params,
        assembledContext,
      );

      return super.sendMessageStream(enhancedParams, prompt_id);
    } catch (error) {
      console.error('RAG-enhanced streaming failed, falling back', { error });
      return super.sendMessageStream(params, prompt_id);
    }
  }

  /**
   * Creates enhanced parameters that include RAG context in the message
   */
  private createEnhancedParameters(
    originalParams: SendMessageParameters,
    assembledContext: ReturnType<
      PromptContextManager['assembleContext']
    > extends Promise<infer T>
      ? T
      : never,
  ): SendMessageParameters {
    // If no RAG context, return original
    if (assembledContext.ragChunksIncluded === 0) {
      return originalParams;
    }

    // Create enhanced message that includes RAG context
    const ragContextText = this.extractRAGContextText(assembledContext);
    const enhancedMessage = ragContextText
      ? `${ragContextText}\n\n---\n\nUser Question: ${originalParams.message}`
      : originalParams.message;

    return {
      ...originalParams,
      message: enhancedMessage,
    };
  }

  /**
   * Extracts RAG context text from assembled context
   */
  private extractRAGContextText(assembledContext: AssembledContext): string {
    if (!assembledContext.contents || assembledContext.contents.length === 0) {
      return '';
    }

    // Look for RAG context in the assembled contents
    for (const content of assembledContext.contents) {
      if (
        content.role === 'user' &&
        content.parts &&
        content.parts.length > 0
      ) {
        const text = content.parts[0]?.text || '';
        if (text.includes('## Relevant Knowledge Base Information')) {
          return text;
        }
      }
    }

    return assembledContext.ragContextSummary || '';
  }

  /**
   * Override getHistory to provide intelligent context instead of
   * the manual "CRITICAL FIX" reversal when curated=true
   */
  getHistory(curated: boolean = false): Content[] {
    if (!curated || !this.ragEnabled) {
      // Use original implementation for uncurated or when RAG disabled
      return super.getHistory(curated);
    }

    // For curated history with RAG enabled, we provide smarter context management
    // by removing the need for manual reversal - the PromptContextManager handles this
    const originalHistory = super.getHistory(false); // Get uncurated

    // Apply basic curation without the problematic reversal
    return this.applySafeCuration(originalHistory);
  }

  /**
   * Applies safe curation without the problematic manual reversal
   */
  private applySafeCuration(history: Content[]): Content[] {
    // Extract valid content without reversal - let PromptContextManager handle ordering
    const curated: Content[] = [];

    for (let i = 0; i < history.length; i++) {
      const content = history[i];
      if (this.isValidContent(content)) {
        curated.push(content);
      }
    }

    return curated;
  }

  /**
   * Checks if content is valid (copied from original implementation logic)
   */
  private isValidContent(content: Content): boolean {
    if (!content.parts || content.parts.length === 0) {
      return false;
    }

    for (const part of content.parts) {
      if (!part || Object.keys(part).length === 0) {
        return false;
      }
      if (!part.thought && part.text !== undefined && part.text === '') {
        return false;
      }
    }

    return true;
  }

  /**
   * Gets RAG enhancement status for this chat instance
   */
  getRAGStatus(): boolean {
    return this.ragEnabled;
  }
}
