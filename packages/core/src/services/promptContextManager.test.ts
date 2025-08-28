/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { PromptContextManager, AssembledContext } from './promptContextManager.js';
import { ChatRecordingService } from './chatRecordingService.js';
import { RAGService } from '../rag/ragService.js';
import { Config } from '../config/config.js';
import { Content } from '@google/genai';

class StubChatRecordingService {
  async getOptimizedHistoryForPrompt(_tokenBudget: number) {
    return {
      history: [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Response' }] },
      ] as Content[],
      metaInfo: {
        totalTokens: 100,
        originalMessageCount: 2,
        finalMessageCount: 2,
        compressionApplied: false,
      },
    };
  }
}

class StubRAGService {}

describe('PromptContextManager integration', () => {
  let pcm: PromptContextManager;
  beforeEach(() => {
    const rag = new StubRAGService() as unknown as RAGService;
    const chatSvc = new StubChatRecordingService() as unknown as ChatRecordingService;
    const cfg = { maxTotalTokens: 8000 } as unknown as Config;
    pcm = new PromptContextManager(rag, chatSvc, cfg, {
      maxTotalTokens: 8000,
      maxRAGChunks: 4,
      ragRelevanceThreshold: 0.5,
      ragWeight: 0.5,
      prioritizeRecentConversation: true,
      useConversationalContext: true,
    });
  });

    it('should assemble context using chatRecordingService optimized history', async () => {
    const assembled: AssembledContext = await pcm.assembleContext('Hi', [
      { role: 'user', parts: [{ text: 'Hello' }] },
    ] as Content[], 'prompt-1');
    expect(assembled).toHaveProperty('contents');
    expect(assembled.contents.length).toBeGreaterThan(0);
    expect(typeof assembled.estimatedTokens).toBe('number');
  });
});
