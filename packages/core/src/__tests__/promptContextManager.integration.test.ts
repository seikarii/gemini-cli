/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PromptContextManager } from '../services/promptContextManager.js';
import { RAGService } from '../rag/ragService.js';
import { ChatRecordingService } from '../services/chatRecordingService.js';
import { Config } from '../config/config.js';
import { Content } from '@google/genai';

// Mock the dependencies
vi.mock('../rag/ragService.js');
vi.mock('../services/chatRecordingService.js');
vi.mock('../config/config.js');

describe('PromptContextManager Integration with Tool Guidance', () => {
  let promptContextManager: PromptContextManager;
  let mockRAGService: vi.Mocked<RAGService>;
  let mockChatRecordingService: vi.Mocked<ChatRecordingService>;
  let mockConfig: vi.Mocked<Config>;

  beforeEach(() => {
    // Create mock instances
    mockRAGService = {
      enhanceQuery: vi.fn(),
    } as unknown as vi.Mocked<RAGService>;

    mockChatRecordingService = {
      getOptimizedHistoryForPrompt: vi.fn(),
    } as unknown as vi.Mocked<ChatRecordingService>;

    mockConfig = {} as unknown as vi.Mocked<Config>;

    // Setup mock responses
    mockRAGService.enhanceQuery.mockResolvedValue({
      content: 'Mock RAG content',
      sourceChunks: [
        {
          content: 'function example() { return "test"; }',
          metadata: { file: { path: 'src/example.ts' } }
        }
      ]
    });

    mockChatRecordingService.getOptimizedHistoryForPrompt.mockResolvedValue({
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Previous user message' }]
        }
      ],
      metaInfo: {
        compressionApplied: false,
        originalMessageCount: 1,
        finalMessageCount: 1
      }
    });

    promptContextManager = new PromptContextManager(
      mockRAGService,
      mockChatRecordingService,
      mockConfig
    );
  });

  it('should include dynamic tool guidance in assembled context', async () => {
    const userMessage = 'Update the handleClick function in Button.tsx';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'I need to modify a React component' }]
      }
    ];

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    expect(result.contents).toBeDefined();
    expect(result.contents.length).toBeGreaterThan(0);
    
    // Check that the first content item contains tool guidance
    const firstContent = result.contents[0];
    expect(firstContent.role).toBe('user');
    expect(firstContent.parts?.[0]?.text).toContain('CONTEXTUAL TOOL GUIDANCE');
    expect(firstContent.parts?.[0]?.text).toContain('Button.tsx');
    expect(firstContent.parts?.[0]?.text).toContain('upsert_code_block');
  });

  it('should include tool guidance even when RAG fails (fallback)', async () => {
    // Make RAG service throw an error
    mockRAGService.enhanceQuery.mockRejectedValue(new Error('RAG service failed'));

    const userMessage = 'Fix the API call in userService.js';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'There is a bug in the user service' }]
      }
    ];

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    expect(result.contents).toBeDefined();
    expect(result.ragChunksIncluded).toBe(0); // RAG failed
    
    // Tool guidance should still be present
    const firstContent = result.contents[0];
    expect(firstContent.role).toBe('user');
    expect(firstContent.parts?.[0]?.text).toContain('CONTEXTUAL TOOL GUIDANCE');
    expect(firstContent.parts?.[0]?.text).toContain('userService.js');
  });

  it('should recommend correct tools based on conversation context', async () => {
    const userMessage = 'Change the port configuration';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'I need to update package.json settings' }]
      },
      {
        role: 'model',
        parts: [{ text: 'Sure, I can help with configuration changes.' }]
      }
    ];

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    const toolGuidanceText = result.contents[0].parts?.[0]?.text || '';
    expect(toolGuidanceText).toContain('replace');
    expect(toolGuidanceText).toContain('config');
  });

  it('should detect and handle previous tool failures', async () => {
    const userMessage = 'Try to fix the validation function again';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'Use replace to update the validation function' }]
      },
      {
        role: 'model',
        parts: [{ text: 'Error: replace tool failed due to multiple matches found' }]
      }
    ];

    // Mock the chat recording service to return the conversation history that includes the error
    mockChatRecordingService.getOptimizedHistoryForPrompt.mockResolvedValue({
      contents: conversationHistory, // Return the actual conversation history with the error
      metaInfo: {
        compressionApplied: false,
        originalMessageCount: 2,
        finalMessageCount: 2
      }
    });

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    const toolGuidanceText = result.contents[0].parts?.[0]?.text || '';
    expect(toolGuidanceText).toContain('Previous replace operation failed');
    expect(toolGuidanceText).toContain('Recent Error Context');
    expect(toolGuidanceText).toContain('upsert_code_block');
  });

  it('should provide specific guidance for different file types', async () => {
    const userMessage = 'Update both config.json and utils.ts files';
    const conversationHistory: Content[] = [];

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    const toolGuidanceText = result.contents[0].parts?.[0]?.text || '';
    expect(toolGuidanceText).toContain('Code files');
    expect(toolGuidanceText).toContain('Config files');
    expect(toolGuidanceText).toContain('utils.ts');
    expect(toolGuidanceText).toContain('config.json');
  });

  it('should maintain proper content order: guidance -> RAG -> conversation', async () => {
    const userMessage = 'Refactor the authentication module';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'I want to improve the auth system' }]
      }
    ];

    const result = await promptContextManager.assembleContext(
      userMessage,
      conversationHistory
    );

    expect(result.contents.length).toBeGreaterThanOrEqual(3);
    
    // First: Tool guidance
    expect(result.contents[0].parts?.[0]?.text).toContain('CONTEXTUAL TOOL GUIDANCE');
    
    // Second: RAG content
    expect(result.contents[1].parts?.[0]?.text).toContain('Relevant Knowledge Base Information');
    
    // Third: Conversation history
    expect(result.contents[2].parts?.[0]?.text).toBe('Previous user message');
  });
});
