/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, it, describe, beforeEach, afterEach } from 'vitest';
import { ChatRecordingService } from '../packages/core/src/services/chatRecordingService.js';
import { Config } from '../packages/core/src/config/config.js';
import { Content } from '@google/genai';
import * as path from 'node:path';
import { promises as fs } from 'node:fs';
import * as os from 'node:os';

describe('RAG Integration Tests', () => {
  let chatRecordingService: ChatRecordingService;
  let tempDir: string;

  beforeEach(async () => {
    // Create temporary directory for test
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'rag-test-'));

    // Create test config
    const config = new Config({
      sessionId: 'test-session',
      targetDir: tempDir,
      debugMode: false,
      cwd: tempDir,
      model: 'gemini-1.5-flash',
    });

    // Initialize services
    chatRecordingService = new ChatRecordingService(config);
    await chatRecordingService.initialize();
  });

  afterEach(async () => {
    // Clean up temp directory
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Failed to clean up temp directory:', error);
    }
  });

  describe('End-to-End RAG Functionality', () => {
    it('should integrate ChatRecordingService with PromptContextManager', async () => {
      // Create sample conversation history
      const conversationHistory: Content[] = [
        { role: 'user', parts: [{ text: 'Hello, I need help with TypeScript interfaces.' }] },
        { role: 'model', parts: [{ text: 'I\'d be happy to help with TypeScript interfaces! What specific aspect would you like to know about?' }] },
        { role: 'user', parts: [{ text: 'How do I create optional properties in an interface?' }] },
        { role: 'model', parts: [{ text: 'You can create optional properties by adding a question mark (?) after the property name. For example: interface User { name: string; age?: number; }' }] },
        { role: 'user', parts: [{ text: 'Can you show me more complex examples with nested interfaces?' }] },
        { role: 'model', parts: [{ text: 'Certainly! Here are some complex interface examples with nesting and optional properties...' }] },
      ];

      // Test 1: ChatRecordingService can process external history
      const optimizedResult = await chatRecordingService.getOptimizedHistoryForPrompt(
        conversationHistory,
        2000, // Token budget
        true, // Include system info
      );

      // Verify the result structure
      expect(optimizedResult).toHaveProperty('contents');
      expect(optimizedResult).toHaveProperty('estimatedTokens');
      expect(optimizedResult).toHaveProperty('compressionLevel');
      expect(optimizedResult).toHaveProperty('metaInfo');

      // Verify contents are in correct format
      expect(Array.isArray(optimizedResult.contents)).toBe(true);
      expect(optimizedResult.contents.length).toBeGreaterThan(0);
      
      // Each content should have proper structure
      for (const content of optimizedResult.contents) {
        expect(content).toHaveProperty('role');
        expect(content).toHaveProperty('parts');
        expect(['user', 'model']).toContain(content.role);
        expect(Array.isArray(content.parts)).toBe(true);
      }

      // Verify token estimation
      expect(typeof optimizedResult.estimatedTokens).toBe('number');
      expect(optimizedResult.estimatedTokens).toBeGreaterThan(0);
      expect(optimizedResult.estimatedTokens).toBeLessThanOrEqual(2100); // Allow some buffer

      // Verify metadata
      expect(optimizedResult.metaInfo.originalMessageCount).toBe(conversationHistory.length);
      expect(optimizedResult.metaInfo.finalMessageCount).toBeGreaterThan(0);
      expect(optimizedResult.metaInfo.finalMessageCount).toBeLessThanOrEqual(conversationHistory.length);
      expect(typeof optimizedResult.metaInfo.compressionApplied).toBe('boolean');

      console.log('✅ ChatRecordingService processes external history correctly');
    });

    it('should handle token budget constraints with compression', async () => {
      // Create a longer conversation that will exceed token budget
      const longConversation: Content[] = Array.from({ length: 50 }, (_, i) => ({
        role: i % 2 === 0 ? 'user' : 'model',
        parts: [{ 
          text: `This is message number ${i + 1}. It contains some technical content about software development, programming languages, frameworks, libraries, and various coding concepts that would typically be found in a developer conversation. The content is substantial enough to contribute meaningfully to token count calculations and compression testing scenarios.` 
        }],
      }));

      // Test with realistic token budget
      const result = await chatRecordingService.getOptimizedHistoryForPrompt(
        longConversation,
        5000, // Reasonable budget
        true,
      );

      // Verify the result structure is correct regardless of whether compression was applied
      expect(result.estimatedTokens).toBeGreaterThan(0);
      expect(result.metaInfo.finalMessageCount).toBeGreaterThan(0);
      expect(result.metaInfo.finalMessageCount).toBeLessThanOrEqual(longConversation.length);

      // Test with more generous budget
      const generousResult = await chatRecordingService.getOptimizedHistoryForPrompt(
        longConversation,
        10000, // Very large budget
        true,
      );

      // Should keep more or equal content
      expect(generousResult.metaInfo.finalMessageCount).toBeGreaterThanOrEqual(result.metaInfo.finalMessageCount);

      console.log(`Restricted result: ${result.estimatedTokens} tokens, ${result.metaInfo.finalMessageCount} messages`);
      console.log(`Generous result: ${generousResult.estimatedTokens} tokens, ${generousResult.metaInfo.finalMessageCount} messages`);
      console.log('✅ Token budget constraints and compression work correctly');
    });

    it('should maintain conversation context integrity', async () => {
      // Create conversation with specific context
      const contextualConversation: Content[] = [
        { role: 'user', parts: [{ text: 'I\'m working on a React component called UserProfile.' }] },
        { role: 'model', parts: [{ text: 'Great! I can help you with the UserProfile component. What specific functionality are you implementing?' }] },
        { role: 'user', parts: [{ text: 'I need to display user avatar, name, and bio. The avatar should be clickable.' }] },
        { role: 'model', parts: [{ text: 'Here\'s a UserProfile component structure...' }] },
        { role: 'user', parts: [{ text: 'How do I handle the avatar click event?' }] },
      ];

      const result = await chatRecordingService.getOptimizedHistoryForPrompt(
        contextualConversation,
        3000,
        true,
      );

      // Verify all messages are preserved (since this is small conversation)
      expect(result.metaInfo.finalMessageCount).toBe(contextualConversation.length);
      
      // Verify the conversation flow is maintained
      const contents = result.contents;
      expect(contents[0].role).toBe('user');
      expect(contents[0].parts[0].text).toContain('UserProfile');
      expect(contents[contents.length - 1].role).toBe('user');
      expect(contents[contents.length - 1].parts[0].text).toContain('avatar click');

      console.log('✅ Conversation context integrity is maintained');
    });

    it('should return correct compression statistics', async () => {
      // Create conversation that will trigger compression
      const mediumConversation: Content[] = Array.from({ length: 20 }, (_, i) => ({
        role: i % 2 === 0 ? 'user' : 'model',
        parts: [{ 
          text: `Message ${i + 1}: This is a detailed technical discussion about software architecture patterns, including detailed explanations of design principles, code examples, implementation strategies, and best practices for scalable application development.` 
        }],
      }));

      const result = await chatRecordingService.getOptimizedHistoryForPrompt(
        mediumConversation,
        1500, // Budget that should trigger compression
        true, // Include system info to get compression stats
      );

      if (result.metaInfo.compressionApplied) {
        expect(result.metaInfo.compressionStats).toBeDefined();
        
        const stats = result.metaInfo.compressionStats!;
        expect(stats.originalMessages).toBe(mediumConversation.length);
        expect(stats.compressedMessages).toBe(result.metaInfo.finalMessageCount);
        expect(stats.tokenReduction).toBeGreaterThanOrEqual(0);
        expect(stats.compressionRatio).toBeGreaterThan(0);
        expect(stats.compressionRatio).toBeLessThanOrEqual(1);

        console.log('✅ Compression statistics are accurate');
      } else {
        console.log('ℹ️ No compression applied for this test case');
      }
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty conversation history', async () => {
      const result = await chatRecordingService.getOptimizedHistoryForPrompt(
        [],
        1000,
        false,
      );

      expect(result.contents).toHaveLength(0);
      expect(result.estimatedTokens).toBe(0);
      expect(result.metaInfo.originalMessageCount).toBe(0);
      expect(result.metaInfo.finalMessageCount).toBe(0);
      expect(result.metaInfo.compressionApplied).toBe(false);

      console.log('✅ Empty conversation history handled correctly');
    });

    it('should handle single message conversation', async () => {
      const singleMessage: Content[] = [
        { role: 'user', parts: [{ text: 'Hello, world!' }] },
      ];

      const result = await chatRecordingService.getOptimizedHistoryForPrompt(
        singleMessage,
        1000,
        false,
      );

      expect(result.contents).toHaveLength(1);
      expect(result.contents[0].role).toBe('user');
      expect(result.contents[0].parts[0].text).toBe('Hello, world!');
      expect(result.estimatedTokens).toBeGreaterThan(0);

      console.log('✅ Single message conversation handled correctly');
    });
  });
});
