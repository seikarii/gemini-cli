/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ContextManager, ContextType, DualContextConfig } from '../services/contextManager.js';

describe('ContextManager', () => {
  let contextManager: ContextManager;
  let mockConfig: DualContextConfig;

  beforeEach(() => {
    mockConfig = {
      promptContextTokens: 1000000,
      toolContextTokens: 28000,
      enableDualContext: true,
      promptModel: 'gemini-2.5-pro',
      toolModel: 'gemini-2.5-flash-lite',
    };

    // Mock token estimator
    const mockTokenEstimator = {
      estimateTokens: vi.fn().mockResolvedValue(1000),
    };

    contextManager = new ContextManager(mockConfig, mockTokenEstimator as any);
  });

  describe('determineContextType', () => {
    it('should return TOOL_EXECUTION for tool-related content', () => {
      const toolContent = 'Execute the edit tool with these parameters';
      const contextType = contextManager.determineContextType('edit', toolContent);
      expect(contextType).toBe(ContextType.TOOL_EXECUTION);
    });

    it('should return PROMPT for general conversation', () => {
      const promptContent = 'Please analyze this code and suggest improvements';
      const contextType = contextManager.determineContextType('analysis', promptContent);
      expect(contextType).toBe(ContextType.PROMPT);
    });

    it('should return PROMPT when dual-context is disabled', () => {
      const disabledConfig = { ...mockConfig, enableDualContext: false };
      const disabledManager = new ContextManager(disabledConfig, { estimateTokens: vi.fn() } as any);

      const toolContent = 'Execute the edit tool';
      const contextType = disabledManager.determineContextType('edit', toolContent);
      expect(contextType).toBe(ContextType.PROMPT);
    });
  });

  describe('getModelForContext', () => {
    it('should return prompt model for PROMPT context', () => {
      const model = contextManager.getModelForContext(ContextType.PROMPT);
      expect(model).toBe('gemini-2.5-pro');
    });

    it('should return tool model for TOOL_EXECUTION context', () => {
      const model = contextManager.getModelForContext(ContextType.TOOL_EXECUTION);
      expect(model).toBe('gemini-2.5-flash-lite');
    });
  });

  describe('canFitInContext', () => {
    it('should return true for content within limits', async () => {
      const mockEstimator = {
        estimateTokens: vi.fn().mockResolvedValue(10000), // Within 28K limit
      };
      const testManager = new ContextManager(mockConfig, mockEstimator as any);

      const canFit = await testManager.canFitInContext('test content', ContextType.TOOL_EXECUTION);
      expect(canFit).toBe(true);
    });

    it('should return false for content exceeding limits', async () => {
      const mockEstimator = {
        estimateTokens: vi.fn().mockResolvedValue(30000), // Exceeds 28K limit
      };
      const testManager = new ContextManager(mockConfig, mockEstimator as any);

      const canFit = await testManager.canFitInContext('large content', ContextType.TOOL_EXECUTION);
      expect(canFit).toBe(false);
    });
  });

  describe('getMaxTokensForContext', () => {
    it('should return correct limits for each context type', () => {
      expect(contextManager.getMaxTokensForContext(ContextType.PROMPT)).toBe(1000000);
      expect(contextManager.getMaxTokensForContext(ContextType.TOOL_EXECUTION)).toBe(28000);
    });
  });
});
