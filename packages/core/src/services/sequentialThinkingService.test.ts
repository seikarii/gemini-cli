/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Sequential Thinking System Test
 * Validates the cognitive system integration
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Content } from '@google/genai';
import { SequentialThinkingService, ThinkingStep } from '../services/sequentialThinkingService.js';
import { CognitiveOrchestrator } from '../services/cognitiveOrchestrator.js';
import { CognitiveSystemBootstrap } from '../services/cognitiveSystemBootstrap.js';

// Mock dependencies
const mockConfig = {
  getModel: vi.fn(() => 'gemini-1.5-flash'),
  sessionId: 'test-session',
} as Record<string, unknown>;

const mockContextManager = {
  assembleContext: vi.fn(async () => ({
    ragChunksIncluded: 0,
    conversationMessagesIncluded: 1,
    estimatedTokens: 100,
  })),
} as Record<string, unknown>;

const mockToolGuidance = {
  analyzeRequestContext: vi.fn(() => ({
    referencedFiles: [],
    hasRecentFailures: false,
  })),
} as Record<string, unknown>;

const mockContentGenerator = {
  generateContent: vi.fn(async () => ({
    text: 'Test response',
    candidates: [{
      content: {
        parts: [{ text: 'Test response' }]
      }
    }]
  })),
} as Record<string, unknown>;

describe('Sequential Thinking System', () => {
  let sequentialThinking: SequentialThinkingService;
  let cognitiveOrchestrator: CognitiveOrchestrator;
  let cognitiveSystem: CognitiveSystemBootstrap;

  beforeEach(() => {
    vi.clearAllMocks();
    
    sequentialThinking = new SequentialThinkingService(
      mockConfig,
      mockContextManager,
      mockToolGuidance,
      mockContentGenerator,
    );

    cognitiveOrchestrator = new CognitiveOrchestrator(
      mockConfig,
      mockContextManager,
      mockToolGuidance,
      mockContentGenerator,
    );

    cognitiveSystem = CognitiveSystemBootstrap.getInstance();
  });

  describe('SequentialThinkingService', () => {
    it('should create thinking sessions', async () => {
      const userMessage = 'Analyze this complex problem';
      const history: Content[] = [];

      const session = await sequentialThinking.think(userMessage, history, 3);

      expect(session).toBeDefined();
      expect(session.sessionId).toMatch(/^thinking_\d+$/);
      expect(session.userMessage).toBe(userMessage);
      expect(session.maxSteps).toBe(3);
      expect(session.steps).toHaveLength(0);
      expect(session.status).toBe('processing');
    });

    it('should add thinking steps', async () => {
      const userMessage = 'Test message';
      const history: Content[] = [];

      const session = await sequentialThinking.think(userMessage, history, 2);
      
      const step: ThinkingStep = {
        stepId: 'step_1',
        type: 'analysis',
        description: 'Analyzing the problem',
        content: 'This is a test analysis',
        confidence: 0.8,
        timestamp: new Date(),
      };

      await sequentialThinking.addThinkingStep(session.sessionId, step);

      const updatedSession = sequentialThinking.getThinkingSession(session.sessionId);
      expect(updatedSession?.steps).toHaveLength(1);
      expect(updatedSession?.steps[0]).toEqual(step);
    });

    it('should handle mission creation', async () => {
      const missionId = await sequentialThinking.startMission(
        'Test mission',
        {
          missionId: 'test_mission',
          type: 'research',
          batchSize: 10,
          maxTokensPerBatch: 1000,
          persistentMemory: true,
          reportingInterval: 5,
        }
      );

      expect(missionId).toBe('test_mission');
    });
  });

  describe('CognitiveOrchestrator', () => {
    it('should determine cognitive mode for thinking requests', async () => {
      const response = await cognitiveOrchestrator.processRequest(
        'I need to think step by step about this problem',
        [],
        'test-prompt'
      );

      expect(response).toBeDefined();
      expect(response.originalContext).toBeDefined();
      expect(response.enhancedPrompt).toContain('step by step');
      expect(response.confidence).toBeGreaterThan(0);
    });

    it('should determine cognitive mode for mission requests', async () => {
      const response = await cognitiveOrchestrator.processRequest(
        'Start a mission to analyze all files in the project',
        [],
        'test-prompt'
      );

      expect(response).toBeDefined();
      expect(response.enhancedPrompt).toContain('Mission-Mode Processing');
      expect(response.cognitiveInsights?.missionType).toBeDefined();
    });

    it('should handle reactive mode for simple requests', async () => {
      const response = await cognitiveOrchestrator.processRequest(
        'What is the weather today?',
        [],
        'test-prompt'
      );

      expect(response).toBeDefined();
      expect(response.enhancedPrompt).toContain('Reactive');
      expect(response.confidence).toBeGreaterThan(0.5);
    });
  });

  describe('CognitiveSystemBootstrap', () => {
    it('should initialize properly', async () => {
      await cognitiveSystem.initialize(
        mockConfig,
        mockContextManager,
        mockToolGuidance,
        mockContentGenerator,
        { enabled: true, debugMode: false }
      );

      expect(cognitiveSystem.isReady()).toBe(true);
      
      const status = cognitiveSystem.getStatus();
      expect(status.initialized).toBe(true);
      expect(status.activeSessions).toBe(0);
      expect(status.activeMissions).toBe(0);
    });

    it('should process requests when ready', async () => {
      await cognitiveSystem.initialize(
        mockConfig,
        mockContextManager,
        mockToolGuidance,
        mockContentGenerator,
        { enabled: true }
      );

      const response = await cognitiveSystem.processRequest(
        'Test request',
        [],
        'test-prompt'
      );

      expect(response).toBeDefined();
      expect(response.originalContext).toBeDefined();
      expect(response.enhancedPrompt).toBeDefined();
    });

    it('should handle enhancement detection', () => {
      const shouldEnhance1 = cognitiveSystem.shouldEnhanceRequest('think about this', []);
      expect(shouldEnhance1).toBe(false); // Not ready yet

      // After initialization
      cognitiveSystem.initialize(
        mockConfig,
        mockContextManager,
        mockToolGuidance,
        mockContentGenerator
      );

      const shouldEnhance2 = cognitiveSystem.shouldEnhanceRequest('think about this', []);
      expect(shouldEnhance2).toBe(true); // Should detect thinking keyword
    });
  });

  describe('Error Handling', () => {
    it('should handle initialization errors gracefully', async () => {
      const badConfig = null as any;

      await expect(
        cognitiveSystem.initialize(
          badConfig,
          mockContextManager,
          mockToolGuidance,
          mockContentGenerator
        )
      ).rejects.toThrow();

      expect(cognitiveSystem.isReady()).toBe(false);
    });

    it('should handle processing errors gracefully', async () => {
      // Try to process without initialization
      await expect(
        cognitiveSystem.processRequest('test', [], 'prompt')
      ).rejects.toThrow('Cognitive system not initialized');
    });
  });

  describe('Integration Features', () => {
    it('should track session metrics', async () => {
      await cognitiveSystem.initialize(
        mockConfig,
        mockContextManager,
        mockToolGuidance,
        mockContentGenerator
      );

      const initialStatus = cognitiveSystem.getStatus();
      const initialOps = initialStatus.cognitiveOperations;

      await cognitiveSystem.processRequest('test', [], 'prompt');

      const finalStatus = cognitiveSystem.getStatus();
      expect(finalStatus.cognitiveOperations).toBe(initialOps + 1);
    });

    it('should update configuration', () => {
      cognitiveSystem.updateConfig({ debugMode: true, autonomousMode: true });
      
      // Configuration should be applied (this is internal, so we check behavior)
      expect(() => cognitiveSystem.updateConfig({ enabled: false })).not.toThrow();
    });
  });
});
