/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Cognitive Orchestrator - Main entry point for the enhanced thinking system
 * Integrates SequentialThinkingService with existing Gemini CLI infrastructure
 */

import { Content } from '@google/genai';
import { SequentialThinkingService, ThinkingSession, MissionConfig, ThinkingStep } from './sequentialThinkingService.js';
import { PromptContextManager, AssembledContext } from './promptContextManager.js';
import { ToolSelectionGuidance } from './toolSelectionGuidance.js';
import { Config } from '../config/config.js';
import type { ContentGenerator } from '../core/contentGenerator.js';

export interface CognitiveMode {
  mode: 'reactive' | 'thinking' | 'mission';
  parameters?: {
    thinkingDepth?: number;
    requiresPlan?: boolean;
    autonomous?: boolean;
  };
}

export interface EnhancedResponse {
  originalContext: AssembledContext;
  thinkingSession?: ThinkingSession;
  enhancedPrompt: string;
  cognitiveInsights?: Record<string, unknown>;
  executionPlan?: Record<string, unknown>;
  confidence: number;
}

/**
 * Main orchestrator that decides when to use sequential thinking
 * and coordinates between different cognitive modes
 */
export class CognitiveOrchestrator {
  private sequentialThinking: SequentialThinkingService;
  private activeMissions: Map<string, { status: string; progress: number }> = new Map();

  constructor(
    private config: Config,
    private contextManager: PromptContextManager,
    private toolGuidance: ToolSelectionGuidance,
    contentGenerator: ContentGenerator,
  ) {
    this.sequentialThinking = new SequentialThinkingService(
      config,
      contextManager as unknown as Record<string, unknown>,
      toolGuidance as unknown as Record<string, unknown>,
      contentGenerator,
    );
  }

  /**
   * Main processing method that decides cognitive approach
   */
  async processRequest(
    userMessage: string,
    conversationHistory: Content[],
    promptId: string,
  ): Promise<EnhancedResponse> {
    // Analyze request to determine cognitive mode
    const cognitiveMode = this.determineCognitiveMode(userMessage, conversationHistory);

    // Always start with standard context assembly
    const originalContext = await this.contextManager.assembleContext(
      userMessage,
      conversationHistory,
      promptId,
    );

    let enhancedResponse: EnhancedResponse = {
      originalContext,
      enhancedPrompt: this.buildBasicPrompt(userMessage, originalContext),
      confidence: 0.7,
    };

    // Apply cognitive enhancement based on mode
    switch (cognitiveMode.mode) {
      case 'thinking':
        enhancedResponse = await this.applySequentialThinking(
          enhancedResponse,
          userMessage,
          conversationHistory,
          cognitiveMode.parameters?.thinkingDepth || 3,
        );
        break;

      case 'mission':
        enhancedResponse = await this.handleMissionRequest(
          enhancedResponse,
          userMessage,
          conversationHistory,
        );
        break;

      case 'reactive':
      default:
        // Standard processing with minimal enhancement
        enhancedResponse = this.applyBasicEnhancement(enhancedResponse, userMessage);
        break;
    }

    return enhancedResponse;
  }

  /**
   * Start a long-term autonomous mission
   */
  async startMission(
    missionDescription: string,
    config: MissionConfig,
    targetFiles?: string[],
  ): Promise<string> {
    const missionId = await this.sequentialThinking.startMission(
      missionDescription,
      config,
      targetFiles,
    );

    this.activeMissions.set(missionId, {
      status: 'active',
      progress: 0,
    });

    return missionId;
  }

  /**
   * Get status of active missions
   */
  getMissionStatus(missionId?: string): Record<string, unknown> {
    if (missionId) {
      const mission = this.activeMissions.get(missionId);
      return mission ? { [missionId]: mission } : {};
    }

    return Object.fromEntries(this.activeMissions.entries());
  }

  /**
   * Execute a thinking session plan
   */
  async executePlan(sessionId: string): Promise<Record<string, unknown>> {
    return await this.sequentialThinking.executePlan(sessionId);
  }

  /**
   * Get insights from a thinking session
   */
  getThinkingInsights(sessionId: string): Record<string, unknown> | null {
    return this.sequentialThinking.getThinkingInsights(sessionId);
  }

  /**
   * Private: Determine what cognitive mode to use
   */
  private determineCognitiveMode(
    userMessage: string,
    _conversationHistory: Content[],
  ): CognitiveMode {
    const message = userMessage.toLowerCase();

    // Check for mission keywords
    if (
      message.includes('mission') ||
      message.includes('analyze') && (message.includes('files') || message.includes('folder')) ||
      message.includes('batch') ||
      message.includes('autonomous')
    ) {
      return { mode: 'mission' };
    }

    // Check for complex thinking requirements
    if (
      message.includes('think') ||
      message.includes('plan') ||
      message.includes('strategy') ||
      message.includes('analyze') ||
      message.includes('complex') ||
      message.includes('step by step') ||
      this.detectComplexityIndicators(userMessage, _conversationHistory)
    ) {
      return {
        mode: 'thinking',
        parameters: {
          thinkingDepth: this.calculateThinkingDepth(userMessage),
          requiresPlan: this.requiresExecutionPlan(userMessage),
        },
      };
    }

    // Default to reactive mode
    return { mode: 'reactive' };
  }

  /**
   * Private: Apply sequential thinking enhancement
   */
  private async applySequentialThinking(
    response: EnhancedResponse,
    userMessage: string,
    conversationHistory: Content[],
    thinkingDepth: number,
  ): Promise<EnhancedResponse> {
    const thinkingSession = await this.sequentialThinking.think(
      userMessage,
      conversationHistory,
      thinkingDepth,
    );

    // Create enhanced prompt with thinking context
    const thinkingContext = this.buildThinkingContext(thinkingSession);
    const enhancedPrompt = this.buildEnhancedPrompt(
      userMessage,
      response.originalContext,
      thinkingContext,
    );

    return {
      ...response,
      thinkingSession,
      enhancedPrompt,
      cognitiveInsights: {
        thinkingSteps: thinkingSession.steps.length,
        confidence: this.calculateConfidence(thinkingSession),
        complexity: this.assessComplexity(thinkingSession),
      },
      confidence: this.calculateConfidence(thinkingSession),
    };
  }

  /**
   * Private: Handle mission-type requests
   */
  private async handleMissionRequest(
    response: EnhancedResponse,
    userMessage: string,
    _conversationHistory: Content[],
  ): Promise<EnhancedResponse> {
    const missionConfig: MissionConfig = {
      missionId: `mission_${Date.now()}`,
      type: this.detectMissionType(userMessage),
      batchSize: 50,
      maxTokensPerBatch: 100000,
      persistentMemory: true,
      reportingInterval: 10,
    };

    const enhancedPrompt = this.buildMissionPrompt(userMessage, response.originalContext, missionConfig);

    return {
      ...response,
      enhancedPrompt,
      cognitiveInsights: {
        missionType: missionConfig.type,
        batchSize: missionConfig.batchSize,
        autonomous: true,
      },
      confidence: 0.9,
    };
  }

  /**
   * Private: Apply basic enhancement for reactive mode
   */
  private applyBasicEnhancement(
    response: EnhancedResponse,
    userMessage: string,
  ): EnhancedResponse {
    // Add basic cognitive context
    const basicEnhancement = `

## Cognitive Context
- Processing Mode: Reactive
- Request Type: ${this.classifyRequestType(userMessage)}
- Recommended Approach: Direct execution with standard tools

`;

    return {
      ...response,
      enhancedPrompt: response.enhancedPrompt + basicEnhancement,
      confidence: 0.8,
    };
  }

  /**
   * Private: Build basic prompt from context
   */
  private buildBasicPrompt(userMessage: string, context: AssembledContext): string {
    return `User Request: ${userMessage}

Context Summary:
- RAG Chunks: ${context.ragChunksIncluded}
- Conversation Messages: ${context.conversationMessagesIncluded}
- Estimated Tokens: ${context.estimatedTokens}

Please process this request using the available tools and context.`;
  }

  /**
   * Private: Build enhanced prompt with thinking context
   */
  private buildEnhancedPrompt(
    userMessage: string,
    context: AssembledContext,
    thinkingContext: string,
  ): string {
    return `# Enhanced Cognitive Processing

## Original Request
${userMessage}

## Thinking Process
${thinkingContext}

## Available Context
- RAG Knowledge: ${context.ragChunksIncluded} chunks
- Conversation History: ${context.conversationMessagesIncluded} messages
- Token Budget: ${context.estimatedTokens}

## Instructions
Based on the thinking process above, execute the most appropriate plan of action.
Use the insights gained during the thinking phase to guide your tool selection and execution strategy.`;
  }

  /**
   * Private: Build thinking context summary
   */
  private buildThinkingContext(session: ThinkingSession): string {
    const steps = session.steps
      .map((step: ThinkingStep, index: number) => `${index + 1}. [${step.type.toUpperCase()}] ${step.description}`)
      .join('\n');

    return `
Thinking Session: ${session.sessionId}
Status: ${session.status}
Steps Completed: ${session.steps.length}

Process:
${steps}

${session.finalPlan ? '✅ Execution plan generated and ready' : '⏳ Still processing...'}
`;
  }

  /**
   * Private: Build mission-oriented prompt
   */
  private buildMissionPrompt(
    userMessage: string,
    context: AssembledContext,
    missionConfig: MissionConfig,
  ): string {
    return `# Mission-Mode Processing

## Mission Request
${userMessage}

## Mission Configuration
- Type: ${missionConfig.type}
- Batch Size: ${missionConfig.batchSize} items
- Reporting Interval: Every ${missionConfig.reportingInterval} batches
- Persistent Memory: ${missionConfig.persistentMemory ? 'Enabled' : 'Disabled'}

## Context Available
- Knowledge Base: ${context.ragChunksIncluded} chunks
- Conversation: ${context.conversationMessagesIncluded} messages

## Mission Execution Protocol
1. Break down the task into manageable batches
2. Process each batch systematically
3. Maintain global context and memory
4. Provide progress reports at intervals
5. Generate comprehensive final report

Begin mission execution with autonomous planning and batch processing.`;
  }

  /**
   * Private helper methods
   */
  private detectComplexityIndicators(userMessage: string, _history: Content[]): boolean {
    // Mock implementation for now
    return userMessage.length > 100;
  }

  private calculateThinkingDepth(userMessage: string): number {
    const indicators = ['complex', 'analyze', 'plan', 'strategy', 'think'];
    const count = indicators.filter((indicator) => userMessage.toLowerCase().includes(indicator)).length;
    return Math.min(Math.max(count, 1), 5);
  }

  private requiresExecutionPlan(userMessage: string): boolean {
    return userMessage.toLowerCase().includes('plan') || userMessage.toLowerCase().includes('steps');
  }

  private detectMissionType(userMessage: string): MissionConfig['type'] {
    const message = userMessage.toLowerCase();
    if (message.includes('analyze') || message.includes('review')) return 'code_analysis';
    if (message.includes('batch') || message.includes('files')) return 'batch_processing';
    if (message.includes('refactor')) return 'refactoring';
    return 'research';
  }

  private classifyRequestType(userMessage: string): string {
    const message = userMessage.toLowerCase();
    if (message.includes('read') || message.includes('show')) return 'information_retrieval';
    if (message.includes('write') || message.includes('create')) return 'content_creation';
    if (message.includes('fix') || message.includes('update')) return 'modification';
    return 'general_query';
  }

  private calculateConfidence(session: ThinkingSession): number {
    if (session.steps.length === 0) return 0.5;
    const avgConfidence = session.steps.reduce((sum: number, step: ThinkingStep) => sum + step.confidence, 0) / session.steps.length;
    return Math.min(Math.max(avgConfidence, 0.1), 0.99);
  }

  private assessComplexity(session: ThinkingSession): number {
    return Math.min(session.steps.length / 10, 1.0);
  }
}
