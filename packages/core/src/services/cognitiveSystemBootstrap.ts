/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Cognitive System Bootstrap - Initialize enhanced thinking capabilities
 * This module integrates the sequential thinking system with the main CLI
 */

import { Config } from '../config/config.js';
import type { ContentGenerator } from '../core/contentGenerator.js';
import { CognitiveOrchestrator, CognitiveMode } from '../services/cognitiveOrchestrator.js';

export interface CognitiveSystemConfig {
  enabled: boolean;
  defaultMode: CognitiveMode['mode'];
  thinkingDepthLimit: number;
  missionBatchSize: number;
  autonomousMode: boolean;
  debugMode: boolean;
}

export interface CognitiveSystemStatus {
  initialized: boolean;
  activeSessions: number;
  activeMissions: number;
  lastActivity: Date;
  totalTokensUsed: number;
  cognitiveOperations: number;
}

/**
 * Bootstrap class that initializes and manages the cognitive system
 */
export class CognitiveSystemBootstrap {
  private static instance: CognitiveSystemBootstrap;
  private orchestrator?: CognitiveOrchestrator;
  private status: CognitiveSystemStatus = {
    initialized: false,
    activeSessions: 0,
    activeMissions: 0,
    lastActivity: new Date(),
    totalTokensUsed: 0,
    cognitiveOperations: 0,
  };

  private config: CognitiveSystemConfig = {
    enabled: true,
    defaultMode: 'reactive',
    thinkingDepthLimit: 5,
    missionBatchSize: 50,
    autonomousMode: false,
    debugMode: false,
  };

  /**
   * Singleton instance getter
   */
  static getInstance(): CognitiveSystemBootstrap {
    if (!CognitiveSystemBootstrap.instance) {
      CognitiveSystemBootstrap.instance = new CognitiveSystemBootstrap();
    }
    return CognitiveSystemBootstrap.instance;
  }

  /**
   * Initialize the cognitive system with the main Gemini CLI services
   */
  async initialize(
    coreConfig: Config,
    contextManager: Record<string, unknown>,
    toolGuidance: Record<string, unknown>,
    contentGenerator: ContentGenerator,
    userConfig?: Partial<CognitiveSystemConfig>,
  ): Promise<void> {
    try {
      // Merge user config with defaults
      this.config = { ...this.config, ...userConfig };

      if (!this.config.enabled) {
        console.debug('Cognitive system disabled by configuration');
        return;
      }

      // Initialize the cognitive orchestrator (mock for now)
      this.orchestrator = new CognitiveOrchestrator(
        coreConfig,
        contextManager as unknown as any,
        toolGuidance as unknown as any,
        contentGenerator,
      );

      // Update status
      this.status.initialized = true;
      this.status.lastActivity = new Date();

      if (this.config.debugMode) {
        console.log('🧠 Cognitive System initialized successfully');
        console.log(`   - Default Mode: ${this.config.defaultMode}`);
        console.log(`   - Thinking Depth: ${this.config.thinkingDepthLimit}`);
        console.log(`   - Autonomous Mode: ${this.config.autonomousMode}`);
      }
    } catch (error) {
      console.error('Failed to initialize cognitive system:', error);
      this.status.initialized = false;
      throw error;
    }
  }

  /**
   * Process a request through the cognitive system
   */
  async processRequest(
    userMessage: string,
    conversationHistory: Array<Record<string, unknown>>,
    promptId: string,
  ): Promise<Record<string, unknown>> {
    if (!this.isReady()) {
      throw new Error('Cognitive system not initialized');
    }

    try {
      this.status.lastActivity = new Date();
      this.status.cognitiveOperations++;

      const response = await this.orchestrator!.processRequest(
        userMessage,
        conversationHistory,
        promptId,
      );

      // Update session tracking
      if (response.thinkingSession) {
        this.status.activeSessions++;
      }

      return { ...response } as Record<string, unknown>;
    } catch (error) {
      console.error('Cognitive processing error:', error);
      throw error;
    }
  }

  /**
   * Start an autonomous mission
   */
  async startMission(
    missionDescription: string,
    config?: Record<string, unknown>,
    targetFiles?: string[],
  ): Promise<string> {
    if (!this.isReady()) {
      throw new Error('Cognitive system not initialized');
    }

    try {
      const missionId = await this.orchestrator!.startMission(
        missionDescription,
        {
          missionId: `mission_${Date.now()}`,
          type: 'research',
          batchSize: this.config.missionBatchSize,
          maxTokensPerBatch: 100000,
          persistentMemory: true,
          reportingInterval: 10,
          ...config,
        },
        targetFiles,
      );

      this.status.activeMissions++;
      this.status.lastActivity = new Date();

      if (this.config.debugMode) {
        console.log(`🚀 Mission started: ${missionId}`);
      }

      return missionId;
    } catch (error) {
      console.error('Mission start error:', error);
      throw error;
    }
  }

  /**
   * Get cognitive system status
   */
  getStatus(): CognitiveSystemStatus {
    return { ...this.status };
  }

  /**
   * Check if cognitive system is ready
   */
  isReady(): boolean {
    return this.status.initialized && this.orchestrator !== undefined;
  }

  /**
   * Get mission status
   */
  getMissionStatus(missionId?: string): Record<string, unknown> {
    if (!this.isReady()) {
      return {};
    }

    return this.orchestrator!.getMissionStatus(missionId);
  }

  /**
   * Execute a thinking session plan
   */
  async executePlan(sessionId: string): Promise<Record<string, unknown>> {
    if (!this.isReady()) {
      throw new Error('Cognitive system not initialized');
    }

    return await this.orchestrator!.executePlan(sessionId);
  }

  /**
   * Get insights from a thinking session
   */
  getThinkingInsights(sessionId: string): Record<string, unknown> | null {
    if (!this.isReady()) {
      return null;
    }

    return this.orchestrator!.getThinkingInsights(sessionId);
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<CognitiveSystemConfig>): void {
    this.config = { ...this.config, ...newConfig };

    if (this.config.debugMode) {
      console.log('🔄 Cognitive system configuration updated:', newConfig);
    }
  }

  /**
   * Check if a request should use enhanced cognitive processing
   */
  shouldEnhanceRequest(userMessage: string, history: Array<Record<string, unknown>>): boolean {
    if (!this.isReady() || !this.config.enabled) {
      return false;
    }

    const message = userMessage.toLowerCase();

    // Keywords that trigger cognitive enhancement
    const cognitiveKeywords = [
      'think',
      'analyze',
      'plan',
      'strategy',
      'complex',
      'step by step',
      'mission',
      'batch',
      'autonomous',
      'review',
      'refactor',
    ];

    // Check for cognitive keywords
    const hasCognitiveKeywords = cognitiveKeywords.some((keyword) =>
      message.includes(keyword),
    );

    // Check conversation complexity
    const isComplexConversation = history.length > 5;

    // Check message length (longer messages often need more thinking)
    const isLongMessage = userMessage.length > 200;

    return hasCognitiveKeywords || isComplexConversation || isLongMessage;
  }

  /**
   * Cleanup and shutdown
   */
  async shutdown(): Promise<void> {
    if (this.config.debugMode) {
      console.log('🛑 Shutting down cognitive system...');
    }

    // Reset status
    this.status = {
      initialized: false,
      activeSessions: 0,
      activeMissions: 0,
      lastActivity: new Date(),
      totalTokensUsed: 0,
      cognitiveOperations: 0,
    };

    this.orchestrator = undefined;
  }
}

/**
 * Convenience function to get the global cognitive system instance
 */
export function getCognitiveSystem(): CognitiveSystemBootstrap {
  return CognitiveSystemBootstrap.getInstance();
}

/**
 * Convenience function to check if cognitive enhancement should be applied
 */
export function shouldUseCognitiveEnhancement(
  userMessage: string,
  history: Array<Record<string, unknown>> = [],
): boolean {
  return getCognitiveSystem().shouldEnhanceRequest(userMessage, history);
}

/**
 * Initialize cognitive system with common defaults
 */
export async function initializeCognitiveSystem(
  coreConfig: Config,
  contextManager: Record<string, unknown>,
  toolGuidance: Record<string, unknown>,
  contentGenerator: ContentGenerator,
  options: {
    enabled?: boolean;
    debugMode?: boolean;
    autonomousMode?: boolean;
  } = {},
): Promise<CognitiveSystemBootstrap> {
  const cognitiveSystem = getCognitiveSystem();

  await cognitiveSystem.initialize(
    coreConfig,
    contextManager,
    toolGuidance,
    contentGenerator,
    {
      enabled: options.enabled ?? true,
      debugMode: options.debugMode ?? false,
      autonomousMode: options.autonomousMode ?? false,
      defaultMode: 'thinking',
      thinkingDepthLimit: 5,
      missionBatchSize: 50,
    },
  );

  return cognitiveSystem;
}

export default CognitiveSystemBootstrap;
