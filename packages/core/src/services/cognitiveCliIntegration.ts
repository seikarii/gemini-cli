/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Cognitive CLI Integration - Hooks into the main CLI flow
 * This module provides the integration points for the cognitive system
 */

import { Content } from '@google/genai';
import { CognitiveSystemBootstrap } from '../services/cognitiveSystemBootstrap.js';

export interface CognitiveCliHooks {
  beforePromptProcessing?: (
    userMessage: string,
    history: Content[],
    promptId: string,
  ) => Promise<{ enhanced: boolean; modifiedMessage?: string }>;
  
  afterPromptProcessing?: (
    response: Record<string, unknown>,
    userMessage: string,
    history: Content[],
  ) => Promise<Record<string, unknown>>;
  
  onMissionStart?: (missionId: string, description: string) => void;
  onMissionComplete?: (missionId: string, results: Record<string, unknown>) => void;
}

/**
 * CLI Integration Manager
 */
export class CognitiveCliIntegration {
  private static instance: CognitiveCliIntegration;
  private hooks: CognitiveCliHooks = {};
  private cognitiveSystem?: CognitiveSystemBootstrap;

  static getInstance(): CognitiveCliIntegration {
    if (!CognitiveCliIntegration.instance) {
      CognitiveCliIntegration.instance = new CognitiveCliIntegration();
    }
    return CognitiveCliIntegration.instance;
  }

  /**
   * Initialize with the cognitive system
   */
  initialize(cognitiveSystem: CognitiveSystemBootstrap): void {
    this.cognitiveSystem = cognitiveSystem;
    this.setupDefaultHooks();
  }

  /**
   * Register custom hooks
   */
  registerHooks(hooks: CognitiveCliHooks): void {
    this.hooks = { ...this.hooks, ...hooks };
  }

  /**
   * Process user input with cognitive enhancement
   */
  async enhanceUserPrompt(
    userMessage: string,
    history: Content[],
    promptId: string,
  ): Promise<{
    enhanced: boolean;
    finalMessage: string;
    cognitiveResponse?: Record<string, unknown>;
    originalMessage: string;
  }> {
    const originalMessage = userMessage;

    try {
      // Check if cognitive enhancement should be applied
      if (!this.cognitiveSystem?.isReady()) {
        return {
          enhanced: false,
          finalMessage: userMessage,
          originalMessage,
        };
      }

      const shouldEnhance = this.cognitiveSystem.shouldEnhanceRequest(userMessage, history as Array<Record<string, unknown>>);

      if (!shouldEnhance) {
        return {
          enhanced: false,
          finalMessage: userMessage,
          originalMessage,
        };
      }

      // Apply before-processing hook
      let processedMessage = userMessage;
      if (this.hooks.beforePromptProcessing) {
        const hookResult = await this.hooks.beforePromptProcessing(
          userMessage,
          history,
          promptId,
        );
        if (hookResult.enhanced && hookResult.modifiedMessage) {
          processedMessage = hookResult.modifiedMessage;
        }
      }

      // Process through cognitive system
      const cognitiveResponse = await this.cognitiveSystem.processRequest(
        processedMessage,
        history as Array<Record<string, unknown>>,
        promptId,
      );

      // Use the enhanced prompt from cognitive processing
      const enhancedMessage = (cognitiveResponse.enhancedPrompt as string) || processedMessage;

      return {
        enhanced: true,
        finalMessage: enhancedMessage,
        cognitiveResponse,
        originalMessage,
      };
    } catch (error) {
      console.error('Cognitive enhancement error:', error);
      // Fallback to original message on error
      return {
        enhanced: false,
        finalMessage: userMessage,
        originalMessage,
      };
    }
  }

  /**
   * Process response with cognitive insights
   */
  async enhanceResponse(
    response: Record<string, unknown>,
    userMessage: string,
    history: Content[],
  ): Promise<Record<string, unknown>> {
    try {
      if (this.hooks.afterPromptProcessing) {
        return await this.hooks.afterPromptProcessing(response, userMessage, history);
      }
      return response;
    } catch (error) {
      console.error('Response enhancement error:', error);
      return response;
    }
  }

  /**
   * Handle mission commands
   */
  async handleMissionCommand(
    command: string,
    args: string[],
  ): Promise<{ handled: boolean; result?: Record<string, unknown> }> {
    if (!this.cognitiveSystem?.isReady()) {
      return { handled: false };
    }

    try {
      switch (command) {
        case 'start-mission': {
          const description = args.join(' ');
          const missionId = await this.cognitiveSystem.startMission(description);
          
          if (this.hooks.onMissionStart) {
            this.hooks.onMissionStart(missionId, description);
          }
          
          return {
            handled: true,
            result: { missionId, description, status: 'started' },
          };
        }

        case 'mission-status': {
          const statusMissionId = args[0];
          const status = this.cognitiveSystem.getMissionStatus(statusMissionId);
          return {
            handled: true,
            result: status,
          };
        }

        case 'list-missions': {
          const allMissions = this.cognitiveSystem.getMissionStatus();
          return {
            handled: true,
            result: allMissions,
          };
        }

        case 'cognitive-status': {
          const cognitiveStatus = this.cognitiveSystem.getStatus();
          return {
            handled: true,
            result: { ...cognitiveStatus } as Record<string, unknown>,
          };
        }

        default:
          return { handled: false };
      }
    } catch (error) {
      console.error(`Mission command error (${command}):`, error);
      return { handled: false };
    }
  }

  /**
   * Check if a message is a cognitive system command
   */
  isCognitiveCommand(message: string): boolean {
    const commands = [
      '/start-mission',
      '/mission-status',
      '/list-missions',
      '/cognitive-status',
      '/think',
      '/analyze',
    ];

    return commands.some((cmd) => message.trim().startsWith(cmd));
  }

  /**
   * Parse cognitive command
   */
  parseCognitiveCommand(message: string): { command: string; args: string[] } | null {
    const trimmed = message.trim();
    
    if (trimmed.startsWith('/start-mission ')) {
      return {
        command: 'start-mission',
        args: trimmed.slice('/start-mission '.length).split(' '),
      };
    }

    if (trimmed.startsWith('/mission-status ')) {
      return {
        command: 'mission-status',
        args: trimmed.slice('/mission-status '.length).split(' '),
      };
    }

    if (trimmed === '/list-missions') {
      return {
        command: 'list-missions',
        args: [],
      };
    }

    if (trimmed === '/cognitive-status') {
      return {
        command: 'cognitive-status',
        args: [],
      };
    }

    return null;
  }

  /**
   * Setup default hooks
   */
  private setupDefaultHooks(): void {
    this.hooks.onMissionStart = (missionId: string, description: string) => {
      console.log(`🚀 Mission started: ${missionId}`);
      console.log(`   Description: ${description}`);
    };

    this.hooks.onMissionComplete = (missionId: string, results: Record<string, unknown>) => {
      console.log(`✅ Mission completed: ${missionId}`);
      console.log(`   Results: ${JSON.stringify(results, null, 2)}`);
    };
  }

  /**
   * Get cognitive system status for display
   */
  getStatusDisplay(): string {
    if (!this.cognitiveSystem?.isReady()) {
      return '🔴 Cognitive System: Offline';
    }

    const status = this.cognitiveSystem.getStatus();
    return `🟢 Cognitive System: Active
   └─ Sessions: ${status.activeSessions}
   └─ Missions: ${status.activeMissions}
   └─ Operations: ${status.cognitiveOperations}
   └─ Last Activity: ${status.lastActivity.toLocaleTimeString()}`;
  }
}

/**
 * Global integration instance
 */
export const cognitiveCliIntegration = CognitiveCliIntegration.getInstance();

/**
 * Convenience function to enhance a user prompt
 */
export async function enhanceUserPrompt(
  userMessage: string,
  history: Content[],
  promptId: string,
): Promise<{
  enhanced: boolean;
  finalMessage: string;
  cognitiveResponse?: Record<string, unknown>;
  originalMessage: string;
}> {
  return await cognitiveCliIntegration.enhanceUserPrompt(userMessage, history, promptId);
}

/**
 * Convenience function to handle cognitive commands
 */
export async function handleCognitiveCommand(
  message: string,
): Promise<{ handled: boolean; result?: Record<string, unknown> }> {
  const command = cognitiveCliIntegration.parseCognitiveCommand(message);
  
  if (!command) {
    return { handled: false };
  }

  return await cognitiveCliIntegration.handleMissionCommand(command.command, command.args);
}

/**
 * Initialize cognitive CLI integration
 */
export async function initializeCognitiveCliIntegration(
  cognitiveSystem: CognitiveSystemBootstrap,
  customHooks?: CognitiveCliHooks,
): Promise<CognitiveCliIntegration> {
  const integration = CognitiveCliIntegration.getInstance();
  integration.initialize(cognitiveSystem);

  if (customHooks) {
    integration.registerHooks(customHooks);
  }

  return integration;
}

export default CognitiveCliIntegration;
