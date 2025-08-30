/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Sequential Thinking System - Main Export
 * Integrates Mew-Upgrade cognitive architecture with Gemini CLI
 */

// Core cognitive services
export { SequentialThinkingService } from './services/sequentialThinkingService.js';
export { CognitiveOrchestrator } from './services/cognitiveOrchestrator.js';
export { CognitiveSystemBootstrap } from './services/cognitiveSystemBootstrap.js';
export { CognitiveCliIntegration } from './services/cognitiveCliIntegration.js';

// Integration utilities
export {
  initializeCognitiveSystem,
  getCognitiveSystem,
  shouldUseCognitiveEnhancement,
} from './services/cognitiveSystemBootstrap.js';

export {
  enhanceUserPrompt,
  handleCognitiveCommand,
  initializeCognitiveCliIntegration,
  cognitiveCliIntegration,
} from './services/cognitiveCliIntegration.js';

// Types and interfaces
export type {
  ThinkingStep,
  ThinkingSession,
  MissionConfig,
} from './services/sequentialThinkingService.js';

export type {
  CognitiveMode,
  EnhancedResponse,
} from './services/cognitiveOrchestrator.js';

export type {
  CognitiveSystemConfig,
  CognitiveSystemStatus,
} from './services/cognitiveSystemBootstrap.js';

export type {
  CognitiveCliHooks,
} from './services/cognitiveCliIntegration.js';

/**
 * Status Check Function
 */
export function getCognitiveSystemStatus(): {
  available: boolean;
  initialized: boolean;
  status?: Record<string, unknown>;
} {
  try {
    // Dynamic import to avoid circular dependencies
    import('./services/cognitiveSystemBootstrap.js').then(({ getCognitiveSystem }) => {
      const system = getCognitiveSystem();
      return {
        available: true,
        initialized: system.isReady(),
        status: system.isReady() ? system.getStatus() : undefined,
      };
    });
    
    return {
      available: true,
      initialized: false,
    };
  } catch {
    return {
      available: false,
      initialized: false,
    };
  }
}

/**
 * Default export for convenient importing
 */
export default {
  getCognitiveSystemStatus,
};
