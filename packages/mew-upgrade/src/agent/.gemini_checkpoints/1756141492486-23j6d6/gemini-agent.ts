/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file Implements the main GeminiAgent, the central hub of the upgraded architecture.
 */

import { MenteOmega } from '../mind/mente-omega.js';
import type { MemoryNodeKind } from '../mind/mental-laby.js';
import { UnifiedPersistence } from '../persistence/unified-persistence.js';
import { createContentGenerator, AuthType, ContentGenerator } from '@google/gemini-cli-core';
import { runNonInteractive } from '../../cli/src/nonInteractiveCli.js';
// @ts-ignore: build/runtime uses ESM paths; keep TS import for types
import { Config } from '../../cli/src/config/config';

// Simple terminal connection used for local testing / examples
class TerminalConnection {
  private onRequest: ((request: string) => void) | null = null;

  onReceiveRequest(callback: (request: string) => void) {
    this.onRequest = callback;
  }

  // Simulate a user typing a command in the terminal
  simulateUserRequest(request: string) {
    console.log(`\n--- Terminal: User entered command: "${request}" ---`);
    if (this.onRequest) {
      this.onRequest(request);
    }
  }
}

export class GeminiAgent {
  private config: Config;
  private brain: MenteOmega | null;
  private persistence: UnifiedPersistence;
  private terminal: TerminalConnection;
  private contentGenerator: ContentGenerator | null; // The connection to Gemini (MenteAlfa)

  constructor(config: Config) {
    this.config = config;
    // The agent's state will be stored in a subdirectory.
    const stateBasePath = '/media/seikarii/Nvme/gemini-cli/Mew/agent_state';

    // contentGenerator will be initialized in start() because it's async
    this.contentGenerator = null;

    // brain will be initialized with contentGenerator in start()
    this.brain = null;

    this.persistence = new UnifiedPersistence(stateBasePath);
    this.terminal = new TerminalConnection();

    // Setup the connection between the terminal and the brain (guarded: brain may be null until start())
    this.terminal.onReceiveRequest((request) => {
      if (this.brain && typeof (this.brain as any).process === 'function') {
        (this.brain as any).process(request);
      } else {
        // If the brain is not ready yet, you could buffer requests here.
        console.warn('Received terminal request before brain was initialized:', request);
      }
    });
  }

  /**
   * Starts the agent's main loop and restores its state.
   */
  async start(): Promise<void> {
    console.log('--- Gemini Agent is starting up... ---');

    // Initialize the content generator (MenteAlfa connection) using the real config
    this.contentGenerator = await createContentGenerator(
      this.config.getContentGeneratorConfig(),
      this.config as any,
      this.config.getSessionId()
    );

    // Initialize MenteOmega with the contentGenerator
    this.brain = new MenteOmega(this.contentGenerator as ContentGenerator);

    // Restore previous state from disk
    await this.persistence.restore(this.getPersistableAPI());

    console.log('--- Agent is running and listening for requests. ---');

    // Periodically save the agent's state
    setInterval(() => {
      this.persistence.backup(this.getPersistableAPI());
    }, 60000); // Backup every 60 seconds

    // --- Example Usage Simulation ---
    // Simulate a user making a request after a short delay.
    setTimeout(() => {
      this.terminal.simulateUserRequest('What do you think about autogenesis?');
    }, 2000);
  }

  /**
   * Provides the persistence system with the components that need to be saved.
   */
  private getPersistableAPI(): any {
    return {
      getPersistableComponents: () => {
        const components: Record<string, unknown> = {};
        if (this.brain) {
          components['brain'] = this.brain;
        }
        return components;
      },
    };
  }
}

  /**
   * Allows external information to be directly ingested into the agent's memory.
   * This is the "whisper" capability.
   * @param data The data to ingest.
   * @param kind The kind of memory node (e.g., 'semantic', 'procedural', 'episodic').
   */
  whisper(data: any, kind?: MemoryNodeKind): void {
    if (this.brain) {
      this.brain.memory.ingest(data, kind);
      console.log('Agent whispered data into memory.');
    } else {
      console.warn('Cannot whisper: Agent brain not initialized.');
    }
  }

// --- Main Execution (example / local test) ---
// Mock Config for testing purposes
const mockConfig: Config = {
  getModel: () => 'gemini-1.5-pro',
  getContentGeneratorConfig: () => ({ authType: AuthType.LOGIN_WITH_GOOGLE }),
  getSessionId: () => 'mew-agent-session',
  getProxy: () => undefined,
  // Add other methods that might be called by ContentGenerator or other parts of the system
  getUsageStatisticsEnabled: () => false,
  getUserMemory: () => ({}),
  getToolRegistry: () => ({ getFunctionDeclarations: () => [] }),
  getChatCompression: () => ({}),
  getMaxSessionTurns: () => 0,
  getSkipNextSpeakerCheck: () => false,
  getDebugMode: () => false,
  flashFallbackHandler: undefined,
  setModel: (model: string) => {},
  setFallbackMode: (mode: boolean) => {},
  getWorkspaceContext: () => ({ getDirectories: () => [], isPathWithinWorkspace: (path: string) => true }),
  getFileService: () => ({ shouldGeminiIgnoreFile: (path: string) => false }),
};

// Only run the example when this module is executed directly (optional guard)
if (typeof require !== 'undefined' && require.main === module) {
  const agent = new GeminiAgent(mockConfig);
  void agent.start();

// Keep the process alive to see async output
setInterval(() => {}, 1000);
}

