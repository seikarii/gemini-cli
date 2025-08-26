/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file Implements the main GeminiAgent, the central hub of the upgraded architecture.
 */

import path from 'path';
// fileURLToPath import intentionally omitted during build; kept out to avoid unused symbol
import { MenteOmega } from '../mind/mente-omega.js';
import type { MemoryNodeKind } from '../mind/mental-laby.js';
import { UnifiedPersistence, Agent } from '../persistence/unified-persistence.js';
import { createContentGenerator, ContentGenerator } from '@google/gemini-cli-core';
import { startWebServer } from '../server/webServer.js';
// We import runNonInteractive dynamically in the example block to avoid
// static type resolution against the monorepo's CLI during package-local builds.
// @ts-ignore: build/runtime uses ESM paths; keep TS import for types
import type { Config } from '../../cli/src/config/config';

// fileURLToPath(import.meta.url) intentionally not used here during builds

// Simple terminal connection used for local testing / examples
class TerminalConnection {
  private onRequest: ((request: string) => void) | null = null;

  onReceiveRequest(callback: (request: string) => void) {
    this.onRequest = callback;
  }

  // Simulate a user typing a command in the terminal
  simulateUserRequest(request: string) {
    console.info(`
--- Terminal: User entered command: "${request}" ---`);
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
    // The agent's state will be stored in a subdirectory of the project root.
    const stateBasePath = path.join(this.config.getTargetDir(), '.gemini', 'agent_state');

    // contentGenerator will be initialized in start() because it's async
    this.contentGenerator = null;

    // brain will be initialized with contentGenerator in start()
    this.brain = null;

    this.persistence = new UnifiedPersistence(stateBasePath);
    this.terminal = new TerminalConnection();

    // Setup the connection between the terminal and the brain (guarded: brain may be null until start())
    this.terminal.onReceiveRequest((request) => {
      if (this.brain) {
        this.brain.process(request);
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
    console.info('--- Gemini Agent is starting up...');

    // Initialize the content generator (MenteAlfa connection) using the real config
    this.contentGenerator = await createContentGenerator(
      this.config.getContentGeneratorConfig(),
      this.config,
      this.config.getSessionId()
    );

    // Initialize MenteOmega with the contentGenerator
    this.brain = new MenteOmega(this.contentGenerator as ContentGenerator);

    // Start the web server and pass the agent instance
    startWebServer(this);

    // Restore previous state from disk
    await this.persistence.restore(this.getPersistableAPI());

    console.log('--- Agent is running and listening for requests. ---');

    // Periodically save the agent's state
    setInterval(() => {
      this.persistence.backup(this.getPersistableAPI());
    }, 60000); // Backup every 60 seconds

  // --- Example Usage Simulation (removed for build cleanliness) ---
  }

  /**
   * Provides the persistence system with the components that need to be saved.
   */
  private getPersistableAPI(): Agent {
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

  /**
   * Allows external information to be directly ingested into the agent's memory.
   * This is the "whisper" capability.
   * @param data The data to ingest.
   * @param kind The kind of memory node (e.g., 'semantic', 'procedural', 'episodic').
   */
  public whisper(data: any, kind?: MemoryNodeKind): void {
    if (this.brain) {
      // Use any because MenteOmega's public API may not expose memory typing here.
      this.brain.memory.ingest(data, kind);
      console.log('Agent whispered data into memory.');
    } else {
      console.warn('Cannot whisper: Agent brain not initialized.');
    }
  }
  /**
   * Retrieves file content for the mini-editor.
   * This is a placeholder for MVP.
   * @param filePath The path to the file.
   * @returns A dummy content string for now.
   */
  public async getFileContent(filePath: string): Promise<string> {
    console.log(`Agent: Request to get content for ${filePath}`);
    return `// Content of ${filePath}\n// This is dummy content for MVP.`;
  }
}
  

// --- Main Execution (example / local test) ---
// Example execution block removed to keep module ESM-compatible and focused on exports. 

