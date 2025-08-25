/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { MemoryNodeKind } from '../mind/mental-laby.js';
import type { Config } from '../../cli/src/config/config';
export declare class GeminiAgent {
    private config;
    private brain;
    private persistence;
    private terminal;
    private contentGenerator;
    constructor(config: Config);
    /**
     * Starts the agent's main loop and restores its state.
     */
    start(): Promise<void>;
    /**
     * Provides the persistence system with the components that need to be saved.
     */
    private getPersistableAPI;
    /**
     * Allows external information to be directly ingested into the agent's memory.
     * This is the "whisper" capability.
     * @param data The data to ingest.
     * @param kind The kind of memory node (e.g., 'semantic', 'procedural', 'episodic').
     */
    whisper(data: any, kind?: MemoryNodeKind): void;
    const mockConfig: any;
}
