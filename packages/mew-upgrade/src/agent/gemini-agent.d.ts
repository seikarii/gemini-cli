/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Config } from '../../cli/src/config/config';
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
}
