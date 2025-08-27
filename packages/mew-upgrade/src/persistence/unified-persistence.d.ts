/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Implements the UnifiedPersistence facade, providing a simple top-level API
 * for the agent to save and load its entire state.
 */
import { Persistable } from './persistence-service.js';
export interface Agent {
    getPersistableComponents(): Record<string, Persistable>;
}
/**
 * The UnifiedPersistence class provides a simple facade for the entire persistence system.
 * The main agent interacts with this class directly.
 */
export declare class UnifiedPersistence {
    private service;
    constructor(basePath: string);
    /**
     * Backs up the entire state of the agent.
     * It iterates through the agent's components and saves each one.
     * @param agent The main agent instance.
     */
    backup(agent: Agent): Promise<void>;
    /**
     * Restores the entire state of the agent from storage.
     * @param agent The main agent instance.
     */
    restore(agent: Agent): Promise<void>;
}
