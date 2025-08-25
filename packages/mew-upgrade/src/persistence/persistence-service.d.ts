/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Implements the PersistenceService, which orchestrates the serialization
 * and storage of the agent's state.
 */
import { StateSerializer } from './state-serializer.js';
import { StorageBackend } from './storage-manager.js';
export interface Persistable {
    exportState(): object;
    importState(state: object): void;
}
export declare class PersistenceService {
    private storage;
    private serializer;
    private basePath;
    constructor(basePath: string, storage?: StorageBackend, serializer?: StateSerializer);
    /**
     * Saves the state of a persistable component.
     * @param component The component to save (e.g., the Brain, Memory).
     * @param key A unique key for the state, used as the filename.
     */
    save(component: Persistable, key: string): Promise<void>;
    /**
     * Loads the state for a persistable component.
     * @param component The component to load state into.
     * @param key The unique key for the state to load.
     * @returns True if the state was loaded successfully, false otherwise.
     */
    load(component: Persistable, key: string): Promise<boolean>;
}
