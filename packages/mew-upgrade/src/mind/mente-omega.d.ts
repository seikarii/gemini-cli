/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { EntityMemory } from './entity-memory.js';
import type { ContentGenerator } from '@google/gemini-cli-core';
import type { Persistable } from '../persistence/persistence-service.js';
export declare class MenteOmega implements Persistable {
    memory: EntityMemory;
    private actions;
    private menteAlfa;
    private hashingEmbedder;
    constructor(menteAlfa: ContentGenerator);
    /**
     * The main processing loop for the agent's mind.
     * @param userRequest The initial request from the user.
     */
    process(userRequest: string): Promise<void>;
    private getPlanFromMenteAlfa;
    exportState(): object;
    importState(state: unknown): void;
}
