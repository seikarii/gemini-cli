/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Implements the MentalLaby, the core associative memory graph.
 */
import { Film } from './film.js';
import { Persistable } from '../persistence/persistence-service.js';
export type MemoryNodeKind = 'semantic' | 'procedural' | 'episodic';
export interface MemoryNode {
    id: string;
    kind: MemoryNodeKind;
    embedding: number[];
    data: Record<string, any> | Film;
    valence: number;
    arousal: number;
    salience: number;
    lastAccessTimestamp: number;
    usageCount: number;
    edges: Map<string, {
        weight: number;
    }>;
}
/**
 * The MentalLaby class manages the graph of memory nodes.
 */
export declare class MentalLaby implements Persistable {
    private nodes;
    private readonly K_NEAREST_NEIGHBORS;
    private embedder;
    constructor();
    /**
     * Stores a new piece of information in the memory graph.
     */
    store(data: any, kind?: MemoryNodeKind, valence?: number, arousal?: number, salience?: number): string;
    /**
     * Recalls information from memory based on a cue.
     */
    recall(cue: any, maxResults?: number): MemoryNode[];
    private reinforceNode;
    private findSimilarNodes;
    private linkNodes;
    private createEmbedding;
    exportState(): object;
    importState(state: unknown): void;
}
