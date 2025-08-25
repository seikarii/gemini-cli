/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Defines the core data structures for the agent's procedural memory.
 * This is based on the "Film" concept from the Mew architecture.
 */
/**
 * Represents a single node or step within a Film.
 * It is a single, atomic action to be executed.
 */
export interface FilmNode {
    id: string;
    action: string;
    params: Record<string, any>;
    costEnergy: number;
    expectedReward: number;
    lastOutcome: number;
    usageCount: number;
    lastExecutedTimestamp: number;
}
/**
 * Represents a "Film" or a learned procedure.
 * It is a sequence of actions (FilmNodes) that, when executed together,
 * accomplish a specific, repeatable task. This is equivalent to a habit or a skill.
 */
export interface Film {
    id: string;
    description: string;
    tags: string[];
    nodes: Record<string, FilmNode>;
    edges: Array<{
        sourceNodeId: string;
        targetNodeId: string;
        condition?: string;
    }>;
    entryNodeId: string;
    fitness: number;
    usageCount: number;
    lastRunTimestamp: number;
}
