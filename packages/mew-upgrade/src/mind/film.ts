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
  action: string; // The name of the tool to call, e.g., 'run_shell_command'
  params: Record<string, any>; // The parameters for the tool
  
  // --- Learning Metadata ---
  costEnergy: number; // Estimated cost to execute this action
  expectedReward: number; // Expected reward/success probability
  lastOutcome: number; // Outcome of the last execution (e.g., 1.0 for success, -1.0 for critical failure)
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
  description: string; // A natural language description of what this film does
  tags: string[]; // Tags for categorization and retrieval
  
  nodes: Record<string, FilmNode>; // A map of node IDs to FilmNode objects
  edges: Array<{
    sourceNodeId: string;
    targetNodeId: string;
    // A condition to determine if this edge should be traversed.
    // For now, a simple string, but could evolve to a more complex condition.
    condition?: string; 
  }>;
  
  entryNodeId: string; // The ID of the first node to execute in the film

  // --- Learning Metadata ---
  fitness: number; // Overall effectiveness of this film. Higher is better.
  usageCount: number;
  lastRunTimestamp: number;
}
