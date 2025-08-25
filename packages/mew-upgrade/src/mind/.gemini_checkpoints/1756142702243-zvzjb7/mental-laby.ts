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
import { HashingEmbedder } from './embeddings.js';

// --- Utility Functions ---

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) {
    return 0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) {
    return 0;
  }

  return dotProduct / denominator;
}

// --- Interfaces ---

export interface MemoryNode {
  id: string;
  kind: 'semantic' | 'procedural' | 'episodic';
  embedding: number[];
  data: Record<string, any> | Film;
  salience: number;
  valence: number;
  arousal: number;
  lastAccessTimestamp: number;
  usageCount: number;
  edges: Map<string, { weight: number }>; // targetNodeId -> { weight }
}

/**
 * The MentalLaby class manages the graph of memory nodes.
 */
export class MentalLaby implements Persistable {
  private nodes: Map<string, MemoryNode> = new Map();
  private readonly K_NEAREST_NEIGHBORS = 3; // Number of neighbors to link to
  private embedder: HashingEmbedder;

  constructor() {
    this.embedder = new HashingEmbedder();
  }

  /**
   * Stores a new piece of information in the memory graph.
   */
  store(data: any, kind: MemoryNode['kind'] = 'semantic'): string {
    const embedding = this.createEmbedding(data);
    const similarNodes = this.findSimilarNodes(embedding, this.K_NEAREST_NEIGHBORS + 1);

    // Check if a very similar node already exists
    if (similarNodes.length > 0 && similarNodes[0].similarity > 0.98) {
      const existingNode = this.nodes.get(similarNodes[0].id)!;
      this.reinforceNode(existingNode);
      return existingNode.id;
    }

    const newNode: MemoryNode = {
      id: `node_${this.nodes.size + 1}`,
      kind,
      embedding,
      data,
      salience: 0.5,
      valence: 0,
      arousal: 0.5,
      lastAccessTimestamp: Date.now(),
      usageCount: 1,
      edges: new Map(),
    };
    this.nodes.set(newNode.id, newNode);

    // Link to the K nearest neighbors
    const neighborsToLink = similarNodes.slice(0, this.K_NEAREST_NEIGHBORS);
    this.linkNodes(newNode, neighborsToLink);

    console.log(`MentalLaby: Stored new memory node ${newNode.id}, linked to ${neighborsToLink.length} neighbors.`);
    return newNode.id;
  }

  /**
   * Recalls information from memory based on a cue.
   */
  recall(cue: any, maxResults = 5): MemoryNode[] {
    if (this.nodes.size === 0) return [];

    const cueEmbedding = this.createEmbedding(cue);
    const initialMatches = this.findSimilarNodes(cueEmbedding, this.K_NEAREST_NEIGHBORS);

    const activationScores: Map<string, number> = new Map();

    // Activate initial matches
    for (const match of initialMatches) {
      activationScores.set(match.id, match.similarity);
    }

    // Spread activation one level deep
    for (const match of initialMatches) {
      const node = this.nodes.get(match.id);
      if (node) {
        for (const [neighborId, edge] of node.edges.entries()) {
          const currentScore = activationScores.get(neighborId) || 0;
          // Activation spreads based on initial similarity and edge weight
          const activationSpread = match.similarity * edge.weight;
          activationScores.set(neighborId, currentScore + activationSpread);
        }
      }
    }

    // Sort by final activation score and return the top nodes
    const sortedRecall = Array.from(activationScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxResults);

    return sortedRecall.map(([id]) => this.nodes.get(id)!);
  }

  private reinforceNode(node: MemoryNode) {
    node.usageCount++;
    node.salience = Math.min(1, node.salience + 0.1);
    node.lastAccessTimestamp = Date.now();
    console.log(`MentalLaby: Reinforced memory node ${node.id}`);
  }

  private findSimilarNodes(embedding: number[], k: number): Array<{ id: string; similarity: number }> {
    const similarities: Array<{ id: string; similarity: number }> = [];
    for (const node of this.nodes.values()) {
      const similarity = cosineSimilarity(embedding, node.embedding);
      if (similarity > 0) { // Only consider nodes with some similarity
        similarities.push({ id: node.id, similarity });
      }
    }
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, k);
  }

  private linkNodes(sourceNode: MemoryNode, targets: Array<{ id: string; similarity: number }>) {
    for (const target of targets) {
      const targetNode = this.nodes.get(target.id);
      if (targetNode) {
        // Create bidirectional links, weight by similarity
        sourceNode.edges.set(targetNode.id, { weight: target.similarity });
        targetNode.edges.set(sourceNode.id, { weight: target.similarity });
      }
    }
  }

  private createEmbedding(data: any): number[] {
    return this.embedder.embed(data);
  }

  // --- Persistence --- 

  exportState(): object {
    const nodesArray = Array.from(this.nodes.values()).map(node => ({
      ...node,
      edges: Array.from(node.edges.entries()),
    }));
    return { nodes: nodesArray };
  }

  importState(state: any): void {
    if (state && state.nodes) {
      this.nodes.clear();
      for (const nodeData of state.nodes) {
        this.nodes.set(nodeData.id, {
          ...nodeData,
          edges: new Map(nodeData.edges),
        });
      }
    }
  }
}
