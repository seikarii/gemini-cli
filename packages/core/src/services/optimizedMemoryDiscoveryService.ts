/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Memory, MemoryStatus } from '../types/memory.js';

/**
 * Options for memory search operations
 */
export interface MemorySearchOptions {
  limit?: number;
  minRelevanceScore?: number;
  includeEmbeddings?: boolean;
  status?: MemoryStatus;
}

/**
 * Statistics for the memory service
 */
export interface MemoryServiceStats {
  totalMemories: number;
  activeMemories: number;
  archivedMemories: number;
  deletedMemories: number;
}

/**
 * Optimized memory discovery service for managing memory items
 */
export class OptimizedMemoryDiscoveryService {
  private static instance: OptimizedMemoryDiscoveryService;
  private memories: Map<string, Memory> = new Map();

  private constructor() {}

  static getInstance(): OptimizedMemoryDiscoveryService {
    if (!OptimizedMemoryDiscoveryService.instance) {
      OptimizedMemoryDiscoveryService.instance = new OptimizedMemoryDiscoveryService();
    }
    return OptimizedMemoryDiscoveryService.instance;
  }

  /**
   * Get a memory by ID
   */
  async getMemoryById(id: string): Promise<Memory | null> {
    return this.memories.get(id) || null;
  }

  /**
   * Get all memories
   */
  async getAllMemories(): Promise<Memory[]> {
    return Array.from(this.memories.values());
  }

  /**
   * Save a memory
   */
  async saveMemory(memory: Memory): Promise<Memory> {
    this.memories.set(memory.id, { ...memory });
    return memory;
  }

  /**
   * Delete a memory by ID
   */
  async deleteMemory(id: string): Promise<boolean> {
    return this.memories.delete(id);
  }

  /**
   * Search memories by query
   */
  async searchMemories(query: string, options?: MemorySearchOptions): Promise<Memory[]> {
    const allMemories = Array.from(this.memories.values());
    let results = allMemories.filter(memory =>
      memory.summary.toLowerCase().includes(query.toLowerCase()) ||
      memory.content.toLowerCase().includes(query.toLowerCase()) ||
      memory.metadata.tags?.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
    );

    if (options?.limit) {
      results = results.slice(0, options.limit);
    }

    if (options?.minRelevanceScore !== undefined) {
      results = results.filter(memory => (memory.relevanceScore || 0) >= options.minRelevanceScore!);
    }

    if (options?.includeEmbeddings !== undefined) {
      results = results.filter(memory =>
        options.includeEmbeddings ? memory.embeddings !== undefined : memory.embeddings === undefined
      );
    }

    return results;
  }

  /**
   * Get service statistics
   */
  getStats(): MemoryServiceStats {
    const allMemories = Array.from(this.memories.values());
    const stats = {
      totalMemories: allMemories.length,
      activeMemories: allMemories.filter(m => m.status === MemoryStatus.ACTIVE).length,
      archivedMemories: allMemories.filter(m => m.status === MemoryStatus.ARCHIVED).length,
      deletedMemories: allMemories.filter(m => m.status === MemoryStatus.DELETED).length,
    };
    return stats;
  }
}
