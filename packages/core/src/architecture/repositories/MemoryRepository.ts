/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { 
  AbstractRepository, 
  RepositoryResult, 
  RepositoryConfig 
} from './BaseRepository.js';
import { OptimizedMemoryDiscoveryService } from '../../services/optimizedMemoryDiscoveryService.js';
import { Memory, MemoryStatus } from '../../types/memory.js';

/**
 * Memory entity representation with enhanced metadata
 */
export interface MemoryEntity {
  id: string;
  summary: string;
  content: string;
  metadata: {
    timestamp: Date;
    source?: string;
    tags?: string[];
    importance?: number;
    type?: string;
    context?: Record<string, unknown>;
  };
  status: MemoryStatus;
  relevanceScore?: number;
  embeddings?: number[];
}

/**
 * Memory search criteria
 */
export interface MemorySearchCriteria {
  query?: string;
  tags?: string[];
  importance?: number;
  type?: string;
  createdAfter?: Date;
  createdBefore?: Date;
  status?: MemoryStatus;
  minRelevanceScore?: number;
  hasEmbeddings?: boolean;
}

/**
 * Repository interface for memory operations
 */
export interface IMemoryRepository {
  findBySummary(summary: string): Promise<RepositoryResult<MemoryEntity>>;
  findByTags(tags: string[]): Promise<RepositoryResult<MemoryEntity[]>>;
  findByImportance(minImportance: number): Promise<RepositoryResult<MemoryEntity[]>>;
  findSimilar(content: string, limit?: number): Promise<RepositoryResult<MemoryEntity[]>>;
  search(criteria: MemorySearchCriteria): Promise<RepositoryResult<MemoryEntity[]>>;
  updateRelevanceScore(id: string, score: number): Promise<RepositoryResult<MemoryEntity>>;
  updateStatus(id: string, status: MemoryStatus): Promise<RepositoryResult<MemoryEntity>>;
  addTags(id: string, tags: string[]): Promise<RepositoryResult<MemoryEntity>>;
  removeTags(id: string, tags: string[]): Promise<RepositoryResult<MemoryEntity>>;
  getStatistics(): Promise<RepositoryResult<MemoryStatistics>>;
}

/**
 * Memory statistics interface
 */
export interface MemoryStatistics {
  totalMemories: number;
  memoriesByStatus: Record<MemoryStatus, number>;
  memoriesByType: Record<string, number>;
  averageImportance: number;
  memoriesWithEmbeddings: number;
  oldestMemory?: Date;
  newestMemory?: Date;
  topTags: Array<{ tag: string; count: number }>;
}

/**
 * Memory repository implementation with vector search and optimization
 */
export class MemoryRepository extends AbstractRepository<MemoryEntity> implements IMemoryRepository {
  private memoryService: OptimizedMemoryDiscoveryService;

  constructor(
    config: RepositoryConfig = {},
    memoryService?: OptimizedMemoryDiscoveryService
  ) {
    super(config);
    this.memoryService = memoryService || OptimizedMemoryDiscoveryService.getInstance();
  }

  /**
   * Convert Memory to MemoryEntity
   */
  private memoryToEntity(memory: Memory): MemoryEntity {
    return {
      id: memory.id,
      summary: memory.summary,
      content: memory.content,
      metadata: {
        timestamp: memory.metadata.timestamp,
        source: memory.metadata.source,
        tags: memory.metadata.tags,
        importance: memory.metadata.importance,
        type: memory.metadata.type,
        context: memory.metadata.context,
      },
      status: memory.status,
      relevanceScore: memory.relevanceScore,
      embeddings: memory.embeddings,
    };
  }

  /**
   * Convert MemoryEntity to Memory
   */
  private entityToMemory(entity: MemoryEntity): Memory {
    return {
      id: entity.id,
      summary: entity.summary,
      content: entity.content,
      metadata: entity.metadata,
      status: entity.status,
      relevanceScore: entity.relevanceScore,
      embeddings: entity.embeddings,
    };
  }

  /**
   * Find memory by ID
   */
  async findById(id: string): Promise<MemoryEntity | null> {
    const memory = await this.memoryService.getMemoryById(id);
    return memory ? this.memoryToEntity(memory) : null;
  }

  /**
   * Find all memories with pagination
   */
  async findAll(): Promise<MemoryEntity[]> {
    const memories = await this.memoryService.getAllMemories();
    return memories.map((memory: Memory) => this.memoryToEntity(memory));
  }

  /**
   * Save memory entity
   */
  async save(entity: MemoryEntity): Promise<MemoryEntity> {
    const memory = this.entityToMemory(entity);
    const savedMemory = await this.memoryService.saveMemory(memory);
    return this.memoryToEntity(savedMemory);
  }

  /**
   * Delete memory
   */
  async delete(id: string): Promise<boolean> {
    return this.memoryService.deleteMemory(id);
  }

  /**
   * Check if memory exists
   */
  async exists(id: string): Promise<boolean> {
    const memory = await this.memoryService.getMemoryById(id);
    return memory !== null;
  }

  /**
   * Find memory by summary
   */
  async findBySummary(summary: string): Promise<RepositoryResult<MemoryEntity>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const memories = await this.memoryService.searchMemories(summary, {
          limit: 1,
          status: MemoryStatus.ACTIVE,
        });

        const exactMatch = memories.find(m => m.summary === summary);
        if (!exactMatch) {
          return this.createErrorResult<MemoryEntity>('Memory not found', {
            operationTime: Date.now() - startTime,
            source: 'MemoryRepository.findBySummary',
          });
        }

        return this.createSuccessResult(this.memoryToEntity(exactMatch), {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findBySummary',
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<MemoryEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findBySummary',
        }
      );
    }
  }

  /**
   * Find memories by tags
   */
  async findByTags(tags: string[]): Promise<RepositoryResult<MemoryEntity[]>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const allMemories = await this.memoryService.getAllMemories();
        const matchingMemories = allMemories.filter((memory: Memory) => 
          memory.metadata.tags && 
          tags.some(tag => memory.metadata.tags!.includes(tag))
        );

        const entities = matchingMemories.map((memory: Memory) => this.memoryToEntity(memory));

        return this.createSuccessResult(entities, {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findByTags',
          resultCount: entities.length,
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<MemoryEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findByTags',
        }
      );
    }
  }

  /**
   * Find memories by minimum importance
   */
  async findByImportance(minImportance: number): Promise<RepositoryResult<MemoryEntity[]>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const allMemories = await this.memoryService.getAllMemories();
        const importantMemories = allMemories.filter((memory: Memory) => 
          memory.metadata.importance !== undefined && 
          memory.metadata.importance >= minImportance
        );

        // Sort by importance descending
        importantMemories.sort((a: Memory, b: Memory) => 
          (b.metadata.importance || 0) - (a.metadata.importance || 0)
        );

        const entities = importantMemories.map((memory: Memory) => this.memoryToEntity(memory));

        return this.createSuccessResult(entities, {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findByImportance',
          resultCount: entities.length,
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<MemoryEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findByImportance',
        }
      );
    }
  }

  /**
   * Find similar memories using vector search
   */
  async findSimilar(
    content: string, 
    limit = 10
  ): Promise<RepositoryResult<MemoryEntity[]>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const similarMemories = await this.memoryService.searchMemories(content, {
          limit,
          status: MemoryStatus.ACTIVE,
        });

        const entities = similarMemories.map((memory: Memory) => this.memoryToEntity(memory));

        return this.createSuccessResult(entities, {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findSimilar',
          resultCount: entities.length,
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<MemoryEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.findSimilar',
        }
      );
    }
  }

  /**
   * Search memories with complex criteria
   */
  async search(criteria: MemorySearchCriteria): Promise<RepositoryResult<MemoryEntity[]>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        let memories = await this.memoryService.getAllMemories();

        // Apply filters
        if (criteria.status) {
          memories = memories.filter((m: Memory) => m.status === criteria.status);
        }

        if (criteria.tags && criteria.tags.length > 0) {
          memories = memories.filter((m: Memory) => 
            m.metadata.tags && 
            criteria.tags!.some(tag => m.metadata.tags!.includes(tag))
          );
        }

        if (criteria.importance !== undefined) {
          memories = memories.filter((m: Memory) => 
            m.metadata.importance !== undefined && 
            m.metadata.importance >= criteria.importance!
          );
        }

        if (criteria.type) {
          memories = memories.filter((m: Memory) => m.metadata.type === criteria.type);
        }

        if (criteria.createdAfter) {
          memories = memories.filter((m: Memory) => 
            m.metadata.timestamp >= criteria.createdAfter!
          );
        }

        if (criteria.createdBefore) {
          memories = memories.filter((m: Memory) => 
            m.metadata.timestamp <= criteria.createdBefore!
          );
        }

        if (criteria.minRelevanceScore !== undefined) {
          memories = memories.filter((m: Memory) => 
            m.relevanceScore !== undefined && 
            m.relevanceScore >= criteria.minRelevanceScore!
          );
        }

        if (criteria.hasEmbeddings !== undefined) {
          memories = memories.filter((m: Memory) => 
            criteria.hasEmbeddings ? 
            (m.embeddings && m.embeddings.length > 0) : 
            (!m.embeddings || m.embeddings.length === 0)
          );
        }

        // If query is provided, perform semantic search
        if (criteria.query) {
          const searchResults = await this.memoryService.searchMemories(
            criteria.query, 
            { limit: 100, status: criteria.status }
          );
          
          // Intersect with filtered results
          const searchIds = new Set(searchResults.map((m: Memory) => m.id));
          memories = memories.filter((m: Memory) => searchIds.has(m.id));
          
          // Preserve relevance order from search
          const relevanceMap = new Map(
            searchResults.map((m: Memory, index: number) => [m.id, index])
          );
          memories.sort((a: Memory, b: Memory) => 
            (relevanceMap.get(a.id) || 999) - (relevanceMap.get(b.id) || 999)
          );
        }

        const entities = memories.map((memory: Memory) => this.memoryToEntity(memory));

        return this.createSuccessResult(entities, {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.search',
          resultCount: entities.length,
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<MemoryEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.search',
        }
      );
    }
  }

  /**
   * Update relevance score for a memory
   */
  async updateRelevanceScore(
    id: string, 
    score: number
  ): Promise<RepositoryResult<MemoryEntity>> {
    const startTime = Date.now();

    try {
      const memory = await this.memoryService.getMemoryById(id);
      if (!memory) {
        return this.createErrorResult<MemoryEntity>('Memory not found', {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.updateRelevanceScore',
        });
      }

      memory.relevanceScore = score;
      const updatedMemory = await this.memoryService.saveMemory(memory);

      return this.createSuccessResult(this.memoryToEntity(updatedMemory), {
        operationTime: Date.now() - startTime,
        source: 'MemoryRepository.updateRelevanceScore',
      });
    } catch (error) {
      return this.createErrorResult<MemoryEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.updateRelevanceScore',
        }
      );
    }
  }

  /**
   * Update status for a memory
   */
  async updateStatus(
    id: string, 
    status: MemoryStatus
  ): Promise<RepositoryResult<MemoryEntity>> {
    const startTime = Date.now();

    try {
      const memory = await this.memoryService.getMemoryById(id);
      if (!memory) {
        return this.createErrorResult<MemoryEntity>('Memory not found', {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.updateStatus',
        });
      }

      memory.status = status;
      const updatedMemory = await this.memoryService.saveMemory(memory);

      return this.createSuccessResult(this.memoryToEntity(updatedMemory), {
        operationTime: Date.now() - startTime,
        source: 'MemoryRepository.updateStatus',
      });
    } catch (error) {
      return this.createErrorResult<MemoryEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.updateStatus',
        }
      );
    }
  }

  /**
   * Add tags to a memory
   */
  async addTags(id: string, tags: string[]): Promise<RepositoryResult<MemoryEntity>> {
    const startTime = Date.now();

    try {
      const memory = await this.memoryService.getMemoryById(id);
      if (!memory) {
        return this.createErrorResult<MemoryEntity>('Memory not found', {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.addTags',
        });
      }

      const existingTags = memory.metadata.tags || [];
      const newTags = [...new Set([...existingTags, ...tags])];
      memory.metadata.tags = newTags;

      const updatedMemory = await this.memoryService.saveMemory(memory);

      return this.createSuccessResult(this.memoryToEntity(updatedMemory), {
        operationTime: Date.now() - startTime,
        source: 'MemoryRepository.addTags',
      });
    } catch (error) {
      return this.createErrorResult<MemoryEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.addTags',
        }
      );
    }
  }

  /**
   * Remove tags from a memory
   */
  async removeTags(id: string, tags: string[]): Promise<RepositoryResult<MemoryEntity>> {
    const startTime = Date.now();

    try {
      const memory = await this.memoryService.getMemoryById(id);
      if (!memory) {
        return this.createErrorResult<MemoryEntity>('Memory not found', {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.removeTags',
        });
      }

      const existingTags = memory.metadata.tags || [];
      const remainingTags = existingTags.filter(tag => !tags.includes(tag));
      memory.metadata.tags = remainingTags;

      const updatedMemory = await this.memoryService.saveMemory(memory);

      return this.createSuccessResult(this.memoryToEntity(updatedMemory), {
        operationTime: Date.now() - startTime,
        source: 'MemoryRepository.removeTags',
      });
    } catch (error) {
      return this.createErrorResult<MemoryEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.removeTags',
        }
      );
    }
  }

  /**
   * Get comprehensive memory statistics
   */
  async getStatistics(): Promise<RepositoryResult<MemoryStatistics>> {
    const startTime = Date.now();

    try {
      const memories = await this.memoryService.getAllMemories();

      const statistics: MemoryStatistics = {
        totalMemories: memories.length,
        memoriesByStatus: {
          [MemoryStatus.ACTIVE]: 0,
          [MemoryStatus.ARCHIVED]: 0,
          [MemoryStatus.DELETED]: 0,
        },
        memoriesByType: {},
        averageImportance: 0,
        memoriesWithEmbeddings: 0,
        topTags: [],
      };

      let totalImportance = 0;
      let importanceCount = 0;
      const tagCounts = new Map<string, number>();

      for (const memory of memories) {
        // Count by status
        statistics.memoriesByStatus[memory.status]++;

        // Count by type
        const type = memory.metadata.type || 'unknown';
        statistics.memoriesByType[type] = (statistics.memoriesByType[type] || 0) + 1;

        // Calculate average importance
        if (memory.metadata.importance !== undefined) {
          totalImportance += memory.metadata.importance;
          importanceCount++;
        }

        // Count embeddings
        if (memory.embeddings && memory.embeddings.length > 0) {
          statistics.memoriesWithEmbeddings++;
        }

        // Count tags
        if (memory.metadata.tags) {
          for (const tag of memory.metadata.tags) {
            tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
          }
        }

        // Track oldest and newest
        if (!statistics.oldestMemory || memory.metadata.timestamp < statistics.oldestMemory) {
          statistics.oldestMemory = memory.metadata.timestamp;
        }
        if (!statistics.newestMemory || memory.metadata.timestamp > statistics.newestMemory) {
          statistics.newestMemory = memory.metadata.timestamp;
        }
      }

      statistics.averageImportance = importanceCount > 0 ? totalImportance / importanceCount : 0;

      // Get top 10 tags
      statistics.topTags = Array.from(tagCounts.entries())
        .map(([tag, count]) => ({ tag, count }))
        .sort((a: { tag: string; count: number }, b: { tag: string; count: number }) => b.count - a.count)
        .slice(0, 10);

      return this.createSuccessResult(statistics, {
        operationTime: Date.now() - startTime,
        source: 'MemoryRepository.getStatistics',
      });
    } catch (error) {
      return this.createErrorResult<MemoryStatistics>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'MemoryRepository.getStatistics',
        }
      );
    }
  }

  /**
   * Get repository health and performance stats
   */
  getRepositoryStats() {
    return {
      config: this.config,
      memoryService: this.memoryService.getStats?.() || null,
    };
  }
}
