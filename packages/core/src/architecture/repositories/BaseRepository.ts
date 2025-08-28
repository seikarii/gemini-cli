/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Base interface for all repository operations
 */
export interface BaseRepository<T, K = string> {
  findById(id: K): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: K): Promise<boolean>;
  exists(id: K): Promise<boolean>;
}

/**
 * Repository result wrapper for consistent error handling
 */
export interface RepositoryResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    fromCache?: boolean;
    operationTime?: number;
    source?: string;
    resultCount?: number;
  };
}

/**
 * Repository configuration options
 */
export interface RepositoryConfig {
  enableCaching?: boolean;
  cacheTTL?: number;
  enableRetries?: boolean;
  maxRetries?: number;
  enableMetrics?: boolean;
}

/**
 * Query parameters for repository operations
 */
export interface QueryOptions {
  limit?: number;
  offset?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  filters?: Record<string, unknown>;
  includeMetadata?: boolean;
}

/**
 * Base repository implementation with common functionality
 */
export abstract class AbstractRepository<T, K = string> implements BaseRepository<T, K> {
  protected config: Required<RepositoryConfig>;

  constructor(config: RepositoryConfig = {}) {
    this.config = {
      enableCaching: config.enableCaching ?? true,
      cacheTTL: config.cacheTTL ?? 5 * 60 * 1000, // 5 minutes
      enableRetries: config.enableRetries ?? true,
      maxRetries: config.maxRetries ?? 3,
      enableMetrics: config.enableMetrics ?? true,
    };
  }

  abstract findById(id: K): Promise<T | null>;
  abstract findAll(): Promise<T[]>;
  abstract save(entity: T): Promise<T>;
  abstract delete(id: K): Promise<boolean>;
  abstract exists(id: K): Promise<boolean>;

  /**
   * Find entities with query options
   */
  async findWithOptions(options: QueryOptions = {}): Promise<T[]> {
    const entities = await this.findAll();
    
    let result = entities;

    // Apply filters
    if (options.filters) {
      result = this.applyFilters(result, options.filters);
    }

    // Apply sorting
    if (options.sortBy) {
      result = this.applySorting(result, options.sortBy, options.sortOrder || 'asc');
    }

    // Apply pagination
    if (options.offset !== undefined || options.limit !== undefined) {
      const offset = options.offset || 0;
      const limit = options.limit || result.length;
      result = result.slice(offset, offset + limit);
    }

    return result;
  }

  /**
   * Apply filters to entity list
   */
  protected applyFilters(entities: T[], filters: Record<string, unknown>): T[] {
    return entities.filter(entity => 
      Object.entries(filters).every(([key, value]) => {
        const entityValue = (entity as Record<string, unknown>)[key];
        if (value instanceof RegExp) {
          return value.test(String(entityValue));
        }
        return entityValue === value;
      })
    );
  }

  /**
   * Apply sorting to entity list
   */
  protected applySorting(entities: T[], sortBy: string, sortOrder: 'asc' | 'desc'): T[] {
    return entities.sort((a, b) => {
      const aValue = (a as Record<string, unknown>)[sortBy];
      const bValue = (b as Record<string, unknown>)[sortBy];
      
      // Convert to comparable values
      const aComp = aValue === null || aValue === undefined ? '' : String(aValue);
      const bComp = bValue === null || bValue === undefined ? '' : String(bValue);
      
      if (aComp < bComp) return sortOrder === 'asc' ? -1 : 1;
      if (aComp > bComp) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
  }

  /**
   * Execute operation with retry logic
   */
  protected async withRetry<R>(operation: () => Promise<R>): Promise<R> {
    if (!this.config.enableRetries) {
      return operation();
    }

    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (attempt === this.config.maxRetries) {
          throw lastError;
        }
        
        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError || new Error('Operation failed after retries');
  }

  /**
   * Create a successful repository result
   */
  protected createSuccessResult<R>(
    data: R, 
    metadata?: RepositoryResult<R>['metadata']
  ): RepositoryResult<R> {
    return {
      success: true,
      data,
      metadata,
    };
  }

  /**
   * Create a failed repository result
   */
  protected createErrorResult<R>(
    error: string, 
    metadata?: RepositoryResult<R>['metadata']
  ): RepositoryResult<R> {
    return {
      success: false,
      error,
      metadata,
    };
  }
}
