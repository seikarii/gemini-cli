import { RAGLogger } from '../logger.js';
import { createHash } from 'crypto';

/**
 * Configuration for embedding cache service
 */
export interface EmbeddingCacheConfig {
  /** Maximum number of cached embeddings */
  maxSize: number;
  /** Cache TTL in milliseconds */
  ttlMs: number;
  /** Enable/disable the cache */
  enabled: boolean;
  /** Enable LRU eviction strategy */
  lruEviction: boolean;
}

/**
 * Cached embedding entry
 */
interface CacheEntry {
  /** The embedding vector */
  embedding: number[];
  /** Timestamp when cached */
  timestamp: number;
  /** Last access timestamp for LRU */
  lastAccessed: number;
  /** Hash of the content for validation */
  contentHash: string;
}

/**
 * High-performance embedding cache service for RAG operations.
 * Implements intelligent caching strategies to reduce API calls and improve response times.
 */
export class EmbeddingCacheService {
  private cache = new Map<string, CacheEntry>();
  private readonly logger: RAGLogger;
  private readonly config: EmbeddingCacheConfig;
  private stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    totalRequests: 0,
  };

  constructor(logger: RAGLogger, config: Partial<EmbeddingCacheConfig> = {}) {
    this.logger = logger;
    this.config = {
      maxSize: 10000,
      ttlMs: 24 * 60 * 60 * 1000, // 24 hours
      enabled: true,
      lruEviction: true,
      ...config,
    };

    this.logger.info('EmbeddingCacheService initialized', {
      config: this.config,
    });
  }

  /**
   * Get embedding from cache if available and valid
   */
  async get(content: string): Promise<number[] | null> {
    if (!this.config.enabled) {
      return null;
    }

    this.stats.totalRequests++;
    const key = this.generateKey(content);
    const entry = this.cache.get(key);

    if (!entry) {
      this.stats.misses++;
      return null;
    }

    // Check TTL
    const now = Date.now();
    if (now - entry.timestamp > this.config.ttlMs) {
      this.cache.delete(key);
      this.stats.misses++;
      this.stats.evictions++;
      return null;
    }

    // Validate content hasn't changed
    const currentHash = this.generateContentHash(content);
    if (currentHash !== entry.contentHash) {
      this.cache.delete(key);
      this.stats.misses++;
      this.stats.evictions++;
      return null;
    }

    // Update LRU access time
    if (this.config.lruEviction) {
      entry.lastAccessed = now;
    }

    this.stats.hits++;
    this.logger.debug('Embedding cache hit', {
      key: key.substring(0, 16) + '...',
    });
    return entry.embedding;
  }

  /**
   * Store embedding in cache
   */
  async set(content: string, embedding: number[]): Promise<void> {
    if (!this.config.enabled) {
      return;
    }

    const key = this.generateKey(content);
    const now = Date.now();
    const contentHash = this.generateContentHash(content);

    const entry: CacheEntry = {
      embedding,
      timestamp: now,
      lastAccessed: now,
      contentHash,
    };

    // Evict if necessary
    if (this.cache.size >= this.config.maxSize) {
      this.evictOldestEntry();
    }

    this.cache.set(key, entry);
    this.logger.debug('Stored embedding in cache', {
      key: key.substring(0, 16) + '...',
      size: this.cache.size,
    });
  }

  /**
   * Clear all cached embeddings
   */
  async clear(): Promise<void> {
    const size = this.cache.size;
    this.cache.clear();
    this.stats = { hits: 0, misses: 0, evictions: 0, totalRequests: 0 };
    this.logger.info('Embedding cache cleared', { previousSize: size });
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const hitRate =
      this.stats.totalRequests > 0
        ? (this.stats.hits / this.stats.totalRequests) * 100
        : 0;

    return {
      ...this.stats,
      hitRate: Math.round(hitRate * 100) / 100,
      currentSize: this.cache.size,
      maxSize: this.config.maxSize,
    };
  }

  /**
   * Remove expired entries
   */
  async cleanup(): Promise<void> {
    const now = Date.now();
    let removedCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.config.ttlMs) {
        this.cache.delete(key);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      this.stats.evictions += removedCount;
      this.logger.debug('Cache cleanup completed', { removedCount });
    }
  }

  /**
   * Bulk get embeddings for multiple contents
   */
  async getMany(contents: string[]): Promise<Array<number[] | null>> {
    if (!this.config.enabled) {
      return contents.map(() => null);
    }

    return Promise.all(contents.map((content) => this.get(content)));
  }

  /**
   * Bulk set embeddings for multiple contents
   */
  async setMany(
    pairs: Array<{ content: string; embedding: number[] }>,
  ): Promise<void> {
    if (!this.config.enabled) {
      return;
    }

    await Promise.all(
      pairs.map((pair) => this.set(pair.content, pair.embedding)),
    );
  }

  private generateKey(content: string): string {
    // Use content hash as key for efficient lookups
    return this.generateContentHash(content);
  }

  private generateContentHash(content: string): string {
    return createHash('sha256').update(content).digest('hex');
  }

  private evictOldestEntry(): void {
    if (this.config.lruEviction) {
      // Find LRU entry
      let oldestKey: string | null = null;
      let oldestTime = Date.now();

      for (const [key, entry] of this.cache.entries()) {
        if (entry.lastAccessed < oldestTime) {
          oldestTime = entry.lastAccessed;
          oldestKey = key;
        }
      }

      if (oldestKey) {
        this.cache.delete(oldestKey);
        this.stats.evictions++;
      }
    } else {
      // Simple FIFO eviction - remove first entry
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
        this.stats.evictions++;
      }
    }
  }
}
