/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { gzipSync, gunzipSync } from 'zlib';

/**
 * Configuration options for the enhanced LRU cache
 */
export interface EnhancedLruCacheOptions {
  /** Enable compression for cached values */
  enableCompression?: boolean;
  /** Minimum size threshold for compression (bytes) */
  compressionThreshold?: number;
  /** Enable TTL (time-to-live) for cache entries */
  enableTTL?: boolean;
  /** Default TTL in milliseconds */
  defaultTTL?: number;
  /** Enable memory usage tracking */
  trackMemory?: boolean;
}

interface CacheEntry<V> {
  value: V;
  compressed: boolean;
  timestamp: number;
  ttl?: number;
  size?: number;
}

/**
 * Enhanced LRU Cache with compression, TTL, and memory tracking capabilities.
 * Optimized for handling large values and memory-constrained environments.
 */
export class EnhancedLruCache<K, V> {
  private cache = new Map<K, CacheEntry<V>>();
  private maxSize: number;
  private options: Required<EnhancedLruCacheOptions>;
  
  // Statistics
  private hits = 0;
  private misses = 0;
  private compressions = 0;
  private memoryUsage = 0;

  constructor(maxSize: number, options: EnhancedLruCacheOptions = {}) {
    this.maxSize = maxSize;
    this.options = {
      enableCompression: options.enableCompression ?? true,
      compressionThreshold: options.compressionThreshold ?? 1024, // 1KB
      enableTTL: options.enableTTL ?? false,
      defaultTTL: options.defaultTTL ?? 5 * 60 * 1000, // 5 minutes
      trackMemory: options.trackMemory ?? true,
    };
  }

  /**
   * Get a value from the cache
   */
  get(key: K): V | undefined {
    const entry = this.cache.get(key);
    
    if (!entry) {
      this.misses++;
      return undefined;
    }

    // Check TTL if enabled
    if (this.options.enableTTL && entry.ttl) {
      if (Date.now() - entry.timestamp > entry.ttl) {
        this.delete(key);
        this.misses++;
        return undefined;
      }
    }

    this.hits++;

    // Move to end to mark as recently used
    this.cache.delete(key);
    this.cache.set(key, entry);

    // Decompress if needed
    if (entry.compressed && typeof entry.value === 'string') {
      try {
        const decompressed = gunzipSync(Buffer.from(entry.value, 'base64')).toString();
        return decompressed as V;
      } catch (error) {
        console.warn('Failed to decompress cache entry:', error);
        return entry.value;
      }
    }

    return entry.value;
  }

  /**
   * Set a value in the cache
   */
  set(key: K, value: V, ttl?: number): void {
    // Remove existing entry if present
    if (this.cache.has(key)) {
      this.delete(key);
    }

    // Evict least recently used if at capacity
    while (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.delete(firstKey);
      }
    }

    let finalValue: V = value;
    let compressed = false;
    let size = 0;

    // Calculate size and compress if needed
    if (typeof value === 'string') {
      size = Buffer.byteLength(value, 'utf8');
      
      if (this.options.enableCompression && size >= this.options.compressionThreshold) {
        try {
          const compressedBuffer = gzipSync(Buffer.from(value, 'utf8'));
          const compressedSize = compressedBuffer.length;
          
          // Only use compression if it actually reduces size
          if (compressedSize < size * 0.9) {
            finalValue = compressedBuffer.toString('base64') as V;
            compressed = true;
            size = compressedSize;
            this.compressions++;
          }
        } catch (error) {
          console.warn('Failed to compress cache entry:', error);
        }
      }
    } else if (this.options.trackMemory) {
      // Rough estimation for non-string objects
      size = JSON.stringify(value).length * 2; // Assuming UTF-16
    }

    const entry: CacheEntry<V> = {
      value: finalValue,
      compressed,
      timestamp: Date.now(),
      ttl: ttl ?? (this.options.enableTTL ? this.options.defaultTTL : undefined),
      size: this.options.trackMemory ? size : undefined,
    };

    this.cache.set(key, entry);
    
    if (this.options.trackMemory) {
      this.memoryUsage += size;
    }
  }

  /**
   * Delete a key from the cache
   */
  delete(key: K): boolean {
    const entry = this.cache.get(key);
    if (entry && this.options.trackMemory && entry.size) {
      this.memoryUsage -= entry.size;
    }
    return this.cache.delete(key);
  }

  /**
   * Clear all entries from the cache
   */
  clear(): void {
    this.cache.clear();
    this.memoryUsage = 0;
    this.hits = 0;
    this.misses = 0;
    this.compressions = 0;
  }

  /**
   * Check if a key exists in the cache (without updating LRU order)
   */
  has(key: K): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    // Check TTL if enabled
    if (this.options.enableTTL && entry.ttl) {
      if (Date.now() - entry.timestamp > entry.ttl) {
        this.delete(key);
        return false;
      }
    }

    return true;
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const totalRequests = this.hits + this.misses;
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: totalRequests > 0 ? (this.hits / totalRequests) * 100 : 0,
      compressions: this.compressions,
      memoryUsage: this.memoryUsage,
      averageEntrySize: this.cache.size > 0 ? this.memoryUsage / this.cache.size : 0,
    };
  }

  /**
   * Clean up expired entries (useful for TTL-enabled caches)
   */
  cleanup(): number {
    if (!this.options.enableTTL) return 0;

    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.ttl && (now - entry.timestamp) > entry.ttl) {
        this.delete(key);
        cleaned++;
      }
    }

    return cleaned;
  }

  /**
   * Get current cache size
   */
  get size(): number {
    return this.cache.size;
  }
}
