/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { EnhancedLruCache } from '../EnhancedLruCache.js';

describe('EnhancedLruCache', () => {
  let cache: EnhancedLruCache<string, string>;

  beforeEach(() => {
    cache = new EnhancedLruCache<string, string>(3, {
      enableCompression: true,
      enableTTL: false,
      trackMemory: true,
    });
  });

  it('should store and retrieve values', () => {
    cache.set('key1', 'value1');
    expect(cache.get('key1')).toBe('value1');
  });

  it('should respect LRU ordering', () => {
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');
    cache.set('key3', 'value3');
    cache.set('key4', 'value4'); // Should evict key1
    
    expect(cache.get('key1')).toBeUndefined();
    expect(cache.get('key2')).toBe('value2');
    expect(cache.get('key3')).toBe('value3');
    expect(cache.get('key4')).toBe('value4');
  });

  it('should update LRU order on access', () => {
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');
    cache.set('key3', 'value3');
    
    // Access key1 to make it most recent
    cache.get('key1');
    
    // Add key4 - should evict key2 (oldest unaccessed)
    cache.set('key4', 'value4');
    
    expect(cache.get('key1')).toBe('value1'); // Should still exist
    expect(cache.get('key2')).toBeUndefined(); // Should be evicted
    expect(cache.get('key3')).toBe('value3');
    expect(cache.get('key4')).toBe('value4');
  });

  it('should track statistics', () => {
    cache.set('key1', 'value1');
    cache.get('key1'); // hit
    cache.get('key2'); // miss
    
    const stats = cache.getStats();
    expect(stats.hits).toBe(1);
    expect(stats.misses).toBe(1);
    expect(stats.hitRate).toBe(50);
  });

  it('should compress large values', () => {
    const largeValue = 'x'.repeat(2000); // Larger than compression threshold
    cache.set('large', largeValue);
    
    const retrieved = cache.get('large');
    expect(retrieved).toBe(largeValue);
    
    const stats = cache.getStats();
    expect(stats.compressions).toBeGreaterThan(0);
  });

  it('should handle TTL when enabled', async () => {
    const ttlCache = new EnhancedLruCache<string, string>(3, {
      enableTTL: true,
      defaultTTL: 50, // 50ms
    });
    
    ttlCache.set('key1', 'value1');
    expect(ttlCache.get('key1')).toBe('value1');
    
    // Wait for TTL to expire
    await new Promise(resolve => setTimeout(resolve, 60));
    
    expect(ttlCache.get('key1')).toBeUndefined();
  });

  it('should track memory usage', () => {
    const testValue = 'x'.repeat(1000);
    cache.set('key1', testValue);
    
    const stats = cache.getStats();
    expect(stats.memoryUsage).toBeGreaterThan(0);
  });

  it('should clean up expired entries', async () => {
    const ttlCache = new EnhancedLruCache<string, string>(10, {
      enableTTL: true,
      defaultTTL: 50,
    });
    
    ttlCache.set('key1', 'value1');
    ttlCache.set('key2', 'value2');
    
    // Wait for expiration
    await new Promise(resolve => setTimeout(resolve, 60));
    
    const cleaned = ttlCache.cleanup();
    expect(cleaned).toBe(2);
    expect(ttlCache.size).toBe(0);
  });
});
