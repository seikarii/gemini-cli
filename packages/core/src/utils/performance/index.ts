/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Concurrency utilities
export { Semaphore } from './concurrency/Semaphore.js';
export { FileOperationPool } from './concurrency/FileOperationPool.js';
export { BufferPool } from './concurrency/BufferPool.js';

// Cache utilities
export { EnhancedLruCache, type EnhancedLruCacheOptions } from './cache/EnhancedLruCache.js';

// File operations
export { 
  OptimizedFileOperations,
  type FileOperationsConfig,
  type FileOperationResult 
} from './fileOperations/OptimizedFileOperations.js';

// Legacy exports (for backwards compatibility)
export { LruCache } from './LruCache.js';
