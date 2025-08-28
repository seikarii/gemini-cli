/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Semaphore } from './Semaphore.js';

/**
 * Pool for managing and deduplicating file operations to improve performance.
 * Prevents duplicate operations and controls concurrency to avoid overwhelming the file system.
 */
export class FileOperationPool {
  private pool = new Map<string, Promise<unknown>>();
  private semaphore: Semaphore;
  private maxPoolSize: number;
  private operationCount = 0;

  constructor(maxConcurrent: number = 10, maxPoolSize: number = 1000) {
    this.semaphore = new Semaphore(maxConcurrent);
    this.maxPoolSize = maxPoolSize;
  }

  /**
   * Execute an operation, deduplicating identical operations and controlling concurrency.
   * @param key - Unique identifier for the operation (e.g., file path)
   * @param operation - The async operation to execute
   * @returns Promise resolving to the operation result
   */
  async execute<T>(key: string, operation: () => Promise<T>): Promise<T> {
    // Check if operation is already in progress
    if (this.pool.has(key)) {
      return this.pool.get(key)! as Promise<T>;
    }

    // Clean pool if it gets too large
    if (this.pool.size >= this.maxPoolSize) {
      this.cleanExpiredOperations();
    }

    const promise = this.executeWithSemaphore(operation);
    
    // Cache the promise
    this.pool.set(key, promise);
    
    // Clean up after completion
    promise.finally(() => {
      this.pool.delete(key);
    });

    return promise;
  }

  /**
   * Execute operation with semaphore control
   */
  private async executeWithSemaphore<T>(operation: () => Promise<T>): Promise<T> {
    await this.semaphore.acquire();
    
    try {
      this.operationCount++;
      return await operation();
    } finally {
      this.semaphore.release();
    }
  }

  /**
   * Clean up expired operations (those that are already settled)
   */
  private cleanExpiredOperations(): void {
    const keysToDelete: string[] = [];
    
    for (const [key, promise] of this.pool.entries()) {
      // Use Promise.race with a resolved promise to check if settled
      Promise.race([promise, Promise.resolve('__RESOLVED__')])
        .then((result) => {
          if (result === '__RESOLVED__') {
            keysToDelete.push(key);
          }
        })
        .catch(() => {
          keysToDelete.push(key);
        });
    }

    // Clean up in next tick to avoid modifying during iteration
    process.nextTick(() => {
      keysToDelete.forEach(key => this.pool.delete(key));
    });
  }

  /**
   * Get current pool statistics
   */
  getStats() {
    return {
      poolSize: this.pool.size,
      availablePermits: this.semaphore.available(),
      waitingOperations: this.semaphore.waitingCount(),
      totalOperations: this.operationCount
    };
  }

  /**
   * Clear all cached operations (use with caution)
   */
  clear(): void {
    this.pool.clear();
  }
}
