/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { Semaphore } from '../Semaphore.js';

describe('Semaphore', () => {
  let semaphore: Semaphore;

  beforeEach(() => {
    semaphore = new Semaphore(2);
  });

  it('should allow acquisition up to permit limit', async () => {
    expect(semaphore.available()).toBe(2);
    
    await semaphore.acquire();
    expect(semaphore.available()).toBe(1);
    
    await semaphore.acquire();
    expect(semaphore.available()).toBe(0);
  });

  it('should block when no permits available', async () => {
    await semaphore.acquire();
    await semaphore.acquire();
    
    let acquired = false;
    const promise = semaphore.acquire().then(() => {
      acquired = true;
    });
    
    // Should not have acquired immediately
    expect(acquired).toBe(false);
    
    // Release a permit
    semaphore.release();
    
    // Now it should acquire
    await promise;
    expect(acquired).toBe(true);
  });

  it('should track waiting operations', async () => {
    await semaphore.acquire();
    await semaphore.acquire();
    
    expect(semaphore.waitingCount()).toBe(0);
    
    const promise1 = semaphore.acquire();
    expect(semaphore.waitingCount()).toBe(1);
    
    const promise2 = semaphore.acquire();
    expect(semaphore.waitingCount()).toBe(2);
    
    semaphore.release();
    await promise1;
    expect(semaphore.waitingCount()).toBe(1);
    
    semaphore.release();
    await promise2;
    expect(semaphore.waitingCount()).toBe(0);
  });

  it('should maintain proper state after multiple operations', async () => {
    const operations = [];
    
    // Create multiple operations that acquire and release
    for (let i = 0; i < 5; i++) {
      operations.push(
        semaphore.acquire()
          .then(() => new Promise(resolve => setTimeout(resolve, 10)))
          .then(() => semaphore.release())
      );
    }
    
    await Promise.all(operations);
    
    // Should be back to initial state
    expect(semaphore.available()).toBe(2);
    expect(semaphore.waitingCount()).toBe(0);
  });
});
