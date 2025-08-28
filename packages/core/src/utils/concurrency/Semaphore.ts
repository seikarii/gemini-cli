/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * A semaphore implementation for controlling concurrency in async operations.
 * Prevents overwhelming the system with too many concurrent file operations.
 */
export class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  /**
   * Acquire a permit. If none available, wait until one is released.
   */
  async acquire(): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.permits > 0) {
        this.permits--;
        resolve();
      } else {
        this.waiting.push(resolve);
      }
    });
  }

  /**
   * Release a permit and wake up waiting operations.
   */
  release(): void {
    this.permits++;
    const next = this.waiting.shift();
    if (next) {
      this.permits--;
      next();
    }
  }

  /**
   * Get current available permits
   */
  available(): number {
    return this.permits;
  }

  /**
   * Get number of waiting operations
   */
  waitingCount(): number {
    return this.waiting.length;
  }
}
