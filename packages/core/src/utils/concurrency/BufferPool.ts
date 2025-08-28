/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pool for managing Buffer objects to reduce memory allocations and garbage collection.
 * Reuses buffers to improve performance in file operations and data processing.
 */
export class BufferPool {
  private buffers: Buffer[] = [];
  private readonly maxSize: number;
  private readonly defaultBufferSize: number;
  private allocated = 0;
  private reused = 0;

  constructor(maxSize: number = 100, defaultBufferSize: number = 64 * 1024) {
    this.maxSize = maxSize;
    this.defaultBufferSize = defaultBufferSize;
  }

  /**
   * Acquire a buffer of at least the specified size.
   * Returns a reused buffer if available, otherwise allocates a new one.
   */
  acquire(size: number = this.defaultBufferSize): Buffer {
    // Look for a suitable buffer in the pool
    const bufferIndex = this.buffers.findIndex(buffer => buffer.length >= size);
    
    if (bufferIndex !== -1) {
      // Remove and return the found buffer
      const buffer = this.buffers.splice(bufferIndex, 1)[0];
      this.reused++;
      
      // Clear the buffer for security
      buffer.fill(0);
      return buffer;
    }

    // No suitable buffer found, allocate a new one
    this.allocated++;
    return Buffer.alloc(Math.max(size, this.defaultBufferSize));
  }

  /**
   * Return a buffer to the pool for reuse.
   * Only stores the buffer if the pool isn't full.
   */
  release(buffer: Buffer): void {
    if (this.buffers.length < this.maxSize) {
      // Clear the buffer for security before storing
      buffer.fill(0);
      this.buffers.push(buffer);
    }
  }

  /**
   * Pre-fill the pool with buffers of the default size.
   * Useful for warming up the pool before heavy operations.
   */
  preFill(count: number = Math.min(10, this.maxSize)): void {
    for (let i = 0; i < count && this.buffers.length < this.maxSize; i++) {
      this.buffers.push(Buffer.alloc(this.defaultBufferSize));
    }
  }

  /**
   * Clear all buffers from the pool.
   */
  clear(): void {
    this.buffers.length = 0;
  }

  /**
   * Get pool statistics for monitoring and debugging.
   */
  getStats() {
    return {
      poolSize: this.buffers.length,
      maxSize: this.maxSize,
      totalAllocated: this.allocated,
      totalReused: this.reused,
      reuseRate: this.allocated > 0 ? (this.reused / (this.allocated + this.reused)) * 100 : 0,
      memoryUsed: this.buffers.reduce((total, buf) => total + buf.length, 0)
    };
  }

  /**
   * Execute an operation with a buffer from the pool.
   * Automatically acquires and releases the buffer.
   */
  async withBuffer<T>(
    size: number,
    operation: (buffer: Buffer) => Promise<T>
  ): Promise<T> {
    const buffer = this.acquire(size);
    try {
      return await operation(buffer);
    } finally {
      this.release(buffer);
    }
  }
}
