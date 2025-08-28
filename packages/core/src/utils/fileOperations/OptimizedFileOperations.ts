/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { FileOperationPool } from '../concurrency/FileOperationPool.js';
import { BufferPool } from '../concurrency/BufferPool.js';
import { EnhancedLruCache } from '../cache/EnhancedLruCache.js';
import { FileSystemService } from '../../services/fileSystemService.js';

/**
 * Configuration for file operations
 */
export interface FileOperationsConfig {
  /** Enable caching for file operations */
  enableCache?: boolean;
  /** Cache TTL in milliseconds */
  cacheTTL?: number;
  /** Maximum number of cached files */
  maxCacheSize?: number;
  /** Enable operation pooling */
  enablePooling?: boolean;
  /** Maximum concurrent operations */
  maxConcurrent?: number;
  /** Enable buffer pooling */
  enableBufferPool?: boolean;
}

/**
 * Result of a file operation
 */
export interface FileOperationResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  fromCache?: boolean;
  operationTime?: number;
}

/**
 * Consolidated file operations utility with performance optimizations.
 * Provides caching, pooling, and error handling for file system operations.
 */
export class OptimizedFileOperations {
  private static instance: OptimizedFileOperations;
  private filePool: FileOperationPool;
  private bufferPool: BufferPool;
  private cache: EnhancedLruCache<string, string>;
  private config: Required<FileOperationsConfig>;

  constructor(config: FileOperationsConfig = {}) {
    this.config = {
      enableCache: config.enableCache ?? true,
      cacheTTL: config.cacheTTL ?? 5 * 60 * 1000, // 5 minutes
      maxCacheSize: config.maxCacheSize ?? 500,
      enablePooling: config.enablePooling ?? true,
      maxConcurrent: config.maxConcurrent ?? 10,
      enableBufferPool: config.enableBufferPool ?? true,
    };

    this.filePool = new FileOperationPool(this.config.maxConcurrent);
    this.bufferPool = new BufferPool();
    this.cache = new EnhancedLruCache<string, string>(this.config.maxCacheSize, {
      enableCompression: true,
      enableTTL: true,
      defaultTTL: this.config.cacheTTL,
      trackMemory: true,
    });

    // Pre-fill buffer pool for better performance
    if (this.config.enableBufferPool) {
      this.bufferPool.preFill(10);
    }
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: FileOperationsConfig): OptimizedFileOperations {
    if (!OptimizedFileOperations.instance) {
      OptimizedFileOperations.instance = new OptimizedFileOperations(config);
    }
    return OptimizedFileOperations.instance;
  }

  /**
   * Safely read a file with caching and error handling
   */
  async safeReadFile(filePath: string, useCache = true): Promise<FileOperationResult<string>> {
    const startTime = Date.now();
    const normalizedPath = path.normalize(filePath);

    // Check cache first
    if (this.config.enableCache && useCache) {
      const cached = this.cache.get(normalizedPath);
      if (cached !== undefined) {
        return {
          success: true,
          data: cached,
          fromCache: true,
          operationTime: Date.now() - startTime,
        };
      }
    }

    try {
      const operation = async () => {
        const content = await fs.readFile(normalizedPath, 'utf-8');
        
        // Cache the result
        if (this.config.enableCache && useCache) {
          this.cache.set(normalizedPath, content);
        }
        
        return content;
      };

      const content = this.config.enablePooling
        ? await this.filePool.execute(normalizedPath, operation)
        : await operation();

      return {
        success: true,
        data: content,
        fromCache: false,
        operationTime: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        operationTime: Date.now() - startTime,
      };
    }
  }

  /**
   * Safely write a file with backup creation
   */
  async safeWriteFile(
    filePath: string,
    content: string,
    createBackup = true
  ): Promise<FileOperationResult<void>> {
    const startTime = Date.now();
    const normalizedPath = path.normalize(filePath);

    try {
      const operation = async () => {
        // Create backup if file exists and backup is enabled
        if (createBackup) {
          try {
            await fs.access(normalizedPath);
            const backupPath = `${normalizedPath}.backup`;
            await fs.copyFile(normalizedPath, backupPath);
          } catch {
            // File doesn't exist, no backup needed
          }
        }

        // Ensure directory exists
        const dir = path.dirname(normalizedPath);
        await fs.mkdir(dir, { recursive: true });

        // Write file
        await fs.writeFile(normalizedPath, content, 'utf-8');

        // Update cache
        if (this.config.enableCache) {
          this.cache.set(normalizedPath, content);
        }
      };

      this.config.enablePooling
        ? await this.filePool.execute(`write:${normalizedPath}`, operation)
        : await operation();

      return {
        success: true,
        operationTime: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        operationTime: Date.now() - startTime,
      };
    }
  }

  /**
   * Check if a file exists with caching
   */
  async exists(filePath: string): Promise<boolean> {
    const normalizedPath = path.normalize(filePath);

    try {
      const operation = async () => {
        await fs.access(normalizedPath);
        return true;
      };

      return this.config.enablePooling
        ? await this.filePool.execute(`exists:${normalizedPath}`, operation)
        : await operation();
    } catch {
      return false;
    }
  }

  /**
   * Find project root directory (consolidated implementation)
   */
  async findProjectRoot(
    startDir: string,
    fileSystemService?: FileSystemService
  ): Promise<string | null> {
    const normalizedStart = path.resolve(startDir);
    const cacheKey = `projectRoot:${normalizedStart}`;

    // Check cache first
    if (this.config.enableCache) {
      const cached = this.cache.get(cacheKey);
      if (cached !== undefined) {
        return cached === 'null' ? null : cached;
      }
    }

    try {
      const operation = async () => {
        let currentDir = normalizedStart;

        while (true) {
          const gitPath = path.join(currentDir, '.git');
          
          try {
            if (fileSystemService) {
              const fileInfo = await fileSystemService.getFileInfo(gitPath);
              if (fileInfo.success && fileInfo.data?.isDirectory) {
                return currentDir;
              }
            } else {
              const stats = await fs.lstat(gitPath);
              if (stats.isDirectory()) {
                return currentDir;
              }
            }
          } catch {
            // .git not found, continue
          }

          const parentDir = path.dirname(currentDir);
          if (parentDir === currentDir) {
            // Reached filesystem root
            return null;
          }
          currentDir = parentDir;
        }
      };

      const result = this.config.enablePooling
        ? await this.filePool.execute(cacheKey, operation)
        : await operation();

      // Cache the result
      if (this.config.enableCache) {
        this.cache.set(cacheKey, result ?? 'null');
      }

      return result;
    } catch (error) {
      console.warn('Error finding project root:', error);
      return null;
    }
  }

  /**
   * Read multiple files efficiently
   */
  async readManyFiles(filePaths: string[]): Promise<Map<string, FileOperationResult<string>>> {
    const results = new Map<string, FileOperationResult<string>>();
    
    // Use Promise.allSettled for parallel execution
    const operations = filePaths.map(async (filePath) => {
      const result = await this.safeReadFile(filePath);
      return { filePath, result };
    });

    const settled = await Promise.allSettled(operations);
    
    settled.forEach((outcome) => {
      if (outcome.status === 'fulfilled') {
        results.set(outcome.value.filePath, outcome.value.result);
      } else {
        results.set('unknown', {
          success: false,
          error: outcome.reason instanceof Error ? outcome.reason.message : String(outcome.reason),
        });
      }
    });

    return results;
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    return {
      filePool: this.filePool.getStats(),
      bufferPool: this.bufferPool.getStats(),
      cache: this.cache.getStats(),
      config: this.config,
    };
  }

  /**
   * Clean up caches and pools
   */
  cleanup(): void {
    this.cache.cleanup();
    this.filePool.clear();
    this.bufferPool.clear();
  }

  /**
   * Reset all caches and statistics
   */
  reset(): void {
    this.cache.clear();
    this.filePool.clear();
    this.bufferPool.clear();
  }
}
