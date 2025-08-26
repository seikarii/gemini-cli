/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';
import { FileFilteringOptions } from '../config/config.js';

// Performance monitoring and logging infrastructure
const logger = {
  debug: (...args: unknown[]) => console.debug('[DEBUG] [BfsFileSearch]', ...args),
  perf: (...args: unknown[]) => console.debug('[PERF] [BfsFileSearch]', ...args),
};

interface BfsFileSearchOptions {
  fileName: string;
  ignoreDirs?: string[];
  maxDirs?: number;
  debug?: boolean;
  fileService?: FileDiscoveryService;
  fileFilteringOptions?: FileFilteringOptions;
  // Performance tuning options
  maxConcurrency?: number;          // Maximum parallel operations (default: 20)
  batchSize?: number;               // Batch size for ignore checking (default: 50)
  cacheSize?: number;               // Maximum cache entries (default: 1000)
  cacheTtlMs?: number;              // Cache TTL in milliseconds (default: 5 minutes)
}

// Performance metrics tracking
interface SearchMetrics {
  totalDirsScanned: number;
  totalFilesFound: number;
  totalIgnoreChecks: number;
  cacheHits: number;
  cacheMisses: number;
  batchesProcessed: number;
  totalTimeMs: number;
  avgIgnoreCheckMs: number;
}

// Cache entry for ignore decisions
interface IgnoreCacheEntry {
  ignored: boolean;
  timestamp: number;
}

// Optimized ignore result cache with TTL and size limits
class IgnoreCache {
  private cache = new Map<string, IgnoreCacheEntry>();
  private readonly maxSize: number;
  private readonly ttlMs: number;

  constructor(maxSize = 1000, ttlMs = 300000) { // 5 minutes default TTL
    this.maxSize = maxSize;
    this.ttlMs = ttlMs;
  }

  get(path: string): boolean | undefined {
    const entry = this.cache.get(path);
    if (!entry) return undefined;
    
    // Check TTL
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.cache.delete(path);
      return undefined;
    }
    
    return entry.ignored;
  }

  set(path: string, ignored: boolean): void {
    // Clean old entries if cache is full
    if (this.cache.size >= this.maxSize) {
      this.evictOldEntries();
    }
    
    this.cache.set(path, {
      ignored,
      timestamp: Date.now(),
    });
  }

  private evictOldEntries(): void {
    const now = Date.now();
    const toDelete: string[] = [];
    
    // Remove expired entries first
    for (const [path, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.ttlMs) {
        toDelete.push(path);
      }
    }
    
    toDelete.forEach(path => this.cache.delete(path));
    
    // If still too large, remove oldest entries
    if (this.cache.size >= this.maxSize) {
      const entries = Array.from(this.cache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      const toRemove = entries.slice(0, Math.floor(this.maxSize * 0.3));
      toRemove.forEach(([path]) => this.cache.delete(path));
    }
  }

  clear(): void {
    this.cache.clear();
  }

  getStats(): { size: number; maxSize: number } {
    return { size: this.cache.size, maxSize: this.maxSize };
  }
}

// File entry with metadata for batch processing
interface FileEntry {
  path: string;
  name: string;
  isDirectory: boolean;
}

// Batch ignore processor for optimized bulk operations
class BatchIgnoreProcessor {
  private readonly fileService: FileDiscoveryService;
  private readonly fileFilteringOptions: FileFilteringOptions;
  private readonly cache: IgnoreCache;
  private readonly metrics: SearchMetrics;

  constructor(
    fileService: FileDiscoveryService,
    fileFilteringOptions: FileFilteringOptions,
    cache: IgnoreCache,
    metrics: SearchMetrics,
  ) {
    this.fileService = fileService;
    this.fileFilteringOptions = fileFilteringOptions;
    this.cache = cache;
    this.metrics = metrics;
  }

  /**
   * Process a batch of entries, checking ignore patterns efficiently
   */
  processBatch(entries: FileEntry[]): { ignored: Set<string>; processed: FileEntry[] } {
    const ignored = new Set<string>();
    const processed: FileEntry[] = [];
    const uncachedEntries: FileEntry[] = [];

    const batchStart = performance.now();

    // First pass: Check cache for quick hits
    for (const entry of entries) {
      const cached = this.cache.get(entry.path);
      if (cached !== undefined) {
        this.metrics.cacheHits++;
        if (cached) {
          ignored.add(entry.path);
        } else {
          processed.push(entry);
        }
      } else {
        uncachedEntries.push(entry);
        this.metrics.cacheMisses++;
      }
    }

    // Second pass: Batch process uncached entries
    if (uncachedEntries.length > 0) {
      for (const entry of uncachedEntries) {
        const isIgnored = this.fileService.shouldIgnoreFile(entry.path, {
          respectGitIgnore: this.fileFilteringOptions?.respectGitIgnore,
          respectGeminiIgnore: this.fileFilteringOptions?.respectGeminiIgnore,
        });

        this.cache.set(entry.path, isIgnored);
        this.metrics.totalIgnoreChecks++;

        if (isIgnored) {
          ignored.add(entry.path);
        } else {
          processed.push(entry);
        }
      }
    }

    const batchTime = performance.now() - batchStart;
    this.metrics.avgIgnoreCheckMs = (this.metrics.avgIgnoreCheckMs + batchTime) / 2;
    this.metrics.batchesProcessed++;

    return { ignored, processed };
  }

  /**
   * Pre-filter directories to avoid scanning ignored directory trees
   */
  shouldSkipDirectory(dirPath: string): boolean {
    const cached = this.cache.get(dirPath);
    if (cached !== undefined) {
      this.metrics.cacheHits++;
      return cached;
    }

    const isIgnored = this.fileService.shouldIgnoreFile(dirPath, {
      respectGitIgnore: this.fileFilteringOptions?.respectGitIgnore,
      respectGeminiIgnore: this.fileFilteringOptions?.respectGeminiIgnore,
    });

    this.cache.set(dirPath, isIgnored);
    this.metrics.cacheMisses++;
    this.metrics.totalIgnoreChecks++;

    return isIgnored;
  }
}

/**
 * Log comprehensive performance metrics
 */
function logPerformanceMetrics(metrics: SearchMetrics, options: BfsFileSearchOptions): void {
  const {
    totalDirsScanned,
    totalFilesFound,
    totalIgnoreChecks,
    cacheHits,
    cacheMisses,
    batchesProcessed,
    totalTimeMs,
    avgIgnoreCheckMs,
  } = metrics;

  const cacheHitRate = cacheHits + cacheMisses > 0 ? (cacheHits / (cacheHits + cacheMisses) * 100).toFixed(1) : '0.0';
  const dirsPerMs = totalDirsScanned > 0 ? (totalDirsScanned / totalTimeMs).toFixed(2) : '0';
  const ignoreChecksPerMs = totalIgnoreChecks > 0 ? (totalIgnoreChecks / totalTimeMs).toFixed(2) : '0';

  if (options.debug) {
    logger.perf(`=== BFS Search Performance Metrics ===`);
    logger.perf(`Total time: ${totalTimeMs.toFixed(2)}ms`);
    logger.perf(`Directories scanned: ${totalDirsScanned} (${dirsPerMs}/ms)`);
    logger.perf(`Files found: ${totalFilesFound}`);
    logger.perf(`Ignore checks: ${totalIgnoreChecks} (${ignoreChecksPerMs}/ms)`);
    logger.perf(`Cache performance: ${cacheHitRate}% hit rate (${cacheHits} hits, ${cacheMisses} misses)`);
    logger.perf(`Batches processed: ${batchesProcessed}`);
    logger.perf(`Avg ignore check time: ${avgIgnoreCheckMs.toFixed(3)}ms`);
    logger.perf(`=======================================`);
  } else {
    // Always log basic performance summary
    console.log(`[BFS Search] ${totalFilesFound} files found in ${totalTimeMs.toFixed(2)}ms across ${totalDirsScanned} directories`);
    if (totalIgnoreChecks > 0) {
      console.log(`[BFS Search] Cache efficiency: ${cacheHitRate}% hit rate (${totalIgnoreChecks} ignore checks)`);
    }
  }
}

/**
 * Performs a breadth-first search for a specific file within a directory structure.
 * Optimized with batch processing, smart caching, and performance monitoring.
 *
 * @param rootDir The directory to start the search from.
 * @param options Configuration for the search.
 * @returns A promise that resolves to an array of paths where the file was found.
 */
export async function bfsFileSearch(
  rootDir: string,
  options: BfsFileSearchOptions,
): Promise<string[]> {
  const {
    fileName,
    ignoreDirs = [],
    maxDirs = Infinity,
    debug = false,
    fileService,
    maxConcurrency = 20,
    batchSize = 50,
    cacheSize = 1000,
    cacheTtlMs = 300000, // 5 minutes
  } = options;

  const searchStart = performance.now();
  const foundFiles: string[] = [];
  const queue: string[] = [rootDir];
  const visited = new Set<string>();
  let queueHead = 0;

  // Performance tracking
  const metrics: SearchMetrics = {
    totalDirsScanned: 0,
    totalFilesFound: 0,
    totalIgnoreChecks: 0,
    cacheHits: 0,
    cacheMisses: 0,
    batchesProcessed: 0,
    totalTimeMs: 0,
    avgIgnoreCheckMs: 0,
  };

  // Initialize optimized components
  const ignoreDirsSet = new Set(ignoreDirs);
  const cache = new IgnoreCache(cacheSize, cacheTtlMs);
  let batchProcessor: BatchIgnoreProcessor | undefined;

  if (fileService) {
    batchProcessor = new BatchIgnoreProcessor(
      fileService,
      options.fileFilteringOptions || { respectGitIgnore: true, respectGeminiIgnore: true },
      cache,
      metrics,
    );
  }

  // Adaptive batch processing with configurable concurrency
  while (queueHead < queue.length && metrics.totalDirsScanned < maxDirs) {
    const currentBatchSize = Math.min(
      maxConcurrency,
      maxDirs - metrics.totalDirsScanned,
      queue.length - queueHead,
    );

    const currentBatch: string[] = [];
    while (currentBatch.length < currentBatchSize && queueHead < queue.length) {
      const currentDir = queue[queueHead];
      queueHead++;
      
      if (!visited.has(currentDir)) {
        visited.add(currentDir);
        currentBatch.push(currentDir);
      }
    }

    if (currentBatch.length === 0) continue;

    metrics.totalDirsScanned += currentBatch.length;

    if (debug) {
      logger.debug(
        `Scanning batch [${metrics.totalDirsScanned}/${maxDirs}]: ${currentBatch.length} directories`,
      );
    }

    // Process directories in parallel with optimized error handling
    const readPromises = currentBatch.map(async (currentDir) => {
      try {
        // Pre-filter directories if we have a file service
        if (batchProcessor?.shouldSkipDirectory(currentDir)) {
          if (debug) {
            logger.debug(`Skipping ignored directory: ${currentDir}`);
          }
          return { currentDir, entries: [], skipped: true };
        }

        const entries = await fs.readdir(currentDir, { withFileTypes: true });
        return { currentDir, entries, skipped: false };
      } catch (error) {
        const message = (error as Error)?.message ?? 'Unknown error';
        console.warn(
          `[WARN] Skipping unreadable directory: ${currentDir} (${message})`,
        );
        if (debug) {
          logger.debug(`Full error for ${currentDir}:`, error);
        }
        return { currentDir, entries: [], skipped: false };
      }
    });

    const results = await Promise.all(readPromises);

    // Process results with batch ignore checking
    for (const { currentDir, entries, skipped } of results) {
      if (skipped || entries.length === 0) continue;

      // Convert to FileEntry objects for batch processing
      const fileEntries: FileEntry[] = entries.map(entry => ({
        path: path.join(currentDir, entry.name),
        name: entry.name,
        isDirectory: entry.isDirectory(),
      }));

      // Batch process ignore checks for performance
      let processedEntries = fileEntries;
      if (batchProcessor && fileEntries.length > 0) {
        const batches: FileEntry[][] = [];
        for (let i = 0; i < fileEntries.length; i += batchSize) {
          batches.push(fileEntries.slice(i, i + batchSize));
        }

        processedEntries = [];
        for (const batch of batches) {
          const { processed } = batchProcessor.processBatch(batch);
          processedEntries.push(...processed);
        }
      }

      // Process final results
      for (const entry of processedEntries) {
        if (entry.isDirectory) {
          if (!ignoreDirsSet.has(entry.name)) {
            queue.push(entry.path);
          }
        } else if (entry.name === fileName) {
          foundFiles.push(entry.path);
          metrics.totalFilesFound++;
          
          if (debug) {
            logger.debug(`Found target file: ${entry.path}`);
          }
        }
      }
    }
  }

  // Calculate final metrics
  metrics.totalTimeMs = performance.now() - searchStart;
  
  // Log performance metrics
  logPerformanceMetrics(metrics, options);

  if (debug) {
    const cacheStats = cache.getStats();
    logger.debug(`Cache utilization: ${cacheStats.size}/${cacheStats.maxSize} entries`);
  }

  return foundFiles;
}
