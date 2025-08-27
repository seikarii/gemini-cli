/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import { promises as fs } from 'node:fs';
import { Ignore } from './ignore.js';
import * as cache from './crawlCache.js';

export interface CrawlOptions {
  // The directory to start the crawl from.
  crawlDirectory: string;
  // The project's root directory, for path relativity.
  cwd: string;
  // The fdir maxDepth option.
  maxDepth?: number;
  // A pre-configured Ignore instance.
  ignore: Ignore;
  // Caching options.
  cache: boolean;
  cacheTtl: number;
  // Performance options.
  maxConcurrency?: number;
}

/**
 * Performance metrics for crawling operations.
 */
interface CrawlMetrics {
  startTime: number;
  totalDirectories: number;
  totalFiles: number;
  cacheHit: boolean;
  concurrentStats: number;
  ignoredItems: number;
}

/**
 * Represents a file system entry with its metadata.
 */
interface FileSystemEntry {
  name: string;
  fullPath: string;
  posixPath: string;
  relativePath: string;
  isDirectory: boolean;
  isFile: boolean;
  error?: Error;
}

/**
 * Optimized POSIX path converter with memoization for better performance.
 */
const pathConversionCache = new Map<string, string>();

function toPosixPath(p: string): string {
  if (pathConversionCache.has(p)) {
    return pathConversionCache.get(p)!;
  }

  const posixPath = p.split(path.sep).join(path.posix.sep);

  // Keep cache size reasonable
  if (pathConversionCache.size > 1000) {
    pathConversionCache.clear();
  }

  pathConversionCache.set(p, posixPath);
  return posixPath;
}

/**
 * Processes file system entries in parallel for better performance.
 */
async function processEntriesInParallel(
  entries: string[],
  dir: string,
  posixCrawlDirectory: string,
  maxConcurrency: number = 50,
): Promise<FileSystemEntry[]> {
  const results: FileSystemEntry[] = [];

  // Process entries in batches to avoid overwhelming the system
  for (let i = 0; i < entries.length; i += maxConcurrency) {
    const batch = entries.slice(i, i + maxConcurrency);

    const batchPromises = batch.map(async (entry): Promise<FileSystemEntry> => {
      const fullPath = path.join(dir, entry);
      const posixPath = toPosixPath(fullPath);
      const relativePath = path.posix.relative(posixCrawlDirectory, posixPath);

      try {
        const stat = await fs.stat(fullPath);
        return {
          name: entry,
          fullPath,
          posixPath,
          relativePath,
          isDirectory: stat.isDirectory(),
          isFile: stat.isFile(),
        };
      } catch (error) {
        return {
          name: entry,
          fullPath,
          posixPath,
          relativePath,
          isDirectory: false,
          isFile: false,
          error: error as Error,
        };
      }
    });

    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }

  return results;
}

export async function crawl(options: CrawlOptions): Promise<string[]> {
  const metrics: CrawlMetrics = {
    startTime: performance.now(),
    totalDirectories: 0,
    totalFiles: 0,
    cacheHit: false,
    concurrentStats: 0,
    ignoredItems: 0,
  };

  // Check cache first
  if (options.cache) {
    const cacheKey = cache.getCacheKey(
      options.crawlDirectory,
      options.ignore.getFingerprint(),
      options.maxDepth,
    );
    const cachedResults = cache.read(cacheKey);

    if (cachedResults) {
      metrics.cacheHit = true;
      logPerformanceMetrics(metrics, cachedResults.length);
      return cachedResults;
    }
  }

  const posixCwd = toPosixPath(options.cwd);
  const posixCrawlDirectory = toPosixPath(options.crawlDirectory);
  const maxConcurrency = options.maxConcurrency || 50;

  // Optimized crawler with parallel processing
  const dirFilter = options.ignore.getDirectoryFilter();
  const resultsList: string[] = [];

  // Always include the crawl root marker '.' so callers know the root was visited.
  resultsList.push('.');

  async function walkOptimized(dir: string, depth: number): Promise<void> {
    if (options.maxDepth !== undefined && depth > options.maxDepth) return;

    let entries: string[];
    try {
      entries = await fs.readdir(dir);
    } catch (error) {
      // Log error for debugging but don't fail the entire crawl
      if (process.env['NODE_ENV'] === 'development') {
        console.debug(`Failed to read directory ${dir}:`, error);
      }
      return;
    }

    if (entries.length === 0) return;

    // Process entries in parallel for better performance
    const processedEntries = await processEntriesInParallel(
      entries,
      dir,
      posixCrawlDirectory,
      maxConcurrency,
    );

    metrics.concurrentStats += processedEntries.length;

    // Separate directories and files for processing
    const directories: FileSystemEntry[] = [];
    const files: FileSystemEntry[] = [];

    for (const entry of processedEntries) {
      if (entry.error) {
        // Skip entries that couldn't be processed
        continue;
      }

      if (entry.isDirectory) {
        directories.push(entry);
      } else if (entry.isFile) {
        files.push(entry);
      }
    }

    // Process directories
    for (const dirEntry of directories) {
      const dirCheck =
        dirEntry.relativePath === '' ? '' : `${dirEntry.relativePath}/`;

      if (dirFilter(dirCheck)) {
        metrics.ignoredItems++;
        continue;
      }

      metrics.totalDirectories++;

      // Push directory entry with trailing slash, use '.' for the root.
      const dirPath =
        dirEntry.relativePath === '' ? '.' : `${dirEntry.relativePath}/`;
      resultsList.push(dirPath);

      // Recursively process subdirectory
      await walkOptimized(dirEntry.fullPath, depth + 1);
    }

    // Process files
    for (const fileEntry of files) {
      metrics.totalFiles++;
      resultsList.push(fileEntry.relativePath);
    }
  }

  await walkOptimized(options.crawlDirectory, 0);

  // Transform results relative to cwd
  const relativeToCrawlDir = path.posix.relative(posixCwd, posixCrawlDirectory);
  const relativeToCwdResults = resultsList.map((p) =>
    relativeToCrawlDir === '' ? p : path.posix.join(relativeToCrawlDir, p),
  );

  // Cache results
  if (options.cache) {
    const cacheKey = cache.getCacheKey(
      options.crawlDirectory,
      options.ignore.getFingerprint(),
      options.maxDepth,
    );
    cache.write(cacheKey, relativeToCwdResults, options.cacheTtl * 1000);
  }

  logPerformanceMetrics(metrics, relativeToCwdResults.length);
  return relativeToCwdResults;
}

/**
 * Logs performance metrics for crawl operations.
 */
function logPerformanceMetrics(
  metrics: CrawlMetrics,
  resultCount: number,
): void {
  const duration = performance.now() - metrics.startTime;

  // Only log if performance is concerning or in development mode
  if (duration > 1000 || process.env['NODE_ENV'] === 'development') {
    const logData = {
      duration: `${duration.toFixed(2)}ms`,
      resultCount,
      directories: metrics.totalDirectories,
      files: metrics.totalFiles,
      ignored: metrics.ignoredItems,
      cacheHit: metrics.cacheHit,
      concurrentStats: metrics.concurrentStats,
    };

    if (process.env['NODE_ENV'] === 'development') {
      console.debug('[crawler] Performance metrics:', logData);
    } else if (duration > 5000) {
      console.warn('[crawler] Slow crawl operation detected:', logData);
    }
  }
}
