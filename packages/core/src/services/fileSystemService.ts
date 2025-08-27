/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'fs/promises';
import path from 'path';

/**
 * Performance mode for file system operations
 */
export type PerformanceMode = 'safe' | 'balanced' | 'fast';

/**
 * Configuration options for file system operations
 */
export interface FileSystemOptions {
  /** Encoding for text files (default: 'utf-8') */
  encoding?: BufferEncoding;
  /** Timeout for operations in milliseconds (default: 30000) */
  timeout?: number;
  /** Whether to create parent directories automatically (default: true) */
  createDirectories?: boolean;
  /** Maximum file size in bytes for safety checks (default: 100MB) */
  maxFileSize?: number;
  /** Whether to perform atomic writes (default: true) */
  atomicWrites?: boolean;
  /** Whether to create a local checkpoint (backup) before overwriting an existing file (default: true) */
  createCheckpoint?: boolean;
  /** Whether to bypass cache for this operation (default: false) */
  bypassCache?: boolean;
  /** Maximum cache size in entries (default: 1000) */
  maxCacheEntries?: number;
  /** Performance mode affecting operation behavior (default: 'balanced') */
  performanceMode?: PerformanceMode;
  /** Enable performance monitoring and metrics collection (default: false) */
  enableMetrics?: boolean;
  /** Batch size for bulk operations (default: 50) */
  batchSize?: number;
  /** Custom path safety configuration */
  pathSafetyConfig?: PathSafetyConfig;
}

/**
 * Configuration for path safety checks
 */
export interface PathSafetyConfig {
  /** Whether to allow path traversal (..) - use with caution (default: false) */
  allowPathTraversal?: boolean;
  /** Whether to allow tilde expansion (default: false) */
  allowTildeExpansion?: boolean;
  /** Additional allowed characters in paths (default: none) */
  allowedChars?: string;
  /** Whether to enable strict path validation (default: true) */
  strictValidation?: boolean;
  /** Custom allowed roots for path validation */
  allowedRoots?: string[];
}

/**
 * File system operation result with metadata
 */
export interface FileOperationResult<T = void> {
  success: boolean;
  data?: T;
  error?: string;
  errorCode?: string; // ENOENT, EACCES, etc.
  metadata?: {
    filePath?: string;
    fileSize?: number;
    operationTime?: number;
    encoding?: string;
    cacheHit?: boolean;
    mtimeMs?: number;
  };
}

/**
 * File information structure
 */
export interface FileInfo {
  path: string;
  size: number;
  isFile: boolean;
  isDirectory: boolean;
  exists: boolean;
  permissions: {
    readable: boolean;
    writable: boolean;
    executable: boolean;
  };
  modified: Date;
  created: Date;
  mtimeMs: number; // High precision mtime for cache validation
}

/**
 * Cache entry for file content with validation metadata
 */
interface FileCacheEntry {
  content: string;
  mtimeMs: number;
  encoding: BufferEncoding;
  cachedAt: number;
  accessCount: number;
  lastAccessed: number;
}

/**
 * Path safety cache entry for performance optimization
 */
interface PathSafetyCacheEntry {
  isSafe: boolean;
  cachedAt: number;
  allowedRootsHash: string;
}

/**
 * Performance metrics for file system operations
 */
export interface FileSystemMetrics {
  operationCounts: Map<string, number>;
  operationTimes: Map<string, number[]>;
  cacheHits: number;
  cacheMisses: number;
  pathSafetyCacheHits: number;
  pathSafetyCacheMisses: number;
  totalOperations: number;
  averageOperationTime: number;
}

/**
 * Batch operation result for bulk file operations
 */
export interface BatchOperationResult<T = void> {
  successful: Array<{ path: string; result: FileOperationResult<T> }>;
  failed: Array<{ path: string; error: string; errorCode?: string }>;
  totalProcessed: number;
  totalTime: number;
  averageTimePerOperation: number;
}

/**
 * Enhanced file system service interface following Crisalida patterns
 */
export interface FileSystemService {
  readTextFile(
    filePath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult<string>>;
  writeTextFile(
    filePath: string,
    content: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult>;
  appendTextFile(
    filePath: string,
    content: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult>;
  readBinaryFile(
    filePath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult<Buffer>>;
  writeBinaryFile(
    filePath: string,
    data: Buffer,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult>;
  exists(filePath: string): Promise<boolean>;
  getFileInfo(filePath: string): Promise<FileOperationResult<FileInfo>>;
  createDirectory(
    dirPath: string,
    options?: { recursive?: boolean },
  ): Promise<FileOperationResult>;
  deleteFile(filePath: string): Promise<FileOperationResult>;
  deleteDirectory(
    dirPath: string,
    options?: { recursive?: boolean },
  ): Promise<FileOperationResult>;
  copyFile(
    sourcePath: string,
    destPath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult>;
  moveFile(
    sourcePath: string,
    destPath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult>;
  listDirectory(dirPath: string): Promise<FileOperationResult<string[]>>;
  listDirectoryRecursive(
    dirPath: string,
    options?: { maxDepth?: number; includeDirectories?: boolean },
  ): Promise<FileOperationResult<string[]>>;
  isPathSafe(filePath: string, allowedRoots?: string[]): boolean;
  clearCache(filePath?: string): void;
  getCacheStats(): { size: number; hitRate: number; totalRequests: number };
  
  // Performance optimization methods
  batchReadTextFiles(
    filePaths: string[],
    options?: FileSystemOptions,
  ): Promise<BatchOperationResult<string>>;
  batchWriteTextFiles(
    operations: Array<{ path: string; content: string }>,
    options?: FileSystemOptions,
  ): Promise<BatchOperationResult>;
  getPerformanceMetrics(): FileSystemMetrics;
  resetPerformanceMetrics(): void;
  optimizeCache(): void;
}

/**
 * Robust file system service implementation with self-validating cache,
 * comprehensive error handling, and defensive programming patterns.
 *
 * Features:
 * - Self-validating in-memory cache with mtime verification
 * - Path traversal protection
 * - Atomic write operations with cache invalidation
 * - File size limits and timeouts
 * - Comprehensive error handling with structured results and error codes
 * - Permission checking and validation
 * - Graceful degradation patterns following Crisalida conventions
 * - Recursive directory operations
 * - Cache statistics and management
 */
export class StandardFileSystemService implements FileSystemService {
  private readonly defaultOptions: Required<
    Omit<FileSystemOptions, 'bypassCache' | 'maxCacheEntries'>
  > = {
    encoding: 'utf-8',
    timeout: 30000,
    createDirectories: true,
    maxFileSize: 100 * 1024 * 1024, // 100MB
    atomicWrites: true,
    createCheckpoint: true,
    performanceMode: 'balanced',
    enableMetrics: false,
    batchSize: 50,
    pathSafetyConfig: {
      allowPathTraversal: false,
      allowTildeExpansion: false,
      allowedChars: '',
      strictValidation: true,
      allowedRoots: [],
    },
  };

  // Self-validating cache implementation
  private readonly fileCache = new Map<string, FileCacheEntry>();
  private readonly maxCacheEntries: number = 1000;
  private cacheStats = {
    hits: 0,
    misses: 0,
    invalidations: 0,
    totalRequests: 0,
  };

  // Path safety cache for performance optimization
  private readonly pathSafetyCache = new Map<string, PathSafetyCacheEntry>();
  private readonly pathSafetyCacheSize = 500;
  private readonly pathSafetyCacheTTL = 300000; // 5 minutes

  // Performance metrics
  private performanceMetrics: FileSystemMetrics = {
    operationCounts: new Map(),
    operationTimes: new Map(),
    cacheHits: 0,
    cacheMisses: 0,
    pathSafetyCacheHits: 0,
    pathSafetyCacheMisses: 0,
    totalOperations: 0,
    averageOperationTime: 0,
  };

  /**
   * Record performance metrics for operations
   */
  private recordMetric(operation: string, startTime: number, options?: FileSystemOptions): void {
    if (!options?.enableMetrics) return;

    const duration = Date.now() - startTime;
    this.performanceMetrics.totalOperations++;
    
    // Update operation count
    const currentCount = this.performanceMetrics.operationCounts.get(operation) || 0;
    this.performanceMetrics.operationCounts.set(operation, currentCount + 1);
    
    // Update operation times
    const currentTimes = this.performanceMetrics.operationTimes.get(operation) || [];
    currentTimes.push(duration);
    this.performanceMetrics.operationTimes.set(operation, currentTimes);
    
    // Update average
    const totalTime = Array.from(this.performanceMetrics.operationTimes.values())
      .flat()
      .reduce((sum, time) => sum + time, 0);
    this.performanceMetrics.averageOperationTime = totalTime / this.performanceMetrics.totalOperations;
  }

  /**
   * Create hash of allowed roots for cache key
   */
  private createAllowedRootsHash(allowedRoots: string[]): string {
    return allowedRoots.sort().join('|');
  }

  /**
   * Optimized atomic write with conditional checkpointing based on performance mode
   */
  private async performOptimizedWrite(
    filePath: string,
    content: string,
    opts: Required<Omit<FileSystemOptions, 'bypassCache' | 'maxCacheEntries'>>,
  ): Promise<void> {
    const shouldUseAtomicWrites = opts.performanceMode === 'safe' ? true : opts.atomicWrites;
    const shouldCreateCheckpoint = opts.performanceMode === 'fast' ? false : opts.createCheckpoint;

    if (shouldUseAtomicWrites) {
      // Optimized checkpoint creation - only for 'safe' and 'balanced' modes
      if (shouldCreateCheckpoint) {
        await this.createOptimizedCheckpoint(filePath, opts);
      }

      // Atomic write with optimized temp file naming
      const tempPath = `${filePath}.tmp.${Date.now()}.${Math.random().toString(36).substr(2, 9)}`;

      try {
        await fs.writeFile(tempPath, content, opts.encoding);
        await fs.rename(tempPath, filePath);
      } catch (error) {
        // Cleanup temp file on error
        try {
          await fs.unlink(tempPath);
        } catch {
          /* intentional */
        }
        throw error;
      }
    } else {
      // Direct write for 'fast' mode
      await fs.writeFile(filePath, content, opts.encoding);
    }
  }

  /**
   * Optimized checkpoint creation with better error handling
   */
  private async createOptimizedCheckpoint(filePath: string, opts: Required<Omit<FileSystemOptions, 'bypassCache' | 'maxCacheEntries'>>): Promise<void> {
    try {
      const stat = await fs.stat(filePath).catch(() => null);
      if (stat && stat.isFile()) {
        // Use cached content if available and valid
        let existing: string | null = null;
        
        if (opts.performanceMode !== 'fast') {
          try {
            const cacheResult = await this.getCachedOrFreshContent(filePath, opts.encoding, false);
            existing = cacheResult.content;
          } catch {
            existing = await fs.readFile(filePath, opts.encoding).catch(() => null);
          }
        } else {
          existing = await fs.readFile(filePath, opts.encoding).catch(() => null);
        }

        if (existing !== null) {
          const checkpointDir = path.join(
            path.dirname(filePath),
            '.gemini_checkpoints',
            `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          );
          
          await fs.mkdir(checkpointDir, { recursive: true }).catch(() => null);
          const checkpointPath = path.join(checkpointDir, path.basename(filePath));
          
          try {
            await fs.writeFile(checkpointPath, existing, opts.encoding);
          } catch {
            // Don't block the main write if checkpoint creation fails
          }
        }
      }
    } catch {
      /* ignore checkpoint errors */
    }
  }

  /**
   * Optimized path safety validation with dynamic performance mode
   */
  isPathSafe(filePath: string, allowedRoots: string[] = []): boolean {
    // Get path safety configuration from options or use defaults
    const pathSafetyConfig = this.defaultOptions.pathSafetyConfig;

    // Dynamic performance mode - could be configured per instance
    const performanceMode: PerformanceMode = this.defaultOptions.performanceMode;

    if (performanceMode === 'fast') {
      // Basic check only for fast mode - respect configuration
      const hasTraversal = !pathSafetyConfig.allowPathTraversal && filePath.includes('..');
      const hasTilde = !pathSafetyConfig.allowTildeExpansion && filePath.includes('~');
      return !hasTraversal && !hasTilde;
    }

    // Create cache key
    const allowedRootsHash = this.createAllowedRootsHash(allowedRoots);
    const cacheKey = `${filePath}:${allowedRootsHash}`;
    const now = Date.now();

    // Check cache first
    const cached = this.pathSafetyCache.get(cacheKey);
    if (cached && (now - cached.cachedAt) < this.pathSafetyCacheTTL) {
      this.performanceMetrics.pathSafetyCacheHits++;
      return cached.isSafe;
    }

    this.performanceMetrics.pathSafetyCacheMisses++;

    try {
      // Normalize and resolve the path
      const normalizedPath = path.resolve(filePath);

      // Check for path traversal attempts (configurable)
      if (!pathSafetyConfig.allowPathTraversal && filePath.includes('..')) {
        this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
        return false;
      }

      // Check for tilde expansion (configurable)
      if (!pathSafetyConfig.allowTildeExpansion && filePath.includes('~')) {
        this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
        return false;
      }

      // Check for suspicious characters (configurable)
      if (pathSafetyConfig.strictValidation) {
        const suspiciousChars = /[<>:"|?*]/;
        const additionalChars = pathSafetyConfig.allowedChars || '';
        const allowedChars = new RegExp(`[<>:"|?*${additionalChars.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]`, 'g');

        if (suspiciousChars.test(filePath) && !allowedChars.test(filePath)) {
          this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
          return false;
        }
      }

      // Validate against allowed roots (merge configured roots with parameter roots)
      const allAllowedRoots = [...(pathSafetyConfig.allowedRoots || []), ...allowedRoots];
      if (allAllowedRoots.length > 0) {
        const isInAllowedRoot = allAllowedRoots.some((root) => {
          const normalizedRoot = path.resolve(root);
          return normalizedPath.startsWith(normalizedRoot);
        });
        if (!isInAllowedRoot) {
          this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
          return false;
        }
      }

      // Prevent access to system directories (skip in fast mode)
      if (performanceMode === 'safe') {
        const systemDirs = ['/etc', '/proc', '/sys', '/dev', '/root'];
        const isSystemDir = systemDirs.some((dir) =>
          normalizedPath.startsWith(dir),
        );
        if (isSystemDir && process.platform !== 'win32') {
          this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
          return false;
        }
      }

      this.cachePathSafetyResult(cacheKey, true, allowedRootsHash, now);
      return true;
    } catch {
      this.cachePathSafetyResult(cacheKey, false, allowedRootsHash, now);
      return false;
    }
  }

  /**
   * Cache path safety result with LRU eviction
   */
  private cachePathSafetyResult(
    cacheKey: string,
    isSafe: boolean,
    allowedRootsHash: string,
    timestamp: number,
  ): void {
    // Evict old entries if cache is full
    if (this.pathSafetyCache.size >= this.pathSafetyCacheSize) {
      this.evictOldPathSafetyEntries();
    }

    this.pathSafetyCache.set(cacheKey, {
      isSafe,
      cachedAt: timestamp,
      allowedRootsHash,
    });
  }

  /**
   * Evict old path safety cache entries
   */
  private evictOldPathSafetyEntries(): void {
    const now = Date.now();
    const toDelete: string[] = [];

    // Remove expired entries first
    for (const [key, entry] of this.pathSafetyCache.entries()) {
      if (now - entry.cachedAt > this.pathSafetyCacheTTL) {
        toDelete.push(key);
      }
    }

    toDelete.forEach(key => this.pathSafetyCache.delete(key));

    // If still too full, remove oldest entries
    if (this.pathSafetyCache.size >= this.pathSafetyCacheSize) {
      const entries = Array.from(this.pathSafetyCache.entries())
        .sort((a, b) => a[1].cachedAt - b[1].cachedAt);
      
      const toRemove = entries.slice(0, Math.floor(this.pathSafetyCacheSize * 0.3));
      toRemove.forEach(([key]) => this.pathSafetyCache.delete(key));
    }
  }

  /**
   * Execute operation with timeout protection and configurable behavior
   */
  private async withTimeout<T>(
    operation: Promise<T>,
    timeoutMs: number,
    operationName: string,
  ): Promise<T> {
    // Allow infinite timeout if set to 0 or negative
    if (timeoutMs <= 0) {
      return operation;
    }

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`${operationName} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });

    return Promise.race([operation, timeoutPromise]);
  }

  /**
   * Validate file size against limits with enhanced configurability
   */
  private async validateFileSize(
    filePath: string,
    maxSize: number,
  ): Promise<void> {
    // Allow unlimited file size if maxSize is 0 or negative
    if (maxSize <= 0) {
      return;
    }

    try {
      const stats = await fs.stat(filePath);
      if (stats.size > maxSize) {
        throw new Error(
          `File size ${stats.size} bytes exceeds maximum allowed ${maxSize} bytes for file: ${filePath}`,
        );
      }
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      if (nodeError.code !== 'ENOENT') {
        throw error;
      }
      // File doesn't exist, which is fine for write operations
    }
  }

  /**
   * Ensure directory exists, following Crisalida defensive patterns
   */
  private async ensureDirectory(dirPath: string): Promise<void> {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      if (nodeError.code !== 'EEXIST') {
        throw new Error(
          `Failed to create directory ${dirPath}: ${nodeError.message}`,
        );
      }
    }
  }

  /**
   * Create structured operation result with error code extraction
   */
  private createResult<T = void>(
    success: boolean,
    data?: T,
    error?: string,
    errorCode?: string,
    metadata?: FileOperationResult<T>['metadata'],
  ): FileOperationResult<T> {
    return {
      success,
      data,
      error,
      errorCode,
      metadata,
    };
  }

  /**
   * Extract error code from NodeJS error for structured reporting
   */
  private extractErrorCode(error: unknown): string {
    if (error && typeof error === 'object' && 'code' in error) {
      return (error as NodeJS.ErrnoException).code || 'UNKNOWN';
    }
    return 'UNKNOWN';
  }

  /**
   * Get high-precision mtime for cache validation
   */
  private async getFileMtimeMs(filePath: string): Promise<number> {
    const stats = await fs.stat(filePath);
    return stats.mtimeMs;
  }

  /**
   * Check if cached content is still valid by comparing mtime
   */
  private async isCacheValid(
    filePath: string,
    cacheEntry: FileCacheEntry,
  ): Promise<boolean> {
    try {
      const currentMtimeMs = await this.getFileMtimeMs(filePath);
      return currentMtimeMs === cacheEntry.mtimeMs;
  } catch (_error) {
      // If we can't stat the file, cache is invalid
      return false;
    }
  }

  /**
   * Manage cache size by evicting least recently used entries
   */
  private evictLeastRecentlyUsed(): void {
    if (this.fileCache.size <= this.maxCacheEntries) {
      return;
    }

    // Find the entry with the oldest lastAccessed time
    let oldestEntry: [string, FileCacheEntry] | null = null;
    let oldestTime = Number.MAX_SAFE_INTEGER;

    for (const [key, entry] of this.fileCache.entries()) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldestEntry = [key, entry];
      }
    }

    if (oldestEntry) {
      this.fileCache.delete(oldestEntry[0]);
    }
  }

  /**
   * Invalidate cache entry for a specific file
   */
  private invalidateCache(filePath: string): void {
    const normalizedPath = path.resolve(filePath);
    if (this.fileCache.delete(normalizedPath)) {
      this.cacheStats.invalidations++;
    }
  }

  /**
   * Get content from cache or disk with self-validation
   */
  private async getCachedOrFreshContent(
    filePath: string,
    encoding: BufferEncoding,
    bypassCache: boolean = false,
  ): Promise<{ content: string; cacheHit: boolean; mtimeMs: number }> {
    const normalizedPath = path.resolve(filePath);
    this.cacheStats.totalRequests++;

    // Check cache if not bypassing
    if (!bypassCache && this.fileCache.has(normalizedPath)) {
      const cacheEntry = this.fileCache.get(normalizedPath)!;

      // Validate cache entry encoding matches request
      if (cacheEntry.encoding === encoding) {
        const isValid = await this.isCacheValid(normalizedPath, cacheEntry);

        if (isValid) {
          // Update access statistics
          cacheEntry.accessCount++;
          cacheEntry.lastAccessed = Date.now();
          this.cacheStats.hits++;

          return {
            content: cacheEntry.content,
            cacheHit: true,
            mtimeMs: cacheEntry.mtimeMs,
          };
        } else {
          // Cache is stale, remove it
          this.fileCache.delete(normalizedPath);
        }
      }
    }

    // Cache miss or bypass - read from disk
    this.cacheStats.misses++;
    const content = await fs.readFile(normalizedPath, encoding);
    const mtimeMs = await this.getFileMtimeMs(normalizedPath);

    // Update cache (if not bypassing and file is cacheable)
    if (!bypassCache) {
      this.evictLeastRecentlyUsed();

      const now = Date.now();
      this.fileCache.set(normalizedPath, {
        content,
        mtimeMs,
        encoding,
        cachedAt: now,
        accessCount: 1,
        lastAccessed: now,
      });
    }

    return {
      content,
      cacheHit: false,
      mtimeMs,
    };
  }

  async exists(filePath: string): Promise<boolean> {
    if (!this.isPathSafe(filePath)) {
      return false;
    }

    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async getFileInfo(filePath: string): Promise<FileOperationResult<FileInfo>> {
    const startTime = Date.now();

    if (!this.isPathSafe(filePath)) {
      return this.createResult<FileInfo>(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      const stats = await fs.stat(filePath);

      // Check permissions
      const permissions = {
        readable: false,
        writable: false,
        executable: false,
      };

      try {
        await fs.access(filePath, fs.constants.R_OK);
        permissions.readable = true;
      } catch {
        /* intentional */
      }

      try {
        await fs.access(filePath, fs.constants.W_OK);
        permissions.writable = true;
      } catch {
        /* intentional */
      }

      try {
        await fs.access(filePath, fs.constants.X_OK);
        permissions.executable = true;
      } catch {
        /* intentional */
      }

      const fileInfo: FileInfo = {
        path: filePath,
        size: stats.size,
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory(),
        exists: true,
        permissions,
        modified: stats.mtime,
        created: stats.birthtime,
        mtimeMs: stats.mtimeMs,
      };

      return this.createResult(true, fileInfo, undefined, undefined, {
        filePath,
        fileSize: stats.size,
        operationTime: Date.now() - startTime,
        mtimeMs: stats.mtimeMs,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      if (nodeError.code === 'ENOENT') {
        const fileInfo: FileInfo = {
          path: filePath,
          size: 0,
          isFile: false,
          isDirectory: false,
          exists: false,
          permissions: { readable: false, writable: false, executable: false },
          modified: new Date(0),
          created: new Date(0),
          mtimeMs: 0,
        };
        return this.createResult(true, fileInfo, undefined, undefined, {
          operationTime: Date.now() - startTime,
        });
      }

      return this.createResult<FileInfo>(
        false,
        undefined,
        `Failed to get file info: ${nodeError.message}`,
        errorCode,
      );
    }
  }

  async readTextFile(
    filePath: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult<string>> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(filePath)) {
      return this.createResult<string>(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      // Validate file size before reading
      await this.validateFileSize(filePath, opts.maxFileSize);

      // Use self-validating cache
      const operation = this.getCachedOrFreshContent(
        filePath,
        opts.encoding,
        options.bypassCache,
      );
      const result = await this.withTimeout(
        operation,
        opts.timeout,
        'readTextFile',
      );

      return this.createResult(true, result.content, undefined, undefined, {
        filePath,
        fileSize: Buffer.byteLength(result.content, opts.encoding),
        operationTime: Date.now() - startTime,
        encoding: opts.encoding,
        cacheHit: result.cacheHit,
        mtimeMs: result.mtimeMs,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);
      let errorMessage = `Failed to read file: ${nodeError.message}`;

      // Provide more specific error messages
      switch (nodeError.code) {
        case 'ENOENT':
          errorMessage = `File not found: ${filePath}`;
          break;
        case 'EACCES':
          errorMessage = `Permission denied reading file: ${filePath}`;
          break;
        case 'EISDIR':
          errorMessage = `Path is a directory, not a file: ${filePath}`;
          break;
        case 'EMFILE':
        case 'ENFILE':
          errorMessage = 'Too many open files, retry later';
          break;
        default:
          break;
      }

      return this.createResult<string>(
        false,
        undefined,
        errorMessage,
        errorCode,
        {
          filePath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async writeTextFile(
    filePath: string,
    content: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(filePath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      // Validate content size
      const contentSize = Buffer.byteLength(content, opts.encoding);
      if (contentSize > opts.maxFileSize) {
        return this.createResult(
          false,
          undefined,
          `Content size ${contentSize} bytes exceeds maximum allowed ${opts.maxFileSize} bytes`,
          'FILE_TOO_LARGE',
        );
      }

      // Create parent directories if needed
      if (opts.createDirectories) {
        const dirPath = path.dirname(filePath);
        await this.ensureDirectory(dirPath);
      }

      if (opts.atomicWrites) {
          // Use optimized write method with conditional checkpointing
          await this.performOptimizedWrite(filePath, content, opts);
      } else {
        // Direct write
        const operation = fs.writeFile(filePath, content, opts.encoding);
        await this.withTimeout(operation, opts.timeout, 'writeTextFile');
      }

      // Explicitly invalidate cache after successful write
      this.invalidateCache(filePath);

      // Record performance metrics
      this.recordMetric('writeTextFile', startTime, options);

      return this.createResult(true, undefined, undefined, undefined, {
        filePath,
        fileSize: contentSize,
        operationTime: Date.now() - startTime,
        encoding: opts.encoding,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);
      let errorMessage = `Failed to write file: ${nodeError.message}`;

      switch (nodeError.code) {
        case 'EACCES':
          errorMessage = `Permission denied writing to file: ${filePath}`;
          break;
        case 'ENOSPC':
          errorMessage = 'No space left on device';
          break;
        case 'ENOTDIR':
          errorMessage = `Parent directory does not exist: ${path.dirname(filePath)}`;
          break;
        case 'EMFILE':
        case 'ENFILE':
          errorMessage = 'Too many open files, retry later';
          break;
        default:
          break;
      }

      return this.createResult(false, undefined, errorMessage, errorCode, {
        filePath,
        operationTime: Date.now() - startTime,
      });
    }
  }

  async appendTextFile(
    filePath: string,
    content: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(filePath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      // Read existing content (using cache if valid)
      let existingContent = '';
      const fileExists = await this.exists(filePath);

      if (fileExists) {
        const readResult = await this.readTextFile(filePath, {
          ...options,
          encoding: opts.encoding,
        });
        if (!readResult.success) {
          return this.createResult(
            false,
            undefined,
            `Failed to read existing content: ${readResult.error}`,
            readResult.errorCode,
          );
        }
        existingContent = readResult.data || '';
      }

      // Prepare new content with proper line ending
      const needsNewline =
        existingContent.length > 0 && !existingContent.endsWith('\n');
      const newContent = existingContent + (needsNewline ? '\n' : '') + content;

      // Validate total content size
      const totalSize = Buffer.byteLength(newContent, opts.encoding);
      if (totalSize > opts.maxFileSize) {
        return this.createResult(
          false,
          undefined,
          `Total content size ${totalSize} bytes exceeds maximum allowed ${opts.maxFileSize} bytes`,
          'FILE_TOO_LARGE',
        );
      }

      // Write the combined content atomically
      const writeResult = await this.writeTextFile(
        filePath,
        newContent,
        options,
      );

      if (writeResult.success) {
        return this.createResult(true, undefined, undefined, undefined, {
          filePath,
          fileSize: totalSize,
          operationTime: Date.now() - startTime,
          encoding: opts.encoding,
        });
      } else {
        return writeResult;
      }
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult(
        false,
        undefined,
        `Failed to append to file: ${nodeError.message}`,
        errorCode,
        {
          filePath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async readBinaryFile(
    filePath: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult<Buffer>> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(filePath)) {
      return this.createResult<Buffer>(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      await this.validateFileSize(filePath, opts.maxFileSize);

      const operation = fs.readFile(filePath);
      const buffer = await this.withTimeout(
        operation,
        opts.timeout,
        'readBinaryFile',
      );

      return this.createResult(true, buffer, undefined, undefined, {
        filePath,
        fileSize: buffer.length,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult<Buffer>(
        false,
        undefined,
        `Failed to read binary file: ${nodeError.message}`,
        errorCode,
        {
          filePath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async writeBinaryFile(
    filePath: string,
    data: Buffer,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(filePath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      if (data.length > opts.maxFileSize) {
        return this.createResult(
          false,
          undefined,
          `Data size ${data.length} bytes exceeds maximum allowed ${opts.maxFileSize} bytes`,
          'FILE_TOO_LARGE',
        );
      }

      if (opts.createDirectories) {
        const dirPath = path.dirname(filePath);
        await this.ensureDirectory(dirPath);
      }

      if (opts.atomicWrites) {
        const tempPath = `${filePath}.tmp.${Date.now()}.${Math.random().toString(36).substr(2, 9)}`;

        try {
          const operation = fs.writeFile(tempPath, data);
          await this.withTimeout(operation, opts.timeout, 'writeBinaryFile');
          await fs.rename(tempPath, filePath);
        } catch (error) {
          try {
            await fs.unlink(tempPath);
          } catch {
            /* intentional */
          }
          throw error;
        }
      } else {
        const operation = fs.writeFile(filePath, data);
        await this.withTimeout(operation, opts.timeout, 'writeBinaryFile');
      }

      // Invalidate text cache for this file (binary write might affect text reads)
      this.invalidateCache(filePath);

      return this.createResult(true, undefined, undefined, undefined, {
        filePath,
        fileSize: data.length,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult(
        false,
        undefined,
        `Failed to write binary file: ${nodeError.message}`,
        errorCode,
        {
          filePath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async createDirectory(
    dirPath: string,
    options: { recursive?: boolean } = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();

    if (!this.isPathSafe(dirPath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe directory path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      await fs.mkdir(dirPath, { recursive: options.recursive });

      return this.createResult(true, undefined, undefined, undefined, {
        filePath: dirPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      if (nodeError.code === 'EEXIST') {
        return this.createResult(true, undefined, undefined, undefined, {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        });
      }

      return this.createResult(
        false,
        undefined,
        `Failed to create directory: ${nodeError.message}`,
        errorCode,
        {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async deleteFile(filePath: string): Promise<FileOperationResult> {
    const startTime = Date.now();

    if (!this.isPathSafe(filePath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      await fs.unlink(filePath);

      // Invalidate cache for deleted file
      this.invalidateCache(filePath);

      return this.createResult(true, undefined, undefined, undefined, {
        filePath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      if (nodeError.code === 'ENOENT') {
        return this.createResult(true, undefined, undefined, undefined, {
          filePath,
          operationTime: Date.now() - startTime,
        });
      }

      return this.createResult(
        false,
        undefined,
        `Failed to delete file: ${nodeError.message}`,
        errorCode,
        {
          filePath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async deleteDirectory(
    dirPath: string,
    options: { recursive?: boolean } = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();

    if (!this.isPathSafe(dirPath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe directory path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      if (options.recursive) {
        await fs.rm(dirPath, { recursive: true, force: true });
      } else {
        await fs.rmdir(dirPath);
      }

      // Invalidate cache for all files under this directory
      const normalizedDir = path.resolve(dirPath);
      for (const cachedPath of this.fileCache.keys()) {
        if (cachedPath.startsWith(normalizedDir)) {
          this.fileCache.delete(cachedPath);
          this.cacheStats.invalidations++;
        }
      }

      return this.createResult(true, undefined, undefined, undefined, {
        filePath: dirPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      if (nodeError.code === 'ENOENT') {
        return this.createResult(true, undefined, undefined, undefined, {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        });
      }

      return this.createResult(
        false,
        undefined,
        `Failed to delete directory: ${nodeError.message}`,
        errorCode,
        {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async copyFile(
    sourcePath: string,
    destPath: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(sourcePath) || !this.isPathSafe(destPath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      await this.validateFileSize(sourcePath, opts.maxFileSize);

      if (opts.createDirectories) {
        const destDir = path.dirname(destPath);
        await this.ensureDirectory(destDir);
      }

      const operation = fs.copyFile(sourcePath, destPath);
      await this.withTimeout(operation, opts.timeout, 'copyFile');

      // Invalidate cache for destination (if it existed)
      this.invalidateCache(destPath);

      return this.createResult(true, undefined, undefined, undefined, {
        filePath: destPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult(
        false,
        undefined,
        `Failed to copy file: ${nodeError.message}`,
        errorCode,
        {
          filePath: destPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async moveFile(
    sourcePath: string,
    destPath: string,
    options: FileSystemOptions = {},
  ): Promise<FileOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };

    if (!this.isPathSafe(sourcePath) || !this.isPathSafe(destPath)) {
      return this.createResult(
        false,
        undefined,
        'Unsafe file path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      if (opts.createDirectories) {
        const destDir = path.dirname(destPath);
        await this.ensureDirectory(destDir);
      }

      const operation = fs.rename(sourcePath, destPath);
      await this.withTimeout(operation, opts.timeout, 'moveFile');

      // Invalidate cache for both source and destination
      this.invalidateCache(sourcePath);
      this.invalidateCache(destPath);

      return this.createResult(true, undefined, undefined, undefined, {
        filePath: destPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult(
        false,
        undefined,
        `Failed to move file: ${nodeError.message}`,
        errorCode,
        {
          filePath: destPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async listDirectory(dirPath: string): Promise<FileOperationResult<string[]>> {
    const startTime = Date.now();

    if (!this.isPathSafe(dirPath)) {
      return this.createResult<string[]>(
        false,
        undefined,
        'Unsafe directory path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      const entries = await fs.readdir(dirPath);

      return this.createResult(true, entries, undefined, undefined, {
        filePath: dirPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult<string[]>(
        false,
        undefined,
        `Failed to list directory: ${nodeError.message}`,
        errorCode,
        {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  async listDirectoryRecursive(
    dirPath: string,
    options: { maxDepth?: number; includeDirectories?: boolean } = {},
  ): Promise<FileOperationResult<string[]>> {
    const startTime = Date.now();
    const { maxDepth = 10, includeDirectories = false } = options;

    if (!this.isPathSafe(dirPath)) {
      return this.createResult<string[]>(
        false,
        undefined,
        'Unsafe directory path detected',
        'UNSAFE_PATH',
      );
    }

    try {
      // Optimized approach: use streaming results instead of accumulating all in memory
      const allEntries: string[] = [];
      const processedDirs = new Set<string>(); // Prevent infinite loops with symlinks

      const scanDirectory = async (
        currentPath: string,
        currentDepth: number,
      ): Promise<void> => {
        if (currentDepth > maxDepth) {
          return;
        }

        // Avoid processing the same directory twice (symlink protection)
        const resolvedPath = await fs.realpath(currentPath).catch(() => currentPath);
        if (processedDirs.has(resolvedPath)) {
          return;
        }
        processedDirs.add(resolvedPath);

        const entries = await fs.readdir(currentPath, { withFileTypes: true });

        // Process entries in batches for better memory management
        const batchSize = 100;
        for (let i = 0; i < entries.length; i += batchSize) {
          const batch = entries.slice(i, i + batchSize);
          
          for (const entry of batch) {
            const fullPath = path.join(currentPath, entry.name);
            const relativePath = path.relative(dirPath, fullPath);

            if (entry.isDirectory()) {
              if (includeDirectories) {
                allEntries.push(relativePath);
              }
              await scanDirectory(fullPath, currentDepth + 1);
            } else if (entry.isFile()) {
              allEntries.push(relativePath);
            }
          }
        }
      };

      await scanDirectory(dirPath, 0);

      // Optimized sorting: only sort if needed and use efficient algorithm
      const sortedEntries = allEntries.length > 1000 
        ? allEntries.sort() // Standard sort for large arrays
        : allEntries.sort((a, b) => a.localeCompare(b)); // Locale-aware sort for smaller arrays

      // Record performance metrics
      this.recordMetric('listDirectoryRecursive', startTime);

      return this.createResult(true, sortedEntries, undefined, undefined, {
        filePath: dirPath,
        operationTime: Date.now() - startTime,
      });
    } catch (error) {
      const nodeError = error as NodeJS.ErrnoException;
      const errorCode = this.extractErrorCode(error);

      return this.createResult<string[]>(
        false,
        undefined,
        `Failed to recursively list directory: ${nodeError.message}`,
        errorCode,
        {
          filePath: dirPath,
          operationTime: Date.now() - startTime,
        },
      );
    }
  }

  // Batch operations for improved throughput
  async batchReadTextFiles(
    filePaths: string[],
    options: FileSystemOptions = {},
  ): Promise<BatchOperationResult<string>> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };
    const successful: Array<{ path: string; result: FileOperationResult<string> }> = [];
    const failed: Array<{ path: string; error: string; errorCode?: string }> = [];

    // Process in batches to control memory usage and concurrency
    const batchSize = opts.batchSize;
    const batches: string[][] = [];
    for (let i = 0; i < filePaths.length; i += batchSize) {
      batches.push(filePaths.slice(i, i + batchSize));
    }

    for (const batch of batches) {
      // Process batch in parallel with controlled concurrency
      const promises = batch.map(async (filePath) => {
        const result = await this.readTextFile(filePath, options);
        if (result.success) {
          successful.push({ path: filePath, result });
        } else {
          failed.push({ 
            path: filePath, 
            error: result.error || 'Unknown error',
            errorCode: result.errorCode 
          });
        }
      });

      await Promise.all(promises);
    }

    const totalTime = Date.now() - startTime;
    return {
      successful,
      failed,
      totalProcessed: filePaths.length,
      totalTime,
      averageTimePerOperation: totalTime / filePaths.length,
    };
  }

  async batchWriteTextFiles(
    operations: Array<{ path: string; content: string }>,
    options: FileSystemOptions = {},
  ): Promise<BatchOperationResult> {
    const startTime = Date.now();
    const opts = { ...this.defaultOptions, ...options };
    const successful: Array<{ path: string; result: FileOperationResult }> = [];
    const failed: Array<{ path: string; error: string; errorCode?: string }> = [];

    // Process in batches to control memory usage and concurrency
    const batchSize = opts.batchSize;
    const batches: Array<Array<{ path: string; content: string }>> = [];
    for (let i = 0; i < operations.length; i += batchSize) {
      batches.push(operations.slice(i, i + batchSize));
    }

    for (const batch of batches) {
      // Process batch in parallel with controlled concurrency
      const promises = batch.map(async (operation) => {
        const result = await this.writeTextFile(operation.path, operation.content, options);
        if (result.success) {
          successful.push({ path: operation.path, result });
        } else {
          failed.push({ 
            path: operation.path, 
            error: result.error || 'Unknown error',
            errorCode: result.errorCode 
          });
        }
      });

      await Promise.all(promises);
    }

    const totalTime = Date.now() - startTime;
    return {
      successful,
      failed,
      totalProcessed: operations.length,
      totalTime,
      averageTimePerOperation: totalTime / operations.length,
    };
  }

  // Performance monitoring methods
  getPerformanceMetrics(): FileSystemMetrics {
    return {
      operationCounts: new Map(this.performanceMetrics.operationCounts),
      operationTimes: new Map(this.performanceMetrics.operationTimes),
      cacheHits: this.performanceMetrics.cacheHits,
      cacheMisses: this.performanceMetrics.cacheMisses,
      pathSafetyCacheHits: this.performanceMetrics.pathSafetyCacheHits,
      pathSafetyCacheMisses: this.performanceMetrics.pathSafetyCacheMisses,
      totalOperations: this.performanceMetrics.totalOperations,
      averageOperationTime: this.performanceMetrics.averageOperationTime,
    };
  }

  resetPerformanceMetrics(): void {
    this.performanceMetrics = {
      operationCounts: new Map(),
      operationTimes: new Map(),
      cacheHits: 0,
      cacheMisses: 0,
      pathSafetyCacheHits: 0,
      pathSafetyCacheMisses: 0,
      totalOperations: 0,
      averageOperationTime: 0,
    };
  }

  optimizeCache(): void {
    // Perform cache optimization based on usage patterns
    
    // Clean up expired path safety cache entries
    this.evictOldPathSafetyEntries();
    
    // Optimize file cache by removing least frequently used entries
    if (this.fileCache.size > this.maxCacheEntries * 0.8) {
      const entries = Array.from(this.fileCache.entries())
        .sort((a, b) => {
          // Sort by access count (ascending) then by last accessed (ascending)
          if (a[1].accessCount !== b[1].accessCount) {
            return a[1].accessCount - b[1].accessCount;
          }
          return a[1].lastAccessed - b[1].lastAccessed;
        });
      
      // Remove bottom 30% of entries
      const toRemove = entries.slice(0, Math.floor(entries.length * 0.3));
      toRemove.forEach(([key]) => {
        this.fileCache.delete(key);
        this.cacheStats.invalidations++;
      });
    }
    
    // Update cache statistics
    this.performanceMetrics.cacheHits = this.cacheStats.hits;
    this.performanceMetrics.cacheMisses = this.cacheStats.misses;
  }

  /**
   * Clear cache entries for a specific file or all files
   */
  clearCache(filePath?: string): void {
    if (filePath) {
      this.invalidateCache(filePath);
    } else {
      const clearedCount = this.fileCache.size;
      this.fileCache.clear();
      this.cacheStats.invalidations += clearedCount;
    }
  }

  /**
   * Get cache performance statistics
   */
  getCacheStats(): { size: number; hitRate: number; totalRequests: number } {
    const hitRate =
      this.cacheStats.totalRequests > 0
        ? (this.cacheStats.hits / this.cacheStats.totalRequests) * 100
        : 0;

    return {
      size: this.fileCache.size,
      hitRate: Math.round(hitRate * 100) / 100,
      totalRequests: this.cacheStats.totalRequests,
    };
  }

  /**
   * Get detailed cache performance metrics
   */
  getCacheMetrics(): {
    fileCache: {
      size: number;
      maxSize: number;
      hitRate: number;
      hits: number;
      misses: number;
    };
    pathSafetyCache: {
      size: number;
      hitRate: number;
      hits: number;
      misses: number;
    };
  } {
    const fileCacheHitRate = this.cacheStats.hits + this.cacheStats.misses > 0
      ? this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses)
      : 0;

    const pathSafetyHitRate = this.performanceMetrics.pathSafetyCacheHits +
      this.performanceMetrics.pathSafetyCacheMisses > 0
      ? this.performanceMetrics.pathSafetyCacheHits /
        (this.performanceMetrics.pathSafetyCacheHits + this.performanceMetrics.pathSafetyCacheMisses)
      : 0;

    return {
      fileCache: {
        size: this.fileCache.size,
        maxSize: this.maxCacheEntries,
        hitRate: fileCacheHitRate,
        hits: this.cacheStats.hits,
        misses: this.cacheStats.misses,
      },
      pathSafetyCache: {
        size: this.pathSafetyCache.size,
        hitRate: pathSafetyHitRate,
        hits: this.performanceMetrics.pathSafetyCacheHits,
        misses: this.performanceMetrics.pathSafetyCacheMisses,
      },
    };
  }

  /**
   * Get diagnostic information about the service
   */
  getDiagnostics(): {
    cacheMetrics: ReturnType<StandardFileSystemService['getCacheMetrics']>;
    performanceMetrics: ReturnType<StandardFileSystemService['getPerformanceMetrics']>;
    configuration: {
      defaultTimeout: number;
      defaultMaxFileSize: number;
      performanceMode: PerformanceMode;
      cacheEnabled: boolean;
      maxCacheEntries: number;
    };
  } {
    return {
      cacheMetrics: this.getCacheMetrics(),
      performanceMetrics: this.getPerformanceMetrics(),
      configuration: {
        defaultTimeout: this.defaultOptions.timeout,
        defaultMaxFileSize: this.defaultOptions.maxFileSize,
        performanceMode: this.defaultOptions.performanceMode,
        cacheEnabled: true,
        maxCacheEntries: this.maxCacheEntries,
      },
    };
  }

  /**
   * Clear all caches - useful for testing or memory management
   */
  clearCaches(): void {
    this.fileCache.clear();
    this.pathSafetyCache.clear();
    this.cacheStats = {
      hits: 0,
      misses: 0,
      invalidations: 0,
      totalRequests: 0,
    };
    this.performanceMetrics = {
      operationCounts: new Map(),
      operationTimes: new Map(),
      cacheHits: 0,
      cacheMisses: 0,
      pathSafetyCacheHits: 0,
      pathSafetyCacheMisses: 0,
      totalOperations: 0,
      averageOperationTime: 0,
    };
  }
}

/**
 * Fallback file system service for environments where full fs access is not available.
 * Following Crisalida's graceful degradation patterns.
 */
export class FallbackFileSystemService implements FileSystemService {
  isPathSafe(): boolean {
    return false;
  }

  async readTextFile(): Promise<FileOperationResult<string>> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async writeTextFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async appendTextFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async readBinaryFile(): Promise<FileOperationResult<Buffer>> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async writeBinaryFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async exists(): Promise<boolean> {
    return false;
  }

  async getFileInfo(): Promise<FileOperationResult<FileInfo>> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async createDirectory(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async deleteFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async deleteDirectory(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async copyFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async moveFile(): Promise<FileOperationResult> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async listDirectory(): Promise<FileOperationResult<string[]>> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  async listDirectoryRecursive(): Promise<FileOperationResult<string[]>> {
    return {
      success: false,
      error: 'File system operations not available in this environment',
      errorCode: 'UNAVAILABLE',
    };
  }

  clearCache(): void {
    // No-op for fallback
  }

  getCacheStats(): { size: number; hitRate: number; totalRequests: number } {
    return { size: 0, hitRate: 0, totalRequests: 0 };
  }

  async batchReadTextFiles(): Promise<BatchOperationResult<string>> {
    return {
      successful: [],
      failed: [],
      totalProcessed: 0,
      totalTime: 0,
      averageTimePerOperation: 0,
    };
  }

  async batchWriteTextFiles(): Promise<BatchOperationResult> {
    return {
      successful: [],
      failed: [],
      totalProcessed: 0,
      totalTime: 0,
      averageTimePerOperation: 0,
    };
  }

  getPerformanceMetrics(): FileSystemMetrics {
    return {
      operationCounts: new Map(),
      operationTimes: new Map(),
      cacheHits: 0,
      cacheMisses: 0,
      pathSafetyCacheHits: 0,
      pathSafetyCacheMisses: 0,
      totalOperations: 0,
      averageOperationTime: 0,
    };
  }

  resetPerformanceMetrics(): void {
    // No-op for fallback
  }

  optimizeCache(): void {
    // No-op for fallback
  }
}

/**
 * Factory function for creating file system service with graceful degradation
 */
export function createFileSystemService(): FileSystemService {
  try {
    // Test if we have fs access (runtime environment where 'fs' might be stubbed)
    if (typeof fs !== 'undefined') {
      return new StandardFileSystemService();
    }
  } catch {
    // Fall through to fallback
  }

  return new FallbackFileSystemService();
}
