/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'fs/promises';
import path from 'path';

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

  /**
   * Validate and sanitize file path for security
   */
  isPathSafe(filePath: string, allowedRoots: string[] = []): boolean {
    try {
      // Normalize and resolve the path
      const normalizedPath = path.resolve(filePath);

      // Check for path traversal attempts
      if (filePath.includes('..') || filePath.includes('~')) {
        return false;
      }

      // Check for suspicious characters
      const suspiciousChars = /[<>:"|?*]/;
      if (suspiciousChars.test(filePath)) {
        return false;
      }

      // Validate against allowed roots if specified
      if (allowedRoots.length > 0) {
        const isInAllowedRoot = allowedRoots.some((root) => {
          const normalizedRoot = path.resolve(root);
          return normalizedPath.startsWith(normalizedRoot);
        });
        if (!isInAllowedRoot) {
          return false;
        }
      }

      // Prevent access to system directories (basic protection)
      const systemDirs = ['/etc', '/proc', '/sys', '/dev', '/root'];
      const isSystemDir = systemDirs.some((dir) =>
        normalizedPath.startsWith(dir),
      );
      if (isSystemDir && process.platform !== 'win32') {
        return false;
      }

      return true;
    } catch {
      return false;
    }
  }

  /**
   * Execute operation with timeout protection
   */
  private async withTimeout<T>(
    operation: Promise<T>,
    timeoutMs: number,
    operationName: string,
  ): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`${operationName} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });

    return Promise.race([operation, timeoutPromise]);
  }

  /**
   * Validate file size against limits
   */
  private async validateFileSize(
    filePath: string,
    maxSize: number,
  ): Promise<void> {
    try {
      const stats = await fs.stat(filePath);
      if (stats.size > maxSize) {
        throw new Error(
          `File size ${stats.size} bytes exceeds maximum allowed ${maxSize} bytes`,
        );
      }
    } catch (_error) {
      if ((_error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw _error;
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
          // If requested, create a checkpoint of the current file content before
          // performing the atomic write. This provides a quick local rollback
          // point in case a write becomes destructive.
          if (opts.createCheckpoint) {
            try {
              // Check whether the target file exists
              const stat = await fs.stat(filePath).catch(() => null);
              if (stat && stat.isFile()) {
                try {
                  const existing = await fs.readFile(filePath, opts.encoding).catch(() => null);
                  if (existing !== null) {
                    const checkpointDir = path.join(
                      path.dirname(filePath),
                      '.gemini_checkpoints',
                      `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                    );
                    // Ensure checkpoint directory exists
                    await fs.mkdir(checkpointDir, { recursive: true }).catch(() => null);
                    const checkpointPath = path.join(checkpointDir, path.basename(filePath));
                    // Write checkpoint using atomic semantics where possible
                    try {
                      await fs.writeFile(checkpointPath, existing, opts.encoding);
                    } catch (_cpErr) {
                      // Don't block the main write if checkpoint creation fails.
                      // Just log debug information to console for now.
                      console.debug('Failed to create checkpoint for', filePath);
                    }
                  }
                } catch {
                  /* ignore checkpoint read/write errors */
                }
              }
            } catch {
              /* ignore checkpoint discovery errors */
            }
          }
        // Atomic write: write to temporary file then rename
        const tempPath = `${filePath}.tmp.${Date.now()}.${Math.random().toString(36).substr(2, 9)}`;

        try {
          const operation = fs.writeFile(tempPath, content, opts.encoding);
          await this.withTimeout(operation, opts.timeout, 'writeTextFile');

          // Atomic rename
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
        // Direct write
        const operation = fs.writeFile(filePath, content, opts.encoding);
        await this.withTimeout(operation, opts.timeout, 'writeTextFile');
      }

      // Explicitly invalidate cache after successful write
      this.invalidateCache(filePath);

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
      const allEntries: string[] = [];

      const scanDirectory = async (
        currentPath: string,
        currentDepth: number,
      ): Promise<void> => {
        if (currentDepth > maxDepth) {
          return;
        }

        const entries = await fs.readdir(currentPath, { withFileTypes: true });

        for (const entry of entries) {
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
      };

      await scanDirectory(dirPath, 0);

      return this.createResult(true, allEntries.sort(), undefined, undefined, {
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
