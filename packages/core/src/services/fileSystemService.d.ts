/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

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
    errorCode?: string;
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
    mtimeMs: number;
}
/**
 * Enhanced file system service interface following Crisalida patterns
 */
export interface FileSystemService {
    readTextFile(filePath: string, options?: FileSystemOptions): Promise<FileOperationResult<string>>;
    writeTextFile(filePath: string, content: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    appendTextFile(filePath: string, content: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    readBinaryFile(filePath: string, options?: FileSystemOptions): Promise<FileOperationResult<Buffer>>;
    writeBinaryFile(filePath: string, data: Buffer, options?: FileSystemOptions): Promise<FileOperationResult>;
    exists(filePath: string): Promise<boolean>;
    getFileInfo(filePath: string): Promise<FileOperationResult<FileInfo>>;
    createDirectory(dirPath: string, options?: {
        recursive?: boolean;
    }): Promise<FileOperationResult>;
    deleteFile(filePath: string): Promise<FileOperationResult>;
    deleteDirectory(dirPath: string, options?: {
        recursive?: boolean;
    }): Promise<FileOperationResult>;
    copyFile(sourcePath: string, destPath: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    moveFile(sourcePath: string, destPath: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    listDirectory(dirPath: string): Promise<FileOperationResult<string[]>>;
    listDirectoryRecursive(dirPath: string, options?: {
        maxDepth?: number;
        includeDirectories?: boolean;
    }): Promise<FileOperationResult<string[]>>;
    isPathSafe(filePath: string, allowedRoots?: string[]): boolean;
    clearCache(filePath?: string): void;
    getCacheStats(): {
        size: number;
        hitRate: number;
        totalRequests: number;
    };
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
export declare class StandardFileSystemService implements FileSystemService {
    private readonly defaultOptions;
    private readonly fileCache;
    private readonly maxCacheEntries;
    private cacheStats;
    /**
     * Validate and sanitize file path for security
     */
    isPathSafe(filePath: string, allowedRoots?: string[]): boolean;
    /**
     * Execute operation with timeout protection
     */
    private withTimeout;
    /**
     * Validate file size against limits
     */
    private validateFileSize;
    /**
     * Ensure directory exists, following Crisalida defensive patterns
     */
    private ensureDirectory;
    /**
     * Create structured operation result with error code extraction
     */
    private createResult;
    /**
     * Extract error code from NodeJS error for structured reporting
     */
    private extractErrorCode;
    /**
     * Get high-precision mtime for cache validation
     */
    private getFileMtimeMs;
    /**
     * Check if cached content is still valid by comparing mtime
     */
    private isCacheValid;
    /**
     * Manage cache size by evicting least recently used entries
     */
    private evictLeastRecentlyUsed;
    /**
     * Invalidate cache entry for a specific file
     */
    private invalidateCache;
    /**
     * Get content from cache or disk with self-validation
     */
    private getCachedOrFreshContent;
    exists(filePath: string): Promise<boolean>;
    getFileInfo(filePath: string): Promise<FileOperationResult<FileInfo>>;
    readTextFile(filePath: string, options?: FileSystemOptions): Promise<FileOperationResult<string>>;
    writeTextFile(filePath: string, content: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    appendTextFile(filePath: string, content: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    readBinaryFile(filePath: string, options?: FileSystemOptions): Promise<FileOperationResult<Buffer>>;
    writeBinaryFile(filePath: string, data: Buffer, options?: FileSystemOptions): Promise<FileOperationResult>;
    createDirectory(dirPath: string, options?: {
        recursive?: boolean;
    }): Promise<FileOperationResult>;
    deleteFile(filePath: string): Promise<FileOperationResult>;
    deleteDirectory(dirPath: string, options?: {
        recursive?: boolean;
    }): Promise<FileOperationResult>;
    copyFile(sourcePath: string, destPath: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    moveFile(sourcePath: string, destPath: string, options?: FileSystemOptions): Promise<FileOperationResult>;
    listDirectory(dirPath: string): Promise<FileOperationResult<string[]>>;
    listDirectoryRecursive(dirPath: string, options?: {
        maxDepth?: number;
        includeDirectories?: boolean;
    }): Promise<FileOperationResult<string[]>>;
    /**
     * Clear cache entries for a specific file or all files
     */
    clearCache(filePath?: string): void;
    /**
     * Get cache performance statistics
     */
    getCacheStats(): {
        size: number;
        hitRate: number;
        totalRequests: number;
    };
}
/**
 * Fallback file system service for environments where full fs access is not available.
 * Following Crisalida's graceful degradation patterns.
 */
export declare class FallbackFileSystemService implements FileSystemService {
    isPathSafe(): boolean;
    readTextFile(): Promise<FileOperationResult<string>>;
    writeTextFile(): Promise<FileOperationResult>;
    appendTextFile(): Promise<FileOperationResult>;
    readBinaryFile(): Promise<FileOperationResult<Buffer>>;
    writeBinaryFile(): Promise<FileOperationResult>;
    exists(): Promise<boolean>;
    getFileInfo(): Promise<FileOperationResult<FileInfo>>;
    createDirectory(): Promise<FileOperationResult>;
    deleteFile(): Promise<FileOperationResult>;
    deleteDirectory(): Promise<FileOperationResult>;
    copyFile(): Promise<FileOperationResult>;
    moveFile(): Promise<FileOperationResult>;
    listDirectory(): Promise<FileOperationResult<string[]>>;
    listDirectoryRecursive(): Promise<FileOperationResult<string[]>>;
    clearCache(): void;
    getCacheStats(): {
        size: number;
        hitRate: number;
        totalRequests: number;
    };
}
/**
 * Factory function for creating file system service with graceful degradation
 */
export declare function createFileSystemService(): FileSystemService;
