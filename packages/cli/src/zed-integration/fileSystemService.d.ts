/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FileOperationResult, FileSystemService, FileInfo, FileSystemOptions, BatchOperationResult, FileSystemMetrics } from '@google/gemini-cli-core';
import * as acp from './acp.js';
/**
 * ACP client-based implementation of FileSystemService
 */
export declare class AcpFileSystemService implements FileSystemService {
    private readonly client;
    private readonly sessionId;
    private readonly capabilities;
    private readonly fallback;
    constructor(client: acp.Client, sessionId: string, capabilities: acp.FileSystemCapability, fallback: FileSystemService);
    readTextFile(filePath: string): Promise<FileOperationResult<string>>;
    writeTextFile(filePath: string, content: string): Promise<FileOperationResult>;
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
    batchReadTextFiles(filePaths: string[], options?: FileSystemOptions): Promise<BatchOperationResult<string>>;
    batchWriteTextFiles(operations: Array<{
        path: string;
        content: string;
    }>, options?: FileSystemOptions): Promise<BatchOperationResult<void>>;
    getPerformanceMetrics(): FileSystemMetrics;
    resetPerformanceMetrics(): void;
    optimizeCache(): void;
}
