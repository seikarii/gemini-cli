/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  FileOperationResult,
  FileSystemService,
  FileInfo,
  FileSystemOptions,
  BatchOperationResult,
  FileSystemMetrics,
} from '@google/gemini-cli-core';
import * as acp from './acp.js';

/**
 * ACP client-based implementation of FileSystemService
 */
export class AcpFileSystemService implements FileSystemService {
  constructor(
    private readonly client: acp.Client,
    private readonly sessionId: string,
    private readonly capabilities: acp.FileSystemCapability,
    private readonly fallback: FileSystemService,
  ) {}

  async readTextFile(filePath: string): Promise<FileOperationResult<string>> {
    if (!this.capabilities.readTextFile) {
      return this.fallback.readTextFile(filePath);
    }

    const response = await this.client.readTextFile({
      path: filePath,
      sessionId: this.sessionId,
      line: null,
      limit: null,
    });

    return {
      success: true,
      data: response.content,
    };
  }

  async writeTextFile(
    filePath: string,
    content: string,
  ): Promise<FileOperationResult> {
    if (!this.capabilities.writeTextFile) {
      return this.fallback.writeTextFile(filePath, content);
    }

    await this.client.writeTextFile({
      path: filePath,
      content,
      sessionId: this.sessionId,
    });

    return {
      success: true,
    };
  }

  appendTextFile(
    filePath: string,
    content: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult> {
    return this.fallback.appendTextFile(filePath, content, options);
  }

  readBinaryFile(
    filePath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult<Buffer>> {
    return this.fallback.readBinaryFile(filePath, options);
  }

  writeBinaryFile(
    filePath: string,
    data: Buffer,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult> {
    return this.fallback.writeBinaryFile(filePath, data, options);
  }

  exists(filePath: string): Promise<boolean> {
    return this.fallback.exists(filePath);
  }

  getFileInfo(filePath: string): Promise<FileOperationResult<FileInfo>> {
    return this.fallback.getFileInfo(filePath);
  }

  createDirectory(
    dirPath: string,
    options?: { recursive?: boolean },
  ): Promise<FileOperationResult> {
    return this.fallback.createDirectory(dirPath, options);
  }

  deleteFile(filePath: string): Promise<FileOperationResult> {
    return this.fallback.deleteFile(filePath);
  }

  deleteDirectory(
    dirPath: string,
    options?: { recursive?: boolean },
  ): Promise<FileOperationResult> {
    return this.fallback.deleteDirectory(dirPath, options);
  }

  copyFile(
    sourcePath: string,
    destPath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult> {
    return this.fallback.copyFile(sourcePath, destPath, options);
  }

  moveFile(
    sourcePath: string,
    destPath: string,
    options?: FileSystemOptions,
  ): Promise<FileOperationResult> {
    return this.fallback.moveFile(sourcePath, destPath, options);
  }

  listDirectory(dirPath: string): Promise<FileOperationResult<string[]>> {
    return this.fallback.listDirectory(dirPath);
  }

  listDirectoryRecursive(
    dirPath: string,
    options?: { maxDepth?: number; includeDirectories?: boolean },
  ): Promise<FileOperationResult<string[]>> {
    return this.fallback.listDirectoryRecursive(dirPath, options);
  }

  isPathSafe(filePath: string, allowedRoots?: string[]): boolean {
    return this.fallback.isPathSafe(filePath, allowedRoots);
  }

  clearCache(filePath?: string): void {
    return this.fallback.clearCache(filePath);
  }

  getCacheStats(): { size: number; hitRate: number; totalRequests: number } {
    return this.fallback.getCacheStats();
  }

  // New batch and performance methods
  async batchReadTextFiles(
    filePaths: string[],
    options?: FileSystemOptions,
  ): Promise<BatchOperationResult<string>> {
    return this.fallback.batchReadTextFiles(filePaths, options);
  }

  async batchWriteTextFiles(
    operations: Array<{ path: string; content: string }>,
    options?: FileSystemOptions,
  ): Promise<BatchOperationResult<void>> {
    return this.fallback.batchWriteTextFiles(operations, options);
  }

  getPerformanceMetrics(): FileSystemMetrics {
    return this.fallback.getPerformanceMetrics();
  }

  resetPerformanceMetrics(): void {
    return this.fallback.resetPerformanceMetrics();
  }

  optimizeCache(): void {
    return this.fallback.optimizeCache();
  }
}
