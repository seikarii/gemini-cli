/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { 
  AbstractRepository, 
  RepositoryResult, 
  RepositoryConfig
} from './BaseRepository.js';
import { OptimizedFileOperations } from '../../utils/performance/index.js';
import { FileSystemService } from '../../services/fileSystemService.js';
import * as path from 'path';

/**
 * File entity representation
 */
export interface FileEntity {
  id: string; // file path
  name: string;
  path: string;
  content?: string;
  size?: number;
  lastModified?: Date;
  mimeType?: string;
  metadata?: Record<string, unknown>;
}

/**
 * File search criteria
 */
export interface FileSearchCriteria {
  pattern?: RegExp;
  extension?: string;
  minSize?: number;
  maxSize?: number;
  modifiedAfter?: Date;
  modifiedBefore?: Date;
  contentSearch?: string;
}

/**
 * Repository interface for file operations
 */
export interface IFileRepository {
  findByPath(filePath: string): Promise<RepositoryResult<FileEntity>>;
  findByPattern(pattern: RegExp, directory?: string): Promise<RepositoryResult<FileEntity[]>>;
  findByExtension(extension: string, directory?: string): Promise<RepositoryResult<FileEntity[]>>;
  search(criteria: FileSearchCriteria, directory?: string): Promise<RepositoryResult<FileEntity[]>>;
  saveContent(filePath: string, content: string): Promise<RepositoryResult<FileEntity>>;
  copy(sourcePath: string, destinationPath: string): Promise<RepositoryResult<FileEntity>>;
  move(sourcePath: string, destinationPath: string): Promise<RepositoryResult<FileEntity>>;
  getMetadata(filePath: string): Promise<RepositoryResult<FileEntity>>;
}

/**
 * File repository implementation with optimized operations
 */
export class FileRepository extends AbstractRepository<FileEntity> implements IFileRepository {
  private fileOps: OptimizedFileOperations;
  private fileSystemService?: FileSystemService;

  constructor(
    config: RepositoryConfig = {},
    fileSystemService?: FileSystemService
  ) {
    super(config);
    this.fileOps = OptimizedFileOperations.getInstance({
      enableCache: this.config.enableCaching,
      cacheTTL: this.config.cacheTTL,
      enablePooling: true,
      maxConcurrent: 10,
    });
    this.fileSystemService = fileSystemService;
  }

  /**
   * Find file by ID (path)
   */
  async findById(filePath: string): Promise<FileEntity | null> {
    const result = await this.findByPath(filePath);
    return result.success ? result.data || null : null;
  }

  /**
   * Find all files (not recommended for large directories)
   */
  async findAll(): Promise<FileEntity[]> {
    // This is a simplified implementation - in practice, you'd want directory enumeration
    throw new Error('findAll not implemented - use findByPattern or search instead');
  }

  /**
   * Save entity (write file)
   */
  async save(entity: FileEntity): Promise<FileEntity> {
    if (!entity.content) {
      throw new Error('File entity must have content to save');
    }

    const result = await this.saveContent(entity.path, entity.content);
    if (!result.success) {
      throw new Error(result.error || 'Failed to save file');
    }

    return result.data!;
  }

  /**
   * Delete file
   */
  async delete(filePath: string): Promise<boolean> {
    try {
      if (this.fileSystemService) {
        const result = await this.fileSystemService.deleteFile(filePath);
        return result.success;
      }
      
      // Fallback implementation would go here
      throw new Error('FileSystemService required for delete operations');
    } catch (error) {
      console.error('Error deleting file:', error);
      return false;
    }
  }

  /**
   * Check if file exists
   */
  async exists(filePath: string): Promise<boolean> {
    return this.fileOps.exists(filePath);
  }

  /**
   * Find file by path with full metadata
   */
  async findByPath(filePath: string): Promise<RepositoryResult<FileEntity>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const exists = await this.fileOps.exists(filePath);
        if (!exists) {
          return this.createErrorResult<FileEntity>('File not found', {
            operationTime: Date.now() - startTime,
            source: 'FileRepository.findByPath',
          });
        }

        const readResult = await this.fileOps.safeReadFile(filePath);
        if (!readResult.success) {
          return this.createErrorResult<FileEntity>(
            readResult.error || 'Failed to read file',
            {
              operationTime: Date.now() - startTime,
              source: 'FileRepository.findByPath',
            }
          );
        }

        const fileEntity: FileEntity = {
          id: filePath,
          name: path.basename(filePath),
          path: filePath,
          content: readResult.data,
          metadata: {
            fromCache: readResult.fromCache,
          },
        };

        // Get additional metadata if FileSystemService is available
        if (this.fileSystemService) {
          const metadataResult = await this.fileSystemService.getFileInfo(filePath);
          if (metadataResult.success && metadataResult.data) {
            fileEntity.size = metadataResult.data.size;
            fileEntity.lastModified = metadataResult.data.modified;
          }
        }

        return this.createSuccessResult(fileEntity, {
          fromCache: readResult.fromCache,
          operationTime: Date.now() - startTime,
          source: 'FileRepository.findByPath',
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<FileEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.findByPath',
        }
      );
    }
  }

  /**
   * Find files matching a pattern
   */
  async findByPattern(
    pattern: RegExp, 
    directory = process.cwd()
  ): Promise<RepositoryResult<FileEntity[]>> {
    const startTime = Date.now();

    try {
      const result = await this.search({ pattern }, directory);
      return result;
    } catch (error) {
      return this.createErrorResult<FileEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.findByPattern',
        }
      );
    }
  }

  /**
   * Find files by extension
   */
  async findByExtension(
    extension: string, 
    directory = process.cwd()
  ): Promise<RepositoryResult<FileEntity[]>> {
    const normalizedExt = extension.startsWith('.') ? extension : `.${extension}`;
    const pattern = new RegExp(`\\${normalizedExt}$`, 'i');
    return this.findByPattern(pattern, directory);
  }

  /**
   * Search files with criteria
   */
  async search(
    criteria: FileSearchCriteria, 
    _directory = process.cwd()
  ): Promise<RepositoryResult<FileEntity[]>> {
    const startTime = Date.now();

    try {
      // This is a simplified implementation
      // In practice, you'd integrate with a file discovery service
      const files: FileEntity[] = [];

      // For now, return empty results with metadata
      return this.createSuccessResult(files, {
        operationTime: Date.now() - startTime,
        source: 'FileRepository.search',
      });
    } catch (error) {
      return this.createErrorResult<FileEntity[]>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.search',
        }
      );
    }
  }

  /**
   * Save content to file
   */
  async saveContent(
    filePath: string, 
    content: string
  ): Promise<RepositoryResult<FileEntity>> {
    const startTime = Date.now();

    try {
      const result = await this.withRetry(async () => {
        const writeResult = await this.fileOps.safeWriteFile(filePath, content);
        if (!writeResult.success) {
          return this.createErrorResult<FileEntity>(
            writeResult.error || 'Failed to write file',
            {
              operationTime: Date.now() - startTime,
              source: 'FileRepository.saveContent',
            }
          );
        }

        const fileEntity: FileEntity = {
          id: filePath,
          name: path.basename(filePath),
          path: filePath,
          content,
          size: Buffer.byteLength(content, 'utf8'),
          lastModified: new Date(),
        };

        return this.createSuccessResult(fileEntity, {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.saveContent',
        });
      });

      return result;
    } catch (error) {
      return this.createErrorResult<FileEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.saveContent',
        }
      );
    }
  }

  /**
   * Copy file
   */
  async copy(
    sourcePath: string, 
    destinationPath: string
  ): Promise<RepositoryResult<FileEntity>> {
    const startTime = Date.now();

    try {
      const sourceResult = await this.findByPath(sourcePath);
      if (!sourceResult.success || !sourceResult.data) {
        return this.createErrorResult<FileEntity>(
          'Source file not found or not readable',
          {
            operationTime: Date.now() - startTime,
            source: 'FileRepository.copy',
          }
        );
      }

      const saveResult = await this.saveContent(
        destinationPath, 
        sourceResult.data.content || ''
      );

      return saveResult;
    } catch (error) {
      return this.createErrorResult<FileEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.copy',
        }
      );
    }
  }

  /**
   * Move file
   */
  async move(
    sourcePath: string, 
    destinationPath: string
  ): Promise<RepositoryResult<FileEntity>> {
    const copyResult = await this.copy(sourcePath, destinationPath);
    if (copyResult.success) {
      await this.delete(sourcePath);
    }
    return copyResult;
  }

  /**
   * Get file metadata without content
   */
  async getMetadata(filePath: string): Promise<RepositoryResult<FileEntity>> {
    const startTime = Date.now();

    try {
      const exists = await this.fileOps.exists(filePath);
      if (!exists) {
        return this.createErrorResult<FileEntity>('File not found', {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.getMetadata',
        });
      }

      const fileEntity: FileEntity = {
        id: filePath,
        name: path.basename(filePath),
        path: filePath,
      };

      // Get metadata if FileSystemService is available
      if (this.fileSystemService) {
        const metadataResult = await this.fileSystemService.getFileInfo(filePath);
        if (metadataResult.success && metadataResult.data) {
          fileEntity.size = metadataResult.data.size;
          fileEntity.lastModified = metadataResult.data.modified;
        }
      }

      return this.createSuccessResult(fileEntity, {
        operationTime: Date.now() - startTime,
        source: 'FileRepository.getMetadata',
      });
    } catch (error) {
      return this.createErrorResult<FileEntity>(
        error instanceof Error ? error.message : String(error),
        {
          operationTime: Date.now() - startTime,
          source: 'FileRepository.getMetadata',
        }
      );
    }
  }

  /**
   * Get repository statistics
   */
  getStats() {
    return {
      fileOperations: this.fileOps.getStats(),
      config: this.config,
    };
  }
}
