/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { FileRepository, FileEntity } from '../../../src/architecture/repositories/FileRepository.js';
import { OptimizedFileOperations } from '../../../src/utils/performance/index.js';

// Mock the dependencies
vi.mock('../../../src/utils/performance/index.js');

describe('FileRepository', () => {
  let fileRepository: FileRepository;
  let mockFileOps: {
    exists: ReturnType<typeof vi.fn>;
    safeReadFile: ReturnType<typeof vi.fn>;
    safeWriteFile: ReturnType<typeof vi.fn>;
    getStats: ReturnType<typeof vi.fn>;
  };
  let mockFileSystemService: {
    getFileInfo: ReturnType<typeof vi.fn>;
    deleteFile: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Mock OptimizedFileOperations
    mockFileOps = {
      exists: vi.fn(),
      safeReadFile: vi.fn(),
      safeWriteFile: vi.fn(),
      getStats: vi.fn(() => ({})),
    };

    // Mock FileSystemService
    mockFileSystemService = {
      getFileInfo: vi.fn(),
      deleteFile: vi.fn(),
    };

    // Mock the singleton getInstance
    (OptimizedFileOperations.getInstance as unknown as ReturnType<typeof vi.fn>) = vi.fn().mockReturnValue(mockFileOps);

    fileRepository = new FileRepository({}, mockFileSystemService);
  });

  describe('findByPath', () => {
    it('should return file entity when file exists and is readable', async () => {
      const filePath = '/test/file.txt';
      const fileContent = 'test content';

      mockFileOps.exists.mockResolvedValue(true);
      mockFileOps.safeReadFile.mockResolvedValue({
        success: true,
        data: fileContent,
        fromCache: false,
      });

      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          size: 100,
          modified: new Date('2025-01-01'),
        },
      });

      const result = await fileRepository.findByPath(filePath);

      expect(result.success).toBe(true);
      expect(result.data).toEqual({
        id: filePath,
        name: 'file.txt',
        path: filePath,
        content: fileContent,
        size: 100,
        lastModified: new Date('2025-01-01'),
        metadata: {
          fromCache: false,
        },
      });
    });

    it('should return error when file does not exist', async () => {
      const filePath = '/test/nonexistent.txt';

      mockFileOps.exists.mockResolvedValue(false);

      const result = await fileRepository.findByPath(filePath);

      expect(result.success).toBe(false);
      expect(result.error).toBe('File not found');
    });

    it('should return error when file read fails', async () => {
      const filePath = '/test/unreadable.txt';

      mockFileOps.exists.mockResolvedValue(true);
      mockFileOps.safeReadFile.mockResolvedValue({
        success: false,
        error: 'Permission denied',
      });

      const result = await fileRepository.findByPath(filePath);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Permission denied');
    });

    it('should handle file info retrieval failure gracefully', async () => {
      const filePath = '/test/file.txt';
      const fileContent = 'test content';

      mockFileOps.exists.mockResolvedValue(true);
      mockFileOps.safeReadFile.mockResolvedValue({
        success: true,
        data: fileContent,
        fromCache: false,
      });

      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: false,
        error: 'Metadata not available',
      });

      const result = await fileRepository.findByPath(filePath);

      expect(result.success).toBe(true);
      expect(result.data?.content).toBe(fileContent);
      expect(result.data?.size).toBeUndefined();
      expect(result.data?.lastModified).toBeUndefined();
    });
  });

  describe('saveContent', () => {
    it('should save file content successfully', async () => {
      const filePath = '/test/new-file.txt';
      const content = 'new content';

      mockFileOps.safeWriteFile.mockResolvedValue({
        success: true,
      });

      const result = await fileRepository.saveContent(filePath, content);

      expect(result.success).toBe(true);
      expect(result.data).toEqual({
        id: filePath,
        name: 'new-file.txt',
        path: filePath,
        content,
        size: Buffer.byteLength(content, 'utf8'),
        lastModified: expect.any(Date),
      });

      expect(mockFileOps.safeWriteFile).toHaveBeenCalledWith(filePath, content);
    });

    it('should return error when write fails', async () => {
      const filePath = '/test/readonly.txt';
      const content = 'content';

      mockFileOps.safeWriteFile.mockResolvedValue({
        success: false,
        error: 'Permission denied',
      });

      const result = await fileRepository.saveContent(filePath, content);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Permission denied');
    });
  });

  describe('copy', () => {
    it('should copy file successfully', async () => {
      const sourcePath = '/test/source.txt';
      const destinationPath = '/test/destination.txt';
      const content = 'file content';

      // Mock findByPath for source file
      const findByPathSpy = vi.spyOn(fileRepository, 'findByPath').mockResolvedValue({
        success: true,
        data: {
          id: sourcePath,
          name: 'source.txt',
          path: sourcePath,
          content,
          metadata: { fromCache: false },
        },
        metadata: {},
      });

      // Mock saveContent for destination
      const saveContentSpy = vi.spyOn(fileRepository, 'saveContent').mockResolvedValue({
        success: true,
        data: {
          id: destinationPath,
          name: 'destination.txt',
          path: destinationPath,
          content,
          size: Buffer.byteLength(content, 'utf8'),
          lastModified: new Date(),
          metadata: {},
        },
        metadata: {},
      });

      const result = await fileRepository.copy(sourcePath, destinationPath);

      expect(result.success).toBe(true);
      expect(result.data?.path).toBe(destinationPath);
      expect(result.data?.content).toBe(content);
      expect(findByPathSpy).toHaveBeenCalledWith(sourcePath);
      expect(saveContentSpy).toHaveBeenCalledWith(destinationPath, content);
    });

    it('should return error when source file does not exist', async () => {
      const sourcePath = '/test/nonexistent.txt';
      const destinationPath = '/test/destination.txt';

      vi.spyOn(fileRepository, 'findByPath').mockResolvedValue({
        success: false,
        error: 'File not found',
        metadata: {},
      });

      const result = await fileRepository.copy(sourcePath, destinationPath);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Source file not found or not readable');
    });
  });

  describe('move', () => {
    it('should move file successfully', async () => {
      const sourcePath = '/test/source.txt';
      const destinationPath = '/test/destination.txt';
      const content = 'file content';

      // Mock copy operation
      const copySpy = vi.spyOn(fileRepository, 'copy').mockResolvedValue({
        success: true,
        data: {
          id: destinationPath,
          name: 'destination.txt',
          path: destinationPath,
          content,
          metadata: {},
        },
        metadata: {},
      });

      // Mock delete operation
      const deleteSpy = vi.spyOn(fileRepository, 'delete').mockResolvedValue(true);

      const result = await fileRepository.move(sourcePath, destinationPath);

      expect(result.success).toBe(true);
      expect(result.data?.path).toBe(destinationPath);
      expect(copySpy).toHaveBeenCalledWith(sourcePath, destinationPath);
      expect(deleteSpy).toHaveBeenCalledWith(sourcePath);
    });
  });

  describe('delete', () => {
    it('should delete file successfully', async () => {
      const filePath = '/test/file.txt';

      mockFileSystemService.deleteFile.mockResolvedValue({
        success: true,
      });

      const result = await fileRepository.delete(filePath);

      expect(result).toBe(true);
      expect(mockFileSystemService.deleteFile).toHaveBeenCalledWith(filePath);
    });

    it('should return false when delete fails', async () => {
      const filePath = '/test/file.txt';

      mockFileSystemService.deleteFile.mockResolvedValue({
        success: false,
        error: 'Permission denied',
      });

      const result = await fileRepository.delete(filePath);

      expect(result).toBe(false);
    });

    it('should throw error when FileSystemService is not provided', async () => {
      const fileRepoWithoutService = new FileRepository();
      const filePath = '/test/file.txt';

      const result = await fileRepoWithoutService.delete(filePath);

      expect(result).toBe(false);
    });
  });

  describe('findByExtension', () => {
    it('should find files with specific extension', async () => {
      const extension = '.js';
      const directory = '/test';

      // Mock the search method since findByExtension delegates to it
      const searchSpy = vi.spyOn(fileRepository, 'search').mockResolvedValue({
        success: true,
        data: [
          {
            id: '/test/file1.js',
            name: 'file1.js',
            path: '/test/file1.js',
            content: 'content1',
            metadata: {},
          } as FileEntity,
          {
            id: '/test/file2.js',
            name: 'file2.js',
            path: '/test/file2.js',
            content: 'content2',
            metadata: {},
          } as FileEntity,
        ],
        metadata: {},
      });

      const result = await fileRepository.findByExtension(extension, directory);

      expect(result.success).toBe(true);
      expect(result.data).toHaveLength(2);
      expect(searchSpy).toHaveBeenCalledWith(
        { pattern: /\.js$/i },
        directory
      );
    });

    it('should handle extension without leading dot', async () => {
      const extension = 'ts';
      const directory = '/test';

      const searchSpy = vi.spyOn(fileRepository, 'search').mockResolvedValue({
        success: true,
        data: [],
        metadata: {},
      });

      await fileRepository.findByExtension(extension, directory);

      expect(searchSpy).toHaveBeenCalledWith(
        { pattern: /\.ts$/i },
        directory
      );
    });
  });

  describe('getMetadata', () => {
    it('should get file metadata without content', async () => {
      const filePath = '/test/file.txt';

      mockFileOps.exists.mockResolvedValue(true);
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          size: 150,
          modified: new Date('2025-01-01'),
        },
      });

      const result = await fileRepository.getMetadata(filePath);

      expect(result.success).toBe(true);
      expect(result.data).toEqual({
        id: filePath,
        name: 'file.txt',
        path: filePath,
        size: 150,
        lastModified: new Date('2025-01-01'),
      });
      expect(result.data?.content).toBeUndefined();
    });

    it('should return error when file does not exist', async () => {
      const filePath = '/test/nonexistent.txt';

      mockFileOps.exists.mockResolvedValue(false);

      const result = await fileRepository.getMetadata(filePath);

      expect(result.success).toBe(false);
      expect(result.error).toBe('File not found');
    });
  });

  describe('exists', () => {
    it('should return true when file exists', async () => {
      const filePath = '/test/file.txt';

      mockFileOps.exists.mockResolvedValue(true);

      const result = await fileRepository.exists(filePath);

      expect(result).toBe(true);
      expect(mockFileOps.exists).toHaveBeenCalledWith(filePath);
    });

    it('should return false when file does not exist', async () => {
      const filePath = '/test/nonexistent.txt';

      mockFileOps.exists.mockResolvedValue(false);

      const result = await fileRepository.exists(filePath);

      expect(result).toBe(false);
    });
  });

  describe('findById', () => {
    it('should find file by ID (path)', async () => {
      const filePath = '/test/file.txt';

      const findByPathSpy = vi.spyOn(fileRepository, 'findByPath').mockResolvedValue({
        success: true,
        data: {
          id: filePath,
          name: 'file.txt',
          path: filePath,
          content: 'content',
          metadata: {},
        } as FileEntity,
        metadata: {},
      });

      const result = await fileRepository.findById(filePath);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(filePath);
      expect(findByPathSpy).toHaveBeenCalledWith(filePath);
    });

    it('should return null when file not found', async () => {
      const filePath = '/test/nonexistent.txt';

      vi.spyOn(fileRepository, 'findByPath').mockResolvedValue({
        success: false,
        error: 'File not found',
        metadata: {},
      });

      const result = await fileRepository.findById(filePath);

      expect(result).toBeNull();
    });
  });

  describe('save', () => {
    it('should save file entity successfully', async () => {
      const entity: FileEntity = {
        id: '/test/file.txt',
        name: 'file.txt',
        path: '/test/file.txt',
        content: 'test content',
        metadata: {},
      };

      const saveContentSpy = vi.spyOn(fileRepository, 'saveContent').mockResolvedValue({
        success: true,
        data: entity,
        metadata: {},
      });

      const result = await fileRepository.save(entity);

      expect(result).toEqual(entity);
      expect(saveContentSpy).toHaveBeenCalledWith(entity.path, entity.content);
    });

    it('should throw error when entity has no content', async () => {
      const entity: FileEntity = {
        id: '/test/file.txt',
        name: 'file.txt',
        path: '/test/file.txt',
        metadata: {},
      };

      await expect(fileRepository.save(entity)).rejects.toThrow(
        'File entity must have content to save'
      );
    });

    it('should throw error when save fails', async () => {
      const entity: FileEntity = {
        id: '/test/file.txt',
        name: 'file.txt',
        path: '/test/file.txt',
        content: 'test content',
        metadata: {},
      };

      vi.spyOn(fileRepository, 'saveContent').mockResolvedValue({
        success: false,
        error: 'Write failed',
        metadata: {},
      });

      await expect(fileRepository.save(entity)).rejects.toThrow('Write failed');
    });
  });

  describe('getStats', () => {
    it('should return repository statistics', () => {
      const mockStats = { operations: 10, cacheHits: 5 };
      mockFileOps.getStats.mockReturnValue(mockStats);

      const stats = fileRepository.getStats();

      expect(stats).toEqual({
        fileOperations: mockStats,
        config: expect.any(Object),
      });
    });
  });
});
