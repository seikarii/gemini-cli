/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

import { LSTool } from './ls.js';
import { Config } from '../config/config.js';
import { WorkspaceContext } from '../utils/workspaceContext.js';
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';
import { ToolErrorType } from './tool-error.js';

describe('LSTool', () => {
  let lsTool: LSTool;
  let mockConfig: Config;
  let mockWorkspaceContext: WorkspaceContext;
  let mockFileService: FileDiscoveryService;
  let mockFileSystemService: any;
  const mockPrimaryDir = '/home/user/project';
  const mockSecondaryDir = '/home/user/other-project';
  
  // Spies
  let statSyncSpy: any;
  let readdirSyncSpy: any;

  beforeEach(() => {
    vi.resetAllMocks();

    // Create spies
    statSyncSpy = vi.spyOn(fs, 'statSync');
    readdirSyncSpy = vi.spyOn(fs, 'readdirSync');

    // Mock WorkspaceContext
    mockWorkspaceContext = {
      getDirectories: vi
        .fn()
        .mockReturnValue([mockPrimaryDir, mockSecondaryDir]),
      isPathWithinWorkspace: vi
        .fn()
        .mockImplementation(
          (path) =>
            path.startsWith(mockPrimaryDir) ||
            path.startsWith(mockSecondaryDir),
        ),
      addDirectory: vi.fn(),
    } as unknown as WorkspaceContext;

    // Mock FileService
    mockFileService = {
      shouldGitIgnoreFile: vi.fn().mockReturnValue(false),
      shouldGeminiIgnoreFile: vi.fn().mockReturnValue(false),
    } as unknown as FileDiscoveryService;

    // Mock FileSystemService
    mockFileSystemService = {
      getFileInfo: vi.fn().mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      }),
      listDirectory: vi.fn().mockResolvedValue({
        success: true,
        data: ['file1.ts', 'file2.ts', 'subdir'],
      }),
    };

    // Mock Config
    mockConfig = {
      getTargetDir: vi.fn().mockReturnValue(mockPrimaryDir),
      getWorkspaceContext: vi.fn().mockReturnValue(mockWorkspaceContext),
      getFileService: vi.fn().mockReturnValue(mockFileService),
      getFileFilteringOptions: vi.fn().mockReturnValue({
        respectGitIgnore: true,
        respectGeminiIgnore: true,
      }),
      getFileSystemService: vi.fn().mockReturnValue(mockFileSystemService),
    } as unknown as Config;

    lsTool = new LSTool(mockConfig);
    
    // Create spies for fs functions
    vi.spyOn(fs, 'statSync');
    vi.spyOn(fs, 'readdirSync');
  });

  describe('parameter validation', () => {
    it('should accept valid absolute paths within workspace', () => {
      const params = {
        path: '/home/user/project/src',
      };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      const invocation = lsTool.build(params);
      expect(invocation).toBeDefined();
    });

    it('should reject relative paths', () => {
      const params = {
        path: './src',
      };

      expect(() => lsTool.build(params)).toThrow(
        'Path must be absolute: ./src',
      );
    });

    it('should reject paths outside workspace with clear error message', () => {
      const params = {
        path: '/etc/passwd',
      };

      expect(() => lsTool.build(params)).toThrow(
        'Path must be within one of the workspace directories: /home/user/project, /home/user/other-project',
      );
    });

    it('should accept paths in secondary workspace directory', () => {
      const params = {
        path: '/home/user/other-project/lib',
      };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      const invocation = lsTool.build(params);
      expect(invocation).toBeDefined();
    });
  });

  describe('execute', () => {
    it('should list files in a directory', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['file1.ts', 'file2.ts', 'subdir'];
      const mockStats = {
        isDirectory: vi.fn(),
        mtime: new Date(),
        size: 1024,
      };

      statSyncSpy.mockImplementation((path: string) => {
        const pathStr = path.toString();
        if (pathStr === testPath) {
          return { isDirectory: () => true } as fs.Stats;
        }
        // For individual files
        if (pathStr.toString().endsWith('subdir')) {
          return { ...mockStats, isDirectory: () => true, size: 0 } as fs.Stats;
        }
        return { ...mockStats, isDirectory: () => false } as fs.Stats;
      });

      readdirSyncSpy.mockReturnValuemockFiles;

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('[DIR] subdir');
      expect(result.llmContent).toContain('file1.ts');
      expect(result.llmContent).toContain('file2.ts');
      expect(result.returnDisplay).toBe('Listed 3 item(s).');
    });

    it('should list files from secondary workspace directory', async () => {
      const testPath = '/home/user/other-project/lib';
      const mockFiles = ['module1.js', 'module2.js'];

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 1024,
          modified: new Date(),
        },
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('module1.js');
      expect(result.llmContent).toContain('module2.js');
      expect(result.returnDisplay).toBe('Listed 2 item(s).');
    });

    it('should handle empty directories', async () => {
      const testPath = '/home/user/project/empty';

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory to return empty array
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: [],
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toBe(
        'Directory /home/user/project/empty is empty.',
      );
      expect(result.returnDisplay).toBe('Directory is empty.');
    });

    it('should respect ignore patterns', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['test.js', 'test.spec.js', 'index.js'];

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 1024,
          modified: new Date(),
        },
      });

      const invocation = lsTool.build({
        path: testPath,
        ignore: ['*.spec.js'],
      });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('test.js');
      expect(result.llmContent).toContain('index.js');
      expect(result.llmContent).not.toContain('test.spec.js');
      expect(result.returnDisplay).toBe('Listed 2 item(s).');
    });

    it('should respect gitignore patterns', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['file1.js', 'file2.js', 'ignored.js'];

      // Reset only the mocks we need to change
      mockFileSystemService.getFileInfo.mockReset();
      mockFileSystemService.listDirectory.mockReset();
      vi.mocked(mockFileService).shouldGitIgnoreFile.mockReset();

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 1024,
          modified: new Date(),
        },
      });

      // Mock gitignore to ignore 'ignored.js'
      vi.mocked(mockFileService).shouldGitIgnoreFile.mockImplementation((filePath: string) =>
        filePath.includes('ignored.js')
      );

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('file1.js');
      expect(result.llmContent).toContain('file2.js');
      expect(result.llmContent).not.toContain('ignored.js');
      expect(result.returnDisplay).toBe('Listed 2 item(s). (1 git-ignored)');
    });

    it('should respect geminiignore patterns', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['file1.js', 'file2.js', 'private.js'];

      // Reset only the mocks we need to change
      mockFileSystemService.getFileInfo.mockReset();
      mockFileSystemService.listDirectory.mockReset();
      vi.mocked(mockFileService).shouldGeminiIgnoreFile.mockReset();

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 1024,
          modified: new Date(),
        },
      });

      // Mock geminiignore to ignore 'private.js'
      vi.mocked(mockFileService).shouldGeminiIgnoreFile.mockImplementation((filePath: string) =>
        filePath.includes('private.js')
      );

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('file1.js');
      expect(result.llmContent).toContain('file2.js');
      expect(result.llmContent).not.toContain('private.js');
      expect(result.returnDisplay).toBe('Listed 2 item(s). (1 gemini-ignored)');
    });

    it('should handle non-directory paths', async () => {
      const testPath = '/home/user/project/file.txt';

      // Mock getFileInfo to return that path exists but is not a directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 1024,
          modified: new Date(),
        },
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('Path is not a directory');
      expect(result.returnDisplay).toBe('Error: Path is not a directory.');
      expect(result.error?.type).toBe(ToolErrorType.PATH_IS_NOT_A_DIRECTORY);
    });

    it('should handle non-existent paths', async () => {
      const testPath = '/home/user/project/does-not-exist';

      // Mock getFileInfo to return that path does not exist
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: false,
          isDirectory: false,
          size: 0,
          modified: new Date(),
        },
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('Directory not found or inaccessible');
      expect(result.returnDisplay).toBe('Error: Directory not found or inaccessible.');
      expect(result.error?.type).toBe(ToolErrorType.FILE_NOT_FOUND);
    });

    it('should sort directories first, then files alphabetically', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['z-file.ts', 'a-dir', 'b-file.ts', 'c-dir'];

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files/directories
      mockFileSystemService.getFileInfo.mockImplementation((path: string) => {
        const fileName = path.split('/').pop() || '';
        const isDir = fileName.endsWith('-dir');
        return Promise.resolve({
          success: true,
          data: {
            exists: true,
            isDirectory: isDir,
            size: isDir ? 0 : 1024,
            modified: new Date(),
          },
        });
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      const lines = (
        typeof result.llmContent === 'string' ? result.llmContent : ''
      ).split('\n');
      const entries = lines.slice(1).filter((line: string) => line.trim()); // Skip header
      expect(entries[0]).toBe('[DIR] a-dir');
      expect(entries[1]).toBe('[DIR] c-dir');
      expect(entries[2]).toBe('b-file.ts');
      expect(entries[3]).toBe('z-file.ts');
    });

    it('should handle permission errors gracefully', async () => {
      const testPath = '/home/user/project/restricted';

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory to throw permission error
      mockFileSystemService.listDirectory.mockRejectedValueOnce(
        new Error('EACCES: permission denied')
      );

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('Error listing directory');
      expect(result.llmContent).toContain('permission denied');
      expect(result.returnDisplay).toBe('Error: Failed to list directory.');
      expect(result.error?.type).toBe(ToolErrorType.LS_EXECUTION_ERROR);
    });

    it('should throw for invalid params at build time', async () => {
      expect(() => lsTool.build({ path: '../outside' })).toThrow(
        'Path must be absolute: ../outside',
      );
    });

    it('should handle errors accessing individual files during listing', async () => {
      const testPath = '/home/user/project/src';
      const mockFiles = ['accessible.ts', 'inaccessible.ts'];

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files - fail for inaccessible.ts
      mockFileSystemService.getFileInfo.mockImplementation((path: string) => {
        if (path.includes('inaccessible.ts')) {
          return Promise.reject(new Error('EACCES: permission denied'));
        }
        return Promise.resolve({
          success: true,
          data: {
            exists: true,
            isDirectory: false,
            size: 1024,
            modified: new Date(),
          },
        });
      });

      // Spy on console.error to verify it's called
      const consoleErrorSpy = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {});

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      // Should still list the accessible file
      expect(result.llmContent).toContain('accessible.ts');
      expect(result.llmContent).not.toContain('inaccessible.ts');
      expect(result.returnDisplay).toBe('Listed 1 item(s).');

      // Verify error was logged
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Error accessing'),
      );

      consoleErrorSpy.mockRestore();
    });
  });

  describe('getDescription', () => {
    it('should return shortened relative path', () => {
      const params = {
        path: `${mockPrimaryDir}/deeply/nested/directory`,
      };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      const invocation = lsTool.build(params);
      const description = invocation.getDescription();
      expect(description).toBe(path.join('deeply', 'nested', 'directory'));
    });

    it('should handle paths in secondary workspace', () => {
      const params = {
        path: `${mockSecondaryDir}/lib`,
      };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      const invocation = lsTool.build(params);
      const description = invocation.getDescription();
      expect(description).toBe(path.join('..', 'other-project', 'lib'));
    });
  });

  describe('workspace boundary validation', () => {
    it('should accept paths in primary workspace directory', () => {
      const params = { path: `${mockPrimaryDir}/src` };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      expect(lsTool.build(params)).toBeDefined();
    });

    it('should accept paths in secondary workspace directory', () => {
      const params = { path: `${mockSecondaryDir}/lib` };
      statSyncSpy.mockReturnValue({
        isDirectory: () => true,
      } as fs.Stats);
      expect(lsTool.build(params)).toBeDefined();
    });

    it('should reject paths outside all workspace directories', () => {
      const params = { path: '/etc/passwd' };
      expect(() => lsTool.build(params)).toThrow(
        'Path must be within one of the workspace directories',
      );
    });

    it('should list files from secondary workspace directory', async () => {
      const testPath = `${mockSecondaryDir}/tests`;
      const mockFiles = ['test1.spec.ts', 'test2.spec.ts'];

      // Mock getFileInfo for the directory
      mockFileSystemService.getFileInfo.mockResolvedValueOnce({
        success: true,
        data: {
          exists: true,
          isDirectory: true,
          size: 0,
          modified: new Date(),
        },
      });

      // Mock listDirectory for the specific files
      mockFileSystemService.listDirectory.mockResolvedValueOnce({
        success: true,
        data: mockFiles,
      });

      // Mock getFileInfo for individual files
      mockFileSystemService.getFileInfo.mockResolvedValue({
        success: true,
        data: {
          exists: true,
          isDirectory: false,
          size: 512,
          modified: new Date(),
        },
      });

      const invocation = lsTool.build({ path: testPath });
      const result = await invocation.execute(new AbortController().signal);

      expect(result.llmContent).toContain('test1.spec.ts');
      expect(result.llmContent).toContain('test2.spec.ts');
      expect(result.returnDisplay).toBe('Listed 2 item(s).');
    });
  });
});
