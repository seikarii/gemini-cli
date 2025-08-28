/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, it, describe, vi, beforeEach, afterEach } from 'vitest';
import { ChatRecordingService } from './chatRecordingService.js';
import { Config } from '../config/config.js';

// Define a stub interface for FileSystemAdapter
interface StubFileSystemAdapter {
  calls: Array<{ method: string; args: unknown[] }>;
  readFile(filePath: string): Promise<string>;
  writeFile(filePath: string, data: string): Promise<void>;
  mkdir(dirPath: string): Promise<void>;
  unlink(filePath: string): Promise<void>;
  exists(filePath: string): Promise<boolean>;
}

describe('ChatRecordingService additions', () => {
  let svc: ChatRecordingService;
  let mockConfig: Config;

  beforeEach(() => {
    mockConfig = {
      getSessionId: vi.fn().mockReturnValue('test-session-id'),
      getProjectRoot: vi.fn().mockReturnValue('/test/project/root'),
      storage: {
        getProjectTempDir: vi
          .fn()
          .mockReturnValue('/test/project/root/.gemini/tmp'),
      },
      getModel: vi.fn().mockReturnValue('gemini-pro'),
      getDebugMode: vi.fn().mockReturnValue(false),
      getChatCompression: vi.fn().mockReturnValue(undefined),
    } as unknown as Config;

    // Create stub file system adapter
    const stubFs = {
      calls: [] as Array<{ method: string; args: unknown[] }>,
      async readFile(filePath: string) {
        this.calls.push({ method: 'readFile', args: [filePath] });
        return JSON.stringify({
          sessionId: 'test-session-id',
          projectHash: 'test-project-hash',
          messages: [
            {
              id: '1',
              type: 'user',
              content: 'Hello',
              timestamp: new Date().toISOString(),
            },
            {
              id: '2',
              type: 'gemini',
              content: 'Response',
              timestamp: new Date().toISOString(),
            },
          ],
        });
      },
      async writeFile(filePath: string, data: string) {
        this.calls.push({ method: 'writeFile', args: [filePath, data] });
      },
      async mkdir(dirPath: string) {
        this.calls.push({ method: 'mkdir', args: [dirPath] });
      },
      async unlink(filePath: string) {
        this.calls.push({ method: 'unlink', args: [filePath] });
      },
      async exists(filePath: string) {
        this.calls.push({ method: 'exists', args: [filePath] });
        return true;
      },
    };

    svc = new ChatRecordingService(mockConfig, stubFs as StubFileSystemAdapter);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('getOptimizedHistoryForPrompt', () => {
    it('should return optimized history with Content[] format', async () => {
      const result = await svc.getOptimizedHistoryForPrompt(2000, true);
      expect(result).toHaveProperty('history');
      expect(Array.isArray(result.history)).toBe(true);
      expect(result.metaInfo).toHaveProperty('totalTokens');
      expect(result.metaInfo).toHaveProperty('originalMessageCount');
      expect(result.metaInfo.finalMessageCount).toBeGreaterThanOrEqual(0);
      expect(typeof result.metaInfo.compressionApplied).toBe('boolean');
    });

    it('should respect token budget parameter', async () => {
      const result = await svc.getOptimizedHistoryForPrompt(1000, false);
      expect(result.metaInfo.totalTokens).toBeLessThanOrEqual(1000);
    });
  });
});
