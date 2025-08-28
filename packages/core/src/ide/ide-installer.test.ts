/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Mock child_process module
vi.mock('child_process', () => ({
  exec: vi.fn((command: string, options?: unknown, callback?: (error: Error | null, stdout: string, stderr: string) => void) => {
    if (typeof options === 'function') {
      callback = options as (error: Error | null, stdout: string, stderr: string) => void;
      options = undefined;
    }
    if (callback) {
      // Call callback immediately with error
      callback(new Error('Command not found'), '', '');
    }
    return {};
  }),
  execSync: vi.fn(() => {
    throw new Error('Command not found');
  }),
}));

// Mock fs module
vi.mock('fs', () => ({
  existsSync: vi.fn(() => false),
  promises: {
    access: vi.fn(() => Promise.reject(new Error('File not accessible'))),
  },
}));

// Mock os module
vi.mock('os', () => ({
  homedir: vi.fn(() => '/home/user'),
}));

// Mock process module
vi.mock('process', () => ({
  platform: 'linux',
}));

import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { getIdeInstaller } from './ide-installer.js';
import { DetectedIde } from './detect-ide.js';

describe('ide-installer', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('getIdeInstaller', () => {
    it('should return a VsCodeInstaller for "vscode"', () => {
      const installer = getIdeInstaller(DetectedIde.VSCode);
      expect(installer).not.toBeNull();
      expect(installer).toBeInstanceOf(Object);
    });
  });

  describe('VsCodeInstaller', () => {
    describe('install', () => {
      it('should return a failure message if VS Code is not installed', async () => {
        const installer = getIdeInstaller(DetectedIde.VSCode)!;
        const result = await installer.install();
        expect(result.success).toBe(false);
        expect(result.message).toContain('VS Code CLI not found');
      });
    });
  });
});
