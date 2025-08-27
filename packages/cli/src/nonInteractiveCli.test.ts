/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="vitest/globals" />

import {
  Config,
} from '@google/gemini-cli-core';
import { describe, it, beforeEach } from 'vitest';
import { runNonInteractive } from './nonInteractiveCli.js';

// Mock core modules
vi.mock('./ui/hooks/atCommandProcessor.js');
vi.mock('@google/gemini-cli-core', async (importOriginal) => {
  const original =
    await importOriginal<typeof import('@google/gemini-cli-core')>();
  return {
    ...original,
    executeToolCall: vi.fn(),
    shutdownTelemetry: vi.fn(),
  };
});

describe('runNonInteractive', () => {
  let mockConfig: Config;

  beforeEach(async () => {
    mockConfig = {
      getProjectRoot: vi.fn(() => '/test'),
      getExtensions: vi.fn(() => []),
    } as unknown as Config;
  });

  it('should run without errors', async () => {
    // Basic test to ensure the function can be called
    await runNonInteractive(null, mockConfig, 'test input', 'test-id');
    // Test passes if no exception is thrown
  });
});
