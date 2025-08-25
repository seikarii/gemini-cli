/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { vi } from 'vitest';
import { CommandContext } from '../ui/commands/types.js';
import { LoadedSettings } from '../config/settings.js';
import { GitService } from '@google/gemini-cli-core';
import { SessionStatsState } from '../ui/contexts/SessionContext.js';

// A utility type to make all properties of an object, and its nested objects, partial.
type DeepPartial<T> = T extends object
  ? {
      [P in keyof T]?: DeepPartial<T[P]>;
    }
  : T;

/**
 * Creates a deep, fully-typed mock of the CommandContext for use in tests.
 * All functions are pre-mocked with `vi.fn()`.
 *
 * @param overrides - A deep partial object to override any default mock values.
 * @returns A complete, mocked CommandContext object.
 */
export const createMockCommandContext = (
  overrides: DeepPartial<CommandContext> = {},
): CommandContext => {
  const defaultMocks: CommandContext = {
    invocation: {
      raw: '',
      name: '',
      args: '',
    },
    services: {
      config: null,
      settings: { merged: {} } as LoadedSettings,
      git: undefined as GitService | undefined,
      import type { Logger } from '@opentelemetry/api-logs';
    ui: {
      addItem: vi.fn(),
      clear: vi.fn(),
      setDebugMessage: vi.fn(),
      pendingItem: null,
      setPendingItem: vi.fn(),
      loadHistory: vi.fn(),
      toggleCorgiMode: vi.fn(),
      toggleVimEnabled: vi.fn(),
       
    } as unknown as CommandContext['ui'],
    session: {
      sessionShellAllowlist: new Set<string>(),
      stats: {
        sessionId: 'test-session',
        sessionStartTime: new Date(),
        lastPromptTokenCount: 0,
        metrics: {
          models: {},
          tools: {
            totalCalls: 0,
            totalSuccess: 0,
            totalFail: 0,
            totalDurationMs: 0,
            totalDecisions: { accept: 0, reject: 0, modify: 0, auto_accept: 0 },
            byName: {},
          },
          files: {
            totalLinesAdded: 0,
            totalLinesRemoved: 0,
          },
        },
      } as SessionStatsState,
    },
  };

   
  const merge = <T extends Record<string, unknown>>(target: T, source: DeepPartial<T>): T => {
    const output = { ...target } as T;

    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        const sourceValue = source[key as keyof DeepPartial<T>];
        const targetValue = output[key as keyof T];

        if (
          // We only want to recursively merge plain objects
          Object.prototype.toString.call(sourceValue) === '[object Object]' &&
          Object.prototype.toString.call(targetValue) === '[object Object]'
        ) {
          // Recursive call needs to maintain the generic type
          output[key as keyof T] = merge(
            targetValue as T[keyof T],
            sourceValue as DeepPartial<T[keyof T]>
          ) as T[keyof T];
        } else {
          // If not, we do a direct assignment. This preserves Date objects and others.
          output[key as keyof T] = sourceValue as T[keyof T];
        }
      }
    }
    return output;
  };

  return merge(defaultMocks, overrides);
};
