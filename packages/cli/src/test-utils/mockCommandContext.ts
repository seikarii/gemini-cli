/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { CommandContext } from '../ui/commands/types.js';
import { LoadedSettings } from '../config/settings.js';
import { SessionStatsState } from '../ui/contexts/SessionContext.js';
import { vi } from 'vitest';

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
  // Minimal, fully-populated defaults with vi.fn() mocks where functions are expected.
  const mockConfig = {
    getDebugMode: () => false,
    getProjectRoot: () => '/test/project',
    getModel: () => 'gemini-pro',
  } as unknown as NonNullable<CommandContext['services']>['config'];

  const mockLogger = ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    log: vi.fn(),
  } as unknown) as NonNullable<CommandContext['services']>['logger'];

  const defaultContext: CommandContext = {
    invocation: { raw: '', name: '', args: '' },
    services: {
      config: mockConfig,
      settings: { merged: {} } as LoadedSettings,
      git: undefined,
      logger: mockLogger,
    },
    ui: {
      addItem: vi.fn(),
      clear: vi.fn(),
      setDebugMessage: vi.fn(),
      pendingItem: null,
      setPendingItem: vi.fn(),
      loadHistory: vi.fn(),
      toggleCorgiMode: vi.fn(),
      toggleVimEnabled: (async () => false) as CommandContext['ui']['toggleVimEnabled'],
      setGeminiMdFileCount: vi.fn(),
      reloadCommands: vi.fn(),
    },
    session: {
      sessionShellAllowlist: new Set<string>(),
      stats: {} as SessionStatsState,
    },
  };

  // Deep merge function for overrides (mutates target). Use Record<string, unknown>
  // to avoid `any` and keep the merge general enough for the overrides object.
  const isObject = (v: unknown): v is Record<string, unknown> =>
    Boolean(v) && typeof v === 'object' && !Array.isArray(v);

  const deepMerge = (
    target: Record<string, unknown>,
    src: Record<string, unknown>,
  ) => {
    if (!isObject(src)) return;
    Object.keys(src).forEach((key) => {
      const val = src[key];
      if (isObject(val)) {
        if (!isObject(target[key])) target[key] = {};
        deepMerge(target[key] as Record<string, unknown>, val);
      } else {
        target[key] = val as unknown;
      }
    });
  };

  deepMerge(defaultContext as unknown as Record<string, unknown>,
    overrides as unknown as Record<string, unknown>);
  return defaultContext;
};
