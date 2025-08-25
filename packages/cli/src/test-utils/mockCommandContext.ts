/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { CommandContext } from '../ui/commands/types.js';
import { LoadedSettings } from '../config/settings.js';
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
  _overrides: DeepPartial<CommandContext> = {},
): CommandContext => ({
  invocation: { raw: '', name: '', args: '' },
  services: { config: null, settings: { merged: {} } as LoadedSettings, git: undefined },
  ui: {} as unknown as CommandContext['ui'],
  session: { sessionShellAllowlist: new Set<string>(), stats: {} as SessionStatsState },
} as CommandContext);
