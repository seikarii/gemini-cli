/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal global shims to help the TypeScript build during incremental repo fixes.
// These use permissive types on purpose; replace with accurate types as we stabilize tests.

declare const vi: {
  fn: any;
  mock: any;
  clearAllMocks: () => void;
  spyOn: (...args: any[]) => any;
  [key: string]: any;
};

// Provide a permissive namespace with common vitest types used in tests.
// These are intentionally broad (any) to avoid cascading type errors while
// we gradually stabilize the test-suite typings.
declare namespace vi {
  export type Mock = any;
  export type Func = any;
  export type SpyInstance = any;
  export type MockedFunction<T = any> = any;
  export type MockInstance<T extends ((...args: any[]) => any) | undefined = any> = any;
  export type MockContext<T = any> = any;
  export function fn(...args: any[]): any;
  export function spyOn(...args: any[]): any;
  export function clearAllMocks(): void;
  export const mocked: any;
}

declare namespace NodeJS {
  interface Global {
    // tests sometimes attach ad-hoc globals; allow indexing to reduce noise temporarily
    [key: string]: unknown;
  }
}

declare const __DEV__: boolean;
