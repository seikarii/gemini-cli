/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

declare module 'vscode' {
  // Minimal ambient typing for build/test environments where the real VS Code
  // API is not available. Keep everything as `any` — tests expect to mock these
  // calls and compile-time types are not important here.
  const _vscode: unknown;
  export = _vscode;
}

declare module 'fdir' {
  export const fdir: unknown;
}

declare module '@testing-library/react' {
  // Provide minimal APIs used by tests. These are all `any` so tests can mock
  // implementations freely.
  export const renderHook: unknown;
  export const render: unknown;
  export const act: unknown;
  export const waitFor: unknown;
  export default {} as unknown;
}
