/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { UpsertCodeBlockTool } from './upsert_code_block.js';

// Mock fs and ast adapter to avoid real file IO and ts-morph
vi.mock('fs', () => ({
  existsSync: (_p: string) => true,
  readFileSync: (_p: string) => 'const a = 1;',
  writeFileSync: (_p: string, _c: string) => {},
}));
vi.mock('../ast/adapter.js', () => ({
  createProject: () => ({}),
  parseFileWithProject: (_proj: any, _filePath: string) => ({ sourceFile: { getText: () => 'const a = 1;' }, text: 'const a = 1;' }),
  dumpSourceFileText: (_sf: any) => 'const a = 2;'
}));
vi.mock('diff', () => ({ createPatch: (_: any, __: any, ___: any) => 'patch' }));

describe('UpsertCodeBlockTool (unit)', () => {
  it('inserts or updates a simple JS block (preview)', async () => {
    const params = {
      file_path: '/tmp/foo.ts',
      block_name: 'a',
      content: 'const a = 2;',
      block_type: 'variable',
      preview: true,
    } as any;

  const tool = new UpsertCodeBlockTool();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const inv = (tool as unknown as { createInvocation: (p: any) => any }).createInvocation(params);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const abortSignal = (new (globalThis as any).AbortController()).signal;
  const res = await inv.execute(abortSignal);
    expect(res.llmContent).toBeDefined();
    expect(res.returnDisplay).toBeDefined();
  });
});
