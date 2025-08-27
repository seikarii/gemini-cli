/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { UpsertCodeBlockTool } from './upsert_code_block.js';
import { type Config } from '../config/config.js';

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
  let mockConfig: Config;

  beforeEach(() => {
    mockConfig = {
      getCoreTools: vi.fn().mockReturnValue([]),
      getExcludeTools: vi.fn().mockReturnValue([]),
      getDebugMode: vi.fn().mockReturnValue(false),
      getTargetDir: vi.fn().mockReturnValue('/test/dir'),
      getSummarizeToolOutputConfig: vi.fn().mockReturnValue(undefined),
      getWorkspaceContext: vi.fn().mockReturnValue({}),
      getGeminiClient: vi.fn(),
      getShouldUseNodePtyShell: vi.fn().mockReturnValue(false),
    } as unknown as Config;
  });

  it('inserts or updates a simple JS block (preview)', async () => {
    const params = {
      file_path: '/tmp/foo.ts',
      block_name: 'a',
      content: 'const a = 2;',
      block_type: 'variable',
      preview: true,
    } as any;

  const tool = new UpsertCodeBlockTool(mockConfig);
   
  const inv = (tool as unknown as { createInvocation: (p: any) => any }).createInvocation(params);
   
  const abortSignal = (new (globalThis as any).AbortController()).signal;
  const res = await inv.execute(abortSignal);
    expect(res.llmContent).toBeDefined();
    expect(res.returnDisplay).toBeDefined();
  });
});
