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

    const mockConfig = {
      getFileSystemService: () => ({
        readTextFile: async (_p: string) => ({ success: true, data: 'const a = 1;' }),
        writeTextFile: async () => ({ success: true }),
      }),
    } as any;

    const tool = new UpsertCodeBlockTool(mockConfig as any);
    const inv = (tool as any).createInvocation(params);
  const res = await inv.execute(new (globalThis as any).AbortController().signal);
    expect(res.llmContent).toBeDefined();
    expect(res.returnDisplay).toBeDefined();
  });
});
