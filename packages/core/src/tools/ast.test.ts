/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { ASTReadTool } from './ast.js';

// Mock the parser used by the tool to avoid heavy ts-morph dependency in tests
vi.mock('../ast/parser.js', () => ({
    parseSourceToSourceFile: (filePath: string, text: string) => ({ sourceFile: { getText: () => text } }),
    extractIntentionsFromSourceFile: (sourceFile: any) => ({ intentions: ['fn1', 'fn2'] }),
  }));

describe('ASTReadTool (unit)', () => {
  it('returns extracted intentions from parser', async () => {
    const mockConfig = {
      getFileSystemService: () => ({
        readTextFile: async (filePath: string) => ({ success: true, data: 'function foo() {}' }),
        writeTextFile: async () => ({ success: true }),
      }),
    } as any;

    const tool = new ASTReadTool(mockConfig as any);
    const invocation = (tool as any).createInvocation({ file_path: '/tmp/foo.ts' });
    const res = await invocation.execute();

    expect(res.llmContent).toBeDefined();
    expect(res.llmContent).toContain('intentions');
  });
});
