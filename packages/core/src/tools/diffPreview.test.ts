/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { generateUnifiedDiff, type DiffOptions } from './diffPreview.js';

describe('generateUnifiedDiff', () => {
  it('should generate a unified diff for simple text changes', () => {
    const oldText = 'const a = 1;\nconst b = 2;';
    const newText = 'const a = 1;\nconst b = 3;';
    const filePath = '/test/file.ts';

    const diff = generateUnifiedDiff(oldText, newText, filePath);

    expect(diff).toContain('--- /test/file.ts');
    expect(diff).toContain('+++ /test/file.ts (modified)');
    expect(diff).toContain('-2');
    expect(diff).toContain('+3');
  });

  it('should handle identical content', () => {
    const text = 'const a = 1;\nconst b = 2;';
    const filePath = '/test/file.ts';

    const diff = generateUnifiedDiff(text, text, filePath);

    expect(diff).toContain('--- /test/file.ts');
    expect(diff).toContain('+++ /test/file.ts (modified)');
    expect(diff).toContain('(No changes)');
  });

  it('should handle empty old text', () => {
    const oldText = '';
    const newText = 'const a = 1;';
    const filePath = '/test/file.ts';

    const diff = generateUnifiedDiff(oldText, newText, filePath);

    expect(diff).toContain('--- /test/file.ts');
    expect(diff).toContain('+++ /test/file.ts (modified)');
    expect(diff).toContain('const a = 1');
  });

  it('should handle empty new text', () => {
    const oldText = 'const a = 1;';
    const newText = '';
    const filePath = '/test/file.ts';

    const diff = generateUnifiedDiff(oldText, newText, filePath);

    expect(diff).toContain('--- /test/file.ts');
    expect(diff).toContain('+++ /test/file.ts (modified)');
    expect(diff).toContain('const a = 1');
  });

  it('should accept custom options', () => {
    const oldText = 'line 1\nline 2\nline 3';
    const newText = 'line 1\nmodified line 2\nline 3';
    const filePath = '/test/file.ts';

    const options: DiffOptions = {
      timeout: 2000,
      cleanupSemantic: false,
      cleanupEfficiency: false,
      editCost: 2,
    };

    const diff = generateUnifiedDiff(oldText, newText, filePath, options);

    expect(diff).toContain('--- /test/file.ts');
    expect(diff).toContain('+++ /test/file.ts (modified)');
    // The diff should still be generated even with different options
    expect(diff.length).toBeGreaterThan(0);
  });

  it('should handle very long content gracefully', () => {
    const oldText = 'x'.repeat(10000);
    const newText = 'y'.repeat(10000);
    const filePath = '/test/large-file.txt';

    const diff = generateUnifiedDiff(oldText, newText, filePath);

    expect(diff).toContain('--- /test/large-file.txt');
    expect(diff).toContain('+++ /test/large-file.txt (modified)');
    // Should generate some form of diff even for large files
    expect(diff.length).toBeGreaterThan(0);
  });
});
