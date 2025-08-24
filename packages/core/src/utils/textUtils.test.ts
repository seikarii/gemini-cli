/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { isBinary } from './textUtils.js';

describe('isBinary', () => {
  it('returns false for null or undefined', () => {
    expect(isBinary(null)).toBe(false);
  expect(isBinary(undefined as unknown as Buffer)).toBe(false);
  });

  it('returns false for plain ASCII text', () => {
    const buf = Buffer.from('hello world', 'utf8');
    expect(isBinary(buf)).toBe(false);
  });

  it('returns true when buffer contains a NULL byte', () => {
    const buf = Buffer.from([0x48, 0x65, 0x00, 0x6c]); // 'He\0l'
    expect(isBinary(buf)).toBe(true);
  });
});
