/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { applyReplacement } from '../edit.js';

const mockFileContent = Array.from({ length: 20 }).map(() => 'HOLA MUNDO').join('\n');

describe('replace occurrence handling', () => {
  it('applyReplacement replaces only the 7th occurrence when requested', () => {
    const result = applyReplacement(mockFileContent, 'HOLA MUNDO', 'LINEA 7 MODIFICADA', false, 7);
    // Count occurrences of the new string
    const newCount = (result.split('LINEA 7 MODIFICADA').length - 1);
    expect(newCount).toBe(1);
    // Ensure total original occurrences were 20
    expect((mockFileContent.split('HOLA MUNDO').length - 1)).toBe(20);
  });
});
