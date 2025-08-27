/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="vitest/globals" />

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect } from 'vitest';

// RegExp to detect invalid characters: backspace, and ANSI escape codes
// eslint-disable-next-line no-control-regex
const invalidCharsRegex = /[\b\x1b]/;

function toHaveOnlyValidCharacters(
  this: { isNot?: boolean },
  buffer: unknown,
) {
  const { isNot } = this;
  let pass = true;
  const invalidLines: Array<{ line: number; content: string }> = [];

  // Check if buffer has lines property
  if (!buffer || typeof buffer !== 'object' || !('lines' in buffer)) {
    return {
      pass: false,
      message: () => `Expected buffer to have a 'lines' property`,
      actual: buffer,
      expected: 'An object with a lines property',
    };
  }

  const lines = (buffer as { lines: string[] }).lines;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.includes('\n')) {
      pass = false;
      invalidLines.push({ line: i, content: line });
      break; // Fail fast on newlines
    }
    if (invalidCharsRegex.test(line)) {
      pass = false;
      invalidLines.push({ line: i, content: line });
    }
  }

  return {
    pass,
    message: () =>
      `Expected buffer ${isNot ? 'not ' : ''}to have only valid characters, but found invalid characters in lines:\n${invalidLines
        .map((l) => `  [${l.line}]: "${l.content}"`) /* This line was changed */
        .join('\n')}`,
    actual: lines,
    expected: 'Lines with no line breaks, backspaces, or escape codes.',
  };
}

expect.extend({ toHaveOnlyValidCharacters });

// Extend Vitest's `expect` interface with the custom matcher's type definition.
declare global {
  namespace jest {
    interface Matchers<R> {
      toHaveOnlyValidCharacters(): R;
    }
  }
}
