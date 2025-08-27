/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */


// Mock the parser used by the tool to avoid heavy ts-morph dependency in tests
vi.mock('../ast/parser.js', () => ({
    parseSourceToSourceFile: (_filePath: string, text: string) => ({ sourceFile: { getText: () => text } }),
  }));

