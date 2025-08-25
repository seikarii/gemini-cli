/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import esbuild from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const pkg = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, 'package.json'), 'utf8'),
);

esbuild
  .build({
    entryPoints: ['packages/cli/index.ts'],
    bundle: true,
    outfile: 'bundle/gemini.js',
    platform: 'node',
    format: 'esm',
    external: [
      '@lydell/node-pty',
      'node-pty',
      '@lydell/node-pty-darwin-arm64',
      '@lydell/node-pty-darwin-x64',
      '@lydell/node-pty-linux-x64',
      '@lydell/node-pty-win32-arm64',
      '@lydell/node-pty-win32-x64',
  // Avoid bundling large or dynamic-require-using libs that can inject
  // helpers (createRequire, dynamic require fallbacks) into the bundle.
  // Externalizing them keeps the runtime behavior native and prevents
  // duplicate identifier / dynamic-require issues.
  'prettier',
  'ink',
  'signal-exit',
  // Externalize google libs that pull in ESM graphs / top-level-await or
  // dynamic-require behaviors. Leaving them external lets Node resolve them
  // natively at runtime and avoids bundling-generated interop wrappers.
  '@google/genai',
  'google-auth-library',
  // File-system search helper used in core; externalize to avoid bundling issues
  'fdir',
    ],
    alias: {
      'is-in-ci': path.resolve(
        __dirname,
        'packages/cli/src/patches/is-in-ci.ts',
      ),
    },
    define: {
      'process.env.CLI_VERSION': JSON.stringify(pkg.version),
    },
  // No banner required for CJS bundle; Node will provide __filename/__dirname
    loader: { '.node': 'file' },
  })
  .catch(() => process.exit(1));
