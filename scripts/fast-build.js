#!/usr/bin/env node

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

// Fast development build script
console.log('🚀 Starting fast development build...\n');

// 1. Skip full typecheck during development
if (process.env.FAST_BUILD === 'true') {
  console.log('⚡ Fast mode: Skipping typecheck and tests...');

  // Build packages in parallel using npm workspaces
  try {
    execSync('npm run build:packages', { stdio: 'inherit', cwd: root });
    console.log('✅ Packages built successfully (fast mode)');
  } catch (error) {
    console.log('⚠️  Package build failed, but continuing...');
  }

  // Try to start the app anyway
  try {
    execSync('npm run start', { stdio: 'inherit', cwd: root });
  } catch (error) {
    console.log('❌ Could not start app. Run full build first.');
  }
} else {
  // Full build with all checks
  console.log('🔍 Running full build with checks...');

  // Check if node_modules exists
  if (!existsSync(join(root, 'node_modules'))) {
    console.log('📦 Installing dependencies...');
    execSync('npm install', { stdio: 'inherit', cwd: root });
  }

  // Generate git info
  execSync('npm run generate', { stdio: 'inherit', cwd: root });

  // Build packages in parallel
  console.log('🏗️  Building packages...');
  execSync('npm run build:packages', { stdio: 'inherit', cwd: root });

  console.log('✅ Build completed successfully!');
}
