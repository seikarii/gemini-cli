/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import { promises as fsp } from 'fs';
import * as path from 'path';

/**
 * Checks if a directory is within a git repository
 * @param directory The directory to check
 * @returns true if the directory is in a git repository, false otherwise
 */
export function isGitRepository(directory: string): boolean {
  try {
    let currentDir = path.resolve(directory);

    while (true) {
      const gitDir = path.join(currentDir, '.git');

      // Check if .git exists (either as directory or file for worktrees)
      if (fs.existsSync(gitDir)) {
        return true;
      }

      const parentDir = path.dirname(currentDir);

      // If we've reached the root directory, stop searching
      if (parentDir === currentDir) {
        break;
      }

      currentDir = parentDir;
    }

    return false;
  } catch (_error) {
    // If any filesystem error occurs, assume not a git repo
    return false;
  }
}

/**
 * Finds the root directory of a git repository
 * @param directory Starting directory to search from
 * @returns The git repository root path, or null if not in a git repository
 */
export function findGitRoot(directory: string): string | null {
  try {
    let currentDir = path.resolve(directory);

    while (true) {
      const gitDir = path.join(currentDir, '.git');

      if (fs.existsSync(gitDir)) {
        return currentDir;
      }

      const parentDir = path.dirname(currentDir);

      if (parentDir === currentDir) {
        break;
      }

      currentDir = parentDir;
    }

    return null;
  } catch (_error) {
    return null;
  }
}

/**
 * Async version of isGitRepository that uses fs.promises for non-blocking file operations.
 * Recommended for use in performance-critical paths and when blocking the event loop is undesirable.
 * @param directory The directory to check
 * @returns Promise that resolves to true if the directory is in a git repository, false otherwise
 */
export async function isGitRepositoryAsync(directory: string): Promise<boolean> {
  try {
    let currentDir = path.resolve(directory);

    while (true) {
      const gitDir = path.join(currentDir, '.git');

      // Check if .git exists (either as directory or file for worktrees)
      try {
        await fsp.access(gitDir);
        return true;
      } catch {
        // .git doesn't exist, continue searching up the directory tree
      }

      const parentDir = path.dirname(currentDir);

      // If we've reached the root directory, stop searching
      if (parentDir === currentDir) {
        break;
      }

      currentDir = parentDir;
    }

    return false;
  } catch (_error) {
    // If any filesystem error occurs, assume not a git repo
    return false;
  }
}

/**
 * Async version of findGitRoot that uses fs.promises for non-blocking file operations.
 * Recommended for use in performance-critical paths and when blocking the event loop is undesirable.
 * @param directory Starting directory to search from
 * @returns Promise that resolves to the git repository root path, or null if not in a git repository
 */
export async function findGitRootAsync(directory: string): Promise<string | null> {
  try {
    let currentDir = path.resolve(directory);

    while (true) {
      const gitDir = path.join(currentDir, '.git');

      try {
        await fsp.access(gitDir);
        return currentDir;
      } catch {
        // .git doesn't exist, continue searching up the directory tree
      }

      const parentDir = path.dirname(currentDir);

      if (parentDir === currentDir) {
        break;
      }

      currentDir = parentDir;
    }

    return null;
  } catch (_error) {
    return null;
  }
}
