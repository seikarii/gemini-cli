/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as path from 'path';
import { FileSystemService, FileOperationResult } from '../services/fileSystemService.js';

/**
 * Creates a timestamped backup of a given file.
 * @param filePath The absolute path to the file to back up.
 * @param fileSystemService The FileSystemService instance to use for file operations.
 * @returns A promise that resolves with the path to the created backup file.
 */
export async function createFileBackup(
  filePath: string,
  fileSystemService: FileSystemService
): Promise<string> {
  const timestamp = new Date().toISOString().replace(/[:.-]/g, ''); // YYYYMMDDTHHMMSSsssZ
  const randomSuffix = Math.random().toString(36).substring(2, 8); // Short random string
  const dir = path.dirname(filePath);
  const ext = path.extname(filePath);
  const base = path.basename(filePath, ext);

  const backupFileName = `${base}${ext}.backup_${timestamp}_${randomSuffix}`;
  const backupPath = path.join(dir, backupFileName);

  const result: FileOperationResult = await fileSystemService.copyFile(filePath, backupPath);

  if (!result.success) {
    throw new Error(`Failed to create backup of ${filePath}: ${result.error}`);
  }

  return backupPath;
}
