/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FileSystemService } from '../services/fileSystemService.js';
/**
 * Creates a timestamped backup of a given file.
 * @param filePath The absolute path to the file to back up.
 * @param fileSystemService The FileSystemService instance to use for file operations.
 * @returns A promise that resolves with the path to the created backup file.
 */
export declare function createFileBackup(filePath: string, fileSystemService: FileSystemService): Promise<string>;
