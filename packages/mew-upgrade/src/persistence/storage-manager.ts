/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file Implements the StorageManager, responsible for physical I/O.
 * This layer abstracts the storage medium (e.g., local disk, cloud storage).
 */

// Node.js file system and path modules will be required.
import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Interface for a storage backend, allowing for different storage strategies.
 */
export interface StorageBackend {
  write(filePath: string, data: string): Promise<void>;
  read(filePath: string): Promise<string>;
  exists(filePath: string): Promise<boolean>;
}

/**
 * A storage backend that interacts with the local filesystem.
 */
export class LocalStorageBackend implements StorageBackend {
  /**
   * Writes data to a file on the local disk, ensuring the directory exists.
   * Implements an atomic write to prevent data corruption during write.
   * @param filePath The absolute path to the file.
   * @param data The string data to write.
   */
  async write(filePath: string, data: string): Promise<void> {
    try {
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });

      // Atomic write: write to a temporary file then rename.
      const tempPath = filePath + '.tmp' + Date.now();
      await fs.writeFile(tempPath, data, 'utf-8');
      await fs.rename(tempPath, filePath);
    } catch (error) {
      console.error(
        `LocalStorageBackend: Failed to write to ${filePath}`,
        error,
      );
      throw new Error(`Failed to write file: ${filePath}`);
    }
  }

  /**
   * Reads data from a file on the local disk.
   * @param filePath The absolute path to the file.
   * @returns The content of the file as a string.
   */
  async read(filePath: string): Promise<string> {
    try {
      return await fs.readFile(filePath, 'utf-8');
    } catch (error) {
      console.error(
        `LocalStorageBackend: Failed to read from ${filePath}`,
        error,
      );
      throw new Error(`Failed to read file: ${filePath}`);
    }
  }

  /**
   * Checks if a file exists on the local disk.
   * @param filePath The absolute path to the file.
   * @returns True if the file exists, false otherwise.
   */
  async exists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}
