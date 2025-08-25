/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
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
export declare class LocalStorageBackend implements StorageBackend {
    /**
     * Writes data to a file on the local disk, ensuring the directory exists.
     * Implements an atomic write to prevent data corruption during write.
     * @param filePath The absolute path to the file.
     * @param data The string data to write.
     */
    write(filePath: string, data: string): Promise<void>;
    /**
     * Reads data from a file on the local disk.
     * @param filePath The absolute path to the file.
     * @returns The content of the file as a string.
     */
    read(filePath: string): Promise<string>;
    /**
     * Checks if a file exists on the local disk.
     * @param filePath The absolute path to the file.
     * @returns True if the file exists, false otherwise.
     */
    exists(filePath: string): Promise<boolean>;
}
