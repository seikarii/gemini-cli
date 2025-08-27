/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FileSystemService } from '../services/fileSystemService.js';
export declare class UserAccountManager {
    private fileSystemService?;
    private getGoogleAccountsCachePath;
    /**
     * Sets the FileSystemService instance for standardized file operations
     * @param service The FileSystemService instance to use
     */
    setFileSystemService(service: FileSystemService): void;
    /**
     * Parses and validates the string content of an accounts file.
     * @param content The raw string content from the file.
     * @returns A valid UserAccounts object.
     */
    private parseAndValidateAccounts;
    private readAccountsSync;
    private readAccounts;
    cacheGoogleAccount(email: string): Promise<void>;
    getCachedGoogleAccount(): string | null;
    getLifetimeGoogleAccounts(): number;
    clearCachedGoogleAccount(): Promise<void>;
}
