/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import { promises as fsp, readFileSync } from 'node:fs';
import { Storage } from '../config/storage.js';
import { FileSystemService } from '../services/fileSystemService.js';

interface UserAccounts {
  active: string | null;
  old: string[];
}

export class UserAccountManager {
  private fileSystemService?: FileSystemService;

  private getGoogleAccountsCachePath(): string {
    return Storage.getGoogleAccountsPath();
  }

  /**
   * Sets the FileSystemService instance for standardized file operations
   * @param service The FileSystemService instance to use
   */
  setFileSystemService(service: FileSystemService): void {
    this.fileSystemService = service;
  }

  /**
   * Parses and validates the string content of an accounts file.
   * @param content The raw string content from the file.
   * @returns A valid UserAccounts object.
   */
  private parseAndValidateAccounts(content: string): UserAccounts {
    const defaultState = { active: null, old: [] };
    if (!content.trim()) {
      return defaultState;
    }

    const parsed = JSON.parse(content);

    // Inlined validation logic
    if (typeof parsed !== 'object' || parsed === null) {
      console.error('Invalid accounts file schema, starting fresh.');
      return defaultState;
    }
    const { active, old } = parsed as Partial<UserAccounts>;
    const isValid =
      (active === undefined || active === null || typeof active === 'string') &&
      (old === undefined ||
        (Array.isArray(old) && old.every((i) => typeof i === 'string')));

    if (!isValid) {
      console.log('Invalid accounts file schema, starting fresh.');
      return defaultState;
    }

    return {
      active: parsed.active ?? null,
      old: parsed.old ?? [],
    };
  }

  private readAccountsSync(filePath: string): UserAccounts {
    const defaultState = { active: null, old: [] };
    try {
      // For synchronous operations, keep using fs for compatibility
      // TODO: Consider providing async versions that use FileSystemService
      const content = readFileSync(filePath, 'utf-8');
      return this.parseAndValidateAccounts(content);
    } catch (error) {
      if (
        error instanceof Error &&
        'code' in error &&
        error.code === 'ENOENT'
      ) {
        return defaultState;
      }
      console.log('Error during sync read of accounts, starting fresh.', error);
      return defaultState;
    }
  }

  private async readAccounts(filePath: string): Promise<UserAccounts> {
    const defaultState = { active: null, old: [] };
    try {
      let content: string;
      if (this.fileSystemService) {
        // Use FileSystemService for standardized file operations
        const readResult = await this.fileSystemService.readTextFile(filePath);
        if (!readResult.success || !readResult.data) {
          throw new Error(`Failed to read file ${filePath}: ${readResult.error || 'No data returned'}`);
        }
        content = readResult.data;
      } else {
        // Fallback to direct fs operations for backward compatibility
        content = await fsp.readFile(filePath, 'utf-8');
      }
      return this.parseAndValidateAccounts(content);
    } catch (error) {
      if (
        error instanceof Error &&
        'code' in error &&
        error.code === 'ENOENT'
      ) {
        return defaultState;
      }
      console.log('Could not parse accounts file, starting fresh.', error);
      return defaultState;
    }
  }

  async cacheGoogleAccount(email: string): Promise<void> {
    const filePath = this.getGoogleAccountsCachePath();

    if (this.fileSystemService) {
      // Use FileSystemService for standardized file operations
      await this.fileSystemService.createDirectory(path.dirname(filePath), { recursive: true });
    } else {
      // Fallback to direct fs operations for backward compatibility
      await fsp.mkdir(path.dirname(filePath), { recursive: true });
    }

    const accounts = await this.readAccounts(filePath);

    if (accounts.active && accounts.active !== email) {
      if (!accounts.old.includes(accounts.active)) {
        accounts.old.push(accounts.active);
      }
    }

    // If the new email was in the old list, remove it
    accounts.old = accounts.old.filter((oldEmail) => oldEmail !== email);

    accounts.active = email;

    if (this.fileSystemService) {
      // Use FileSystemService for standardized file operations
      const writeResult = await this.fileSystemService.writeTextFile(filePath, JSON.stringify(accounts, null, 2));
      if (!writeResult.success) {
        throw new Error(`Failed to write accounts file ${filePath}: ${writeResult.error}`);
      }
    } else {
      // Fallback to direct fs operations for backward compatibility
      await fsp.writeFile(filePath, JSON.stringify(accounts, null, 2), 'utf-8');
    }
  }

  getCachedGoogleAccount(): string | null {
    const filePath = this.getGoogleAccountsCachePath();
    const accounts = this.readAccountsSync(filePath);
    return accounts.active;
  }

  getLifetimeGoogleAccounts(): number {
    const filePath = this.getGoogleAccountsCachePath();
    const accounts = this.readAccountsSync(filePath);
    const allAccounts = new Set(accounts.old);
    if (accounts.active) {
      allAccounts.add(accounts.active);
    }
    return allAccounts.size;
  }

  async clearCachedGoogleAccount(): Promise<void> {
    const filePath = this.getGoogleAccountsCachePath();
    const accounts = await this.readAccounts(filePath);

    if (accounts.active) {
      if (!accounts.old.includes(accounts.active)) {
        accounts.old.push(accounts.active);
      }
      accounts.active = null;
    }

    if (this.fileSystemService) {
      // Use FileSystemService for standardized file operations
      const writeResult = await this.fileSystemService.writeTextFile(filePath, JSON.stringify(accounts, null, 2));
      if (!writeResult.success) {
        throw new Error(`Failed to write accounts file ${filePath}: ${writeResult.error}`);
      }
    } else {
      // Fallback to direct fs operations for backward compatibility
      await fsp.writeFile(filePath, JSON.stringify(accounts, null, 2), 'utf-8');
    }
  }
}
