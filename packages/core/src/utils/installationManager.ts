/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import { promises as fsp } from 'fs';
import { randomUUID } from 'crypto';
import * as path from 'node:path';
import { Storage } from '../config/storage.js';
import { FileSystemService } from '../services/fileSystemService.js';

export class InstallationManager {
  private fileSystemService?: FileSystemService;

  private getInstallationIdPath(): string {
    return Storage.getInstallationIdPath();
  }

  /**
   * Sets the FileSystemService instance for standardized file operations
   * @param service The FileSystemService instance to use
   */
  setFileSystemService(service: FileSystemService): void {
    this.fileSystemService = service;
  }

  private readInstallationIdFromFile(): string | null {
    const installationIdFile = this.getInstallationIdPath();
    try {
      // For synchronous operations, keep using fs for compatibility
      // TODO: Consider providing async versions that use FileSystemService
      if (fs.existsSync(installationIdFile)) {
        const installationid = fs
          .readFileSync(installationIdFile, 'utf-8')
          .trim();
        return installationid || null;
      }
    } catch (error) {
      console.error('Error reading installation ID file:', error);
    }
    return null;
  }

  /**
   * Async version of readInstallationIdFromFile that uses fs.promises for non-blocking file operations.
   * This is the recommended version for new code and performance-critical paths.
   * @returns Promise that resolves to the installation ID string or null if not found
   */
  private async readInstallationIdFromFileAsync(): Promise<string | null> {
    const installationIdFile = this.getInstallationIdPath();
    try {
      if (this.fileSystemService) {
        // Use FileSystemService for standardized file operations
        const fileInfo =
          await this.fileSystemService.getFileInfo(installationIdFile);
        if (fileInfo.success && fileInfo.data?.exists) {
          const readResult =
            await this.fileSystemService.readTextFile(installationIdFile);
          return readResult.success ? readResult.data?.trim() || null : null;
        }
      } else {
        // Fallback to direct fs.promises operations
        try {
          const installationid = await fsp.readFile(
            installationIdFile,
            'utf-8',
          );
          return installationid.trim() || null;
        } catch {
          // File doesn't exist or can't be read
          return null;
        }
      }
    } catch (error) {
      console.error('Error reading installation ID file:', error);
    }
    return null;
  }

  private writeInstallationIdToFile(installationId: string) {
    const installationIdFile = this.getInstallationIdPath();
    const dir = path.dirname(installationIdFile);

    if (this.fileSystemService) {
      // Use FileSystemService for standardized file operations
      void this.fileSystemService
        .createDirectory(dir, { recursive: true })
        .catch(() => {});
      void this.fileSystemService
        .writeTextFile(installationIdFile, installationId)
        .catch(() => {});
    } else {
      // Fallback to direct fs operations for backward compatibility
      void fsp.mkdir(dir, { recursive: true }).catch(() => {});
      void fsp
        .writeFile(installationIdFile, installationId, 'utf-8')
        .catch(() => {});
    }
  }

  /**
   * Retrieves the installation ID from a file, creating it if it doesn't exist.
   * This ID is used for unique user installation tracking.
   * @returns A UUID string for the user.
   */
  getInstallationId(): string {
    try {
      let installationId = this.readInstallationIdFromFile();

      if (!installationId) {
        installationId = randomUUID();
        this.writeInstallationIdToFile(installationId);
      }

      return installationId;
    } catch (error) {
      console.error(
        'Error accessing installation ID file, generating ephemeral ID:',
        error,
      );
      return '123456789';
    }
  }

  /**
   * Async version of getInstallationId that uses fs.promises for non-blocking file operations.
   * This is the recommended version for new code and performance-critical paths.
   * @returns Promise that resolves to a UUID string for the user
   */
  async getInstallationIdAsync(): Promise<string> {
    try {
      let installationId = await this.readInstallationIdFromFileAsync();

      if (!installationId) {
        installationId = randomUUID();
        this.writeInstallationIdToFile(installationId);
      }

      return installationId;
    } catch (error) {
      console.error(
        'Error accessing installation ID file, generating ephemeral ID:',
        error,
      );
      return '123456789';
    }
  }
}
