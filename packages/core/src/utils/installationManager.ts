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
}
