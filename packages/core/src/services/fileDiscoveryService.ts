/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GitIgnoreParser, GitIgnoreFilter } from '../utils/gitIgnoreParser.js';
import { isGitRepository } from '../utils/gitUtils.js';
import * as path from 'path';

const GEMINI_IGNORE_FILE_NAME = '.geminiignore';

export interface FilterFilesOptions {
  respectGitIgnore?: boolean;
  respectGeminiIgnore?: boolean;
}

export interface FileDiscoveryServiceConfig {
  enableLogging?: boolean;
  strictMode?: boolean;
}

export class FileDiscoveryService {
  private gitIgnoreFilter: GitIgnoreFilter | null = null;
  private geminiIgnoreFilter: GitIgnoreFilter | null = null;
  private projectRoot: string;
  private config: FileDiscoveryServiceConfig;
  private loadErrors: string[] = [];

  constructor(projectRoot: string, config: FileDiscoveryServiceConfig = {}) {
    this.config = {
      enableLogging: true,
      strictMode: false,
      ...config,
    };

    this.projectRoot = path.resolve(projectRoot);
    this.loadIgnoreFiles();
  }

  /**
   * Enhanced ignore file loading with proper error handling and logging
   */
  private loadIgnoreFiles(): void {
    this.loadErrors = [];

    // Load Git ignore patterns if in a Git repository
    this.loadGitIgnore();

    // Load Gemini ignore patterns
    this.loadGeminiIgnore();
  }

  /**
   * Load Git ignore patterns with enhanced error handling
   */
  private loadGitIgnore(): void {
    const isGitRepo = isGitRepository(this.projectRoot);

    if (this.config.enableLogging) {
      console.debug(
        `FileDiscoveryService: Git repository detection: ${isGitRepo ? 'YES' : 'NO'} at ${this.projectRoot}`,
      );
    }

    if (isGitRepo) {
      const parser = new GitIgnoreParser(this.projectRoot);
      try {
        parser.loadGitRepoPatterns();
        this.gitIgnoreFilter = parser;

        if (this.config.enableLogging) {
          const patternCount = parser.getPatterns().length;
          console.debug(
            `FileDiscoveryService: Loaded ${patternCount} Git ignore patterns`,
          );
        }
      } catch (error) {
        const errorMessage = `Failed to load .gitignore patterns: ${error instanceof Error ? error.message : 'Unknown error'}`;
        this.loadErrors.push(errorMessage);

        if (this.config.enableLogging) {
          console.warn(`FileDiscoveryService: ${errorMessage}`);
        }

        if (this.config.strictMode) {
          throw new Error(
            `FileDiscoveryService initialization failed: ${errorMessage}`,
          );
        }
      }
    } else {
      if (this.config.enableLogging) {
        console.debug(
          'FileDiscoveryService: Skipping .gitignore loading (not a Git repository)',
        );
      }
    }
  }

  /**
   * Load Gemini ignore patterns with enhanced error handling
   */
  private loadGeminiIgnore(): void {
    const parser = new GitIgnoreParser(this.projectRoot);
    try {
      parser.loadPatterns(GEMINI_IGNORE_FILE_NAME);
      this.geminiIgnoreFilter = parser;

      if (this.config.enableLogging) {
        const patternCount = parser.getPatterns().length;
        console.debug(
          `FileDiscoveryService: Loaded ${patternCount} Gemini ignore patterns`,
        );
      }
    } catch (error) {
      const errorMessage = `Failed to load .geminiignore patterns: ${error instanceof Error ? error.message : 'Unknown error'}`;
      this.loadErrors.push(errorMessage);

      if (this.config.enableLogging) {
        console.warn(`FileDiscoveryService: ${errorMessage}`);
      }

      if (this.config.strictMode) {
        throw new Error(
          `FileDiscoveryService initialization failed: ${errorMessage}`,
        );
      }
    }
  }

  /**
   * Filters a list of file paths based on git ignore rules with enhanced path validation
   */
  filterFiles(
    filePaths: string[],
    options: FilterFilesOptions = {
      respectGitIgnore: true,
      respectGeminiIgnore: true,
    },
  ): string[] {
    if (!Array.isArray(filePaths)) {
      if (this.config.enableLogging) {
        console.warn(
          'FileDiscoveryService: filterFiles called with non-array input',
        );
      }
      return [];
    }

    return filePaths.filter((filePath) => {
      // Validate and normalize the file path
      const normalizedPath = this.normalizeFilePath(filePath);

      if (
        options.respectGitIgnore &&
        this.shouldGitIgnoreFile(normalizedPath)
      ) {
        return false;
      }
      if (
        options.respectGeminiIgnore &&
        this.shouldGeminiIgnoreFile(normalizedPath)
      ) {
        return false;
      }
      return true;
    });
  }

  /**
   * Normalizes file paths to ensure consistent handling
   */
  private normalizeFilePath(filePath: string): string {
    if (!filePath || typeof filePath !== 'string') {
      return '';
    }

    // Convert to absolute path relative to project root if it's not already absolute
    if (!path.isAbsolute(filePath)) {
      return path.resolve(this.projectRoot, filePath);
    }

    return filePath;
  }

  /**
   * Checks if a single file should be git-ignored with enhanced validation
   */
  shouldGitIgnoreFile(filePath: string): boolean {
    if (!filePath || typeof filePath !== 'string') {
      return false;
    }

    if (!this.gitIgnoreFilter) {
      return false;
    }

    try {
      return this.gitIgnoreFilter.isIgnored(filePath);
    } catch (error) {
      if (this.config.enableLogging) {
        console.warn(
          `FileDiscoveryService: Error checking Git ignore for ${filePath}:`,
          error,
        );
      }
      return false;
    }
  }

  /**
   * Checks if a single file should be gemini-ignored with enhanced validation
   */
  shouldGeminiIgnoreFile(filePath: string): boolean {
    if (!filePath || typeof filePath !== 'string') {
      return false;
    }

    if (!this.geminiIgnoreFilter) {
      return false;
    }

    try {
      return this.geminiIgnoreFilter.isIgnored(filePath);
    } catch (error) {
      if (this.config.enableLogging) {
        console.warn(
          `FileDiscoveryService: Error checking Gemini ignore for ${filePath}:`,
          error,
        );
      }
      return false;
    }
  }

  /**
   * Unified method to check if a file should be ignored based on filtering options
   */
  shouldIgnoreFile(
    filePath: string,
    options: FilterFilesOptions = {},
  ): boolean {
    if (!filePath || typeof filePath !== 'string') {
      return false;
    }

    const { respectGitIgnore = true, respectGeminiIgnore = true } = options;

    if (respectGitIgnore && this.shouldGitIgnoreFile(filePath)) {
      return true;
    }
    if (respectGeminiIgnore && this.shouldGeminiIgnoreFile(filePath)) {
      return true;
    }
    return false;
  }

  /**
   * Returns loaded patterns from .geminiignore
   */
  getGeminiIgnorePatterns(): string[] {
    return this.geminiIgnoreFilter?.getPatterns() ?? [];
  }

  /**
   * Returns loaded patterns from .gitignore
   */
  getGitIgnorePatterns(): string[] {
    return this.gitIgnoreFilter?.getPatterns() ?? [];
  }

  /**
   * Returns any errors that occurred during loading
   */
  getLoadErrors(): string[] {
    return [...this.loadErrors];
  }

  /**
   * Returns diagnostic information about the service state
   */
  getDiagnostics(): {
    projectRoot: string;
    isGitRepository: boolean;
    gitIgnoreLoaded: boolean;
    geminiIgnoreLoaded: boolean;
    gitIgnorePatternCount: number;
    geminiIgnorePatternCount: number;
    loadErrors: string[];
  } {
    return {
      projectRoot: this.projectRoot,
      isGitRepository: isGitRepository(this.projectRoot),
      gitIgnoreLoaded: this.gitIgnoreFilter !== null,
      geminiIgnoreLoaded: this.geminiIgnoreFilter !== null,
      gitIgnorePatternCount: this.gitIgnoreFilter?.getPatterns().length ?? 0,
      geminiIgnorePatternCount:
        this.geminiIgnoreFilter?.getPatterns().length ?? 0,
      loadErrors: [...this.loadErrors],
    };
  }

  /**
   * Reloads ignore files - useful for testing or when files change
   */
  reloadIgnoreFiles(): void {
    if (this.config.enableLogging) {
      console.debug('FileDiscoveryService: Reloading ignore files');
    }
    this.loadIgnoreFiles();
  }
}
