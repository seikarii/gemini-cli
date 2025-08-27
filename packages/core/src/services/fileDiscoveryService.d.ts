/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export interface FilterFilesOptions {
    respectGitIgnore?: boolean;
    respectGeminiIgnore?: boolean;
}
export interface FileDiscoveryServiceConfig {
    enableLogging?: boolean;
    strictMode?: boolean;
}
export declare class FileDiscoveryService {
    private gitIgnoreFilter;
    private geminiIgnoreFilter;
    private projectRoot;
    private config;
    private loadErrors;
    constructor(projectRoot: string, config?: FileDiscoveryServiceConfig);
    /**
     * Enhanced ignore file loading with proper error handling and logging
     */
    private loadIgnoreFiles;
    /**
     * Load Git ignore patterns with enhanced error handling
     */
    private loadGitIgnore;
    /**
     * Load Gemini ignore patterns with enhanced error handling
     */
    private loadGeminiIgnore;
    /**
     * Filters a list of file paths based on git ignore rules with enhanced path validation
     */
    filterFiles(filePaths: string[], options?: FilterFilesOptions): string[];
    /**
     * Normalizes file paths to ensure consistent handling
     */
    private normalizeFilePath;
    /**
     * Checks if a single file should be git-ignored with enhanced validation
     */
    shouldGitIgnoreFile(filePath: string): boolean;
    /**
     * Checks if a single file should be gemini-ignored with enhanced validation
     */
    shouldGeminiIgnoreFile(filePath: string): boolean;
    /**
     * Unified method to check if a file should be ignored based on filtering options
     */
    shouldIgnoreFile(filePath: string, options?: FilterFilesOptions): boolean;
    /**
     * Returns loaded patterns from .geminiignore
     */
    getGeminiIgnorePatterns(): string[];
    /**
     * Returns loaded patterns from .gitignore
     */
    getGitIgnorePatterns(): string[];
    /**
     * Returns any errors that occurred during loading
     */
    getLoadErrors(): string[];
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
    };
    /**
     * Reloads ignore files - useful for testing or when files change
     */
    reloadIgnoreFiles(): void;
}
