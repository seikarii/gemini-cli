/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { ToolError } from './tool-error.js';
import { FileSystemService } from '../services/fileSystemService.js';
/**
 * Validation result interface
 */
export interface ValidationResult {
    isValid: boolean;
    error?: ToolError;
}
/**
 * Pre-execution validation utilities
 */
export declare class ToolValidationUtils {
    private fileSystemService;
    constructor(fileSystemService: FileSystemService);
    /**
     * Validate file existence and accessibility
     */
    validateFileExists(filePath: string): Promise<ValidationResult>;
    /**
     * Validate directory existence and accessibility
     */
    validateDirectoryExists(dirPath: string): Promise<ValidationResult>;
    /**
     * Validate path accessibility (read/write permissions)
     */
    validatePathAccessibility(filePath: string, requireWrite?: boolean): Promise<ValidationResult>;
    /**
     * Validate that old_string exists in file content (for edit operations)
     */
    validateOldStringExists(filePath: string, oldString: string, expectedOccurrences?: number): Promise<ValidationResult>;
    /**
     * Validate content for write operations
     */
    validateWriteContent(content: string): ValidationResult;
    /**
     * Validate shell command syntax and binary existence
     */
    validateShellCommand(command: string): ValidationResult;
    /**
     * Validate edit range parameters
     */
    validateEditRange(content: string, startLine: number, startColumn: number, endLine: number, endColumn: number): ValidationResult;
    /**
     * Perform a dry run validation for edit operations (simulates the replacement)
     */
    validateEditDryRun(filePath: string, oldString: string, newString: string, expectedOccurrences?: number): Promise<ValidationResult>;
    /**
     * Escape special regex characters
     */
    private escapeRegExp;
    /**
     * Count occurrences of a substring in content
     */
    private countOccurrences;
}
