/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Tool validation utilities for pre-execution parameter validation
 */

import { execSync, exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { ToolError, ToolErrorType } from './tool-error.js';
import { FileSystemService } from '../services/fileSystemService.js';

// Create promisified version of exec for async operations
const execAsync = promisify(exec);

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
export class ToolValidationUtils {
  constructor(private fileSystemService: FileSystemService) {}

  /**
   * Validate file existence and accessibility
   */
  async validateFileExists(filePath: string): Promise<ValidationResult> {
    try {
      const exists = await this.fileSystemService.exists(filePath);
      if (!exists) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.VALIDATION_FILE_NOT_EXISTS,
            `File does not exist: ${filePath}`,
            filePath,
          ),
        };
      }

      const fileInfo = await this.fileSystemService.getFileInfo(filePath);
      if (!fileInfo.success || !fileInfo.data!.isFile) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.TARGET_IS_DIRECTORY,
            `Path is a directory, not a file: ${filePath}`,
            filePath,
          ),
        };
      }

      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: ToolError.fromUnknownError(
          error,
          ToolErrorType.VALIDATION_FILE_NOT_EXISTS,
        ),
      };
    }
  }

  /**
   * Validate directory existence and accessibility
   */
  async validateDirectoryExists(dirPath: string): Promise<ValidationResult> {
    try {
      const exists = await this.fileSystemService.exists(dirPath);
      if (!exists) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.SEARCH_PATH_NOT_FOUND,
            `Directory does not exist: ${dirPath}`,
            dirPath,
          ),
        };
      }

      const dirInfo = await this.fileSystemService.getFileInfo(dirPath);
      if (!dirInfo.success || !dirInfo.data!.isDirectory) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.PATH_IS_NOT_A_DIRECTORY,
            `Path is not a directory: ${dirPath}`,
            dirPath,
          ),
        };
      }

      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: ToolError.fromUnknownError(
          error,
          ToolErrorType.SEARCH_PATH_NOT_FOUND,
        ),
      };
    }
  }

  /**
   * Validate path accessibility (read/write permissions)
   */
  async validatePathAccessibility(
    filePath: string,
    requireWrite: boolean = false,
  ): Promise<ValidationResult> {
    try {
      const exists = await this.fileSystemService.exists(filePath);
      if (!exists) {
        if (requireWrite) {
          // For write operations, if file doesn't exist, check parent directory
          const parentDir = path.dirname(filePath);
          const parentExists = await this.fileSystemService.exists(parentDir);
          if (!parentExists) {
            return {
              isValid: false,
              error: ToolError.fileSystemError(
                ToolErrorType.VALIDATION_PATH_NOT_ACCESSIBLE,
                `Parent directory does not exist: ${parentDir}`,
                filePath,
              ),
            };
          }

          const parentInfo =
            await this.fileSystemService.getFileInfo(parentDir);
          if (!parentInfo.success) {
            return {
              isValid: false,
              error: ToolError.fileSystemError(
                ToolErrorType.VALIDATION_PATH_NOT_ACCESSIBLE,
                `Cannot get parent directory information: ${parentDir}`,
                filePath,
              ),
            };
          }

          if (!parentInfo.data!.permissions.writable) {
            return {
              isValid: false,
              error: ToolError.fileSystemError(
                ToolErrorType.PERMISSION_DENIED,
                `No write permission for parent directory: ${parentDir}`,
                filePath,
              ),
            };
          }

          return { isValid: true };
        } else {
          // For read operations, file must exist
          return {
            isValid: false,
            error: ToolError.fileSystemError(
              ToolErrorType.VALIDATION_PATH_NOT_ACCESSIBLE,
              `Path is not accessible: ${filePath}`,
              filePath,
            ),
          };
        }
      }

      const fileInfo = await this.fileSystemService.getFileInfo(filePath);
      if (!fileInfo.success) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.VALIDATION_PATH_NOT_ACCESSIBLE,
            `Cannot get file information: ${filePath}`,
            filePath,
          ),
        };
      }

      const permissions = fileInfo.data!.permissions;
      if (!permissions.readable) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.PERMISSION_DENIED,
            `No read permission for path: ${filePath}`,
            filePath,
          ),
        };
      }

      if (requireWrite && !permissions.writable) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.PERMISSION_DENIED,
            `No write permission for path: ${filePath}`,
            filePath,
          ),
        };
      }

      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: ToolError.fromUnknownError(
          error,
          ToolErrorType.VALIDATION_PATH_NOT_ACCESSIBLE,
        ),
      };
    }
  }

  /**
   * Validate that old_string exists in file content (for edit operations)
   */
  async validateOldStringExists(
    filePath: string,
    oldString: string,
    expectedOccurrences?: number,
  ): Promise<ValidationResult> {
    try {
      const readResult = await this.fileSystemService.readTextFile(filePath);
      if (!readResult.success) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.FILE_NOT_FOUND,
            `Failed to read file for validation: ${filePath}`,
            filePath,
          ),
        };
      }

      const content = readResult.data!;
      const occurrences = this.countOccurrences(content, oldString);

      if (occurrences === 0) {
        return {
          isValid: false,
          error: ToolError.validationError(
            ToolErrorType.EDIT_NO_OCCURRENCE_FOUND,
            `The string to replace was not found in the file`,
            {
              filePath,
              metadata: { oldString, contentLength: content.length },
            },
          ),
        };
      }

      if (
        expectedOccurrences !== undefined &&
        occurrences !== expectedOccurrences
      ) {
        return {
          isValid: false,
          error: ToolError.validationError(
            ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
            `Expected ${expectedOccurrences} occurrences but found ${occurrences}`,
            {
              filePath,
              metadata: {
                oldString,
                expectedOccurrences,
                actualOccurrences: occurrences,
              },
            },
          ),
        };
      }

      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: ToolError.fromUnknownError(
          error,
          ToolErrorType.EDIT_PREPARATION_FAILURE,
        ),
      };
    }
  }

  /**
   * Validate content for write operations
   */
  validateWriteContent(content: string): ValidationResult {
    if (content === undefined || content === null) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.WRITE_FILE_CONTENT_EMPTY,
          'Content cannot be null or undefined',
        ),
      };
    }

    if (typeof content !== 'string') {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.VALIDATION_CONTENT_INVALID,
          'Content must be a string',
        ),
      };
    }

    // Check for extremely large content that might cause issues
    const maxSize = 10 * 1024 * 1024; // 10MB limit
    if (content.length > maxSize) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.FILE_TOO_LARGE,
          `Content too large: ${content.length} characters (max: ${maxSize})`,
        ),
      };
    }

    return { isValid: true };
  }

  /**
   * Async version of validateShellCommand using promises instead of execSync
   */
  async validateShellCommandAsync(command: string): Promise<ValidationResult> {
    if (!command || typeof command !== 'string') {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.VALIDATION_PARAMETERS_INVALID,
          'Command must be a non-empty string',
        ),
      };
    }

    // Basic syntax validation - check for obviously malformed commands
    const trimmedCommand = command.trim();
    if (!trimmedCommand) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.SHELL_COMMAND_SYNTAX_ERROR,
          'Command cannot be empty after trimming',
        ),
      };
    }

    // Extract the main command (first word before any arguments)
    const commandParts = trimmedCommand.split(/\s+/);
    const mainCommand = commandParts[0];

    // Check if the binary exists in PATH using async exec
    try {
      await execAsync(`which "${mainCommand}" 2>/dev/null`);
    } catch {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.SHELL_COMMAND_BINARY_NOT_FOUND,
          `Command not found in PATH: ${mainCommand}`,
          {
            metadata: { command: mainCommand },
            suggestedAction:
              'Check if the command is installed and available in PATH',
          },
        ),
      };
    }

    return { isValid: true };
  }

  /**
   * Validate shell command syntax and binary existence
   */
  validateShellCommand(command: string): ValidationResult {
    if (!command || typeof command !== 'string') {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.VALIDATION_PARAMETERS_INVALID,
          'Command must be a non-empty string',
        ),
      };
    }

    // Basic syntax validation - check for obviously malformed commands
    const trimmedCommand = command.trim();
    if (!trimmedCommand) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.SHELL_COMMAND_SYNTAX_ERROR,
          'Command cannot be empty after trimming',
        ),
      };
    }

    // Extract the main command (first word before any arguments)
    const commandParts = trimmedCommand.split(/\s+/);
    const mainCommand = commandParts[0];

    // Check if the binary exists in PATH
    try {
      execSync(`which "${mainCommand}"`, { stdio: 'ignore' });
    } catch {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.SHELL_COMMAND_BINARY_NOT_FOUND,
          `Command not found in PATH: ${mainCommand}`,
          {
            metadata: { command: mainCommand },
            suggestedAction:
              'Check if the command is installed and available in PATH',
          },
        ),
      };
    }

    return { isValid: true };
  }

  /**
   * Validate edit range parameters
   */
  validateEditRange(
    content: string,
    startLine: number,
    startColumn: number,
    endLine: number,
    endColumn: number,
  ): ValidationResult {
    const lines = content.split('\n');
    const totalLines = lines.length;

    if (startLine < 0 || startLine >= totalLines) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `Start line ${startLine} is out of bounds (0-${totalLines - 1})`,
        ),
      };
    }

    if (endLine < 0 || endLine >= totalLines) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `End line ${endLine} is out of bounds (0-${totalLines - 1})`,
        ),
      };
    }

    if (startLine > endLine) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `Start line ${startLine} cannot be greater than end line ${endLine}`,
        ),
      };
    }

    const startLineContent = lines[startLine] || '';
    const endLineContent = lines[endLine] || '';

    if (startColumn < 0 || startColumn > startLineContent.length) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `Start column ${startColumn} is out of bounds for line ${startLine} (length: ${startLineContent.length})`,
        ),
      };
    }

    if (endColumn < 0 || endColumn > endLineContent.length) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `End column ${endColumn} is out of bounds for line ${endLine} (length: ${endLineContent.length})`,
        ),
      };
    }

    if (startLine === endLine && startColumn > endColumn) {
      return {
        isValid: false,
        error: ToolError.validationError(
          ToolErrorType.EDIT_INVALID_RANGE,
          `Start column ${startColumn} cannot be greater than end column ${endColumn} on the same line`,
        ),
      };
    }

    return { isValid: true };
  }

  /**
   * Perform a dry run validation for edit operations (simulates the replacement)
   */
  async validateEditDryRun(
    filePath: string,
    oldString: string,
    newString: string,
    expectedOccurrences?: number,
  ): Promise<ValidationResult> {
    try {
      const readResult = await this.fileSystemService.readTextFile(filePath);
      if (!readResult.success) {
        return {
          isValid: false,
          error: ToolError.fileSystemError(
            ToolErrorType.FILE_NOT_FOUND,
            `Failed to read file for dry run validation: ${filePath}`,
            filePath,
          ),
        };
      }

      const content = readResult.data!;
      const occurrences = this.countOccurrences(content, oldString);

      if (occurrences === 0) {
        return {
          isValid: false,
          error: ToolError.validationError(
            ToolErrorType.EDIT_NO_OCCURRENCE_FOUND,
            `Dry run: The string to replace was not found in the file`,
            {
              filePath,
              metadata: { oldString, contentLength: content.length },
            },
          ),
        };
      }

      if (
        expectedOccurrences !== undefined &&
        occurrences !== expectedOccurrences
      ) {
        return {
          isValid: false,
          error: ToolError.validationError(
            ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
            `Dry run: Expected ${expectedOccurrences} occurrences but found ${occurrences}`,
            {
              filePath,
              metadata: {
                oldString,
                expectedOccurrences,
                actualOccurrences: occurrences,
              },
            },
          ),
        };
      }

      // Simulate the replacement to check for potential issues
      const newContent = content.replace(
        new RegExp(this.escapeRegExp(oldString), 'g'),
        newString,
      );

      // Check if the replacement would result in an empty file when it shouldn't
      if (newContent.trim().length === 0 && content.trim().length > 0) {
        return {
          isValid: false,
          error: ToolError.validationError(
            ToolErrorType.VALIDATION_CONTENT_INVALID,
            `Dry run: Replacement would result in an empty file`,
            {
              filePath,
              metadata: { oldString, newString },
            },
          ),
        };
      }

      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: ToolError.fromUnknownError(
          error,
          ToolErrorType.EDIT_PREPARATION_FAILURE,
        ),
      };
    }
  }

  /**
   * Escape special regex characters
   */
  private escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /**
   * Count occurrences of a substring in content
   */
  private countOccurrences(content: string, substring: string): number {
    if (!substring) return 0;
    let count = 0;
    let index = 0;
    while ((index = content.indexOf(substring, index)) !== -1) {
      count++;
      index += substring.length;
    }
    return count;
  }
}
