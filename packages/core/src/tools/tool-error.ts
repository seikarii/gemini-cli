/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * A type-safe enum for tool-related errors.
 */
export enum ToolErrorType {
  // General Errors
  INVALID_TOOL_PARAMS = 'invalid_tool_params',
  UNKNOWN = 'unknown',
  UNHANDLED_EXCEPTION = 'unhandled_exception',
  TOOL_NOT_REGISTERED = 'tool_not_registered',
  EXECUTION_FAILED = 'execution_failed',

  // File System Errors
  FILE_NOT_FOUND = 'file_not_found',
  FILE_WRITE_FAILURE = 'file_write_failure',
  READ_CONTENT_FAILURE = 'read_content_failure',
  ATTEMPT_TO_CREATE_EXISTING_FILE = 'attempt_to_create_existing_file',
  FILE_TOO_LARGE = 'file_too_large',
  PERMISSION_DENIED = 'permission_denied',
  NO_SPACE_LEFT = 'no_space_left',
  TARGET_IS_DIRECTORY = 'target_is_directory',
  PATH_NOT_IN_WORKSPACE = 'path_not_in_workspace',
  SEARCH_PATH_NOT_FOUND = 'search_path_not_found',
  SEARCH_PATH_NOT_A_DIRECTORY = 'search_path_not_a_directory',
  DIRECTORY_NOT_EMPTY = 'directory_not_empty',
  INVALID_PATH_FORMAT = 'invalid_path_format',

  // Edit-specific Errors
  EDIT_PREPARATION_FAILURE = 'edit_preparation_failure',
  EDIT_NO_OCCURRENCE_FOUND = 'edit_no_occurrence_found',
  EDIT_EXPECTED_OCCURRENCE_MISMATCH = 'edit_expected_occurrence_mismatch',
  EDIT_NO_CHANGE = 'edit_no_change',
  EDIT_INVALID_RANGE = 'edit_invalid_range',
  EDIT_CONTENT_MISMATCH = 'edit_content_mismatch',

  // Write File specific Errors
  WRITE_FILE_CONTENT_EMPTY = 'write_file_content_empty',
  WRITE_FILE_INVALID_MODE = 'write_file_invalid_mode',

  // Shell Command Errors
  SHELL_COMMAND_SYNTAX_ERROR = 'shell_command_syntax_error',
  SHELL_COMMAND_BINARY_NOT_FOUND = 'shell_command_binary_not_found',
  SHELL_COMMAND_PERMISSION_DENIED = 'shell_command_permission_denied',
  SHELL_COMMAND_TIMEOUT = 'shell_command_timeout',
  SHELL_COMMAND_RESOURCE_EXHAUSTED = 'shell_command_resource_exhausted',

  // Validation Errors
  VALIDATION_FILE_NOT_EXISTS = 'validation_file_not_exists',
  VALIDATION_PATH_NOT_ACCESSIBLE = 'validation_path_not_accessible',
  VALIDATION_CONTENT_INVALID = 'validation_content_invalid',
  VALIDATION_PARAMETERS_MISSING = 'validation_parameters_missing',
  VALIDATION_PARAMETERS_INVALID = 'validation_parameters_invalid',

  // Retry-related Errors
  RETRY_EXHAUSTED = 'retry_exhausted',
  RETRY_TRANSIENT_FAILURE = 'retry_transient_failure',

  // Glob-specific Errors
  GLOB_EXECUTION_ERROR = 'glob_execution_error',

  // Grep-specific Errors
  GREP_EXECUTION_ERROR = 'grep_execution_error',

  // Ls-specific Errors
  LS_EXECUTION_ERROR = 'ls_execution_error',
  PATH_IS_NOT_A_DIRECTORY = 'path_is_not_a_directory',

  // MCP-specific Errors
  MCP_TOOL_ERROR = 'mcp_tool_error',

  // Memory-specific Errors
  MEMORY_TOOL_EXECUTION_ERROR = 'memory_tool_execution_error',

  // ReadManyFiles-specific Errors
  READ_MANY_FILES_SEARCH_ERROR = 'read_many_files_search_error',

  // DiscoveredTool-specific Errors
  DISCOVERED_TOOL_EXECUTION_ERROR = 'discovered_tool_execution_error',

  // WebFetch-specific Errors
  WEB_FETCH_NO_URL_IN_PROMPT = 'web_fetch_no_url_in_prompt',
  WEB_FETCH_FALLBACK_FAILED = 'web_fetch_fallback_failed',
  WEB_FETCH_PROCESSING_ERROR = 'web_fetch_processing_error',

  // WebSearch-specific Errors
  WEB_SEARCH_FAILED = 'web_search_failed',
}

/**
 * Structured error context information
 */
export interface ErrorContext {
  /** The file path related to the error (if applicable) */
  filePath?: string;
  /** Line number where the error occurred (if applicable) */
  lineNumber?: number;
  /** Column number where the error occurred (if applicable) */
  columnNumber?: number;
  /** Additional metadata about the error */
  metadata?: Record<string, unknown>;
  /** The underlying error that caused this tool error */
  cause?: Error;
  /** Suggested recovery action */
  suggestedAction?: string;
}

/**
 * Enhanced tool error class with structured information
 */
export class ToolError extends Error {
  readonly type: ToolErrorType;
  readonly context?: ErrorContext;
  readonly isRetryable: boolean;

  constructor(
    type: ToolErrorType,
    message: string,
    context?: ErrorContext,
    isRetryable: boolean = false,
  ) {
    super(message);
    this.name = 'ToolError';
    this.type = type;
    this.context = context;
    this.isRetryable = isRetryable;

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ToolError);
    }
  }

  /**
   * Create a ToolError from an unknown error
   */
  static fromUnknownError(
    error: unknown,
    type: ToolErrorType = ToolErrorType.UNKNOWN,
  ): ToolError {
    const message = error instanceof Error ? error.message : String(error);
    return new ToolError(type, message, {
      cause: error instanceof Error ? error : undefined,
    });
  }

  /**
   * Create a ToolError for file system related issues
   */
  static fileSystemError(
    type: ToolErrorType,
    message: string,
    filePath?: string,
    cause?: Error,
  ): ToolError {
    return new ToolError(type, message, {
      filePath,
      cause,
      suggestedAction: this.getSuggestedAction(type),
    });
  }

  /**
   * Create a ToolError for validation failures
   */
  static validationError(
    type: ToolErrorType,
    message: string,
    context?: Partial<ErrorContext>,
  ): ToolError {
    return new ToolError(type, message, {
      ...context,
      suggestedAction: this.getSuggestedAction(type),
    });
  }

  /**
   * Get suggested action based on error type
   */
  private static getSuggestedAction(type: ToolErrorType): string | undefined {
    switch (type) {
      case ToolErrorType.FILE_NOT_FOUND:
        return 'Check if the file path exists and is accessible';
      case ToolErrorType.PERMISSION_DENIED:
        return 'Verify file permissions and user access rights';
      case ToolErrorType.EDIT_NO_OCCURRENCE_FOUND:
        return 'Verify the old_string exists in the file and try a smaller unique substring';
      case ToolErrorType.SHELL_COMMAND_BINARY_NOT_FOUND:
        return 'Check if the command is installed and available in PATH';
      case ToolErrorType.VALIDATION_PARAMETERS_INVALID:
        return 'Review the tool parameters and ensure they meet the requirements';
      default:
        return undefined;
    }
  }

  /**
   * Convert ToolError to a plain object for serialization
   */
  toJSON() {
    return {
      name: this.name,
      type: this.type,
      message: this.message,
      context: this.context,
      isRetryable: this.isRetryable,
      stack: this.stack,
    };
  }
}
