/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Shared constants used across the CLI configuration modules.
 */

// Directory name for storing Gemini CLI settings and configuration files
export const SETTINGS_DIRECTORY_NAME = '.gemini';

// Configuration file names
export const EXTENSIONS_CONFIG_FILENAME = 'gemini-extension.json';
export const TRUSTED_FOLDERS_FILENAME = 'trustedFolders.json';

// Default context file name
export const DEFAULT_CONTEXT_FILENAME = 'GEMINI.md';

// Scope labels for dialog components
export const SCOPE_LABELS = {
  User: 'User Settings',
  Workspace: 'Workspace Settings',
  System: 'System Settings',
} as const;

// Prompt processor constants
/**
 * The placeholder string for shorthand argument injection in custom commands.
 * When used outside of !{...}, arguments are injected raw.
 * When used inside !{...}, arguments are shell-escaped.
 */
export const SHORTHAND_ARGS_PLACEHOLDER = '{{args}}';

/**
 * The trigger string for shell command injection in custom commands.
 */
export const SHELL_INJECTION_TRIGGER = '!{';

// Environment variable names
export const ENV_GEMINI_CLI_SYSTEM_SETTINGS_PATH =
  'GEMINI_CLI_SYSTEM_SETTINGS_PATH';
export const ENV_GOOGLE_CLOUD_PROJECT = 'GOOGLE_CLOUD_PROJECT';
export const ENV_GOOGLE_CLOUD_LOCATION = 'GOOGLE_CLOUD_LOCATION';
export const ENV_GEMINI_API_KEY = 'GEMINI_API_KEY';
export const ENV_GOOGLE_API_KEY = 'GOOGLE_API_KEY';

// System paths for different platforms
export const SYSTEM_SETTINGS_PATH_DARWIN =
  '/Library/Application Support/GeminiCli/settings.json';
export const SYSTEM_SETTINGS_PATH_WIN32 =
  'C:\\ProgramData\\gemini-cli\\settings.json';
export const SYSTEM_SETTINGS_PATH_LINUX = '/etc/gemini-cli/settings.json';

// Error messages for authentication validation
export const ERROR_GOOGLE_CLOUD_PROJECT_NOT_SET = `[Error] GOOGLE_CLOUD_PROJECT is not set.
Please set it using:
  export GOOGLE_CLOUD_PROJECT=<your-project-id>
and try again.`;

export const ERROR_GEMINI_API_KEY_NOT_FOUND =
  'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';

export const ERROR_VERTEX_AI_CONFIG_MISSING = `When using Vertex AI, you must specify either:
• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
• GOOGLE_API_KEY environment variable (if using express mode).
Update your environment and try again (no reload needed if using .env)!`;

export const ERROR_INVALID_AUTH_METHOD = 'Invalid auth method selected.';

// Default excluded environment variables
export const DEFAULT_EXCLUDED_ENV_VARS = ['DEBUG', 'DEBUG_MODE'];

// Error messages for file operations
export const ERROR_SAVING_USER_SETTINGS = 'Error saving user settings file:';
export const ERROR_SAVING_TRUSTED_FOLDERS =
  'Error saving trusted folders file:';

// Error messages for sandbox configuration
export const ERROR_INVALID_SANDBOX_COMMAND = (
  command: string,
  validCommands: readonly string[],
) =>
  `ERROR: invalid sandbox command '${command}'. Must be one of ${validCommands.join(', ')}`;

export const ERROR_MISSING_SANDBOX_COMMAND = (command: string) =>
  `ERROR: missing sandbox command '${command}' (from GEMINI_SANDBOX)`;

export const ERROR_SANDBOX_COMMAND_NOT_FOUND =
  'ERROR: GEMINI_SANDBOX is true but failed to determine command for sandbox; install docker or podman or specify command in GEMINI_SANDBOX';

// Error messages for configuration validation
export const ERROR_INVALID_CONFIGURATION =
  'Invalid configuration detected. Please check your settings.';
export const ERROR_MISSING_REQUIRED_CONFIG = (configName: string) =>
  `Required configuration '${configName}' is missing or invalid.`;
export const ERROR_CONFIG_FILE_NOT_FOUND = (filePath: string) =>
  `Configuration file not found: ${filePath}`;
export const ERROR_CONFIG_FILE_INVALID_JSON = (filePath: string) =>
  `Configuration file contains invalid JSON: ${filePath}`;
export const ERROR_CONFIG_VALIDATION_FAILED = (field: string, reason: string) =>
  `Configuration validation failed for '${field}': ${reason}`;

// Error messages for settings operations
export const ERROR_SETTINGS_LOAD_FAILED = 'Failed to load settings from file.';
export const ERROR_SETTINGS_SAVE_FAILED = 'Failed to save settings to file.';
export const ERROR_SETTINGS_MERGE_FAILED =
  'Failed to merge settings from different scopes.';
