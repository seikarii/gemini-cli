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
