/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Utilities for consistent error handling across the config module.
 */

import { logger } from './logger.js';

export type ErrorSeverity = 'fatal' | 'warning' | 'info';

/**
 * Handles errors consistently across the config module.
 * For fatal errors, logs the error and exits the process.
 * For other severities, only logs the error.
 */
export function handleConfigError(
  message: string,
  severity: ErrorSeverity = 'fatal',
  exitCode: number = 1,
): void {
  switch (severity) {
    case 'fatal':
      logger.error(message);
      process.exit(exitCode);
      break;
    case 'warning':
      logger.warn(message);
      break;
    case 'info':
      logger.info(message);
      break;
    default:
      logger.error(`Unknown error severity: ${severity}`);
      break;
  }
}

/**
 * Creates a standardized error message for configuration validation failures.
 */
export function createValidationErrorMessage(field: string, reason: string): string {
  return `Configuration validation failed for '${field}': ${reason}`;
}

/**
 * Creates a standardized error message for missing required configuration.
 */
export function createMissingConfigErrorMessage(configName: string): string {
  return `Required configuration '${configName}' is missing. Please check your settings.`;
}
