/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Simple logger interface for RAG system components.
 * This provides a consistent logging interface that can be easily
 * replaced with more sophisticated logging systems in the future.
 */
export interface RAGLogger {
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

/**
 * Console-based implementation of RAGLogger.
 */
export class ConsoleRAGLogger implements RAGLogger {
  constructor(private readonly prefix: string = '[RAG]') {}

  debug(message: string, ...args: unknown[]): void {
    console.debug(`${this.prefix} ${message}`, ...args);
  }

  info(message: string, ...args: unknown[]): void {
    console.info(`${this.prefix} ${message}`, ...args);
  }

  warn(message: string, ...args: unknown[]): void {
    console.warn(`${this.prefix} ${message}`, ...args);
  }

  error(message: string, ...args: unknown[]): void {
    console.error(`${this.prefix} ${message}`, ...args);
  }
}

/**
 * No-op logger that discards all log messages.
 */
export class NoOpRAGLogger implements RAGLogger {
  debug(): void {
    // No-op
  }

  info(): void {
    // No-op
  }

  warn(): void {
    // No-op
  }

  error(): void {
    // No-op
  }
}
