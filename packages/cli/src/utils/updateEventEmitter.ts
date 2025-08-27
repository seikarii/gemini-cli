/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'events';

/**
 * Event types for update-related events
 */
export interface UpdateEvents {
  'update-received': [info: { message: string }];
  'update-failed': [error: { message: string }];
  'update-success': [success: { message: string }];
  'update-info': [info: { message: string }];
}

/**
 * A shared event emitter for application-wide communication
 * between decoupled parts of the CLI.
 *
 * Provides type-safe event emission and listening for update-related events.
 */
export const updateEventEmitter = new EventEmitter() as EventEmitter & {
  emit<K extends keyof UpdateEvents>(
    event: K,
    ...args: UpdateEvents[K]
  ): boolean;
  on<K extends keyof UpdateEvents>(
    event: K,
    listener: (...args: UpdateEvents[K]) => void
  ): EventEmitter;
  off<K extends keyof UpdateEvents>(
    event: K,
    listener: (...args: UpdateEvents[K]) => void
  ): EventEmitter;
};
