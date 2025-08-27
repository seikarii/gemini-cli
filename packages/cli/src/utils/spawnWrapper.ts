/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn, SpawnOptions } from 'child_process';
import { logger } from '../config/logger.js';

/**
 * Enhanced spawn wrapper that provides centralized logging and error handling
 * for child process spawning throughout the application.
 */
export function spawnWrapper(
  command: string,
  args: readonly string[] = [],
  options: SpawnOptions = {}
): ReturnType<typeof spawn> {
  logger.debug(`Spawning process: ${command} ${args.join(' ')}`);

  const childProcess = spawn(command, args, options);

  // Add error logging
  childProcess.on('error', (error) => {
    logger.error(`Spawn error for command '${command}':`, error);
  });

  return childProcess;
}

// Re-export original spawn for backward compatibility
export { spawn };
