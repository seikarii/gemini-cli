/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Utility functions for argument processing and manipulation
 */

/**
 * Injects stdin data into command line arguments by adding or modifying the --prompt flag
 * @param args - Original command line arguments
 * @param stdin - Stdin data to inject
 * @returns Modified arguments array with stdin data injected
 */
export function injectStdinIntoArgs(args: string[], stdin: string): string[] {
  if (!stdin) {
    return [...args];
  }

  const finalArgs = [...args];
  const promptIndex = finalArgs.findIndex(
    (arg) => arg === '--prompt' || arg === '-p',
  );

  if (promptIndex > -1 && finalArgs.length > promptIndex + 1) {
    // Append stdin to existing prompt
    finalArgs[promptIndex + 1] = `${stdin}\n\n${finalArgs[promptIndex + 1]}`;
  } else {
    // Add new prompt argument
    finalArgs.push('--prompt', stdin);
  }

  return finalArgs;
}
