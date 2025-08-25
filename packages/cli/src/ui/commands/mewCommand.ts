/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { SlashCommand, CommandKind } from './types.js';

// Helper to get the project root, so we can reliably find the mew-app
const getProjectRoot = () => {
  const __filename = fileURLToPath(import.meta.url);
  // This path is /packages/cli/src/ui/commands/mewCommand.ts
  // We want to go up 5 levels to the project root
  return path.resolve(__filename, '../../../../../../');
};

export const mewCommand: SlashCommand = {
  kind: CommandKind.BUILT_IN,
  name: 'mew',
  description: 'Launches the Mew agent window.',
  action: async (context) => {
    const projectRoot = getProjectRoot();
    const mewAppPath = path.join(projectRoot, 'mew-upgrade', 'src', 'app', 'index.ts');

    context.ui.addItem(
      {
        type: 'info',
        text: `Spawning Mew window from: ${mewAppPath}`,
      },
      Date.now(),
    );

    // We use tsx to run the TypeScript file directly
    // Launch the web server
    const child = spawn('npm', ['run', 'start:web', '--workspace=@google/gemini-cli-mew-upgrade'], {
      stdio: 'inherit', // Inherit stdio to display server logs
      detached: true, // Allows the child to run independently
    });

    // Open the browser
    const open = (await import('open')).default;
    open('http://localhost:3000');


    // This is crucial: the parent process should not wait for the detached child to exit.
    child.unref();

    context.ui.addItem(
      {
        type: 'info',
        text: `Mew window process spawned with PID: ${child.pid}`,
      },
      Date.now(),
    );

    // We don't have a great way to detect if the window was successfully created,
    // but we can at least report if the spawn itself failed.
    child.on('error', (err) => {
      context.ui.addItem(
        {
          type: 'error',
          text: `Failed to spawn Mew window: ${err.message}`,
        },
        Date.now(),
      );
    });

    // When the main CLI process exits, make sure to kill the child.
    const cleanup = () => {
      if (child.pid) {
        try {
          // Kill the entire process group to prevent orphaned processes
          process.kill(-child.pid, 'SIGKILL');
        } catch (_e) {
          // Ignore errors, the process might have already exited
        }
      }
    };

    process.on('exit', cleanup);

    return {
      type: 'message',
      messageType: 'info',
      content: 'Mew window launched.',
    };
  },
};
