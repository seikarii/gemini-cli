/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { platform } from 'node:os';
import { URL } from 'node:url';
import { shouldAttemptBrowserLaunch } from './browserDetection.js';

const execFileAsync = promisify(execFile);

/**
 * Validates that a URL is safe to open in a browser.
 * Only allows HTTP and HTTPS URLs to prevent command injection.
 *
 * @param url The URL to validate
 * @throws Error if the URL is invalid or uses an unsafe protocol
 */
function validateUrl(url: string): void {
  let parsedUrl: URL;

  try {
    parsedUrl = new URL(url);
  } catch (_error) {
    throw new Error(`Invalid URL: ${url}`);
  }

  // Only allow HTTP and HTTPS protocols
  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw new Error(
      `Unsafe protocol: ${parsedUrl.protocol}. Only HTTP and HTTPS are allowed.`,
    );
  }

  // Additional validation: ensure no newlines or control characters
  // eslint-disable-next-line no-control-regex
  if (/[\r\n\x00-\x1f]/.test(url)) {
    throw new Error('URL contains invalid characters');
  }
}

/**
 * Opens a URL in the default browser using platform-specific commands.
 * This implementation avoids shell injection vulnerabilities by:
 * 1. Validating the URL to ensure it's HTTP/HTTPS only
 * 2. Using execFile instead of exec to avoid shell interpretation
 * 3. Passing the URL as an argument rather than constructing a command string
 *
 * @param url The URL to open
 * @throws Error if the URL is invalid or if opening the browser fails
 */
export async function openBrowserSecurely(url: string): Promise<void> {
  // Validate the URL first
  validateUrl(url);

  const platformName = platform();
  let command: string;
  let args: string[];

  switch (platformName) {
    case 'darwin':
      // macOS
      command = 'open';
      args = [url];
      break;

    case 'win32':
      // Windows - use PowerShell with Start-Process
      // This avoids the cmd.exe shell which is vulnerable to injection
      command = 'powershell.exe';
      args = [
        '-NoProfile',
        '-NonInteractive',
        '-WindowStyle',
        'Hidden',
        '-Command',
        `Start-Process '${url.replace(/'/g, "''")}'`,
      ];
      break;

    case 'linux':
    case 'freebsd':
    case 'openbsd':
      // Linux and BSD variants
      // Try xdg-open first, fall back to other options
      command = 'xdg-open';
      args = [url];
      break;

    default:
      throw new Error(`Unsupported platform: ${platformName}`);
  }

  const options: Record<string, unknown> = {
    // Don't inherit parent's environment to avoid potential issues
    env: {
      ...process.env,
      // Ensure we're not in a shell that might interpret special characters
      SHELL: undefined,
    },
    // Detach the browser process so it doesn't block
    detached: true,
    stdio: 'ignore',
  };

  try {
    await execFileAsync(command, args, options);
  } catch (error) {
    // For Linux, try fallback commands if xdg-open fails
    if (
      (platformName === 'linux' ||
        platformName === 'freebsd' ||
        platformName === 'openbsd') &&
      command === 'xdg-open'
    ) {
      const fallbackCommands = [
        'gnome-open',
        'kde-open',
        'firefox',
        'chromium',
        'google-chrome',
      ];

      for (const fallbackCommand of fallbackCommands) {
        try {
          await execFileAsync(fallbackCommand, [url], options);
          return; // Success!
        } catch {
          // Try next command
          continue;
        }
      }
    }

    // Re-throw the error if all attempts failed
    throw new Error(
      `Failed to open browser: ${error instanceof Error ? error.message : 'Unknown error'}`,
    );
  }
}

/**
 * Checks if the current environment should attempt to launch a browser.
 * This is the same logic as in browser.ts for consistency.
 *
 * @returns True if the tool should attempt to launch a browser
 */
export function shouldLaunchBrowser(): boolean {
  return shouldAttemptBrowserLaunch();
}
