/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { isGitRepository } from '@google/gemini-cli-core';
import * as fs from 'fs';
import * as path from 'path';
import * as childProcess from 'child_process';
import { promisify } from 'util';
import { logger } from '../config/logger.js';

const fsp = fs.promises;
const exec = promisify(childProcess.exec);

export enum PackageManager {
  NPM = 'npm',
  YARN = 'yarn',
  PNPM = 'pnpm',
  PNPX = 'pnpx',
  BUN = 'bun',
  BUNX = 'bunx',
  HOMEBREW = 'homebrew',
  NPX = 'npx',
  UNKNOWN = 'unknown',
}

export interface InstallationInfo {
  packageManager: PackageManager;
  isGlobal: boolean;
  updateCommand?: string;
  updateMessage?: string;
}

export async function getInstallationInfo(
  projectRoot: string,
  isAutoUpdateDisabled: boolean,
): Promise<InstallationInfo> {
  const cliPath = process.argv[1];
  if (!cliPath) {
    return { packageManager: PackageManager.UNKNOWN, isGlobal: false };
  }

  try {
    // Normalize path separators to forward slashes for consistent matching.
    const realPathRaw = await fsp.realpath(cliPath);
    const realPath = realPathRaw.replace(/\\/g, '/');
    const normalizedProjectRoot = projectRoot?.replace(/\\/g, '/');
    const isGit = isGitRepository(process.cwd());

    // Check for local git clone first
    if (
      isGit &&
      normalizedProjectRoot &&
      realPath.startsWith(normalizedProjectRoot) &&
      !realPath.includes('/node_modules/')
    ) {
      return {
        packageManager: PackageManager.UNKNOWN, // Not managed by a package manager in this sense
        isGlobal: false,
        updateMessage:
          'Running from a local git clone. Please update with "git pull".',
      };
    }

    // Check for npx/pnpx
    if (realPath.includes('/.npm/_npx') || realPath.includes('/npm/_npx')) {
      return {
        packageManager: PackageManager.NPX,
        isGlobal: false,
        updateMessage: 'Running via npx, update not applicable.',
      };
    }
    if (realPath.includes('/.pnpm/_pnpx')) {
      return {
        packageManager: PackageManager.PNPX,
        isGlobal: false,
        updateMessage: 'Running via pnpx, update not applicable.',
      };
    }

    // Check for Homebrew
    if (process.platform === 'darwin') {
      try {
        // The package name in homebrew is gemini-cli
        await exec('brew list -1 | grep -q "^gemini-cli$"');
        return {
          packageManager: PackageManager.HOMEBREW,
          isGlobal: true,
          updateMessage:
            'Installed via Homebrew. Please update with "brew upgrade".',
        };
      } catch (error) {
        // Brew is not installed or gemini-cli is not installed via brew.
        // Continue to the next check.
        logger.debug('Homebrew check failed:', error);
      }
    }

    // Check for pnpm
    if (realPath.includes('/.pnpm/global')) {
      const updateCommand = 'pnpm add -g @google/gemini-cli@latest';
      return {
        packageManager: PackageManager.PNPM,
        isGlobal: true,
        updateCommand,
        updateMessage: isAutoUpdateDisabled
          ? `Please run ${updateCommand} to update`
          : 'Installed with pnpm. Attempting to automatically update now...',
      };
    }

    // Check for yarn
    if (realPath.includes('/.yarn/global')) {
      const updateCommand = 'yarn global add @google/gemini-cli@latest';
      return {
        packageManager: PackageManager.YARN,
        isGlobal: true,
        updateCommand,
        updateMessage: isAutoUpdateDisabled
          ? `Please run ${updateCommand} to update`
          : 'Installed with yarn. Attempting to automatically update now...',
      };
    }

    // Check for bun
    if (realPath.includes('/.bun/install/cache')) {
      return {
        packageManager: PackageManager.BUNX,
        isGlobal: false,
        updateMessage: 'Running via bunx, update not applicable.',
      };
    }
    if (realPath.includes('/.bun/bin')) {
      const updateCommand = 'bun add -g @google/gemini-cli@latest';
      return {
        packageManager: PackageManager.BUN,
        isGlobal: true,
        updateCommand,
        updateMessage: isAutoUpdateDisabled
          ? `Please run ${updateCommand} to update`
          : 'Installed with bun. Attempting to automatically update now...',
      };
    }

    // Check for local install
    if (
      normalizedProjectRoot &&
      realPath.startsWith(`${normalizedProjectRoot}/node_modules`)
    ) {
      let pm = PackageManager.NPM;
      const yarnLock = path.join(projectRoot, 'yarn.lock');
      const pnpmLock = path.join(projectRoot, 'pnpm-lock.yaml');
      const bunLock = path.join(projectRoot, 'bun.lockb');
      try {
        await fsp.access(yarnLock);
        pm = PackageManager.YARN;
      } catch (error) {
        logger.debug('yarn.lock not found:', error);
        try {
          await fsp.access(pnpmLock);
          pm = PackageManager.PNPM;
        } catch (error) {
          logger.debug('pnpm-lock.yaml not found:', error);
          try {
            await fsp.access(bunLock);
            pm = PackageManager.BUN;
          } catch (error) {
            // no lockfile found, keep default
            logger.debug('bun.lockb not found:', error);
          }
        }
      }
      return {
        packageManager: pm,
        isGlobal: false,
        updateMessage:
          "Locally installed. Please update via your project's package.json.",
      };
    }

    // Assume global npm
    const updateCommand = 'npm install -g @google/gemini-cli@latest';
    return {
      packageManager: PackageManager.NPM,
      isGlobal: true,
      updateCommand,
      updateMessage: isAutoUpdateDisabled
        ? `Please run ${updateCommand} to update`
        : 'Installed with npm. Attempting to automatically update now...',
    };
  } catch (error) {
    logger.error('Error determining installation info:', error);
    return { packageManager: PackageManager.UNKNOWN, isGlobal: false };
  }
}
