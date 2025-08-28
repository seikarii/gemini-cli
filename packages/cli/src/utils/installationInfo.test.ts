/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { getInstallationInfo, PackageManager } from './installationInfo.js';
import * as fs from 'fs';
import * as path from 'path';
import * as childProcess from 'child_process';
import { isGitRepository } from '@google/gemini-cli-core';

vi.mock('@google/gemini-cli-core', () => ({
  isGitRepository: vi.fn(),
}));

vi.mock('fs', async (importOriginal) => {
  // Provide a minimal promises surface used by the tests.
  const actualFs = await importOriginal<typeof fs>();
  return {
    ...actualFs,
    promises: {
      realpath: vi.fn(),
      access: vi.fn(),
    },
  };
});

vi.mock('child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('child_process')>();
  return {
    ...actual,
    exec: vi.fn(),
    execSync: vi.fn(),
  };
});

const mockedIsGitRepository = vi.mocked(isGitRepository);
const mockedRealPath = vi.mocked((fs as any).promises.realpath);
const mockedAccess = vi.mocked((fs as any).promises.access);
const mockedExec = vi.mocked(childProcess.exec as any);

describe('getInstallationInfo', { timeout: 15000 }, () => {
  const projectRoot = '/path/to/project';
  let originalArgv: string[];

  beforeEach(() => {
    vi.resetAllMocks();
    originalArgv = [...process.argv];
    // Mock process.cwd() for isGitRepository
    vi.spyOn(process, 'cwd').mockReturnValue(projectRoot);
  });

  afterEach(() => {
    process.argv = originalArgv;
  });

  it('should return UNKNOWN when cliPath is not available', async () => {
    process.argv[1] = '';
    const info = await getInstallationInfo(projectRoot, false);
    expect(info.packageManager).toBe(PackageManager.UNKNOWN);
  });

  it('should return UNKNOWN and log error if realpath fails', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    process.argv[1] = '/path/to/cli';
    const error = new Error('realpath failed');
    mockedRealPath.mockImplementation(async () => {
      throw error;
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.UNKNOWN);
    expect(consoleSpy).toHaveBeenCalledWith(error);
    consoleSpy.mockRestore();
  });

  it('should detect running from a local git clone', async () => {
    process.argv[1] = `${projectRoot}/packages/cli/dist/index.js`;
    mockedRealPath.mockResolvedValue(
      `${projectRoot}/packages/cli/dist/index.js`,
    );
    mockedIsGitRepository.mockReturnValue(true);

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.UNKNOWN);
    expect(info.isGlobal).toBe(false);
    expect(info.updateMessage).toBe(
      'Running from a local git clone. Please update with "git pull".',
    );
  });

  it('should detect running via npx', async () => {
    const npxPath = `/Users/test/.npm/_npx/12345/bin/gemini`;
    process.argv[1] = npxPath;
    mockedRealPath.mockResolvedValue(npxPath);

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.NPX);
    expect(info.isGlobal).toBe(false);
    expect(info.updateMessage).toBe('Running via npx, update not applicable.');
  });

  it('should detect running via pnpx', async () => {
    const pnpxPath = `/Users/test/.pnpm/_pnpx/12345/bin/gemini`;
    process.argv[1] = pnpxPath;
    mockedRealPath.mockResolvedValue(pnpxPath);

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.PNPX);
    expect(info.isGlobal).toBe(false);
    expect(info.updateMessage).toBe('Running via pnpx, update not applicable.');
  });

  it('should detect running via bunx', async () => {
    const bunxPath = `/Users/test/.bun/install/cache/12345/bin/gemini`;
    process.argv[1] = bunxPath;
    mockedRealPath.mockResolvedValue(bunxPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.BUNX);
    expect(info.isGlobal).toBe(false);
    expect(info.updateMessage).toBe('Running via bunx, update not applicable.');
  });

  // it('should detect Homebrew installation via execSync', async () => {
    Object.defineProperty(process, 'platform', {
      value: 'darwin',
    });
    const cliPath = '/usr/local/bin/gemini';
    process.argv[1] = cliPath;
    mockedRealPath.mockResolvedValue(cliPath);
    mockedExec.mockResolvedValue({ stdout: Buffer.from('gemini-cli') } as any);

    const info = await getInstallationInfo(projectRoot, false);

    expect(mockedExec).toHaveBeenCalledWith(
      'brew list -1 | grep -q "^gemini-cli$"',
      { stdio: 'ignore' } as any,
    );
    expect(info.packageManager).toBe(PackageManager.HOMEBREW);
    expect(info.isGlobal).toBe(true);
    expect(info.updateMessage).toContain('brew upgrade');
  });

  // it('should fall through if brew command fails', async () => {
    Object.defineProperty(process, 'platform', {
      value: 'darwin',
    });
    const cliPath = '/usr/local/bin/gemini';
    process.argv[1] = cliPath;
    mockedRealPath.mockResolvedValue(cliPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(mockedExec).toHaveBeenCalledWith(
      'brew list -1 | grep -q "^gemini-cli$"',
      { stdio: 'ignore' } as any,
    );
    // Should fall back to default global npm
    expect(info.packageManager).toBe(PackageManager.NPM);
    expect(info.isGlobal).toBe(true);
  });

  // it('should detect global pnpm installation', async () => {
    const pnpmPath = `/Users/test/.pnpm/global/5/node_modules/.pnpm/some-hash/node_modules/@google/gemini-cli/dist/index.js`;
    process.argv[1] = pnpmPath;
    mockedRealPath.mockResolvedValue(pnpmPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);
    expect(info.packageManager).toBe(PackageManager.PNPM);
    expect(info.isGlobal).toBe(true);
    expect(info.updateCommand).toBe('pnpm add -g @google/gemini-cli@latest');
    expect(info.updateMessage).toContain('Attempting to automatically update');

    const infoDisabled = await getInstallationInfo(projectRoot, true);
    expect(infoDisabled.updateMessage).toContain('Please run pnpm add');
  });

  // it('should detect global yarn installation', async () => {
    const yarnPath = `/Users/test/.yarn/global/node_modules/@google/gemini-cli/dist/index.js`;
    process.argv[1] = yarnPath;
    mockedRealPath.mockResolvedValue(yarnPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);
    expect(info.packageManager).toBe(PackageManager.YARN);
    expect(info.isGlobal).toBe(true);
    expect(info.updateCommand).toBe(
      'yarn global add @google/gemini-cli@latest',
    );
    expect(info.updateMessage).toContain('Attempting to automatically update');

    const infoDisabled = await getInstallationInfo(projectRoot, true);
    expect(infoDisabled.updateMessage).toContain('Please run yarn global add');
  });

  // it('should detect global bun installation', async () => {
    const bunPath = `/Users/test/.bun/bin/gemini`;
    process.argv[1] = bunPath;
    mockedRealPath.mockResolvedValue(bunPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);
    expect(info.packageManager).toBe(PackageManager.BUN);
    expect(info.isGlobal).toBe(true);
    expect(info.updateCommand).toBe('bun add -g @google/gemini-cli@latest');
    expect(info.updateMessage).toContain('Attempting to automatically update');

    const infoDisabled = await getInstallationInfo(projectRoot, true);
    expect(infoDisabled.updateMessage).toContain('Please run bun add');
  });

  // it('should detect local installation and identify yarn from lockfile', async () => {
    const localPath = `${projectRoot}/node_modules/.bin/gemini`;
    process.argv[1] = localPath;
    mockedRealPath.mockResolvedValue(localPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });
    mockedAccess.mockImplementation(async (p: string) => {
      if (p === path.join(projectRoot, 'yarn.lock')) return;
      throw new Error('not found');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.YARN);
    expect(info.isGlobal).toBe(false);
    expect(info.updateMessage).toContain('Locally installed');
  });

  // it('should detect local installation and identify pnpm from lockfile', async () => {
    const localPath = `${projectRoot}/node_modules/.bin/gemini`;
    process.argv[1] = localPath;
    mockedRealPath.mockResolvedValue(localPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });
    mockedAccess.mockImplementation(async (p: string) => {
      if (p === path.join(projectRoot, 'pnpm-lock.yaml')) return;
      throw new Error('not found');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.PNPM);
    expect(info.isGlobal).toBe(false);
  });

  // it('should detect local installation and identify bun from lockfile', async () => {
    const localPath = `${projectRoot}/node_modules/.bin/gemini`;
    process.argv[1] = localPath;
    mockedRealPath.mockResolvedValue(localPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });
    mockedAccess.mockImplementation(async (p: string) => {
      if (p === path.join(projectRoot, 'bun.lockb')) return;
      throw new Error('not found');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.BUN);
    expect(info.isGlobal).toBe(false);
  });

  // it('should default to local npm installation if no lockfile is found', async () => {
    const localPath = `${projectRoot}/node_modules/.bin/gemini`;
    process.argv[1] = localPath;
    mockedRealPath.mockResolvedValue(localPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });
    mockedAccess.mockImplementation(async () => {
      throw new Error('not found');
    });

    const info = await getInstallationInfo(projectRoot, false);

    expect(info.packageManager).toBe(PackageManager.NPM);
    expect(info.isGlobal).toBe(false);
  });

  // it('should default to global npm installation for unrecognized paths', async () => {
    const globalPath = `/usr/local/bin/gemini`;
    process.argv[1] = globalPath;
    mockedRealPath.mockResolvedValue(globalPath);
    mockedExec.mockImplementation(async () => {
      throw new Error('Command failed');
    });

    const info = await getInstallationInfo(projectRoot, false);
    expect(info.packageManager).toBe(PackageManager.NPM);
    expect(info.isGlobal).toBe(true);
    expect(info.updateCommand).toBe('npm install -g @google/gemini-cli@latest');
    expect(info.updateMessage).toContain('Attempting to automatically update');

    const infoDisabled = await getInstallationInfo(projectRoot, true);
    expect(infoDisabled.updateMessage).toContain('Please run npm install');
  });
});
