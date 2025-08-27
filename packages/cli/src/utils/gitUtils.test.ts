/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { vi, describe, expect, it, afterEach, beforeEach } from 'vitest';
import * as child_process from 'child_process';
import {
  isGitHubRepository,
  getGitRepoRoot,
  getLatestGitHubRelease,
  getGitHubRepoInfo,
} from './gitUtils.js';

vi.mock('child_process');

describe('isGitHubRepository', async () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns false if the git command fails', async () => {
    vi.spyOn(child_process, 'exec').mockImplementation((...args: unknown[]) => {
      const cb = args[args.length - 1] as unknown;
      if (typeof cb === 'function')
        (cb as (err: unknown, stdout?: string) => void)(new Error('oops'));
      return undefined as unknown as child_process.ChildProcess;
    });
    await expect(isGitHubRepository()).resolves.toBe(false);
  });

  it('returns false if the remote is not github.com', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(
            null,
            'https://gitlab.com',
          );
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(isGitHubRepository()).resolves.toBe(false);
  });

  it('returns true if the remote is github.com', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(
            null,
            `
      origin  https://github.com/sethvargo/gemini-cli (fetch)
      origin  https://github.com/sethvargo/gemini-cli (push)
    `,
          );
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(isGitHubRepository()).resolves.toBe(true);
  });
});

describe('getGitHubRepoInfo', async () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('throws an error if github repo info cannot be determined', async () => {
    vi.spyOn(child_process, 'exec').mockImplementation((...args: unknown[]) => {
      const cb = args[args.length - 1] as unknown;
      if (typeof cb === 'function')
        (cb as (err: unknown, stdout?: string) => void)(new Error('oops'));
      return undefined as unknown as child_process.ChildProcess;
    });
    await expect(getGitHubRepoInfo()).rejects.toThrowError(/oops/);
  });

  it('throws an error if owner/repo could not be determined', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(null, '');
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(getGitHubRepoInfo()).rejects.toThrowError(
      /Owner & repo could not be extracted from remote URL/,
    );
  });

  it('returns the owner and repo', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(
            null,
            'https://github.com/owner/repo.git ',
          );
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(getGitHubRepoInfo()).resolves.toEqual({
      owner: 'owner',
      repo: 'repo',
    });
  });
});

describe('getGitRepoRoot', async () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('throws an error if git root cannot be determined', async () => {
    vi.spyOn(child_process, 'exec').mockImplementation((...args: unknown[]) => {
      const cb = args[args.length - 1] as unknown;
      if (typeof cb === 'function')
        (cb as (err: unknown, stdout?: string) => void)(new Error('oops'));
      return undefined as unknown as child_process.ChildProcess;
    });
    await expect(getGitRepoRoot()).rejects.toThrowError(/oops/);
  });

  it('throws an error if git root is empty', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(null, '');
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(getGitRepoRoot()).rejects.toThrowError(
      /Git repo returned empty value/,
    );
  });

  it('returns the root', async () => {
    vi.spyOn(child_process, 'exec').mockImplementationOnce(
      (...args: unknown[]) => {
        const cb = args[args.length - 1] as unknown;
        if (typeof cb === 'function')
          (cb as (err: unknown, stdout?: string) => void)(
            null,
            '/path/to/git/repo',
          );
        return undefined as unknown as child_process.ChildProcess;
      },
    );
    await expect(getGitRepoRoot()).resolves.toBe('/path/to/git/repo');
  });
});

describe('getLatestRelease', async () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('throws an error if the fetch fails', async () => {
    global.fetch = vi.fn(() => Promise.reject('nope'));
    expect(getLatestGitHubRelease()).rejects.toThrowError(
      /Unable to determine the latest/,
    );
  });

  it('throws an error if the fetch does not return a json body', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ foo: 'bar' }),
      } as Response),
    );
    expect(getLatestGitHubRelease()).rejects.toThrowError(
      /Unable to determine the latest/,
    );
  });

  it('returns the release version', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ tag_name: 'v1.2.3' }),
      } as Response),
    );
    expect(getLatestGitHubRelease()).resolves.toBe('v1.2.3');
  });
});
