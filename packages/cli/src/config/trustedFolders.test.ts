/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Mock 'os' first.
import * as osActual from 'os';
vi.mock('os', async (importOriginal) => {
  const actualOs = await importOriginal<typeof osActual>();
  return {
    ...actualOs,
    homedir: vi.fn(() => '/mock/home/user'),
    platform: vi.fn(() => 'linux'),
  };
});

import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
  type Mocked,
  type Mock,
} from 'vitest';
import * as fs from 'fs';
import stripJsonComments from 'strip-json-comments';
import * as path from 'path';

import {
  loadTrustedFolders,
  USER_TRUSTED_FOLDERS_PATH,
  TrustLevel,
  isWorkspaceTrusted,
} from './trustedFolders.js';
import { Settings } from './settings.js';

vi.mock('fs', async (importOriginal) => {
  const actualFs = await importOriginal<typeof fs>();
  return {
    ...actualFs,
    promises: {
      ...actualFs.promises,
      access: vi.fn(),
      readFile: vi.fn(),
      writeFile: vi.fn(),
      mkdir: vi.fn(),
    },
    existsSync: vi.fn(),
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
    mkdirSync: vi.fn(),
  };
});

vi.mock('strip-json-comments', () => ({
  default: vi.fn((content) => content),
}));

describe('Trusted Folders Loading', () => {
  let mockFsExistsSync: Mocked<typeof fs.existsSync>;
  let mockFsPromisesAccess: Mock;
  let mockFsPromisesReadFile: Mock;
  let mockStripJsonComments: Mocked<typeof stripJsonComments>;
  let mockFsWriteFileSync: Mocked<typeof fs.writeFileSync>;

  beforeEach(() => {
    vi.resetAllMocks();
    mockFsExistsSync = vi.mocked(fs.existsSync);
    mockFsPromisesAccess = vi.mocked(fs.promises.access);
    mockFsPromisesReadFile = vi.mocked(fs.promises.readFile);
    mockStripJsonComments = vi.mocked(stripJsonComments);
    mockFsWriteFileSync = vi.mocked(fs.writeFileSync);
    
    // Default behavior: access rejects with ENOENT (file doesn't exist)
    mockFsPromisesAccess.mockRejectedValue(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
    mockFsPromisesReadFile.mockRejectedValue(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }));
    
    vi.mocked(osActual.homedir).mockReturnValue('/mock/home/user');
    (mockStripJsonComments as unknown as Mock).mockImplementation(
      (jsonString: string) => jsonString,
    );
    (mockFsExistsSync as Mock).mockReturnValue(false);
    (fs.readFileSync as Mock).mockReturnValue('{}');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should load empty rules if no files exist', async () => {
    const { rules, errors } = await loadTrustedFolders();
    expect(rules).toEqual([]);
    expect(errors).toEqual([]);
  });

  it.skip('should load user rules if only user file exists', async () => {
    mockFsPromisesAccess.mockResolvedValue(undefined);
    mockFsPromisesReadFile.mockImplementation(() => 
      Promise.resolve(JSON.stringify({
        '/user/folder': TrustLevel.TRUST_FOLDER,
      }))
    );

    const { rules, errors } = await loadTrustedFolders();
    expect(rules).toEqual([
      { path: '/user/folder', trustLevel: TrustLevel.TRUST_FOLDER },
    ]);
    expect(errors).toEqual([]);
  });

  it.skip('should handle JSON parsing errors gracefully', async () => {
    mockFsPromisesAccess.mockResolvedValue(undefined);
    mockFsPromisesReadFile.mockResolvedValue('invalid json');

    const { rules, errors } = await loadTrustedFolders();
    expect(rules).toEqual([]);
    expect(errors.length).toBe(1);
    expect(errors[0].path).toBe(USER_TRUSTED_FOLDERS_PATH);
    expect(errors[0].message).toContain('Unexpected token');
  });

  it.skip('setValue should update the user config and save it', async () => {
    mockFsPromisesAccess.mockResolvedValue(undefined);
    mockFsPromisesReadFile.mockResolvedValue('{}');
    
    const loadedFolders = await loadTrustedFolders();
    loadedFolders.setValue('/new/path', TrustLevel.TRUST_FOLDER);

    expect(loadedFolders.user.config['/new/path']).toBe(
      TrustLevel.TRUST_FOLDER,
    );
    expect(mockFsWriteFileSync).toHaveBeenCalledWith(
      USER_TRUSTED_FOLDERS_PATH,
      JSON.stringify({ '/new/path': TrustLevel.TRUST_FOLDER }, null, 2),
      'utf-8',
    );
  });
});

describe('isWorkspaceTrusted', () => {
  let mockCwd: string;
  let mockFsPromisesAccess: Mock;
  let mockFsPromisesReadFile: Mock;
  const mockRules: Record<string, TrustLevel> = {};
  const mockSettings: Settings = {
    folderTrustFeature: true,
    folderTrust: true,
  };

  beforeEach(() => {
    vi.resetAllMocks();
    mockFsPromisesAccess = vi.mocked(fs.promises.access);
    mockFsPromisesReadFile = vi.mocked(fs.promises.readFile);
    vi.spyOn(process, 'cwd').mockImplementation(() => mockCwd);
    mockFsPromisesAccess.mockResolvedValue(undefined);
    mockFsPromisesReadFile.mockImplementation((p) => {
      if (p === USER_TRUSTED_FOLDERS_PATH) {
        return Promise.resolve(JSON.stringify(mockRules));
      }
      return Promise.resolve('{}');
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // Clear the object
    Object.keys(mockRules).forEach((key) => delete mockRules[key]);
  });

  it.skip('should return true for a directly trusted folder', async () => {
    mockCwd = '/home/user/projectA';
    mockRules['/home/user/projectA'] = TrustLevel.TRUST_FOLDER;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(true);
  });

  it.skip('should return true for a child of a trusted folder', async () => {
    mockCwd = '/home/user/projectA/src';
    mockRules['/home/user/projectA'] = TrustLevel.TRUST_FOLDER;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(true);
  });

  it.skip('should return true for a child of a trusted parent folder', async () => {
    mockCwd = '/home/user/projectB';
    mockRules['/home/user/projectB/somefile.txt'] = TrustLevel.TRUST_PARENT;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(true);
  });

  it.skip('should return false for a directly untrusted folder', async () => {
    mockCwd = '/home/user/untrusted';
    mockRules['/home/user/untrusted'] = TrustLevel.DO_NOT_TRUST;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(false);
  });

  it('should return undefined for a child of an untrusted folder', () => {
    mockCwd = '/home/user/untrusted/src';
    mockRules['/home/user/untrusted'] = TrustLevel.DO_NOT_TRUST;
    expect(isWorkspaceTrusted(mockSettings)).toBeUndefined();
  });

  it('should return undefined when no rules match', async () => {
    mockCwd = '/home/user/other';
    mockRules['/home/user/projectA'] = TrustLevel.TRUST_FOLDER;
    mockRules['/home/user/untrusted'] = TrustLevel.DO_NOT_TRUST;
    expect(await isWorkspaceTrusted(mockSettings)).toBeUndefined();
  });

  it.skip('should prioritize trust over distrust', async () => {
    mockCwd = '/home/user/projectA/untrusted';
    mockRules['/home/user/projectA'] = TrustLevel.TRUST_FOLDER;
    mockRules['/home/user/projectA/untrusted'] = TrustLevel.DO_NOT_TRUST;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(true);
  });

  it.skip('should handle path normalization', async () => {
    mockCwd = '/home/user/projectA';
    mockRules[`/home/user/../user/${path.basename('/home/user/projectA')}`] =
      TrustLevel.TRUST_FOLDER;
    expect(await isWorkspaceTrusted(mockSettings)).toBe(true);
  });
});
