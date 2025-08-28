/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  vi,
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  type MockInstance,
} from 'vitest';
import { listMcpServers } from './list.js';
import { loadSettings } from '../../config/settings.js';
import { loadExtensions } from '../../config/extension.js';
import { createTransport } from '@google/gemini-cli-core';

vi.mock('../../config/settings.js', () => ({
  loadSettings: vi.fn(),
}));
vi.mock('../../config/extension.js', () => ({
  loadExtensions: vi.fn(),
}));
vi.mock('@google/gemini-cli-core', () => ({
  createTransport: vi.fn(),
  MCPServerStatus: {
    CONNECTED: 'CONNECTED',
    CONNECTING: 'CONNECTING',
    DISCONNECTED: 'DISCONNECTED',
  },
  Storage: vi.fn().mockImplementation((_cwd: string) => ({
    getGlobalSettingsPath: () => '/tmp/gemini/settings.json',
    getWorkspaceSettingsPath: () => '/tmp/gemini/workspace-settings.json',
    getProjectTempDir: () => '/test/home/.gemini/tmp/mocked_hash',
  })),
  GEMINI_CONFIG_DIR: '.gemini',
  getErrorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e)),
}));
vi.mock('@modelcontextprotocol/sdk/client/index.js', () => ({
  Client: vi.fn().mockImplementation(() => ({
    connect: vi.fn(),
    ping: vi.fn(),
    close: vi.fn(),
  })),
}));

const mockedLoadSettings = loadSettings as unknown as MockInstance<
  () => Promise<unknown>
>;
const mockedLoadExtensions = loadExtensions as unknown as MockInstance<
  () => Promise<unknown>
>;
const mockedCreateTransport = createTransport as unknown as MockInstance<
  () => Promise<unknown>
>;

interface MockClient {
  connect: MockInstance;
  ping: MockInstance;
  close: MockInstance;
}

interface MockTransport {
  close: MockInstance;
}

describe('mcp list command', () => {
  let consoleSpy: MockInstance;
  let mockClient: MockClient;
  let mockTransport: MockTransport;

  beforeEach(() => {
    vi.resetAllMocks();

    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    mockTransport = { close: vi.fn() };
    mockClient = {
      connect: vi.fn(),
      ping: vi.fn(),
      close: vi.fn(),
    };

    // Client is already mocked in vi.mock above
    mockedCreateTransport.mockResolvedValue(mockTransport);
    mockedLoadExtensions.mockReturnValue(Promise.resolve([]));
  });

  afterEach(() => {
    consoleSpy.mockRestore();
  });

  it('should display message when no servers configured', async () => {
    mockedLoadSettings.mockReturnValue(
      Promise.resolve({ merged: { mcpServers: {} } }),
    );

    await listMcpServers();

    expect(consoleSpy).toHaveBeenCalledWith('No MCP servers configured.');
  });

  it('should display different server types with connected status', async () => {
    mockedLoadSettings.mockReturnValue(
      Promise.resolve({
        merged: {
          mcpServers: {
            'stdio-server': { command: '/path/to/server', args: ['arg1'] },
            'sse-server': { url: 'https://example.com/sse' },
            'http-server': { httpUrl: 'https://example.com/http' },
          },
        },
      }),
    );

    mockClient.connect.mockResolvedValue(undefined);
    mockClient.ping.mockResolvedValue(undefined);

    await listMcpServers();

    expect(consoleSpy).toHaveBeenCalledWith('Configured MCP servers:\n');
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'stdio-server: /path/to/server arg1 (stdio) - Connected',
      ),
    );
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'sse-server: https://example.com/sse (sse) - Connected',
      ),
    );
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'http-server: https://example.com/http (http) - Connected',
      ),
    );
  });

  it('should display disconnected status when connection fails', async () => {
    mockedLoadSettings.mockReturnValue(
      Promise.resolve({
        merged: {
          mcpServers: {
            'test-server': { command: '/test/server' },
          },
        },
      }),
    );

    mockClient.connect.mockRejectedValue(new Error('Connection failed'));

    await listMcpServers();

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'test-server: /test/server  (stdio) - Disconnected',
      ),
    );
  });

  it('should merge extension servers with config servers', async () => {
    mockedLoadSettings.mockReturnValue(
      Promise.resolve({
        merged: {
          mcpServers: { 'config-server': { command: '/config/server' } },
        },
      }),
    );

    mockedLoadExtensions.mockReturnValue(
      Promise.resolve([
        {
          config: {
            name: 'test-extension',
            mcpServers: { 'extension-server': { command: '/ext/server' } },
          },
        },
      ]),
    );

    mockClient.connect.mockResolvedValue(undefined);
    mockClient.ping.mockResolvedValue(undefined);

    await listMcpServers();

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'config-server: /config/server  (stdio) - Connected',
      ),
    );
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        'extension-server: /ext/server  (stdio) - Connected',
      ),
    );
  });
});
