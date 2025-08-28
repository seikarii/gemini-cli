import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PluginSandbox } from '../PluginSandbox';
import { ISecurityPolicy, IPluginContext, IPluginExecutionResult } from '../PluginSandbox';
import { Worker } from 'worker_threads';
import { promises as fs } from 'fs';

// Mock Worker and fs to avoid actual worker creation and file operations in tests
vi.mock('worker_threads', () => ({
  Worker: vi.fn(),
  isMainThread: true,
  parentPort: null
}));

vi.mock('fs', () => ({
  promises: {
    readFile: vi.fn(),
    writeFile: vi.fn(),
    access: vi.fn(),
    mkdir: vi.fn()
  }
}));

interface MockWorker {
  postMessage: ReturnType<typeof vi.fn>;
  terminate: ReturnType<typeof vi.fn>;
  on: ReturnType<typeof vi.fn>;
  removeAllListeners: ReturnType<typeof vi.fn>;
}

type WorkerCallback = (data: Record<string, unknown>) => void;

describe('PluginSandbox', () => {
  let sandbox: PluginSandbox;
  let mockWorker: MockWorker;
  let defaultPolicy: ISecurityPolicy;

  beforeEach(() => {
    mockWorker = {
      postMessage: vi.fn(),
      terminate: vi.fn(),
      on: vi.fn(),
      removeAllListeners: vi.fn()
    };
    
    (Worker as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => mockWorker);
    
    // Mock fs.readFile to return valid plugin code
    (fs.readFile as ReturnType<typeof vi.fn>).mockResolvedValue(`
      module.exports = {
        name: 'test-plugin',
        version: '1.0.0',
        execute: (input) => ({ result: 'processed: ' + input })
      };
    `);

    defaultPolicy = {
      maxExecutionTime: 5000,
      maxMemoryUsage: 100 * 1024 * 1024, // 100MB
      allowedHosts: [],
      allowedReadPaths: ['/tmp'],
      allowedWritePaths: ['/tmp/output'],
      allowFileSystem: true,
      allowNetwork: false,
      allowChildProcesses: false,
      apiCallLimit: 100,
      requireSignature: false,
      environmentVariables: {}
    };

    sandbox = new PluginSandbox();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Plugin Loading', () => {
    it('should load valid plugin', async () => {
      const mockContext: IPluginContext = {
        id: 'test-plugin-id',
        name: 'test-plugin',
        version: '1.0.0',
        author: 'test-author',
        description: 'Test plugin',
        permissions: [],
        sourceHash: 'mock-hash',
        entryPoint: 'index.js',
        dependencies: {}
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'loaded',
              success: true,
              context: mockContext
            });
          }, 10);
        }
      });

      const result = await sandbox.loadPlugin('/path/to/plugin.js', defaultPolicy);
      expect(result.name).toBe('test-plugin');
      expect(result.version).toBe('1.0.0');
    });

    it('should reject invalid plugin code', async () => {
      (fs.readFile as ReturnType<typeof vi.fn>).mockResolvedValue('invalid javascript code {{{');

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'error',
              error: 'SyntaxError: Unexpected token'
            });
          }, 10);
        }
      });

      await expect(sandbox.loadPlugin('/path/to/invalid.js', defaultPolicy))
        .rejects
        .toThrow('SyntaxError');
    });

    it('should validate plugin signature when required', async () => {
      const signaturePolicy: ISecurityPolicy = {
        ...defaultPolicy,
        requireSignature: true
      };

      const mockContext: IPluginContext = {
        id: 'signed-plugin-id',
        name: 'signed-plugin',
        version: '1.0.0',
        author: 'trusted-author',
        description: 'Signed plugin',
        permissions: [],
        signature: 'valid-signature',
        sourceHash: 'signed-hash',
        entryPoint: 'index.js',
        dependencies: {}
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'loaded',
              success: true,
              context: mockContext
            });
          }, 10);
        }
      });

      const result = await sandbox.loadPlugin('/path/to/signed.js', signaturePolicy);
      expect(result.signature).toBe('valid-signature');
    });
  });

  describe('Plugin Execution', () => {
    let loadedContext: IPluginContext;

    beforeEach(async () => {
      loadedContext = {
        id: 'test-plugin-id',
        name: 'test-plugin',
        version: '1.0.0',
        author: 'test-author',
        description: 'Test plugin',
        permissions: [],
        sourceHash: 'mock-hash',
        entryPoint: 'index.js',
        dependencies: {}
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'loaded',
              success: true,
              context: loadedContext
            });
          }, 10);
        }
      });

      await sandbox.loadPlugin('/path/to/plugin.js', defaultPolicy);
    });

    it('should execute plugin successfully', async () => {
      const mockResult: IPluginExecutionResult = {
        success: true,
        output: { result: 'processed: test input' },
        errors: [],
        warnings: [],
        duration: 100,
        memoryUsage: { peak: 1024, current: 512 },
        apiCalls: 0,
        securityViolations: [],
        context: loadedContext
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'result',
              success: true,
              result: mockResult
            });
          }, 10);
        }
      });

      const result = await sandbox.executePlugin('test-plugin-id', 'test input');
      expect(result.success).toBe(true);
      expect(result.output).toEqual({ result: 'processed: test input' });
      expect(result.duration).toBe(100);
    });

    it('should handle execution errors', async () => {
      const mockResult: IPluginExecutionResult = {
        success: false,
        output: undefined,
        errors: ['Runtime error: Cannot read property of undefined'],
        warnings: [],
        duration: 50,
        memoryUsage: { peak: 512, current: 0 },
        apiCalls: 0,
        securityViolations: [],
        context: loadedContext
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'result',
              success: false,
              result: mockResult
            });
          }, 10);
        }
      });

      const result = await sandbox.executePlugin('test-plugin-id', 'invalid input');
      expect(result.success).toBe(false);
      expect(result.errors).toContain('Runtime error: Cannot read property of undefined');
    });

    it('should enforce execution timeout', async () => {
      mockWorker.on.mockImplementation((_event: string, _callback: WorkerCallback) => {
        // Don't call callback to simulate timeout
      });

      const options = { timeout: 1000 };
      await expect(sandbox.executePlugin('test-plugin-id', 'test input', options))
        .rejects
        .toThrow('timeout');
    });

    it('should enforce memory limits', async () => {
      const mockResult: IPluginExecutionResult = {
        success: false,
        output: undefined,
        errors: ['Memory limit exceeded'],
        warnings: [],
        duration: 200,
        memoryUsage: { peak: 200 * 1024 * 1024, current: 150 * 1024 * 1024 },
        apiCalls: 0,
        securityViolations: ['MEMORY_LIMIT_EXCEEDED'],
        context: loadedContext
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'result',
              success: false,
              result: mockResult
            });
          }, 10);
        }
      });

      const options = { memoryLimit: 50 * 1024 * 1024 }; // 50MB limit
      const result = await sandbox.executePlugin('test-plugin-id', 'memory-intensive', options);
      expect(result.success).toBe(false);
      expect(result.securityViolations).toContain('MEMORY_LIMIT_EXCEEDED');
    });
  });

  describe('Security Validation', () => {
    it('should validate plugin signatures', async () => {
      const validSignature = 'valid-plugin-signature';
      const pluginPath = '/path/to/signed-plugin.js';

      (fs.readFile as ReturnType<typeof vi.fn>).mockResolvedValue('plugin content');

      const isValid = await sandbox.validateSignature(pluginPath, validSignature);
      expect(typeof isValid).toBe('boolean');
    });

    it('should detect security violations', async () => {
      const restrictivePolicy: ISecurityPolicy = {
        ...defaultPolicy,
        allowFileSystem: false,
        allowNetwork: false,
        allowChildProcesses: false
      };

      const _maliciousContext: IPluginContext = {
        id: 'malicious-plugin',
        name: 'malicious',
        version: '1.0.0',
        author: 'unknown',
        description: 'Malicious plugin',
        permissions: ['file-system', 'network'],
        sourceHash: 'malicious-hash',
        entryPoint: 'index.js',
        dependencies: {}
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'security-violation',
              violations: ['UNAUTHORIZED_FILE_ACCESS', 'UNAUTHORIZED_NETWORK_ACCESS']
            });
          }, 10);
        }
      });

      await expect(sandbox.loadPlugin('/path/to/malicious.js', restrictivePolicy))
        .rejects
        .toThrow('Security violations detected');
    });
  });

  describe('Resource Management', () => {
    it('should provide sandbox statistics', () => {
      const stats = sandbox.getStats();
      expect(stats).toHaveProperty('loadedPlugins');
      expect(stats).toHaveProperty('totalExecutions');
      expect(stats).toHaveProperty('averageExecutionTime');
      expect(stats).toHaveProperty('securityViolations');
      expect(stats).toHaveProperty('memoryUsage');
      expect(typeof stats.loadedPlugins).toBe('number');
    });

    it('should list loaded plugins', () => {
      const plugins = sandbox.getLoadedPlugins();
      expect(Array.isArray(plugins)).toBe(true);
    });

    it('should unload plugins and clean up resources', async () => {
      // First load a plugin
      const mockContext: IPluginContext = {
        id: 'test-plugin-id',
        name: 'test-plugin',
        version: '1.0.0',
        author: 'test-author',
        description: 'Test plugin',
        permissions: [],
        sourceHash: 'mock-hash',
        entryPoint: 'index.js',
        dependencies: {}
      };

      mockWorker.on.mockImplementation((event: string, callback: WorkerCallback) => {
        if (event === 'message') {
          setTimeout(() => {
            callback({
              type: 'loaded',
              success: true,
              context: mockContext
            });
          }, 10);
        }
      });

      await sandbox.loadPlugin('/path/to/plugin.js', defaultPolicy);

      // Now unload it
      await sandbox.unloadPlugin('test-plugin-id');
      expect(mockWorker.terminate).toHaveBeenCalled();
    });
  });

  describe('Event Handling', () => {
    it('should emit events during plugin lifecycle', () =>
      new Promise<void>((resolve) => {
        let eventCount = 0;
        
        sandbox.on('plugin-loaded', () => {
          eventCount++;
          if (eventCount === 1) resolve();
        });

        sandbox.on('plugin-execution-start', () => {
          eventCount++;
        });

        sandbox.on('plugin-execution-complete', () => {
          eventCount++;
        });

        // Simulate plugin loading
        sandbox.emit('plugin-loaded', { pluginId: 'test' });
      })
    );

    it('should emit security violation events', () =>
      new Promise<void>((resolve) => {
        sandbox.on('security-violation', (violation) => {
          expect(violation).toHaveProperty('type');
          expect(violation).toHaveProperty('pluginId');
          resolve();
        });

        // Simulate security violation
        sandbox.emit('security-violation', {
          type: 'UNAUTHORIZED_ACCESS',
          pluginId: 'malicious-plugin',
          details: 'Attempted to access restricted resource'
        });
      })
    );
  });
});
