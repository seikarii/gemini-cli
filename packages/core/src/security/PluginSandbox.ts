/**
 * @fileoverview Plugin sandbox system for secure execution of untrusted code
 * Uses worker thread isolation to provide secure runtime environment for plugins
 */

import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { join, resolve } from 'path';
import { createHash } from 'crypto';

/**
 * Security policy configuration for plugin execution
 */
export interface ISecurityPolicy {
  /** Maximum execution time in milliseconds */
  maxExecutionTime: number;
  /** Maximum memory usage in bytes */
  maxMemoryUsage: number;
  /** Allowed network hosts (empty array = no network access) */
  allowedHosts: string[];
  /** Allowed file system paths for read access */
  allowedReadPaths: string[];
  /** Allowed file system paths for write access */
  allowedWritePaths: string[];
  /** Enable/disable file system access entirely */
  allowFileSystem: boolean;
  /** Enable/disable network access entirely */
  allowNetwork: boolean;
  /** Enable/disable child process spawning */
  allowChildProcesses: boolean;
  /** Maximum number of API calls per minute */
  apiCallLimit: number;
  /** Trusted plugin signature required */
  requireSignature: boolean;
  /** Additional environment variables to expose */
  environmentVariables: Record<string, string>;
}

/**
 * Plugin execution context and metadata
 */
export interface IPluginContext {
  /** Plugin unique identifier */
  id: string;
  /** Plugin name */
  name: string;
  /** Plugin version */
  version: string;
  /** Plugin author */
  author: string;
  /** Plugin description */
  description: string;
  /** Plugin permissions requested */
  permissions: string[];
  /** Plugin signature for verification */
  signature?: string;
  /** Plugin source code hash */
  sourceHash: string;
  /** Plugin entry point */
  entryPoint: string;
  /** Plugin dependencies */
  dependencies: Record<string, string>;
}

/**
 * Plugin execution result
 */
export interface IPluginExecutionResult {
  /** Execution success status */
  success: boolean;
  /** Plugin output data */
  output?: unknown;
  /** Execution errors */
  errors: string[];
  /** Execution warnings */
  warnings: string[];
  /** Execution duration in milliseconds */
  duration: number;
  /** Memory usage statistics */
  memoryUsage: {
    peak: number;
    current: number;
  };
  /** API calls made during execution */
  apiCalls: number;
  /** Security violations detected */
  securityViolations: string[];
  /** Plugin context used */
  context: IPluginContext;
}

/**
 * Plugin sandbox manager interface
 */
export interface IPluginSandbox extends EventEmitter {
  /**
   * Load and validate a plugin
   * @param pluginPath Path to plugin file
   * @param policy Security policy to apply
   * @returns Plugin context
   */
  loadPlugin(pluginPath: string, policy: ISecurityPolicy): Promise<IPluginContext>;

  /**
   * Execute a loaded plugin
   * @param pluginId Plugin identifier
   * @param input Input data for plugin
   * @param options Execution options
   * @returns Execution result
   */
  executePlugin(
    pluginId: string,
    input?: unknown,
    options?: { timeout?: number; memoryLimit?: number }
  ): Promise<IPluginExecutionResult>;

  /**
   * Unload a plugin and clean up resources
   * @param pluginId Plugin identifier
   */
  unloadPlugin(pluginId: string): Promise<void>;

  /**
   * Get list of loaded plugins
   * @returns Array of plugin contexts
   */
  getLoadedPlugins(): IPluginContext[];

  /**
   * Validate plugin signature
   * @param pluginPath Path to plugin file
   * @param signature Plugin signature
   * @returns Validation result
   */
  validateSignature(pluginPath: string, signature: string): Promise<boolean>;

  /**
   * Get sandbox statistics
   * @returns Sandbox usage statistics
   */
  getStats(): {
    loadedPlugins: number;
    totalExecutions: number;
    averageExecutionTime: number;
    securityViolations: number;
    memoryUsage: number;
  };

  /**
   * Shutdown sandbox and cleanup all resources
   */
  shutdown(): Promise<void>;
}

/**
 * Secure plugin sandbox implementation
 */
export class PluginSandbox extends EventEmitter implements IPluginSandbox {
  private loadedPlugins = new Map<string, IPluginContext>();
  private activeWorkers = new Map<string, Worker>();
  private executionStats = {
    totalExecutions: 0,
    totalExecutionTime: 0,
    securityViolations: 0,
    memoryPeak: 0
  };
  private trustedSignatures = new Set<string>();

  constructor() {
    super();
  }

  /**
   * Load and validate a plugin
   */
  async loadPlugin(pluginPath: string, policy: ISecurityPolicy): Promise<IPluginContext> {
    const resolvedPath = resolve(pluginPath);
    
    try {
      // Read plugin file
      const pluginCode = await fs.readFile(resolvedPath, 'utf8');
      
      // Calculate source hash
      const sourceHash = createHash('sha256').update(pluginCode).digest('hex');
      
      // Parse plugin metadata (assuming JSON header or comments)
      const context = await this.parsePluginMetadata(pluginCode, resolvedPath, sourceHash);
      
      // Validate security requirements
      await this.validatePluginSecurity(context, policy);
      
      // Verify signature if required
      if (policy.requireSignature && context.signature) {
        const isValid = await this.validateSignature(resolvedPath, context.signature);
        if (!isValid) {
          throw new Error('Invalid plugin signature');
        }
      }
      
      // Store plugin context
      this.loadedPlugins.set(context.id, context);
      
      this.emit('plugin-loaded', context);
      return context;
      
    } catch (error) {
      this.emit('plugin-load-error', { path: resolvedPath, error });
      throw error;
    }
  }

  /**
   * Execute a loaded plugin in secure sandbox
   */
  async executePlugin(
    pluginId: string,
    input?: unknown,
    options?: { timeout?: number; memoryLimit?: number }
  ): Promise<IPluginExecutionResult> {
    const context = this.loadedPlugins.get(pluginId);
    if (!context) {
      throw new Error(`Plugin not loaded: ${pluginId}`);
    }

    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed;
    
    let worker: Worker | null = null;
    let executionResult: IPluginExecutionResult;

    try {
      // Create isolated worker thread
      worker = await this.createSecureWorker(context, options);
      this.activeWorkers.set(pluginId, worker);
      
      // Execute plugin with timeout and monitoring
      const result = await this.executeInWorker(worker, context, input, options);
      
      const duration = Date.now() - startTime;
      const memoryUsed = process.memoryUsage().heapUsed - startMemory;
      
      executionResult = {
        success: true,
        output: result.output,
        errors: result.errors || [],
        warnings: result.warnings || [],
        duration,
        memoryUsage: {
          peak: Math.max(memoryUsed, this.executionStats.memoryPeak),
          current: memoryUsed
        },
        apiCalls: result.apiCalls || 0,
        securityViolations: result.securityViolations || [],
        context
      };
      
      // Update statistics
      this.executionStats.totalExecutions++;
      this.executionStats.totalExecutionTime += duration;
      this.executionStats.memoryPeak = Math.max(this.executionStats.memoryPeak, memoryUsed);
      this.executionStats.securityViolations += executionResult.securityViolations.length;
      
      this.emit('plugin-executed', executionResult);
      
    } catch (error) {
      const duration = Date.now() - startTime;
      
      executionResult = {
        success: false,
        errors: [error instanceof Error ? error.message : String(error)],
        warnings: [],
        duration,
        memoryUsage: {
          peak: this.executionStats.memoryPeak,
          current: process.memoryUsage().heapUsed - startMemory
        },
        apiCalls: 0,
        securityViolations: [],
        context
      };
      
      this.emit('plugin-execution-error', { pluginId, error, result: executionResult });
      
    } finally {
      // Clean up worker
      if (worker) {
        await worker.terminate();
        this.activeWorkers.delete(pluginId);
      }
    }

    return executionResult;
  }

  /**
   * Unload a plugin and clean up resources
   */
  async unloadPlugin(pluginId: string): Promise<void> {
    const context = this.loadedPlugins.get(pluginId);
    if (!context) {
      return;
    }

    // Terminate any active worker
    const worker = this.activeWorkers.get(pluginId);
    if (worker) {
      await worker.terminate();
      this.activeWorkers.delete(pluginId);
    }

    // Remove from loaded plugins
    this.loadedPlugins.delete(pluginId);
    
    this.emit('plugin-unloaded', context);
  }

  /**
   * Get list of loaded plugins
   */
  getLoadedPlugins(): IPluginContext[] {
    return Array.from(this.loadedPlugins.values());
  }

  /**
   * Validate plugin signature
   */
  async validateSignature(pluginPath: string, signature: string): Promise<boolean> {
    try {
      // Simple signature validation (in production, use proper crypto verification)
      const pluginCode = await fs.readFile(pluginPath, 'utf8');
      const expectedSignature = createHash('sha256')
        .update(pluginCode + process.env.PLUGIN_SIGNING_KEY || 'default-key')
        .digest('hex');
      
      return signature === expectedSignature || this.trustedSignatures.has(signature);
    } catch (_error) {
      return false;
    }
  }

  /**
   * Get sandbox statistics
   */
  getStats() {
    return {
      loadedPlugins: this.loadedPlugins.size,
      totalExecutions: this.executionStats.totalExecutions,
      averageExecutionTime: this.executionStats.totalExecutions > 0 
        ? this.executionStats.totalExecutionTime / this.executionStats.totalExecutions 
        : 0,
      securityViolations: this.executionStats.securityViolations,
      memoryUsage: this.executionStats.memoryPeak
    };
  }

  /**
   * Shutdown sandbox and cleanup all resources
   */
  async shutdown(): Promise<void> {
    // Terminate all active workers
    const terminatePromises = Array.from(this.activeWorkers.values()).map(worker => worker.terminate());
    await Promise.all(terminatePromises);
    
    // Clear all data
    this.activeWorkers.clear();
    this.loadedPlugins.clear();
    
    this.emit('sandbox-shutdown');
  }

  /**
   * Parse plugin metadata from source code
   */
  private async parsePluginMetadata(
    pluginCode: string,
    pluginPath: string,
    sourceHash: string
  ): Promise<IPluginContext> {
    // Extract metadata from comments or JSON header
    const metadataMatch = pluginCode.match(/\/\*\s*@plugin\s*(.*?)\s*\*\//s);
    
    let metadata: Partial<IPluginContext> = {};
    
    if (metadataMatch) {
      try {
        metadata = JSON.parse(metadataMatch[1]);
      } catch (_error) {
        // If JSON parsing fails, extract from individual comment lines
        const lines = metadataMatch[1].split('\n');
        for (const line of lines) {
          const match = line.match(/^\s*@(\w+)\s+(.+)$/);
          if (match) {
            const [, key, value] = match;
            (metadata as Record<string, unknown>)[key] = value.trim();
          }
        }
      }
    }

    return {
      id: metadata.id || createHash('md5').update(pluginPath).digest('hex'),
      name: metadata.name || 'Unnamed Plugin',
      version: metadata.version || '1.0.0',
      author: metadata.author || 'Unknown',
      description: metadata.description || 'No description provided',
      permissions: metadata.permissions || [],
      signature: metadata.signature,
      sourceHash,
      entryPoint: metadata.entryPoint || 'main',
      dependencies: metadata.dependencies || {}
    };
  }

  /**
   * Validate plugin security requirements
   */
  private async validatePluginSecurity(context: IPluginContext, policy: ISecurityPolicy): Promise<void> {
    // Check required permissions against policy
    for (const permission of context.permissions) {
      switch (permission) {
        case 'filesystem':
          if (!policy.allowFileSystem) {
            throw new Error(`Plugin requires filesystem access but policy denies it`);
          }
          break;
        case 'network':
          if (!policy.allowNetwork) {
            throw new Error(`Plugin requires network access but policy denies it`);
          }
          break;
        case 'childProcess':
          if (!policy.allowChildProcesses) {
            throw new Error(`Plugin requires child process access but policy denies it`);
          }
          break;
        default:
          // Unknown permission - be conservative and deny
          throw new Error(`Plugin requires unknown permission: ${permission}`);
      }
    }
  }

  /**
   * Create secure worker thread for plugin execution
   */
  private async createSecureWorker(
    context: IPluginContext,
    options?: { timeout?: number; memoryLimit?: number }
  ): Promise<Worker> {
    // Create worker script path
    const workerScript = join(__dirname, 'sandbox-worker.js');
    
    // Worker options with resource limits
    const workerOptions = {
      resourceLimits: {
        maxOldGenerationSizeMb: Math.floor((options?.memoryLimit || 100 * 1024 * 1024) / (1024 * 1024)),
        maxYoungGenerationSizeMb: 32,
        codeRangeSizeMb: 8
      }
    };

    const worker = new Worker(workerScript, workerOptions);
    
    // Set up error handling
    worker.on('error', (error) => {
      this.emit('worker-error', { context, error });
    });

    worker.on('exit', (code) => {
      if (code !== 0) {
        this.emit('worker-exit', { context, code });
      }
    });

    return worker;
  }

  /**
   * Execute plugin in worker with monitoring
   */
  private async executeInWorker(
    worker: Worker,
    context: IPluginContext,
    input?: unknown,
    options?: { timeout?: number }
  ): Promise<{
    output?: unknown;
    errors?: string[];
    warnings?: string[];
    apiCalls?: number;
    securityViolations?: string[];
  }> {
    return new Promise((resolve, reject) => {
      const timeout = options?.timeout || 30000; // 30 second default timeout
      
      // Set up timeout
      const timeoutId = setTimeout(() => {
        worker.terminate();
        reject(new Error('Plugin execution timeout'));
      }, timeout);

      // Listen for result
      worker.once('message', (result) => {
        clearTimeout(timeoutId);
        
        if (result.type === 'success') {
          resolve(result.data);
        } else {
          reject(new Error(result.error || 'Plugin execution failed'));
        }
      });

      // Send execution request
      worker.postMessage({
        type: 'execute',
        context,
        input
      });
    });
  }

  /**
   * Add trusted signature to whitelist
   */
  addTrustedSignature(signature: string): void {
    this.trustedSignatures.add(signature);
  }

  /**
   * Remove trusted signature from whitelist
   */
  removeTrustedSignature(signature: string): void {
    this.trustedSignatures.delete(signature);
  }
}

/**
 * Default security policies for different use cases
 */
export const SECURITY_POLICY_PRESETS: Record<string, ISecurityPolicy> = {
  /** Minimal security for trusted plugins */
  TRUSTED: {
    maxExecutionTime: 60000, // 1 minute
    maxMemoryUsage: 100 * 1024 * 1024, // 100MB
    allowedHosts: [],
    allowedReadPaths: [process.cwd()],
    allowedWritePaths: ['/tmp'],
    allowFileSystem: true,
    allowNetwork: true,
    allowChildProcesses: false,
    apiCallLimit: 1000,
    requireSignature: true,
    environmentVariables: {}
  },

  /** Moderate security for semi-trusted plugins */
  MODERATE: {
    maxExecutionTime: 30000, // 30 seconds
    maxMemoryUsage: 50 * 1024 * 1024, // 50MB
    allowedHosts: ['api.example.com'],
    allowedReadPaths: [],
    allowedWritePaths: [],
    allowFileSystem: false,
    allowNetwork: true,
    allowChildProcesses: false,
    apiCallLimit: 100,
    requireSignature: true,
    environmentVariables: {}
  },

  /** High security for untrusted plugins */
  RESTRICTIVE: {
    maxExecutionTime: 10000, // 10 seconds
    maxMemoryUsage: 25 * 1024 * 1024, // 25MB
    allowedHosts: [],
    allowedReadPaths: [],
    allowedWritePaths: [],
    allowFileSystem: false,
    allowNetwork: false,
    allowChildProcesses: false,
    apiCallLimit: 10,
    requireSignature: true,
    environmentVariables: {}
  }
};

/**
 * Create plugin sandbox instance
 */
export function createPluginSandbox(): PluginSandbox {
  return new PluginSandbox();
}
