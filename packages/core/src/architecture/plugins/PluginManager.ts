/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'events';
import {
  IPlugin,
  IPluginManager,
  PluginConfig,
  PluginContext,
  PluginManagerConfig,
  PluginRegistryEntry,
  PluginHealthStatus,
  HookExecutionResult,
  PluginLifecycleStage,
  PluginPriority,
} from './PluginInterfaces.js';

/**
 * Default plugin configuration
 */
const DEFAULT_PLUGIN_CONFIG: PluginConfig = {
  enabled: true,
  priority: PluginPriority.NORMAL,
  autoStart: true,
  retryAttempts: 3,
  timeout: 30000,
};

/**
 * Default plugin manager configuration
 */
const DEFAULT_MANAGER_CONFIG: PluginManagerConfig = {
  maxPlugins: 100,
  defaultTimeout: 30000,
  enableHotReload: false,
  enableSandbox: true,
  allowRemotePlugins: false,
  pluginDirectories: ['./plugins'],
  autoDiscovery: true,
  healthCheckInterval: 60000, // 1 minute
  retryAttempts: 3,
};

/**
 * Mutable plugin interface for internal stage management
 */
interface MutablePlugin extends IPlugin {
  stage: PluginLifecycleStage;
}

/**
 * Plugin manager implementation with advanced lifecycle management
 */
export class PluginManager extends EventEmitter implements IPluginManager {
  private readonly registry = new Map<string, PluginRegistryEntry>();
  private readonly hooks = new Map<string, Map<string, IPlugin>>();
  private readonly config: PluginManagerConfig;
  private healthMonitorInterval?: NodeJS.Timeout;
  private readonly startTime = Date.now();
  private hookExecutions = 0;
  private pluginStorage: Record<string, unknown> = {};

  constructor(config: Partial<PluginManagerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_MANAGER_CONFIG, ...config };
    
    if (this.config.healthCheckInterval && this.config.healthCheckInterval > 0) {
      this.startHealthMonitoring();
    }
  }

  /**
   * Register a plugin with the manager
   */
  async register(plugin: IPlugin, config: Partial<PluginConfig> = {}): Promise<void> {
    const pluginId = plugin.metadata.id;

    // Check if plugin already exists
    if (this.registry.has(pluginId)) {
      throw new Error(`Plugin '${pluginId}' is already registered`);
    }

    // Check max plugins limit
    if (this.config.maxPlugins && this.registry.size >= this.config.maxPlugins) {
      throw new Error(`Maximum number of plugins (${this.config.maxPlugins}) reached`);
    }

    // Validate plugin
    await this.validatePlugin(plugin);

    // Create plugin configuration
    const pluginConfig: PluginConfig = { ...DEFAULT_PLUGIN_CONFIG, ...config };

    // Create plugin context
    const context = this.createPluginContext(pluginId, pluginConfig);

    // Create registry entry
    const entry: PluginRegistryEntry = {
      plugin,
      config: pluginConfig,
      context,
      health: {
        healthy: true,
        lastCheck: new Date(),
        errors: [],
        warnings: [],
      },
      statistics: {
        activationCount: 0,
        deactivationCount: 0,
        errorCount: 0,
      },
    };

    // Register the plugin
    this.registry.set(pluginId, entry);

    // Register plugin hooks
    for (const [hookName] of plugin.hooks) {
      this.registerHookInternal(hookName, plugin);
    }

    // Emit registration event
    this.emit('plugin.registered', plugin);

    // Auto-initialize if enabled
    if (pluginConfig.autoStart) {
      try {
        await this.initializePlugin(pluginId);
        await this.activate(pluginId);
      } catch (error) {
        console.error(`Failed to auto-start plugin '${pluginId}':`, error);
        entry.statistics.errorCount++;
        entry.statistics.lastError = new Date();
        this.emit('plugin.error', plugin, error as Error);
      }
    }
  }

  /**
   * Unregister a plugin
   */
  async unregister(pluginId: string): Promise<void> {
    const entry = this.registry.get(pluginId);
    if (!entry) {
      throw new Error(`Plugin '${pluginId}' is not registered`);
    }

    try {
      // Deactivate if active
      if (entry.plugin.stage === PluginLifecycleStage.ACTIVE) {
        await this.deactivate(pluginId);
      }

      // Destroy the plugin
      await entry.plugin.destroy();

      // Unregister hooks
      for (const hookName of entry.plugin.hooks.keys()) {
        this.unregisterHookInternal(hookName, pluginId);
      }

      // Remove from registry
      this.registry.delete(pluginId);

      // Emit unregistration event
      this.emit('plugin.unregistered', pluginId);
    } catch (error) {
      console.error(`Error unregistering plugin '${pluginId}':`, error);
      throw error;
    }
  }

  /**
   * Get a plugin by ID
   */
  getPlugin(pluginId: string): IPlugin | null {
    return this.registry.get(pluginId)?.plugin || null;
  }

  /**
   * Get all plugins
   */
  getPlugins(): IPlugin[] {
    return Array.from(this.registry.values()).map(entry => entry.plugin);
  }

  /**
   * Get plugins by tag
   */
  getPluginsByTag(tag: string): IPlugin[] {
    return this.getPlugins().filter(plugin => 
      plugin.metadata.tags?.includes(tag)
    );
  }

  /**
   * Get active plugins
   */
  getActivePlugins(): IPlugin[] {
    return this.getPlugins().filter(plugin => 
      plugin.stage === PluginLifecycleStage.ACTIVE
    );
  }

  /**
   * Initialize all plugins
   */
  async initializeAll(): Promise<void> {
    const plugins = Array.from(this.registry.keys());
    const results = await Promise.allSettled(
      plugins.map(pluginId => this.initializePlugin(pluginId))
    );

    // Log any failures
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Failed to initialize plugin '${plugins[index]}':`, result.reason);
      }
    });
  }

  /**
   * Activate a plugin
   */
  async activate(pluginId: string): Promise<void> {
    const entry = this.registry.get(pluginId);
    if (!entry) {
      throw new Error(`Plugin '${pluginId}' is not registered`);
    }

    if (!entry.config.enabled) {
      throw new Error(`Plugin '${pluginId}' is disabled`);
    }

    if (entry.plugin.stage === PluginLifecycleStage.ACTIVE) {
      return; // Already active
    }

    try {
      // Initialize if not already initialized
      if (entry.plugin.stage === PluginLifecycleStage.CREATED) {
        await this.initializePlugin(pluginId);
      }

      // Set activating stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.ACTIVATING;

      // Activate with timeout
      await this.withTimeout(
        entry.plugin.activate(),
        entry.config.timeout || this.config.defaultTimeout!
      );

      // Set active stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.ACTIVE;

      // Update statistics
      entry.statistics.activationCount++;
      entry.statistics.lastActivated = new Date();

      // Emit activation event
      this.emit('plugin.activated', entry.plugin);
    } catch (error) {
      // Set error stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.ERROR;
      entry.statistics.errorCount++;
      entry.statistics.lastError = new Date();
      entry.health.healthy = false;
      entry.health.errors.push(error instanceof Error ? error.message : String(error));

      this.emit('plugin.error', entry.plugin, error as Error);
      throw error;
    }
  }

  /**
   * Deactivate a plugin
   */
  async deactivate(pluginId: string): Promise<void> {
    const entry = this.registry.get(pluginId);
    if (!entry) {
      throw new Error(`Plugin '${pluginId}' is not registered`);
    }

    if (entry.plugin.stage !== PluginLifecycleStage.ACTIVE) {
      return; // Not active
    }

    try {
      // Set deactivating stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.DEACTIVATING;

      // Deactivate with timeout
      await this.withTimeout(
        entry.plugin.deactivate(),
        entry.config.timeout || this.config.defaultTimeout!
      );

      // Set deactivated stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.DEACTIVATED;

      // Update statistics
      entry.statistics.deactivationCount++;
      entry.statistics.lastDeactivated = new Date();

      // Emit deactivation event
      this.emit('plugin.deactivated', entry.plugin);
    } catch (error) {
      // Set error stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.ERROR;
      entry.statistics.errorCount++;
      entry.statistics.lastError = new Date();
      entry.health.healthy = false;
      entry.health.errors.push(error instanceof Error ? error.message : String(error));

      this.emit('plugin.error', entry.plugin, error as Error);
      throw error;
    }
  }

  /**
   * Activate all plugins
   */
  async activateAll(): Promise<void> {
    const plugins = Array.from(this.registry.keys());
    const results = await Promise.allSettled(
      plugins.map(pluginId => this.activate(pluginId))
    );

    // Log any failures
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Failed to activate plugin '${plugins[index]}':`, result.reason);
      }
    });
  }

  /**
   * Deactivate all plugins
   */
  async deactivateAll(): Promise<void> {
    const plugins = Array.from(this.registry.keys());
    const results = await Promise.allSettled(
      plugins.map(pluginId => this.deactivate(pluginId))
    );

    // Log any failures
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Failed to deactivate plugin '${plugins[index]}':`, result.reason);
      }
    });
  }

  /**
   * Execute hooks
   */
  async executeHook(hookName: string, ...args: unknown[]): Promise<HookExecutionResult[]> {
    const hookPlugins = this.hooks.get(hookName);
    if (!hookPlugins || hookPlugins.size === 0) {
      return [];
    }

    const results: HookExecutionResult[] = [];

    for (const [pluginId, plugin] of hookPlugins) {
      const entry = this.registry.get(pluginId);
      if (!entry || plugin.stage !== PluginLifecycleStage.ACTIVE) {
        continue;
      }

      const hook = plugin.hooks.get(hookName);
      if (!hook) {
        continue;
      }

      const startTime = Date.now();
      let result: HookExecutionResult;

      try {
        const hookResult = await hook.handler(entry.context, ...args);
        result = {
          hookName,
          pluginId,
          success: true,
          result: hookResult,
          executionTime: Date.now() - startTime,
        };

        this.emit('hook.executed', hookName, plugin, hookResult);
      } catch (error) {
        result = {
          hookName,
          pluginId,
          success: false,
          error: error as Error,
          executionTime: Date.now() - startTime,
        };

        this.emit('hook.error', hookName, plugin, error as Error);
      }

      results.push(result);
      this.hookExecutions++;
    }

    return results;
  }

  /**
   * Execute hooks with priority order
   */
  async executeHookOrdered(hookName: string, ...args: unknown[]): Promise<HookExecutionResult[]> {
    const hookPlugins = this.hooks.get(hookName);
    if (!hookPlugins || hookPlugins.size === 0) {
      return [];
    }

    // Sort plugins by hook priority
    const sortedPlugins = Array.from(hookPlugins.entries())
      .map(([pluginId, plugin]) => {
        const hook = plugin.hooks.get(hookName);
        return { pluginId, plugin, priority: hook?.priority || PluginPriority.NORMAL };
      })
      .sort((a, b) => b.priority - a.priority);

    const results: HookExecutionResult[] = [];

    for (const { pluginId, plugin } of sortedPlugins) {
      const entry = this.registry.get(pluginId);
      if (!entry || plugin.stage !== PluginLifecycleStage.ACTIVE) {
        continue;
      }

      const hook = plugin.hooks.get(hookName);
      if (!hook) {
        continue;
      }

      const startTime = Date.now();
      let result: HookExecutionResult;

      try {
        const hookResult = await hook.handler(entry.context, ...args);
        result = {
          hookName,
          pluginId,
          success: true,
          result: hookResult,
          executionTime: Date.now() - startTime,
        };

        this.emit('hook.executed', hookName, plugin, hookResult);
      } catch (error) {
        result = {
          hookName,
          pluginId,
          success: false,
          error: error as Error,
          executionTime: Date.now() - startTime,
        };

        this.emit('hook.error', hookName, plugin, error as Error);
      }

      results.push(result);
      this.hookExecutions++;
    }

    return results;
  }

  /**
   * Execute hooks until first success
   */
  async executeHookUntilSuccess(hookName: string, ...args: unknown[]): Promise<HookExecutionResult | null> {
    const results = await this.executeHookOrdered(hookName, ...args);
    return results.find(result => result.success) || null;
  }

  /**
   * Get plugin health
   */
  async getPluginHealth(pluginId: string): Promise<PluginHealthStatus | null> {
    const entry = this.registry.get(pluginId);
    if (!entry) {
      return null;
    }

    try {
      const health = await entry.plugin.getHealth();
      entry.health = health;
      return health;
    } catch (error) {
      const errorHealth: PluginHealthStatus = {
        healthy: false,
        lastCheck: new Date(),
        errors: [error instanceof Error ? error.message : String(error)],
        warnings: [],
      };
      entry.health = errorHealth;
      return errorHealth;
    }
  }

  /**
   * Get all plugin health statuses
   */
  async getAllHealth(): Promise<Map<string, PluginHealthStatus>> {
    const healthMap = new Map<string, PluginHealthStatus>();

    const healthPromises = Array.from(this.registry.keys()).map(async pluginId => {
      const health = await this.getPluginHealth(pluginId);
      if (health) {
        healthMap.set(pluginId, health);
      }
    });

    await Promise.allSettled(healthPromises);
    return healthMap;
  }

  /**
   * Start health monitoring
   */
  startHealthMonitoring(): void {
    if (this.healthMonitorInterval) {
      return; // Already started
    }

    this.healthMonitorInterval = setInterval(async () => {
      const healthMap = await this.getAllHealth();
      
      for (const [pluginId, health] of healthMap) {
        const entry = this.registry.get(pluginId);
        if (entry) {
          const wasHealthy = entry.health.healthy;
          entry.health = health;
          
          if (wasHealthy !== health.healthy) {
            this.emit('plugin.health.changed', entry.plugin, health);
          }
        }
      }
    }, this.config.healthCheckInterval);
  }

  /**
   * Stop health monitoring
   */
  stopHealthMonitoring(): void {
    if (this.healthMonitorInterval) {
      clearInterval(this.healthMonitorInterval);
      this.healthMonitorInterval = undefined;
    }
  }

  /**
   * Get manager statistics
   */
  getStatistics() {
    const plugins = Array.from(this.registry.values());
    const activePlugins = plugins.filter(entry => 
      entry.plugin.stage === PluginLifecycleStage.ACTIVE
    );
    const errorPlugins = plugins.filter(entry => 
      entry.plugin.stage === PluginLifecycleStage.ERROR
    );
    const totalHooks = Array.from(this.hooks.values())
      .reduce((sum, pluginMap) => sum + pluginMap.size, 0);

    return {
      totalPlugins: this.registry.size,
      activePlugins: activePlugins.length,
      errorPlugins: errorPlugins.length,
      totalHooks,
      hookExecutions: this.hookExecutions,
      uptime: Date.now() - this.startTime,
    };
  }

  /**
   * Initialize a plugin
   */
  private async initializePlugin(pluginId: string): Promise<void> {
    const entry = this.registry.get(pluginId);
    if (!entry) {
      throw new Error(`Plugin '${pluginId}' is not registered`);
    }

    if (entry.plugin.stage !== PluginLifecycleStage.CREATED) {
      return; // Already initialized
    }

    try {
      // Set initializing stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.INITIALIZING;

      // Initialize with timeout
      await this.withTimeout(
        entry.plugin.initialize(entry.context),
        entry.config.timeout || this.config.defaultTimeout!
      );

      // Set initialized stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.INITIALIZED;

      // Emit initialization event
      this.emit('plugin.initialized', entry.plugin);
    } catch (error) {
      // Set error stage
      (entry.plugin as MutablePlugin).stage = PluginLifecycleStage.ERROR;
      entry.statistics.errorCount++;
      entry.statistics.lastError = new Date();
      entry.health.healthy = false;
      entry.health.errors.push(error instanceof Error ? error.message : String(error));

      this.emit('plugin.error', entry.plugin, error as Error);
      throw error;
    }
  }

  /**
   * Create plugin context
   */
  private createPluginContext(pluginId: string, config: PluginConfig): PluginContext {
    return {
      config,
      logger: {
        debug: (message: string, ...args: unknown[]) => console.debug(`[${pluginId}]`, message, ...args),
        info: (message: string, ...args: unknown[]) => console.info(`[${pluginId}]`, message, ...args),
        warn: (message: string, ...args: unknown[]) => console.warn(`[${pluginId}]`, message, ...args),
        error: (message: string, ...args: unknown[]) => console.error(`[${pluginId}]`, message, ...args),
      },
      events: {
        emit: (event: string, data?: unknown) => this.emit(`plugin.${pluginId}.${event}`, data),
        on: (event: string, handler: (data?: unknown) => void) => this.on(`plugin.${pluginId}.${event}`, handler),
        off: (event: string, handler: (data?: unknown) => void) => this.off(`plugin.${pluginId}.${event}`, handler),
      },
      storage: {
        get: async <T>(key: string): Promise<T | null> => {
          const storageKey = `plugin.${pluginId}.${key}`;
          const stored = this.pluginStorage[storageKey] as T;
          return stored || null;
        },
        set: async <T>(key: string, value: T): Promise<void> => {
          const storageKey = `plugin.${pluginId}.${key}`;
          this.pluginStorage[storageKey] = value;
        },
        delete: async (key: string): Promise<void> => {
          const storageKey = `plugin.${pluginId}.${key}`;
          delete this.pluginStorage[storageKey];
        },
        clear: async (): Promise<void> => {
          const prefix = `plugin.${pluginId}.`;
          for (const key of Object.keys(this.pluginStorage)) {
            if (key.startsWith(prefix)) {
              delete this.pluginStorage[key];
            }
          }
        },
      },
      api: {
        callPlugin: async (targetPluginId: string, method: string, ...args: unknown[]): Promise<unknown> => {
          const targetPlugin = this.getPlugin(targetPluginId);
          if (!targetPlugin) {
            throw new Error(`Plugin '${targetPluginId}' not found`);
          }
          
          // Type-safe method calling - this is a simplified version
          // In a real implementation, you'd want proper method registration/discovery
          const pluginWithMethods = targetPlugin as unknown as Record<string, unknown>;
          const methodFn = pluginWithMethods[method];
          
          if (typeof methodFn !== 'function') {
            throw new Error(`Method '${method}' not found on plugin '${targetPluginId}'`);
          }
          
          return (methodFn as (...args: unknown[]) => unknown)(...args);
        },
        getPlugin: (targetPluginId: string) => this.getPlugin(targetPluginId),
        getPlugins: () => this.getPlugins(),
      },
    };
  }

  /**
   * Validate plugin
   */
  private async validatePlugin(plugin: IPlugin): Promise<void> {
    if (!plugin.metadata?.id) {
      throw new Error('Plugin must have a valid ID');
    }

    if (!plugin.metadata?.name) {
      throw new Error('Plugin must have a valid name');
    }

    if (!plugin.metadata?.version) {
      throw new Error('Plugin must have a valid version');
    }

    if (typeof plugin.initialize !== 'function') {
      throw new Error('Plugin must implement initialize method');
    }

    if (typeof plugin.activate !== 'function') {
      throw new Error('Plugin must implement activate method');
    }

    if (typeof plugin.deactivate !== 'function') {
      throw new Error('Plugin must implement deactivate method');
    }

    if (typeof plugin.destroy !== 'function') {
      throw new Error('Plugin must implement destroy method');
    }

    if (typeof plugin.getHealth !== 'function') {
      throw new Error('Plugin must implement getHealth method');
    }
  }

  /**
   * Register hook internally
   */
  private registerHookInternal(hookName: string, plugin: IPlugin): void {
    if (!this.hooks.has(hookName)) {
      this.hooks.set(hookName, new Map());
    }
    this.hooks.get(hookName)!.set(plugin.metadata.id, plugin);
  }

  /**
   * Unregister hook internally
   */
  private unregisterHookInternal(hookName: string, pluginId: string): void {
    const hookPlugins = this.hooks.get(hookName);
    if (hookPlugins) {
      hookPlugins.delete(pluginId);
      if (hookPlugins.size === 0) {
        this.hooks.delete(hookName);
      }
    }
  }

  /**
   * Execute with timeout
   */
  private async withTimeout<T>(promise: Promise<T>, timeout: number): Promise<T> {
    return Promise.race([
      promise,
      new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error(`Operation timed out after ${timeout}ms`)), timeout)
      ),
    ]);
  }

  /**
   * Cleanup on destroy
   */
  async destroy(): Promise<void> {
    this.stopHealthMonitoring();
    await this.deactivateAll();
    
    // Clear all plugins
    for (const pluginId of Array.from(this.registry.keys())) {
      await this.unregister(pluginId);
    }
    
    this.registry.clear();
    this.hooks.clear();
    this.removeAllListeners();
  }
}
