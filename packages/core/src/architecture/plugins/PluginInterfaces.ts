/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Plugin lifecycle stages
 */
export enum PluginLifecycleStage {
  CREATED = 'created',
  INITIALIZING = 'initializing',
  INITIALIZED = 'initialized',
  ACTIVATING = 'activating',
  ACTIVE = 'active',
  DEACTIVATING = 'deactivating',
  DEACTIVATED = 'deactivated',
  ERROR = 'error',
  DESTROYED = 'destroyed',
}

/**
 * Plugin priorities for execution order
 */
export enum PluginPriority {
  HIGHEST = 100,
  HIGH = 75,
  NORMAL = 50,
  LOW = 25,
  LOWEST = 0,
}

/**
 * Plugin metadata
 */
export interface PluginMetadata {
  id: string;
  name: string;
  version: string;
  description?: string;
  author?: string;
  tags?: string[];
  dependencies?: string[];
  optionalDependencies?: string[];
  conflicts?: string[];
  minCoreVersion?: string;
  maxCoreVersion?: string;
}

/**
 * Plugin configuration
 */
export interface PluginConfig {
  enabled: boolean;
  priority: PluginPriority;
  settings?: Record<string, unknown>;
  autoStart?: boolean;
  retryAttempts?: number;
  timeout?: number;
}

/**
 * Plugin execution context
 */
export interface PluginContext {
  config: PluginConfig;
  logger: {
    debug: (message: string, ...args: unknown[]) => void;
    info: (message: string, ...args: unknown[]) => void;
    warn: (message: string, ...args: unknown[]) => void;
    error: (message: string, ...args: unknown[]) => void;
  };
  events: {
    emit: (event: string, data?: unknown) => void;
    on: (event: string, handler: (data?: unknown) => void) => void;
    off: (event: string, handler: (data?: unknown) => void) => void;
  };
  storage: {
    get: <T>(key: string) => Promise<T | null>;
    set: <T>(key: string, value: T) => Promise<void>;
    delete: (key: string) => Promise<void>;
    clear: () => Promise<void>;
  };
  api: {
    callPlugin: (pluginId: string, method: string, ...args: unknown[]) => Promise<unknown>;
    getPlugin: (pluginId: string) => IPlugin | null;
    getPlugins: () => IPlugin[];
  };
}

/**
 * Plugin hook for extending functionality
 */
export interface PluginHook {
  name: string;
  priority: PluginPriority;
  handler: (context: PluginContext, ...args: unknown[]) => Promise<unknown> | unknown;
}

/**
 * Plugin interface that all plugins must implement
 */
export interface IPlugin {
  readonly metadata: PluginMetadata;
  readonly stage: PluginLifecycleStage;
  readonly hooks: Map<string, PluginHook>;

  /**
   * Initialize the plugin with context
   */
  initialize(context: PluginContext): Promise<void>;

  /**
   * Activate the plugin
   */
  activate(): Promise<void>;

  /**
   * Deactivate the plugin
   */
  deactivate(): Promise<void>;

  /**
   * Destroy the plugin and clean up resources
   */
  destroy(): Promise<void>;

  /**
   * Get plugin health status
   */
  getHealth(): Promise<PluginHealthStatus>;

  /**
   * Register a hook
   */
  registerHook(hook: PluginHook): void;

  /**
   * Unregister a hook
   */
  unregisterHook(hookName: string): void;

  /**
   * Handle configuration changes
   */
  onConfigChange?(newConfig: PluginConfig): Promise<void>;

  /**
   * Handle dependency updates
   */
  onDependencyUpdate?(dependency: string, version: string): Promise<void>;
}

/**
 * Plugin health status
 */
export interface PluginHealthStatus {
  healthy: boolean;
  lastCheck: Date;
  errors: string[];
  warnings: string[];
  metrics?: {
    uptime: number;
    memoryUsage?: number;
    executionCount?: number;
    errorCount?: number;
    lastExecution?: Date;
  };
}

/**
 * Plugin registry entry
 */
export interface PluginRegistryEntry {
  plugin: IPlugin;
  config: PluginConfig;
  context: PluginContext;
  health: PluginHealthStatus;
  statistics: {
    activationCount: number;
    deactivationCount: number;
    errorCount: number;
    lastActivated?: Date;
    lastDeactivated?: Date;
    lastError?: Date;
  };
}

/**
 * Plugin manager events
 */
export interface PluginManagerEvents {
  'plugin.registered': (plugin: IPlugin) => void;
  'plugin.unregistered': (pluginId: string) => void;
  'plugin.initialized': (plugin: IPlugin) => void;
  'plugin.activated': (plugin: IPlugin) => void;
  'plugin.deactivated': (plugin: IPlugin) => void;
  'plugin.error': (plugin: IPlugin, error: Error) => void;
  'plugin.health.changed': (plugin: IPlugin, health: PluginHealthStatus) => void;
  'hook.executed': (hookName: string, plugin: IPlugin, result: unknown) => void;
  'hook.error': (hookName: string, plugin: IPlugin, error: Error) => void;
}

/**
 * Plugin loader interface for dynamic plugin loading
 */
export interface IPluginLoader {
  /**
   * Load plugin from path
   */
  loadFromPath(path: string): Promise<IPlugin>;

  /**
   * Load plugin from URL
   */
  loadFromUrl(url: string): Promise<IPlugin>;

  /**
   * Load plugin from source code
   */
  loadFromSource(source: string, metadata: PluginMetadata): Promise<IPlugin>;

  /**
   * Validate plugin before loading
   */
  validatePlugin(plugin: IPlugin): Promise<boolean>;

  /**
   * Get supported plugin formats
   */
  getSupportedFormats(): string[];
}

/**
 * Plugin dependency resolver
 */
export interface IDependencyResolver {
  /**
   * Resolve plugin dependencies
   */
  resolveDependencies(plugin: IPlugin): Promise<string[]>;

  /**
   * Check if dependencies are satisfied
   */
  areDependenciesSatisfied(plugin: IPlugin): Promise<boolean>;

  /**
   * Get dependency tree
   */
  getDependencyTree(plugin: IPlugin): Promise<Map<string, string[]>>;

  /**
   * Detect circular dependencies
   */
  detectCircularDependencies(plugins: IPlugin[]): Promise<string[][]>;
}

/**
 * Hook execution result
 */
export interface HookExecutionResult {
  hookName: string;
  pluginId: string;
  success: boolean;
  result?: unknown;
  error?: Error;
  executionTime: number;
}

/**
 * Plugin manager configuration
 */
export interface PluginManagerConfig {
  maxPlugins?: number;
  defaultTimeout?: number;
  enableHotReload?: boolean;
  enableSandbox?: boolean;
  allowRemotePlugins?: boolean;
  pluginDirectories?: string[];
  autoDiscovery?: boolean;
  healthCheckInterval?: number;
  retryAttempts?: number;
}

/**
 * Plugin manager interface
 */
export interface IPluginManager {
  /**
   * Register a plugin
   */
  register(plugin: IPlugin, config?: Partial<PluginConfig>): Promise<void>;

  /**
   * Unregister a plugin
   */
  unregister(pluginId: string): Promise<void>;

  /**
   * Get a plugin by ID
   */
  getPlugin(pluginId: string): IPlugin | null;

  /**
   * Get all plugins
   */
  getPlugins(): IPlugin[];

  /**
   * Get plugins by tag
   */
  getPluginsByTag(tag: string): IPlugin[];

  /**
   * Get active plugins
   */
  getActivePlugins(): IPlugin[];

  /**
   * Initialize all plugins
   */
  initializeAll(): Promise<void>;

  /**
   * Activate a plugin
   */
  activate(pluginId: string): Promise<void>;

  /**
   * Deactivate a plugin
   */
  deactivate(pluginId: string): Promise<void>;

  /**
   * Activate all plugins
   */
  activateAll(): Promise<void>;

  /**
   * Deactivate all plugins
   */
  deactivateAll(): Promise<void>;

  /**
   * Execute hooks
   */
  executeHook(hookName: string, ...args: unknown[]): Promise<HookExecutionResult[]>;

  /**
   * Execute hooks with priority order
   */
  executeHookOrdered(hookName: string, ...args: unknown[]): Promise<HookExecutionResult[]>;

  /**
   * Execute hooks until first success
   */
  executeHookUntilSuccess(hookName: string, ...args: unknown[]): Promise<HookExecutionResult | null>;

  /**
   * Get plugin health
   */
  getPluginHealth(pluginId: string): Promise<PluginHealthStatus | null>;

  /**
   * Get all plugin health statuses
   */
  getAllHealth(): Promise<Map<string, PluginHealthStatus>>;

  /**
   * Start health monitoring
   */
  startHealthMonitoring(): void;

  /**
   * Stop health monitoring
   */
  stopHealthMonitoring(): void;

  /**
   * Get manager statistics
   */
  getStatistics(): {
    totalPlugins: number;
    activePlugins: number;
    errorPlugins: number;
    totalHooks: number;
    hookExecutions: number;
    uptime: number;
  };

  /**
   * Event management
   */
  on<K extends keyof PluginManagerEvents>(event: K, handler: PluginManagerEvents[K]): void;
  off<K extends keyof PluginManagerEvents>(event: K, handler: PluginManagerEvents[K]): void;
  emit<K extends keyof PluginManagerEvents>(event: K, ...args: Parameters<PluginManagerEvents[K]>): void;
}
