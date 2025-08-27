/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { render } from 'ink';
import { AppWrapper } from './ui/App.js';
import { loadCliConfig, parseArguments } from './config/config.js';
import { basename } from 'node:path';
import v8 from 'node:v8';
import os from 'node:os';
import dns from 'node:dns';
import { spawn } from 'node:child_process';
import { start_sandbox } from './utils/sandbox.js';
import {
  DnsResolutionOrder,
  LoadedSettings,
  loadSettings,
  SettingScope,
} from './config/settings.js';
import { themeManager } from './ui/themes/theme-manager.js';
import { getStartupWarnings } from './utils/startupWarnings.js';
import { getUserStartupWarnings } from './utils/userStartupWarnings.js';
import { ConsolePatcher } from './ui/utils/ConsolePatcher.js';
import { runNonInteractive } from './nonInteractiveCli.js';
import { loadExtensions } from './config/extension.js';
import { cleanupCheckpoints, registerCleanup } from './utils/cleanup.js';
import { getCliVersion } from './utils/version.js';
import {
  Config,
  sessionId,
  logUserPrompt,
  getOauthClient,
  AuthType,
  logIdeConnection,
  IdeConnectionEvent,
  IdeConnectionType,
} from '@google/gemini-cli-core';
import { validateAuthMethod } from './config/auth.js';
import { setMaxSizedBoxDebugging } from './ui/components/shared/MaxSizedBox.js';
import { validateNonInteractiveAuth } from './validateNonInterActiveAuth.js';
import { detectAndEnableKittyProtocol } from './ui/utils/kittyProtocolDetector.js';
import { checkForUpdates } from './ui/utils/updateCheck.js';
import { handleAutoUpdate } from './utils/handleAutoUpdate.js';
import { appEvents, AppEvent } from './utils/events.js';
import { SettingsContext } from './ui/contexts/SettingsContext.js';

// ===========================
// PERFORMANCE OPTIMIZATIONS
// ===========================

// Optimized logging system with structured output and async capabilities
class OptimizedLogger {
  private static instance: OptimizedLogger;
  private debugMode = false;
  private logQueue: Array<{
    level: string;
    message: string;
    timestamp: number;
  }> = [];
  private isFlushingLogs = false;

  static getInstance(): OptimizedLogger {
    if (!OptimizedLogger.instance) {
      OptimizedLogger.instance = new OptimizedLogger();
    }
    return OptimizedLogger.instance;
  }

  setDebugMode(debug: boolean): void {
    this.debugMode = debug;
  }

  private async flushLogs(): Promise<void> {
    if (this.isFlushingLogs || this.logQueue.length === 0) return;

    this.isFlushingLogs = true;
    const logsToFlush = [...this.logQueue];
    this.logQueue = [];

    try {
      // Batch write logs for better performance
      for (const log of logsToFlush) {
        const timestamp = new Date(log.timestamp).toISOString();
        const message = `[${timestamp}] [${log.level}] ${log.message}`;

        if (log.level === 'ERROR') {
          console.error(message);
        } else if (log.level === 'WARN') {
          console.warn(message);
        } else if (log.level === 'DEBUG' && this.debugMode) {
          console.debug(message);
        } else if (log.level === 'INFO') {
          console.log(message);
        }
      }
    } finally {
      this.isFlushingLogs = false;
    }
  }

  private queueLog(level: string, message: string): void {
    this.logQueue.push({
      level,
      message,
      timestamp: Date.now(),
    });

    // Async flush to avoid blocking main thread
    process.nextTick(() => this.flushLogs());
  }

  debug(message: string): void {
    if (this.debugMode) {
      this.queueLog('DEBUG', message);
    }
  }

  info(message: string): void {
    this.queueLog('INFO', message);
  }

  warn(message: string): void {
    this.queueLog('WARN', message);
  }

  error(message: string): void {
    this.queueLog('ERROR', message);
  }
}

// Optimized memory management with caching and better heuristics
class MemoryManager {
  private static memoryStatsCache: {
    totalMemoryMB: number;
    currentMaxOldSpaceSizeMb: number;
    targetMaxOldSpaceSizeInMB: number;
    timestamp: number;
  } | null = null;

  private static readonly CACHE_TTL_MS = 30000; // 30 seconds cache

  static getOptimizedMemoryArgs(config: Config): string[] {
    const now = Date.now();

    // Use cached values if still valid
    if (
      MemoryManager.memoryStatsCache &&
      now - MemoryManager.memoryStatsCache.timestamp <
        MemoryManager.CACHE_TTL_MS
    ) {
      const { currentMaxOldSpaceSizeMb, targetMaxOldSpaceSizeInMB } =
        MemoryManager.memoryStatsCache;

      if (config.getDebugMode()) {
        OptimizedLogger.getInstance().debug(
          `Using cached memory stats: current=${currentMaxOldSpaceSizeMb}MB, target=${targetMaxOldSpaceSizeInMB}MB`,
        );
      }

      return MemoryManager.shouldRelaunch(
        currentMaxOldSpaceSizeMb,
        targetMaxOldSpaceSizeInMB,
      )
        ? [`--max-old-space-size=${targetMaxOldSpaceSizeInMB}`]
        : [];
    }

    // Calculate fresh memory stats
    const totalMemoryMB = os.totalmem() / (1024 * 1024);
    const heapStats = v8.getHeapStatistics();
    const currentMaxOldSpaceSizeMb = Math.floor(
      heapStats.heap_size_limit / 1024 / 1024,
    );

    // Optimized memory target: 50% of total memory, but with minimum and maximum bounds
    const minMemoryMB = 1024; // 1GB minimum
    const maxMemoryMB = Math.min(8192, totalMemoryMB * 0.8); // 8GB maximum or 80% of total
    const targetMaxOldSpaceSizeInMB = Math.max(
      minMemoryMB,
      Math.min(maxMemoryMB, Math.floor(totalMemoryMB * 0.5)),
    );

    // Cache the calculated values
    MemoryManager.memoryStatsCache = {
      totalMemoryMB,
      currentMaxOldSpaceSizeMb,
      targetMaxOldSpaceSizeInMB,
      timestamp: now,
    };

    if (config.getDebugMode()) {
      OptimizedLogger.getInstance().debug(
        `Memory analysis: total=${totalMemoryMB.toFixed(0)}MB, current=${currentMaxOldSpaceSizeMb}MB, target=${targetMaxOldSpaceSizeInMB}MB`,
      );
    }

    if (process.env['GEMINI_CLI_NO_RELAUNCH']) {
      return [];
    }

    return MemoryManager.shouldRelaunch(
      currentMaxOldSpaceSizeMb,
      targetMaxOldSpaceSizeInMB,
    )
      ? [`--max-old-space-size=${targetMaxOldSpaceSizeInMB}`]
      : [];
  }

  private static shouldRelaunch(current: number, target: number): boolean {
    // Add a buffer to prevent unnecessary relaunches (10% threshold)
    const threshold = current * 1.1;
    return target > threshold;
  }
}

// Optimized stdin reader with streaming support for large inputs
class OptimizedStdinReader {
  private static cache: string | null = null;
  private static cacheTimestamp = 0;
  private static readonly CACHE_TTL_MS = 5000; // 5 seconds

  static async readStdinOptimized(): Promise<string> {
    const now = Date.now();

    // Return cached value if still valid
    if (
      OptimizedStdinReader.cache !== null &&
      now - OptimizedStdinReader.cacheTimestamp <
        OptimizedStdinReader.CACHE_TTL_MS
    ) {
      return OptimizedStdinReader.cache;
    }

    try {
      const chunks: string[] = [];
      let totalSize = 0;
      const maxSize = 50 * 1024 * 1024; // 50MB limit

      return new Promise((resolve, reject) => {
        process.stdin.setEncoding('utf8');

        const timeout = setTimeout(() => {
          reject(new Error('Stdin read timeout after 10 seconds'));
        }, 10000);

        process.stdin.on('data', (chunk: string) => {
          totalSize += chunk.length;

          if (totalSize > maxSize) {
            clearTimeout(timeout);
            reject(
              new Error(
                `Input too large: ${totalSize} bytes (max: ${maxSize})`,
              ),
            );
            return;
          }

          chunks.push(chunk);
        });

        process.stdin.on('end', () => {
          clearTimeout(timeout);
          const result = chunks.join('');

          // Cache the result
          OptimizedStdinReader.cache = result;
          OptimizedStdinReader.cacheTimestamp = now;

          resolve(result);
        });

        process.stdin.on('error', (err) => {
          clearTimeout(timeout);
          reject(err);
        });
      });
    } catch (error) {
      OptimizedLogger.getInstance().error(`Failed to read stdin: ${error}`);
      return '';
    }
  }
}

// Dynamic import manager with better error handling and caching
class DynamicImportManager {
  private static importCache = new Map<string, Promise<unknown>>();

  static async loadGeminiAgent(config: Config): Promise<unknown> {
    const cacheKey = 'gemini-agent';

    if (DynamicImportManager.importCache.has(cacheKey)) {
      return DynamicImportManager.importCache.get(cacheKey);
    }

    const importPromise = (async () => {
      try {
        const startTime = performance.now();

        // Dynamic import with timeout protection
        const importPromise = import(
          '@google/gemini-cli-mew-upgrade/agent/gemini-agent.js'
        );
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Import timeout')), 5000);
        });

        const moduleResult = (await Promise.race([
          importPromise,
          timeoutPromise,
        ])) as {
          GeminiAgent: new (config: unknown) => unknown;
        };

        const loadTime = performance.now() - startTime;
        if (config.getDebugMode()) {
          OptimizedLogger.getInstance().debug(
            `GeminiAgent loaded in ${loadTime.toFixed(2)}ms`,
          );
        }

        return new moduleResult.GeminiAgent(config as unknown);
      } catch (err) {
        if (config.getDebugMode()) {
          OptimizedLogger.getInstance().warn(
            `Failed to load GeminiAgent: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        return undefined;
      }
    })();

    DynamicImportManager.importCache.set(cacheKey, importPromise);
    return importPromise;
  }
}

// Parallel startup orchestrator for better performance
class StartupOrchestrator {
  static async executeParallelStartup(workspaceRoot: string) {
    const startTime = performance.now();
    const logger = OptimizedLogger.getInstance();

    // Phase 1: Independent operations that can run in parallel
    const [settings, argv, extensions, checkpointCleanup, version] =
      await Promise.allSettled([
        loadSettings(workspaceRoot),
        parseArguments(),
        loadExtensions(workspaceRoot),
        cleanupCheckpoints(),
        getCliVersion(),
      ]);
    console.log('[DEBUG] StartupOrchestrator: Phase 1 completed.');

    // Handle errors from parallel operations
    const settingsValue =
      settings.status === 'fulfilled'
        ? settings.value
        : (() => {
            logger.error('Failed to load settings');
            process.exit(1);
          })();

    const argvValue =
      argv.status === 'fulfilled'
        ? argv.value
        : (() => {
            logger.error('Failed to parse arguments');
            process.exit(1);
          })();

    const extensionsValue =
      extensions.status === 'fulfilled' ? extensions.value : [];
    const versionValue =
      version.status === 'fulfilled' ? version.value : 'unknown';

    if (checkpointCleanup.status === 'rejected') {
      logger.warn(`Checkpoint cleanup failed: ${checkpointCleanup.reason}`);
    }

    // Phase 2: Operations that depend on Phase 1 results
    const config = await loadCliConfig(
      settingsValue.merged,
      extensionsValue,
      sessionId,
      argvValue,
    );

    logger.setDebugMode(config.getDebugMode());

    // Phase 3: Independent setup operations
    const setupPromises = [
      // DNS configuration
      Promise.resolve().then(() => {
        dns.setDefaultResultOrder(
          validateDnsResolutionOrder(settingsValue.merged.dnsResolutionOrder),
        );
      }),

      // Console patcher setup
      Promise.resolve().then(() => {
        const consolePatcher = new ConsolePatcher({
          stderr: true,
          debugMode: config.getDebugMode(),
        });
        consolePatcher.patch();
        registerCleanup(consolePatcher.cleanup);
        return consolePatcher;
      }),

      // Max sized box debugging setup
      Promise.resolve().then(() => {
        setMaxSizedBoxDebugging(config.getDebugMode());
      }),

      // Theme loading
      Promise.resolve().then(() => {
        themeManager.loadCustomThemes(settingsValue.merged.customThemes);
        if (settingsValue.merged.theme) {
          if (!themeManager.setActiveTheme(settingsValue.merged.theme)) {
            logger.warn(`Theme "${settingsValue.merged.theme}" not found`);
          }
        }
      }),
    ];

    await Promise.allSettled(setupPromises);

    const totalTime = performance.now() - startTime;
    logger.debug(`Parallel startup completed in ${totalTime.toFixed(2)}ms`);

    return {
      settings: settingsValue,
      argv: argvValue,
      extensions: extensionsValue,
      config,
      version: versionValue,
    };
  }
}

export function validateDnsResolutionOrder(
  order: string | undefined,
): DnsResolutionOrder {
  const defaultValue: DnsResolutionOrder = 'ipv4first';
  if (order === undefined) {
    return defaultValue;
  }
  if (order === 'ipv4first' || order === 'verbatim') {
    return order;
  }
  // Use optimized logger instead of direct console.warn
  OptimizedLogger.getInstance().warn(
    `Invalid value for dnsResolutionOrder in settings: "${order}". Using default "${defaultValue}".`,
  );
  return defaultValue;
}

// Optimized memory args function using the MemoryManager
function getNodeMemoryArgs(config: Config): string[] {
  return MemoryManager.getOptimizedMemoryArgs(config);
}

// Optimized relaunch function with better error handling
async function relaunchWithAdditionalArgs(
  additionalArgs: string[],
): Promise<void> {
  const logger = OptimizedLogger.getInstance();

  try {
    const nodeArgs = [...additionalArgs, ...process.argv.slice(1)];
    const newEnv = { ...process.env, GEMINI_CLI_NO_RELAUNCH: 'true' };

    logger.debug(`Relaunching with args: ${nodeArgs.join(' ')}`);

    const child = spawn(process.execPath, nodeArgs, {
      stdio: 'inherit',
      env: newEnv,
    });

    await new Promise<void>((resolve, reject) => {
      child.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Child process exited with code ${code}`));
        }
      });

      child.on('error', reject);
    });

    process.exit(0);
  } catch (error) {
    logger.error(`Failed to relaunch: ${error}`);
    process.exit(1);
  }
}
import { runZedIntegration } from './zed-integration/zedIntegration.js';

export function setupUnhandledRejectionHandler(): void {
  let unhandledRejectionOccurred = false;
  const logger = OptimizedLogger.getInstance();

  process.on('unhandledRejection', (reason, _promise) => {
    const errorMessage = `=========================================
This is an unexpected error. Please file a bug report using the /bug tool.
CRITICAL: Unhandled Promise Rejection!
=========================================
Reason: ${reason}${
      reason instanceof Error && reason.stack
        ? `
Stack trace:
${reason.stack}`
        : ''
    }`;

    logger.error(errorMessage);
    appEvents.emit(AppEvent.LogError, errorMessage);

    if (!unhandledRejectionOccurred) {
      unhandledRejectionOccurred = true;
      appEvents.emit(AppEvent.OpenDebugConsole);
    }
  });
}

// Optimized interactive UI startup with parallel operations
export async function startInteractiveUI(
  config: Config,
  settings: LoadedSettings,
  startupWarnings: string[],
  workspaceRoot: string,
): Promise<void> {
  const logger = OptimizedLogger.getInstance();
  const startTime = performance.now();

  try {
    // Parallel initialization of UI components
    const [version, agentInstance, kittyResult] = await Promise.allSettled([
      getCliVersion(),
      DynamicImportManager.loadGeminiAgent(config),
      detectAndEnableKittyProtocol(),
    ]);

    // Handle results from parallel operations
    const versionValue =
      version.status === 'fulfilled' ? version.value : 'unknown';
    const agentValue =
      agentInstance.status === 'fulfilled' ? agentInstance.value : undefined;

    if (kittyResult.status === 'rejected') {
      logger.debug(`Kitty protocol detection failed: ${kittyResult.reason}`);
    }

    // Set window title (optimized to avoid blocking)
    process.nextTick(() => setWindowTitle(basename(workspaceRoot), settings));

    const initTime = performance.now() - startTime;
    logger.debug(`Interactive UI initialization took ${initTime.toFixed(2)}ms`);

    // Render the React application
    const instance = render(
      <React.StrictMode>
        <SettingsContext.Provider value={settings}>
          <AppWrapper
            config={config}
            settings={settings}
            startupWarnings={startupWarnings}
            version={versionValue}
            agent={agentValue}
          />
        </SettingsContext.Provider>
      </React.StrictMode>,
      {
        exitOnCtrlC: false,
        isScreenReaderEnabled: config.getScreenReader(),
      },
    );

    // Async update check (non-blocking)
    checkForUpdates()
      .then((info) => handleAutoUpdate(info, settings, config.getProjectRoot()))
      .catch((err) => {
        if (config.getDebugMode()) {
          logger.debug(`Update check failed: ${err}`);
        }
      });

    registerCleanup(() => instance.unmount());

    const totalTime = performance.now() - startTime;
    logger.debug(`Interactive UI fully started in ${totalTime.toFixed(2)}ms`);
  } catch (error) {
    logger.error(`Failed to start interactive UI: ${error}`);
    throw error;
  }
}

// Optimized main function with comprehensive parallel processing
export async function main(): Promise<void> {
  const mainStartTime = performance.now();
  setupUnhandledRejectionHandler();

  const workspaceRoot = process.cwd();
  const logger = OptimizedLogger.getInstance();

  try {
    // Execute optimized parallel startup
    const {
      settings,
      argv,
      extensions,
      config,
      version: _version,
    } = await StartupOrchestrator.executeParallelStartup(workspaceRoot);

    // Early validation of settings errors
    if (settings.errors.length > 0) {
      for (const error of settings.errors) {
        const colorizedMessage = process.env['NO_COLOR']
          ? `Error in ${error.path}: ${error.message}`
          : `\x1b[31mError in ${error.path}: ${error.message}\x1b[0m`;

        logger.error(colorizedMessage);
        logger.error(`Please fix ${error.path} and try again.`);
      }
      process.exit(1);
    }

    // Early exit for list extensions
    if (config.getListExtensions()) {
      logger.info('Installed extensions:');
      for (const extension of extensions) {
        logger.info(`- ${extension.config.name}`);
      }
      process.exit(0);
    }

    // Parallel validation and checks
    const [promptInteractiveCheck, _authTypeSetup, configInitialization] =
      await Promise.allSettled([
        // Validate prompt interactive flag
        Promise.resolve().then(() => {
          if (argv.promptInteractive && !process.stdin.isTTY) {
            throw new Error(
              'The --prompt-interactive flag is not supported when piping input from stdin.',
            );
          }
        }),

        // Set default auth type if needed
        Promise.resolve().then(() => {
          if (
            !settings.merged.selectedAuthType &&
            process.env['CLOUD_SHELL'] === 'true'
          ) {
            settings.setValue(
              SettingScope.User,
              'selectedAuthType',
              AuthType.CLOUD_SHELL,
            );
          }
        }),

        // Initialize config
        config.initialize(),
      ]);

    // Handle validation errors
    if (promptInteractiveCheck.status === 'rejected') {
      logger.error(promptInteractiveCheck.reason.message);
      process.exit(1);
    }

    if (configInitialization.status === 'rejected') {
      logger.error(
        `Failed to initialize config: ${configInitialization.reason}`,
      );
      process.exit(1);
    }

    // Parallel IDE and sandbox setup
    const setupPromises: Array<Promise<void>> = [];

    // IDE mode setup
    if (config.getIdeMode()) {
      setupPromises.push(
        (async () => {
          await config.getIdeClient().connect();
          logIdeConnection(
            config,
            new IdeConnectionEvent(IdeConnectionType.START),
          );
        })(),
      );
    }

    // Experimental Zed integration
    if (config.getExperimentalZedIntegration()) {
      await Promise.allSettled(setupPromises);
      return runZedIntegration(config, settings, extensions, argv);
    }

    // Memory and sandbox handling
    if (!process.env['SANDBOX']) {
      const memoryArgs = settings.merged.autoConfigureMaxOldSpaceSize
        ? getNodeMemoryArgs(config)
        : [];

      const sandboxConfig = config.getSandbox();

      if (sandboxConfig) {
        // Parallel authentication and stdin reading for sandbox
        const sandboxPrep = await Promise.allSettled([
          // Authentication
          (async () => {
            if (
              settings.merged.selectedAuthType &&
              !settings.merged.useExternalAuth
            ) {
              const authError = validateAuthMethod(
                settings.merged.selectedAuthType,
              );
              if (authError) {
                throw new Error(authError);
              }
              await config.refreshAuth(settings.merged.selectedAuthType);
            }
          })(),

          // Stdin reading
          process.stdin.isTTY
            ? Promise.resolve('')
            : OptimizedStdinReader.readStdinOptimized(),
        ]);

        // Handle authentication errors
        if (sandboxPrep[0].status === 'rejected') {
          logger.error(`Authentication error: ${sandboxPrep[0].reason}`);
          process.exit(1);
        }

        const stdinData =
          sandboxPrep[1].status === 'fulfilled' ? sandboxPrep[1].value : '';

        // Optimized stdin injection
        const injectStdinIntoArgs = (
          args: string[],
          stdin?: string,
        ): string[] => {
          if (!stdin) return [...args];

          const finalArgs = [...args];
          const promptIndex = finalArgs.findIndex(
            (arg) => arg === '--prompt' || arg === '-p',
          );

          if (promptIndex > -1 && finalArgs.length > promptIndex + 1) {
            finalArgs[promptIndex + 1] =
              `${stdin}\n\n${finalArgs[promptIndex + 1]}`;
          } else {
            finalArgs.push('--prompt', stdin);
          }

          return finalArgs;
        };

        const sandboxArgs = injectStdinIntoArgs(process.argv, stdinData);
        await start_sandbox(sandboxConfig, memoryArgs, config, sandboxArgs);
        process.exit(0);
      } else if (memoryArgs.length > 0) {
        await relaunchWithAdditionalArgs(memoryArgs);
        process.exit(0);
      }
    }

    // Complete remaining setup operations
    await Promise.allSettled(setupPromises);

    // OAuth handling for login with Google
    if (
      settings.merged.selectedAuthType === AuthType.LOGIN_WITH_GOOGLE &&
      config.isBrowserLaunchSuppressed()
    ) {
      await getOauthClient(settings.merged.selectedAuthType, config);
    }

    // Parallel startup warnings and input processing
    const [startupWarnings, inputProcessing] = await Promise.allSettled([
      Promise.all([
        getStartupWarnings(),
        getUserStartupWarnings(workspaceRoot),
      ]).then(([startup, user]) => [...startup, ...user]),

      (async () => {
        let input = config.getQuestion();

        if (!config.isInteractive() && !process.stdin.isTTY) {
          const stdinData = await OptimizedStdinReader.readStdinOptimized();
          if (stdinData) {
            input = `${stdinData}\n\n${input}`;
          }
        }

        return input;
      })(),
    ]);

    const startupWarningsValue =
      startupWarnings.status === 'fulfilled' ? startupWarnings.value : [];
    const inputValue =
      inputProcessing.status === 'fulfilled' ? inputProcessing.value : '';

    // Launch interactive UI
    if (config.isInteractive()) {
      const setupTime = performance.now() - mainStartTime;
      logger.debug(`Total startup time: ${setupTime.toFixed(2)}ms`);

      await startInteractiveUI(
        config,
        settings,
        startupWarningsValue,
        workspaceRoot,
      );
      return;
    }

    // Non-interactive mode
    if (!inputValue) {
      logger.error(
        'No input provided via stdin. Input can be provided by piping data into gemini or using the --prompt option.',
      );
      process.exit(1);
    }

    // Parallel non-interactive setup
    const promptId = Math.random().toString(16).slice(2);

    const [_loggingResult, nonInteractiveConfig, agentInstance] =
      await Promise.allSettled([
        // User prompt logging
        Promise.resolve().then(() => {
          logUserPrompt(config, {
            'event.name': 'user_prompt',
            'event.timestamp': new Date().toISOString(),
            prompt: inputValue,
            prompt_id: promptId,
            auth_type: config.getContentGeneratorConfig()?.authType,
            prompt_length: inputValue.length,
          });
        }),

        // Validate non-interactive auth
        validateNonInteractiveAuth(
          settings.merged.selectedAuthType,
          settings.merged.useExternalAuth,
          config,
        ),

        // Load agent
        DynamicImportManager.loadGeminiAgent(config),
      ]);

    // Handle errors
    if (nonInteractiveConfig.status === 'rejected') {
      logger.error(
        `Authentication validation failed: ${nonInteractiveConfig.reason}`,
      );
      process.exit(1);
    }

    const agent =
      agentInstance.status === 'fulfilled'
        ? agentInstance.value
        : (() => {
            logger.error('Failed to load GeminiAgent for non-interactive mode');
            process.exit(1);
          })();

    const totalSetupTime = performance.now() - mainStartTime;
    logger.debug(
      `Non-interactive setup completed in ${totalSetupTime.toFixed(2)}ms`,
    );

    await runNonInteractive(
      agent,
      nonInteractiveConfig.value,
      inputValue,
      promptId,
    );
    process.exit(0);
  } catch (error) {
    logger.error(`Fatal error in main: ${error}`);
    process.exit(1);
  }
}

// Optimized window title function with better sanitization
function setWindowTitle(title: string, settings: LoadedSettings): void {
  if (settings.merged.hideWindowTitle) return;

  try {
    const rawTitle = process.env['CLI_TITLE'] || `Gemini - ${title}`;
    // More efficient control character removal using character codes
    const sanitizedTitle = rawTitle
      .split('')
      .filter((char) => {
        const code = char.charCodeAt(0);
        return code >= 32 && code !== 127; // Remove control characters
      })
      .join('');

    process.stdout.write(`\x1b]2;${sanitizedTitle}\x07`);

    // Clean up on exit
    if (!process.listeners('exit').length) {
      process.on('exit', () => {
        process.stdout.write('\x1b]2;\x07');
      });
    }
  } catch (error) {
    // Silent failure for window title setting
    OptimizedLogger.getInstance().debug(`Failed to set window title: ${error}`);
  }
}
