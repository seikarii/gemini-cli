/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import { homedir } from 'node:os';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import process from 'node:process';
import { mcpCommand } from '../commands/mcp.js';
import {
  Config,
  loadServerHierarchicalMemory,
  setGeminiMdFilename as setServerGeminiMdFilename,
  getCurrentGeminiMdFilename,
  ApprovalMode,
  DEFAULT_GEMINI_MODEL,
  DEFAULT_GEMINI_EMBEDDING_MODEL,
  DEFAULT_MEMORY_FILE_FILTERING_OPTIONS,
  FileDiscoveryService,
  TelemetryTarget,
  FileFilteringOptions,
  ShellTool,
  EditTool,
  WriteFileTool,
  MCPServerConfig,
} from '@google/gemini-cli-core';
import { Settings } from './settings.js';

import { Extension, annotateActiveExtensions } from './extension.js';
import { getCliVersion } from '../utils/version.js';
import { loadSandboxConfig } from './sandboxConfig.js';
import { resolvePath } from '../utils/resolvePath.js';

import { isWorkspaceTrusted } from './trustedFolders.js';

import { logger } from './logger.js';
import { handleConfigError, createValidationErrorMessage } from './errorHandling.js';

// Centralized environment variable access
interface EnvironmentVariables {
  GEMINI_MODEL?: string;
  DEBUG?: string;
  DEBUG_MODE?: string;
  OTEL_EXPORTER_OTLP_ENDPOINT?: string;
  HTTPS_PROXY?: string;
  https_proxy?: string;
  HTTP_PROXY?: string;
  http_proxy?: string;
  NO_BROWSER?: string;
}

function getEnvironmentVariables(): EnvironmentVariables {
  return {
    GEMINI_MODEL: process.env['GEMINI_MODEL'],
    DEBUG: process.env['DEBUG'],
    DEBUG_MODE: process.env['DEBUG_MODE'],
    OTEL_EXPORTER_OTLP_ENDPOINT: process.env['OTEL_EXPORTER_OTLP_ENDPOINT'],
    HTTPS_PROXY: process.env['HTTPS_PROXY'],
    https_proxy: process.env['https_proxy'],
    HTTP_PROXY: process.env['HTTP_PROXY'],
    http_proxy: process.env['http_proxy'],
    NO_BROWSER: process.env['NO_BROWSER'],
  };
}

export interface CliArgs {
  model: string | undefined;
  sandbox: boolean | string | undefined;
  sandboxImage: string | undefined;
  debug: boolean | undefined;
  prompt: string | undefined;
  promptInteractive: string | undefined;
  allFiles: boolean | undefined;
  showMemoryUsage: boolean | undefined;
  yolo: boolean | undefined;
  approvalMode: string | undefined;
  telemetry: boolean | undefined;
  checkpointing: boolean | undefined;
  telemetryTarget: string | undefined;
  telemetryOtlpEndpoint: string | undefined;
  telemetryOtlpProtocol: string | undefined;
  telemetryLogPrompts: boolean | undefined;
  telemetryOutfile: string | undefined;
  allowedMcpServerNames: string[] | undefined;
  experimentalAcp: boolean | undefined;
  extensions: string[] | undefined;
  listExtensions: boolean | undefined;
  proxy: string | undefined;
  includeDirectories: string[] | undefined;
  screenReader: boolean | undefined;
}

export async function parseArguments(): Promise<CliArgs> {
  const yargsInstance = yargs(hideBin(process.argv))
    .locale('en')
    .scriptName('gemini')
    .usage(
      'Usage: gemini [options] [command]\n\nGemini CLI - Launch an interactive CLI, use -p/--prompt for non-interactive mode',
    );

  // Configure all option groups
  configureModelOptions(yargsInstance);
  configurePromptOptions(yargsInstance);
  configureSandboxOptions(yargsInstance);
  configureDebugOptions(yargsInstance);
  configureFileOptions(yargsInstance);
  configureMemoryOptions(yargsInstance);
  configureApprovalOptions(yargsInstance);
  configureTelemetryOptions(yargsInstance);
  configureExtensionOptions(yargsInstance);
  configureNetworkOptions(yargsInstance);
  configureAccessibilityOptions(yargsInstance);

  // Configure validation and commands
  configureValidation(yargsInstance);
  await configureCommands(yargsInstance);

  yargsInstance.wrap(yargsInstance.terminalWidth());
  const result = await yargsInstance.parse();

  // Handle case where MCP subcommands are executed - they should exit the process
  // and not return to main CLI logic
  if (result._.length > 0 && result._[0] === 'mcp') {
    // MCP commands handle their own execution and process exit
    process.exit(0);
  }

  // The import format is now only controlled by settings.memoryImportFormat
  // We no longer accept it as a CLI argument
  return result as unknown as CliArgs;
}

// Helper functions to configure different option groups
function configureModelOptions(yargsInstance: ReturnType<typeof yargs>) {
  const env = getEnvironmentVariables();
  return yargsInstance.option('model', {
    alias: 'm',
    type: 'string',
    description: 'Model',
    default: env.GEMINI_MODEL,
  });
}

function configurePromptOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('prompt', {
      alias: 'p',
      type: 'string',
      description: 'Prompt. Appended to input on stdin (if any).',
    })
    .option('prompt-interactive', {
      alias: 'i',
      type: 'string',
      description: 'Execute the provided prompt and continue in interactive mode',
    });
}

function configureSandboxOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('sandbox', {
      alias: 's',
      type: 'boolean',
      description: 'Run in sandbox?',
    })
    .option('sandbox-image', {
      type: 'string',
      description: 'Sandbox image URI.',
    });
}

function configureDebugOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance.option('debug', {
    alias: 'd',
    type: 'boolean',
    description: 'Run in debug mode?',
    default: false,
  });
}

function configureFileOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('all-files', {
      alias: ['a'],
      type: 'boolean',
      description: 'Include ALL files in context?',
      default: false,
    })
    .option('include-directories', {
      type: 'array',
      string: true,
      description:
        'Additional directories to include in the workspace (comma-separated or multiple --include-directories)',
      coerce: (dirs: string[]) =>
        // Handle comma-separated values
        dirs.flatMap((dir) => dir.split(',').map((d) => d.trim())),
    });
}

function configureMemoryOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('show-memory-usage', {
      type: 'boolean',
      description: 'Show memory usage in status bar',
      default: false,
    })
    .option('checkpointing', {
      alias: 'c',
      type: 'boolean',
      description: 'Enables checkpointing of file edits',
      default: false,
    });
}

function configureApprovalOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('yolo', {
      alias: 'y',
      type: 'boolean',
      description:
        'Automatically accept all actions (aka YOLO mode, see https://www.youtube.com/watch?v=xvFZjo5PgG0 for more details)?',
      default: false,
    })
    .option('approval-mode', {
      type: 'string',
      choices: ['default', 'auto_edit', 'yolo'],
      description:
        'Set the approval mode: default (prompt for approval), auto_edit (auto-approve edit tools), yolo (auto-approve all tools)',
    });
}

function configureTelemetryOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('telemetry', {
      type: 'boolean',
      description:
        'Enable telemetry? This flag specifically controls if telemetry is sent. Other --telemetry-* flags set specific values but do not enable telemetry on their own.',
    })
    .option('telemetry-target', {
      type: 'string',
      choices: ['local', 'gcp'],
      description:
        'Set the telemetry target (local or gcp). Overrides settings files.',
    })
    .option('telemetry-otlp-endpoint', {
      type: 'string',
      description:
        'Set the OTLP endpoint for telemetry. Overrides environment variables and settings files.',
    })
    .option('telemetry-otlp-protocol', {
      type: 'string',
      choices: ['grpc', 'http'],
      description:
        'Set the OTLP protocol for telemetry (grpc or http). Overrides settings files.',
    })
    .option('telemetry-log-prompts', {
      type: 'boolean',
      description:
        'Enable or disable logging of user prompts for telemetry. Overrides settings files.',
    })
    .option('telemetry-outfile', {
      type: 'string',
      description: 'Redirect all telemetry output to the specified file.',
    });
}

function configureExtensionOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .option('experimental-acp', {
      type: 'boolean',
      description: 'Starts the agent in ACP mode',
    })
    .option('allowed-mcp-server-names', {
      type: 'array',
      string: true,
      description: 'Allowed MCP server names',
    })
    .option('extensions', {
      alias: 'e',
      type: 'array',
      string: true,
      description:
        'A list of extensions to use. If not provided, all extensions are used.',
    })
    .option('list-extensions', {
      alias: 'l',
      type: 'boolean',
      description: 'List all available extensions and exit.',
    });
}

function configureNetworkOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance.option('proxy', {
    type: 'string',
    description:
      'Proxy for gemini client, like schema://user:password@host:port',
  });
}

function configureAccessibilityOptions(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance.option('screen-reader', {
    type: 'boolean',
    description: 'Enable screen reader mode for accessibility.',
    default: false,
  });
}

function configureValidation(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance.check((argv: { [argName: string]: unknown; _: Array<string | number>; $0: string; }) => {
    if (argv.prompt && argv['promptInteractive']) {
      handleConfigError(
        'Cannot use both --prompt (-p) and --prompt-interactive (-i) together',
        'fatal'
      );
      return false;
    }
    if (argv.yolo && argv['approvalMode']) {
      handleConfigError(
        'Cannot use both --yolo (-y) and --approval-mode together. Use --approval-mode=yolo instead.',
        'fatal'
      );
      return false;
    }
    return true;
  });
}

async function configureCommands(yargsInstance: ReturnType<typeof yargs>) {
  return yargsInstance
    .command('$0', 'Launch Gemini CLI')
    .command(mcpCommand)
    .version(await getCliVersion())
    .alias('v', 'version')
    .help()
    .alias('h', 'help')
    .strict()
    .demandCommand(0, 0);
}

// This function is now a thin wrapper around the server's implementation.
// It's kept in the CLI for now as App.tsx directly calls it for memory refresh.
// TODO: Consider if App.tsx should get memory via a server call or if Config should refresh itself.
export async function loadHierarchicalGeminiMemory(
  currentWorkingDirectory: string,
  includeDirectoriesToReadGemini: readonly string[] = [],
  debugMode: boolean,
  fileService: FileDiscoveryService,
  settings: Settings,
  extensionContextFilePaths: string[] = [],
  memoryImportFormat: 'flat' | 'tree' = 'tree',
  fileFilteringOptions?: FileFilteringOptions,
): Promise<{ memoryContent: string; fileCount: number }> {
  // FIX: Use real, canonical paths for a reliable comparison to handle symlinks.
  const realCwd = fs.realpathSync(path.resolve(currentWorkingDirectory));
  const realHome = fs.realpathSync(path.resolve(homedir()));
  const isHomeDirectory = realCwd === realHome;

  // If it is the home directory, pass an empty string to the core memory
  // function to signal that it should skip the workspace search.
  const effectiveCwd = isHomeDirectory ? '' : currentWorkingDirectory;

  if (debugMode) {
    logger.debug(
      `CLI: Delegating hierarchical memory load to server for CWD: ${currentWorkingDirectory} (memoryImportFormat: ${memoryImportFormat})`,
    );
  }

  // Directly call the server function with the corrected path.
  return loadServerHierarchicalMemory(
    effectiveCwd,
    includeDirectoriesToReadGemini,
    debugMode,
    fileService,
    extensionContextFilePaths,
    memoryImportFormat,
    fileFilteringOptions,
    settings.memoryDiscoveryMaxDirs,
  );
}

export async function loadCliConfig(
  settings: Settings,
  extensions: Extension[],
  sessionId: string,
  argv: CliArgs,
  cwd: string = process.cwd(),
): Promise<Config> {
  // Resolve configuration values
  const debugMode = resolveDebugMode(argv);
  const memoryImportFormat = settings.memoryImportFormat || 'tree';
  const ideMode = settings.ideMode ?? false;
  const folderTrust = resolveFolderTrust(settings);
  const trustedFolder = await isWorkspaceTrusted(settings);

  // Process extensions
  const allExtensions = annotateActiveExtensions(
    extensions,
    argv.extensions || [],
  );
  const activeExtensions = extensions.filter(
    (_, i) => allExtensions[i].isActive,
  );

  // Configure context filename
  configureContextFilename(settings);

  // Load memory and file information
  const { memoryContent, fileCount, extensionContextFilePaths, fileService, includeDirectories } =
    await loadMemoryAndFiles(settings, activeExtensions, argv, cwd, debugMode, memoryImportFormat);

  // Configure MCP servers
  const { mcpServers, blockedMcpServers } = configureMcpServers(settings, activeExtensions, argv);

  // Resolve approval and interaction settings
  const question = argv.promptInteractive || argv.prompt || '';
  const approvalMode = resolveApprovalMode(argv);
  const interactive = resolveInteractiveMode(argv, question);
  const extraExcludes = resolveExtraExcludes(interactive, argv, approvalMode);

  // Configure tools and sandbox
  const excludeTools = mergeExcludeTools(
    settings,
    activeExtensions,
    extraExcludes.length > 0 ? extraExcludes : undefined,
  );
  const sandboxConfig = await loadSandboxConfig(settings, argv);

  // Build final configuration
  return buildConfigObject({
    settings,
    argv,
    cwd,
    sessionId,
    debugMode,
    ideMode,
    folderTrust,
    trustedFolder,
    allExtensions,
    memoryContent,
    fileCount,
    extensionContextFilePaths,
    fileService,
    includeDirectories,
    mcpServers,
    blockedMcpServers,
    question,
    approvalMode,
    interactive,
    excludeTools,
    sandboxConfig,
  });
}

// Helper functions for configuration resolution
function resolveDebugMode(argv: CliArgs): boolean {
  const env = getEnvironmentVariables();
  return argv.debug ||
    [env.DEBUG, env.DEBUG_MODE].some(
      (v) => v === 'true' || v === '1',
    ) ||
    false;
}

function resolveFolderTrust(settings: Settings): boolean {
  const folderTrustFeature = settings.folderTrustFeature ?? false;
  const folderTrustSetting = settings.folderTrust ?? true;
  return folderTrustFeature && folderTrustSetting;
}

function configureContextFilename(settings: Settings): void {
  if (settings.contextFileName) {
    setServerGeminiMdFilename(settings.contextFileName);
  } else {
    setServerGeminiMdFilename(getCurrentGeminiMdFilename());
  }
}

async function loadMemoryAndFiles(
  settings: Settings,
  activeExtensions: Extension[],
  argv: CliArgs,
  cwd: string,
  debugMode: boolean,
  memoryImportFormat: 'flat' | 'tree',
) {
  const extensionContextFilePaths = activeExtensions.flatMap(
    (e) => e.contextFiles,
  );

  const fileService = new FileDiscoveryService(cwd);

  const fileFiltering = {
    ...DEFAULT_MEMORY_FILE_FILTERING_OPTIONS,
    ...settings.fileFiltering,
  };

  const includeDirectories = (settings.includeDirectories || [])
    .map(resolvePath)
    .concat((argv.includeDirectories || []).map(resolvePath));

  const { memoryContent, fileCount } = await loadHierarchicalGeminiMemory(
    cwd,
    settings.loadMemoryFromIncludeDirectories ? includeDirectories : [],
    debugMode,
    fileService,
    settings,
    extensionContextFilePaths,
    memoryImportFormat,
    fileFiltering,
  );

  return {
    memoryContent,
    fileCount,
    extensionContextFilePaths,
    fileService,
    includeDirectories,
  };
}

function configureMcpServers(
  settings: Settings,
  activeExtensions: Extension[],
  argv: CliArgs,
) {
  let mcpServers = mergeMcpServers(settings, activeExtensions);
  const blockedMcpServers: Array<{ name: string; extensionName: string }> = [];

  if (!argv.allowedMcpServerNames) {
    if (settings.allowMCPServers) {
      mcpServers = allowedMcpServers(
        mcpServers,
        settings.allowMCPServers,
        blockedMcpServers,
      );
    }

    if (settings.excludeMCPServers) {
      const excludedNames = new Set(settings.excludeMCPServers.filter(Boolean));
      if (excludedNames.size > 0) {
        mcpServers = Object.fromEntries(
          Object.entries(mcpServers).filter(([key]) => !excludedNames.has(key)),
        );
      }
    }
  }

  if (argv.allowedMcpServerNames) {
    mcpServers = allowedMcpServers(
      mcpServers,
      argv.allowedMcpServerNames,
      blockedMcpServers,
    );
  }

  return { mcpServers, blockedMcpServers };
}

function resolveApprovalMode(argv: CliArgs): ApprovalMode {
  if (argv.approvalMode) {
    switch (argv.approvalMode) {
      case 'yolo':
        return ApprovalMode.YOLO;
      case 'auto_edit':
        return ApprovalMode.AUTO_EDIT;
      case 'default':
        return ApprovalMode.DEFAULT;
      default:
        handleConfigError(
          createValidationErrorMessage('approvalMode', `Invalid value '${argv.approvalMode}'. Valid values are: yolo, auto_edit, default`),
          'fatal'
        );
        return ApprovalMode.DEFAULT; // This won't be reached, but needed for TypeScript
    }
  } else {
    return argv.yolo || false ? ApprovalMode.YOLO : ApprovalMode.DEFAULT;
  }
}

function resolveInteractiveMode(argv: CliArgs, question: string): boolean {
  return !!argv.promptInteractive || (process.stdin.isTTY && question.length === 0);
}

function resolveExtraExcludes(
  interactive: boolean,
  argv: CliArgs,
  approvalMode: ApprovalMode,
): string[] {
  const extraExcludes: string[] = [];
  if (!interactive && !argv.experimentalAcp) {
    switch (approvalMode) {
      case ApprovalMode.DEFAULT:
        extraExcludes.push(ShellTool.Name, EditTool.Name, WriteFileTool.Name);
        break;
      case ApprovalMode.AUTO_EDIT:
        extraExcludes.push(ShellTool.Name);
        break;
      case ApprovalMode.YOLO:
        // No extra excludes for YOLO mode.
        break;
      default:
        // This should never happen due to validation earlier, but satisfies the linter
        break;
    }
  }
  return extraExcludes;
}

function buildConfigObject(params: {
  settings: Settings;
  argv: CliArgs;
  cwd: string;
  sessionId: string;
  debugMode: boolean;
  ideMode: boolean;
  folderTrust: boolean;
  trustedFolder: boolean;
  allExtensions: ReturnType<typeof annotateActiveExtensions>;
  memoryContent: string;
  fileCount: number;
  extensionContextFilePaths: string[];
  fileService: FileDiscoveryService;
  includeDirectories: string[];
  mcpServers: { [x: string]: MCPServerConfig };
  blockedMcpServers: Array<{ name: string; extensionName: string }>;
  question: string;
  approvalMode: ApprovalMode;
  interactive: boolean;
  excludeTools: string[];
  sandboxConfig: Awaited<ReturnType<typeof loadSandboxConfig>>;
}): Config {
  const {
    settings,
    argv,
    cwd,
    sessionId,
    debugMode,
    ideMode,
    folderTrust,
    trustedFolder,
    allExtensions,
    memoryContent,
    fileCount,
    extensionContextFilePaths,
    fileService,
    includeDirectories,
    mcpServers,
    blockedMcpServers,
    question,
    approvalMode,
    interactive,
    excludeTools,
    sandboxConfig,
  } = params;

  const screenReader =
    argv.screenReader ?? settings.accessibility?.screenReader ?? false;

  return new Config({
    sessionId,
    embeddingModel: DEFAULT_GEMINI_EMBEDDING_MODEL,
    sandbox: sandboxConfig,
    targetDir: cwd,
    includeDirectories,
    loadMemoryFromIncludeDirectories:
      settings.loadMemoryFromIncludeDirectories || false,
    debugMode,
    question,
    fullContext: argv.allFiles || false,
    coreTools: settings.coreTools || undefined,
    excludeTools,
    toolDiscoveryCommand: settings.toolDiscoveryCommand,
    toolCallCommand: settings.toolCallCommand,
    mcpServerCommand: settings.mcpServerCommand,
    mcpServers,
    userMemory: memoryContent,
    geminiMdFileCount: fileCount,
    approvalMode,
    showMemoryUsage:
      argv.showMemoryUsage ||
      settings.showMemoryUsage ||
      false,
    accessibility: {
      ...settings.accessibility,
      screenReader,
    },
    telemetry: {
      enabled: argv.telemetry ?? settings.telemetry?.enabled,
      target: (argv.telemetryTarget ??
        settings.telemetry?.target) as TelemetryTarget,
      otlpEndpoint:
        argv.telemetryOtlpEndpoint ??
        process.env['OTEL_EXPORTER_OTLP_ENDPOINT'] ??
        settings.telemetry?.otlpEndpoint,
      otlpProtocol: (['grpc', 'http'] as const).find(
        (p) =>
          p ===
          (argv.telemetryOtlpProtocol ?? settings.telemetry?.otlpProtocol),
      ),
      logPrompts: argv.telemetryLogPrompts ?? settings.telemetry?.logPrompts,
      outfile: argv.telemetryOutfile ?? settings.telemetry?.outfile,
    },
    usageStatisticsEnabled: settings.usageStatisticsEnabled ?? true,
    fileFiltering: {
      respectGitIgnore: settings.fileFiltering?.respectGitIgnore,
      respectGeminiIgnore: settings.fileFiltering?.respectGeminiIgnore,
      enableRecursiveFileSearch:
        settings.fileFiltering?.enableRecursiveFileSearch,
    },
    checkpointing: argv.checkpointing || settings.checkpointing?.enabled,
    proxy:
      argv.proxy ||
      process.env['HTTPS_PROXY'] ||
      process.env['https_proxy'] ||
      process.env['HTTP_PROXY'] ||
      process.env['http_proxy'],
    cwd,
    fileDiscoveryService: fileService,
    bugCommand: settings.bugCommand,
    model: argv.model || settings.model || DEFAULT_GEMINI_MODEL,
    extensionContextFilePaths,
    maxSessionTurns: settings.maxSessionTurns ?? -1,
    experimentalZedIntegration: argv.experimentalAcp || false,
    listExtensions: argv.listExtensions || false,
    extensions: allExtensions,
    blockedMcpServers,
    noBrowser: !!process.env['NO_BROWSER'],
    summarizeToolOutput: settings.summarizeToolOutput,
    ideMode,
    chatCompression: settings.chatCompression,
    folderTrustFeature: settings.folderTrustFeature ?? false,
    folderTrust,
    interactive,
    trustedFolder,
    useRipgrep: settings.useRipgrep,
    shouldUseNodePtyShell: settings.shouldUseNodePtyShell,
    skipNextSpeakerCheck: settings.skipNextSpeakerCheck,
    enablePromptCompletion: settings.enablePromptCompletion ?? false,
  });
}

function allowedMcpServers(
  mcpServers: { [x: string]: MCPServerConfig },
  allowMCPServers: string[],
  blockedMcpServers: Array<{ name: string; extensionName: string }>,
) {
  const allowedNames = new Set(allowMCPServers.filter(Boolean));
  if (allowedNames.size > 0) {
    mcpServers = Object.fromEntries(
      Object.entries(mcpServers).filter(([key, server]) => {
        const isAllowed = allowedNames.has(key);
        if (!isAllowed) {
          blockedMcpServers.push({
            name: key,
            extensionName: server.extensionName || '',
          });
        }
        return isAllowed;
      }),
    );
  } else {
    blockedMcpServers.push(
      ...Object.entries(mcpServers).map(([key, server]) => ({
        name: key,
        extensionName: server.extensionName || '',
      })),
    );
    mcpServers = {};
  }
  return mcpServers;
}

function mergeMcpServers(settings: Settings, extensions: Extension[]) {
  const mcpServers = { ...(settings.mcpServers || {}) };
  for (const extension of extensions) {
    Object.entries(extension.config.mcpServers || {}).forEach(
      ([key, server]) => {
        if (mcpServers[key]) {
          logger.warn(
            `Skipping extension MCP config for server with key "${key}" as it already exists.`,
          );
          return;
        }
        mcpServers[key] = {
          ...server,
          extensionName: extension.config.name,
        };
      },
    );
  }
  return mcpServers;
}

function mergeExcludeTools(
  settings: Settings,
  extensions: Extension[],
  extraExcludes?: string[] | undefined,
): string[] {
  const allExcludeTools = new Set([
    ...(settings.excludeTools || []),
    ...(extraExcludes || []),
  ]);
  for (const extension of extensions) {
    for (const tool of extension.config.excludeTools || []) {
      allExcludeTools.add(tool);
    }
  }
  return [...allExcludeTools];
}
