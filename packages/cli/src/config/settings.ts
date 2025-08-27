/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
// Removed unused import promisify
import * as path from 'path';
import { homedir, platform } from 'os';
import * as dotenv from 'dotenv';
import {
  GEMINI_CONFIG_DIR as GEMINI_DIR,
  getErrorMessage,
  Storage,
} from '@google/gemini-cli-core';
import stripJsonComments from 'strip-json-comments';
import { DefaultLight } from '../ui/themes/default-light.js';
import { DefaultDark } from '../ui/themes/default.js';
import { isWorkspaceTrusted } from './trustedFolders.js';
import { Settings, MemoryImportFormat } from './settingsSchema.js';
import { logger } from './logger.js';

export type { Settings, MemoryImportFormat };

export const USER_SETTINGS_PATH = Storage.getGlobalSettingsPath();
export const USER_SETTINGS_DIR = path.dirname(USER_SETTINGS_PATH);
export const DEFAULT_EXCLUDED_ENV_VARS = ['DEBUG', 'DEBUG_MODE'];

export function getSystemSettingsPath(): string {
  if (process.env['GEMINI_CLI_SYSTEM_SETTINGS_PATH']) {
    return process.env['GEMINI_CLI_SYSTEM_SETTINGS_PATH'];
  }
  if (platform() === 'darwin') {
    return '/Library/Application Support/GeminiCli/settings.json';
  } else if (platform() === 'win32') {
    return 'C:\\ProgramData\\gemini-cli\\settings.json';
  } else {
    return '/etc/gemini-cli/settings.json';
  }
}

export type { DnsResolutionOrder } from './settingsSchema.js';

export enum SettingScope {
  User = 'User',
  Workspace = 'Workspace',
  System = 'System',
}

export interface CheckpointingSettings {
  enabled?: boolean;
}

export interface SummarizeToolOutputSettings {
  tokenBudget?: number;
}

export interface AccessibilitySettings {
  disableLoadingPhrases?: boolean;
  screenReader?: boolean;
}

export interface SettingsError {
  message: string;
  path: string;
}

export interface SettingsFile {
  settings: Settings;
  path: string;
}

function mergeSettings(
  system: Settings,
  user: Settings,
  workspace: Settings,
  isTrusted: boolean,
): Settings {
  const safeWorkspace = isTrusted ? workspace : ({} as Settings);

  // folderTrust is not supported at workspace level.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { folderTrust, ...safeWorkspaceWithoutFolderTrust } = safeWorkspace;

  return {
    ...user,
    ...safeWorkspaceWithoutFolderTrust,
    ...system,
    customThemes: {
      ...(user.customThemes || {}),
      ...(safeWorkspace.customThemes || {}),
      ...(system.customThemes || {}),
    },
    mcpServers: {
      ...(user.mcpServers || {}),
      ...(safeWorkspace.mcpServers || {}),
      ...(system.mcpServers || {}),
    },
    includeDirectories: [
      ...(system.includeDirectories || []),
      ...(user.includeDirectories || []),
      ...(safeWorkspace.includeDirectories || []),
    ],
    chatCompression: {
      ...(system.chatCompression || {}),
      ...(user.chatCompression || {}),
      ...(safeWorkspace.chatCompression || {}),
    },
  };
}

export class LoadedSettings {
  constructor(
    system: SettingsFile,
    user: SettingsFile,
    workspace: SettingsFile,
    errors: SettingsError[],
    isTrusted: boolean,
  ) {
    this.system = system;
    this.user = user;
    this.workspace = workspace;
    this.errors = errors;
    this.isTrusted = isTrusted;
    this._merged = this.computeMergedSettings();
  }

  readonly system: SettingsFile;
  readonly user: SettingsFile;
  readonly workspace: SettingsFile;
  readonly errors: SettingsError[];
  readonly isTrusted: boolean;

  private _merged: Settings;

  get merged(): Settings {
    return this._merged;
  }

  private computeMergedSettings(): Settings {
    return mergeSettings(
      this.system.settings,
      this.user.settings,
      this.workspace.settings,
      this.isTrusted,
    );
  }

  forScope(scope: SettingScope): SettingsFile {
    switch (scope) {
      case SettingScope.User:
        return this.user;
      case SettingScope.Workspace:
        return this.workspace;
      case SettingScope.System:
        return this.system;
      default:
        throw new Error(`Invalid scope: ${scope}`);
    }
  }

  setValue<K extends keyof Settings>(
    scope: SettingScope,
    key: K,
    value: Settings[K],
  ): void {
    const settingsFile = this.forScope(scope);
    settingsFile.settings[key] = value;
    this._merged = this.computeMergedSettings();
    saveSettings(settingsFile);
  }
}

function resolveEnvVarsInString(value: string): string {
  const envVarRegex = /\$(?:(\w+)|{([^}]+)})/g; // Find $VAR_NAME or ${VAR_NAME}
  return value.replace(envVarRegex, (match, varName1, varName2) => {
    const varName = varName1 || varName2;
    if (process && process.env && typeof process.env[varName] === 'string') {
      return process.env[varName]!;
    }
    return match;
  });
}

function resolveEnvVarsInObject<T>(obj: T): T {
  if (
    obj === null ||
    obj === undefined ||
    typeof obj === 'boolean' ||
    typeof obj === 'number'
  ) {
    return obj;
  }

  if (typeof obj === 'string') {
    return resolveEnvVarsInString(obj) as unknown as T;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => resolveEnvVarsInObject(item)) as unknown as T;
  }

  if (typeof obj === 'object') {
    const newObj = { ...obj } as T;
    for (const key in newObj) {
      if (Object.prototype.hasOwnProperty.call(newObj, key)) {
        newObj[key] = resolveEnvVarsInObject(newObj[key]);
      }
    }
    return newObj;
  }

  return obj;
}

export async function findEnvFile(startDir: string): Promise<string | null> {
  // synchronous traversal is small and fast, but we provide an async-friendly
  // alternative by using fs.promises.access. Implement as a synchronous loop
  // but using non-blocking access checks to avoid event-loop stalls.
  let currentDir = path.resolve(startDir);
  const access = fs.promises.access;
  while (true) {
    const geminiEnvPath = path.join(currentDir, GEMINI_DIR, '.env');
    try {
      await access(geminiEnvPath);
      return geminiEnvPath;
    } catch (_) {
      // file not found at this path
      /* no-op */
    }
    const envPath = path.join(currentDir, '.env');
    try {
      await access(envPath);
      return envPath;
    } catch (_) {
      // file not found at this path
      /* no-op */
    }
    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir || !parentDir) {
  const homeGeminiEnvPath = path.join(homedir(), GEMINI_DIR, '.env');
  const homeEnvPath = path.join(homedir(), '.env');
      try {
        await access(homeGeminiEnvPath);
        return homeGeminiEnvPath;
      } catch (_) {
        // file not found in home directory
        /* no-op */
      }
      try {
        await access(homeEnvPath);
        return homeEnvPath;
      } catch (_) {
        // file not found in home directory
        /* no-op */
      }
      return null; // Exit the loop if no .env file is found after checking home directories
    }
    currentDir = parentDir;
  }
}

export async function setUpCloudShellEnvironment(
  envFilePath: string | null,
): Promise<void> {
  // Special handling for GOOGLE_CLOUD_PROJECT in Cloud Shell:
  // Because GOOGLE_CLOUD_PROJECT in Cloud Shell tracks the project
  // set by the user using "gcloud config set project" we do not want to
  // use its value. So, unless the user overrides GOOGLE_CLOUD_PROJECT in
  // one of the .env files, we set the Cloud Shell-specific default here.
  if (envFilePath) {
    try {
      const envFileContent = await fs.promises.readFile(envFilePath);
      const parsedEnv = dotenv.parse(envFileContent as Buffer | string);
    if (parsedEnv['GOOGLE_CLOUD_PROJECT']) {
      // .env file takes precedence in Cloud Shell
      process.env['GOOGLE_CLOUD_PROJECT'] = parsedEnv['GOOGLE_CLOUD_PROJECT'];
    } else {
      // If not in .env, set to default and override global
      process.env['GOOGLE_CLOUD_PROJECT'] = 'cloudshell-gca';
    }
    } catch (_e) {
      // If read fails, still set default
      process.env['GOOGLE_CLOUD_PROJECT'] = 'cloudshell-gca';
    }
  }
}

export async function loadEnvironment(settings?: Settings): Promise<void> {
  const envFilePath = await findEnvFile(process.cwd());

  // Cloud Shell environment variable handling
  if (process.env['CLOUD_SHELL'] === 'true') {
    await setUpCloudShellEnvironment(envFilePath);
  }

  // If no settings provided, try to load workspace settings for exclusions
  let resolvedSettings = settings;
  if (!resolvedSettings) {
    const workspaceSettingsPath = new Storage(process.cwd()).getWorkspaceSettingsPath();
    try {
      try {
        const workspaceContent = await fs.promises.readFile(workspaceSettingsPath, 'utf-8');
        const parsedWorkspaceSettings = JSON.parse(stripJsonComments(workspaceContent)) as Settings;
        resolvedSettings = resolveEnvVarsInObject(parsedWorkspaceSettings);
      } catch (error) {
        // Ignore errors loading workspace settings but log for debugging
        logger.debug('Failed to load workspace settings:', error);
      }
    } catch (error) {
      // Ignore access errors but log for debugging
      logger.debug('Failed to access workspace settings file:', error);
    }
  }

  if (envFilePath) {
    try {
      const envFileContent = await fs.promises.readFile(envFilePath, 'utf-8');
      const parsedEnv = dotenv.parse(envFileContent);

      const excludedVars = resolvedSettings?.excludedProjectEnvVars || DEFAULT_EXCLUDED_ENV_VARS;
      const isProjectEnvFile = !envFilePath.includes(GEMINI_DIR);

      for (const key in parsedEnv) {
        if (Object.hasOwn(parsedEnv, key)) {
          if (isProjectEnvFile && excludedVars.includes(key)) {
            continue;
          }
          if (!Object.hasOwn(process.env, key)) {
            process.env[key] = parsedEnv[key];
          }
        }
      }
    } catch (error) {
      // Log parsing errors for debugging but continue execution
      logger.debug('Failed to parse environment file:', error);
    }
  }
}

/**
 * Resolves workspace and home directory paths to their canonical representation.
 * This handles symlinks and ensures consistent path resolution.
 */
async function resolveDirectories(workspaceDir: string): Promise<{
  realWorkspaceDir: string;
  realHomeDir: string;
  workspaceSettingsPath: string;
}> {
  // Resolve paths to their canonical representation to handle symlinks
  const resolvedWorkspaceDir = path.resolve(workspaceDir);
  const resolvedHomeDir = path.resolve(homedir());

  let realWorkspaceDir = resolvedWorkspaceDir;
  try {
    realWorkspaceDir = await fs.promises.realpath(resolvedWorkspaceDir);
  } catch (_e) {
    // path might not exist yet
  }

  // We expect homedir to always exist and be resolvable.
  let realHomeDir = resolvedHomeDir;
  try {
    realHomeDir = await fs.promises.realpath(resolvedHomeDir);
  } catch (_e) {
    // ignore
  }

  const workspaceSettingsPath = new Storage(workspaceDir).getWorkspaceSettingsPath();

  return {
    realWorkspaceDir,
    realHomeDir,
    workspaceSettingsPath,
  };
}

/**
 * Loads settings from a specific file path with error handling.
 */
async function loadSettingsFromFile(
  filePath: string,
  settingsErrors: SettingsError[],
): Promise<Settings> {
  try {
    const content = await fs.promises.readFile(filePath, 'utf-8');
    return JSON.parse(stripJsonComments(content)) as Settings;
  } catch (error: unknown) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      // ignore missing settings file
      return {};
    }
    settingsErrors.push({
      message: getErrorMessage(error),
      path: filePath,
    });
    return {};
  }
}

/**
 * Applies legacy theme name mappings to settings.
 */
function applyLegacyThemeMappings(settings: Settings): void {
  if (settings.theme && settings.theme === 'VS') {
    settings.theme = DefaultLight.name;
  } else if (settings.theme && settings.theme === 'VS2015') {
    settings.theme = DefaultDark.name;
  } else if (settings.theme && settings.theme === 'VSDark') {
    settings.theme = DefaultDark.name;
  }
}

/**
 * Loads settings from user and workspace directories.
 * Project settings override user settings.
 */
export async function loadSettings(workspaceDir: string): Promise<LoadedSettings> {
  let systemSettings: Settings = {};
  let userSettings: Settings = {};
  let workspaceSettings: Settings = {};
  const settingsErrors: SettingsError[] = [];
  const systemSettingsPath = getSystemSettingsPath();

  // Resolve paths to their canonical representation to handle symlinks
  const { realWorkspaceDir, realHomeDir, workspaceSettingsPath } = await resolveDirectories(workspaceDir);

  // Load system settings
  systemSettings = await loadSettingsFromFile(systemSettingsPath, settingsErrors);

  // Load user settings
  userSettings = await loadSettingsFromFile(USER_SETTINGS_PATH, settingsErrors);
  applyLegacyThemeMappings(userSettings);

  if (realWorkspaceDir !== realHomeDir) {
    // Load workspace settings
    workspaceSettings = await loadSettingsFromFile(workspaceSettingsPath, settingsErrors);
    applyLegacyThemeMappings(workspaceSettings);
  }

  // For the initial trust check, we can only use user and system settings.
  const initialTrustCheckSettings = { ...systemSettings, ...userSettings };
  const isTrusted = (await isWorkspaceTrusted(initialTrustCheckSettings)) ?? true;

  // Create a temporary merged settings object to pass to loadEnvironment.
  const tempMergedSettings = mergeSettings(
    systemSettings,
    userSettings,
    workspaceSettings,
    isTrusted,
  );

  // loadEnviroment depends on settings so we have to create a temp version of
  // the settings to avoid a cycle
  await loadEnvironment(tempMergedSettings);

  // Now that the environment is loaded, resolve variables in the settings.
  systemSettings = resolveEnvVarsInObject(systemSettings);
  userSettings = resolveEnvVarsInObject(userSettings);
  workspaceSettings = resolveEnvVarsInObject(workspaceSettings);

  // Create LoadedSettings first
  const loadedSettings = new LoadedSettings(
    {
      path: systemSettingsPath,
      settings: systemSettings,
    },
    {
      path: USER_SETTINGS_PATH,
      settings: userSettings,
    },
    {
      path: workspaceSettingsPath,
      settings: workspaceSettings,
    },
    settingsErrors,
    isTrusted,
  );

  // Validate chatCompression settings
  const chatCompression = loadedSettings.merged.chatCompression;
  const threshold = chatCompression?.contextPercentageThreshold;
  if (
    threshold != null &&
    (typeof threshold !== 'number' || threshold < 0 || threshold > 1)
  ) {
    logger.warn(
      `Invalid value for chatCompression.contextPercentageThreshold: "${threshold}". Please use a value between 0 and 1. Using default compression settings.`,
    );
    delete loadedSettings.merged.chatCompression;
  }

  return loadedSettings;
}
export async function saveSettings(settingsFile: SettingsFile): Promise<void> {
  try {
    const dirPath = path.dirname(settingsFile.path);
    try {
      await fs.promises.access(dirPath);
    } catch (_) {
      await fs.promises.mkdir(dirPath, { recursive: true });
    }

    await fs.promises.writeFile(
      settingsFile.path,
      JSON.stringify(settingsFile.settings, null, 2),
      'utf-8',
    );
  } catch (error) {
    logger.error('Error saving user settings file:', error);
  }
}
