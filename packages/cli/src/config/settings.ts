/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
// Removed unused import promisify
import * as path from 'path';
import { homedir } from 'os';
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

// System paths configuration by platform
const SYSTEM_PATHS_CONFIG = {
  darwin: '/Library/Application Support/GeminiCli/settings.json',
  win32: 'C:\\ProgramData\\gemini-cli\\settings.json',
  linux: '/etc/gemini-cli/settings.json',
} as const;

export function getSystemSettingsPath(): string {
  if (process.env['GEMINI_CLI_SYSTEM_SETTINGS_PATH']) {
    return process.env['GEMINI_CLI_SYSTEM_SETTINGS_PATH'];
  }

  const platform = process.platform as keyof typeof SYSTEM_PATHS_CONFIG;
  return SYSTEM_PATHS_CONFIG[platform] || SYSTEM_PATHS_CONFIG.linux;
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

// Utility function for deep merging settings objects
function deepMergeSettings(target: Settings, ...sources: Settings[]): Settings {
  const result = { ...target } as Record<string, unknown>;

  for (const source of sources) {
    const sourceRecord = source as Record<string, unknown>;
    for (const key in sourceRecord) {
      if (Object.prototype.hasOwnProperty.call(sourceRecord, key)) {
        const sourceValue = sourceRecord[key];
        const targetValue = result[key];

        if (
          sourceValue !== null &&
          typeof sourceValue === 'object' &&
          !Array.isArray(sourceValue)
        ) {
          // Deep merge objects
          const targetObj =
            targetValue &&
            typeof targetValue === 'object' &&
            !Array.isArray(targetValue)
              ? (targetValue as Record<string, unknown>)
              : {};
          result[key] = deepMergeSettings(
            targetObj as Settings,
            sourceValue as Settings,
          );
        } else if (Array.isArray(sourceValue)) {
          // Concatenate arrays
          const targetArray = Array.isArray(targetValue)
            ? (targetValue as unknown[])
            : [];
          result[key] = [...targetArray, ...sourceValue];
        } else {
          // Override with source value
          result[key] = sourceValue;
        }
      }
    }
  }

  return result as Settings;
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

  return deepMergeSettings(
    {} as Settings,
    system,
    user,
    safeWorkspaceWithoutFolderTrust,
  );
}

export class SettingsManager {
  /**
   * Saves settings to a file with proper error handling.
   */
  static async saveSettings(settingsFile: SettingsFile): Promise<void> {
    try {
      const dirPath = path.dirname(settingsFile.path);
      try {
        await fs.promises.access(dirPath);
      } catch (error) {
        // Only ignore ENOENT (directory doesn't exist), log other access errors
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
          await fs.promises.mkdir(dirPath, { recursive: true });
        } else {
          logger.debug('Error accessing settings directory:', error);
          throw error;
        }
      }

      await fs.promises.writeFile(
        settingsFile.path,
        JSON.stringify(settingsFile.settings, null, 2),
        'utf-8',
      );
    } catch (error) {
      logger.error('Error saving settings file:', error);
    }
  }
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
    SettingsManager.saveSettings(settingsFile);
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
    return resolveEnvVarsInString(obj) as T;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => resolveEnvVarsInObject(item)) as T;
  }

  if (typeof obj === 'object') {
    const newObj: Record<string, unknown> = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        newObj[key] = resolveEnvVarsInObject(
          (obj as Record<string, unknown>)[key],
        );
      }
    }
    return newObj as T;
  }

  return obj;
}

export async function findEnvFile(startDir: string): Promise<string | null> {
  // synchronous traversal is small and fast, but we provide an async-friendly
  // alternative by using fs.promises.access. Implement as a synchronous loop
  // but using non-blocking access checks to avoid event-loop stalls.
  let currentDir = path.resolve(startDir);

  // Check for .env files in directory hierarchy
  while (true) {
    const geminiEnvPath = path.join(currentDir, GEMINI_DIR, '.env');
    if (await tryAccessPath(geminiEnvPath)) {
      return geminiEnvPath;
    }

    const envPath = path.join(currentDir, '.env');
    if (await tryAccessPath(envPath)) {
      return envPath;
    }

    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir || !parentDir) {
      break; // Reached root directory
    }
    currentDir = parentDir;
  }

  // Check home directory .env files
  return await findEnvFileInHome();
}

async function tryAccessPath(filePath: string): Promise<boolean> {
  try {
    await fs.promises.access(filePath);
    return true;
  } catch (error) {
    // Only ignore ENOENT (file not found), log other errors for debugging
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      logger.debug('Error accessing env file:', error);
    }
    return false;
  }
}

async function findEnvFileInHome(): Promise<string | null> {
  const homeGeminiEnvPath = path.join(homedir(), GEMINI_DIR, '.env');
  if (await tryAccessPath(homeGeminiEnvPath)) {
    return homeGeminiEnvPath;
  }

  const homeEnvPath = path.join(homedir(), '.env');
  if (await tryAccessPath(homeEnvPath)) {
    return homeEnvPath;
  }

  return null;
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
    } catch (error) {
      // Log read/parse errors for debugging but continue with default
      logger.debug('Error reading or parsing .env file in Cloud Shell:', error);
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
    const workspaceSettingsPath = new Storage(
      process.cwd(),
    ).getWorkspaceSettingsPath();
    try {
      try {
        const workspaceContent = await fs.promises.readFile(
          workspaceSettingsPath,
          'utf-8',
        );
        const parsedWorkspaceSettings = JSON.parse(
          stripJsonComments(workspaceContent),
        ) as Settings;
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

      const excludedVars =
        resolvedSettings?.excludedProjectEnvVars || DEFAULT_EXCLUDED_ENV_VARS;
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
  } catch (error) {
    // Log error if path resolution fails, but continue with unresolved path
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      logger.debug('Error resolving workspace directory path:', error);
    }
    // Keep the unresolved path if realpath fails
  }

  // We expect homedir to always exist and be resolvable.
  let realHomeDir = resolvedHomeDir;
  try {
    realHomeDir = await fs.promises.realpath(resolvedHomeDir);
  } catch (error) {
    // Log error but continue - homedir might not be resolvable in some environments
    logger.debug('Error resolving home directory path:', error);
  }

  const workspaceSettingsPath = new Storage(
    workspaceDir,
  ).getWorkspaceSettingsPath();

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
 *
 * The loading process follows these steps:
 * 1. Resolve canonical paths for workspace and home directories
 * 2. Load system, user, and workspace settings files
 * 3. Apply legacy theme mappings to user and workspace settings
 * 4. Perform initial trust check using only user and system settings
 * 5. Load environment variables (may depend on merged settings)
 * 6. Resolve environment variables in all settings
 * 7. Create and validate final LoadedSettings object
 */
export async function loadSettings(
  workspaceDir: string,
): Promise<LoadedSettings> {
  const settingsErrors: SettingsError[] = [];

  // Step 1: Resolve canonical paths
  const { realWorkspaceDir, realHomeDir, workspaceSettingsPath } =
    await resolveDirectories(workspaceDir);

  // Step 2: Load settings from different scopes
  const systemSettingsPath = getSystemSettingsPath();
  const systemSettings = await loadSettingsFromFile(
    systemSettingsPath,
    settingsErrors,
  );
  const userSettings = await loadSettingsFromFile(
    USER_SETTINGS_PATH,
    settingsErrors,
  );
  const workspaceSettings =
    realWorkspaceDir !== realHomeDir
      ? await loadSettingsFromFile(workspaceSettingsPath, settingsErrors)
      : {};

  // Step 3: Apply legacy theme mappings
  applyLegacyThemeMappings(userSettings);
  applyLegacyThemeMappings(workspaceSettings);

  // Step 4: Initial trust check (before loading environment)
  const initialTrustCheckSettings = deepMergeSettings(
    {} as Settings,
    systemSettings,
    userSettings,
  );
  const isTrusted =
    (await isWorkspaceTrusted(initialTrustCheckSettings)) ?? true;

  // Step 5: Load environment (depends on settings to avoid circular dependency)
  const tempMergedSettings = mergeSettings(
    systemSettings,
    userSettings,
    workspaceSettings,
    isTrusted,
  );
  await loadEnvironment(tempMergedSettings);

  // Step 6: Resolve environment variables in settings
  const resolvedSystemSettings = resolveEnvVarsInObject(systemSettings);
  const resolvedUserSettings = resolveEnvVarsInObject(userSettings);
  const resolvedWorkspaceSettings = resolveEnvVarsInObject(workspaceSettings);

  // Step 7: Create and validate LoadedSettings
  const loadedSettings = new LoadedSettings(
    { path: systemSettingsPath, settings: resolvedSystemSettings },
    { path: USER_SETTINGS_PATH, settings: resolvedUserSettings },
    { path: workspaceSettingsPath, settings: resolvedWorkspaceSettings },
    settingsErrors,
    isTrusted,
  );

  // Validate critical settings
  validateChatCompressionSettings(loadedSettings);

  return loadedSettings;
}

/**
 * Validates chat compression settings and logs warnings for invalid values.
 */
function validateChatCompressionSettings(loadedSettings: LoadedSettings): void {
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
}
