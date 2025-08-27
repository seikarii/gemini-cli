/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Settings, MemoryImportFormat } from './settingsSchema.js';
export type { Settings, MemoryImportFormat };
export declare const SETTINGS_DIRECTORY_NAME = ".gemini";
export declare const USER_SETTINGS_PATH: string;
export declare const USER_SETTINGS_DIR: string;
export declare const DEFAULT_EXCLUDED_ENV_VARS: string[];
export declare function getSystemSettingsPath(): string;
export type { DnsResolutionOrder } from './settingsSchema.js';
export declare enum SettingScope {
    User = "User",
    Workspace = "Workspace",
    System = "System"
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
export declare class LoadedSettings {
    constructor(system: SettingsFile, user: SettingsFile, workspace: SettingsFile, errors: SettingsError[], isTrusted: boolean);
    readonly system: SettingsFile;
    readonly user: SettingsFile;
    readonly workspace: SettingsFile;
    readonly errors: SettingsError[];
    readonly isTrusted: boolean;
    private _merged;
    get merged(): Settings;
    private computeMergedSettings;
    forScope(scope: SettingScope): SettingsFile;
    setValue<K extends keyof Settings>(scope: SettingScope, key: K, value: Settings[K]): void;
}
export declare function findEnvFile(startDir: string): Promise<string | null>;
export declare function setUpCloudShellEnvironment(envFilePath: string | null): Promise<void>;
export declare function loadEnvironment(settings?: Settings): Promise<void>;
/**
 * Loads settings from user and workspace directories.
 * Project settings override user settings.
 */
export declare function loadSettings(workspaceDir: string): Promise<LoadedSettings>;
export declare function saveSettings(settingsFile: SettingsFile): Promise<void>;
