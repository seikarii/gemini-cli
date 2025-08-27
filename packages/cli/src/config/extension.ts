/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  MCPServerConfig,
  GeminiCLIExtension,
  Storage,
} from '@google/gemini-cli-core';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { EXTENSIONS_CONFIG_FILENAME, DEFAULT_CONTEXT_FILENAME } from './constants.js';
import { logger } from './logger.js';

export interface Extension {
  path: string;
  config: ExtensionConfig;
  contextFiles: string[];
}

export interface ExtensionConfig {
  name: string;
  version: string;
  mcpServers?: Record<string, MCPServerConfig>;
  contextFileName?: string | string[];
  excludeTools?: string[];
}

export async function loadExtensions(workspaceDir: string): Promise<Extension[]> {
  const fromWorkspace = await loadExtensionsFromDir(workspaceDir);
  const fromHome = await loadExtensionsFromDir(os.homedir());

  const allExtensions = [...fromWorkspace, ...fromHome];

  const uniqueExtensions = new Map<string, Extension>();
  for (const extension of allExtensions) {
    if (!uniqueExtensions.has(extension.config.name)) {
      uniqueExtensions.set(extension.config.name, extension);
    }
  }

  return Array.from(uniqueExtensions.values());
}

async function loadExtensionsFromDir(dir: string): Promise<Extension[]> {
  const storage = new Storage(dir);
  const extensionsDir = storage.getExtensionsDir();
  try {
    await fs.promises.access(extensionsDir);
  } catch (error) {
    // Only ignore ENOENT (directory doesn't exist), log other errors
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return [];
    }
    logger.warn(`Error accessing extensions directory ${extensionsDir}: ${error}`);
    return [];
  }

  const entries = await fs.promises.readdir(extensionsDir);

  const extensions: Extension[] = [];
  for (const subdir of entries) {
    const extensionDir = path.join(extensionsDir, subdir);

    const extension = await loadExtension(extensionDir);
    if (extension != null) {
      extensions.push(extension);
    }
  }
  return extensions;
}

async function loadExtension(extensionDir: string): Promise<Extension | null> {
  try {
    const stat = await fs.promises.stat(extensionDir);
    if (!stat.isDirectory()) {
      logger.warn(
        `unexpected file ${extensionDir} in extensions directory.`,
      );
      return null;
    }
  } catch (error) {
    // Only ignore ENOENT (path doesn't exist), log other errors as they might indicate permission issues
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return null;
    }
    logger.warn(`Error accessing extension directory ${extensionDir}: ${error}`);
    return null;
  }

  const configFilePath = path.join(extensionDir, EXTENSIONS_CONFIG_FILENAME);
  try {
    await fs.promises.access(configFilePath);
  } catch (error) {
    // Only ignore ENOENT (config file doesn't exist), log other errors
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      logger.warn(
        `extension directory ${extensionDir} does not contain a config file ${configFilePath}.`,
      );
      return null;
    }
    logger.warn(`Error accessing config file ${configFilePath}: ${error}`);
    return null;
  }

  try {
    const configContent = await fs.promises.readFile(configFilePath, 'utf-8');
    const config = JSON.parse(configContent) as ExtensionConfig;
    if (!config.name || !config.version) {
      logger.error(
        `Invalid extension config in ${configFilePath}: missing name or version.`,
      );
      return null;
    }

    const contextFileNames = getContextFileNames(config);
    const contextFiles: string[] = [];
    for (const contextFileName of contextFileNames) {
      const contextFilePath = path.join(extensionDir, contextFileName);
      try {
        await fs.promises.access(contextFilePath);
        contextFiles.push(contextFilePath);
      } catch (error) {
        // Only ignore ENOENT (context file doesn't exist), log other errors
        if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) {
          logger.warn(`Error accessing context file ${contextFilePath}: ${error}`);
        }
        // If ENOENT or other error, skip this context file
      }
    }

    return {
      path: extensionDir,
      config,
      contextFiles,
    };
  } catch (e) {
    logger.error(
      `error parsing extension config in ${configFilePath}: ${e}`,
    );
    return null;
  }
}

function getContextFileNames(config: ExtensionConfig): string[] {
  if (!config.contextFileName) {
    return [DEFAULT_CONTEXT_FILENAME];
  } else if (!Array.isArray(config.contextFileName)) {
    return [config.contextFileName];
  }
  return config.contextFileName;
}

export function annotateActiveExtensions(
  extensions: Extension[],
  enabledExtensionNames: string[],
): GeminiCLIExtension[] {
  const annotatedExtensions: GeminiCLIExtension[] = [];

  // If no specific extensions are enabled, enable all extensions by default
  if (enabledExtensionNames.length === 0) {
    return extensions.map((extension) => ({
      name: extension.config.name,
      version: extension.config.version,
      isActive: true,
      path: extension.path,
    }));
  }

  // Normalize extension names to lowercase for case-insensitive comparison
  const lowerCaseEnabledExtensions = new Set(
    enabledExtensionNames.map((e) => e.trim().toLowerCase()),
  );

  // Special case: if only "none" is specified, disable all extensions
  if (
    lowerCaseEnabledExtensions.size === 1 &&
    lowerCaseEnabledExtensions.has('none')
  ) {
    return extensions.map((extension) => ({
      name: extension.config.name,
      version: extension.config.version,
      isActive: false,
      path: extension.path,
    }));
  }

  // Track which requested extensions were not found
  const notFoundNames = new Set(lowerCaseEnabledExtensions);

  // Process each available extension and mark it as active if it was requested
  for (const extension of extensions) {
    const lowerCaseName = extension.config.name.toLowerCase();
    const isActive = lowerCaseEnabledExtensions.has(lowerCaseName);

    if (isActive) {
      notFoundNames.delete(lowerCaseName); // Remove from not-found set
    }

    annotatedExtensions.push({
      name: extension.config.name,
      version: extension.config.version,
      isActive,
      path: extension.path,
    });
  }

  // Report any requested extensions that were not found
  for (const requestedName of notFoundNames) {
    logger.error(`Extension not found: ${requestedName}`);
  }

  return annotatedExtensions;
}
