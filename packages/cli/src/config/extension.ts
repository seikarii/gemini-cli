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

export const EXTENSIONS_CONFIG_FILENAME = 'gemini-extension.json';

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
  } catch (_) {
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
      console.error(
        `Warning: unexpected file ${extensionDir} in extensions directory.`,
      );
      return null;
    }
  } catch (_e) {
    // If the path doesn't exist or is not accessible, skip it
    return null;
  }

  const configFilePath = path.join(extensionDir, EXTENSIONS_CONFIG_FILENAME);
  try {
    await fs.promises.access(configFilePath);
  } catch (_) {
    console.error(
      `Warning: extension directory ${extensionDir} does not contain a config file ${configFilePath}.`,
    );
    return null;
  }

  try {
    const configContent = await fs.promises.readFile(configFilePath, 'utf-8');
    const config = JSON.parse(configContent) as ExtensionConfig;
    if (!config.name || !config.version) {
      console.error(
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
      } catch (_) {
        // not present, skip
      }
    }

    return {
      path: extensionDir,
      config,
      contextFiles,
    };
  } catch (e) {
    console.error(
      `Warning: error parsing extension config in ${configFilePath}: ${e}`,
    );
    return null;
  }
}

function getContextFileNames(config: ExtensionConfig): string[] {
  if (!config.contextFileName) {
    return ['GEMINI.md'];
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

  if (enabledExtensionNames.length === 0) {
    return extensions.map((extension) => ({
      name: extension.config.name,
      version: extension.config.version,
      isActive: true,
      path: extension.path,
    }));
  }

  const lowerCaseEnabledExtensions = new Set(
    enabledExtensionNames.map((e) => e.trim().toLowerCase()),
  );

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

  const notFoundNames = new Set(lowerCaseEnabledExtensions);

  for (const extension of extensions) {
    const lowerCaseName = extension.config.name.toLowerCase();
    const isActive = lowerCaseEnabledExtensions.has(lowerCaseName);

    if (isActive) {
      notFoundNames.delete(lowerCaseName);
    }

    annotatedExtensions.push({
      name: extension.config.name,
      version: extension.config.version,
      isActive,
      path: extension.path,
    });
  }

  for (const requestedName of notFoundNames) {
    console.error(`Extension not found: ${requestedName}`);
  }

  return annotatedExtensions;
}
