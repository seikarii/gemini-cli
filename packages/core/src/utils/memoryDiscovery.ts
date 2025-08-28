/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import { homedir } from 'os';
import { bfsFileSearch } from './bfsFileSearch.js';
import {
  GEMINI_CONFIG_DIR,
  getAllGeminiMdFilenames,
} from '../tools/memoryTool.js';
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';
import { processImports } from './memoryImportProcessor.js';
import {
  DEFAULT_MEMORY_FILE_FILTERING_OPTIONS,
  FileFilteringOptions,
} from '../config/config.js';
import { FileSystemService } from '../services/fileSystemService.js';
import { OptimizedFileOperations } from './fileOperations/OptimizedFileOperations.js';

// Simple console logger, similar to the one previously in CLI's config.ts
// TODO: Integrate with a more robust server-side logger if available/appropriate.
const logger = {
  debug: (...args: unknown[]) =>
    console.debug('[DEBUG] [MemoryDiscovery]', ...args),
  warn: (...args: unknown[]) =>
    console.warn('[WARN] [MemoryDiscovery]', ...args),
  error: (...args: unknown[]) =>
    console.error('[ERROR] [MemoryDiscovery]', ...args),
};

interface GeminiFileContent {
  filePath: string;
  content: string | null;
}

// Use optimized file operations instead of simple Map cache
const optimizedFileOps = OptimizedFileOperations.getInstance({
  enableCache: true,
  cacheTTL: 10 * 60 * 1000, // 10 minutes for project roots
  maxCacheSize: 1000,
});

async function findProjectRoot(
  startDir: string,
  fileSystemService?: FileSystemService,
): Promise<string | null> {
  // Use optimized file operations for caching
  const result = await optimizedFileOps.findProjectRoot(startDir, fileSystemService);
  return result;
}

async function getGeminiMdFilePathsInternal(
  currentWorkingDirectory: string,
  includeDirectoriesToReadGemini: readonly string[],
  userHomePath: string,
  debugMode: boolean,
  fileService: FileDiscoveryService,
  extensionContextFilePaths: string[] = [],
  fileFilteringOptions: FileFilteringOptions,
  maxDirs: number,
  fileSystemService?: FileSystemService,
): Promise<string[]> {
  const dirs = new Set<string>([
    ...includeDirectoriesToReadGemini,
    currentWorkingDirectory,
  ]);

  // Process directories in parallel with concurrency limit to prevent EMFILE errors
  const CONCURRENT_LIMIT = 10;
  const dirsArray = Array.from(dirs);
  const pathsArrays: string[][] = [];

  for (let i = 0; i < dirsArray.length; i += CONCURRENT_LIMIT) {
    const batch = dirsArray.slice(i, i + CONCURRENT_LIMIT);
    const batchPromises = batch.map((dir) =>
      getGeminiMdFilePathsInternalForEachDir(
        dir,
        userHomePath,
        debugMode,
        fileService,
        extensionContextFilePaths,
        fileFilteringOptions,
        maxDirs,
        fileSystemService,
      ),
    );

    const batchResults = await Promise.allSettled(batchPromises);

    for (const result of batchResults) {
      if (result.status === 'fulfilled') {
        pathsArrays.push(result.value);
      } else {
        const error = result.reason;
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`Error discovering files in directory: ${message}`);
        // Continue processing other directories
      }
    }
  }

  const paths = pathsArrays.flat();
  return Array.from(new Set<string>(paths));
}

async function getGeminiMdFilePathsInternalForEachDir(
  dir: string,
  userHomePath: string,
  debugMode: boolean,
  fileService: FileDiscoveryService,
  extensionContextFilePaths: string[] = [],
  fileFilteringOptions: FileFilteringOptions,
  maxDirs: number,
  fileSystemService?: FileSystemService,
): Promise<string[]> {
  const allPaths = new Set<string>();
  const geminiMdFilenames = getAllGeminiMdFilenames();

  for (const geminiMdFilename of geminiMdFilenames) {
    const resolvedHome = path.resolve(userHomePath);
    const globalMemoryPath = path.join(
      resolvedHome,
      GEMINI_CONFIG_DIR,
      geminiMdFilename,
    );

    // This part that finds the global file always runs.
    try {
      if (fileSystemService) {
        // Use FileSystemService for standardized file operations
        const fileExists = await fileSystemService.exists(globalMemoryPath);
        if (!fileExists) {
          throw new Error('File does not exist');
        }
      } else {
        // Fallback to direct fs.access for backward compatibility
        await fs.access(globalMemoryPath, fsSync.constants.R_OK);
      }
      allPaths.add(globalMemoryPath);
      if (debugMode)
        logger.debug(
          `Found readable global ${geminiMdFilename}: ${globalMemoryPath}`,
        );
    } catch {
      // It's okay if it's not found.
    }

    // FIX: Only perform the workspace search (upward and downward scans)
    // if a valid currentWorkingDirectory is provided.
    if (dir) {
      const resolvedCwd = path.resolve(dir);
      if (debugMode)
        logger.debug(
          `Searching for ${geminiMdFilename} starting from CWD: ${resolvedCwd}`,
        );

      const projectRoot = await findProjectRoot(resolvedCwd, fileSystemService);
      if (debugMode)
        logger.debug(`Determined project root: ${projectRoot ?? 'None'}`);

      const upwardPaths: string[] = [];
      let currentDir = resolvedCwd;
      const ultimateStopDir = projectRoot
        ? path.dirname(projectRoot)
        : path.dirname(resolvedHome);

      while (currentDir && currentDir !== path.dirname(currentDir)) {
        if (currentDir === path.join(resolvedHome, GEMINI_CONFIG_DIR)) {
          break;
        }

        const potentialPath = path.join(currentDir, geminiMdFilename);
        try {
          if (fileSystemService) {
            // Use FileSystemService for standardized file operations
            const fileExists = await fileSystemService.exists(potentialPath);
            if (!fileExists) {
              throw new Error('File does not exist');
            }
          } else {
            // Fallback to direct fs.access for backward compatibility
            await fs.access(potentialPath, fsSync.constants.R_OK);
          }
          if (potentialPath !== globalMemoryPath) {
            upwardPaths.unshift(potentialPath);
          }
        } catch {
          // Not found, continue.
        }

        if (currentDir === ultimateStopDir) {
          break;
        }

        currentDir = path.dirname(currentDir);
      }
      upwardPaths.forEach((p) => allPaths.add(p));

      const mergedOptions = {
        ...DEFAULT_MEMORY_FILE_FILTERING_OPTIONS,
        ...fileFilteringOptions,
      };

      const downwardPaths = await bfsFileSearch(resolvedCwd, {
        fileName: geminiMdFilename,
        maxDirs,
        debug: debugMode,
        fileService,
        fileFilteringOptions: mergedOptions,
      });
      downwardPaths.sort();
      for (const dPath of downwardPaths) {
        allPaths.add(dPath);
      }
    }
  }

  // Add extension context file paths.
  for (const extensionPath of extensionContextFilePaths) {
    allPaths.add(extensionPath);
  }

  const finalPaths = Array.from(allPaths);

  if (debugMode)
    logger.debug(
      `Final ordered ${getAllGeminiMdFilenames()} paths to read: ${JSON.stringify(
        finalPaths,
      )}`,
    );
  return finalPaths;
}

async function readGeminiMdFiles(
  filePaths: string[],
  debugMode: boolean,
  importFormat: 'flat' | 'tree' = 'tree',
  fileSystemService?: FileSystemService,
): Promise<GeminiFileContent[]> {
  // Use optimized file operations for batch reading
  const fileResults = await optimizedFileOps.readManyFiles(filePaths);
  const results: GeminiFileContent[] = [];

  for (const [filePath, result] of fileResults.entries()) {
    if (result.success && result.data) {
      try {
        // Process imports in the content
        const processedResult = await processImports(
          result.data,
          path.dirname(filePath),
          debugMode,
          undefined,
          undefined,
          importFormat,
          fileSystemService,
        );
        
        if (debugMode && result.fromCache) {
          logger.debug(
            `Successfully read from cache and processed imports: ${filePath} (Length: ${processedResult.content.length})`,
          );
        } else if (debugMode) {
          logger.debug(
            `Successfully read and processed imports: ${filePath} (Length: ${processedResult.content.length})`,
          );
        }

        results.push({ filePath, content: processedResult.content });
      } catch (error: unknown) {
        const isTestEnv =
          process.env['NODE_ENV'] === 'test' || process.env['VITEST'];
        if (!isTestEnv) {
          const message =
            error instanceof Error ? error.message : String(error);
          logger.warn(
            `Warning: Could not process imports for ${getAllGeminiMdFilenames()} file at ${filePath}. Error: ${message}`,
          );
        }
        if (debugMode) logger.debug(`Failed to process: ${filePath}`);
        results.push({ filePath, content: result.data }); // Include raw content
      }
    } else {
      const isTestEnv =
        process.env['NODE_ENV'] === 'test' || process.env['VITEST'];
      if (!isTestEnv) {
        logger.warn(
          `Warning: Could not read ${getAllGeminiMdFilenames()} file at ${filePath}. Error: ${result.error}`,
        );
      }
      if (debugMode) logger.debug(`Failed to read: ${filePath}`);
      results.push({ filePath, content: null }); // Still include it with null content
    }
  }

  return results;
}

function concatenateInstructions(
  instructionContents: GeminiFileContent[],
  // CWD is needed to resolve relative paths for display markers
  currentWorkingDirectoryForDisplay: string,
): string {
  return instructionContents
    .filter((item) => typeof item.content === 'string')
    .map((item) => {
      const trimmedContent = (item.content as string).trim();
      if (trimmedContent.length === 0) {
        return null;
      }
      const displayPath = path.isAbsolute(item.filePath)
        ? path.relative(currentWorkingDirectoryForDisplay, item.filePath)
        : item.filePath;
      return `--- Context from: ${displayPath} ---\n${trimmedContent}\n--- End of Context from: ${displayPath} ---`;
    })
    .filter((block): block is string => block !== null)
    .join('\n\n');
}

/**
 * Loads hierarchical GEMINI.md files and concatenates their content.
 * This function is intended for use by the server.
 */
export async function loadServerHierarchicalMemory(
  currentWorkingDirectory: string,
  includeDirectoriesToReadGemini: readonly string[],
  debugMode: boolean,
  fileService: FileDiscoveryService,
  extensionContextFilePaths: string[] = [],
  importFormat: 'flat' | 'tree' = 'tree',
  fileFilteringOptions?: FileFilteringOptions,
  maxDirs: number = 200,
  fileSystemService?: FileSystemService,
): Promise<{ memoryContent: string; fileCount: number }> {
  if (debugMode)
    logger.debug(
      `Loading server hierarchical memory for CWD: ${currentWorkingDirectory} (importFormat: ${importFormat})`,
    );

  // For the server, homedir() refers to the server process's home.
  // This is consistent with how MemoryTool already finds the global path.
  const userHomePath = homedir();
  const filePaths = await getGeminiMdFilePathsInternal(
    currentWorkingDirectory,
    includeDirectoriesToReadGemini,
    userHomePath,
    debugMode,
    fileService,
    extensionContextFilePaths,
    fileFilteringOptions || DEFAULT_MEMORY_FILE_FILTERING_OPTIONS,
    maxDirs,
    fileSystemService,
  );
  if (filePaths.length === 0) {
    if (debugMode)
      logger.debug('No GEMINI.md files found in hierarchy of the workspace.');
    return { memoryContent: '', fileCount: 0 };
  }
  const contentsWithPaths = await readGeminiMdFiles(
    filePaths,
    debugMode,
    importFormat,
    fileSystemService,
  );
  // Pass CWD for relative path display in concatenated content
  const combinedInstructions = concatenateInstructions(
    contentsWithPaths,
    currentWorkingDirectory,
  );
  if (debugMode)
    logger.debug(
      `Combined instructions length: ${combinedInstructions.length}`,
    );
  if (debugMode && combinedInstructions.length > 0)
    logger.debug(
      `Combined instructions (snippet): ${combinedInstructions.substring(0, 500)}...`,
    );
  return {
    memoryContent: combinedInstructions,
    fileCount: contentsWithPaths.length,
  };
}
