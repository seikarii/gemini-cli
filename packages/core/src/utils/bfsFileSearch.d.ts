/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';
import { FileFilteringOptions } from '../config/config.js';
import { FileSystemService } from '../services/fileSystemService.js';
interface BfsFileSearchOptions {
    fileName: string;
    ignoreDirs?: string[];
    maxDirs?: number;
    debug?: boolean;
    fileService?: FileDiscoveryService;
    fileFilteringOptions?: FileFilteringOptions;
    maxConcurrency?: number;
    batchSize?: number;
    cacheSize?: number;
    cacheTtlMs?: number;
    fileSystemService?: FileSystemService;
}
/**
 * Performs a breadth-first search for a specific file within a directory structure.
 * Optimized with batch processing, smart caching, and performance monitoring.
 *
 * @param rootDir The directory to start the search from.
 * @param options Configuration for the search.
 * @returns A promise that resolves to an array of paths where the file was found.
 */
export declare function bfsFileSearch(rootDir: string, options: BfsFileSearchOptions): Promise<string[]>;
export {};
