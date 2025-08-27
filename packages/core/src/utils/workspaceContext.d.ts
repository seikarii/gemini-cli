/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { FileSystemService } from '../services/fileSystemService.js';
export type Unsubscribe = () => void;
/**
 * WorkspaceContext manages multiple workspace directories and validates paths
 * against them. This allows the CLI to operate on files from multiple directories
 * in a single session.
 */
export declare class WorkspaceContext {
    private directories;
    private initialDirectories;
    private onDirectoriesChangedListeners;
    private fileSystemService?;
    /**
     * Creates a new WorkspaceContext with the given initial directory and optional additional directories.
     * @param directory The initial working directory (usually cwd)
     * @param additionalDirectories Optional array of additional directories to include
     */
    constructor(directory: string, additionalDirectories?: string[]);
    /**
     * Registers a listener that is called when the workspace directories change.
     * @param listener The listener to call.
     * @returns A function to unsubscribe the listener.
     */
    onDirectoriesChanged(listener: () => void): Unsubscribe;
    /**
     * Sets the FileSystemService to use for file operations.
     * @param fileSystemService The FileSystemService instance to use
     */
    setFileSystemService(fileSystemService: FileSystemService): void;
    private notifyDirectoriesChanged;
    /**
     * Adds a directory to the workspace.
     * @param directory The directory path to add (can be relative or absolute)
     * @param basePath Optional base path for resolving relative paths (defaults to cwd)
     */
    addDirectory(directory: string, basePath?: string): void;
    /**
     * Async version of addDirectory that uses FileSystemService when available
     * @param directory The directory path to add (can be relative or absolute)
     * @param basePath Optional base path for resolving relative paths (defaults to cwd)
     */
    addDirectoryAsync(directory: string, basePath?: string): Promise<void>;
    private resolveAndValidateDir;
    /**
     * Async version of resolveAndValidateDir that uses FileSystemService when available
     */
    private resolveAndValidateDirAsync;
    /**
     * Gets a copy of all workspace directories.
     * @returns Array of absolute directory paths
     */
    getDirectories(): readonly string[];
    getInitialDirectories(): readonly string[];
    setDirectories(directories: readonly string[]): void;
    /**
     * Async version of setDirectories that uses FileSystemService when available
     * @param directories Array of directory paths to set
     */
    setDirectoriesAsync(directories: readonly string[]): Promise<void>;
    /**
     * Checks if a given path is within any of the workspace directories.
     * @param pathToCheck The path to validate
     * @returns True if the path is within the workspace, false otherwise
     */
    isPathWithinWorkspace(pathToCheck: string): boolean;
    /**
     * Async version of isPathWithinWorkspace that uses FileSystemService when available
     * @param pathToCheck The path to validate
     * @returns True if the path is within the workspace, false otherwise
     */
    isPathWithinWorkspaceAsync(pathToCheck: string): Promise<boolean>;
    /**
     * Fully resolves a path, including symbolic links.
     * If the path does not exist, it returns the fully resolved path as it would be
     * if it did exist.
     */
    private fullyResolvedPath;
    /**
     * Async version of fullyResolvedPath that uses FileSystemService when available
     */
    private fullyResolvedPathAsync;
    /**
     * Checks if a path is within a given root directory.
     * @param pathToCheck The absolute path to check
     * @param rootDirectory The absolute root directory
     * @returns True if the path is within the root directory, false otherwise
     */
    private isPathWithinRoot;
    /**
     * Checks if a file path is a symbolic link that points to a file.
     */
    private isFileSymlink;
    /**
     * Async version of isFileSymlink that uses FileSystemService when available
     */
    private isFileSymlinkAsync;
}
