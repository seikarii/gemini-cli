/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export interface LoadIgnoreRulesOptions {
    projectRoot: string;
    useGitignore: boolean;
    useGeminiignore: boolean;
    ignoreDirs: string[];
}
export declare function loadIgnoreRules(options: LoadIgnoreRulesOptions): Ignore;
/**
 * Async variant of `loadIgnoreRules` which performs non-blocking file I/O.
 * This is intended for use in hot-path initialization where blocking the
 * event loop is undesirable. The existing synchronous `loadIgnoreRules`
 * is preserved for tests and callers that need a sync API.
 */
export declare function loadIgnoreRulesAsync(options: LoadIgnoreRulesOptions): Promise<Ignore>;
export declare class Ignore {
    private readonly allPatterns;
    private dirIgnorer;
    private fileIgnorer;
    /**
     * Adds one or more ignore patterns.
     * @param patterns A single pattern string or an array of pattern strings.
     *                 Each pattern can be a glob-like string similar to .gitignore rules.
     * @returns The `Ignore` instance for chaining.
     */
    add(patterns: string | string[]): this;
    /**
     * Returns a predicate that matches explicit directory ignore patterns (patterns ending with '/').
     * @returns {(dirPath: string) => boolean}
     */
    getDirectoryFilter(): (dirPath: string) => boolean;
    /**
     * Returns a predicate that matches file ignore patterns (all patterns not ending with '/').
     * Note: This may also match directories if a file pattern matches a directory name, but all explicit directory patterns are handled by getDirectoryFilter.
     * @returns {(filePath: string) => boolean}
     */
    getFileFilter(): (filePath: string) => boolean;
    /**
     * Returns a string representing the current set of ignore patterns.
     * This can be used to generate a unique identifier for the ignore configuration,
     * useful for caching purposes.
     * @returns A string fingerprint of the ignore patterns.
     */
    getFingerprint(): string;
}
