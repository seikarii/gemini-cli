/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Lightweight AST adapter helpers to provide a small stable surface for
 * tools that need to parse and modify TypeScript/JavaScript files.
 */
import { Project, SourceFile } from 'ts-morph';
export interface ParseResult {
    project?: Project;
    sourceFile?: SourceFile | null;
    text?: string;
    error?: string | null;
}
export declare function createProject(): Project;
export declare function parseFileWithProject(project: Project, filePath: string): ParseResult;
export declare function parseSourceToSourceFileWithProject(project: Project, source: string, filePath?: string): ParseResult;
export declare function dumpSourceFileText(sourceFile: SourceFile): string;
declare const _default: {
    createProject: typeof createProject;
    parseFileWithProject: typeof parseFileWithProject;
    parseSourceToSourceFileWithProject: typeof parseSourceToSourceFileWithProject;
    dumpSourceFileText: typeof dumpSourceFileText;
};
export default _default;
