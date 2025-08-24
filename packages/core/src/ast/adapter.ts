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
import * as fs from 'fs';
import * as path from 'path';

export interface ParseResult {
  project?: Project;
  sourceFile?: SourceFile | null;
  text?: string;
  error?: string | null;
}

export function createProject(): Project {
  return new Project({ useInMemoryFileSystem: true });
}

export function parseFileWithProject(project: Project, filePath: string): ParseResult {
  try {
    const original = fs.readFileSync(filePath, 'utf-8');
    const normalizedPath = path.isAbsolute(filePath) ? filePath : path.resolve('/', filePath);
    const sf = project.createSourceFile(normalizedPath, original, { overwrite: true });
    return { project, sourceFile: sf, text: original, error: null };
  } catch (e: unknown) {
    return { error: String(e) };
  }
}

export function parseSourceToSourceFileWithProject(project: Project, source: string, filePath?: string): ParseResult {
  const fp = filePath ?? '/virtual-file.ts';
  try {
    const normalizedPath = path.isAbsolute(fp) ? fp : path.resolve('/', fp);
    const sf = project.createSourceFile(normalizedPath, source, { overwrite: true });
    return { project, sourceFile: sf, text: source, error: null };
  } catch (e: unknown) {
    return { error: String(e) };
  }
}

export function dumpSourceFileText(sourceFile: SourceFile): string {
  return sourceFile.getFullText();
}

export default {
  createProject,
  parseFileWithProject,
  parseSourceToSourceFileWithProject,
  dumpSourceFileText,
};
