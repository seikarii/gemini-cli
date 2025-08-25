/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal shims for heavy external libs used by the two new files.


/* Minimal shims for isolated type-check of new tools. These are intentionally
   permissive and only include the members referenced by the new files so we
   can run a localized tsc without pulling heavy deps. */

declare module 'ts-morph' {
  export interface ProjectOptions { useInMemoryFileSystem?: boolean; compilerOptions?: any }
  export class Project {
    constructor(opts?: ProjectOptions);
    createSourceFile(path: string, text: string, opts?: any): SourceFile;
  }

  export class SourceFile {
    getText(): string;
    getFullText(): string;
    getFunctions(): any[];
    getClasses(): any[];
    getVariableStatements(): any[];
    getInterfaces(): any[];
    getTypeAliases(): any[];
    getImportDeclarations(): any[];
    getExportDeclarations(): any[];
    getStatements(): any[];
    getChildren(): any[];
    insertStatements(index: number, text: string): void;
    addStatements(text: string): void;
    insertText(pos: number, text: string): void;
    getDescendants(): any[];
    forEachDescendant(cb: (n: any) => void): void;
    getStartLineNumber(): number;
    getEndLineNumber(): number;
    getTextRange?(): { getStart(): number; getEnd(): number };
  }

  export type Node = any;
  // Provide Node as a runtime value as some code calls Node.isXyz
  export const Node: any;
  export const SyntaxKind: any;
  export const Statement: any;
  export type ClassDeclaration = any;
  export const ClassDeclaration: any;
}

declare module 'diff' {
  export interface PatchOptions { context?: number; ignoreWhitespace?: boolean }
  export interface Hunk { newStart: number; oldStart: number; lines: string[] }
  export interface ParsedDiff { hunks: Hunk[] }
  export function createPatch(fileName: string, oldStr: string, newStr: string, oldHeader?: string, newHeader?: string, options?: PatchOptions): string;
  export function structuredPatch(oldFile: string, newFile: string, oldStr: string, newStr: string, oldHeader?: string, newHeader?: string, options?: PatchOptions): ParsedDiff;
  const d: any;
  export default d;
}

declare module 'fdir' {
  export class fdir {
    constructor();
    withRelativePaths(): this;
    exclude(cb: (p: string, dirPath?: string) => boolean): this;
    glob(pattern: string): this;
    walk(root: string): Promise<string[]>;
  }
}

// Local project AST modules used by the new tools. Provide minimal signatures.
declare module '../ast/parser.js' {
  export function parseSourceToSourceFile(filePath: string, text: string): { sourceFile?: any } | undefined;
  export function extractIntentionsFromSourceFile(sourceFile: any): Record<string, unknown>;
}

declare module '../ast/adapter.js' {
  export function createProject(): any;
  export function parseFileWithProject(project: any, filePath: string): { sourceFile?: any; text?: string; error?: string };
  export function dumpSourceFileText(sourceFile: any): string;
}

declare module '../ast/models.js' {
  export type ASTQuery = any;
  export type DictionaryQuery = any;
  export type ModificationSpec = any;
  export enum ModificationOperation { REPLACE = 'REPLACE' }
}

// Minimal node module shims to satisfy imports used in these files
declare module 'fs';
declare module 'path';

// Local tooling shims used by the two new tool files
declare module './tools.js' {
  export type ToolResult = { llmContent?: string; returnDisplay?: any };
  export type ToolResultDisplay = any;
  export type ToolInvocation<P = any, R = ToolResult> = any;
  export enum Kind { Read = 'read', Edit = 'edit', Other = 'other' }

  export class BaseToolInvocation<P = any, R = ToolResult> {
    constructor(params?: P);
    protected params: P;
    getDescription(): string;
    execute(signal?: AbortSignal): Promise<R>;
  }

  export class BaseDeclarativeTool<P = any, R = ToolResult> {
    constructor(name: string, id: string, description: string, kind: Kind, schema?: any);
  }

  export {};
}

declare module '../config/config.js' {
  export class Config {
    getFileSystemService(): { readTextFile(path: string): Promise<{ success: boolean; data?: string; error?: string }>; writeTextFile(path: string, text: string): Promise<{ success: boolean; error?: string }>; };
  }
}

declare module '../ast/modifier.js' {
  export class ASTModifier {
    constructor();
    applyModifications(text: string, mods: any[], opts?: any): Promise<{ success: boolean; modifiedText?: string; backupId?: string; error?: string }>;
  }
}

declare module '../ast/finder.js' {
  export function findNodes(sourceFile: any, query: any): any[];
}

declare module './diffOptions.js' {
  export const DEFAULT_DIFF_OPTIONS: any;
  export function getDiffStat(fileName: string, original: string, content: string, modified: string): any;
}
