/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export interface Intentions {
    functions: FunctionInfo[];
    classes: ClassInfo[];
    imports: ImportInfo[];
    constants: ConstantInfo[];
    exports: ExportInfo[];
    interfaces: InterfaceInfo[];
    types: TypeInfo[];
    enums: EnumInfo[];
    parsingErrors: string[];
    complexity: ComplexityMetrics;
    [key: string]: unknown;
}
export interface FunctionInfo {
    name: string;
    isAsync: boolean;
    isExported: boolean;
    isGenerator: boolean;
    startLine: number | null;
    endLine: number | null;
    params: ParameterInfo[];
    returnType?: string;
    visibility?: 'public' | 'private' | 'protected';
    isStatic?: boolean;
    complexity?: number;
    documentation?: string;
}
export interface ParameterInfo {
    name: string;
    type: string;
    isOptional: boolean;
    hasDefault: boolean;
    defaultValue?: string;
}
export interface ClassInfo {
    name: string;
    isExported: boolean;
    isAbstract: boolean;
    startLine: number | null;
    endLine: number | null;
    methods: MethodInfo[];
    properties: PropertyInfo[];
    constructors: ConstructorInfo[];
    extends?: string;
    implements: string[];
    documentation?: string;
}
export interface MethodInfo {
    name: string;
    isAsync: boolean;
    isStatic: boolean;
    isAbstract: boolean;
    visibility: 'public' | 'private' | 'protected';
    params: ParameterInfo[];
    returnType?: string;
    complexity?: number;
}
export interface PropertyInfo {
    name: string;
    type?: string;
    isStatic: boolean;
    isReadonly: boolean;
    visibility: 'public' | 'private' | 'protected';
    hasInitializer: boolean;
}
export interface ConstructorInfo {
    params: ParameterInfo[];
    visibility: 'public' | 'private' | 'protected';
    complexity?: number;
}
export interface ImportInfo {
    moduleSpecifier: string;
    namedImports: NamedImportInfo[];
    defaultImport?: string;
    namespaceImport?: string;
    isTypeOnly: boolean;
    startLine: number | null;
}
export interface NamedImportInfo {
    name: string;
    alias?: string;
    isTypeOnly: boolean;
}
export interface ConstantInfo {
    name: string;
    value?: string;
    type?: string;
    isExported: boolean;
    startLine: number | null;
    isReadonly: boolean;
    documentation?: string;
}
export interface ExportInfo {
    name?: string;
    isDefault: boolean;
    isReExport: boolean;
    moduleSpecifier?: string;
    type: 'variable' | 'function' | 'class' | 'interface' | 'type' | 'namespace';
}
export interface InterfaceInfo {
    name: string;
    isExported: boolean;
    startLine: number | null;
    endLine: number | null;
    properties: InterfacePropertyInfo[];
    methods: InterfaceMethodInfo[];
    extends: string[];
    documentation?: string;
}
export interface InterfacePropertyInfo {
    name: string;
    type?: string;
    isOptional: boolean;
    isReadonly: boolean;
}
export interface InterfaceMethodInfo {
    name: string;
    params: ParameterInfo[];
    returnType?: string;
    isOptional: boolean;
}
export interface TypeInfo {
    name: string;
    isExported: boolean;
    definition: string;
    startLine: number | null;
    documentation?: string;
}
export interface EnumInfo {
    name: string;
    isExported: boolean;
    isConst: boolean;
    startLine: number | null;
    endLine: number | null;
    members: EnumMemberInfo[];
    documentation?: string;
}
export interface EnumMemberInfo {
    name: string;
    value?: string | number;
    hasInitializer: boolean;
}
export interface ComplexityMetrics {
    cyclomaticComplexity: number;
    cognitiveComplexity: number;
    halsteadMetrics: HalsteadMetrics;
    linesOfCode: number;
    maintainabilityIndex: number;
}
export interface HalsteadMetrics {
    operatorsCount: number;
    operandsCount: number;
    vocabularySize: number;
    programLength: number;
    difficulty: number;
    effort: number;
    timeRequired: number;
    bugsDelivered: number;
}
import { SourceFile } from 'ts-morph';
export interface ParseResult {
    source?: string | null;
    sourceFile?: SourceFile | null;
    parseError?: string | null;
    comments: string[];
    jsdocs: string[];
    intentions: unknown;
    fileInfo: {
        path: string;
        sizeBytes: number;
        lineCount: number;
        processingTimeMs: number;
    };
}
/**
 * Simple defensive file size check.
 */
export declare function checkFileSize(filePath: string): Promise<{
    ok: boolean;
    error?: string;
}>;
/**
 * Read file with encoding fallbacks (utf-8, latin1).
 */
export declare function readFileWithEncodingFallback(filePath: string): Promise<{
    content: string | null;
    error: string | null;
}>;
/**
 * Compatibility wrapper for parseSourceToSourceFile.
 * Accepts either (source, filePath) or the older callers that pass (filePath, source).
 */
export declare function parseSourceToSourceFile(a: string, b?: string): {
    sourceFile?: SourceFile | null;
    error?: string | null;
};
/**
 * Extract comments (line comments and block comments) and JSDoc blocks using regex + ts-morph where possible.
 */
export declare function extractCommentsAndJsDoc(source: string, sourceFile?: SourceFile | null): {
    comments: string[];
    jsdocs: string[];
};
/**
 * Extract intentions (functions, classes, imports, constants) from a ts-morph SourceFile.
 * Defensive: isolate node-level errors to avoid complete failure.
 */
export declare function extractIntentionsFromSourceFile(sourceFile?: SourceFile | null): Intentions;
/**
 * Main orchestration: read file, parse, extract comments and intentions.
 */
export declare function readAndParseFile(filePath: string): Promise<ParseResult>;
/**
 * Minimal ASTReader class matching the Python tool shape.
 * Adjust to your tool framework (BaseTool) if needed.
 */
export interface ASTReaderParams {
    filePath: string;
    includeSource?: boolean;
    extractIntentions?: boolean;
    extractComments?: boolean;
}
export declare class ASTReader {
    name: string;
    description: string;
    execute(params: ASTReaderParams): Promise<{
        success: boolean;
        output: string;
        metadata?: Partial<ParseResult>;
        errorMessage?: string;
        executionTime?: number;
    }>;
}
