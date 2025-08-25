/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { SourceFile } from 'ts-morph';
/**
 * Result shape returned by readAndParseFile / ASTReader.execute
 */
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
export declare function extractIntentionsFromSourceFile(sourceFile?: SourceFile | null): any;
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
        metadata?: any;
        errorMessage?: string;
        executionTime?: number;
    }>;
}
