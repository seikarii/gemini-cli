/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { GeminiClient } from '../core/client.js';
import { EditToolParams } from '../tools/edit.js';
import { FileSystemService } from '../services/fileSystemService.js';
import { countOccurrences } from './stringUtils.js';
export { countOccurrences };
/**
 * Defines the structure of the parameters within CorrectedEditResult
 */
interface CorrectedEditParams {
    file_path: string;
    old_string: string;
    new_string: string;
}
/**
 * Defines the result structure for ensureCorrectEdit.
 */
export interface CorrectedEditResult {
    params: CorrectedEditParams;
    occurrences: number;
}
/**
 * Attempts to correct edit parameters if the original old_string is not found.
 * Uses an AST-first approach with multiple correction strategies before falling back to LLM.
 * Results are cached to avoid redundant processing.
 */
export declare function ensureCorrectEdit(filePath: string, currentContent: string, originalParams: EditToolParams, client: GeminiClient, abortSignal: AbortSignal, fileSystemService?: FileSystemService): Promise<CorrectedEditResult>;
/**
 * Unescapes a string that might have been overly escaped by an LLM.
 */
export declare function unescapeStringForGeminiBug(inputString: string): string;
/**
 * Counts non-overlapping occurrences of substr in str.
 */
export declare function resetEditCorrectorCaches(): void;
/**
 * Ensure file content escaping is corrected when necessary via LLM.
 */
export declare function ensureCorrectFileContent(content: string, client: GeminiClient, abortSignal: AbortSignal): Promise<string>;
