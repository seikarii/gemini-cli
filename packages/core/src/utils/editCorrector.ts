/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GenerateContentConfig } from '@google/genai';
import { GeminiClient } from '../core/client.js';
import { EditToolParams, EditTool } from '../tools/edit.js';
import { WriteFileTool } from '../tools/write-file.js';
import { ReadFileTool } from '../tools/read-file.js';
import { ReadManyFilesTool } from '../tools/read-many-files.js';
import { GrepTool } from '../tools/grep.js';
import { LruCache } from './LruCache.js';
import { DEFAULT_GEMINI_FLASH_LITE_MODEL } from '../config/models.js';
import {
  isFunctionResponse,
  isFunctionCall,
} from '../utils/messageInspectors.js';
import * as fs from 'fs';
import { parseSourceToSourceFile } from '../ast/parser.js';
import { findNodes } from '../ast/finder.js';
import { FileSystemService } from '../services/fileSystemService.js';
import {
  normalizeWhitespace,
  countOccurrences,
  calculateStringSimilarity,
  escapeRegExp,
} from './stringUtils.js';

// Re-export certain utils for backwards compatibility with existing imports/tests
export { countOccurrences };

const _EditModel = DEFAULT_GEMINI_FLASH_LITE_MODEL; // Kept for backward compatibility
const EditConfig: GenerateContentConfig = {
  thinkingConfig: {
    thinkingBudget: 0,
  },
};

/**
 * Determines the appropriate model for edit correction based on operation complexity
 */
function getEditCorrectionModel(operation: string, content: string): string {
  // For complex corrections involving large contexts or analysis, use Pro model
  if (content.length > 10000 || operation.includes('analysis') || operation.includes('complex')) {
    return 'gemini-2.5-pro';
  }

  // For simple corrections, use Flash-Lite for speed
  return DEFAULT_GEMINI_FLASH_LITE_MODEL;
}

const MAX_CACHE_SIZE = 50;

// Cache for ensureCorrectEdit results
const editCorrectionCache = new LruCache<string, CorrectedEditResult>(
  MAX_CACHE_SIZE,
);

// Cache for ensureCorrectFileContent results
const fileContentCorrectionCache = new LruCache<string, string>(MAX_CACHE_SIZE);

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
 * Extracts the timestamp from the .id value, which is in format
 * <tool.name>-<timestamp>-<uuid>
 * @param fcnId the ID value of a functionCall or functionResponse object
 * @returns -1 if the timestamp could not be extracted, else the timestamp (as a number)
 */
function getTimestampFromFunctionId(fcnId: string): number {
  const idParts = fcnId.split('-');
  if (idParts.length > 2) {
    const timestamp = parseInt(idParts[1], 10);
    if (!isNaN(timestamp)) {
      return timestamp;
    }
  }
  return -1;
}

/**
 * Will look through the gemini client history and determine when the most recent
 * edit to a target file occurred. If no edit happened, it will return -1
 * @param filePath the path to the file
 * @param client the geminiClient, so that we can get the history
 * @returns a DateTime (as a number) of when the last edit occurred, or -1 if no edit was found.
 */
async function findLastEditTimestamp(
  filePath: string,
  client: GeminiClient,
): Promise<number> {
  const history = (await client.getHistory()) ?? [];

  // Tools that may reference the file path in their FunctionResponse `output`.
  const toolsInResp = new Set([
    WriteFileTool.Name,
    EditTool.Name,
    ReadManyFilesTool.Name,
    GrepTool.Name,
  ]);
  // Tools that may reference the file path in their FunctionCall `args`.
  const toolsInCall = new Set([...toolsInResp, ReadFileTool.Name]);

  // Iterate backwards to find the most recent relevant action.
  for (const entry of history.slice().reverse()) {
    if (!entry.parts) continue;

    for (const part of entry.parts) {
      let id: string | undefined;
      let content: unknown;

      // Check for a relevant FunctionCall with the file path in its arguments.
      if (
        isFunctionCall(entry) &&
        part.functionCall?.name &&
        toolsInCall.has(part.functionCall.name)
      ) {
        id = part.functionCall.id;
        content = part.functionCall.args;
      }
      // Check for a relevant FunctionResponse with the file path in its output.
      else if (
        isFunctionResponse(entry) &&
        part.functionResponse?.name &&
        toolsInResp.has(part.functionResponse.name)
      ) {
        const { response } = part.functionResponse;
        if (response && !('error' in response) && 'output' in response) {
          id = part.functionResponse.id;
          content = response['output'];
        }
      }

      if (!id || content === undefined) continue;

      // Use the "blunt hammer" approach to find the file path in the content.
      // Note that the tool response data is inconsistent in their formatting
      // with successes and errors - so, we just check for the existence
      // as the best guess to if error/failed occurred with the response.
      const stringified = JSON.stringify(content);
      if (
        !stringified.includes('Error') && // only applicable for functionResponse
        !stringified.includes('Failed') && // only applicable for functionResponse
        stringified.includes(filePath)
      ) {
        return getTimestampFromFunctionId(id);
      }
    }
  }

  return -1;
}

/**
 * Attempts to correct edit parameters if the original old_string is not found.
 * Uses an AST-first approach with multiple correction strategies before falling back to LLM.
 * Results are cached to avoid redundant processing.
 */
export async function ensureCorrectEdit(
  filePath: string,
  currentContent: string,
  originalParams: EditToolParams,
  client: GeminiClient,
  abortSignal: AbortSignal,
  fileSystemService?: FileSystemService,
): Promise<CorrectedEditResult> {
  const cacheKey = `${currentContent}---${originalParams.old_string}---${originalParams.new_string}`;
  const cachedResult = editCorrectionCache.get(cacheKey);
  if (cachedResult) {
    return cachedResult;
  }

  const finalNewString = originalParams.new_string ?? '';
  const newStringPotentiallyEscaped =
    unescapeStringForGeminiBug(finalNewString) !== finalNewString;

  const expectedReplacements = originalParams.expected_replacements ?? 1;
  let finalOldString = originalParams.old_string ?? '';
  let occurrences = countOccurrences(currentContent, finalOldString);

  // Early return if we already have the expected number of occurrences
  if (occurrences === expectedReplacements) {
    return await finalizeResult(
      originalParams,
      finalOldString,
      finalNewString,
      occurrences,
      newStringPotentiallyEscaped,
      client,
      abortSignal,
      cacheKey,
    );
  }

  // AST-first correction with multiple strategies
  // If the caller requested a specific occurrence (e.g. target_occurrence = 7)
  // we MUST NOT apply AST-based corrections that can replace the original
  // `old_string` with a large AST node text. Doing so collapses multiple
  // simple repeated occurrences into a single match and breaks targeted
  // replacements. In that case skip AST correction and rely on string-based
  // heuristics (unescaping/trimming) which are safe for occurrence-aware ops.
  const callerRequestedSpecificOccurrence =
    originalParams &&
    (originalParams as EditToolParams).target_occurrence !== undefined;

  if (!callerRequestedSpecificOccurrence) {
    const astCorrectionResult = await tryASTCorrection(
      currentContent,
      filePath,
      finalOldString,
      expectedReplacements,
    );

    if (astCorrectionResult.success) {
      if (
        astCorrectionResult.correctedOldString !== undefined &&
        astCorrectionResult.occurrences !== undefined
      ) {
        // correctedOldString and occurrences are optional on the result type;
        // narrow and cast to satisfy the compiler that they're present.
        finalOldString = astCorrectionResult.correctedOldString as string;
        occurrences = astCorrectionResult.occurrences as number;
      }

      return await finalizeResult(
        originalParams,
        finalOldString,
        finalNewString,
        occurrences,
        newStringPotentiallyEscaped,
        client,
        abortSignal,
        cacheKey,
      );
    }
  }

  // String-based corrections (unescaping, trimming)
  const stringCorrectionResult = tryStringBasedCorrections(
    currentContent,
    finalOldString,
    expectedReplacements,
  );

  if (stringCorrectionResult.success) {
    if (
      stringCorrectionResult.correctedOldString !== undefined &&
      stringCorrectionResult.occurrences !== undefined
    ) {
      // same non-null assertion as above for string-based corrections
      finalOldString = stringCorrectionResult.correctedOldString as string;
      occurrences = stringCorrectionResult.occurrences as number;
    }

    return await finalizeResult(
      originalParams,
      finalOldString,
      finalNewString,
      occurrences,
      newStringPotentiallyEscaped,
      client,
      abortSignal,
      cacheKey,
    );
  }

  // Handle cases with too many occurrences
  if (occurrences > expectedReplacements) {
    return createResult(
      originalParams,
      finalOldString,
      finalNewString,
      occurrences,
      cacheKey,
    );
  }

  // Fallback to LLM-based correction
  return await tryLLMCorrection(
    filePath,
    currentContent,
    originalParams,
    finalOldString,
    finalNewString,
    expectedReplacements,
    newStringPotentiallyEscaped,
    client,
    abortSignal,
    cacheKey,
    fileSystemService,
  );
}

/**
 * Comprehensive AST-based correction using multiple strategies
 */
async function tryASTCorrection(
  currentContent: string,
  filePath: string,
  oldString: string,
  expectedReplacements: number,
): Promise<{
  success: boolean;
  correctedOldString?: string;
  occurrences?: number;
}> {
  try {
    const { sourceFile, error: parseErr } = parseSourceToSourceFile(
      currentContent,
      filePath ?? '/virtual-file.ts',
    );

    if (!sourceFile || parseErr) {
      return { success: false };
    }

    // Strategy 1: Direct text matching with node specificity
    const directMatchResult = tryDirectNodeMatching(
      sourceFile,
      oldString,
      expectedReplacements,
      currentContent,
    );
    if (directMatchResult.success) {
      return directMatchResult;
    }

    // Strategy 2: Unescaped string matching
    const unescapedOldString = unescapeStringForGeminiBug(oldString);
    if (unescapedOldString !== oldString) {
      const unescapedMatchResult = tryDirectNodeMatching(
        sourceFile,
        unescapedOldString,
        expectedReplacements,
        currentContent,
      );
      if (unescapedMatchResult.success) {
        return unescapedMatchResult;
      }
    }

    // Strategy 3: Semantic node matching for common code patterns
    const semanticResult = trySemanticNodeMatching(
      sourceFile,
      oldString,
      expectedReplacements,
      currentContent,
    );
    if (semanticResult.success) {
      return semanticResult;
    }

    // Strategy 4: Fuzzy matching with whitespace normalization
    const fuzzyResult = tryFuzzyNodeMatching(
      sourceFile,
      oldString,
      expectedReplacements,
      currentContent,
    );
    if (fuzzyResult.success) {
      return fuzzyResult;
    }

    return { success: false };
  } catch (_error) {
    console.debug('AST correction failed:', _error);
    return { success: false };
  }
}

/**
 * Strategy 1: Direct node text matching with specificity ranking
 */
function tryDirectNodeMatching(
  sourceFile: any,
  searchString: string,
  expectedReplacements: number,
  currentContent: string,
): { success: boolean; correctedOldString?: string; occurrences?: number } {
  const candidates: Array<{ node: any; text: string; specificity: number }> =
    [];

  for (const node of sourceFile.getDescendants()) {
    try {
      const nodeText = node.getText();
      if (nodeText && nodeText.includes(searchString)) {
        // Calculate specificity score (prefer smaller, more specific nodes)
        const specificity = calculateNodeSpecificity(
          node,
          searchString,
          nodeText,
        );
        candidates.push({ node, text: nodeText, specificity });
      }
    } catch {
      // Ignore problematic nodes
    }
  }

  if (candidates.length === 0) {
    return { success: false };
  }

  // Sort by specificity (higher score = more specific)
  candidates.sort((a, b) => b.specificity - a.specificity);

  // Try candidates in order of specificity
  for (const candidate of candidates) {
    const occurrences = countOccurrences(currentContent, candidate.text);
    if (occurrences === expectedReplacements) {
      return {
        success: true,
        correctedOldString: candidate.text,
        occurrences,
      };
    }
  }

  return { success: false };
}

/**
 * Strategy 3: Semantic node matching for common code patterns
 */
function trySemanticNodeMatching(
  sourceFile: any,
  oldString: string,
  expectedReplacements: number,
  currentContent: string,
): { success: boolean; correctedOldString?: string; occurrences?: number } {
  try {
    // Look for specific node types that commonly contain the target string
    const semanticQueries = [
      '//FunctionDeclaration',
      '//MethodDeclaration',
      '//VariableStatement',
      '//ClassDeclaration',
      '//PropertyAssignment',
      '//CallExpression',
      '//StringLiteral',
      '//TemplateExpression',
    ];

    for (const query of semanticQueries) {
      try {
        const nodes = findNodes(sourceFile, query);
        for (const node of nodes) {
          try {
            const nodeText = node.getText();
            if (nodeText && nodeText.includes(oldString)) {
              const occurrences = countOccurrences(currentContent, nodeText);
              if (occurrences === expectedReplacements) {
                return {
                  success: true,
                  correctedOldString: nodeText,
                  occurrences,
                };
              }
            }
          } catch {
            // Continue with next node
          }
        }
      } catch {
        // Continue with next query
      }
    }

    return { success: false };
  } catch {
    return { success: false };
  }
}

/**
 * Strategy 4: Fuzzy matching with whitespace and formatting normalization
 */
function tryFuzzyNodeMatching(
  sourceFile: any,
  oldString: string,
  expectedReplacements: number,
  currentContent: string,
): { success: boolean; correctedOldString?: string; occurrences?: number } {
  const normalizedTarget = normalizeWhitespace(oldString);
  const candidates: Array<{ text: string; similarity: number }> = [];

  for (const node of sourceFile.getDescendants()) {
    try {
      const nodeText = node.getText();
      if (!nodeText) continue;

      const normalizedNodeText = normalizeWhitespace(nodeText);

      // Check if normalized versions match or have high similarity
      if (normalizedNodeText.includes(normalizedTarget)) {
        const similarity = calculateStringSimilarity(
          normalizedTarget,
          normalizedNodeText,
        );
        if (similarity > 0.8) {
          // 80% similarity threshold
          candidates.push({ text: nodeText, similarity });
        }
      }
    } catch {
      // Ignore problematic nodes
    }
  }

  // Sort by similarity (highest first)
  candidates.sort((a, b) => b.similarity - a.similarity);

  for (const candidate of candidates) {
    const occurrences = countOccurrences(currentContent, candidate.text);
    if (occurrences === expectedReplacements) {
      return {
        success: true,
        correctedOldString: candidate.text,
        occurrences,
      };
    }
  }

  return { success: false };
}

/**
 * Calculate node specificity score for ranking candidates
 */
function calculateNodeSpecificity(
  node: any,
  searchString: string,
  nodeText: string,
): number {
  let score = 0;

  // Prefer smaller nodes (inverse relationship with length)
  score += Math.max(0, 1000 - nodeText.length);

  // Prefer nodes where the search string is a larger portion of the content
  const searchRatio = searchString.length / nodeText.length;
  score += searchRatio * 500;

  // Prefer certain node types
  try {
    const kindName = node.getKindName ? node.getKindName() : '';
    const preferredTypes = [
      'VariableStatement',
      'FunctionDeclaration',
      'MethodDeclaration',
      'PropertyAssignment',
      'StringLiteral',
      'CallExpression',
    ];
    if (preferredTypes.includes(kindName)) {
      score += 200;
    }
  } catch {
    // Ignore if we can't get kind name
  }

  // Prefer exact matches at word boundaries
  const wordBoundaryRegex = new RegExp(`\\b${escapeRegExp(searchString)}\\b`);
  if (wordBoundaryRegex.test(nodeText)) {
    score += 300;
  }

  return score;
}

/**
 * String-based correction strategies
 */
function tryStringBasedCorrections(
  currentContent: string,
  oldString: string,
  expectedReplacements: number,
): { success: boolean; correctedOldString?: string; occurrences?: number } {
  const strategies = [
    // Unescaping
    () => unescapeStringForGeminiBug(oldString),
    // Trimming
    () => oldString.trim(),
    // Normalize line endings
    () => oldString.replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
    // Remove extra spaces
    () => oldString.replace(/\s+/g, ' '),
    // Normalize quotes
    () => oldString.replace(/[""]/g, '"').replace(/['']/g, "'"),
  ];

  for (const strategy of strategies) {
    try {
      const corrected = strategy();
      if (corrected !== oldString) {
        const occurrences = countOccurrences(currentContent, corrected);
        if (occurrences === expectedReplacements) {
          return {
            success: true,
            correctedOldString: corrected,
            occurrences,
          };
        }
      }
    } catch {
      // Continue with next strategy
    }
  }

  return { success: false };
}

/**
 * Fallback to LLM-based correction (existing logic)
 */
async function tryLLMCorrection(
  filePath: string,
  currentContent: string,
  originalParams: EditToolParams,
  finalOldString: string,
  finalNewString: string,
  expectedReplacements: number,
  newStringPotentiallyEscaped: boolean,
  client: GeminiClient,
  abortSignal: AbortSignal,
  cacheKey: string,
  fileSystemService?: FileSystemService,
): Promise<CorrectedEditResult> {
  let occurrences = countOccurrences(currentContent, finalOldString);

  if (occurrences === 0) {
    // Check for external file modifications
    if (filePath) {
      const lastEditedByUsTime = await findLastEditTimestamp(filePath, client);
      if (lastEditedByUsTime > 0) {
        let fileMtimeMs: number;

        if (fileSystemService) {
          // Use FileSystemService for standardized file operations
          const fileInfo = await fileSystemService.getFileInfo(filePath);
          if (!fileInfo.success) {
            console.warn(
              `Failed to get file info for ${filePath}: ${fileInfo.error}`,
            );
            fileMtimeMs = 0;
          } else {
            fileMtimeMs = fileInfo.data?.mtimeMs || 0;
          }
        } else {
          // Fallback to direct fs.statSync for backward compatibility
          const stats = fs.statSync(filePath);
          fileMtimeMs = stats.mtimeMs;
        }

        const diff = fileMtimeMs - lastEditedByUsTime;
        if (diff > 2000) {
          return createResult(
            originalParams,
            finalOldString,
            finalNewString,
            0,
            cacheKey,
          );
        }
      }
    }

    // Try LLM correction
    const unescapedOldString = unescapeStringForGeminiBug(finalOldString);
    // Ask LLM once for a corrected old string. Tests expect a single LLM call
    // in the 'no match' case rather than multiple retries.
    const llmCorrectedOldString = await correctOldStringMismatch(
      client,
      currentContent,
      unescapedOldString,
      abortSignal,
    );
    const llmOldOccurrences = countOccurrences(
      currentContent,
      llmCorrectedOldString,
    );

    if (llmOldOccurrences === expectedReplacements) {
      finalOldString = llmCorrectedOldString;
      occurrences = llmOldOccurrences;
      if (newStringPotentiallyEscaped) {
        // Ask the LLM once to adjust the replacement text to the corrected old
        // string. Pass the ORIGINAL requested old_string so the LLM can reason
        // about how the replacement should change relative to the original.
        const correctedNewString = await correctNewString(
          client,
          // original_old_string should be the value originally requested
          // (not the already-corrected one).
          originalParams.old_string ?? finalOldString,
          llmCorrectedOldString,
          unescapeStringForGeminiBug(finalNewString),
          abortSignal,
        );
        if (
          typeof correctedNewString === 'string' &&
          correctedNewString.length > 0
        ) {
          finalNewString = correctedNewString;
        }
      }
    } else {
      return createResult(
        originalParams,
        finalOldString,
        finalNewString,
        0,
        cacheKey,
      );
    }
  }

  // Apply trimming optimization
  const { targetString, pair } = trimPairIfPossible(
    finalOldString,
    finalNewString,
    currentContent,
    expectedReplacements,
  );

  return createResult(
    originalParams,
    targetString,
    pair,
    countOccurrences(currentContent, targetString),
    cacheKey,
  );
}

/**
 * Finalize result with potential new string correction
 */
async function finalizeResult(
  originalParams: EditToolParams,
  finalOldString: string,
  finalNewString: string,
  occurrences: number,
  newStringPotentiallyEscaped: boolean,
  client: GeminiClient,
  abortSignal: AbortSignal,
  cacheKey: string,
): Promise<CorrectedEditResult> {
  if (newStringPotentiallyEscaped) {
    const maybeCorrected = await correctNewStringEscaping(
      client,
      finalOldString,
      finalNewString,
      abortSignal,
    );
    if (typeof maybeCorrected === 'string' && maybeCorrected.length > 0) {
      finalNewString = maybeCorrected;
    }
  }

  return createResult(
    originalParams,
    finalOldString,
    finalNewString,
    occurrences,
    cacheKey,
  );
}

/**
 * Create and cache the final result
 */
function createResult(
  originalParams: EditToolParams,
  finalOldString: string,
  finalNewString: string,
  occurrences: number,
  cacheKey: string,
): CorrectedEditResult {
  // Helper to choose a non-empty string preference order:
  // 1) finalNewString (result of correction) if non-empty
  // 2) originalParams.new_string (what was requested) if non-empty
  // 3) fallback to empty string (explicit)
  function chooseNonEmpty(
    candidate: string | undefined,
    fallback?: string | undefined,
  ): string {
    if (typeof candidate === 'string' && candidate.length > 0) return candidate;
    if (typeof fallback === 'string' && fallback.length > 0) return fallback;
    return '';
  }

  const safeNewString = chooseNonEmpty(
    finalNewString,
    originalParams.new_string as string | undefined,
  );
  const safeOldString =
    typeof finalOldString === 'string'
      ? finalOldString
      : (originalParams.old_string ?? '');

  // Defensive guard: never return or cache an empty old_string because
  // performing a replacement with an empty target can be destructive
  // (it may match everywhere or be misinterpreted by callers). If we
  // detect an empty old_string, return a result with occurrences=0
  // and do not cache it so callers will avoid applying the replace.
  if (!safeOldString || safeOldString.length === 0) {
    try {
      console.warn(
        `editCorrector: blocked empty old_string for file ${originalParams.file_path}; returning occurrences=0 to avoid destructive replacements.`,
      );
    } catch {
      /* ignore logging failures */
    }

    const fallbackResult: CorrectedEditResult = {
      params: {
        file_path: originalParams.file_path,
        old_string: originalParams.old_string ?? '',
        new_string: safeNewString,
      },
      occurrences: 0,
    };

    // Do not cache intentionally.
    return fallbackResult;
  }
  // If we had to fallback to the original param because the correction
  // returned empty, log a warning in debug modes to aid troubleshooting.
  try {
    if (
      (finalNewString === undefined || finalNewString === '') &&
      originalParams.new_string
    ) {
      // Avoid noisy logging in production; rely on consumers to enable debug
      // mode if they want detailed telemetry. Use console.warn for visibility.
      console.warn(
        `editCorrector: correction produced empty new_string for file ${originalParams.file_path}; falling back to original requested new_string.`,
      );
    }
  } catch {
    /* ignore logging failures */
  }

  const result: CorrectedEditResult = {
    params: {
      file_path: originalParams.file_path,
      old_string: safeOldString,
      new_string: safeNewString,
    },
    occurrences,
  };

  // Cache the safe result
  editCorrectionCache.set(cacheKey, result);
  return result;
}

// Utility functions

// normalizeWhitespace is provided by ./stringUtils.ts

// normalizeWhitespace is provided by ./stringUtils.ts

/**
 * Unescapes a string that might have been overly escaped by an LLM.
 */
function escapeBackticks(input: string): string {
  return input.replace(/```/g, "'''"); // Replace triple backticks with triple single quotes
}

export function unescapeStringForGeminiBug(inputString: string): string {
  return inputString.replace(
    /\\+(n|t|r|'|"|`|\\|\n)/g,
    (match, capturedChar) => {
      switch (capturedChar) {
        case 'n':
          return '\n';
        case 't':
          return '\t';
        case 'r':
          return '\r';
        case "'":
          return "'";
        case '"':
          return '"';
        case '`':
          return '`';
        case '\\':
          return '\\';
        case '\n':
          return '\n';
        default:
          return match;
      }
    },
  );
}

/**
 * Counts non-overlapping occurrences of substr in str.
 */
// countOccurrences and normalizeWhitespace are provided by ./stringUtils.ts
export function resetEditCorrectorCaches() {
  try {
    editCorrectionCache.clear();
  } catch (e) {
    console.debug('resetEditCorrectorCaches: ignore', e);
  }
  try {
    fileContentCorrectionCache.clear();
  } catch (e) {
    console.debug('resetEditCorrectorCaches: ignore', e);
  }
}

// Schemas for LLM generateJson calls
const OLD_STRING_CORRECTION_SCHEMA = {
  type: 'object',
  properties: {
    corrected_target_snippet: {
      type: 'string',
      description:
        'The corrected version of the target snippet that exactly and uniquely matches a segment within the provided file content.',
    },
  },
  required: ['corrected_target_snippet'],
};

const NEW_STRING_CORRECTION_SCHEMA = {
  type: 'object',
  properties: {
    corrected_new_string: {
      type: 'string',
      description:
        'The original_new_string adjusted to be a suitable replacement for the corrected_old_string, while maintaining the original intent of the change.',
    },
  },
  required: ['corrected_new_string'],
};

const CORRECT_NEW_STRING_ESCAPING_SCHEMA = {
  type: 'object',
  properties: {
    corrected_new_string_escaping: {
      type: 'string',
      description:
        'The new_string with corrected escaping, ensuring it is a proper replacement for the old_string, especially considering potential over-escaping issues from previous LLM generations.',
    },
  },
  required: ['corrected_new_string_escaping'],
};

const CORRECT_STRING_ESCAPING_SCHEMA = {
  type: 'object',
  properties: {
    corrected_string_escaping: {
      type: 'string',
      description:
        'The string with corrected escaping, ensuring it is valid, specially considering potential over-escaping issues from previous LLM generations.',
    },
  },
  required: ['corrected_string_escaping'],
};

/**
 * Try to correct an old_string that did not match by asking the LLM for the
 * likely segment in the file content that the snippet intended to match.
 */
async function correctOldStringMismatch(
  geminiClient: GeminiClient,
  fileContent: string,
  problematicSnippet: string,
  abortSignal: AbortSignal,
): Promise<string> {
  const prompt = `
# Edit Correction Assistant - Old String Mismatch

## Context
A text replacement operation failed because the target snippet could not be found exactly in the file. This commonly occurs due to:
- Extra escape characters in the snippet
- Minor whitespace or formatting differences
- Case sensitivity issues
- Hidden characters or encoding problems

## System Capabilities
You have access to powerful tools for file analysis and editing. Use the dual-context strategy:
- **Short-term context (current session)**: For immediate tool execution and analysis
- **Long-term context (knowledge base)**: For comprehensive understanding and pattern recognition

### Available Tools:
- **ReadFileTool**: Examine file contents with precise line ranges
  - *Use when*: Need specific file sections or syntax verification
  - *Failure recovery*: Try broader line ranges if initial read fails
- **GrepTool**: Search for patterns across files using regex
  - *Use when*: Need to find similar patterns or code structures
  - *Failure recovery*: Simplify regex patterns, use plain text search
- **ReadManyFilesTool**: Analyze multiple files simultaneously
  - *Use when*: Need to understand cross-file relationships
  - *Failure recovery*: Reduce file count, focus on most relevant files
- **EditTool**: Perform precise text replacements with validation
  - *Use when*: Ready to make the corrected replacement
  - *Failure recovery*: Break down complex edits into smaller operations

### Tool Selection Strategy:
1. **Start with ReadFileTool** for focused analysis
2. **Use GrepTool** for pattern discovery when text matching fails
3. **Apply ReadManyFilesTool** when context spans multiple files
4. **Execute EditTool** only after validation with other tools

## Task
Analyze the provided file content and problematic target snippet. Identify the exact segment in the file that the snippet was intended to match.

**Problematic target snippet:**
\`\`\`
${problematicSnippet}
\`\`\`

**File Content:**
\`\`\`
${fileContent}
\`\`\`

## Analysis Steps
1. **Compare character-by-character** between snippet and file content
2. **Identify discrepancies** in escaping, whitespace, or formatting
3. **Locate the most similar segment** in the file content
4. **Verify uniqueness** of the match

## Output Requirements
Return ONLY a JSON object with the key 'corrected_target_snippet' containing the exact literal text from the file that should be replaced. If no clear match exists, return an empty string.

## Error Recovery Strategy
If the snippet cannot be matched:
- Consider using GrepTool to search for similar patterns
- Use ReadFileTool to examine the exact area of interest
- Check for encoding or invisible character issues
`.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      OLD_STRING_CORRECTION_SCHEMA,
      abortSignal,
      getEditCorrectionModel('old_string_correction', fileContent),
      EditConfig,
    )) as Record<string, unknown> | undefined;
    if (result) {
      const corrected = result['corrected_target_snippet'];
      if (typeof corrected === 'string' && corrected.length > 0)
        return corrected;
    }
    return problematicSnippet;
  } catch (error) {
    if (abortSignal.aborted) throw error;
    console.error(
      'Error during LLM call for old string snippet correction:',
      error,
    );
    return problematicSnippet;
  }
}

async function correctNewString(
  geminiClient: GeminiClient,
  originalOldString: string,
  correctedOldString: string,
  originalNewString: string,
  abortSignal: AbortSignal,
): Promise<string> {
  if (originalOldString === correctedOldString) return originalNewString;

  const prompt = `
# Edit Correction Assistant - New String Adjustment

## Context
A text replacement was planned, but the original target text (original_old_string) didn't match exactly with the actual file content (corrected_old_string). The target has been corrected, and now the replacement text needs adjustment to maintain the original intent.

## System Capabilities
When string corrections are needed, leverage the dual-context strategy and these recovery tools:

### Available Tools (prioritized by failure recovery):
- **ReadFileTool**: Get fresh file content to verify current state
  - *Best for*: Real-time file validation and syntax checking
  - *Recovery*: Use broader line ranges if specific sections fail
- **GrepTool**: Search for related patterns or similar code
  - *Best for*: Finding comparable implementations and patterns
  - *Recovery*: Simplify search terms, use literal strings instead of regex
- **EditTool**: Apply the corrected replacement with validation
  - *Best for*: Precise, validated text replacements
  - *Recovery*: Break complex replacements into smaller, safer operations
- **ReadManyFilesTool**: Check related files for consistency
  - *Best for*: Understanding cross-file dependencies and standards
  - *Recovery*: Focus on most critical files if batch operations fail

### Intelligent Tool Selection:
1. **Context Analysis**: Use ReadFileTool to understand current file state
2. **Pattern Discovery**: Apply GrepTool when unsure about text variations
3. **Consistency Check**: Leverage ReadManyFilesTool for multi-file validation
4. **Safe Execution**: Use EditTool only after thorough validation

## Text Comparison Analysis

**Original intended target (what was planned to be replaced):**
\`\`\`
${originalOldString}
\`\`\`

**Actual target found (what will actually be replaced):**
\`\`\`
${correctedOldString}
\`\`\`

**Original replacement text (needs adjustment):**
\`\`\`
${originalNewString}
\`\`\`

## Task
Based on the differences between the original and corrected target strings, adjust the replacement text to:
1. **Maintain the original transformation intent**
2. **Account for any structural differences**
3. **Preserve code syntax and formatting**
4. **Ensure logical consistency**

## Analysis Guidelines
- Compare the structural differences between original_old_string and corrected_old_string
- Determine how these differences affect the replacement logic
- Adjust the replacement text accordingly while preserving the core change
- Consider edge cases and potential side effects

## Output Requirements
Return ONLY a JSON object with the key 'corrected_new_string' containing the adjusted replacement text. If no adjustment is needed, return the original_new_string.

## Error Recovery Strategy
If the adjustment seems complex:
- Use ReadFileTool to examine the broader context
- Consider using GrepTool to find similar patterns
- Validate the change with a test edit operation
`.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      NEW_STRING_CORRECTION_SCHEMA,
      abortSignal,
      getEditCorrectionModel('new_string_correction', originalNewString),
      EditConfig,
    )) as Record<string, unknown> | undefined;
    if (result) {
      const corrected = result['corrected_new_string'];
      if (typeof corrected === 'string' && corrected.length > 0)
        return corrected;
    }
    return originalNewString;
  } catch (error) {
    if (abortSignal.aborted) throw error;
    console.error('Error during LLM call for new_string correction:', error);
    return originalNewString;
  }
}

async function correctNewStringEscaping(
  geminiClient: GeminiClient,
  oldString: string,
  potentiallyProblematicNewString: string | undefined,
  abortSignal: AbortSignal,
): Promise<string | undefined> {
  const oldStringEscaped = escapeBackticks(oldString);
  let newString = potentiallyProblematicNewString ?? '';
  newString = escapeBackticks(newString);

  const prompt = `
# Edit Correction Assistant - String Escaping Validation

## Context
A text replacement operation is being prepared. The target text (old_string) has been correctly identified, but the replacement text (new_string) may have incorrect escaping that could cause syntax errors or unexpected behavior.

## Common Escaping Issues
- **Over-escaping**: Too many backslashes (\\\\ instead of \\)
- **Under-escaping**: Missing escape characters for special regex/symbols
- **Quote mismatches**: Incorrect quote escaping in strings
- **Path separators**: Incorrect escaping of file paths
- **Regex patterns**: Improper escaping of special regex characters

## System Capabilities for Validation
If escaping issues are detected, utilize these tools with intelligent selection:

### Advanced Validation Tools:
- **ReadFileTool**: Examine the target file's syntax and context
  - *Strategy*: Read surrounding context to understand syntax patterns
  - *Recovery*: If syntax validation fails, read broader context for patterns
- **GrepTool**: Search for similar patterns in the codebase
  - *Strategy*: Find comparable escaping examples across the project
  - *Recovery*: Use simpler search patterns if complex regex fails
- **EditTool**: Test the replacement with validation
  - *Strategy*: Apply escaping corrections with immediate validation
  - *Recovery*: Revert to safer escaping if validation fails
- **FileSystemService**: Check file system compatibility
  - *Strategy*: Validate path separators and special characters
  - *Recovery*: Use platform-agnostic alternatives when needed

### Dual-Context Escaping Strategy:
- **Short-term memory**: Focus on immediate syntax requirements
- **Long-term memory**: Leverage project-wide escaping patterns and conventions

## Text Analysis

**Target text (will be replaced):**
\`\`\`
${oldStringEscaped}
\`\`\`

**Replacement text (potentially problematic):**
\`\`\`
${newString}
\`\`\`

## Task
Analyze the replacement text for escaping issues that could cause:
1. **Syntax errors** in the target language/file type
2. **Runtime errors** when the code is executed
3. **Unexpected behavior** due to incorrect character interpretation
4. **File system issues** with paths or special characters

## Validation Steps
1. **Identify the target language/file type** from context
2. **Check escaping rules** for that language
3. **Validate special characters** and their escaping
4. **Test string boundaries** and quote consistency
5. **Verify path separators** and file system compatibility

## Output Requirements
Return ONLY a JSON object with the key 'corrected_new_string_escaping' containing the corrected replacement text with proper escaping. If no correction is needed, return the original text.

## Error Recovery Strategy
For complex escaping issues:
- Use ReadFileTool to examine similar patterns in the file
- Check the broader codebase with GrepTool for consistency
- Consider the file type and apply language-specific escaping rules
- Test the correction with a small edit operation first
`.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      CORRECT_NEW_STRING_ESCAPING_SCHEMA,
      abortSignal,
      getEditCorrectionModel('escaping_correction', newString),
      EditConfig,
    )) as Record<string, unknown> | undefined;
    if (result) {
      const corrected1 = result['corrected_new_string_escaping'];
      if (typeof corrected1 === 'string' && corrected1.length > 0)
        return corrected1;
      const corrected2 = result['corrected_new_string'];
      if (typeof corrected2 === 'string' && corrected2.length > 0)
        return corrected2;
    }
    return potentiallyProblematicNewString;
  } catch (error) {
    if (abortSignal.aborted) throw error;
    console.error(
      'Error during LLM call for new_string escaping correction:',
      error,
    );
    return potentiallyProblematicNewString;
  }
}

async function correctStringEscaping(
  potentiallyProblematicString: string,
  client: GeminiClient,
  abortSignal: AbortSignal,
): Promise<string> {
  const prompt = `
# String Escaping Correction Assistant

## Context
An LLM has generated text that may contain improper escaping, potentially causing syntax errors, parsing issues, or unexpected behavior when used in code or configuration files.

## System Capabilities
For complex escaping validation, leverage these intelligent tools:

### Professional Validation Suite:
- **ReadFileTool**: Examine target files for syntax validation
  - *Application*: Verify escaping in context of actual file syntax
  - *Recovery*: Read broader context if specific syntax checking fails
- **GrepTool**: Search codebase for similar patterns and consistency
  - *Application*: Find established escaping conventions in the project
  - *Recovery*: Use literal string searches if pattern matching fails
- **FileSystemService**: Validate file paths and system compatibility
  - *Application*: Ensure path separators work across platforms
  - *Recovery*: Apply platform-specific escaping when generic fails
- **AST Parser**: Check code syntax and structure
  - *Application*: Parse code to validate syntactic correctness
  - *Recovery*: Use simpler text-based validation if parsing fails

### Dual-Context Validation Strategy:
- **Short-term context**: Immediate escaping requirements and syntax
- **Long-term context**: Project conventions and established patterns

## Text Analysis

**Potentially problematic text:**
\`\`\`
${potentiallyProblematicString}
\`\`\`

## Task
Analyze the text for escaping issues that could cause:
1. **Syntax errors** in programming languages
2. **Parsing errors** in configuration files
3. **Runtime exceptions** during execution
4. **File system errors** with paths
5. **Data corruption** in serialized formats

## Common Issues to Check
- **String literals**: Proper quote escaping
- **Regular expressions**: Special character escaping
- **File paths**: OS-specific path separator handling
- **JSON/XML**: Proper entity encoding
- **SQL queries**: Safe parameter escaping
- **HTML**: Entity encoding for special characters

## Validation Guidelines
1. **Determine context**: What type of content is this text for?
2. **Apply appropriate escaping rules** for that context
3. **Check for over-escaping** (too many backslashes)
4. **Verify under-escaping** (missing required escapes)
5. **Test string boundaries** and delimiter consistency

## Output Requirements
Return ONLY a JSON object with the key 'corrected_string_escaping' containing the properly escaped text. If no correction is needed, return the original text.

## Error Recovery Strategy
For ambiguous cases:
- Use ReadFileTool to check how similar strings are handled in the codebase
- Search with GrepTool for patterns and best practices
- Consider the file type and apply domain-specific escaping rules
- When in doubt, prefer safer (more escaped) versions
`.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await client.generateJson(
      contents,
      CORRECT_STRING_ESCAPING_SCHEMA,
      abortSignal,
      getEditCorrectionModel('string_escaping', potentiallyProblematicString),
      EditConfig,
    )) as Record<string, unknown> | undefined;
    if (result) {
      const corrected = result['corrected_string_escaping'];
      if (typeof corrected === 'string' && corrected.length > 0)
        return corrected;
    }
    return potentiallyProblematicString;
  } catch (error) {
    if (abortSignal.aborted) throw error;
    console.error(
      'Error during LLM call for string escaping correction:',
      error,
    );
    return potentiallyProblematicString;
  }
}

function trimPairIfPossible(
  target: string,
  trimIfTargetTrims: string,
  currentContent: string,
  expectedReplacements: number,
): { targetString: string; pair: string } {
  const trimmedTargetString = target.trim();
  if (target.length !== trimmedTargetString.length) {
    const trimmedTargetOccurrences = countOccurrences(
      currentContent,
      trimmedTargetString,
    );
    if (trimmedTargetOccurrences === expectedReplacements) {
      const trimmedReactiveString = trimIfTargetTrims.trim();
      return { targetString: trimmedTargetString, pair: trimmedReactiveString };
    }
  }
  return { targetString: target, pair: trimIfTargetTrims };
}

/**
 * Ensure file content escaping is corrected when necessary via LLM.
 */
export async function ensureCorrectFileContent(
  content: string,
  client: GeminiClient,
  abortSignal: AbortSignal,
): Promise<string> {
  const cached = fileContentCorrectionCache.get(content);
  if (cached) return cached;

  const contentPotentiallyEscaped =
    unescapeStringForGeminiBug(content) !== content;
  if (!contentPotentiallyEscaped) {
    fileContentCorrectionCache.set(content, content);
    return content;
  }

  const correctedContent = await correctStringEscaping(
    content,
    client,
    abortSignal,
  );
  fileContentCorrectionCache.set(content, correctedContent);
  return correctedContent;
}
