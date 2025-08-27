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

const EditModel = DEFAULT_GEMINI_FLASH_LITE_MODEL;
const EditConfig: GenerateContentConfig = {
  thinkingConfig: {
    thinkingBudget: 0,
  },
};

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
Context: A process needs to find an exact literal, unique match for a specific text snippet within a file's content. The provided snippet failed to match exactly. This is most likely because it has been overly escaped.

Task: Analyze the provided file content and the problematic target snippet. Identify the segment in the file content that the snippet was *most likely* intended to match. Output the *exact*, literal text of that segment from the file content. Focus *only* on removing extra escape characters and correcting formatting, whitespace, or minor differences to achieve a PERFECT literal match. The output must be the exact literal text as it appears in the file.

Problematic target snippet:
\`\`\`
${problematicSnippet}
\`\`\`

File Content:
\`\`\`
${fileContent}
\`\`\`

Return ONLY the corrected target snippet in the specified JSON format with the key 'corrected_target_snippet'. If no clear, unique match can be found, return an empty string for 'corrected_target_snippet'.
`.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      OLD_STRING_CORRECTION_SCHEMA,
      abortSignal,
      EditModel,
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
Context: A text replacement operation was planned. The original text to be replaced (original_old_string) was slightly different from the actual text in the file (corrected_old_string). The original_old_string has now been corrected to match the file content.
We now need to adjust the replacement text (original_new_string) so that it makes sense as a replacement for the corrected_old_string, while preserving the original intent of the change.

original_old_string (what was initially intended to be found):
\`\`\`
${originalOldString}
\`\`\`

corrected_old_string (what was actually found in the file and will be replaced):
\`\`\`
${correctedOldString}
\`\`\`

original_new_string (what was intended to replace original_old_string):
\`\`\`
${originalNewString}
\`\`\`

Task: Based on the differences between original_old_string and corrected_old_string, and the content of original_new_string, generate a corrected_new_string. This corrected_new_string should be what original_new_string would have been if it was designed to replace corrected_old_string directly, while maintaining the spirit of the original transformation.

Return ONLY the corrected string in the specified JSON format with the key 'corrected_new_string'. If no adjustment is deemed necessary or possible, return the original_new_string.
  `.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      NEW_STRING_CORRECTION_SCHEMA,
      abortSignal,
      EditModel,
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
  const prompt = `
Context: A text replacement operation is planned. The text to be replaced (old_string) has been correctly identified in the file. However, the replacement text (new_string) might have been improperly escaped by a previous LLM generation.

old_string (this is the exact text that will be replaced):
\`\`${oldString}

potentially_problematic_new_string (this is the text that should replace old_string, but MIGHT have bad escaping, or might be entirely correct):
\`\`${potentiallyProblematicNewString ?? ''}

Task: Analyze the potentially_problematic_new_string. If it's syntactically invalid due to incorrect escaping, correct the invalid syntax. Return ONLY the corrected string in the specified JSON format with the key 'corrected_new_string_escaping'. If no escaping correction is needed, return the original potentially_problematic_new_string.
  `.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await geminiClient.generateJson(
      contents,
      CORRECT_NEW_STRING_ESCAPING_SCHEMA,
      abortSignal,
      EditModel,
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
Context: An LLM has just generated potentially_problematic_string and the text might have been improperly escaped.

potentially_problematic_string (this text MIGHT have bad escaping, or might be entirely correct):
\`\`\`
${potentiallyProblematicString}
\`\`\`

Task: Analyze the potentially_problematic_string. If it's syntactically invalid due to incorrect escaping, correct the invalid syntax.

Return ONLY the corrected string in the specified JSON format with the key 'corrected_string_escaping'. If no escaping correction is needed, return the original potentially_problematic_string.
  `.trim();

  const contents = [{ role: 'user', parts: [{ text: prompt }] }];
  try {
    const result = (await client.generateJson(
      contents,
      CORRECT_STRING_ESCAPING_SCHEMA,
      abortSignal,
      EditModel,
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
