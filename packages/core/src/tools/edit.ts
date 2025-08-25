/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import * as Diff from 'diff';
// Correct import for string-similarity (default import)
import stringSimilarity from 'string-similarity';
// Add diff-match-patch for better matching
import * as DiffMatchPatch from 'diff-match-patch';

import {
  BaseDeclarativeTool,
  Kind,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolEditConfirmationDetails,
  ToolInvocation,
  ToolLocation,
  ToolResult,
  ToolResultDisplay,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { isNodeError } from '../utils/errors.js';
import { Config, ApprovalMode } from '../config/config.js';
import {
  ensureCorrectEdit,
  CorrectedEditResult,
} from '../utils/editCorrector.js';
import { DEFAULT_DIFF_OPTIONS, getDiffStat } from './diffOptions.js';
import { ReadFileTool } from './read-file.js';
import { ModifiableDeclarativeTool, ModifyContext } from './modifiable-tool.js';
import { IDEConnectionStatus } from '../ide/ide-client.js';
import { FileOperation } from '../telemetry/metrics.js';
import { logFileOperation } from '../telemetry/loggers.js';
import { FileOperationEvent } from '../telemetry/types.js';
import { getProgrammingLanguage } from '../telemetry/telemetry-utils.js';
import { getSpecificMimeType } from '../utils/fileUtils.js';
import { resetEditCorrectorCaches } from '../utils/editCorrector.js';

export function applyReplacement(
  currentContent: string | null,
  oldString: string,
  newString: string,
  isNewFile: boolean,
  targetOccurrence?: number | 'first' | 'last' | 'all',
): string {
  if (isNewFile) {
    return newString;
  }
  if (currentContent === null) {
    // Should not happen if not a new file, but defensively return empty or newString if oldString is also empty
    return oldString === '' ? newString : '';
  }
  // If oldString is empty and it's not a new file, do not modify the content.
  if (oldString === '' && !isNewFile) {
    return currentContent;
  }

  // Handle target occurrence modes
  const content = currentContent;

  // If no targetOccurrence provided, preserve historical behavior and replace ALL occurrences by default.
  if (targetOccurrence === undefined) {
    return content.split(oldString).join(newString);
  }

  if (targetOccurrence === 'all') {
    // replace all occurrences explicitly
    return content.split(oldString).join(newString);
  }

  if (targetOccurrence === 'last') {
    const idx = content.lastIndexOf(oldString);
    if (idx === -1) return content;
    return (
      content.slice(0, idx) + newString + content.slice(idx + oldString.length)
    );
  }

  if (
    typeof targetOccurrence === 'number' &&
    Number.isInteger(targetOccurrence) &&
    targetOccurrence > 0
  ) {
    // replace N-th (1-based) occurrence
    let count = 0;
    let start = 0;
    let idx = -1;
    while (true) {
      const found = content.indexOf(oldString, start);
      if (found === -1) break;
      count++;
      if (count === targetOccurrence) {
        idx = found;
        break;
      }
      start = found + oldString.length;
    }
    if (idx === -1) return content;
    return (
      content.slice(0, idx) + newString + content.slice(idx + oldString.length)
    );
  }

  // default / 'first' or undefined -> replace first occurrence only
  return content.replace(oldString, newString);
}

/**
 * Applies a range-based edit to content using line/column coordinates
 */
export function applyRangeEdit(
  currentContent: string,
  startLine: number,
  startColumn: number,
  endLine: number,
  endColumn: number,
  newContent: string,
): string {
  const lines = currentContent.split('\n');

  // Validate range bounds
  if (startLine < 0 || startLine >= lines.length) {
    throw new Error(
      `Start line ${startLine} is out of bounds (0-${lines.length - 1})`,
    );
  }
  if (endLine < 0 || endLine >= lines.length) {
    throw new Error(
      `End line ${endLine} is out of bounds (0-${lines.length - 1})`,
    );
  }
  if (startLine > endLine) {
    throw new Error(
      `Start line ${startLine} cannot be greater than end line ${endLine}`,
    );
  }
  if (startLine === endLine && startColumn > endColumn) {
    throw new Error(
      `Start column ${startColumn} cannot be greater than end column ${endColumn} on the same line`,
    );
  }

  // Extract parts: before range, after range
  const beforeLines = lines.slice(0, startLine);
  const afterLines = lines.slice(endLine + 1);

  // Handle the start and end lines
  const startLineContent = lines[startLine] || '';
  const endLineContent = lines[endLine] || '';

  const beforeRange = startLineContent.substring(0, startColumn);
  const afterRange = endLineContent.substring(endColumn);

  // Combine the result
  const newLines = [
    ...beforeLines,
    beforeRange + newContent + afterRange,
    ...afterLines,
  ];

  return newLines.join('\n');
}

/**
 * Parameters for the Edit tool
 */
export interface EditToolParams {
  /**
   * The absolute path to the file to modify
   */
  file_path: string;

  /**
   * The text to replace (for string-based editing)
   */
  old_string?: string;

  /**
   * The text to replace it with (for string-based editing)
   */
  new_string?: string;

  /**
   * Number of replacements expected. Defaults to 1 if not specified.
   * Use when you want to replace multiple occurrences.
   */
  expected_replacements?: number;

  /**
   * Controls which occurrence(s) to replace:
   * - number (1-based): replace that specific occurrence
   * - 'first': replace the first occurrence (default)
   * - 'last': replace the last occurrence
   * - 'all': replace all occurrences
   */
  target_occurrence?: number | 'first' | 'last' | 'all';

  /**
   * Start line for range-based editing (0-indexed)
   */
  start_line?: number;

  /**
   * Start column for range-based editing (0-indexed)
   */
  start_column?: number;

  /**
   * End line for range-based editing (0-indexed)
   */
  end_line?: number;

  /**
   * End column for range-based editing (0-indexed)
   */
  end_column?: number;

  /**
   * New content to insert (for range-based editing)
   */
  new_content?: string;

  /**
   * Whether the edit was modified manually by the user.
   */
  modified_by_user?: boolean;

  /**
   * Initially proposed string.
   */
  ai_proposed_string?: string;
}

interface CalculatedEdit {
  currentContent: string | null;
  newContent: string;
  occurrences: number;
  error?: { display: string; raw: string; type: ToolErrorType };
  isNewFile: boolean;
  isRangeEdit: boolean;
}

/**
 * Configuration for autofix behavior
 */
interface AutofixConfig {
  // Minimum similarity threshold for fuzzy matching (0-1)
  minSimilarityThreshold: number;
  // Maximum number of candidate matches to consider
  maxCandidates: number;
  // Whether to enable fuzzy matching
  enableFuzzyMatching: boolean;
  // Whether to normalize whitespace
  normalizeWhitespace: boolean;
  // Whether to adjust indentation
  adjustIndentation: boolean;
}

const DEFAULT_AUTOFIX_CONFIG: AutofixConfig = {
  minSimilarityThreshold: 0.7,
  maxCandidates: 5,
  enableFuzzyMatching: true,
  normalizeWhitespace: true,
  adjustIndentation: true,
};

/**
 * Normalizes whitespace in a string while preserving relative indentation
 */
function normalizeWhitespace(text: string): string {
  return text
    .replace(/\r\n/g, '\n') // Normalize line endings
    .replace(/\t/g, '  ') // Convert tabs to spaces
    .replace(/[ ]+$/gm, '') // Remove trailing spaces
    .replace(/^\s*\n/gm, '\n') // Remove empty lines with only whitespace
    .trim();
}

/**
 * Detects the indentation pattern of a text block with improved base detection
 */
function detectIndentationAdvanced(text: string): { baseIndent: string } {
  const lines = text.split('\n').filter((line) => line.trim().length > 0);
  const indentations = lines
    .map((line) => {
      const match = line.match(/^(\s*)/);
      return match ? match[1] : '';
    })
    .filter((indent) => indent.length > 0);

  if (indentations.length === 0) {
    return { baseIndent: '' };
  }

  // Find the minimum common indentation (base indent)
  const baseIndent = indentations.reduce((min, current) => {
    if (min === '') return current;
    if (current === '') return min;

    let i = 0;
    while (i < Math.min(min.length, current.length) && min[i] === current[i]) {
      i++;
    }
    return min.substring(0, i);
  }, indentations[0]);

  return { baseIndent };
}

/**
 * Adjusts the indentation of a text block to match a target indentation
 */
function adjustIndentationAdvanced(text: string, targetIndent: string): string {
  const lines = text.split('\n');
  if (lines.length <= 1) return text;

  const { baseIndent } = detectIndentationAdvanced(text);

  return lines
    .map((line, index) => {
      if (line.trim() === '') return line; // keep empty lines as-is

      const currentIndent = line.match(/^(\s*)/)?.[1] || '';

      if (index === 0) {
        // For first line, replace its indentation with target
        return targetIndent + line.replace(/^\s*/, '');
      } else {
        // For subsequent lines, maintain relative indentation from base
        let relativeIndent = '';
        if (currentIndent.length > baseIndent.length) {
          relativeIndent = currentIndent.slice(baseIndent.length);
        }
        return targetIndent + relativeIndent + line.replace(/^\s*/, '');
      }
    })
    .join('\n');
}

/**
 * Optimized fuzzy matching with sliding window buffer to avoid excessive array creation
 */
function findFuzzyMatchesOptimized(
  content: string,
  searchString: string,
  config: AutofixConfig,
): Array<{
  match: string;
  similarity: number;
  startIndex: number;
  endIndex: number;
}> {
  const normalizedSearch = config.normalizeWhitespace
    ? normalizeWhitespace(searchString)
    : searchString;
  const lines = content.split('\n');
  const searchLines = normalizedSearch.split('\n');
  const candidates: Array<{
    match: string;
    similarity: number;
    startIndex: number;
    endIndex: number;
  }> = [];

  // Try different window sizes around the expected length
  const searchLineCount = searchLines.length;
  const windowSizes = [
    searchLineCount,
    searchLineCount + 1,
    searchLineCount - 1,
    searchLineCount + 2,
    searchLineCount - 2,
  ].filter((size) => size > 0);

  for (const windowSize of windowSizes) {
    if (windowSize > lines.length) continue;

    // Initialize sliding window buffer
    const windowBuffer = lines.slice(0, windowSize);
    let windowText = windowBuffer.join('\n');
    let normalizedCandidate = config.normalizeWhitespace
      ? normalizeWhitespace(windowText)
      : windowText;

    // Check first window
    let similarity = stringSimilarity.compareTwoStrings(
      normalizedSearch,
      normalizedCandidate,
    );
    if (similarity >= config.minSimilarityThreshold) {
      candidates.push({
        match: windowText,
        similarity,
        startIndex: 0,
        endIndex: windowSize - 1,
      });
    }

    // Slide window through remaining lines
    for (let i = 1; i <= lines.length - windowSize; i++) {
      // Slide window: remove first line, add new line at end
      windowBuffer.shift();
      windowBuffer.push(lines[i + windowSize - 1]);
      windowText = windowBuffer.join('\n');
      normalizedCandidate = config.normalizeWhitespace
        ? normalizeWhitespace(windowText)
        : windowText;

      // Calculate similarity for current window
      similarity = stringSimilarity.compareTwoStrings(
        normalizedSearch,
        normalizedCandidate,
      );

      if (similarity >= config.minSimilarityThreshold) {
        candidates.push({
          match: windowText,
          similarity,
          startIndex: i,
          endIndex: i + windowSize - 1,
        });
      }
    }
  }

  // Sort by similarity (descending) and return top candidates
  return candidates
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, config.maxCandidates);
}

/**
 * Advanced diff-based matching using diff-match-patch for better precision
 */
function findBestMatchWithDMP(
  content: string,
  searchString: string,
  config: AutofixConfig,
): { match: string; confidence: number; startIndex: number } | null {
  try {
    const dmp = new DiffMatchPatch.diff_match_patch();

    // Configure diff-match-patch for better matching
    dmp.Match_Threshold = 0.8; // Higher threshold for quality matches
    dmp.Match_Distance = 1000; // Allow matches within reasonable distance

    const normalizedContent = config.normalizeWhitespace
      ? normalizeWhitespace(content)
      : content;
    const normalizedSearch = config.normalizeWhitespace
      ? normalizeWhitespace(searchString)
      : searchString;

    // Find the best location for the search string in content
    const matchLocation = dmp.match_main(
      normalizedContent,
      normalizedSearch,
      0,
    );

    if (matchLocation === -1) {
      return null; // No match found
    }

    // Extract the actual match from original content
    const searchLength = normalizedSearch.length;
    const actualMatch = content.substring(
      matchLocation,
      matchLocation + searchLength,
    );

    // Calculate confidence based on similarity
    const confidence = stringSimilarity.compareTwoStrings(
      normalizedSearch,
      normalizeWhitespace(actualMatch),
    );

    if (confidence >= config.minSimilarityThreshold) {
      return {
        match: actualMatch,
        confidence,
        startIndex: matchLocation,
      };
    }

    return null;
  } catch (error) {
    // Fallback if diff-match-patch fails
    console.debug('DMP matching failed:', error);
    return null;
  }
}

/**
 * Attempts to autofix an edit by adjusting whitespace and indentation.
 * Enhanced version with optimizations and better diff-based matching.
 */
async function autofixEdit(
  currentContent: string,
  oldString: string,
  newString: string,
): Promise<string> {
  const config = DEFAULT_AUTOFIX_CONFIG;
  const appliedFixes: string[] = [];
  let workingOldString = oldString;

  // Step 1: Try exact match first
  if (currentContent.includes(oldString)) {
    return oldString; // No fix needed
  }

  appliedFixes.push('exact_match_failed');

  // Step 2: Basic normalization
  if (config.normalizeWhitespace) {
    const normalizedOld = normalizeWhitespace(oldString);
    const normalizedContent = normalizeWhitespace(currentContent);

    if (normalizedContent.includes(normalizedOld)) {
      // Find the actual text in the original content that matches our normalized version
      const lines = currentContent.split('\n');
      const searchLines = normalizedOld.split('\n');

      // Look for a sequence of lines that when normalized match our target
      for (let i = 0; i <= lines.length - searchLines.length; i++) {
        const candidate = lines.slice(i, i + searchLines.length).join('\n');
        if (normalizeWhitespace(candidate) === normalizedOld) {
          appliedFixes.push('whitespace_normalization');
          return candidate;
        }
      }
    }
  }

  // Step 3: Advanced diff-based matching with diff-match-patch (moved earlier for better precision)
  if (config.enableFuzzyMatching) {
    const dmpMatch = findBestMatchWithDMP(currentContent, oldString, config);
    if (dmpMatch && dmpMatch.confidence >= 0.85) {
      appliedFixes.push(
        `dmp_match_${Math.round(dmpMatch.confidence * 100)}pct`,
      );
      workingOldString = dmpMatch.match;

      // Apply advanced indentation adjustment if needed
      if (config.adjustIndentation && workingOldString !== oldString) {
        const lines = currentContent.split('\n');
        const matchLines = workingOldString.split('\n');

        if (matchLines.length > 0) {
          // Find where this content appears in the file to get proper indentation
          const firstLineContent = matchLines[0].trim();
          const matchingLineIndex = lines.findIndex(
            (line) => line.trim() === firstLineContent,
          );

          if (matchingLineIndex !== -1) {
            const targetIndent =
              lines[matchingLineIndex].match(/^(\s*)/)?.[1] || '';
            const adjustedOldString = adjustIndentationAdvanced(
              oldString,
              targetIndent,
            );

            if (currentContent.includes(adjustedOldString)) {
              appliedFixes.push('advanced_indentation_adjustment');
              workingOldString = adjustedOldString;
            }
          }
        }
      }

      // If we found a good match with DMP, return it
      return workingOldString;
    }
  }

  // Step 4: Optimized fuzzy matching as fallback
  if (config.enableFuzzyMatching && workingOldString === oldString) {
    const fuzzyMatches = findFuzzyMatchesOptimized(
      currentContent,
      oldString,
      config,
    );

    if (fuzzyMatches.length > 0) {
      const bestMatch = fuzzyMatches[0];
      appliedFixes.push(
        `optimized_fuzzy_match_${Math.round(bestMatch.similarity * 100)}pct`,
      );

      // If we have a high confidence match, use it
      if (bestMatch.similarity >= 0.85) {
        workingOldString = bestMatch.match;
      }
    }
  }

  // Step 5: Basic indentation adjustment (fallback for cases where advanced didn't apply)
  if (config.adjustIndentation && workingOldString !== oldString) {
    const lines = currentContent.split('\n');
    const oldLines = workingOldString.split('\n');

    if (oldLines.length > 0) {
      // Find where this content appears in the file to get proper indentation
      const firstLineContent = oldLines[0].trim();
      const matchingLineIndex = lines.findIndex(
        (line) => line.trim() === firstLineContent,
      );

      if (matchingLineIndex !== -1) {
        const targetIndent =
          lines[matchingLineIndex].match(/^(\s*)/)?.[1] || '';
        const adjustedOldString = adjustIndentationAdvanced(
          oldString,
          targetIndent,
        );

        if (currentContent.includes(adjustedOldString)) {
          appliedFixes.push('basic_indentation_adjustment');
          workingOldString = adjustedOldString;
        }
      }
    }
  }

  // Log telemetry for debugging and improvement
  if (appliedFixes.length > 1) {
    // More than just 'exact_match_failed'
    console.debug('AutofixEdit applied fixes:', appliedFixes.join(', '));

    // Log confidence metrics for future improvements
    if (workingOldString !== oldString) {
      const finalSimilarity = stringSimilarity.compareTwoStrings(
        normalizeWhitespace(oldString),
        normalizeWhitespace(workingOldString),
      );
      console.debug(
        `AutofixEdit final similarity: ${Math.round(finalSimilarity * 100)}%`,
      );
    }
  }

  return workingOldString;
}

class EditToolInvocation implements ToolInvocation<EditToolParams, ToolResult> {
  constructor(
    private readonly config: Config,
    public params: EditToolParams,
  ) {}

  toolLocations(): ToolLocation[] {
    return [{ path: this.params.file_path }];
  }

  /**
   * Determines if this is a range-based edit or string-based edit
   */
  private isRangeEdit(params: EditToolParams): boolean {
    return (
      params.start_line !== undefined &&
      params.start_column !== undefined &&
      params.end_line !== undefined &&
      params.end_column !== undefined &&
      params.new_content !== undefined
    );
  }

  /**
   * Validates range edit parameters
   */

  /**
   * Counts occurrences of a string in content
   */
  private countOccurrences(content: string, searchString: string): number {
    if (!searchString) return 0;
    let count = 0;
    let pos = 0;
    while ((pos = content.indexOf(searchString, pos)) !== -1) {
      count++;
      pos += searchString.length;
    }
    return count;
  }

  /**
   * Calculates how many replacements will actually be performed given target_occurrence
   */
  private calculateExpectedReplacements(
    totalOccurrences: number,
    targetOccurrence: number | 'first' | 'last' | 'all' | undefined,
  ): number {
    // Default historical behavior: if not specified, user intended to replace all occurrences.
    const target = targetOccurrence ?? 'all';

    if (totalOccurrences === 0) return 0;

    if (target === 'all') return totalOccurrences;
    if (target === 'first' || target === 'last') return 1;
    if (typeof target === 'number') {
      return totalOccurrences >= target ? 1 : 0;
    }

    return 1; // default fallback
  }

  private validateRangeParams(params: EditToolParams): string | null {
    if (!this.isRangeEdit(params)) {
      return null; // Not a range edit, no validation needed
    }

    const { start_line, start_column, end_line, end_column } = params;

    if (
      start_line! < 0 ||
      start_column! < 0 ||
      end_line! < 0 ||
      end_column! < 0
    ) {
      return 'Line and column numbers must be non-negative';
    }

    if (start_line! > end_line!) {
      return 'Start line cannot be greater than end line';
    }

    if (start_line === end_line && start_column! > end_column!) {
      return 'Start column cannot be greater than end column on the same line';
    }

    return null;
  }

  /**
   * Calculates the potential outcome of an edit operation.
   */
  private async calculateEdit(
    params: EditToolParams,
    abortSignal: AbortSignal,
  ): Promise<CalculatedEdit> {
    const isRangeEdit = this.isRangeEdit(params);
    let currentContent: string | null = null;
    let fileExists = false;
    let isNewFile = false;
    let finalNewString = params.new_string || '';
    let finalOldString = params.old_string || '';
    let occurrences = 0;
    let error:
      | { display: string; raw: string; type: ToolErrorType }
      | undefined = undefined;

    // Validate range parameters if this is a range edit
    if (isRangeEdit) {
      const rangeError = this.validateRangeParams(params);
      if (rangeError) {
        error = {
          display: rangeError,
          raw: `Range validation error: ${rangeError}`,
          type: ToolErrorType.EDIT_PREPARATION_FAILURE,
        };
        return {
          currentContent: null,
          newContent: '',
          occurrences: 0,
          error,
          isNewFile: false,
          isRangeEdit: true,
        };
      }
    }

    try {
      const res = await this.config
        .getFileSystemService()
        .readTextFile(params.file_path);
      if (res.success) {
        currentContent = res.data!;
        // Normalize line endings to LF for consistent processing.
        currentContent = currentContent.replace(/\r\n/g, '\n');
        fileExists = true;
      } else {
        if (res.errorCode !== 'ENOENT') {
          throw new Error(res.error);
        }
        fileExists = false;
      }
    } catch (err: unknown) {
      if (!isNodeError(err) || err.code !== 'ENOENT') {
        // Rethrow unexpected FS errors (permissions, etc.)
        throw err;
      }
      fileExists = false;
    }

    // Handle file creation logic
    if (!fileExists) {
      if (isRangeEdit) {
        error = {
          display:
            'File not found. Cannot apply range edit to non-existent file.',
          raw: `File not found: ${params.file_path}`,
          type: ToolErrorType.FILE_NOT_FOUND,
        };
      } else if (params.old_string === '') {
        // Creating a new file with string-based edit
        isNewFile = true;
      } else {
        // Trying to edit a nonexistent file (and old_string is not empty)
        error = {
          display: `File not found. Cannot apply edit. Use an empty old_string to create a new file.`,
          raw: `File not found: ${params.file_path}`,
          type: ToolErrorType.FILE_NOT_FOUND,
        };
      }
    } else if (currentContent !== null) {
      // File exists, handle editing
      if (isRangeEdit) {
        // Range-based editing
        try {
          const lines = currentContent.split('\n');
          const { start_line, start_column, end_line, end_column } = params;

          // Additional runtime validation against actual file content
          if (start_line! >= lines.length || end_line! >= lines.length) {
            error = {
              display: `Line numbers out of bounds. File has ${lines.length} lines.`,
              raw: `Line numbers out of bounds for file: ${params.file_path}`,
              type: ToolErrorType.EDIT_PREPARATION_FAILURE,
            };
          } else if (
            start_column! > lines[start_line!].length ||
            end_column! > lines[end_line!].length
          ) {
            error = {
              display: `Column numbers out of bounds for specified lines.`,
              raw: `Column numbers out of bounds for file: ${params.file_path}`,
              type: ToolErrorType.EDIT_PREPARATION_FAILURE,
            };
          } else {
            occurrences = 1; // Range edits always have exactly 1 "occurrence"
          }
        } catch (rangeError) {
          error = {
            display: `Range edit validation failed: ${rangeError instanceof Error ? rangeError.message : String(rangeError)}`,
            raw: `Range edit validation failed for file: ${params.file_path}`,
            type: ToolErrorType.EDIT_PREPARATION_FAILURE,
          };
        }
      } else {
        // String-based editing
        if (params.old_string === '') {
          // Error: Trying to create a file that already exists
          error = {
            display: `Failed to edit. Attempted to create a file that already exists.`,
            raw: `File already exists, cannot create: ${params.file_path}`,
            type: ToolErrorType.ATTEMPT_TO_CREATE_EXISTING_FILE,
          };
        } else {
          await autofixEdit(
            currentContent,
            params.old_string || '',
            params.new_string || '',
          );

          // Call ensureCorrectEdit with the original params object (tests expect the same reference).
          const correctedEdit: CorrectedEditResult = await ensureCorrectEdit(
            params.file_path,
            currentContent,
            params,
            this.config.getGeminiClient(),
            abortSignal,
          );
          finalOldString = correctedEdit.params.old_string;
          finalNewString = correctedEdit.params.new_string;

          // Calculate total occurrences in content. Prefer the occurrences info returned by ensureCorrectEdit if provided.
          const totalOccurrences =
            typeof (correctedEdit as CorrectedEditResult).occurrences ===
            'number'
              ? (correctedEdit as CorrectedEditResult).occurrences
              : this.countOccurrences(currentContent, finalOldString);

          // Additional defensive guards to avoid over-broad replacements which
          // historically have caused truncation or destructive writes when a
          // short or overly-common snippet is used as the target. These guards
          // are conservative: they require callers to be explicit when the
          // match is widespread.
          try {
            const lineCount = (currentContent.match(/\n/g)?.length ?? 0) + 1;

            // If the target occurs on (approximately) every line of the file,
            // this is suspicious for a single-line replacement. Require explicit
            // target occurrence or expected_replacements to proceed.
            if (
              totalOccurrences >= lineCount &&
              params.target_occurrence === undefined &&
              params.expected_replacements === undefined
            ) {
              error = {
                display: `Suspicious replacement: target snippet appears on every line. Please specify target_occurrence or expected_replacements to proceed.`,
                raw: `Suspicious replacement: final_old_string appears on ${totalOccurrences} lines (file has ${lineCount} lines) for file: ${params.file_path}`,
                type: ToolErrorType.EDIT_PREPARATION_FAILURE,
              };
            }

            // If the corrected old string is very short (likely punctuation or a
            // small token) and occurs many times, block it to avoid accidental
            // global replacements. This protects against single-character or
            // tiny-token replacements that can drastically alter files.
            if (!error && finalOldString.length > 0 && finalOldString.length < 4 && totalOccurrences > 3) {
              error = {
                display: `Suspicious replacement: target snippet is very short and occurs ${totalOccurrences} times. Provide a more specific old_string or an explicit target_occurrence.`,
                raw: `Suspicious replacement: final_old_string (len=${finalOldString.length}) occurs ${totalOccurrences} times in ${params.file_path}`,
                type: ToolErrorType.EDIT_PREPARATION_FAILURE,
              };
            }
          } catch (guardErr) {
            // Don't let guard logic throw; proceed optimistically if it fails
            console.debug('Replacement guards failed unexpectedly', guardErr);
          }

          // Calculate actual replacements that will be performed
          occurrences = this.calculateExpectedReplacements(
            totalOccurrences,
            params.target_occurrence,
          );

          // New handling for target_occurrence
          // Default to 'all' to preserve historical behavior unless caller specified otherwise.
          const target = params.target_occurrence ?? 'all';

          if (totalOccurrences === 0) {
            error = {
              display: `Failed to edit, could not find the string to replace.`,
              raw: `Failed to edit, 0 occurrences found for old_string in ${params.file_path}. No edits made. The exact text in old_string was not found. Ensure you're not escaping content incorrectly and check whitespace, indentation, and context. Use ${ReadFileTool.Name} tool to verify.`,
              type: ToolErrorType.EDIT_NO_OCCURRENCE_FOUND,
            };
          } else if (
            totalOccurrences > 1 &&
            params.target_occurrence === undefined &&
            params.expected_replacements === undefined
          ) {
            // For backward compatibility, respond with the expected-occurrence mismatch error
            error = {
              display: `Failed to edit, expected 1 occurrence but found ${totalOccurrences} for old_string in file`,
              raw: `Expected 1 occurrence but found ${totalOccurrences} for old_string in file: ${params.file_path}`,
              type: ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
            };
          } else if (target === 'all') {
            // For 'all', validate against actual total if expected_replacements is specified
            if (
              params.expected_replacements !== undefined &&
              totalOccurrences !== params.expected_replacements
            ) {
              error = {
                display: `Failed to edit, expected ${params.expected_replacements} occurrences but found ${totalOccurrences}`,
                raw: `Expected ${params.expected_replacements} occurrences but found ${totalOccurrences} for old_string in file: ${params.file_path}`,
                type: ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
              };
            }
          } else if (typeof target === 'number') {
            if (totalOccurrences < target) {
              error = {
                display: `Failed to edit, target occurrence ${target} not found (only ${totalOccurrences} occurrences present).`,
                raw: `Failed to edit, target occurrence ${target} not found (only ${totalOccurrences} occurrences) for old_string in file: ${params.file_path}`,
                type: ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
              };
            } else {
              // Single replacement expected - validate if user provided expected_replacements
              if (
                params.expected_replacements !== undefined &&
                params.expected_replacements !== 1
              ) {
                error = {
                  display: `expected_replacements conflicts with target_occurrence.`,
                  raw: `Parameter mismatch: expected_replacements=${params.expected_replacements} but target_occurrence=${target} will perform 1 replacement`,
                  type: ToolErrorType.EDIT_PREPARATION_FAILURE,
                };
              }
            }
          } else {
            // 'first' or 'last' or default behavior -> single replacement expected
            if (
              params.expected_replacements !== undefined &&
              params.expected_replacements !== 1
            ) {
              error = {
                display: `Failed to edit, expected ${params.expected_replacements} occurrences but target_occurrence="${target}" will perform 1 replacement.`,
                raw: `Failed to edit, Expected ${params.expected_replacements} occurrences but target_occurrence="${target}" performs 1 replacement for old_string in file: ${params.file_path}`,
                type: ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
              };
            }
          }

          // Check for no-op replacement
          if (!error && finalOldString === finalNewString) {
            error = {
              display: `No changes to apply. The old_string and new_string are identical.`,
              raw: `No changes to apply. The old_string and new_string are identical in file: ${params.file_path}`,
              type: ToolErrorType.EDIT_NO_CHANGE,
            };
          }
        }
      }
    } else {
      // Should not happen if fileExists and no exception was thrown, but defensively:
      error = {
        display: `Failed to read content of file.`,
        raw: `Failed to read content of existing file: ${params.file_path}`,
        type: ToolErrorType.READ_CONTENT_FAILURE,
      };
    }

    // Calculate new content
    let newContent = currentContent ?? '';
    if (!error) {
      if (isRangeEdit && currentContent !== null) {
        try {
          newContent = applyRangeEdit(
            currentContent,
            params.start_line!,
            params.start_column!,
            params.end_line!,
            params.end_column!,
            params.new_content!,
          );
        } catch (rangeError) {
          error = {
            display: `Range edit failed: ${rangeError instanceof Error ? rangeError.message : String(rangeError)}`,
            raw: `Range edit failed for file: ${params.file_path}`,
            type: ToolErrorType.EDIT_PREPARATION_FAILURE,
          };
          newContent = currentContent;
        }
      } else if (!isRangeEdit) {
        newContent = applyReplacement(
          currentContent,
          finalOldString,
          finalNewString,
          isNewFile,
          params.target_occurrence,
        );
      }
    }

    // Check if content actually changed
    if (!error && fileExists && currentContent === newContent) {
      console.debug('DEBUG calculateEdit: setting EDIT_NO_CHANGE', {
        file: params.file_path,
      });
      error = {
        display:
          'No changes to apply. The new content is identical to the current content.',
        raw: `No changes to apply. The new content is identical to the current content in file: ${params.file_path}`,
        type: ToolErrorType.EDIT_NO_CHANGE,
      };
    }

    return {
      currentContent,
      newContent,
      occurrences,
      error,
      isNewFile,
      isRangeEdit,
    };
  }

  /**
   * Handles the confirmation prompt for the Edit tool in the CLI.
   * It needs to calculate the diff to show the user.
   */
  async shouldConfirmExecute(
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    if (this.config.getApprovalMode() === ApprovalMode.AUTO_EDIT) {
      return false;
    }

    let editData: CalculatedEdit;
    try {
      editData = await this.calculateEdit(this.params, abortSignal);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.log(`Error preparing edit: ${errorMsg}`);
      return false;
    }

    if (editData.error) {
      console.log(`Error: ${editData.error.display}`);
      return false;
    }

    const fileName = path.basename(this.params.file_path);
    const fileDiff = Diff.createPatch(
      fileName,
      editData.currentContent ?? '',
      editData.newContent,
      'Current',
      'Proposed',
      DEFAULT_DIFF_OPTIONS,
    );
    const ideClient = this.config.getIdeClient();
    const ideConfirmation =
      this.config.getIdeMode() &&
      ideClient?.getConnectionStatus().status === IDEConnectionStatus.Connected
        ? ideClient.openDiff(this.params.file_path, editData.newContent)
        : undefined;

    const confirmationDetails: ToolEditConfirmationDetails = {
      type: 'edit',
      title: `Confirm Edit: ${shortenPath(makeRelative(this.params.file_path, this.config.getTargetDir()))}`,
      fileName,
      filePath: this.params.file_path,
      fileDiff,
      originalContent: editData.currentContent ?? '',
      newContent: editData.newContent,
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlways) {
          this.config.setApprovalMode(ApprovalMode.AUTO_EDIT);
        }

        if (ideConfirmation) {
          const result = await ideConfirmation;
          if (result.status === 'accepted' && result.content) {
            // Update params based on edit type
            if (editData.isRangeEdit) {
              this.params.new_content = result.content;
            } else {
              this.params.old_string = editData.currentContent ?? '';
              this.params.new_string = result.content;
            }
          }
        }
      },
      ideConfirmation,
    };
    return confirmationDetails;
  }

  getDescription(): string {
    const relativePath = makeRelative(
      this.params.file_path,
      this.config.getTargetDir(),
    );

    if (this.isRangeEdit(this.params)) {
      return `Range edit ${shortenPath(relativePath)} (${this.params.start_line}:${this.params.start_column}-${this.params.end_line}:${this.params.end_column})`;
    }

    if (this.params.old_string === '') {
      return `Create ${shortenPath(relativePath)}`;
    }

    const oldStringSnippet =
      (this.params.old_string || '').split('\n')[0].substring(0, 30) +
      ((this.params.old_string || '').length > 30 ? '...' : '');
    const newStringSnippet =
      (this.params.new_string || '').split('\n')[0].substring(0, 30) +
      ((this.params.new_string || '').length > 30 ? '...' : '');

    if (this.params.old_string === this.params.new_string) {
      return `No file changes to ${shortenPath(relativePath)}`;
    }
    return `${shortenPath(relativePath)}: ${oldStringSnippet} => ${newStringSnippet}`;
  }

  /**
   * Executes the edit operation with the given parameters.
   * @param params Parameters for the edit operation
   * @returns Result of the edit operation
   */
  async execute(signal: AbortSignal): Promise<ToolResult> {
    let editData: CalculatedEdit;
    try {
      editData = await this.calculateEdit(this.params, signal);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error preparing edit: ${errorMsg}`,
        returnDisplay: `Error preparing edit: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.EDIT_PREPARATION_FAILURE,
        },
      };
    }

    if (editData.error) {
      console.debug('DEBUG execute: editData.error present:', editData.error);
      return {
        llmContent: editData.error.raw,
        returnDisplay: `Error: ${editData.error.display}`,
        error: {
          message: editData.error.raw,
          type: editData.error.type,
        },
      };
    }

    // Defensive: if edit calculations result in no-op, return EDIT_NO_CHANGE
    const normalizeLineEndings = (s: string | null) =>
      (s ?? '').replace(/\r\n/g, '\n');
    if (
      !editData.isNewFile &&
      normalizeLineEndings(editData.currentContent) ===
        normalizeLineEndings(editData.newContent)
    ) {
      // Debug logging to aid test diagnosis
      console.debug('DEBUG EDIT_NO_CHANGE', {
        file: this.params.file_path,
        occurrences: editData.occurrences,
        currentContentSnippet: (editData.currentContent || '').slice(0, 120),
        newContentSnippet: (editData.newContent || '').slice(0, 120),
      });
      return {
        llmContent: `No changes to apply. The new content is identical to the current content.`,
        returnDisplay: `No changes to apply. The new content is identical to the current content.`,
        error: {
          message: 'No changes to apply',
          type: ToolErrorType.EDIT_NO_CHANGE,
        },
      };
    }

    try {
      this.ensureParentDirectoriesExist(this.params.file_path);
      // If target file exists but is not writable, return FILE_WRITE_FAILURE to match expected behavior
      try {
        if (fs.existsSync(this.params.file_path)) {
          fs.accessSync(this.params.file_path, fs.constants.W_OK);
        }
      } catch (_accessErr) {
        return {
          llmContent: `Error executing edit: Permission denied writing to file: ${this.params.file_path}`,
          returnDisplay: `Error writing file: Permission denied writing to file: ${this.params.file_path}`,
          error: {
            message: `Permission denied writing to file: ${this.params.file_path}`,
            type: ToolErrorType.FILE_WRITE_FAILURE,
          },
        };
      }

      const writeResult = await this.config
        .getFileSystemService()
        .writeTextFile(this.params.file_path, editData.newContent);
      if (!writeResult.success) {
        throw new Error(writeResult.error);
      }
      resetEditCorrectorCaches();

      // Verify that the file on disk actually changed. Some write operations
      // may succeed but result in identical content (no-op). In that case,
      // return EDIT_NO_CHANGE so callers/tests receive the appropriate error.
      try {
        const verifyRes = await this.config
          .getFileSystemService()
          .readTextFile(this.params.file_path);
        if (verifyRes.success) {
          const onDisk = (verifyRes.data ?? '').replace(/\r\n/g, '\n');
          const before = (editData.currentContent ?? '').replace(/\r\n/g, '\n');
          if (!editData.isNewFile && onDisk === before) {
            return {
              llmContent: `No changes to apply. The new content is identical to the current content.`,
              returnDisplay: `No changes to apply. The new content is identical to the current content.`,
              error: {
                message: 'No changes to apply',
                type: ToolErrorType.EDIT_NO_CHANGE,
              },
            };
          }
        }
      } catch {
        // ignore verification errors - fall through to normal behavior
      }

      let displayResult: ToolResultDisplay;
      const fileName = path.basename(this.params.file_path);
      const originallyProposedContent =
        this.params.ai_proposed_string ||
        this.params.new_string ||
        this.params.new_content ||
        '';
      const diffStat = getDiffStat(
        fileName,
        editData.currentContent ?? '',
        originallyProposedContent,
        editData.newContent,
      );

      if (editData.isNewFile) {
        displayResult = `Created ${shortenPath(makeRelative(this.params.file_path, this.config.getTargetDir()))}`;
      } else {
        // Generate diff for display, even though core logic doesn't technically need it
        // The CLI wrapper will use this part of the ToolResult
        const fileDiff = Diff.createPatch(
          fileName,
          editData.currentContent ?? '', // Should not be null here if not isNewFile
          editData.newContent,
          'Current',
          'Proposed',
          DEFAULT_DIFF_OPTIONS,
        );
        displayResult = {
          fileDiff,
          fileName,
          originalContent: editData.currentContent ?? '',
          newContent: editData.newContent,
          diffStat,
        };
      }

      const editType = editData.isRangeEdit ? 'range' : 'string';
      const llmSuccessMessageParts = [
        editData.isNewFile
          ? `Created new file: ${this.params.file_path} with provided content.`
          : `Successfully modified file: ${this.params.file_path} using ${editType} edit (${editData.occurrences} replacements).`,
      ];
      if (this.params.modified_by_user) {
        // Include the canonical phrase expected by tests while still providing context
        llmSuccessMessageParts.push(
          `User modified the ` + '`new_string`' + ` content`,
        );
        const modifiedContent = editData.isRangeEdit
          ? this.params.new_content
          : this.params.new_string;
        // Provide the modified content for additional context
        llmSuccessMessageParts.push(`Modified content: ${modifiedContent}.`);
      }

      const lines = editData.newContent.split('\n').length;
      const mimetype = getSpecificMimeType(this.params.file_path);
      const extension = path.extname(this.params.file_path);
      const programming_language = getProgrammingLanguage({
        file_path: this.params.file_path,
      });

      logFileOperation(
        this.config,
        new FileOperationEvent(
          EditTool.Name,
          editData.isNewFile ? FileOperation.CREATE : FileOperation.UPDATE,
          lines,
          mimetype,
          extension,
          diffStat,
          programming_language,
        ),
      );

      return {
        llmContent: llmSuccessMessageParts.join(' '),
        returnDisplay: displayResult,
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error executing edit: ${errorMsg}`,
        returnDisplay: `Error writing file: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_WRITE_FAILURE,
        },
      };
    }
  }

  /**
   * Creates parent directories if they don't exist
   */
  private ensureParentDirectoriesExist(filePath: string): void {
    const dirName = path.dirname(filePath);
    if (!fs.existsSync(dirName)) {
      fs.mkdirSync(dirName, { recursive: true });
    }
  }
}

/**
 * Implementation of the Edit tool logic
 */
export class EditTool
  extends BaseDeclarativeTool<EditToolParams, ToolResult>
  implements ModifiableDeclarativeTool<EditToolParams>
{
  static readonly Name = 'replace';
  constructor(private readonly config: Config) {
    super(
      EditTool.Name,
      'Edit',
      `Replaces text within a file using either string-based replacement or precise range-based editing.

**String-based editing:**
By default, replaces the first occurrence of \`old_string\`. Use \`target_occurrence\` to control which occurrence(s) to replace:
- \`target_occurrence: 'first'\` (default): Replace the first occurrence only
- \`target_occurrence: 'last'\`: Replace the last occurrence only  
- \`target_occurrence: 'all'\`: Replace all occurrences
- \`target_occurrence: N\` (number): Replace the N-th occurrence (1-based)

The \`expected_replacements\` parameter serves as validation - ensure it matches the actual number of replacements that will be performed.

**Range-based editing (robust mode):**
Allows precise specification of what to delete and insert using line/column coordinates. This mode is more robust as it doesn't depend on exact string matching.

Always use the ${ReadFileTool.Name} tool to examine the file's current content before attempting any edit.

**For string-based editing:**
1. \`file_path\` MUST be an absolute path.
2. \`old_string\` MUST be the exact literal text to replace (including all whitespace, indentation, newlines, etc.).
3. \`new_string\` MUST be the exact literal text to replace \`old_string\` with.
4. \`target_occurrence\` controls which occurrences to replace.
5. \`expected_replacements\` validates the number of actual replacements (optional but recommended).
6. NEVER escape \`old_string\` or \`new_string\`.

**For range-based editing:**
1. \`file_path\` MUST be an absolute path.
2. \`start_line\`, \`start_column\`, \`end_line\`, \`end_column\` specify the range to delete (0-indexed).
3. \`new_content\` is the content to insert at the start position.

**Examples:**
- Replace all occurrences: \`{ target_occurrence: 'all' }\`
- Replace 3rd occurrence: \`{ target_occurrence: 3 }\`
- Replace last occurrence: \`{ target_occurrence: 'last' }\`
- Validate replacement count: \`{ target_occurrence: 'all', expected_replacements: 5 }\``,
      Kind.Edit,
      {
        properties: {
          file_path: {
            description:
              "The absolute path to the file to modify. Must start with '/'.",
            type: 'string',
          },
          old_string: {
            description:
              'The exact literal text to replace (string-based editing). Include context for precise targeting.',
            type: 'string',
          },
          new_string: {
            description:
              'The exact literal text to replace `old_string` with (string-based editing).',
            type: 'string',
          },
          expected_replacements: {
            type: 'number',
            description:
              'Expected number of replacements for validation. For target_occurrence="all", this validates the total count. For specific occurrences, this should be 1.',
            minimum: 1,
          },
          target_occurrence: {
            description:
              'Controls which occurrence(s) to replace: number (1-based) for specific occurrence, "first" for the first (default), "last" for the last, or "all" for all occurrences.',
            oneOf: [
              { type: 'number', minimum: 1 },
              { type: 'string', enum: ['first', 'last', 'all'] },
            ],
          },
          start_line: {
            type: 'number',
            description: 'Start line for range-based editing (0-indexed).',
            minimum: 0,
          },
          start_column: {
            type: 'number',
            description: 'Start column for range-based editing (0-indexed).',
            minimum: 0,
          },
          end_line: {
            type: 'number',
            description: 'End line for range-based editing (0-indexed).',
            minimum: 0,
          },
          end_column: {
            type: 'number',
            description: 'End column for range-based editing (0-indexed).',
            minimum: 0,
          },
          new_content: {
            type: 'string',
            description: 'New content to insert for range-based editing.',
          },
        },
        required: ['file_path'],
        type: 'object',
      },
    );
  }

  /**
   * Validates the parameters for the Edit tool
   */
  protected override validateToolParamValues(
    params: EditToolParams,
  ): string | null {
    if (!params.file_path) {
      return "The 'file_path' parameter must be non-empty.";
    }

    if (!path.isAbsolute(params.file_path)) {
      return `File path must be absolute: ${params.file_path}`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(params.file_path)) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(', ')}`;
    }

    // Check if this is range-based or string-based editing
    const hasRangeParams =
      params.start_line !== undefined ||
      params.start_column !== undefined ||
      params.end_line !== undefined ||
      params.end_column !== undefined ||
      params.new_content !== undefined;

    const hasStringParams =
      params.old_string !== undefined || params.new_string !== undefined;

    if (hasRangeParams && hasStringParams) {
      return 'Cannot mix range-based and string-based editing parameters. Use either (start_line, start_column, end_line, end_column, new_content) or (old_string, new_string).';
    }

    if (hasRangeParams) {
      // Validate range parameters
      if (
        params.start_line === undefined ||
        params.start_column === undefined ||
        params.end_line === undefined ||
        params.end_column === undefined ||
        params.new_content === undefined
      ) {
        return 'Range-based editing requires all of: start_line, start_column, end_line, end_column, new_content';
      }

      // target_occurrence doesn't apply to range edits
      if (params.target_occurrence !== undefined) {
        return 'target_occurrence parameter is not applicable to range-based editing';
      }
    } else {
      // Validate string parameters
      if (params.old_string === undefined || params.new_string === undefined) {
        return 'String-based editing requires both old_string and new_string';
      }

      // Validate target_occurrence if provided
      if (params.target_occurrence !== undefined) {
        const target = params.target_occurrence;
        if (typeof target === 'number') {
          if (!Number.isInteger(target) || target < 1) {
            return 'target_occurrence must be a positive integer (1-based) when specified as a number';
          }
        } else if (typeof target === 'string') {
          if (!['first', 'last', 'all'].includes(target)) {
            return 'target_occurrence must be "first", "last", "all", or a positive integer';
          }
        } else {
          return 'target_occurrence must be a number or one of: "first", "last", "all"';
        }
      }

      // Validate compatibility between target_occurrence and expected_replacements
      if (
        params.target_occurrence !== undefined &&
        params.expected_replacements !== undefined
      ) {
        const target = params.target_occurrence;
        if (
          (target === 'first' ||
            target === 'last' ||
            typeof target === 'number') &&
          params.expected_replacements !== 1
        ) {
          return 'expected_replacements must be 1 when target_occurrence is "first", "last", or a specific number';
        }
      }
    }

    return null;
  }

  getModifyContext(_abortSignal: AbortSignal): ModifyContext<EditToolParams> {
    return {
      getFilePath: (params: EditToolParams) => params.file_path,
      getCurrentContent: async (params: EditToolParams): Promise<string> => {
        try {
          const res = await this.config
            .getFileSystemService()
            .readTextFile(params.file_path);
          if (res.success) {
            return res.data ?? '';
          }
          return '';
        } catch (err) {
          if (!isNodeError(err) || err.code !== 'ENOENT') throw err;
          return '';
        }
      },
      getProposedContent: async (params: EditToolParams): Promise<string> => {
        try {
          const res = await this.config
            .getFileSystemService()
            .readTextFile(params.file_path);
          let currentContent: string;
          if (res.success) {
            currentContent = res.data ?? '';
          } else {
            currentContent = '';
          }

          // Determine edit type and apply appropriate transformation
          if (
            params.start_line !== undefined &&
            params.start_column !== undefined &&
            params.end_line !== undefined &&
            params.end_column !== undefined &&
            params.new_content !== undefined
          ) {
            // Range-based edit
            return applyRangeEdit(
              currentContent,
              params.start_line,
              params.start_column,
              params.end_line,
              params.end_column,
              params.new_content,
            );
          } else {
            // String-based edit
            return applyReplacement(
              currentContent,
              params.old_string || '',
              params.new_string || '',
              params.old_string === '' && currentContent === '',
              params.target_occurrence,
            );
          }
        } catch (err) {
          if (!isNodeError(err) || err.code !== 'ENOENT') throw err;
          return '';
        }
      },
      createUpdatedParams: (
        oldContent: string,
        modifiedProposedContent: string,
        originalParams: EditToolParams,
      ): EditToolParams => {
        const isRangeEdit =
          originalParams.start_line !== undefined &&
          originalParams.start_column !== undefined &&
          originalParams.end_line !== undefined &&
          originalParams.end_column !== undefined;

        if (isRangeEdit) {
          return {
            ...originalParams,
            ai_proposed_string: originalParams.new_content,
            new_content: modifiedProposedContent,
            modified_by_user: true,
          };
        } else {
          return {
            ...originalParams,
            ai_proposed_string: originalParams.new_string,
            old_string: oldContent,
            new_string: modifiedProposedContent,
            modified_by_user: true,
          };
        }
      },
    };
  }

  protected override createInvocation(
    params: EditToolParams,
  ): ToolInvocation<EditToolParams, ToolResult> {
    return new EditToolInvocation(this.config, params);
  }
}
