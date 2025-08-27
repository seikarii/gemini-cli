/**
 * Shared string utilities used across tools.
 */
export declare function normalizeWhitespace(text: string): string;
/**
 * Counts non-overlapping occurrences of substr in str.
 */
export declare function countOccurrences(str: string, substr: string): number;
/**
 * Calculate Levenshtein distance between two strings
 */
export declare function levenshteinDistance(str1: string, str2: string): number;
/**
 * Calculate string similarity using Levenshtein distance
 */
export declare function calculateStringSimilarity(str1: string, str2: string): number;
/**
 * Escape special regex characters
 */
export declare function escapeRegExp(input: string): string;
