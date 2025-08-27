/**
 * Configuration options for diff generation
 */
export interface DiffOptions {
    /** Timeout for diff computation in milliseconds (default: 1000) */
    timeout?: number;
    /** Whether to perform semantic cleanup (default: true) */
    cleanupSemantic?: boolean;
    /** Whether to perform efficiency cleanup (default: true) */
    cleanupEfficiency?: boolean;
    /** Edit cost for diff computation (default: 4) */
    editCost?: number;
}
/**
 * Generates a unified diff using diff-match-patch with improved configuration and error handling.
 * This provides more accurate and readable diffs for preview purposes.
 */
export declare function generateUnifiedDiff(oldText: string, newText: string, filePath: string, options?: DiffOptions): string;
