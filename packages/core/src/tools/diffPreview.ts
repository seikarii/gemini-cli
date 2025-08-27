import { diff_match_patch as DiffMatchPatch } from 'diff-match-patch';

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
export function generateUnifiedDiff(
  oldText: string,
  newText: string,
  filePath: string,
  options: DiffOptions = {}
): string {
  const {
    timeout = 1000,
    cleanupSemantic = true,
    cleanupEfficiency = true,
    editCost = 4,
  } = options;

  try {
    const dmp = new DiffMatchPatch();

    // Configure diff-match-patch for better performance and accuracy
    dmp.Diff_Timeout = timeout / 1000; // Convert to seconds
    dmp.Diff_EditCost = editCost;

    // Generate the diff
    const diffs = dmp.diff_main(oldText, newText);

    // Apply cleanup for better readability
    if (cleanupSemantic) {
      dmp.diff_cleanupSemantic(diffs);
    }
    if (cleanupEfficiency) {
      dmp.diff_cleanupEfficiency(diffs);
    }

    // Create patches for unified diff format
    const patchList = dmp.patch_make(oldText, diffs);
    const patchText = dmp.patch_toText(patchList);

    // Format as unified diff with proper headers
    const header = `--- ${filePath}\n+++ ${filePath} (modified)`;

    return patchText.trim() ? `${header}\n${patchText}` : `${header}\n(No changes)`;
  } catch (_error) {
    // Fallback to simple diff if diff-match-patch fails
    const oldLines = oldText.split('\n').length;
    const newLines = newText.split('\n').length;
    const lines = [
      `--- ${filePath}`,
      `+++ ${filePath} (modified)`,
      `@@ -1,${oldLines} +1,${newLines} @@`,
      `// Diff generation failed, showing simplified comparison:`,
      `// Old content length: ${oldText.length} characters`,
      `// New content length: ${newText.length} characters`,
      `// Lines changed: ${Math.abs(oldLines - newLines)}`,
    ];
    return lines.join('\n');
  }
}
