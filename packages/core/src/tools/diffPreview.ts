import DiffMatchPatch from 'diff-match-patch';

/**
 * Generates a unified diff using diff-match-patch for better preview accuracy.
 */
export function generateUnifiedDiff(oldText: string, newText: string, filePath: string): string {
  const dmp = new DiffMatchPatch();
  const diffs = dmp.diff_main(oldText, newText);
  dmp.diff_cleanupSemantic(diffs);
  const patchList = dmp.patch_make(oldText, diffs);
  const patchText = dmp.patch_toText(patchList);
  return `--- ${filePath}\n+++ ${filePath} (modified)\n${patchText}`;
}
