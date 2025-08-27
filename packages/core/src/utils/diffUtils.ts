import * as Diff from 'diff';

/**
 * Wrapper around Diff.createPatch to centralize patch formatting and options.
 */
export function createPatch(
  fileName: string,
  oldStr: string,
  newStr: string,
  oldHeader?: string,
  newHeader?: string,
) {
  // Use default options from the diff library; callers can post-process if needed
  return Diff.createPatch(
    fileName,
    oldStr,
    newStr,
    oldHeader ?? '',
    newHeader ?? '',
  );
}

export function getSimpleDiffStat(patchText: string) {
  // Minimal stat: count of lines starting with + or - (excluding headers)
  const lines = patchText.split('\n');
  let added = 0;
  let removed = 0;
  for (const l of lines) {
    if (l.startsWith('+') && !l.startsWith('+++')) added++;
    if (l.startsWith('-') && !l.startsWith('---')) removed++;
  }
  return { added, removed };
}
