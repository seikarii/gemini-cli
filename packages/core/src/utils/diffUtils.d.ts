/**
 * Wrapper around Diff.createPatch to centralize patch formatting and options.
 */
export declare function createPatch(fileName: string, oldStr: string, newStr: string, oldHeader?: string, newHeader?: string): string;
export declare function getSimpleDiffStat(patchText: string): {
    added: number;
    removed: number;
};
