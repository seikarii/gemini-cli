/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { ModificationSpec } from './models.js';
type ModifierResult = {
    success: boolean;
    output?: string;
    error?: string;
    modifiedText?: string;
    backupId?: string;
};
export declare class ASTModifier {
    private readonly projectOptions;
    private backups;
    constructor(projectOptions?: {
        useInMemoryFileSystem?: boolean;
    });
    private createProject;
    private createBackup;
    private restoreBackup;
    /**
     * Apply a list of modifications to a source text. Returns modified text on success,
     * or restores the backup and returns an error on failure.
     */
    applyModifications(sourceText: string, modifications: ModificationSpec[], opts?: {
        filePath?: string;
        format?: boolean;
    }): Promise<ModifierResult>;
    private applyFileLevelModification;
    private applyModificationToNode;
    private insertRelative;
    private wrapNode;
    private extractNodeAsFunction;
    private renameSymbolScoped;
    private updateMethodSignature;
    private insertStatementIntoFunction;
}
export {};
