/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export interface PtyProcess {
    readonly pid: number;
    onData(callback: (data: string) => void): void;
    onExit(callback: (e: {
        exitCode: number;
        signal?: number;
    }) => void): void;
    kill(signal?: string): void;
}
export interface PtyModule {
    spawn(command: string, args: string[], opts: Record<string, unknown>): PtyProcess;
}
export type PtyImplementation = {
    module: PtyModule;
    name: 'lydell-node-pty' | 'node-pty';
} | null;
export declare const getPty: () => Promise<PtyImplementation>;
