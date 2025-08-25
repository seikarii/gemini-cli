/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { MentalLaby, MemoryNodeKind } from './mental-laby.js';
export declare class EntityMemory {
    entityId: string;
    mind: MentalLaby;
    affectBias: number;
    valenceBias: number;
    storageBudget: number;
    constructor(entityId: string, mind: MentalLaby, affectBias?: number, valenceBias?: number, storageBudget?: number);
    ingest(data: any, valence?: number, arousal?: number, kind?: MemoryNodeKind, salience?: number): string;
    recall(cue: any): any;
    dream(mode?: string): void;
}
