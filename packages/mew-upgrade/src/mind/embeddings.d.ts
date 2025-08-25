/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
export declare class HashingEmbedder {
    private d;
    private ngram;
    private useWords;
    private key;
    constructor(d?: number, ngram?: [number, number], useWords?: boolean, seed?: number);
    embed(data: any): number[];
    private _h;
    private _indexSign;
    private addText;
    private addSequence;
    private addNumber;
    private addDict;
}
export declare class ARPredictor {
    private d;
    private lr;
    private l2;
    private initScale;
    private A_diag;
    constructor(d: number, lr?: number, l2?: number, initScale?: number);
    private _matvec;
    loss(x: number[], y: number[]): number;
    update(x: number[], y: number[], steps?: number): number;
    computePUAndUpdate(prevX: number[] | null, currY: number[]): number;
}
