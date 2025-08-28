/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Translation of HashingEmbedder from crisalida_lib/ADAM/mente/memory.py to TypeScript

import * as crypto from 'crypto';

// Utility function for clamping values
export function clamp(x: number, lo: number = 0.0, hi: number = 1.0): number {
  return Math.max(lo, Math.min(hi, x));
}

export class HashingEmbedder {
  private d: number;
  private ngram: [number, number];
  private useWords: boolean;
  private key: Buffer; // 16-byte key for blake2b

  constructor(
    d: number = 256,
    ngram: [number, number] = [3, 5],
    useWords: boolean = true,
    seed: number = 1337,
  ) {
    this.d = d;
    this.ngram = ngram;
    this.useWords = useWords;

    // Generate a deterministic 16-byte key from the seed using a simple LCG.
    // Avoid extending webcrypto.Crypto which is not extendable in TS.
    const keyBuffer = new Uint8Array(16);
    let s = seed >>> 0;
    for (let i = 0; i < keyBuffer.length; i++) {
      // LCG constants (numerical recipes)
      s = (s * 1664525 + 1013904223) >>> 0;
      keyBuffer[i] = s & 0xff;
    }
    this.key = Buffer.from(keyBuffer);
    // reference key to satisfy linters (no-op)
    void this.key;
  }

  // Main embedding method
  embed(data: any): number[] {
    const v = new Array(this.d).fill(0.0);

    if (typeof data === 'string') {
      this.addText(v, data);
    } else if (Array.isArray(data)) {
      this.addSequence(v, data);
    } else if (typeof data === 'number') {
      this.addNumber(v, data);
    } else if (typeof data === 'object' && data !== null) {
      this.addDict(v, data);
    } else {
      this.addText(v, String(data));
    }

    // Normalize
    const n = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    if (n > 0) {
      return v.map((x) => x / n);
    }
    return v; // Return as is if norm is zero
  }

  private _h(s: string): number {
    // Using blake2b for hashing, similar to Python's hashlib.blake2b
    const hash = crypto.createHash('blake2b512').update(s).digest();
    // Take first 8 bytes for consistency with Python's digest_size=8
    const eightBytes = hash.slice(0, 8);
    // Convert 8 bytes to a BigInt, then to a number (might lose precision for very large numbers)
    // For hashing, we primarily care about distribution, so this should be fine.
    return Number(BigInt('0x' + eightBytes.toString('hex')));
  }

  private _indexSign(token: string): [number, number] {
    const h = this._h(token);
    const idx = h % this.d;
    const sign = ((h >> 63) & 1) === 0 ? 1.0 : -1.0; // Check 64th bit for sign
    return [idx, sign];
  }

  private addText(v: number[], text: string): void {
    text = text.toLowerCase();
    if (this.useWords) {
      const words = text.match(/[a-záéíóúñü0-9]+/g) || [];
      for (const w of words) {
        const [idx, s] = this._indexSign('w|' + w);
        v[idx] += s;
      }
    }
    const t = `^${text}$`;
    const [lo, hi] = this.ngram;
    for (let n = lo; n <= hi; n++) {
      for (let i = 0; i < Math.max(0, t.length - n + 1); i++) {
        const ng = t.substring(i, i + n);
        const [idx, s] = this._indexSign(`c${n}|${ng}`);
        v[idx] += 0.5 * s;
      }
    }
  }

  private addSequence(v: number[], arr: any[]): void {
    for (let i = 0; i < arr.length; i++) {
      const val = arr[i];
      try {
        const valf = Number(val);
        if (isNaN(valf) || valf === 0.0) {
          continue;
        }
        const [idx, s] = this._indexSign(`idx|${i}`);
        v[idx] += s * valf;
      } catch (_e) {
        continue;
      }
    }
  }

  private addNumber(v: number[], x: number): void {
    const bucket =
      x !== 0 ? Math.sign(x) * Math.floor(Math.log1p(Math.abs(x) + 1e-12)) : 0;
    const [idx, s] = this._indexSign(`num|${bucket}`);
    v[idx] += s * (1.0 + Math.min(1.0, Math.abs(x)));
  }

  private addDict(v: number[], d: Record<string, unknown>): void {
    // Sort keys for deterministic embedding
    const sortedKeys = Object.keys(d).sort();
    for (const k of sortedKeys) {
      const val = d[k];
      if (typeof val === 'number') {
        const tok = `${k}:${val.toFixed(3)}`; // Round to 3 decimal places
        const [idx, s] = this._indexSign('kv|' + tok);
        v[idx] += s;
      } else {
        const tok = `${k}:${String(val).substring(0, 64)}`; // Truncate string values
        const [idx, s] = this._indexSign('kv|' + tok);
        v[idx] += 0.5 * s;
      }
    }
  }
}

export function cosSim(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export class ARPredictor {
  private d: number;
  private lr: number;
  private l2: number;
  private initScale: number;
  private A_diag: number[]; // Fallback to diagonal representation

  constructor(
    d: number,
    lr: number = 0.01,
    l2: number = 1e-4,
    initScale: number = 0.9,
  ) {
    this.d = d;
    this.lr = lr;
    this.l2 = l2;
    this.initScale = initScale;
    this.A_diag = new Array(this.d).fill(this.initScale);
  }

  private _matvec(x: number[]): number[] {
    // Fallback: diagonal multiply
    return x.map((val, i) => this.A_diag[i] * val);
  }

  loss(x: number[], y: number[]): number {
    try {
      const y_hat = this._matvec(x);
      const e = y_hat.map((yh, i) => yh - y[i]);
      return e.reduce((sum, ei) => sum + ei * ei, 0) / Math.max(1, e.length);
    } catch (_e) {
      return Infinity;
    }
  }

  update(x: number[], y: number[], steps: number = 1): number {
    const x_list = x.map(Number);
    const y_list = y.map(Number);

    for (let step = 0; step < steps; step++) {
      const y_hat = this._matvec(x_list);
      for (let i = 0; i < this.d; i++) {
        const err = y_hat[i] - y_list[i];
        this.A_diag[i] -=
          this.lr * (2.0 * err * x_list[i] + 2.0 * this.l2 * this.A_diag[i]);
      }
    }
    return this.loss(x_list, y_list);
  }

  computePUAndUpdate(prevX: number[] | null, currY: number[]): number {
    if (prevX === null) {
      return 0.0;
    }
    try {
      const before = this.loss(prevX, currY);
      this.update(prevX, currY, 1);
      const after = this.loss(prevX, currY);
      if (before <= 1e-9) {
        return 0.0;
      }
      const pu = (before - after) / before;
      return clamp(pu, 0.0, 1.0);
    } catch (_e) {
      return 0.0;
    }
  }
}
