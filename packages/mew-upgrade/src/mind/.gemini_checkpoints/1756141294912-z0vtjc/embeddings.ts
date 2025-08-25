/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Translation of HashingEmbedder from crisalida_lib/ADAM/mente/memory.py to TypeScript

import * as crypto from 'crypto';

// Utility function for clamping values
function clamp(x: number, lo: number = 0.0, hi: number = 1.0): number {
  try {
    return Math.max(lo, Math.min(hi, x));
  } catch (e) {
    return lo;
  }
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

    // Generate a deterministic 16-byte key from the seed
    const rng = new (class extends crypto.webcrypto.Crypto {
      private seed: number;
      constructor(seed: number) {
        super();
        this.seed = seed;
      }
      getRandomValues<T extends ArrayBufferView | null>(array: T): T {
        if (array === null) return null;
        const view = new DataView(array.buffer);
        for (let i = 0; i < array.byteLength; i += 4) {
          this.seed = (this.seed * 9301 + 49297) % 233280;
          view.setUint32(i, this.seed, true);
        }
        return array;
      }
    })(seed);
    const keyBuffer = new Uint8Array(16);
    rng.getRandomValues(keyBuffer);
    this.key = Buffer.from(keyBuffer);
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
      } catch (e) {
        continue;
      }
    }
  }

  private addNumber(v: number[], x: number): void {
    const bucket = x !== 0 ? Math.sign(x) * Math.floor(Math.log1p(Math.abs(x) + 1e-12)) : 0;
    const [idx, s] = this._indexSign(`num|${bucket}`);
    v[idx] += s * (1.0 + Math.min(1.0, Math.abs(x)));
  }

  private addDict(v: number[], d: Record<string, any>): void {
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
