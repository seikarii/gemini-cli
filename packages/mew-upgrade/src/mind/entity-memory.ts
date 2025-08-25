/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Translation of EntityMemory from crisalida_lib/ADAM/mente/memory.py to TypeScript

import { MentalLaby, MemoryNodeKind } from './mental-laby.js';
import { clamp } from './embeddings.js'; // Assuming clamp is exported from embeddings.ts

export class EntityMemory {
  entityId: string;
  mind: MentalLaby;
  affectBias: number;
  valenceBias: number;
  storageBudget: number;

  constructor(
    entityId: string,
    mind: MentalLaby,
    affectBias: number = 0.0,
    valenceBias: number = 0.0,
    storageBudget: number = 100_000,
  ) {
    this.entityId = entityId;
    this.mind = mind;
    this.affectBias = affectBias;
    this.valenceBias = valenceBias;
    this.storageBudget = storageBudget;
  }

  ingest(
    data: any,
    valence: number = 0.0,
    arousal: number = 0.0,
    kind: MemoryNodeKind = 'semantic',
    salience: number = 0.5, // Importance from SignificanceResult
  ): string {
    valence = clamp(valence + this.valenceBias, -1.0, 1.0);
    arousal = clamp(arousal + this.affectBias, 0.0, 1.0);
    // this.mind.max_nodes = this.storageBudget; // MentalLaby in TS doesn't have max_nodes directly
    return this.mind.store(
      data,
      kind,
      valence,
      arousal,
      salience,
    );
  }

  recall(cue: any): any {
    return this.mind.recall(cue);
  }

  // Convenience export/import wrappers for persistence
  exportState(): object {
    // Delegate to mental laby export
    // @ts-ignore - mental laby export shape is dynamic
    return (this.mind as any).exportState ? (this.mind as any).exportState() : { nodes: [] };
  }

  importState(state: any): void {
    // Delegate import to MentalLaby if available
    if ((this.mind as any).importState) {
      (this.mind as any).importState(state);
    }
  }

  dream(mode: string = 'MIXED'): void {
    // MentalLaby in TS doesn't have dream_cycle yet, so this is a stub
    // this.mind.dream_cycle(mode);
  }

  // Omitted on_cosmic_influence and on_qualia_update for MVP
}
