/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Memory status enumeration
 */
export enum MemoryStatus {
  ACTIVE = 'active',
  ARCHIVED = 'archived',
  DELETED = 'deleted',
}

/**
 * Memory interface representing a stored memory item
 */
export interface Memory {
  id: string;
  summary: string;
  content: string;
  metadata: {
    timestamp: Date;
    source?: string;
    tags?: string[];
    importance?: number;
    type?: string;
    context?: Record<string, unknown>;
  };
  status: MemoryStatus;
  relevanceScore?: number;
  embeddings?: number[];
}
