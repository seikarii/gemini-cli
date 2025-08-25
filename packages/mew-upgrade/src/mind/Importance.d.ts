/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Importance Assessment Module
 * Evaluates incoming data significance for selective memory storage.
 */
import { HashingEmbedder } from './embeddings.js';
export interface SignificanceResult {
    importance: number;
    valence: number;
    arousal: number;
}
interface DataContext {
    dataType?: 'file_content' | 'tool_output' | 'user_input' | 'system_event' | 'error' | 'success';
    filePath?: string;
    toolName?: string;
    timestamp?: number;
    size?: number;
}
/**
 * Calculates the significance of incoming data for memory storage decisions.
 */
export declare function calculateDataSignificance(incomingData: any, currentProjectStateEmbeddings: number[][], currentMission: string, agentHistorySummary: string, hashingEmbedder: HashingEmbedder, context?: DataContext): SignificanceResult;
/**
 * Helper function to easily integrate with MenteOmega's memory storage.
 * Evaluates data and returns whether it should be stored based on importance threshold.
 */
export declare function shouldStoreInMemory(significance: SignificanceResult, importanceThreshold?: number): boolean;
/**
 * Helper to create enhanced memory node data with significance scores.
 */
export declare function enhanceMemoryNodeData(originalData: any, significance: SignificanceResult): any;
export {};
