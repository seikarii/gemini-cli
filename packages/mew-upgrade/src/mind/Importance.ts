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
import { cosSim, clamp } from './embeddings.js'; // Import from embeddings.ts

export interface SignificanceResult {
  importance: number; // 0.0-1.0 (higher = more important)
  valence: number;    // -1.0 to 1.0 (negative = bad/error, positive = good/success)
  arousal: number;    // 0.0-1.0 (higher = more emotionally intense/urgent)
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
export function calculateDataSignificance(
  incomingData: any,
  currentProjectStateEmbeddings: number[][],
  currentMission: string,
  agentHistorySummary: string,
  hashingEmbedder: HashingEmbedder,
  context?: DataContext
): SignificanceResult {
  
  // Generate embedding for incoming data
  const dataEmbedding = hashingEmbedder.embed(incomingData);
  const missionEmbedding = hashingEmbedder.embed(currentMission);
  const historyEmbedding = hashingEmbedder.embed(agentHistorySummary);
  
  // Calculate base components
  const missionRelevance = calculateMissionRelevance(dataEmbedding, missionEmbedding);
  const novelty = calculateNovelty(dataEmbedding, currentProjectStateEmbeddings);
  const criticalityScore = calculateCriticality(incomingData, context);
  const temporalRelevance = calculateTemporalRelevance(context);
  const contextualSignals = extractContextualSignals(incomingData, context);
  
  // Combine factors for final importance score
  const importance = computeImportance(
    missionRelevance,
    novelty,
    criticalityScore,
    temporalRelevance,
    contextualSignals
  );
  
  // Calculate emotional dimensions
  const valence = calculateValence(incomingData, context, criticalityScore);
  const arousal = calculateArousal(criticalityScore, novelty, contextualSignals);
  
  return {
    importance: clamp(importance, 0, 1),
    valence: clamp(valence, -1, 1),
    arousal: clamp(arousal, 0, 1)
  };
}

/**
 * Measures how relevant the data is to the current mission.
 */
function calculateMissionRelevance(dataEmbedding: number[], missionEmbedding: number[]): number {
  const similarity = cosSim(dataEmbedding, missionEmbedding); // Use cosSim
  // Transform similarity to emphasize high relevance
  return Math.pow(Math.max(0, similarity), 1.5);
}

/**
 * Measures how novel/new the information is compared to existing project state.
 */
function calculateNovelty(dataEmbedding: number[], projectStateEmbeddings: number[][]): number {
  if (projectStateEmbeddings.length === 0) return 1.0; // Everything is novel if no history
  
  // Find maximum similarity with existing memories
  const maxSimilarity = Math.max(
    ...projectStateEmbeddings.map(embedding => 
      cosSim(dataEmbedding, embedding) // Use cosSim
    )
  );
  
  // Novelty is inverse of similarity (high similarity = low novelty)
  const novelty = 1.0 - Math.max(0, maxSimilarity);
  
  // Apply curve to make highly novel items stand out more
  return Math.pow(novelty, 0.7);
}

/**
 * Assesses criticality based on data content and context.
 */
function calculateCriticality(data: any, context?: DataContext): number {
  let score = 0.0;
  const dataStr = String(data).toLowerCase();
  
  // Error indicators (high criticality)
  if (hasErrorSignals(dataStr)) score += 0.8;
  
  // Success indicators (moderate-high criticality)
  if (hasSuccessSignals(dataStr)) score += 0.6;
  
  // Change indicators (moderate criticality)
  if (hasChangeSignals(dataStr)) score += 0.4;
  
  // Critical file paths
  if (context?.filePath && isCriticalFile(context.filePath)) score += 0.5;
  
  // Important tools
  if (context?.toolName && isCriticalTool(context.toolName)) score += 0.3;
  
  // Data type criticality
  if (context?.dataType) {
    switch (context.dataType) {
      case 'error': score += 0.9; break;
      case 'success': score += 0.7; break;
      case 'user_input': score += 0.6; break;
      case 'system_event': score += 0.4; break;
      case 'file_content': score += 0.3; break;
      case 'tool_output': score += 0.2; break;
    }
  }
  
  return Math.min(1.0, score);
}

/**
 * Calculates temporal relevance (recency and frequency considerations).
 */
function calculateTemporalRelevance(context?: DataContext): number {
  if (!context?.timestamp) return 0.5; // Neutral if no timestamp
  
  const now = Date.now();
  const ageMs = now - context.timestamp;
  const ageHours = ageMs / (1000 * 60 * 60);
  
  // Recency decay curve - more recent data is more relevant
  if (ageHours < 1) return 1.0;        // Very recent
  if (ageHours < 6) return 0.8;        // Recent
  if (ageHours < 24) return 0.6;       // Today
  if (ageHours < 168) return 0.4;      // This week
  return 0.2;                          // Older
}

/**
 * Extracts contextual signals that affect importance.
 */
function extractContextualSignals(data: any, context?: DataContext): number {
  let signals = 0.0;
  const dataStr = String(data).toLowerCase();
  
  // Keywords that suggest high importance
  const importantKeywords = [
    'critical', 'urgent', 'important', 'breaking', 'failed', 'error',
    'success', 'completed', 'fixed', 'resolved', 'breaking change',
    'security', 'performance', 'bug', 'feature', 'release'
  ];
  
  for (const keyword of importantKeywords) {
    if (dataStr.includes(keyword)) {
      signals += 0.1;
    }
  }
  
  // File size considerations (very large or very small files might be important)
  if (context?.size) {
    if (context.size > 100000 || context.size < 100) { // Very large or very small
      signals += 0.2;
    }
  }
  
  return Math.min(1.0, signals);
}

/**
 * Computes final importance score by combining all factors.
 */
function computeImportance(
  missionRelevance: number,
  novelty: number,
  criticality: number,
  temporalRelevance: number,
  contextualSignals: number
): number {
  // Weighted combination - criticality and mission alignment are most important
  const weightedScore = (
    missionRelevance * 0.35 +    // Mission alignment is crucial
    criticality * 0.30 +         // Errors/successes are important
    novelty * 0.20 +             // New information matters
    temporalRelevance * 0.10 +   // Recent data is more relevant
    contextualSignals * 0.05     // Contextual hints
  );
  
  // Apply boost for highly critical items regardless of other factors
  if (criticality > 0.8) {
    return Math.max(weightedScore, 0.8);
  }
  
  return weightedScore;
}

/**
 * Calculates emotional valence (positive/negative).
 */
function calculateValence(data: any, context?: DataContext, criticality: number): number {
  const dataStr = String(data).toLowerCase();
  let valence = 0.0;
  
  // Negative indicators
  if (hasErrorSignals(dataStr) || context?.dataType === 'error') {
    valence -= 0.8;
  }
  
  // Positive indicators
  if (hasSuccessSignals(dataStr) || context?.dataType === 'success') {
    valence += 0.7;
  }
  
  // Neutral adjustment based on criticality
  if (criticality > 0.6 && Math.abs(valence) < 0.3) {
    valence = valence < 0 ? -0.3 : 0.3; // Ensure critical items have some emotional weight
  }
  
  return valence;
}

/**
 * Calculates emotional arousal (intensity/urgency).
 */
function calculateArousal(criticality: number, novelty: number, contextualSignals: number): number {
  // High criticality = high arousal
  // High novelty = moderate arousal
  // Strong contextual signals = moderate arousal
  return Math.min(1.0, criticality * 0.6 + novelty * 0.3 + contextualSignals * 0.1);
}

// === Utility Functions ===

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hasErrorSignals(text: string): boolean {
  const errorPatterns = [
    'error', 'failed', 'failure', 'exception', 'crash', 'bug',
    'broken', 'issue', 'problem', 'warning', 'fatal', 'critical error'
  ];
  return errorPatterns.some(pattern => text.includes(pattern));
}

function hasSuccessSignals(text: string): boolean {
  const successPatterns = [
    'success', 'completed', 'done', 'finished', 'resolved', 'fixed',
    'working', 'passed', 'ok', 'successful', 'achieved', 'accomplished'
  ];
  return successPatterns.some(pattern => text.includes(pattern));
}

function hasChangeSignals(text: string): boolean {
  const changePatterns = [
    'modified', 'changed', 'updated', 'added', 'removed', 'deleted',
    'created', 'new', 'edit', 'patch', 'diff'
  ];
  return changePatterns.some(pattern => text.includes(pattern));
}

function isCriticalFile(filePath: string): boolean {
  const criticalPatterns = [
    'package.json', 'tsconfig.json', '.env', 'config',
    'index.', 'main.', 'app.', 'server.',
    '/src/', '/lib/', '/core/'
  ];
  return criticalPatterns.some(pattern => filePath.includes(pattern));
}

function isCriticalTool(toolName: string): boolean {
  const criticalTools = [
    'git', 'npm', 'build', 'test', 'deploy', 'compile',
    'lint', 'format', 'typecheck'
  ];
  return criticalTools.some(tool => toolName.includes(tool));
}

// === Integration Helper ===

/**
 * Helper function to easily integrate with MenteOmega's memory storage.
 * Evaluates data and returns whether it should be stored based on importance threshold.
 */
export function shouldStoreInMemory(
  significance: SignificanceResult,
  importanceThreshold: number = 0.3
): boolean {
  return significance.importance >= importanceThreshold;
}

/**
 * Helper to create enhanced memory node data with significance scores.
 */
export function enhanceMemoryNodeData(
  originalData: any,
  significance: SignificanceResult
): any {
  return {
    ...originalData,
    _significance: significance,
    _timestamp: Date.now()
  };
}
