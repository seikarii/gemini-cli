/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  RAGVectorStore,
  RAGChunk,
  ScoredChunk,
  QueryFilters,
  VectorStoreStats,
  VectorStoreConfig,
  RAGVectorStoreError,
} from '../types.js';
import { RAGLogger } from '../logger.js';

/**
 * Simple in-memory vector store implementation.
 * This is primarily for development and testing purposes.
 * For production use, consider using a persistent vector database.
 */
export class RAGMemoryVectorStore extends RAGVectorStore {
  private chunks: Map<string, RAGChunk>;
  private initialized = false;
  private hybridWeights: {
    semantic: number;
    keyword: number;
    graph: number;
    recency: number;
  };

  constructor(
    config: VectorStoreConfig,
    private logger: RAGLogger,
    hybridWeights?: {
      semantic: number;
      keyword: number;
      graph: number;
      recency: number;
    }
  ) {
    super(config);
    this.chunks = new Map();
    
    // Use provided weights or sensible defaults
    this.hybridWeights = hybridWeights || {
      semantic: 0.7,
      keyword: 0.2,
      graph: 0.05,
      recency: 0.05
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    this.logger.info('Initializing in-memory vector store...');
    this.chunks.clear();
    this.initialized = true;
    this.logger.info('In-memory vector store initialized');
  }

  async addChunks(chunks: RAGChunk[]): Promise<void> {
    this.ensureInitialized();

    const startTime = Date.now();
    let addedCount = 0;

    for (const chunk of chunks) {
      if (!chunk.id) {
        throw new RAGVectorStoreError('Chunk must have an ID');
      }
      
      this.chunks.set(chunk.id, chunk);
      addedCount++;
    }

    const duration = Date.now() - startTime;
    this.logger.debug(`Added ${addedCount} chunks in ${duration}ms`);
  }

  async updateChunks(chunks: RAGChunk[]): Promise<void> {
    this.ensureInitialized();

    const startTime = Date.now();
    let updatedCount = 0;

    for (const chunk of chunks) {
      if (!chunk.id) {
        throw new RAGVectorStoreError('Chunk must have an ID');
      }
      
      if (this.chunks.has(chunk.id)) {
        this.chunks.set(chunk.id, chunk);
        updatedCount++;
      }
    }

    const duration = Date.now() - startTime;
    this.logger.debug(`Updated ${updatedCount} chunks in ${duration}ms`);
  }

  async deleteChunks(chunkIds: string[]): Promise<void> {
    this.ensureInitialized();

    const startTime = Date.now();
    let deletedCount = 0;

    for (const id of chunkIds) {
      if (this.chunks.delete(id)) {
        deletedCount++;
      }
    }

    const duration = Date.now() - startTime;
    this.logger.debug(`Deleted ${deletedCount} chunks in ${duration}ms`);
  }

  async search(
    query: string,
    embedding: number[],
    filters?: QueryFilters,
    limit?: number
  ): Promise<ScoredChunk[]> {
    this.ensureInitialized();

    const startTime = Date.now();
    const maxResults = limit || 10;
    const results: ScoredChunk[] = [];

    // Convert chunks to array for processing
    const chunksArray = Array.from(this.chunks.values());

    // Apply filters
    const filteredChunks = this.applyFilters(chunksArray, filters);

    // Calculate similarity scores
    for (const chunk of filteredChunks) {
      if (!chunk.embedding) {
        continue; // Skip chunks without embeddings
      }

      const semanticScore = this.cosineSimilarity(embedding, chunk.embedding);
      const keywordScore = this.keywordSimilarity(query, chunk.content);
      const recencyScore = this.calculateRecencyScore(chunk.timestamp);
      const graphScore = this.calculateGraphScore(chunk); // Basic implementation
      
      // Combine scores using configured weights
      const combinedScore = 
        semanticScore * this.hybridWeights.semantic +
        keywordScore * this.hybridWeights.keyword +
        graphScore * this.hybridWeights.graph +
        recencyScore * this.hybridWeights.recency;

      if (combinedScore >= 0.7) {
        results.push({
          chunk,
          score: combinedScore,
          scoreBreakdown: {
            semantic: semanticScore,
            keyword: keywordScore,
            graph: 0, // Not implemented in memory store
            recency: recencyScore,
            quality: chunk.metadata.quality?.relevance || 0.5,
          },
          explanation: `Combined score: ${combinedScore.toFixed(3)} (semantic: ${semanticScore.toFixed(3)}, keyword: ${keywordScore.toFixed(3)}, recency: ${recencyScore.toFixed(3)})`,
          highlights: this.extractHighlights(query, chunk.content),
        });
      }
    }

    // Sort by score and limit results
    results.sort((a, b) => b.score - a.score);
    const limitedResults = results.slice(0, maxResults);

    const duration = Date.now() - startTime;
    this.logger.debug(
      `Search completed: ${limitedResults.length}/${results.length} results in ${duration}ms`
    );

    return limitedResults;
  }

  async getChunk(id: string): Promise<RAGChunk | null> {
    this.ensureInitialized();
    return this.chunks.get(id) || null;
  }

  async listChunks(filters?: QueryFilters): Promise<string[]> {
    this.ensureInitialized();
    
    const chunksArray = Array.from(this.chunks.values());
    const filteredChunks = this.applyFilters(chunksArray, filters);
    
    return filteredChunks.map(chunk => chunk.id);
  }

  async getStats(): Promise<VectorStoreStats> {
    this.ensureInitialized();

    const chunks = Array.from(this.chunks.values());
    const totalSize = chunks.reduce((sum, chunk) => sum + chunk.content.length, 0);

    return {
      totalChunks: chunks.length,
      indexSize: totalSize, // In memory, index size equals content size
      lastUpdated: new Date().toISOString(),
    };
  }

  async close(): Promise<void> {
    this.logger.info('Closing in-memory vector store...');
    this.chunks.clear();
    this.initialized = false;
    this.logger.info('In-memory vector store closed');
  }

  // Private helper methods

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new RAGVectorStoreError('Vector store not initialized');
    }
  }

  private applyFilters(chunks: RAGChunk[], filters?: QueryFilters): RAGChunk[] {
    if (!filters) {
      return chunks;
    }

    return chunks.filter(chunk => {
      // Filter by chunk types
      if (filters.chunkTypes && !filters.chunkTypes.includes(chunk.type)) {
        return false;
      }

      // Filter by languages
      if (filters.languages && chunk.language && !filters.languages.includes(chunk.language)) {
        return false;
      }

      // Filter by file patterns
      if (filters.filePatterns && chunk.source.type === 'file') {
        const matchesPattern = filters.filePatterns.some(pattern => {
          const regex = new RegExp(pattern.replace(/\*/g, '.*'));
          return regex.test(chunk.source.id);
        });
        if (!matchesPattern) {
          return false;
        }
      }

      // Filter by time range
      if (filters.timeRange) {
        const chunkTime = new Date(chunk.timestamp);
        if (filters.timeRange.start && chunkTime < new Date(filters.timeRange.start)) {
          return false;
        }
        if (filters.timeRange.end && chunkTime > new Date(filters.timeRange.end)) {
          return false;
        }
      }

      // Filter by quality threshold
      if (filters.minQuality && chunk.metadata.quality) {
        if (chunk.metadata.quality.relevance < filters.minQuality) {
          return false;
        }
      }

      return true;
    });
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      return 0;
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  private keywordSimilarity(query: string, content: string): number {
    const queryWords = this.tokenize(query.toLowerCase());
    const contentWords = this.tokenize(content.toLowerCase());
    
    if (queryWords.length === 0) {
      return 0;
    }

    const matches = queryWords.filter(word => contentWords.includes(word));
    return matches.length / queryWords.length;
  }

  private calculateRecencyScore(timestamp: string): number {
    const chunkTime = new Date(timestamp).getTime();
    const now = Date.now();
    const ageInDays = (now - chunkTime) / (1000 * 60 * 60 * 24);
    
    // Exponential decay: newer chunks get higher scores
    return Math.exp(-ageInDays / 30); // 30-day half-life
  }

  private extractHighlights(query: string, content: string): string[] {
    const queryWords = this.tokenize(query.toLowerCase());
    const highlights: string[] = [];

    // Find sentences containing query words
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    for (const sentence of sentences) {
      const sentenceLower = sentence.toLowerCase();
      const hasMatch = queryWords.some(word => sentenceLower.includes(word));
      
      if (hasMatch && highlights.length < 3) { // Limit to 3 highlights
        highlights.push(sentence.trim());
      }
    }

    return highlights;
  }

  private tokenize(text: string): string[] {
    return text
      .split(/\W+/)
      .filter(word => word.length > 2) // Filter out short words
      .map(word => word.toLowerCase());
  }

  /**
   * Calculate graph score based on chunk relationships.
   * Basic implementation - can be enhanced with actual graph algorithms.
   */
  private calculateGraphScore(chunk: RAGChunk): number {
    // Basic graph scoring based on metadata relationships
    let score = 0.5; // Base score
    
    // Boost score for chunks with more connections (e.g., imports, dependencies)
    const metadata = chunk.metadata as Record<string, unknown>; // Allow access to extended metadata
    if (metadata?.dependencies && Array.isArray(metadata.dependencies)) {
      const connectionCount = metadata.dependencies.length;
      score += Math.min(connectionCount * 0.1, 0.3); // Cap at 0.3
    }
    
    // Boost score for chunks with quality indicators
    const quality = metadata?.quality as Record<string, unknown>;
    if (quality?.relevance && typeof quality.relevance === 'number') {
      score += quality.relevance * 0.2;
    }
    
    // Boost score for central components (classes, main functions)
    if (chunk.type === 'code_class' || chunk.type === 'code_module') {
      score += 0.1;
    }
    
    return Math.min(score, 1.0); // Cap at 1.0
  }
}
