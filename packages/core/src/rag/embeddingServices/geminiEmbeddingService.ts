/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  RAGEmbeddingService,
  EmbeddingModelInfo,
  RAGConfig,
  RAGEmbeddingError,
} from '../types.js';
import { Config } from '../../config/config.js';
import { RAGLogger } from '../logger.js';

/**
 * Embedding service that uses Gemini's embedding model.
 */
export class RAGGeminiEmbeddingService extends RAGEmbeddingService {
  private readonly modelInfo: EmbeddingModelInfo;
  private embeddingCache: Map<string, number[]> = new Map();

  constructor(
    private readonly config: Config,
    private readonly ragConfig: RAGConfig,
    private readonly logger: RAGLogger
  ) {
    super();
    
    this.modelInfo = {
      name: this.ragConfig.embedding.model,
      dimension: this.ragConfig.embedding.dimension,
      maxTokens: 8192, // Typical Gemini embedding limit
      isCodeSpecific: this.ragConfig.embedding.model.includes('code'),
      provider: 'google',
    };

    // Set up cache with size limit
    if (this.ragConfig.performance.enableCaching) {
      this.setupCacheEviction();
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    // Check cache first
    if (this.ragConfig.performance.enableCaching) {
      const cached = this.embeddingCache.get(text);
      if (cached) {
        this.logger.debug('Embedding cache hit');
        return cached;
      }
    }

    try {
      const embedding = await this.generateEmbeddingWithRetry(text);
      
      // Cache the result
      if (this.ragConfig.performance.enableCaching) {
        this.embeddingCache.set(text, embedding);
      }
      
      return embedding;
    } catch (error) {
      this.logger.error('Failed to generate embedding:', error);
      throw new RAGEmbeddingError(
        `Failed to generate embedding: ${(error as Error).message}`,
        error as Error
      );
    }
  }

  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const batchSize = this.ragConfig.embedding.batchSize;
    const results: number[][] = [];

    // Process in batches for efficiency
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(text => this.generateEmbedding(text))
      );
      results.push(...batchResults);
      
      this.logger.debug(`Generated embeddings for batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)}`);
    }

    return results;
  }

  getEmbeddingDimension(): number {
    return this.modelInfo.dimension;
  }

  getModelInfo(): EmbeddingModelInfo {
    return { ...this.modelInfo };
  }

  // Private methods

  private async generateEmbeddingWithRetry(text: string): Promise<number[]> {
    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= this.ragConfig.embedding.maxRetries; attempt++) {
      try {
        return await this.callEmbeddingAPI(text);
      } catch (error) {
        lastError = error as Error;
        this.logger.warn(`Embedding attempt ${attempt}/${this.ragConfig.embedding.maxRetries} failed:`, error);
        
        if (attempt < this.ragConfig.embedding.maxRetries) {
          // Exponential backoff
          const delay = Math.pow(2, attempt - 1) * 1000;
          await this.sleep(delay);
        }
      }
    }

    throw lastError || new Error('All embedding attempts failed');
  }

  private async callEmbeddingAPI(text: string): Promise<number[]> {
    // TODO: Replace with actual Gemini embedding API call
    // For now, return a mock embedding
    this.logger.debug(`Generating embedding for text (${text.length} chars)`);
    
    // Mock embedding generation - replace with actual API call
    const embedding = this.generateMockEmbedding(text);
    
    return embedding;
  }

  private generateMockEmbedding(text: string): number[] {
    // Generate a deterministic mock embedding based on text content
    // This is for development/testing purposes only
    const dimension = this.getEmbeddingDimension();
    const embedding = new Array(dimension);
    
    // Use a simple hash-based approach for consistency
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Generate normalized embedding vector
    for (let i = 0; i < dimension; i++) {
      const seed = hash + i;
      embedding[i] = Math.sin(seed) * Math.cos(seed * 0.7);
    }
    
    // Normalize the vector
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < dimension; i++) {
        embedding[i] = embedding[i] / magnitude;
      }
    }
    
    return embedding;
  }

  private setupCacheEviction(): void {
    // Simple LRU-style cache eviction
    const maxCacheSize = this.ragConfig.embedding.cacheSize;
    
    setInterval(() => {
      if (this.embeddingCache.size > maxCacheSize) {
        const entries = Array.from(this.embeddingCache.entries());
        const toDelete = entries.slice(0, Math.floor(maxCacheSize * 0.2)); // Remove 20%
        
        for (const [key] of toDelete) {
          this.embeddingCache.delete(key);
        }
        
        this.logger.debug(`Cache eviction: removed ${toDelete.length} entries`);
      }
    }, 60000); // Check every minute
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
