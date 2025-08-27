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
import { EmbeddingCacheService } from '../services/embeddingCacheService.js';

/**
 * High-performance embedding service that uses Gemini's embedding model
 * with intelligent caching optimizations.
 */
export class RAGGeminiEmbeddingService extends RAGEmbeddingService {
  private readonly modelInfo: EmbeddingModelInfo;
  private readonly cacheService: EmbeddingCacheService;

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

    // Initialize cache service
    this.cacheService = new EmbeddingCacheService(this.logger, {
      enabled: this.ragConfig.performance.enableCaching,
      maxSize: 10000, // Fixed cache size
      ttlMs: 24 * 60 * 60 * 1000, // 24 hours
      lruEviction: true
    });

    this.logger.info('Enhanced GeminiEmbeddingService initialized', {
      model: this.modelInfo.name,
      caching: this.ragConfig.performance.enableCaching
    });
  }

  async generateEmbedding(text: string): Promise<number[]> {
    // Try cache first
    const cached = await this.cacheService.get(text);
    if (cached) {
      return cached;
    }

    // Generate new embedding using actual API call
    const embedding = await this.generateEmbeddingDirect(text);

    // Cache the result
    await this.cacheService.set(text, embedding);

    return embedding;
  }

  /**
   * Generate embeddings for multiple texts efficiently
   */
  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    // Check cache for all texts
    const cachedResults = await this.cacheService.getMany(texts);
    const uncachedTexts: string[] = [];

    // Identify which texts need embedding
    for (let i = 0; i < texts.length; i++) {
      if (cachedResults[i] === null) {
        uncachedTexts.push(texts[i]);
      }
    }

    // Generate embeddings for uncached texts
    let newEmbeddings: number[][] = [];
    if (uncachedTexts.length > 0) {
      newEmbeddings = await Promise.all(
        uncachedTexts.map(text => this.generateEmbeddingDirect(text))
      );
      
      // Cache new embeddings
      const cacheEntries = uncachedTexts.map((text, idx) => ({
        content: text,
        embedding: newEmbeddings[idx]
      }));
      await this.cacheService.setMany(cacheEntries);
    }

    // Combine cached and new results
    const results: number[][] = new Array(texts.length);
    let newEmbeddingIndex = 0;
    
    for (let i = 0; i < texts.length; i++) {
      if (cachedResults[i] !== null) {
        results[i] = cachedResults[i]!; // Non-null assertion since we checked
      } else {
        results[i] = newEmbeddings[newEmbeddingIndex++];
      }
    }

    this.logger.debug('Batch embedding generation completed', {
      total: texts.length,
      cached: texts.length - uncachedTexts.length,
      generated: uncachedTexts.length
    });

    return results;
  }

  /**
   * Direct embedding generation using Gemini API
   */
  private async generateEmbeddingDirect(text: string): Promise<number[]> {
    try {
      const embedding = await this.generateEmbeddingWithRetry(text);
      return embedding;
    } catch (error) {
      this.logger.error('Failed to generate embedding:', error);
      throw new RAGEmbeddingError(
        `Embedding generation failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async generateEmbeddingWithRetry(
    text: string,
    maxRetries: number = 3
  ): Promise<number[]> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        this.logger.debug(`Generating embedding (attempt ${attempt}/${maxRetries})`, {
          textLength: text.length,
          model: this.modelInfo.name,
        });

        // Use real Gemini client for embedding generation
        const geminiClient = this.config.getGeminiClient();
        const embedding = await geminiClient.generateEmbedding([text]);

        this.logger.debug('Embedding generated successfully', {
          dimension: embedding.length,
          attempt,
        });

        return embedding[0]; // Return first embedding from batch result
      } catch (error) {
        lastError = error as Error;
        this.logger.warn(`Embedding generation attempt ${attempt} failed:`, {
          error: lastError.message,
          attempt,
          willRetry: attempt < maxRetries,
        });

        if (attempt < maxRetries) {
          // Exponential backoff
          const delay = Math.pow(2, attempt - 1) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    // If we reach here, all retries failed
    throw lastError || new Error('Unknown error during embedding generation');
  }

  getModelInfo(): EmbeddingModelInfo {
    return this.modelInfo;
  }

  getEmbeddingDimension(): number {
    return this.modelInfo.dimension;
  }

  /**
   * Get performance statistics
   */
  getStats() {
    return {
      modelInfo: this.modelInfo,
      cache: this.cacheService.getStats()
    };
  }

  /**
   * Clean up resources
   */
  async cleanup(): Promise<void> {
    await this.cacheService.cleanup();
  }
}
