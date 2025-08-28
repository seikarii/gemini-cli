/**
 * @fileoverview Embedding Generation Worker for handling text embeddings in parallel
 * @version 1.0.0
 * @license MIT
 */

import { isMainThread, parentPort } from 'node:worker_threads';

import {
  WorkerMessage,
  WorkerResult,
  EmbeddingInput,
  EmbeddingOutput,
  WorkerTask,
} from '../WorkerInterfaces.js';

/**
 * Simple embedding generation using TF-IDF approach
 * This is a basic implementation for demonstration purposes
 * In production, you would use more sophisticated models
 */
class EmbeddingWorker {
  private readonly defaultDimensions = 512;
  private readonly vocabulary: Map<string, number> = new Map();
  private readonly idfCache: Map<string, number> = new Map();

  /**
   * Generate text embeddings
   */
  async generateEmbedding(input: EmbeddingInput): Promise<EmbeddingOutput> {
    const startTime = Date.now();
    
    try {
      const text = input.text.toLowerCase().trim();
      if (!text) {
        throw new Error('Input text is empty');
      }

      const dimensions = input.options?.dimensions || this.defaultDimensions;
      const normalize = input.options?.normalize !== false;
      const _batchSize = input.options?.batchSize || 1;
      
      // Tokenize text
      const tokens = this.tokenize(text);
      
      if (tokens.length === 0) {
        throw new Error('No valid tokens found in input text');
      }

      // Generate embeddings using TF-IDF approach
      const embeddings = this.computeTfIdfEmbedding(tokens, dimensions);
      
      // Normalize if requested
      const finalEmbeddings = normalize ? this.normalizeVector(embeddings) : embeddings;
      
      // Calculate processing time
      const processingTime = Date.now() - startTime;
      
      return {
        embeddings: finalEmbeddings,
        metadata: {
          model: input.model || 'tf-idf',
          dimensions: finalEmbeddings.length,
          tokenCount: tokens.length,
          processingTime,
        },
      };
    } catch (error) {
      throw new Error(`Embedding generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Tokenize text into words
   */
  private tokenize(text: string): string[] {
    // Basic tokenization - split by whitespace and punctuation
    return text
      .replace(/[^\w\s]/g, ' ') // Replace punctuation with spaces
      .split(/\s+/) // Split by whitespace
      .filter(token => token.length > 0) // Remove empty tokens
      .filter(token => token.length > 2) // Remove very short tokens
      .filter(token => !/^\d+$/.test(token)); // Remove pure numbers
  }

  /**
   * Compute TF-IDF based embedding
   */
  private computeTfIdfEmbedding(tokens: string[], dimensions: number): number[] {
    // Build term frequency map
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
    }

    // Update vocabulary and calculate TF-IDF scores
    const tfidfScores = new Map<string, number>();
    
    for (const [term, freq] of termFreq) {
      // Add to vocabulary if new
      if (!this.vocabulary.has(term)) {
        this.vocabulary.set(term, this.vocabulary.size);
      }
      
      // Calculate TF (term frequency)
      const tf = freq / tokens.length;
      
      // Calculate IDF (inverse document frequency) - simplified
      // In a real implementation, this would be based on a large corpus
      const idf = this.getOrCalculateIdf(term);
      
      // Calculate TF-IDF score
      tfidfScores.set(term, tf * idf);
    }

    // Create embedding vector
    const embedding = new Array<number>(dimensions).fill(0);
    
    // Map TF-IDF scores to embedding dimensions using hashing
    for (const [term, score] of tfidfScores) {
      const hash = this.hashToDimension(term, dimensions);
      embedding[hash] += score;
    }

    return embedding;
  }

  /**
   * Get or calculate IDF for a term
   */
  private getOrCalculateIdf(term: string): number {
    if (this.idfCache.has(term)) {
      return this.idfCache.get(term)!;
    }

    // Simplified IDF calculation based on term characteristics
    let idf = 1.0;
    
    // Common words get lower IDF scores
    if (this.isCommonWord(term)) {
      idf = 0.5;
    } else if (term.length > 8) {
      // Longer words are typically more specific
      idf = 2.0;
    } else if (term.length > 5) {
      idf = 1.5;
    }

    // Add some randomness based on the term itself for consistency
    const termHash = this.simpleHash(term);
    idf += (termHash % 100) / 200; // Add 0.0 to 0.5

    this.idfCache.set(term, idf);
    return idf;
  }

  /**
   * Check if a word is common (simplified approach)
   */
  private isCommonWord(word: string): boolean {
    const commonWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
    ]);
    
    return commonWords.has(word.toLowerCase());
  }

  /**
   * Hash a term to a dimension index
   */
  private hashToDimension(term: string, dimensions: number): number {
    const hash = this.simpleHash(term);
    return hash % dimensions;
  }

  /**
   * Simple hash function for strings
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Normalize vector to unit length
   */
  private normalizeVector(vector: number[]): number[] {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    
    if (magnitude === 0) {
      return vector; // Return as-is if zero vector
    }
    
    return vector.map(val => val / magnitude);
  }

  /**
   * Health check for the worker
   */
  async healthCheck(): Promise<boolean> {
    try {
      // Test embedding generation with a simple phrase
      const testInput: EmbeddingInput = {
        text: 'health check test',
        options: { dimensions: 64 },
      };
      
      const result = await this.generateEmbedding(testInput);
      return result.embeddings.length === 64;
    } catch (_error) {
      return false;
    }
  }

  /**
   * Get worker statistics
   */
  getStats(): {
    vocabularySize: number;
    idfCacheSize: number;
    memoryUsage: NodeJS.MemoryUsage;
  } {
    return {
      vocabularySize: this.vocabulary.size,
      idfCacheSize: this.idfCache.size,
      memoryUsage: process.memoryUsage(),
    };
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.vocabulary.clear();
    this.idfCache.clear();
  }
}

// Worker thread execution
if (!isMainThread && parentPort) {
  const worker = new EmbeddingWorker();

  parentPort.on('message', async (message: WorkerMessage) => {
    try {
      if (message.type === 'task' && message.payload) {
        const task = message.payload as WorkerTask<EmbeddingInput>;
        const result = await worker.generateEmbedding(task.input);
        
        const response: WorkerResult<EmbeddingOutput> = {
          taskId: task.id,
          success: true,
          result,
          executionTime: Date.now() - (task.metadata?.submittedAt ? Number(task.metadata.submittedAt) : Date.now()),
          workerId: 'embedding-generator',
        };
        
        parentPort?.postMessage({
          type: 'result',
          taskId: task.id,
          payload: response,
        });
      } else if (message.type === 'ping') {
        // Health check
        const healthy = await worker.healthCheck();
        const stats = worker.getStats();
        
        parentPort?.postMessage({
          type: 'pong',
          payload: {
            healthy,
            stats,
            uptime: process.uptime(),
          },
        });
      } else if (message.type === 'shutdown') {
        worker.cleanup();
        process.exit(0);
      }
    } catch (error) {
      const errorResponse: WorkerResult<EmbeddingOutput> = {
        taskId: 'unknown',
        success: false,
        error: {
          message: error instanceof Error ? error.message : String(error),
          code: 'EMBEDDING_GENERATION_ERROR',
          stack: error instanceof Error ? error.stack : undefined,
        },
        executionTime: 0,
        workerId: 'embedding-generator',
      };
      
      parentPort?.postMessage({
        type: 'error',
        payload: errorResponse,
      });
    }
  });

  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    parentPort?.postMessage({
      type: 'error',
      error: {
        message: error.message,
        stack: error.stack,
      },
    });
    process.exit(1);
  });

  // Signal ready
  parentPort.postMessage({ type: 'ping' });
}

export { EmbeddingWorker };
