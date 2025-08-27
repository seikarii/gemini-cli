import { RAGLogger } from '../logger.js';

/**
 * Interface for the underlying embedding generator
 */
interface EmbeddingGenerator {
  generateEmbedding(text: string): Promise<number[]>;
}

/**
 * Configuration for embedding batch processing
 */
export interface EmbeddingBatchConfig {
  /** Maximum batch size for API calls */
  maxBatchSize: number;
  /** Maximum wait time before processing partial batch (ms) */
  batchTimeoutMs: number;
  /** Maximum concurrent batches */
  maxConcurrentBatches: number;
  /** Enable adaptive batching based on API response times */
  adaptiveBatching: boolean;
  /** Target latency for adaptive batching (ms) */
  targetLatencyMs: number;
}

/**
 * Batch request for embeddings
 */
interface BatchRequest {
  /** Unique request ID */
  id: string;
  /** Content to embed */
  content: string;
  /** Promise resolver */
  resolve: (embedding: number[]) => void;
  /** Promise rejecter */
  reject: (error: Error) => void;
  /** Timestamp when added to batch */
  timestamp: number;
}

/**
 * Intelligent embedding batch processor that optimizes API calls
 * through strategic batching and adaptive load management.
 */
export class EmbeddingBatchService {
  private readonly logger: RAGLogger;
  private readonly embeddingGenerator: EmbeddingGenerator;
  private readonly config: EmbeddingBatchConfig;
  
  private currentBatch: BatchRequest[] = [];
  private batchTimer: NodeJS.Timeout | null = null;
  private activeBatches = 0;
  private requestQueue: BatchRequest[] = [];
  
  // Adaptive batching metrics
  private recentLatencies: number[] = [];
  private currentOptimalBatchSize: number;
  
  // Performance metrics
  private stats = {
    totalRequests: 0,
    totalBatches: 0,
    avgBatchSize: 0,
    avgLatency: 0,
    queueDrops: 0,
    adaptiveBatchChanges: 0
  };

  constructor(
    embeddingGenerator: EmbeddingGenerator,
    logger: RAGLogger,
    config: Partial<EmbeddingBatchConfig> = {}
  ) {
    this.embeddingGenerator = embeddingGenerator;
    this.logger = logger;
    this.config = {
      maxBatchSize: 50,
      batchTimeoutMs: 100,
      maxConcurrentBatches: 3,
      adaptiveBatching: true,
      targetLatencyMs: 500,
      ...config
    };
    
    this.currentOptimalBatchSize = this.config.maxBatchSize;
    this.logger.info('EmbeddingBatchService initialized', { config: this.config });
  }

  /**
   * Generate embedding for content with intelligent batching
   */
  async generateEmbedding(content: string): Promise<number[]> {
    return new Promise((resolve, reject) => {
      const request: BatchRequest = {
        id: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        content,
        resolve,
        reject,
        timestamp: Date.now()
      };

      this.stats.totalRequests++;
      this.addToBatch(request);
    });
  }

  /**
   * Generate embeddings for multiple contents efficiently
   */
  async generateEmbeddings(contents: string[]): Promise<number[][]> {
    // For large batches, split into optimal sizes
    if (contents.length <= this.currentOptimalBatchSize) {
      return Promise.all(contents.map(content => this.generateEmbedding(content)));
    }

    // Process in chunks
    const results: number[][] = [];
    for (let i = 0; i < contents.length; i += this.currentOptimalBatchSize) {
      const chunk = contents.slice(i, i + this.currentOptimalBatchSize);
      const chunkResults = await Promise.all(
        chunk.map(content => this.generateEmbedding(content))
      );
      results.push(...chunkResults);
    }

    return results;
  }

  /**
   * Get current service statistics
   */
  getStats() {
    return {
      ...this.stats,
      currentBatchSize: this.currentBatch.length,
      queueSize: this.requestQueue.length,
      activeBatches: this.activeBatches,
      optimalBatchSize: this.currentOptimalBatchSize,
      avgLatency: this.recentLatencies.length > 0 
        ? this.recentLatencies.reduce((a, b) => a + b, 0) / this.recentLatencies.length 
        : 0
    };
  }

  /**
   * Force process current batch (useful for testing or shutdown)
   */
  async flush(): Promise<void> {
    if (this.currentBatch.length > 0) {
      await this.processBatch();
    }
    
    // Process any remaining queued requests
    while (this.requestQueue.length > 0 && this.activeBatches < this.config.maxConcurrentBatches) {
      this.fillCurrentBatch();
      if (this.currentBatch.length > 0) {
        await this.processBatch();
      }
    }
  }

  private addToBatch(request: BatchRequest): void {
    // Check if we're at capacity
    if (this.activeBatches >= this.config.maxConcurrentBatches) {
      this.requestQueue.push(request);
      this.logger.debug('Request queued due to batch concurrency limit', {
        queueSize: this.requestQueue.length
      });
      return;
    }

    this.currentBatch.push(request);

    // Start batch timer if this is the first request
    if (this.currentBatch.length === 1 && !this.batchTimer) {
      this.batchTimer = setTimeout(() => this.processBatch(), this.config.batchTimeoutMs);
    }

    // Process immediately if batch is full
    if (this.currentBatch.length >= this.currentOptimalBatchSize) {
      this.processBatch();
    }
  }

  private async processBatch(): Promise<void> {
    if (this.currentBatch.length === 0) return;

    // Clear timer and prepare batch
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }

    const batch = this.currentBatch;
    this.currentBatch = [];
    this.activeBatches++;

    const startTime = Date.now();
    
    try {
      this.logger.debug('Processing embedding batch', { 
        size: batch.length,
        activeBatches: this.activeBatches 
      });

      // Process all embeddings in the batch
      const embeddings = await Promise.all(
        batch.map(request => this.embeddingGenerator.generateEmbedding(request.content))
      );

      // Resolve all promises
      batch.forEach((request, index) => {
        request.resolve(embeddings[index]);
      });

      // Update statistics
      const latency = Date.now() - startTime;
      this.updateStats(batch.length, latency);

      this.logger.debug('Batch processed successfully', {
        size: batch.length,
        latency,
        avgLatency: this.stats.avgLatency
      });

    } catch (error) {
      this.logger.error('Batch processing failed', { error, batchSize: batch.length });
      
      // Reject all promises in the batch
      batch.forEach(request => {
        request.reject(error as Error);
      });
    } finally {
      this.activeBatches--;
      
      // Process next batch from queue
      if (this.requestQueue.length > 0 && this.activeBatches < this.config.maxConcurrentBatches) {
        this.fillCurrentBatch();
        if (this.currentBatch.length > 0) {
          // Process after a small delay to allow more requests to accumulate
          setTimeout(() => this.processBatch(), 10);
        }
      }
    }
  }

  private fillCurrentBatch(): void {
    const availableSlots = this.currentOptimalBatchSize;
    const requestsToMove = Math.min(availableSlots, this.requestQueue.length);
    
    for (let i = 0; i < requestsToMove; i++) {
      const request = this.requestQueue.shift();
      if (request) {
        this.currentBatch.push(request);
      }
    }
  }

  private updateStats(batchSize: number, latency: number): void {
    this.stats.totalBatches++;
    this.stats.avgBatchSize = (
      (this.stats.avgBatchSize * (this.stats.totalBatches - 1) + batchSize) / 
      this.stats.totalBatches
    );
    this.stats.avgLatency = (
      (this.stats.avgLatency * (this.stats.totalBatches - 1) + latency) / 
      this.stats.totalBatches
    );

    // Update adaptive batching metrics
    this.recentLatencies.push(latency);
    if (this.recentLatencies.length > 20) {
      this.recentLatencies.shift(); // Keep only recent samples
    }

    // Adjust batch size based on latency if adaptive batching is enabled
    if (this.config.adaptiveBatching) {
      this.adjustOptimalBatchSize(latency);
    }
  }

  private adjustOptimalBatchSize(latency: number): void {
    const targetLatency = this.config.targetLatencyMs;
    const tolerance = 0.2; // 20% tolerance

    if (latency > targetLatency * (1 + tolerance) && this.currentOptimalBatchSize > 1) {
      // Latency too high, reduce batch size
      this.currentOptimalBatchSize = Math.max(1, Math.floor(this.currentOptimalBatchSize * 0.8));
      this.stats.adaptiveBatchChanges++;
      
      this.logger.debug('Reduced optimal batch size due to high latency', {
        newSize: this.currentOptimalBatchSize,
        latency,
        targetLatency
      });
    } else if (latency < targetLatency * (1 - tolerance) && 
               this.currentOptimalBatchSize < this.config.maxBatchSize) {
      // Latency acceptable, try increasing batch size
      this.currentOptimalBatchSize = Math.min(
        this.config.maxBatchSize, 
        Math.floor(this.currentOptimalBatchSize * 1.2)
      );
      this.stats.adaptiveBatchChanges++;
      
      this.logger.debug('Increased optimal batch size due to low latency', {
        newSize: this.currentOptimalBatchSize,
        latency,
        targetLatency
      });
    }
  }
}
