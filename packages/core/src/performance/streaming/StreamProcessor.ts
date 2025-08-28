/**
 * @fileoverview Core stream processor implementation with backpressure management
 * and efficient memory usage for handling large files and data streams
 */

import { Readable, Transform } from 'stream';
import { createReadStream, promises as fs } from 'fs';
import { createHash } from 'crypto';
import { EventEmitter } from 'events';
import { pipeline } from 'stream/promises';
import { 
  IStreamProcessor, 
  IStreamConfig, 
  IStreamResult, 
  IStreamChunk, 
  IStreamStats,
  IBackpressureManager,
  StreamMode,
  StreamPriority,
  DEFAULT_STREAM_CONFIGS
} from './StreamInterfaces.js';

/**
 * Core stream processor with comprehensive backpressure management
 */
export class StreamProcessor<TInput = Buffer | string, TOutput = Buffer | string> 
  extends EventEmitter implements IStreamProcessor<TInput, TOutput> {
  
  private readonly backpressureManager: BackpressureManager;
  private stats: IStreamStats;
  private isProcessing = false;
  private isPaused = false;
  private shouldStop = false;
  private currentChunks = new Map<string, IStreamChunk<TInput>>();

  constructor() {
    super();
    this.backpressureManager = new BackpressureManager();
    this.stats = this.createInitialStats();
  }

  /**
   * Process a readable stream with chunked processing and backpressure management
   */
  async processStream(input: Readable, config: IStreamConfig): Promise<IStreamResult<TOutput>> {
    if (this.isProcessing) {
      throw new Error('Stream processor is already processing a stream');
    }

    this.isProcessing = true;
    this.shouldStop = false;
    this.stats = this.createInitialStats();
    this.stats.startTime = Date.now();

    try {
      const result = await this.processStreamInternal(input, config);
      this.stats.endTime = Date.now();
      
      this.emit('processing-complete', result);
      return result;
    } catch (error) {
      this.emit('processing-error', error);
      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Process a file as a stream
   */
  async processFile(filePath: string, config: IStreamConfig): Promise<IStreamResult<TOutput>> {
    const stats = await fs.stat(filePath);
    if (!stats.isFile()) {
      throw new Error(`Path is not a file: ${filePath}`);
    }

    const stream = createReadStream(filePath, { 
      highWaterMark: config.chunkSize,
      encoding: config.mode === StreamMode.REAL_TIME ? 'utf8' : undefined
    });

    return this.processStream(stream, config);
  }

  /**
   * Process raw data as chunks
   */
  async processData(data: TInput[], config: IStreamConfig): Promise<IStreamResult<TOutput>> {
    const stream = new Readable({
      objectMode: true,
      read() {
        // Implementation handled by push below
      }
    });

    // Push data to stream
    for (const item of data) {
      stream.push(item);
    }
    stream.push(null); // End stream

    return this.processStream(stream, config);
  }

  /**
   * Get current processing statistics
   */
  getStats(): IStreamStats {
    return { ...this.stats };
  }

  /**
   * Pause stream processing
   */
  pause(): void {
    this.isPaused = true;
    this.emit('processing-paused');
  }

  /**
   * Resume stream processing
   */
  resume(): void {
    this.isPaused = false;
    this.emit('processing-resumed');
  }

  /**
   * Stop stream processing
   */
  async stop(force = false): Promise<void> {
    this.shouldStop = true;
    
    if (force) {
      this.currentChunks.clear();
    }
    
    this.emit('processing-stopped');
  }

  /**
   * Check if backpressure is active
   */
  isBackpressureActive(): boolean {
    return this.backpressureManager.getBackpressureLevel() > 0;
  }

  /**
   * Internal stream processing implementation
   */
  private async processStreamInternal(
    input: Readable, 
    config: IStreamConfig
  ): Promise<IStreamResult<TOutput>> {
    const chunks: Array<IStreamChunk<TInput>> = [];
    const results: TOutput[] = [];
    const errors: string[] = [];
    let chunkSequence = 0;

    // Create transform stream for chunk processing
    const chunkProcessor = new Transform({
      objectMode: true,
      transform: async (chunk: Buffer | string, _encoding, callback) => {
        try {
          if (this.shouldStop) {
            callback();
            return;
          }

          while (this.isPaused) {
            await this.delay(100);
          }

          // Check backpressure
          const memoryUsage = process.memoryUsage().heapUsed;
          if (this.backpressureManager.shouldApplyBackpressure(this.currentChunks.size, memoryUsage)) {
            const level = this.backpressureManager.getBackpressureLevel();
            const delay = this.backpressureManager.calculateDelay(level);
            
            this.emit('backpressure-start', level);
            await this.delay(delay);
            this.emit('backpressure-end');
          }

          const streamChunk = this.createStreamChunk(chunk as TInput, chunkSequence++, config);
          chunks.push(streamChunk);
          this.currentChunks.set(streamChunk.id, streamChunk);

          const processed = await this.processChunk(streamChunk, config);
          if (processed) {
            results.push(processed);
          }

          this.currentChunks.delete(streamChunk.id);
          this.updateStats(streamChunk);
          this.emit('chunk-processed', streamChunk);

          callback();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          errors.push(errorMessage);
          this.stats.errorCount++;
          this.emit('chunk-error', error as Error, chunks[chunks.length - 1]);
          callback();
        }
      }
    });

    // Process stream through pipeline
    try {
      await pipeline(input, chunkProcessor);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      errors.push(errorMessage);
    }

    const duration = Date.now() - this.stats.startTime;
    
    return {
      success: errors.length === 0,
      chunksProcessed: chunks.length,
      bytesProcessed: this.stats.bytesProcessed,
      duration,
      data: results.length > 0 ? results[results.length - 1] : undefined,
      errors,
      stats: this.getStats()
    };
  }

  /**
   * Process individual chunk
   */
  protected async processChunk(
    chunk: IStreamChunk<TInput>, 
    config: IStreamConfig
  ): Promise<TOutput | undefined> {
    // Default implementation - override in specialized processors
    if (config.transformFunction) {
      const transformed = await config.transformFunction(chunk);
      return transformed.data as TOutput;
    }
    
    return chunk.data as unknown as TOutput;
  }

  /**
   * Create stream chunk with metadata
   */
  private createStreamChunk(data: TInput, sequence: number, config: IStreamConfig): IStreamChunk<TInput> {
    const id = `chunk_${Date.now()}_${sequence}`;
    const size = this.calculateChunkSize(data);
    const timestamp = Date.now();
    
    const chunk: IStreamChunk<TInput> = {
      id,
      sequence,
      data,
      size,
      timestamp
    };

    if (config.enableIntegrityCheck) {
      chunk.checksum = this.calculateChecksum(data);
    }

    return chunk;
  }

  /**
   * Calculate chunk size in bytes
   */
  private calculateChunkSize(data: TInput): number {
    if (Buffer.isBuffer(data)) {
      return data.length;
    }
    if (typeof data === 'string') {
      return Buffer.byteLength(data, 'utf8');
    }
    return JSON.stringify(data).length;
  }

  /**
   * Calculate chunk checksum
   */
  private calculateChecksum(data: TInput): string {
    const hash = createHash('sha256');
    
    if (Buffer.isBuffer(data)) {
      hash.update(data);
    } else {
      hash.update(String(data));
    }
    
    return hash.digest('hex');
  }

  /**
   * Update processing statistics
   */
  private updateStats(chunk: IStreamChunk<TInput>): void {
    this.stats.chunksProcessed++;
    this.stats.bytesProcessed += chunk.size;
    this.stats.chunksInProgress = this.currentChunks.size;
    this.stats.memoryUsage = process.memoryUsage().heapUsed;
    
    const now = Date.now();
    const totalTime = now - this.stats.startTime;
    this.stats.averageChunkTime = totalTime / this.stats.chunksProcessed;
    this.stats.throughput = this.stats.bytesProcessed / (totalTime / 1000);
    this.stats.backpressureLevel = this.backpressureManager.getBackpressureLevel();

    this.emit('stats-update', this.getStats());
  }

  /**
   * Create initial statistics object
   */
  private createInitialStats(): IStreamStats {
    return {
      chunksProcessed: 0,
      bytesProcessed: 0,
      chunksInProgress: 0,
      averageChunkTime: 0,
      memoryUsage: 0,
      errorCount: 0,
      throughput: 0,
      backpressureLevel: 0,
      startTime: Date.now()
    };
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Backpressure manager for controlling stream flow
 */
export class BackpressureManager implements IBackpressureManager {
  private maxLoad = 100;
  private maxMemory = 100 * 1024 * 1024; // 100MB
  private minDelay = 10;
  private maxDelay = 1000;
  private currentLevel = 0;

  shouldApplyBackpressure(currentLoad: number, memoryUsage: number): boolean {
    const loadPressure = currentLoad / this.maxLoad;
    const memoryPressure = memoryUsage / this.maxMemory;
    
    this.currentLevel = Math.max(loadPressure, memoryPressure);
    
    return this.currentLevel > 0.7; // Apply backpressure at 70% capacity
  }

  calculateDelay(level: number): number {
    const normalizedLevel = Math.min(Math.max(level, 0), 1);
    return this.minDelay + (this.maxDelay - this.minDelay) * normalizedLevel;
  }

  getBackpressureLevel(): number {
    return this.currentLevel;
  }

  updateParameters(params: {
    maxLoad?: number;
    maxMemory?: number;
    minDelay?: number;
    maxDelay?: number;
  }): void {
    if (params.maxLoad !== undefined) this.maxLoad = params.maxLoad;
    if (params.maxMemory !== undefined) this.maxMemory = params.maxMemory;
    if (params.minDelay !== undefined) this.minDelay = params.minDelay;
    if (params.maxDelay !== undefined) this.maxDelay = params.maxDelay;
  }
}

/**
 * Create stream processor with default configuration
 */
export function createStreamProcessor<TInput = Buffer | string, TOutput = Buffer | string>(
  _mode: StreamMode = StreamMode.STANDARD
): StreamProcessor<TInput, TOutput> {
  const processor = new StreamProcessor<TInput, TOutput>();
  
  // Configuration is applied when processing starts
  return processor;
}

/**
 * Utility function to create optimized stream configuration
 */
export function createOptimizedConfig(
  fileSize: number,
  availableMemory: number,
  priority: StreamPriority = StreamPriority.NORMAL
): IStreamConfig {
  // Choose mode based on file size and available memory
  let mode: StreamMode;
  
  if (fileSize > 100 * 1024 * 1024) { // Files > 100MB
    mode = availableMemory > 500 * 1024 * 1024 ? StreamMode.HIGH_THROUGHPUT : StreamMode.MEMORY_EFFICIENT;
  } else if (fileSize > 10 * 1024 * 1024) { // Files > 10MB
    mode = StreamMode.STANDARD;
  } else {
    mode = StreamMode.REAL_TIME;
  }
  
  // Calculate optimal chunk size
  const chunkSize = Math.min(
    Math.max(fileSize / 100, 8 * 1024), // Between 8KB and 1% of file size
    availableMemory / 20 // Don't use more than 5% of available memory per chunk
  );
  
  const baseConfig = DEFAULT_STREAM_CONFIGS[mode];
  
  return {
    ...baseConfig,
    mode,
    priority,
    chunkSize: Math.floor(chunkSize),
    maxMemoryUsage: Math.floor(availableMemory * 0.3) // Use max 30% of available memory
  };
}
