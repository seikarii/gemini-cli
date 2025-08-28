/**
 * @fileoverview Stream processing interfaces for handling large files and data streams
 * Provides comprehensive interfaces for backpressure management, chunked processing,
 * and memory-efficient streaming operations
 */

import { Readable } from 'stream';
import { EventEmitter } from 'events';

/**
 * Stream processing modes for different use cases
 */
export enum StreamMode {
  /** Standard streaming with basic backpressure */
  STANDARD = 'standard',
  /** High-throughput streaming for large files */
  HIGH_THROUGHPUT = 'high_throughput',
  /** Memory-efficient streaming for constrained environments */
  MEMORY_EFFICIENT = 'memory_efficient',
  /** Real-time streaming with minimal latency */
  REAL_TIME = 'real_time'
}

/**
 * Stream processing priority levels
 */
export enum StreamPriority {
  LOW = 0,
  NORMAL = 1,
  HIGH = 2,
  CRITICAL = 3
}

/**
 * Stream chunk metadata for tracking and processing
 */
export interface IStreamChunk<T = Buffer | string | Record<string, unknown>> {
  /** Unique identifier for the chunk */
  id: string;
  /** Chunk sequence number */
  sequence: number;
  /** Chunk data payload */
  data: T;
  /** Size of the chunk in bytes */
  size: number;
  /** Timestamp when chunk was created */
  timestamp: number;
  /** Chunk checksum for integrity verification */
  checksum?: string;
  /** Metadata associated with the chunk */
  metadata?: Record<string, unknown>;
  /** Whether this is the final chunk in the stream */
  isLast?: boolean;
}

/**
 * Stream processing configuration
 */
export interface IStreamConfig {
  /** Processing mode */
  mode: StreamMode;
  /** Stream priority level */
  priority: StreamPriority;
  /** Chunk size in bytes */
  chunkSize: number;
  /** Maximum number of concurrent chunks being processed */
  maxConcurrentChunks: number;
  /** High water mark for backpressure (in chunks) */
  highWaterMark: number;
  /** Enable compression for chunks */
  enableCompression: boolean;
  /** Enable chunk integrity checking */
  enableIntegrityCheck: boolean;
  /** Timeout for chunk processing in milliseconds */
  chunkTimeout: number;
  /** Maximum memory usage in bytes */
  maxMemoryUsage: number;
  /** Custom transform function for chunk processing */
  transformFunction?: (chunk: IStreamChunk) => Promise<IStreamChunk>;
}

/**
 * Stream processing statistics
 */
export interface IStreamStats {
  /** Total number of chunks processed */
  chunksProcessed: number;
  /** Total bytes processed */
  bytesProcessed: number;
  /** Number of chunks currently being processed */
  chunksInProgress: number;
  /** Average processing time per chunk in milliseconds */
  averageChunkTime: number;
  /** Current memory usage in bytes */
  memoryUsage: number;
  /** Number of processing errors */
  errorCount: number;
  /** Current throughput in bytes per second */
  throughput: number;
  /** Backpressure level (0-1, where 1 is maximum backpressure) */
  backpressureLevel: number;
  /** Stream start timestamp */
  startTime: number;
  /** Stream end timestamp (if completed) */
  endTime?: number;
}

/**
 * Stream processing result
 */
export interface IStreamResult<T = Buffer | string | Record<string, unknown>> {
  /** Processing success status */
  success: boolean;
  /** Total number of chunks processed */
  chunksProcessed: number;
  /** Total bytes processed */
  bytesProcessed: number;
  /** Processing duration in milliseconds */
  duration: number;
  /** Final processed data (if applicable) */
  data?: T;
  /** Processing errors encountered */
  errors: string[];
  /** Processing statistics */
  stats: IStreamStats;
}

/**
 * Stream processor interface for handling different types of streams
 */
export interface IStreamProcessor<TInput = Buffer | string, TOutput = Buffer | string> extends EventEmitter {
  /**
   * Process a readable stream
   * @param input Input stream to process
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processStream(input: Readable, config: IStreamConfig): Promise<IStreamResult<TOutput>>;

  /**
   * Process a file as a stream
   * @param filePath Path to the file to process
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processFile(filePath: string, config: IStreamConfig): Promise<IStreamResult<TOutput>>;

  /**
   * Process raw data as chunks
   * @param data Data to process
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processData(data: TInput[], config: IStreamConfig): Promise<IStreamResult<TOutput>>;

  /**
   * Get current processing statistics
   * @returns Current stream processing statistics
   */
  getStats(): IStreamStats;

  /**
   * Pause stream processing
   */
  pause(): void;

  /**
   * Resume stream processing
   */
  resume(): void;

  /**
   * Stop stream processing
   * @param force Whether to force stop immediately
   */
  stop(force?: boolean): Promise<void>;

  /**
   * Check if backpressure is active
   * @returns True if backpressure is active
   */
  isBackpressureActive(): boolean;
}

/**
 * File stream processor for handling large files efficiently
 */
export interface IFileStreamProcessor extends IStreamProcessor<Buffer, Buffer> {
  /**
   * Process a large file with chunked reading
   * @param filePath Path to the file
   * @param outputPath Optional output file path
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processLargeFile(
    filePath: string,
    outputPath?: string,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<Buffer>>;

  /**
   * Calculate file checksum using streaming
   * @param filePath Path to the file
   * @param algorithm Hash algorithm to use
   * @returns Promise resolving to file checksum
   */
  calculateChecksum(filePath: string, algorithm?: string): Promise<string>;

  /**
   * Compress file using streaming
   * @param inputPath Input file path
   * @param outputPath Output file path
   * @param compressionLevel Compression level (1-9)
   * @returns Promise resolving to compression result
   */
  compressFile(inputPath: string, outputPath: string, compressionLevel?: number): Promise<IStreamResult>;
}

/**
 * Text stream processor for handling large text files and content
 */
export interface ITextStreamProcessor extends IStreamProcessor<string, string[]> {
  /**
   * Process text file line by line
   * @param filePath Path to the text file
   * @param lineProcessor Function to process each line
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processLines(
    filePath: string,
    lineProcessor: (line: string, lineNumber: number) => Promise<string | null>,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<string[]>>;

  /**
   * Search patterns in large text files
   * @param filePath Path to the text file
   * @param patterns Patterns to search for
   * @param config Processing configuration
   * @returns Promise resolving to search results
   */
  searchPatterns(
    filePath: string,
    patterns: RegExp[],
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<Array<{ line: number; match: string; pattern: RegExp }>>>;

  /**
   * Transform text content using streaming
   * @param inputPath Input file path
   * @param outputPath Output file path
   * @param transformer Transform function
   * @param config Processing configuration
   * @returns Promise resolving to transformation result
   */
  transformText(
    inputPath: string,
    outputPath: string,
    transformer: (text: string) => Promise<string>,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult>;
}

/**
 * JSON stream processor for handling large JSON files and arrays
 */
export interface IJSONStreamProcessor extends IStreamProcessor<Record<string, unknown>, Array<Record<string, unknown>>> {
  /**
   * Process large JSON array
   * @param filePath Path to the JSON file
   * @param itemProcessor Function to process each JSON item
   * @param config Processing configuration
   * @returns Promise resolving to processing result
   */
  processJSONArray(
    filePath: string,
    itemProcessor: (item: Record<string, unknown>, index: number) => Promise<Record<string, unknown>>,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<Array<Record<string, unknown>>>>;

  /**
   * Stream JSON objects from file
   * @param filePath Path to the JSON file
   * @param config Processing configuration
   * @returns Async iterator for JSON objects
   */
  streamJSONObjects(filePath: string, config?: Partial<IStreamConfig>): AsyncIterable<Record<string, unknown>>;

  /**
   * Validate JSON structure using streaming
   * @param filePath Path to the JSON file
   * @param schema JSON schema for validation
   * @param config Processing configuration
   * @returns Promise resolving to validation result
   */
  validateJSON(
    filePath: string,
    schema: Record<string, unknown>,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<Array<{ path: string; error: string }>>>;
}

/**
 * Stream backpressure manager for controlling flow
 */
export interface IBackpressureManager {
  /**
   * Check if backpressure should be applied
   * @param currentLoad Current processing load
   * @param memoryUsage Current memory usage
   * @returns True if backpressure should be applied
   */
  shouldApplyBackpressure(currentLoad: number, memoryUsage: number): boolean;

  /**
   * Calculate delay for backpressure
   * @param level Backpressure level (0-1)
   * @returns Delay in milliseconds
   */
  calculateDelay(level: number): number;

  /**
   * Get current backpressure level
   * @returns Backpressure level (0-1)
   */
  getBackpressureLevel(): number;

  /**
   * Update backpressure parameters
   * @param params New backpressure parameters
   */
  updateParameters(params: {
    maxLoad?: number;
    maxMemory?: number;
    minDelay?: number;
    maxDelay?: number;
  }): void;
}

/**
 * Default stream configurations for different use cases
 */
export const DEFAULT_STREAM_CONFIGS: Record<StreamMode, IStreamConfig> = {
  [StreamMode.STANDARD]: {
    mode: StreamMode.STANDARD,
    priority: StreamPriority.NORMAL,
    chunkSize: 64 * 1024, // 64KB
    maxConcurrentChunks: 4,
    highWaterMark: 16,
    enableCompression: false,
    enableIntegrityCheck: false,
    chunkTimeout: 30000, // 30 seconds
    maxMemoryUsage: 100 * 1024 * 1024, // 100MB
  },
  [StreamMode.HIGH_THROUGHPUT]: {
    mode: StreamMode.HIGH_THROUGHPUT,
    priority: StreamPriority.HIGH,
    chunkSize: 1024 * 1024, // 1MB
    maxConcurrentChunks: 8,
    highWaterMark: 32,
    enableCompression: true,
    enableIntegrityCheck: true,
    chunkTimeout: 60000, // 60 seconds
    maxMemoryUsage: 500 * 1024 * 1024, // 500MB
  },
  [StreamMode.MEMORY_EFFICIENT]: {
    mode: StreamMode.MEMORY_EFFICIENT,
    priority: StreamPriority.NORMAL,
    chunkSize: 16 * 1024, // 16KB
    maxConcurrentChunks: 2,
    highWaterMark: 8,
    enableCompression: true,
    enableIntegrityCheck: false,
    chunkTimeout: 45000, // 45 seconds
    maxMemoryUsage: 50 * 1024 * 1024, // 50MB
  },
  [StreamMode.REAL_TIME]: {
    mode: StreamMode.REAL_TIME,
    priority: StreamPriority.CRITICAL,
    chunkSize: 8 * 1024, // 8KB
    maxConcurrentChunks: 1,
    highWaterMark: 4,
    enableCompression: false,
    enableIntegrityCheck: false,
    chunkTimeout: 5000, // 5 seconds
    maxMemoryUsage: 25 * 1024 * 1024, // 25MB
  },
};

/**
 * Stream processing events
 */
export interface IStreamEvents<TInput = Buffer | string | Record<string, unknown>, TOutput = Buffer | string> {
  'chunk-processed': (chunk: IStreamChunk<TInput>) => void;
  'chunk-error': (error: Error, chunk: IStreamChunk<TInput>) => void;
  'backpressure-start': (level: number) => void;
  'backpressure-end': () => void;
  'stats-update': (stats: IStreamStats) => void;
  'processing-complete': (result: IStreamResult<TOutput>) => void;
  'processing-error': (error: Error) => void;
  'processing-paused': () => void;
  'processing-resumed': () => void;
  'processing-stopped': () => void;
}
