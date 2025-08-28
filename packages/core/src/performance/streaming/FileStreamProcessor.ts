/**
 * @fileoverview File stream processor for efficient handling of large files
 * with compression, checksum calculation, and optimized I/O operations
 */

import { createReadStream, createWriteStream, promises as fs } from 'fs';
import { createHash } from 'crypto';
import { createGzip, createGunzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { Transform } from 'stream';
import { join, dirname } from 'path';
import { 
  IFileStreamProcessor, 
  IStreamConfig, 
  IStreamResult,
  StreamMode,
  StreamPriority,
  DEFAULT_STREAM_CONFIGS
} from './StreamInterfaces.js';
import { StreamProcessor } from './StreamProcessor.js';

/**
 * Specialized file stream processor for large file operations
 */
export class FileStreamProcessor extends StreamProcessor<Buffer, Buffer> implements IFileStreamProcessor {
  
  /**
   * Process a large file with chunked reading and optional output
   */
  async processLargeFile(
    filePath: string,
    outputPath?: string,
    config?: Partial<IStreamConfig>
  ): Promise<IStreamResult<Buffer>> {
    const stats = await fs.stat(filePath);
    const fileSize = stats.size;
    
    // Create optimized configuration for the file size
    const optimizedConfig = this.createFileConfig(fileSize, config);
    
    const inputStream = createReadStream(filePath, {
      highWaterMark: optimizedConfig.chunkSize
    });

    if (outputPath) {
      // Ensure output directory exists
      await fs.mkdir(dirname(outputPath), { recursive: true });
      
      const outputStream = createWriteStream(outputPath);
      const transformStream = new Transform({
        transform(chunk, _encoding, callback) {
          // Apply any transformation logic here
          callback(null, chunk);
        }
      });

      try {
        await pipeline(inputStream, transformStream, outputStream);
      } catch (error) {
        throw new Error(`File processing failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    return this.processStream(inputStream, optimizedConfig);
  }

  /**
   * Calculate file checksum using streaming approach
   */
  async calculateChecksum(filePath: string, algorithm = 'sha256'): Promise<string> {
    const hash = createHash(algorithm);
    const stream = createReadStream(filePath);

    return new Promise((resolve, reject) => {
      stream.on('data', (data: Buffer) => {
        hash.update(data);
      });

      stream.on('end', () => {
        resolve(hash.digest('hex'));
      });

      stream.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Compress file using streaming with configurable compression level
   */
  async compressFile(
    inputPath: string,
    outputPath: string,
    compressionLevel = 6
  ): Promise<IStreamResult> {
    const stats = await fs.stat(inputPath);
    const startTime = Date.now();

    // Ensure output directory exists
    await fs.mkdir(dirname(outputPath), { recursive: true });

    const inputStream = createReadStream(inputPath);
    const gzipStream = createGzip({ level: compressionLevel });
    const outputStream = createWriteStream(outputPath);

    try {
      await pipeline(inputStream, gzipStream, outputStream);
      
      const endTime = Date.now();
      
      return {
        success: true,
        chunksProcessed: Math.ceil(stats.size / 65536), // Estimate based on 64KB chunks
        bytesProcessed: stats.size,
        duration: endTime - startTime,
        errors: [],
        stats: {
          chunksProcessed: Math.ceil(stats.size / 65536),
          bytesProcessed: stats.size,
          chunksInProgress: 0,
          averageChunkTime: (endTime - startTime) / Math.ceil(stats.size / 65536),
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 0,
          throughput: stats.size / ((endTime - startTime) / 1000),
          backpressureLevel: 0,
          startTime,
          endTime
        }
      };
    } catch (error) {
      return {
        success: false,
        chunksProcessed: 0,
        bytesProcessed: 0,
        duration: Date.now() - startTime,
        errors: [error instanceof Error ? error.message : String(error)],
        stats: {
          chunksProcessed: 0,
          bytesProcessed: 0,
          chunksInProgress: 0,
          averageChunkTime: 0,
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 1,
          throughput: 0,
          backpressureLevel: 0,
          startTime,
          endTime: Date.now()
        }
      };
    }
  }

  /**
   * Decompress a gzip file using streaming
   */
  async decompressFile(inputPath: string, outputPath: string): Promise<IStreamResult> {
    const startTime = Date.now();

    // Ensure output directory exists
    await fs.mkdir(dirname(outputPath), { recursive: true });

    const inputStream = createReadStream(inputPath);
    const gunzipStream = createGunzip();
    const outputStream = createWriteStream(outputPath);

    try {
      await pipeline(inputStream, gunzipStream, outputStream);
      
      const endTime = Date.now();
      const inputStats = await fs.stat(inputPath);
      const outputStats = await fs.stat(outputPath);
      
      return {
        success: true,
        chunksProcessed: Math.ceil(inputStats.size / 65536),
        bytesProcessed: inputStats.size,
        duration: endTime - startTime,
        errors: [],
        stats: {
          chunksProcessed: Math.ceil(inputStats.size / 65536),
          bytesProcessed: inputStats.size,
          chunksInProgress: 0,
          averageChunkTime: (endTime - startTime) / Math.ceil(inputStats.size / 65536),
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 0,
          throughput: inputStats.size / ((endTime - startTime) / 1000),
          backpressureLevel: 0,
          startTime,
          endTime
        }
      };
    } catch (error) {
      return {
        success: false,
        chunksProcessed: 0,
        bytesProcessed: 0,
        duration: Date.now() - startTime,
        errors: [error instanceof Error ? error.message : String(error)],
        stats: {
          chunksProcessed: 0,
          bytesProcessed: 0,
          chunksInProgress: 0,
          averageChunkTime: 0,
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 1,
          throughput: 0,
          backpressureLevel: 0,
          startTime,
          endTime: Date.now()
        }
      };
    }
  }

  /**
   * Copy file using streaming with progress tracking
   */
  async copyFile(sourcePath: string, destPath: string): Promise<IStreamResult> {
    const stats = await fs.stat(sourcePath);
    const config = this.createFileConfig(stats.size);

    // Ensure destination directory exists
    await fs.mkdir(dirname(destPath), { recursive: true });

    const inputStream = createReadStream(sourcePath, {
      highWaterMark: config.chunkSize
    });
    const outputStream = createWriteStream(destPath);

    const startTime = Date.now();

    try {
      await pipeline(inputStream, outputStream);
      
      const endTime = Date.now();
      
      return {
        success: true,
        chunksProcessed: Math.ceil(stats.size / config.chunkSize),
        bytesProcessed: stats.size,
        duration: endTime - startTime,
        errors: [],
        stats: {
          chunksProcessed: Math.ceil(stats.size / config.chunkSize),
          bytesProcessed: stats.size,
          chunksInProgress: 0,
          averageChunkTime: (endTime - startTime) / Math.ceil(stats.size / config.chunkSize),
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 0,
          throughput: stats.size / ((endTime - startTime) / 1000),
          backpressureLevel: 0,
          startTime,
          endTime
        }
      };
    } catch (error) {
      return {
        success: false,
        chunksProcessed: 0,
        bytesProcessed: 0,
        duration: Date.now() - startTime,
        errors: [error instanceof Error ? error.message : String(error)],
        stats: {
          chunksProcessed: 0,
          bytesProcessed: 0,
          chunksInProgress: 0,
          averageChunkTime: 0,
          memoryUsage: process.memoryUsage().heapUsed,
          errorCount: 1,
          throughput: 0,
          backpressureLevel: 0,
          startTime,
          endTime: Date.now()
        }
      };
    }
  }

  /**
   * Split large file into smaller chunks
   */
  async splitFile(
    inputPath: string,
    outputDir: string,
    chunkSizeBytes: number
  ): Promise<Array<{ path: string; size: number }>> {
    const chunks: Array<{ path: string; size: number }> = [];

    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    const inputStream = createReadStream(inputPath);
    let currentChunk = 0;
    let currentChunkSize = 0;
    let currentChunkStream: ReturnType<typeof createWriteStream> | null = null;

    return new Promise((resolve, reject) => {
      inputStream.on('data', (data: string | Buffer) => {
        const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data);
        
        if (!currentChunkStream || currentChunkSize >= chunkSizeBytes) {
          // Close previous chunk
          if (currentChunkStream) {
            currentChunkStream.end();
          }

          // Start new chunk
          const chunkPath = join(outputDir, `chunk_${currentChunk.toString().padStart(4, '0')}`);
          currentChunkStream = createWriteStream(chunkPath);
          chunks.push({ path: chunkPath, size: 0 });
          currentChunkSize = 0;
          currentChunk++;
        }

        currentChunkStream!.write(buffer);
        currentChunkSize += buffer.length;
        chunks[chunks.length - 1].size += buffer.length;
      });

      inputStream.on('end', () => {
        if (currentChunkStream) {
          currentChunkStream.end();
        }
        resolve(chunks);
      });

      inputStream.on('error', (error) => {
        if (currentChunkStream) {
          currentChunkStream.destroy();
        }
        reject(error);
      });
    });
  }

  /**
   * Create optimized configuration for file processing
   */
  private createFileConfig(
    fileSize: number,
    userConfig?: Partial<IStreamConfig>
  ): IStreamConfig {
    // Determine optimal mode based on file size
    let mode: StreamMode;
    
    if (fileSize > 1024 * 1024 * 1024) { // Files > 1GB
      mode = StreamMode.HIGH_THROUGHPUT;
    } else if (fileSize > 100 * 1024 * 1024) { // Files > 100MB
      mode = StreamMode.STANDARD;
    } else if (fileSize > 10 * 1024 * 1024) { // Files > 10MB
      mode = StreamMode.MEMORY_EFFICIENT;
    } else {
      mode = StreamMode.REAL_TIME;
    }

    const baseConfig = DEFAULT_STREAM_CONFIGS[mode];
    
    // Calculate optimal chunk size based on file size
    const optimalChunkSize = Math.min(
      Math.max(fileSize / 50, 8 * 1024), // Between 8KB and 2% of file size
      2 * 1024 * 1024 // Max 2MB chunks
    );

    return {
      ...baseConfig,
      chunkSize: Math.floor(optimalChunkSize),
      priority: StreamPriority.HIGH,
      enableIntegrityCheck: fileSize > 100 * 1024 * 1024, // Enable for files > 100MB
      ...userConfig
    };
  }
}

/**
 * Create file stream processor instance
 */
export function createFileStreamProcessor(): FileStreamProcessor {
  return new FileStreamProcessor();
}

/**
 * Utility function to get file processing recommendations
 */
export async function getFileProcessingRecommendations(filePath: string): Promise<{
  recommendedMode: StreamMode;
  recommendedChunkSize: number;
  enableCompression: boolean;
  enableIntegrityCheck: boolean;
  estimatedMemoryUsage: number;
  estimatedProcessingTime: number;
}> {
  const stats = await fs.stat(filePath);
  const fileSize = stats.size;
  
  let recommendedMode: StreamMode;
  let recommendedChunkSize: number;
  
  if (fileSize > 1024 * 1024 * 1024) { // > 1GB
    recommendedMode = StreamMode.HIGH_THROUGHPUT;
    recommendedChunkSize = 1024 * 1024; // 1MB
  } else if (fileSize > 100 * 1024 * 1024) { // > 100MB
    recommendedMode = StreamMode.STANDARD;
    recommendedChunkSize = 512 * 1024; // 512KB
  } else if (fileSize > 10 * 1024 * 1024) { // > 10MB
    recommendedMode = StreamMode.MEMORY_EFFICIENT;
    recommendedChunkSize = 64 * 1024; // 64KB
  } else {
    recommendedMode = StreamMode.REAL_TIME;
    recommendedChunkSize = 16 * 1024; // 16KB
  }

  const enableCompression = fileSize > 50 * 1024 * 1024; // Enable for files > 50MB
  const enableIntegrityCheck = fileSize > 100 * 1024 * 1024; // Enable for files > 100MB
  
  // Estimate memory usage (approximately 4x chunk size for buffering)
  const estimatedMemoryUsage = recommendedChunkSize * 4;
  
  // Estimate processing time (rough calculation based on typical disk speeds)
  const estimatedThroughput = 100 * 1024 * 1024; // 100MB/s typical SSD speed
  const estimatedProcessingTime = (fileSize / estimatedThroughput) * 1000; // in milliseconds

  return {
    recommendedMode,
    recommendedChunkSize,
    enableCompression,
    enableIntegrityCheck,
    estimatedMemoryUsage,
    estimatedProcessingTime
  };
}
