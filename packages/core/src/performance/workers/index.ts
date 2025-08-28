/**
 * @fileoverview Worker Thread System - Main Export
 * @version 1.0.0
 * @license MIT
 */

// Core interfaces and types
export type {
  WorkerTask,
  WorkerResult,
  WorkerConfig,
  WorkerPoolConfig,
  WorkerInfo,
  PoolStats,
  QueueStats,
  ITaskQueue,
  IWorkerPool,
  WorkerMessage,
  IWorkerScript,
  FileProcessingInput,
  FileProcessingOutput,
  EmbeddingInput,
  EmbeddingOutput,
  CodeAnalysisInput,
  CodeAnalysisOutput,
  MemoryAnalysisInput,
  MemoryAnalysisOutput,
} from './WorkerInterfaces.js';

export {
  WorkerType,
  TaskPriority,
  TaskStatus,
} from './WorkerInterfaces.js';

// Core components
export { TaskQueue } from './TaskQueue.js';
export { WorkerPool } from './WorkerPool.js';

// Worker implementations
export { FileProcessingWorker } from './scripts/FileProcessingWorker.js';
export { EmbeddingWorker } from './scripts/EmbeddingWorker.js';
export { CodeAnalysisWorker } from './scripts/CodeAnalysisWorker.js';
export { default as RAGEmbeddingWorker } from './RAGEmbeddingWorker.js';

/**
 * Default worker pool configuration
 */
export const DEFAULT_WORKER_CONFIG: WorkerPoolConfig = {
  maxWorkers: 4,
  idleTimeout: 30000, // 30 seconds
  queueMaxSize: 1000,
  healthCheckInterval: 10000, // 10 seconds
  enableMetrics: true,
  workerConfigs: [
    {
      type: WorkerType.FILE_PROCESSING,
      scriptPath: new URL('./scripts/FileProcessingWorker.js', import.meta.url).pathname,
      maxConcurrency: 2,
      timeout: 60000, // 1 minute
      retries: 3,
      restartOnCrash: true,
      memoryLimit: 512 * 1024 * 1024, // 512MB
    },
    {
      type: WorkerType.EMBEDDING_GENERATION,
      scriptPath: new URL('./scripts/EmbeddingWorker.js', import.meta.url).pathname,
      maxConcurrency: 1,
      timeout: 120000, // 2 minutes
      retries: 2,
      restartOnCrash: true,
      memoryLimit: 1024 * 1024 * 1024, // 1GB
    },
    {
      type: WorkerType.CODE_ANALYSIS,
      scriptPath: new URL('./scripts/CodeAnalysisWorker.js', import.meta.url).pathname,
      maxConcurrency: 2,
      timeout: 90000, // 1.5 minutes
      retries: 3,
      restartOnCrash: true,
      memoryLimit: 256 * 1024 * 1024, // 256MB
    },
  ],
};

/**
 * Create a worker pool with default configuration
 */
export function createWorkerPool(customConfig?: Partial<WorkerPoolConfig>): WorkerPool {
  const taskQueue = new TaskQueue({
    maxSize: customConfig?.queueMaxSize || DEFAULT_WORKER_CONFIG.queueMaxSize!,
  });

  const config: WorkerPoolConfig = {
    ...DEFAULT_WORKER_CONFIG,
    ...customConfig,
    workerConfigs: customConfig?.workerConfigs || DEFAULT_WORKER_CONFIG.workerConfigs,
  };

  return new WorkerPool(config, taskQueue);
}

/**
 * Create a task queue with default configuration
 */
export function createTaskQueue(maxSize = 1000): TaskQueue {
  return new TaskQueue({ maxSize });
}
