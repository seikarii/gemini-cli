/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Task priority levels for worker queue
 */
export enum TaskPriority {
  IMMEDIATE = 1000,
  HIGH = 750,
  NORMAL = 500,
  LOW = 250,
  BACKGROUND = 100,
}

/**
 * Task status enumeration
 */
export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  TIMEOUT = 'timeout',
}

/**
 * Worker types for different kinds of processing
 */
export enum WorkerType {
  FILE_PROCESSING = 'file_processing',
  EMBEDDING_GENERATION = 'embedding_generation',
  CODE_ANALYSIS = 'code_analysis',
  MEMORY_ANALYSIS = 'memory_analysis',
  GENERIC = 'generic',
}

/**
 * Base interface for all worker tasks
 */
export interface WorkerTask<TInput = unknown, _TOutput = unknown> {
  id: string;
  type: WorkerType;
  priority: TaskPriority;
  input: TInput;
  timeout?: number;
  retries?: number;
  metadata?: Record<string, unknown>;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
}

/**
 * Result of worker task execution
 */
export interface WorkerResult<TOutput = unknown> {
  taskId: string;
  success: boolean;
  result?: TOutput;
  error?: {
    message: string;
    stack?: string;
    code?: string;
  };
  executionTime: number;
  workerId: string;
  metadata?: Record<string, unknown>;
}

/**
 * Worker configuration
 */
export interface WorkerConfig {
  type: WorkerType;
  scriptPath: string;
  maxConcurrency?: number;
  timeout?: number;
  retries?: number;
  restartOnCrash?: boolean;
  memoryLimit?: number;
  env?: Record<string, string>;
}

/**
 * Worker pool configuration
 */
export interface WorkerPoolConfig {
  maxWorkers?: number;
  idleTimeout?: number;
  queueMaxSize?: number;
  healthCheckInterval?: number;
  enableMetrics?: boolean;
  workerConfigs: WorkerConfig[];
}

/**
 * Worker instance information
 */
export interface WorkerInfo {
  id: string;
  type: WorkerType;
  pid?: number;
  status: 'idle' | 'busy' | 'crashed' | 'terminated';
  currentTask?: string;
  createdAt: Date;
  lastUsed: Date;
  tasksCompleted: number;
  tasksErrored: number;
  memoryUsage?: number;
}

/**
 * Task queue statistics
 */
export interface QueueStats {
  totalTasks: number;
  pendingTasks: number;
  runningTasks: number;
  completedTasks: number;
  failedTasks: number;
  averageWaitTime: number;
  averageExecutionTime: number;
  tasksByPriority: Record<TaskPriority, number>;
  tasksByType: Record<WorkerType, number>;
}

/**
 * Worker pool statistics
 */
export interface PoolStats {
  totalWorkers: number;
  activeWorkers: number;
  idleWorkers: number;
  crashedWorkers: number;
  queue: QueueStats;
  uptime: number;
  totalTasksProcessed: number;
  totalErrors: number;
  averageTaskTime: number;
  memoryUsage: {
    total: number;
    perWorker: number;
    peak: number;
  };
}

/**
 * Worker pool events
 */
export interface WorkerPoolEvents {
  'task.queued': (task: WorkerTask) => void;
  'task.started': (task: WorkerTask, workerId: string) => void;
  'task.completed': (result: WorkerResult) => void;
  'task.failed': (result: WorkerResult) => void;
  'task.timeout': (taskId: string, workerId: string) => void;
  'worker.created': (workerId: string, type: WorkerType) => void;
  'worker.destroyed': (workerId: string, type: WorkerType) => void;
  'worker.crashed': (workerId: string, error: Error) => void;
  'pool.drained': () => void;
  'pool.error': (error: Error) => void;
}

/**
 * Task queue interface
 */
export interface ITaskQueue {
  enqueue<TInput, TOutput>(task: WorkerTask<TInput, TOutput>): Promise<void>;
  dequeue(workerType?: WorkerType): Promise<WorkerTask | null>;
  peek(workerType?: WorkerType): WorkerTask | null;
  remove(taskId: string): boolean;
  clear(): void;
  size(): number;
  isEmpty(): boolean;
  getStats(): QueueStats;
  getTasksByPriority(priority: TaskPriority): WorkerTask[];
  getTasksByType(type: WorkerType): WorkerTask[];
}

/**
 * Worker pool interface
 */
export interface IWorkerPool {
  /**
   * Execute a task
   */
  execute<TInput, TOutput>(
    type: WorkerType,
    input: TInput,
    options?: {
      priority?: TaskPriority;
      timeout?: number;
      retries?: number;
      metadata?: Record<string, unknown>;
    }
  ): Promise<WorkerResult<TOutput>>;

  /**
   * Submit a task and get a promise
   */
  submit<TInput, TOutput>(task: WorkerTask<TInput, TOutput>): Promise<WorkerResult<TOutput>>;

  /**
   * Batch execute multiple tasks
   */
  batch<TInput, TOutput>(
    tasks: Array<{
      type: WorkerType;
      input: TInput;
      options?: {
        priority?: TaskPriority;
        timeout?: number;
        retries?: number;
        metadata?: Record<string, unknown>;
      };
    }>
  ): Promise<Array<WorkerResult<TOutput>>>;

  /**
   * Get worker pool statistics
   */
  getStats(): PoolStats;

  /**
   * Get information about all workers
   */
  getWorkers(): WorkerInfo[];

  /**
   * Get information about a specific worker
   */
  getWorker(workerId: string): WorkerInfo | null;

  /**
   * Terminate a specific worker
   */
  terminateWorker(workerId: string): Promise<void>;

  /**
   * Restart all crashed workers
   */
  restartCrashedWorkers(): Promise<void>;

  /**
   * Pause task processing
   */
  pause(): void;

  /**
   * Resume task processing
   */
  resume(): void;

  /**
   * Wait for all current tasks to complete
   */
  drain(): Promise<void>;

  /**
   * Gracefully shutdown the pool
   */
  shutdown(): Promise<void>;

  /**
   * Force shutdown the pool
   */
  forceShutdown(): Promise<void>;

  /**
   * Health check
   */
  healthCheck(): Promise<{
    healthy: boolean;
    workers: Array<{ id: string; healthy: boolean; error?: string }>;
    queue: { size: number; oldestTask?: Date };
  }>;

  /**
   * Event management
   */
  on<K extends keyof WorkerPoolEvents>(event: K, handler: WorkerPoolEvents[K]): void;
  off<K extends keyof WorkerPoolEvents>(event: K, handler: WorkerPoolEvents[K]): void;
  emit<K extends keyof WorkerPoolEvents>(event: K, ...args: Parameters<WorkerPoolEvents[K]>): void;
}

/**
 * Generic worker message types
 */
export interface WorkerMessage {
  type: 'task' | 'result' | 'error' | 'ping' | 'pong' | 'shutdown';
  taskId?: string;
  payload?: unknown;
  error?: {
    message: string;
    stack?: string;
    code?: string;
  };
}

/**
 * Worker script interface that all workers must implement
 */
export interface IWorkerScript {
  /**
   * Process a task
   */
  process<TInput, TOutput>(input: TInput, metadata?: Record<string, unknown>): Promise<TOutput>;

  /**
   * Health check
   */
  healthCheck(): Promise<boolean>;

  /**
   * Cleanup resources
   */
  cleanup(): Promise<void>;
}

/**
 * File processing task input
 */
export interface FileProcessingInput {
  operation: 'read' | 'write' | 'analyze' | 'transform' | 'validate';
  filePath: string;
  content?: string;
  options?: {
    encoding?: BufferEncoding;
    maxSize?: number;
    parseAs?: 'text' | 'json' | 'yaml' | 'xml' | 'binary';
    transformations?: string[];
    validationRules?: string[];
  };
}

/**
 * File processing task output
 */
export interface FileProcessingOutput {
  success: boolean;
  content?: string;
  metadata?: {
    size: number;
    encoding: string;
    mimeType?: string;
    checksum?: string;
    lastModified?: Date;
  };
  analysis?: {
    lineCount?: number;
    wordCount?: number;
    complexity?: number;
    issues?: Array<{
      type: string;
      message: string;
      line?: number;
      column?: number;
    }>;
  };
  transformationResults?: Array<{
    transformation: string;
    success: boolean;
    result?: string;
    error?: string;
  }>;
  validationResults?: Array<{
    rule: string;
    passed: boolean;
    message?: string;
  }>;
}

/**
 * Embedding generation task input
 */
export interface EmbeddingInput {
  text: string;
  model?: string;
  options?: {
    maxTokens?: number;
    dimensions?: number;
    normalize?: boolean;
    batchSize?: number;
  };
}

/**
 * Embedding generation task output
 */
export interface EmbeddingOutput {
  embeddings: number[];
  metadata: {
    model: string;
    dimensions: number;
    tokenCount: number;
    processingTime: number;
  };
}

/**
 * Code analysis task input
 */
export interface CodeAnalysisInput {
  code: string;
  language: string;
  analysisTypes: Array<'syntax' | 'complexity' | 'security' | 'performance' | 'style'>;
  options?: {
    includeMetrics?: boolean;
    includeAST?: boolean;
    securityRules?: string[];
    styleRules?: string[];
  };
}

/**
 * Code analysis task output
 */
export interface CodeAnalysisOutput {
  syntax: {
    valid: boolean;
    errors: Array<{
      message: string;
      line: number;
      column: number;
      severity: 'error' | 'warning';
    }>;
  };
  complexity?: {
    cyclomatic: number;
    cognitive: number;
    maintainabilityIndex: number;
  };
  security?: {
    vulnerabilities: Array<{
      type: string;
      severity: 'critical' | 'high' | 'medium' | 'low';
      message: string;
      line?: number;
      recommendation?: string;
    }>;
  };
  performance?: {
    issues: Array<{
      type: string;
      message: string;
      impact: 'high' | 'medium' | 'low';
      line?: number;
    }>;
  };
  style?: {
    violations: Array<{
      rule: string;
      message: string;
      line: number;
      fixable: boolean;
    }>;
  };
  metrics?: {
    linesOfCode: number;
    linesOfComments: number;
    functions: number;
    classes: number;
    imports: number;
  };
  ast?: Record<string, unknown>;
}

/**
 * Memory analysis task input
 */
export interface MemoryAnalysisInput {
  operation: 'discover' | 'search' | 'optimize' | 'validate';
  data?: unknown;
  query?: string;
  options?: {
    includeEmbeddings?: boolean;
    maxResults?: number;
    threshold?: number;
    optimizationLevel?: 'basic' | 'aggressive';
  };
}

/**
 * Memory analysis task output
 */
export interface MemoryAnalysisOutput {
  operation: string;
  results?: unknown[];
  optimizations?: Array<{
    type: string;
    description: string;
    impact: number;
    applied: boolean;
  }>;
  metadata: {
    processingTime: number;
    memoryUsed: number;
    resultCount: number;
  };
}
