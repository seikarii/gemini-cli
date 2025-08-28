/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import { cpus } from 'os';
import {
  WorkerType,
  TaskPriority,
  WorkerTask,
  WorkerResult,
  WorkerConfig,
  WorkerPoolConfig,
  WorkerInfo,
  PoolStats,
  IWorkerPool,
  WorkerMessage,
} from './WorkerInterfaces.js';
import { TaskQueue } from './TaskQueue.js';

/**
 * Worker pool implementation for managing Node.js worker threads
 */
export class WorkerPool extends EventEmitter implements IWorkerPool {
  private workers = new Map<string, Worker>();
  private workerInfo = new Map<string, WorkerInfo>();
  private workerConfigs = new Map<WorkerType, WorkerConfig>();
  private taskQueue: TaskQueue;
  private pendingTasks = new Map<string, {
    resolve: (result: WorkerResult) => void;
    reject: (error: Error) => void;
    timeout?: NodeJS.Timeout;
  }>();
  
  private config: Required<WorkerPoolConfig>;
  private running = false;
  private paused = false;
  private healthCheckInterval?: NodeJS.Timeout;
  private startTime = Date.now();
  private totalTasksProcessed = 0;
  private totalErrors = 0;
  private totalExecutionTime = 0;
  private peakMemoryUsage = 0;

  constructor(config: WorkerPoolConfig) {
    super();
    
    this.config = {
      maxWorkers: config.maxWorkers || cpus().length,
      idleTimeout: config.idleTimeout || 60000, // 1 minute
      queueMaxSize: config.queueMaxSize || 10000,
      healthCheckInterval: config.healthCheckInterval || 30000, // 30 seconds
      enableMetrics: config.enableMetrics ?? true,
      workerConfigs: config.workerConfigs,
    };

    this.taskQueue = new TaskQueue(this.config.queueMaxSize);
    
    // Store worker configurations
    this.config.workerConfigs.forEach(workerConfig => {
      this.workerConfigs.set(workerConfig.type, workerConfig);
    });

    this.startHealthMonitoring();
    this.running = true;
  }

  /**
   * Execute a task with specified options
   */
  async execute<TInput, TOutput>(
    type: WorkerType,
    input: TInput,
    options: {
      priority?: TaskPriority;
      timeout?: number;
      retries?: number;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<WorkerResult<TOutput>> {
    const task: WorkerTask<TInput, TOutput> = {
      id: randomUUID(),
      type,
      priority: options.priority || TaskPriority.NORMAL,
      input,
      timeout: options.timeout,
      retries: options.retries || 3,
      metadata: options.metadata,
      createdAt: new Date(),
    };

    return this.submit(task);
  }

  /**
   * Submit a task and get a promise for the result
   */
  async submit<TInput, TOutput>(task: WorkerTask<TInput, TOutput>): Promise<WorkerResult<TOutput>> {
    if (!this.running) {
      throw new Error('Worker pool is not running');
    }

    if (this.paused) {
      throw new Error('Worker pool is paused');
    }

    return new Promise<WorkerResult<TOutput>>((resolve, reject) => {
      // Set up timeout if specified
      let timeoutHandle: NodeJS.Timeout | undefined;
      if (task.timeout) {
        timeoutHandle = setTimeout(() => {
          this.pendingTasks.delete(task.id);
          const timeoutResult: WorkerResult<TOutput> = {
            taskId: task.id,
            success: false,
            error: {
              message: `Task timed out after ${task.timeout}ms`,
              code: 'TASK_TIMEOUT',
            },
            executionTime: task.timeout || 30000,
            workerId: 'timeout',
          };
          this.emit('task.timeout', task.id, 'timeout');
          resolve(timeoutResult);
        }, task.timeout);
      }

      // Store the promise resolvers
      this.pendingTasks.set(task.id, {
        resolve: (result: WorkerResult<unknown>) => {
          if (timeoutHandle) {
            clearTimeout(timeoutHandle);
          }
          resolve(result as WorkerResult<TOutput>);
        },
        reject: (error: Error) => {
          if (timeoutHandle) {
            clearTimeout(timeoutHandle);
          }
          reject(error);
        },
        timeout: timeoutHandle,
      });

      // Add task to queue
      this.taskQueue.enqueue(task).then(() => {
        this.emit('task.queued', task);
        this.processQueue();
      }).catch(reject);
    });
  }

  /**
   * Batch execute multiple tasks
   */
  async batch<TInput, TOutput>(
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
  ): Promise<Array<WorkerResult<TOutput>>> {
    const promises = tasks.map(taskSpec => 
      this.execute<TInput, TOutput>(
        taskSpec.type,
        taskSpec.input,
        taskSpec.options || {}
      )
    );

    return Promise.all(promises);
  }

  /**
   * Process the task queue
   */
  private async processQueue(): Promise<void> {
    while (!this.paused && this.taskQueue.size() > 0) {
      // Find available worker or create new one
      const availableWorker = this.findAvailableWorker();
      
      if (!availableWorker) {
        // Try to create a new worker
        const task = this.taskQueue.peek();
        if (task && this.canCreateWorker(task.type)) {
          const worker = await this.createWorker(task.type);
          if (worker) {
            await this.assignTaskToWorker(worker.id, task);
          }
        }
        break; // No available workers and can't create more
      } else {
        // Get next appropriate task for this worker
        const workerInfo = this.workerInfo.get(availableWorker.id);
        const task = await this.taskQueue.dequeue(workerInfo?.type);
        
        if (task) {
          await this.assignTaskToWorker(availableWorker.id, task);
        } else {
          break; // No appropriate tasks for available workers
        }
      }
    }
  }

  /**
   * Find an available worker
   */
  private findAvailableWorker(): { id: string; worker: Worker } | null {
    for (const [workerId, worker] of this.workers) {
      const info = this.workerInfo.get(workerId);
      if (info && info.status === 'idle') {
        return { id: workerId, worker };
      }
    }
    return null;
  }

  /**
   * Check if we can create a new worker
   */
  private canCreateWorker(type: WorkerType): boolean {
    const totalWorkers = this.workers.size;
    const typeWorkers = Array.from(this.workerInfo.values())
      .filter(info => info.type === type).length;
    
    const workerConfig = this.workerConfigs.get(type);
    const maxConcurrency = workerConfig?.maxConcurrency || this.config.maxWorkers;

    return totalWorkers < this.config.maxWorkers && typeWorkers < maxConcurrency;
  }

  /**
   * Create a new worker
   */
  private async createWorker(type: WorkerType): Promise<{ id: string; worker: Worker } | null> {
    const workerConfig = this.workerConfigs.get(type);
    if (!workerConfig) {
      throw new Error(`No configuration found for worker type: ${type}`);
    }

    try {
      const workerId = randomUUID();
      const worker = new Worker(workerConfig.scriptPath, {
        env: workerConfig.env,
        resourceLimits: workerConfig.memoryLimit ? {
          maxOldGenerationSizeMb: workerConfig.memoryLimit,
          maxYoungGenerationSizeMb: Math.floor(workerConfig.memoryLimit * 0.1),
        } : undefined,
      });

      // Set up worker event handlers
      this.setupWorkerEventHandlers(workerId, worker, type);

      // Store worker info
      const info: WorkerInfo = {
        id: workerId,
        type,
        pid: worker.threadId,
        status: 'idle',
        createdAt: new Date(),
        lastUsed: new Date(),
        tasksCompleted: 0,
        tasksErrored: 0,
      };

      this.workers.set(workerId, worker);
      this.workerInfo.set(workerId, info);

      this.emit('worker.created', workerId, type);

      return { id: workerId, worker };
    } catch (error) {
      console.error(`Failed to create worker for type ${type}:`, error);
      return null;
    }
  }

  /**
   * Set up event handlers for a worker
   */
  private setupWorkerEventHandlers(workerId: string, worker: Worker, _type: WorkerType): void {
    worker.on('message', (message: WorkerMessage) => {
      this.handleWorkerMessage(workerId, message);
    });

    worker.on('error', (error: Error) => {
      this.handleWorkerError(workerId, error);
    });

    worker.on('exit', (code: number) => {
      this.handleWorkerExit(workerId, code);
    });

    worker.on('messageerror', (error: Error) => {
      console.error(`Message error from worker ${workerId}:`, error);
    });
  }

  /**
   * Handle messages from workers
   */
  private handleWorkerMessage(workerId: string, message: WorkerMessage): void {
    const workerInfo = this.workerInfo.get(workerId);
    if (!workerInfo) {
      return;
    }

    switch (message.type) {
      case 'result':
        this.handleTaskResult(workerId, message);
        break;
      case 'error':
        this.handleTaskError(workerId, message);
        break;
      case 'pong':
        // Health check response
        break;
      default:
        console.warn(`Unknown message type from worker ${workerId}:`, message.type);
    }
  }

  /**
   * Handle task result from worker
   */
  private handleTaskResult(workerId: string, message: WorkerMessage): void {
    if (!message.taskId) {
      return;
    }

    const pending = this.pendingTasks.get(message.taskId);
    if (!pending) {
      return;
    }

    const result: WorkerResult = {
      taskId: message.taskId,
      success: true,
      result: message.payload,
      executionTime: Date.now(), // This should be sent from worker
      workerId,
      metadata: {},
    };

    // Update statistics
    this.updateWorkerStats(workerId, true, result.executionTime);
    this.taskQueue.markTaskCompleted(message.taskId, result.executionTime);

    // Clean up and resolve
    this.pendingTasks.delete(message.taskId);
    pending.resolve(result);

    this.emit('task.completed', result);
    this.continueProcessing(workerId);
  }

  /**
   * Handle task error from worker
   */
  private handleTaskError(workerId: string, message: WorkerMessage): void {
    if (!message.taskId) {
      return;
    }

    const pending = this.pendingTasks.get(message.taskId);
    if (!pending) {
      return;
    }

    const result: WorkerResult = {
      taskId: message.taskId,
      success: false,
      error: message.error || { message: 'Unknown worker error' },
      executionTime: Date.now(), // This should be sent from worker
      workerId,
    };

    // Update statistics
    this.updateWorkerStats(workerId, false, result.executionTime);
    this.taskQueue.markTaskFailed(message.taskId, result.executionTime);

    // Clean up and resolve
    this.pendingTasks.delete(message.taskId);
    pending.resolve(result); // Resolve with error result instead of rejecting

    this.emit('task.failed', result);
    this.continueProcessing(workerId);
  }

  /**
   * Handle worker errors
   */
  private handleWorkerError(workerId: string, error: Error): void {
    console.error(`Worker ${workerId} error:`, error);
    
    const workerInfo = this.workerInfo.get(workerId);
    if (workerInfo) {
      workerInfo.status = 'crashed';
      this.emit('worker.crashed', workerId, error);
    }

    // Handle any pending task for this worker
    const currentTask = workerInfo?.currentTask;
    if (currentTask) {
      const pending = this.pendingTasks.get(currentTask);
      if (pending) {
        const errorResult: WorkerResult = {
          taskId: currentTask,
          success: false,
          error: {
            message: `Worker crashed: ${error.message}`,
            code: 'WORKER_CRASHED',
          },
          executionTime: 0,
          workerId,
        };
        
        this.pendingTasks.delete(currentTask);
        pending.resolve(errorResult);
        this.emit('task.failed', errorResult);
      }
    }

    // Clean up worker
    this.cleanupWorker(workerId);
  }

  /**
   * Handle worker exit
   */
  private handleWorkerExit(workerId: string, code: number): void {
    console.log(`Worker ${workerId} exited with code ${code}`);
    this.cleanupWorker(workerId);
  }

  /**
   * Assign a task to a worker
   */
  private async assignTaskToWorker(workerId: string, task: WorkerTask): Promise<void> {
    const worker = this.workers.get(workerId);
    const workerInfo = this.workerInfo.get(workerId);
    
    if (!worker || !workerInfo) {
      throw new Error(`Worker ${workerId} not found`);
    }

    // Update worker status
    workerInfo.status = 'busy';
    workerInfo.currentTask = task.id;
    workerInfo.lastUsed = new Date();
    task.startedAt = new Date();

    // Send task to worker
    const message: WorkerMessage = {
      type: 'task',
      taskId: task.id,
      payload: {
        input: task.input,
        metadata: task.metadata,
      },
    };

    worker.postMessage(message);
    this.emit('task.started', task, workerId);
  }

  /**
   * Continue processing after a task completes
   */
  private continueProcessing(workerId: string): void {
    const workerInfo = this.workerInfo.get(workerId);
    if (workerInfo) {
      workerInfo.status = 'idle';
      workerInfo.currentTask = undefined;
    }

    // Process more tasks
    setImmediate(() => this.processQueue());
  }

  /**
   * Update worker statistics
   */
  private updateWorkerStats(workerId: string, success: boolean, executionTime: number): void {
    const workerInfo = this.workerInfo.get(workerId);
    if (!workerInfo) {
      return;
    }

    if (success) {
      workerInfo.tasksCompleted++;
      this.totalTasksProcessed++;
    } else {
      workerInfo.tasksErrored++;
      this.totalErrors++;
    }

    this.totalExecutionTime += executionTime;
  }

  /**
   * Clean up a worker
   */
  private cleanupWorker(workerId: string): void {
    const worker = this.workers.get(workerId);
    const workerInfo = this.workerInfo.get(workerId);

    if (worker) {
      worker.removeAllListeners();
      worker.terminate();
      this.workers.delete(workerId);
    }

    if (workerInfo) {
      this.emit('worker.destroyed', workerId, workerInfo.type);
      this.workerInfo.delete(workerId);
    }
  }

  /**
   * Start health monitoring
   */
  private startHealthMonitoring(): void {
    if (this.config.healthCheckInterval > 0) {
      this.healthCheckInterval = setInterval(() => {
        this.performHealthCheck();
      }, this.config.healthCheckInterval);
    }
  }

  /**
   * Perform health check on all workers
   */
  private async performHealthCheck(): Promise<void> {
    const promises = Array.from(this.workers.entries()).map(async ([workerId, worker]) => {
      try {
        // Send ping to worker
        const message: WorkerMessage = { type: 'ping' };
        worker.postMessage(message);
        
        // Update memory usage if available
        const workerInfo = this.workerInfo.get(workerId);
        if (workerInfo && this.config.enableMetrics) {
          // Memory usage would need to be reported by the worker
          // This is a placeholder for actual memory monitoring
        }
      } catch (error) {
        console.error(`Health check failed for worker ${workerId}:`, error);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Get worker pool statistics
   */
  getStats(): PoolStats {
    const workers = Array.from(this.workerInfo.values());
    const queueStats = this.taskQueue.getStats();

    return {
      totalWorkers: this.workers.size,
      activeWorkers: workers.filter(w => w.status === 'busy').length,
      idleWorkers: workers.filter(w => w.status === 'idle').length,
      crashedWorkers: workers.filter(w => w.status === 'crashed').length,
      queue: queueStats,
      uptime: Date.now() - this.startTime,
      totalTasksProcessed: this.totalTasksProcessed,
      totalErrors: this.totalErrors,
      averageTaskTime: this.totalTasksProcessed > 0 ? 
        this.totalExecutionTime / this.totalTasksProcessed : 0,
      memoryUsage: {
        total: 0, // Would need to aggregate from workers
        perWorker: 0, // Would need to calculate average
        peak: this.peakMemoryUsage,
      },
    };
  }

  /**
   * Get information about all workers
   */
  getWorkers(): WorkerInfo[] {
    return Array.from(this.workerInfo.values());
  }

  /**
   * Get information about a specific worker
   */
  getWorker(workerId: string): WorkerInfo | null {
    return this.workerInfo.get(workerId) || null;
  }

  /**
   * Terminate a specific worker
   */
  async terminateWorker(workerId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (worker) {
      await worker.terminate();
      this.cleanupWorker(workerId);
    }
  }

  /**
   * Restart all crashed workers
   */
  async restartCrashedWorkers(): Promise<void> {
    const crashedWorkers = Array.from(this.workerInfo.values())
      .filter(info => info.status === 'crashed');

    for (const workerInfo of crashedWorkers) {
      this.cleanupWorker(workerInfo.id);
      // The processQueue will create new workers as needed
    }

    this.processQueue();
  }

  /**
   * Pause task processing
   */
  pause(): void {
    this.paused = true;
  }

  /**
   * Resume task processing
   */
  resume(): void {
    this.paused = false;
    this.processQueue();
  }

  /**
   * Wait for all current tasks to complete
   */
  async drain(): Promise<void> {
    return new Promise<void>((resolve) => {
      const checkDrained = () => {
        if (this.taskQueue.isEmpty() && this.pendingTasks.size === 0) {
          this.emit('pool.drained');
          resolve();
        } else {
          setTimeout(checkDrained, 100);
        }
      };
      checkDrained();
    });
  }

  /**
   * Gracefully shutdown the pool
   */
  async shutdown(): Promise<void> {
    this.running = false;
    
    // Stop health monitoring
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Wait for current tasks to complete
    await this.drain();

    // Terminate all workers
    const terminationPromises = Array.from(this.workers.keys())
      .map(workerId => this.terminateWorker(workerId));
    
    await Promise.all(terminationPromises);

    // Clear all data
    this.workers.clear();
    this.workerInfo.clear();
    this.pendingTasks.clear();
    this.taskQueue.clear();

    this.removeAllListeners();
  }

  /**
   * Force shutdown the pool
   */
  async forceShutdown(): Promise<void> {
    this.running = false;
    
    // Stop health monitoring
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Cancel all pending tasks
    for (const [taskId, pending] of this.pendingTasks) {
      if (pending.timeout) {
        clearTimeout(pending.timeout);
      }
      const errorResult: WorkerResult = {
        taskId,
        success: false,
        error: { message: 'Pool force shutdown', code: 'POOL_SHUTDOWN' },
        executionTime: 0,
        workerId: 'shutdown',
      };
      pending.resolve(errorResult);
    }

    // Terminate all workers immediately
    const terminationPromises = Array.from(this.workers.values())
      .map(worker => worker.terminate());
    
    await Promise.allSettled(terminationPromises);

    // Clear all data
    this.workers.clear();
    this.workerInfo.clear();
    this.pendingTasks.clear();
    this.taskQueue.clear();

    this.removeAllListeners();
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    workers: Array<{ id: string; healthy: boolean; error?: string }>;
    queue: { size: number; oldestTask?: Date };
  }> {
    const workers = Array.from(this.workerInfo.values()).map(info => ({
      id: info.id,
      healthy: info.status !== 'crashed',
      error: info.status === 'crashed' ? 'Worker crashed' : undefined,
    }));

    const queueInfo = this.taskQueue.getDetailedInfo();
    const oldestTask = Object.values(queueInfo.oldestTaskByPriority)
      .filter(date => date !== null)
      .sort((a, b) => a!.getTime() - b!.getTime())[0] || undefined;

    return {
      healthy: this.running && !this.paused && workers.every(w => w.healthy),
      workers,
      queue: {
        size: this.taskQueue.size(),
        oldestTask,
      },
    };
  }
}
