/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  TaskPriority,
  WorkerType,
  WorkerTask,
  QueueStats,
  ITaskQueue,
} from './WorkerInterfaces.js';

/**
 * Priority queue implementation for worker tasks
 */
export class TaskQueue implements ITaskQueue {
  private queues = new Map<TaskPriority, WorkerTask[]>();
  private taskMap = new Map<string, WorkerTask>();
  private stats: QueueStats;
  private maxSize: number;

  constructor(maxSize = 10000) {
    this.maxSize = maxSize;
    this.stats = this.initializeStats();
    this.initializeQueues();
  }

  /**
   * Initialize priority queues
   */
  private initializeQueues(): void {
    Object.values(TaskPriority).forEach(priority => {
      if (typeof priority === 'number') {
        this.queues.set(priority, []);
      }
    });
  }

  /**
   * Initialize statistics
   */
  private initializeStats(): QueueStats {
    return {
      totalTasks: 0,
      pendingTasks: 0,
      runningTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      averageWaitTime: 0,
      averageExecutionTime: 0,
      tasksByPriority: {
        [TaskPriority.IMMEDIATE]: 0,
        [TaskPriority.HIGH]: 0,
        [TaskPriority.NORMAL]: 0,
        [TaskPriority.LOW]: 0,
        [TaskPriority.BACKGROUND]: 0,
      },
      tasksByType: {
        [WorkerType.FILE_PROCESSING]: 0,
        [WorkerType.EMBEDDING_GENERATION]: 0,
        [WorkerType.CODE_ANALYSIS]: 0,
        [WorkerType.MEMORY_ANALYSIS]: 0,
        [WorkerType.GENERIC]: 0,
      },
    };
  }

  /**
   * Add a task to the queue
   */
  async enqueue<TInput, TOutput>(task: WorkerTask<TInput, TOutput>): Promise<void> {
    if (this.size() >= this.maxSize) {
      throw new Error(`Queue is full. Maximum size: ${this.maxSize}`);
    }

    if (this.taskMap.has(task.id)) {
      throw new Error(`Task with ID '${task.id}' already exists in queue`);
    }

    const priorityQueue = this.queues.get(task.priority);
    if (!priorityQueue) {
      throw new Error(`Invalid priority: ${task.priority}`);
    }

    // Insert task in priority order (higher priority first)
    const insertIndex = this.findInsertIndex(priorityQueue, task);
    priorityQueue.splice(insertIndex, 0, task);
    
    this.taskMap.set(task.id, task);
    this.updateStatsOnEnqueue(task);
  }

  /**
   * Find the correct index to insert a task to maintain priority order
   */
  private findInsertIndex(queue: WorkerTask[], task: WorkerTask): number {
    let left = 0;
    let right = queue.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      const midTask = queue[mid];
      
      // If tasks have same priority, use creation time (FIFO)
      if (midTask.priority === task.priority) {
        if (midTask.createdAt <= task.createdAt) {
          left = mid + 1;
        } else {
          right = mid;
        }
      } else if (midTask.priority < task.priority) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    return left;
  }

  /**
   * Remove and return the highest priority task
   */
  async dequeue(workerType?: WorkerType): Promise<WorkerTask | null> {
    // Get priorities in descending order
    const priorities = Array.from(this.queues.keys()).sort((a, b) => b - a);

    for (const priority of priorities) {
      const queue = this.queues.get(priority)!;
      
      if (queue.length === 0) {
        continue;
      }

      // If worker type is specified, find matching task
      if (workerType) {
        const index = queue.findIndex(task => task.type === workerType);
        if (index !== -1) {
          const task = queue.splice(index, 1)[0];
          this.taskMap.delete(task.id);
          this.updateStatsOnDequeue(task);
          return task;
        }
      } else {
        // Return the first task (highest priority within this level)
        const task = queue.shift()!;
        this.taskMap.delete(task.id);
        this.updateStatsOnDequeue(task);
        return task;
      }
    }

    return null;
  }

  /**
   * Peek at the highest priority task without removing it
   */
  peek(workerType?: WorkerType): WorkerTask | null {
    const priorities = Array.from(this.queues.keys()).sort((a, b) => b - a);

    for (const priority of priorities) {
      const queue = this.queues.get(priority)!;
      
      if (queue.length === 0) {
        continue;
      }

      if (workerType) {
        const task = queue.find(task => task.type === workerType);
        if (task) {
          return task;
        }
      } else {
        return queue[0];
      }
    }

    return null;
  }

  /**
   * Remove a specific task from the queue
   */
  remove(taskId: string): boolean {
    const task = this.taskMap.get(taskId);
    if (!task) {
      return false;
    }

    const queue = this.queues.get(task.priority);
    if (!queue) {
      return false;
    }

    const index = queue.findIndex(t => t.id === taskId);
    if (index === -1) {
      return false;
    }

    queue.splice(index, 1);
    this.taskMap.delete(taskId);
    this.updateStatsOnRemove(task);
    return true;
  }

  /**
   * Clear all tasks from the queue
   */
  clear(): void {
    this.queues.forEach(queue => queue.length = 0);
    this.taskMap.clear();
    this.stats = this.initializeStats();
  }

  /**
   * Get the total number of tasks in the queue
   */
  size(): number {
    return this.taskMap.size;
  }

  /**
   * Check if the queue is empty
   */
  isEmpty(): boolean {
    return this.taskMap.size === 0;
  }

  /**
   * Get tasks by priority
   */
  getTasksByPriority(priority: TaskPriority): WorkerTask[] {
    return [...(this.queues.get(priority) || [])];
  }

  /**
   * Get tasks by type
   */
  getTasksByType(type: WorkerType): WorkerTask[] {
    const tasks: WorkerTask[] = [];
    this.queues.forEach(queue => {
      tasks.push(...queue.filter(task => task.type === type));
    });
    return tasks;
  }

  /**
   * Get queue statistics
   */
  getStats(): QueueStats {
    this.updateCurrentStats();
    return { ...this.stats };
  }

  /**
   * Update statistics when a task is enqueued
   */
  private updateStatsOnEnqueue(task: WorkerTask): void {
    this.stats.totalTasks++;
    this.stats.pendingTasks++;
    this.stats.tasksByPriority[task.priority]++;
    this.stats.tasksByType[task.type]++;
  }

  /**
   * Update statistics when a task is dequeued
   */
  private updateStatsOnDequeue(task: WorkerTask): void {
    this.stats.pendingTasks--;
    this.stats.runningTasks++;
    
    // Update average wait time
    const waitTime = Date.now() - task.createdAt.getTime();
    this.stats.averageWaitTime = this.updateAverage(
      this.stats.averageWaitTime,
      waitTime,
      this.stats.runningTasks + this.stats.completedTasks + this.stats.failedTasks
    );
  }

  /**
   * Update statistics when a task is removed
   */
  private updateStatsOnRemove(task: WorkerTask): void {
    this.stats.pendingTasks--;
    this.stats.tasksByPriority[task.priority]--;
    this.stats.tasksByType[task.type]--;
  }

  /**
   * Update current statistics
   */
  private updateCurrentStats(): void {
    this.stats.pendingTasks = this.size();
    
    // Recalculate tasks by priority
    Object.keys(this.stats.tasksByPriority).forEach(priority => {
      const priorityValue = parseInt(priority, 10) as TaskPriority;
      const queue = this.queues.get(priorityValue);
      this.stats.tasksByPriority[priorityValue] = queue ? queue.length : 0;
    });

    // Recalculate tasks by type
    Object.keys(this.stats.tasksByType).forEach(type => {
      const typeValue = type as WorkerType;
      this.stats.tasksByType[typeValue] = this.getTasksByType(typeValue).length;
    });
  }

  /**
   * Update a running average
   */
  private updateAverage(currentAverage: number, newValue: number, count: number): number {
    if (count === 1) {
      return newValue;
    }
    return ((currentAverage * (count - 1)) + newValue) / count;
  }

  /**
   * Mark a task as completed (for statistics)
   */
  markTaskCompleted(taskId: string, executionTime: number): void {
    this.stats.runningTasks--;
    this.stats.completedTasks++;
    
    this.stats.averageExecutionTime = this.updateAverage(
      this.stats.averageExecutionTime,
      executionTime,
      this.stats.completedTasks
    );
  }

  /**
   * Mark a task as failed (for statistics)
   */
  markTaskFailed(taskId: string, executionTime: number): void {
    this.stats.runningTasks--;
    this.stats.failedTasks++;
    
    this.stats.averageExecutionTime = this.updateAverage(
      this.stats.averageExecutionTime,
      executionTime,
      this.stats.completedTasks + this.stats.failedTasks
    );
  }

  /**
   * Get detailed queue information for debugging
   */
  getDetailedInfo(): {
    queueSizes: Record<TaskPriority, number>;
    oldestTaskByPriority: Record<TaskPriority, Date | null>;
    tasksByWorkerType: Record<WorkerType, number>;
  } {
    const queueSizes: Record<TaskPriority, number> = {} as Record<TaskPriority, number>;
    const oldestTaskByPriority: Record<TaskPriority, Date | null> = {} as Record<TaskPriority, Date | null>;
    const tasksByWorkerType: Record<WorkerType, number> = {} as Record<WorkerType, number>;

    // Initialize counters
    Object.values(WorkerType).forEach(type => {
      tasksByWorkerType[type] = 0;
    });

    // Analyze each priority queue
    this.queues.forEach((queue, priority) => {
      queueSizes[priority] = queue.length;
      oldestTaskByPriority[priority] = queue.length > 0 ? queue[0].createdAt : null;
      
      // Count tasks by worker type in this priority
      queue.forEach(task => {
        tasksByWorkerType[task.type]++;
      });
    });

    return {
      queueSizes,
      oldestTaskByPriority,
      tasksByWorkerType,
    };
  }

  /**
   * Get tasks that have been waiting longer than the specified time
   */
  getStaleTasksOlderThan(maxWaitTimeMs: number): WorkerTask[] {
    const now = Date.now();
    const staleTasks: WorkerTask[] = [];

    this.queues.forEach(queue => {
      queue.forEach(task => {
        if (now - task.createdAt.getTime() > maxWaitTimeMs) {
          staleTasks.push(task);
        }
      });
    });

    return staleTasks;
  }

  /**
   * Compact the queue by removing completed/failed tasks older than specified time
   */
  compact(_retentionTimeMs: number): number {
    // This implementation assumes we're only storing pending tasks
    // In a full implementation, you might want to keep completed/failed tasks for a while
    const before = this.size();
    
    // For now, just return 0 as we don't store completed tasks in the queue
    return before - this.size();
  }
}
