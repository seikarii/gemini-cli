/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Action System - Sistema de Acción Principal
 * ==========================================
 *
 * Gestiona y ejecuta las herramientas disponibles del agente.
 * Actúa como un despachador de acciones inteligente, con integración de priorización adaptativa,
 * registro de uso de herramientas, y soporte para métricas avanzadas de rendimiento.
 */

import { v4 as uuidv4 } from 'uuid';

/**
 * Prioridades de acciones
 */
export enum ActionPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * Estados de acciones
 */
export enum ActionStatus {
  PENDING = 'pending',
  EXECUTING = 'executing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Representa una acción a ejecutar
 */
export interface Action {
  id: string;
  toolName: string;
  parameters: Record<string, unknown>;
  priority: ActionPriority;
  status: ActionStatus;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  result?: unknown;
  error?: string;
  metadata: Record<string, unknown>;
  callback?: (result: unknown) => void;
}

/**
 * Configuración del ActionSystem
 */
export interface ActionSystemConfig {
  maxConcurrentActions: number;
  maxQueueSize: number;
  enableActionHistory: boolean;
  maxHistorySize: number;
  autoCleanupCompleted: boolean;
  cleanupInterval: number; // segundos
  enablePriorityScheduling: boolean;
  defaultTimeout: number; // segundos
  adaptivePriority: boolean;
}

/**
 * Estadísticas del ActionSystem
 */
export interface ActionSystemStats {
  totalActionsCreated: number;
  totalActionsCompleted: number;
  totalActionsFailed: number;
  totalActionsCancelled: number;
  averageExecutionTime: number;
  toolsUsage: Record<string, number>;
  priorityStats: Record<string, number>;
  lastToolUsed?: string;
  successRate: number;
  failureRate: number;
  isRunning: boolean;
  queueSize: number;
  activeActionsCount: number;
  completedActionsCount: number;
}

/**
 * Sistema de Acción principal para el Agente
 */
export class ActionSystem {
  private actionQueue: Action[] = [];
  private activeActions: Map<string, Action> = new Map();
  private completedActions: Action[] = [];
  private toolRegistry: Map<string, (...args: unknown[]) => Promise<unknown>> =
    new Map();

  private config: ActionSystemConfig = {
    maxConcurrentActions: 3,
    maxQueueSize: 100,
    enableActionHistory: true,
    maxHistorySize: 1000,
    autoCleanupCompleted: true,
    cleanupInterval: 3600, // 1 hora
    enablePriorityScheduling: true,
    defaultTimeout: 30.0, // segundos
    adaptivePriority: true,
  };

  private stats = {
    totalActionsCreated: 0,
    totalActionsCompleted: 0,
    totalActionsFailed: 0,
    totalActionsCancelled: 0,
    averageExecutionTime: 0.0,
    toolsUsage: {} as Record<string, number>,
    priorityStats: {
      [ActionPriority.LOW]: 0,
      [ActionPriority.NORMAL]: 0,
      [ActionPriority.HIGH]: 0,
      [ActionPriority.CRITICAL]: 0,
    },
    lastToolUsed: undefined as string | undefined,
  };

  private isRunning = false;

  /**
   * Inicializa el Sistema de Acción
   */
  constructor(config?: Partial<ActionSystemConfig>) {
    if (config) {
      this.config = { ...this.config, ...config };
    }
    console.log('⚡ Sistema de Acción inicializado');
  }

  /**
   * Registra una herramienta disponible para ejecución
   */
  registerTool(
    toolName: string,
    toolFn: (...args: unknown[]) => Promise<unknown>,
  ): void {
    this.toolRegistry.set(toolName, toolFn);
    console.log(`🔧 Herramienta registrada: ${toolName}`);
  }

  /**
   * Crea una acción sin ejecutarla inmediatamente
   */
  createAction(
    toolName: string,
    parameters: Record<string, unknown> = {},
    priority: ActionPriority = ActionPriority.NORMAL,
    metadata: Record<string, unknown> = {},
    callback?: (result: unknown) => void,
  ): string {
    const actionId = uuidv4();

    const action: Action = {
      id: actionId,
      toolName,
      parameters,
      priority,
      status: ActionStatus.PENDING,
      createdAt: new Date().toISOString(),
      metadata,
      callback,
    };

    // Añadir a la cola
    this.actionQueue.push(action);

    // Actualizar estadísticas
    this.stats.totalActionsCreated++;
    this.stats.priorityStats[priority]++;

    // Ordenar cola por prioridad
    if (this.config.enablePriorityScheduling) {
      this.sortActionQueue();
    }

    // Limitar tamaño de la cola
    if (this.actionQueue.length > this.config.maxQueueSize) {
      this.actionQueue = this.actionQueue.slice(-this.config.maxQueueSize);
    }

    console.log(`⚡ Acción creada: ${actionId} (${toolName})`);
    return actionId;
  }

  /**
   * Ejecuta la siguiente acción en la cola si hay capacidad
   */
  async executeNextAction(): Promise<string | null> {
    if (!this.isRunning || this.actionQueue.length === 0) {
      return null;
    }

    if (this.activeActions.size >= this.config.maxConcurrentActions) {
      console.log('⏳ Límite de acciones concurrentes alcanzado');
      return null;
    }

    const action = this.actionQueue.shift()!;
    action.status = ActionStatus.EXECUTING;
    action.startedAt = new Date().toISOString();
    this.activeActions.set(action.id, action);
    this.stats.lastToolUsed = action.toolName;

    try {
      const toolFn = this.toolRegistry.get(action.toolName);
      if (!toolFn) {
        throw new Error(`Herramienta '${action.toolName}' no registrada`);
      }

      const result = await toolFn(...Object.values(action.parameters));
      action.result = result;
      action.status = ActionStatus.COMPLETED;
      action.completedAt = new Date().toISOString();

      this.stats.totalActionsCompleted++;
      this.stats.toolsUsage[action.toolName] =
        (this.stats.toolsUsage[action.toolName] || 0) + 1;

      if (action.callback) {
        action.callback(result);
      }

      console.log(`✅ Acción ejecutada: ${action.id} (${action.toolName})`);
    } catch (error) {
      action.error = error instanceof Error ? error.message : String(error);
      action.status = ActionStatus.FAILED;
      action.completedAt = new Date().toISOString();
      this.stats.totalActionsFailed++;
      console.error(`❌ Error en acción ${action.id}: ${action.error}`);
    }

    this.completedActions.push(this.activeActions.get(action.id)!);
    this.activeActions.delete(action.id);

    console.log(`Action ${action.id} status after execution: ${action.status}`);
    this.updateAverageExecutionTime(action);

    return action.id;
  }

  /**
   * Cancela una acción
   */
  cancelAction(actionId: string): boolean {
    // Buscar en cola
    const queueIndex = this.actionQueue.findIndex(
      (action) => action.id === actionId,
    );
    if (queueIndex !== -1) {
      const action = this.actionQueue[queueIndex];
      action.status = ActionStatus.CANCELLED;
      action.completedAt = new Date().toISOString();
      this.completedActions.push(this.actionQueue.splice(queueIndex, 1)[0]);
      this.stats.totalActionsCancelled++;
      console.log(`⚡ Acción cancelada: ${actionId}`);
      return true;
    }

    // Buscar en acciones activas
    if (this.activeActions.has(actionId)) {
      const action = this.activeActions.get(actionId)!;
      action.status = ActionStatus.CANCELLED;
      action.completedAt = new Date().toISOString();
      this.completedActions.push(action);
      this.activeActions.delete(actionId);
      this.stats.totalActionsCancelled++;
      console.log(`⚡ Acción activa cancelada: ${actionId}`);
      return true;
    }

    console.warn(`⚠️ Acción no encontrada para cancelar: ${actionId}`);
    return false;
  }

  /**
   * Obtiene el estado de una acción
   */
  getActionStatus(actionId: string): Record<string, unknown> | null {
    // Buscar en cola
    const queueAction = this.actionQueue.find(
      (action) => action.id === actionId,
    );
    if (queueAction) {
      return this.actionToDict(queueAction);
    }

    // Buscar en activas
    if (this.activeActions.has(actionId)) {
      return this.actionToDict(this.activeActions.get(actionId)!);
    }

    // Buscar en completadas
    const completedAction = this.completedActions.find(
      (action) => action.id === actionId,
    );
    if (completedAction) {
      return this.actionToDict(completedAction);
    }

    return null;
  }

  /**
   * Lista acciones con filtrado opcional por estado
   */
  listActions(status?: ActionStatus): Array<Record<string, unknown>> {
    const actions: Action[] = [];

    // Acciones en cola
    if (!status || status === ActionStatus.PENDING) {
      actions.push(...this.actionQueue);
    }

    // Acciones activas
    if (!status || status === ActionStatus.EXECUTING) {
      actions.push(...Array.from(this.activeActions.values()));
    }

    // Acciones completadas
    if (!status) {
      actions.push(...this.completedActions);
    } else if (
      [
        ActionStatus.COMPLETED,
        ActionStatus.FAILED,
        ActionStatus.CANCELLED,
      ].includes(status)
    ) {
      actions.push(
        ...this.completedActions.filter((action) => action.status === status),
      );
    }

    return actions.map((action) => this.actionToDict(action));
  }

  /**
   * Inicia el sistema de acción
   */
  start(): void {
    if (this.isRunning) {
      console.warn('⚠️ El sistema de acción ya está en ejecución');
      return;
    }

    this.isRunning = true;
    console.log('⚡ Sistema de Acción iniciado');
  }

  /**
   * Detiene el sistema de acción
   */
  stop(): void {
    if (!this.isRunning) {
      console.warn('⚠️ El sistema de acción no está en ejecución');
      return;
    }

    this.isRunning = false;

    // Cancelar todas las acciones activas
    for (const actionId of this.activeActions.keys()) {
      this.cancelAction(actionId);
    }

    console.log('⚡ Sistema de Acción detenido');
  }

  /**
   * Devuelve estadísticas del sistema de acción
   */
  getStats(): ActionSystemStats {
    const stats: ActionSystemStats = {
      totalActionsCreated: this.stats.totalActionsCreated,
      totalActionsCompleted: this.stats.totalActionsCompleted,
      totalActionsFailed: this.stats.totalActionsFailed,
      totalActionsCancelled: this.stats.totalActionsCancelled,
      averageExecutionTime: this.stats.averageExecutionTime,
      toolsUsage: { ...this.stats.toolsUsage },
      priorityStats: { ...this.stats.priorityStats },
      lastToolUsed: this.stats.lastToolUsed,
      successRate: 0,
      failureRate: 0,
      isRunning: this.isRunning,
      queueSize: this.actionQueue.length,
      activeActionsCount: this.activeActions.size,
      completedActionsCount: this.completedActions.length,
    };

    // Calcular tasas
    const totalCompleted = stats.totalActionsCompleted;
    const totalFailed = stats.totalActionsFailed;
    const totalProcessed = totalCompleted + totalFailed;

    if (totalProcessed > 0) {
      stats.successRate = totalCompleted / totalProcessed;
      stats.failureRate = totalFailed / totalProcessed;
    } else {
      stats.successRate = 0.0;
      stats.failureRate = 0.0;
    }

    return stats;
  }

  /**
   * Configura parámetros del sistema
   */
  configure(config: Partial<ActionSystemConfig>): void {
    this.config = { ...this.config, ...config };
    console.log(`⚙️ Configuración de sistema de acción actualizada:`, config);
  }

  /**
   * Reinicia las estadísticas
   */
  resetStats(): void {
    this.stats = {
      totalActionsCreated: 0,
      totalActionsCompleted: 0,
      totalActionsFailed: 0,
      totalActionsCancelled: 0,
      averageExecutionTime: 0.0,
      toolsUsage: {},
      priorityStats: {
        [ActionPriority.LOW]: 0,
        [ActionPriority.NORMAL]: 0,
        [ActionPriority.HIGH]: 0,
        [ActionPriority.CRITICAL]: 0,
      },
      lastToolUsed: undefined,
    };

    // Limpiar acciones completadas
    this.completedActions = [];
    console.log('📊 Estadísticas de sistema de acción reiniciadas');
  }

  /**
   * Ordena la cola de acciones por prioridad
   */
  private sortActionQueue(): void {
    const priorityOrder = {
      [ActionPriority.CRITICAL]: 0,
      [ActionPriority.HIGH]: 1,
      [ActionPriority.NORMAL]: 2,
      [ActionPriority.LOW]: 3,
    };

    this.actionQueue.sort(
      (a, b) => priorityOrder[a.priority] - priorityOrder[b.priority],
    );
  }

  /**
   * Convierte una acción a diccionario
   */
  private actionToDict(action: Action): Record<string, unknown> {
    return {
      id: action.id,
      toolName: action.toolName,
      parameters: action.parameters,
      priority: action.priority,
      status: action.status,
      createdAt: action.createdAt,
      startedAt: action.startedAt,
      completedAt: action.completedAt,
      hasResult: action.result !== undefined,
      hasError: action.error !== undefined,
      error: action.error,
      metadata: action.metadata,
    };
  }

  /**
   * Actualiza la métrica de tiempo promedio de ejecución
   */
  private updateAverageExecutionTime(action: Action): void {
    try {
      if (action.startedAt && action.completedAt) {
        const start = new Date(action.startedAt).getTime();
        const end = new Date(action.completedAt).getTime();
        const elapsed = (end - start) / 1000; // segundos

        const total =
          this.stats.totalActionsCompleted + this.stats.totalActionsFailed;
        const prevAvg = this.stats.averageExecutionTime;
        this.stats.averageExecutionTime =
          total > 0 ? (prevAvg * (total - 1) + elapsed) / total : elapsed;
      }
    } catch (error) {
      console.error(
        `Error al actualizar tiempo promedio de ejecución: ${error}`,
      );
    }
  }
}
