/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Prioridades de acciones
 */
export declare enum ActionPriority {
    LOW = "low",
    NORMAL = "normal",
    HIGH = "high",
    CRITICAL = "critical"
}
/**
 * Estados de acciones
 */
export declare enum ActionStatus {
    PENDING = "pending",
    EXECUTING = "executing",
    COMPLETED = "completed",
    FAILED = "failed",
    CANCELLED = "cancelled"
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
    cleanupInterval: number;
    enablePriorityScheduling: boolean;
    defaultTimeout: number;
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
export declare class ActionSystem {
    private actionQueue;
    private activeActions;
    private completedActions;
    private toolRegistry;
    private config;
    private stats;
    private isRunning;
    /**
     * Inicializa el Sistema de Acción
     */
    constructor(config?: Partial<ActionSystemConfig>);
    /**
     * Registra una herramienta disponible para ejecución
     */
    registerTool(toolName: string, toolFn: (...args: unknown[]) => Promise<unknown>): void;
    /**
     * Crea una acción sin ejecutarla inmediatamente
     */
    createAction(toolName: string, parameters?: Record<string, unknown>, priority?: ActionPriority, metadata?: Record<string, unknown>, callback?: (result: unknown) => void): string;
    /**
     * Ejecuta la siguiente acción en la cola si hay capacidad
     */
    executeNextAction(): Promise<string | null>;
    /**
     * Cancela una acción
     */
    cancelAction(actionId: string): boolean;
    /**
     * Obtiene el estado de una acción
     */
    getActionStatus(actionId: string): Record<string, unknown> | null;
    /**
     * Lista acciones con filtrado opcional por estado
     */
    listActions(status?: ActionStatus): Array<Record<string, unknown>>;
    /**
     * Inicia el sistema de acción
     */
    start(): void;
    /**
     * Detiene el sistema de acción
     */
    stop(): void;
    /**
     * Devuelve estadísticas del sistema de acción
     */
    getStats(): ActionSystemStats;
    /**
     * Configura parámetros del sistema
     */
    configure(config: Partial<ActionSystemConfig>): void;
    /**
     * Reinicia las estadísticas
     */
    resetStats(): void;
    /**
     * Ordena la cola de acciones por prioridad
     */
    private sortActionQueue;
    /**
     * Convierte una acción a diccionario
     */
    private actionToDict;
    /**
     * Actualiza la métrica de tiempo promedio de ejecución
     */
    private updateAverageExecutionTime;
}
