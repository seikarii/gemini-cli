/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * Sistema de Prompts Avanzado - Mini-Lenguaje de Scripting para Acciones
 * ====================================================================
 *
 * Permite al LLM expresar secuencias complejas de acciones, lógica condicional
 * y ejecución paralela dentro de una única salida generada.
 */
import { ActionPriority } from './action-system.js';
/**
 * Tipos de nodos en el script de acciones
 */
export declare enum ScriptNodeType {
    ACTION = "action",
    SEQUENCE = "sequence",
    PARALLEL = "parallel",
    CONDITION = "condition",
    LOOP = "loop",
    VARIABLE = "variable"
}
/**
 * Nodo de acción básico
 */
export interface ActionNode {
    type: ScriptNodeType.ACTION;
    toolName: string;
    parameters: Record<string, unknown>;
    priority?: ActionPriority;
    id?: string;
    description?: string;
}
/**
 * Nodo de secuencia (ejecución secuencial)
 */
export interface SequenceNode {
    type: ScriptNodeType.SEQUENCE;
    nodes: ScriptNode[];
    description?: string;
}
/**
 * Nodo paralelo (ejecución concurrente)
 */
export interface ParallelNode {
    type: ScriptNodeType.PARALLEL;
    nodes: ScriptNode[];
    maxConcurrency?: number;
    description?: string;
}
/**
 * Nodo condicional
 */
export interface ConditionNode {
    type: ScriptNodeType.CONDITION;
    condition: string;
    thenNode: ScriptNode;
    elseNode?: ScriptNode;
    description?: string;
}
/**
 * Nodo de bucle
 */
export interface LoopNode {
    type: ScriptNodeType.LOOP;
    variable: string;
    iterable: unknown[];
    body: ScriptNode;
    description?: string;
}
/**
 * Nodo de variable
 */
export interface VariableNode {
    type: ScriptNodeType.VARIABLE;
    name: string;
    value: unknown;
    description?: string;
}
/**
 * Unión de todos los tipos de nodos
 */
export type ScriptNode = ActionNode | SequenceNode | ParallelNode | ConditionNode | LoopNode | VariableNode;
/**
 * Script completo de acciones
 */
export interface ActionScript {
    id: string;
    name?: string;
    description?: string;
    version?: string;
    rootNode: ScriptNode;
    variables?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
}
/**
 * Contexto de ejecución del script
 */
export interface ScriptExecutionContext {
    variables: Map<string, unknown>;
    results: Map<string, unknown>;
    currentNode?: ScriptNode;
    parentContext?: ScriptExecutionContext;
}
/**
 * Resultado de ejecución de script
 */
export interface ScriptExecutionResult {
    success: boolean;
    result?: unknown;
    error?: string;
    executionTime: number;
    executedNodes: number;
    failedNodes: number;
}
/**
 * Parser para el mini-lenguaje de scripting
 */
export declare class ActionScriptParser {
    /**
     * Parsea un script desde formato string
     */
    parse(scriptText: string): ActionScript;
    /**
     * Valida la estructura del script
     */
    private validateScript;
    /**
     * Valida un nodo recursivamente
     */
    private validateNode;
    /**
     * Convierte un script a formato string
     */
    stringify(script: ActionScript): string;
}
/**
 * Generador de scripts de acciones
 */
export declare class ActionScriptBuilder {
    private script;
    constructor(name?: string, description?: string);
    /**
     * Establece la raíz del script
     */
    setRoot(node: ScriptNode): ActionScriptBuilder;
    /**
     * Crea un nodo de acción
     */
    static action(toolName: string, parameters: Record<string, unknown>, priority?: ActionPriority, description?: string): ActionNode;
    /**
     * Crea un nodo de secuencia
     */
    static sequence(nodes: ScriptNode[], description?: string): SequenceNode;
    /**
     * Crea un nodo paralelo
     */
    static parallel(nodes: ScriptNode[], maxConcurrency?: number, description?: string): ParallelNode;
    /**
     * Crea un nodo condicional
     */
    static condition(condition: string, thenNode: ScriptNode, elseNode?: ScriptNode, description?: string): ConditionNode;
    /**
     * Crea un nodo de bucle
     */
    static loop(variable: string, iterable: unknown[], body: ScriptNode, description?: string): LoopNode;
    /**
     * Crea un nodo de variable
     */
    static variable(name: string, value: unknown, description?: string): VariableNode;
    /**
     * Construye el script final
     */
    build(): ActionScript;
}
/**
 * Evaluador de condiciones para nodos condicionales
 */
export declare class ConditionEvaluator {
    /**
     * Evalúa una condición en el contexto dado
     */
    evaluate(condition: string, context: ScriptExecutionContext): boolean;
}
