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
export enum ScriptNodeType {
  ACTION = 'action',
  SEQUENCE = 'sequence',
  PARALLEL = 'parallel',
  CONDITION = 'condition',
  LOOP = 'loop',
  VARIABLE = 'variable',
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
  condition: string; // Expresión de condición
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
export type ScriptNode =
  | ActionNode
  | SequenceNode
  | ParallelNode
  | ConditionNode
  | LoopNode
  | VariableNode;

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
export class ActionScriptParser {
  /**
   * Parsea un script desde formato string
   */
  parse(scriptText: string): ActionScript {
    // Implementación básica - se puede extender con un parser más sofisticado
    try {
      const parsed = JSON.parse(scriptText);
      return this.validateScript(parsed);
    } catch (error) {
      throw new Error(`Error parsing action script: ${error}`);
    }
  }

  /**
   * Valida la estructura del script
   */
  private validateScript(script: unknown): ActionScript {
    const scriptObj = script as Record<string, unknown>;
    if (!scriptObj['id'] || !scriptObj['rootNode']) {
      throw new Error('Invalid script: missing id or rootNode');
    }

    this.validateNode(scriptObj['rootNode']);

    return script as ActionScript;
  }

  /**
   * Valida un nodo recursivamente
   */
  private validateNode(node: unknown): void {
    const nodeObj = node as Record<string, unknown>;
    if (!nodeObj['type']) {
      throw new Error('Invalid node: missing type');
    }

    switch (nodeObj['type']) {
      case ScriptNodeType.ACTION:
        if (!nodeObj['toolName']) {
          throw new Error('Invalid action node: missing toolName');
        }
        break;

      case ScriptNodeType.SEQUENCE:
      case ScriptNodeType.PARALLEL:
        if (!Array.isArray(nodeObj['nodes'])) {
          throw new Error(`Invalid ${nodeObj['type']} node: nodes must be an array`);
        }
        (nodeObj['nodes'] as unknown[]).forEach((childNode: unknown) => this.validateNode(childNode));
        break;

      case ScriptNodeType.CONDITION:
        if (!nodeObj['condition'] || !nodeObj['thenNode']) {
          throw new Error('Invalid condition node: missing condition or thenNode');
        }
        this.validateNode(nodeObj['thenNode']);
        if (nodeObj['elseNode']) {
          this.validateNode(nodeObj['elseNode']);
        }
        break;

      case ScriptNodeType.LOOP:
        if (!nodeObj['variable'] || !Array.isArray(nodeObj['iterable']) || !nodeObj['body']) {
          throw new Error('Invalid loop node: missing variable, iterable, or body');
        }
        this.validateNode(nodeObj['body']);
        break;

      case ScriptNodeType.VARIABLE:
        if (!nodeObj['name']) {
          throw new Error('Invalid variable node: missing name');
        }
        break;

      default:
        throw new Error(`Unknown node type: ${nodeObj['type']}`);
    }
  }

  /**
   * Convierte un script a formato string
   */
  stringify(script: ActionScript): string {
    return JSON.stringify(script, null, 2);
  }
}

/**
 * Generador de scripts de acciones
 */
export class ActionScriptBuilder {
  private script: Partial<ActionScript>;

  constructor(name?: string, description?: string) {
    this.script = {
      id: `script_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      description,
      rootNode: {} as ScriptNode,
    };
  }

  /**
   * Establece la raíz del script
   */
  setRoot(node: ScriptNode): ActionScriptBuilder {
    this.script.rootNode = node;
    return this;
  }

  /**
   * Crea un nodo de acción
   */
  static action(
    toolName: string,
    parameters: Record<string, unknown>,
    priority?: ActionPriority,
    description?: string
  ): ActionNode {
    return {
      type: ScriptNodeType.ACTION,
      toolName,
      parameters,
      priority,
      description,
    };
  }

  /**
   * Crea un nodo de secuencia
   */
  static sequence(nodes: ScriptNode[], description?: string): SequenceNode {
    return {
      type: ScriptNodeType.SEQUENCE,
      nodes,
      description,
    };
  }

  /**
   * Crea un nodo paralelo
   */
  static parallel(
    nodes: ScriptNode[],
    maxConcurrency?: number,
    description?: string
  ): ParallelNode {
    return {
      type: ScriptNodeType.PARALLEL,
      nodes,
      maxConcurrency,
      description,
    };
  }

  /**
   * Crea un nodo condicional
   */
  static condition(
    condition: string,
    thenNode: ScriptNode,
    elseNode?: ScriptNode,
    description?: string
  ): ConditionNode {
    return {
      type: ScriptNodeType.CONDITION,
      condition,
      thenNode,
      elseNode,
      description,
    };
  }

  /**
   * Crea un nodo de bucle
   */
  static loop(
    variable: string,
    iterable: unknown[],
    body: ScriptNode,
    description?: string
  ): LoopNode {
    return {
      type: ScriptNodeType.LOOP,
      variable,
      iterable,
      body,
      description,
    };
  }

  /**
   * Crea un nodo de variable
   */
  static variable(name: string, value: unknown, description?: string): VariableNode {
    return {
      type: ScriptNodeType.VARIABLE,
      name,
      value,
      description,
    };
  }

  /**
   * Construye el script final
   */
  build(): ActionScript {
    if (!this.script.rootNode || Object.keys(this.script.rootNode).length === 0) {
      throw new Error('Cannot build script: rootNode is not set');
    }

    return this.script as ActionScript;
  }
}

/**
 * Evaluador de condiciones para nodos condicionales
 */
export class ConditionEvaluator {
  /**
   * Evalúa una condición en el contexto dado
   */
  evaluate(condition: string, context: ScriptExecutionContext): boolean {
    try {
      // Implementación básica - se puede extender con un evaluador más sofisticado
      const variables = Object.fromEntries(context.variables);
      const results = Object.fromEntries(context.results);

      // Crear función segura para evaluación
      const safeEval = new Function(
        'variables',
        'results',
        `return ${condition};`
      );

      return !!safeEval(variables, results);
    } catch (error) {
      console.error(`Error evaluating condition "${condition}":`, error);
      return false;
    }
  }
}
