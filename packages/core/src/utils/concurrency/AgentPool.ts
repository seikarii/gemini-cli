/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Semaphore } from './Semaphore.js';
import { SubAgentScope, ContextState, PromptConfig, ModelConfig, RunConfig, ToolConfig, OutputConfig, SubagentTerminateMode } from '../../core/subagent.js';
import { Config } from '../../config/config.js';

/**
 * Configuration for spawning a parallel agent.
 */
export interface AgentConfig {
  name: string;
  promptConfig: PromptConfig;
  modelConfig: ModelConfig;
  runConfig: RunConfig;
  toolConfig?: ToolConfig;
  outputConfig?: OutputConfig;
  context: ContextState;
}

/**
 * Result from an agent execution.
 */
export interface AgentResult {
  name: string;
  success: boolean;
  output: unknown;
  error?: Error;
  executionTime: number;
}

/**
 * Configuration for running multiple agents in parallel with task decomposition.
 */
export interface ParallelAgentConfig {
  /** Main task description to be decomposed */
  taskDescription: string;
  /** Base configuration to use for all agents */
  baseConfig: {
    modelConfig: ModelConfig;
    runConfig: RunConfig;
    toolConfig?: ToolConfig;
  };
  /** Shared context for all agents */
  sharedContext: ContextState;
  /** Maximum number of agents to spawn */
  maxAgents?: number;
  /** Strategy for task decomposition */
  decompositionStrategy?: 'analysis' | 'modification' | 'testing' | 'custom';
}

/**
 * Agent pool for managing concurrent execution of multiple subagents.
 * Uses semaphore-based concurrency control to prevent resource exhaustion.
 */
export class AgentPool {
  private semaphore: Semaphore;
  private activeAgents: Map<string, SubAgentScope> = new Map();
  private results: Map<string, AgentResult> = new Map();

  constructor(
    private runtimeContext: Config,
    private maxConcurrent: number = 5
  ) {
    this.semaphore = new Semaphore(maxConcurrent);
  }

  /**
   * Spawn a single agent with the given configuration.
   */
  async spawnAgent(config: AgentConfig): Promise<SubAgentScope> {
    const agent = await SubAgentScope.create(
      config.name,
      this.runtimeContext,
      config.promptConfig,
      config.modelConfig,
      config.runConfig,
      config.toolConfig,
      config.outputConfig
    );
    
    this.activeAgents.set(config.name, agent);
    return agent;
  }

  /**
   * Run multiple agents in parallel with controlled concurrency.
   * Each agent execution is managed by the semaphore to prevent resource exhaustion.
   */
  async runParallel(configs: AgentConfig[]): Promise<AgentResult[]> {
    const promises = configs.map(config => this.executeAgent(config));
    const results = await Promise.allSettled(promises);
    
    // Process results and handle any failures
    const agentResults: AgentResult[] = [];
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const config = configs[i];
      
      if (result.status === 'fulfilled') {
        agentResults.push(result.value);
      } else {
        agentResults.push({
          name: config.name,
          success: false,
          output: null,
          error: result.reason,
          executionTime: 0
        });
      }
    }
    
    return agentResults;
  }

  /**
   * Execute a single agent with semaphore control.
   */
  private async executeAgent(config: AgentConfig): Promise<AgentResult> {
    await this.semaphore.acquire();
    const startTime = Date.now();
    
    try {
      const agent = await this.spawnAgent(config);
      await agent.runNonInteractive(config.context);
      
      const executionTime = Date.now() - startTime;
      const result: AgentResult = {
        name: config.name,
        success: agent.output.terminate_reason === SubagentTerminateMode.GOAL,
        output: agent.output,
        executionTime
      };
      
      this.results.set(config.name, result);
      return result;
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const result: AgentResult = {
        name: config.name,
        success: false,
        output: null,
        error: error instanceof Error ? error : new Error(String(error)),
        executionTime
      };
      
      this.results.set(config.name, result);
      return result;
    } finally {
      this.activeAgents.delete(config.name);
      this.semaphore.release();
    }
  }

  /**
   * Decompose a complex task into parallel agents based on strategy.
   */
  static decomposeTask(
    taskDescription: string, 
    strategy: 'analysis' | 'modification' | 'testing' | 'custom' = 'analysis'
  ): Array<{ name: string; promptTemplate: string }> {
    switch (strategy) {
      case 'analysis':
        return [
          {
            name: 'code-analyzer',
            promptTemplate: `Analyze the codebase for: ${taskDescription}. Focus on identifying strengths, weaknesses, and areas for improvement. Use grep, read, and glob tools to understand the code structure.`
          },
          {
            name: 'architecture-reviewer', 
            promptTemplate: `Review the architectural patterns for: ${taskDescription}. Identify design patterns, dependencies, and potential architectural improvements.`
          },
          {
            name: 'performance-assessor',
            promptTemplate: `Assess performance implications for: ${taskDescription}. Look for bottlenecks, optimization opportunities, and resource usage patterns.`
          }
        ];
        
      case 'modification':
        return [
          {
            name: 'implementation-agent',
            promptTemplate: `Implement the core functionality for: ${taskDescription}. Focus on robust, maintainable code with proper error handling.`
          },
          {
            name: 'test-agent',
            promptTemplate: `Create comprehensive tests for: ${taskDescription}. Include unit tests, integration tests, and edge case coverage.`
          },
          {
            name: 'documentation-agent',
            promptTemplate: `Create documentation and comments for: ${taskDescription}. Ensure code is well-documented and maintainable.`
          }
        ];
        
      case 'testing':
        return [
          {
            name: 'unit-test-agent',
            promptTemplate: `Create unit tests for: ${taskDescription}. Focus on individual component testing and edge cases.`
          },
          {
            name: 'integration-test-agent',
            promptTemplate: `Create integration tests for: ${taskDescription}. Test component interactions and system-wide behavior.`
          },
          {
            name: 'validation-agent',
            promptTemplate: `Validate the implementation of: ${taskDescription}. Run tests and ensure quality standards are met.`
          }
        ];
        
      default:
        return [
          {
            name: 'general-agent',
            promptTemplate: `Handle the task: ${taskDescription}. Use appropriate tools and strategies to complete the request effectively.`
          }
        ];
    }
  }

  /**
   * Run agents in parallel with automatic task decomposition.
   */
  async runParallelWithDecomposition(config: ParallelAgentConfig): Promise<AgentResult[]> {
    const subtasks = AgentPool.decomposeTask(
      config.taskDescription, 
      config.decompositionStrategy || 'analysis'
    );
    
    // Limit agents if maxAgents is specified
    const limitedSubtasks = config.maxAgents 
      ? subtasks.slice(0, config.maxAgents)
      : subtasks;
    
    const agentConfigs: AgentConfig[] = limitedSubtasks.map(subtask => ({
      name: subtask.name,
      promptConfig: { systemPrompt: subtask.promptTemplate },
      modelConfig: config.baseConfig.modelConfig,
      runConfig: config.baseConfig.runConfig,
      toolConfig: config.baseConfig.toolConfig,
      context: config.sharedContext
    }));
    
    return this.runParallel(agentConfigs);
  }

  /**
   * Get the current number of active agents.
   */
  getActiveCount(): number {
    return this.activeAgents.size;
  }

  /**
   * Get available semaphore permits.
   */
  getAvailableSlots(): number {
    return this.semaphore.available();
  }

  /**
   * Get results from completed agents.
   */
  getResults(): Map<string, AgentResult> {
    return new Map(this.results);
  }

  /**
   * Clear all stored results.
   */
  clearResults(): void {
    this.results.clear();
  }

  /**
   * Wait for all active agents to complete.
   */
  async waitForCompletion(): Promise<void> {
    // Wait until all agents have released their semaphore permits
    while (this.getActiveCount() > 0) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
}
