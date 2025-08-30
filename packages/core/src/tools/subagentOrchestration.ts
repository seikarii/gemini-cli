/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { 
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolResult
} from './tools.js';
import { SubAgentScope, ContextState } from '../core/subagent.js';
import { Config } from '../config/config.js';

/**
 * Parameters for RunParallelTool
 */
export interface RunParallelParams {
  taskDescription: string;
  strategy: 'analysis' | 'modification' | 'testing' | 'custom';
  maxConcurrent?: number;
  toolsNeeded?: string[];
}

/**
 * Parameters for DelegateSubagentTool  
 */
export interface DelegateSubagentParams {
  taskDescription: string;
  agentType: 'specialistAgent' | 'reviewAgent' | 'testAgent' | 'customAgent';
  context?: string;
  toolsNeeded?: string[];
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * Parameters for CreateAnalysisAgentTool
 */
export interface CreateAnalysisAgentParams {
  analysisType: 'codebaseAnalysis' | 'performanceAnalysis' | 'securityAnalysis' | 'architectureAnalysis' | 'customAnalysis';
  targetPath: string;
  depth?: 'shallow' | 'medium' | 'deep';
  includePatterns?: string[];
  excludePatterns?: string[];
  customInstructions?: string;
}

// Tool Invocation Classes

class RunParallelToolInvocation extends BaseToolInvocation<RunParallelParams, ToolResult> {
  constructor(
    private readonly config: Config,
    params: RunParallelParams,
  ) {
    super(params);
  }

  getDescription(): string {
    return `Running parallel subagents for: ${this.params.taskDescription}`;
  }

  async execute(_signal: AbortSignal): Promise<ToolResult> {
    try {
      console.log('🤖 Running parallel subagents with strategy:', this.params.strategy);
      
      const contextState = new ContextState();
      const maxConcurrent = this.params.maxConcurrent || 3;
      const toolsNeeded = this.params.toolsNeeded || ['read_file', 'list_directory', 'grep'];
      
      // Create multiple subagents for parallel execution
      const subagentPromises: Array<Promise<SubAgentScope>> = [];
      
      for (let i = 0; i < maxConcurrent; i++) {
        const agentName = `parallel-agent-${i + 1}`;
        const systemPrompt = `You are subagent ${i + 1} of ${maxConcurrent} working on: ${this.params.taskDescription}. Focus on strategy: ${this.params.strategy}. Coordinate with other agents by emitting your findings.`;
        
        const subagentPromise = SubAgentScope.create(
          agentName,
          this.config,
          { 
            systemPrompt,
          },
          { 
            model: 'gemini-1.5-flash-latest', 
            temp: 0.7, 
            top_p: 1 
          },
          { 
            max_time_minutes: 3, 
            max_turns: 8 
          },
          {
            tools: toolsNeeded
          },
          {
            outputs: {
              result: `Findings from ${agentName} for the ${this.params.strategy} strategy`,
              status: 'Completion status of the assigned task'
            }
          }
        );
        
        subagentPromises.push(subagentPromise);
      }
      
      // Wait for all subagents to be created
      const subagents = await Promise.all(subagentPromises);
      
      // Run all subagents in parallel
      const runPromises = subagents.map(agent => agent.runNonInteractive(contextState));
      await Promise.all(runPromises);
      
      // Collect results
      const results = subagents.map(agent => ({
        name: agent.name,
        terminationReason: agent.output.terminate_reason,
        emittedVars: agent.output.emitted_vars
      }));
      
      const summary = `✅ Parallel execution completed with ${results.length} subagents using ${this.params.strategy} strategy`;
      
      return {
        llmContent: summary + '\n\nResults:\n' + results.map((r: unknown, i: number) => `Agent ${i + 1}: ${JSON.stringify(r, null, 2)}`).join('\n'),
        returnDisplay: summary
      };
    } catch (error) {
      const errorMsg = `Failed to execute parallel subagents: ${error instanceof Error ? error.message : String(error)}`;
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg
      };
    }
  }
}

class DelegateSubagentToolInvocation extends BaseToolInvocation<DelegateSubagentParams, ToolResult> {
  constructor(
    private readonly config: Config,
    params: DelegateSubagentParams,
  ) {
    super(params);
  }

  getDescription(): string {
    return `Delegating to ${this.params.agentType}: ${this.params.taskDescription}`;
  }

  async execute(_signal: AbortSignal): Promise<ToolResult> {
    try {
      console.log('🎯 Delegating to subagent:', this.params.agentType);
      
      const contextState = new ContextState();
      const priority = this.params.priority || 'medium';
      const toolsNeeded = this.params.toolsNeeded || ['read_file', 'write_file', 'list_directory'];
      
      // Create specialized subagent based on type
      let systemPrompt = '';
      let maxTimeMinutes = 5;
      let maxTurns = 10;
      
      switch (this.params.agentType) {
        case 'specialistAgent':
          systemPrompt = `You are a specialist agent focused on: ${this.params.taskDescription}. ${this.params.context || 'Use your specialized knowledge to complete the task efficiently.'}`;
          maxTimeMinutes = 7;
          maxTurns = 15;
          break;
        case 'reviewAgent':
          systemPrompt = `You are a review agent tasked with: ${this.params.taskDescription}. ${this.params.context || 'Carefully review and provide detailed feedback and suggestions.'}`;
          maxTimeMinutes = 5;
          maxTurns = 12;
          break;
        case 'testAgent':
          systemPrompt = `You are a testing agent responsible for: ${this.params.taskDescription}. ${this.params.context || 'Focus on creating comprehensive tests and validating functionality.'}`;
          maxTimeMinutes = 6;
          maxTurns = 14;
          break;
        case 'customAgent':
        default:
          systemPrompt = `You are a custom agent working on: ${this.params.taskDescription}. ${this.params.context || 'Complete the task according to the requirements.'}`;
          break;
      }
      
      systemPrompt += `\n\nPriority Level: ${priority.toUpperCase()}. ${priority === 'critical' ? 'This task requires immediate attention and high-quality output.' : priority === 'high' ? 'This task is important and should be handled promptly.' : 'Handle this task with appropriate care.'}`;
      
      // Create and run the subagent
      const subagent = await SubAgentScope.create(
        `${this.params.agentType}-delegate`,
        this.config,
        { systemPrompt },
        { 
          model: 'gemini-1.5-flash-latest', 
          temp: 0.5, 
          top_p: 1 
        },
        { 
          max_time_minutes: maxTimeMinutes, 
          max_turns: maxTurns 
        },
        {
          tools: toolsNeeded
        },
        {
          outputs: {
            result: `Results from ${this.params.agentType} delegation`,
            status: 'Completion status and any recommendations',
            summary: 'Brief summary of work completed'
          }
        }
      );
      
      await subagent.runNonInteractive(contextState);
      
      const result = {
        agentType: this.params.agentType,
        priority,
        terminationReason: subagent.output.terminate_reason,
        emittedVars: subagent.output.emitted_vars
      };
      
      const summary = `✅ Delegation to ${this.params.agentType} completed with priority ${priority}`;
      
      return {
        llmContent: summary + '\n\nResult:\n' + JSON.stringify(result, null, 2),
        returnDisplay: summary
      };
    } catch (error) {
      const errorMsg = `Failed to delegate to subagent: ${error instanceof Error ? error.message : String(error)}`;
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg
      };
    }
  }
}

class CreateAnalysisAgentToolInvocation extends BaseToolInvocation<CreateAnalysisAgentParams, ToolResult> {
  constructor(
    private readonly config: Config,
    params: CreateAnalysisAgentParams,
  ) {
    super(params);
  }

  getDescription(): string {
    return `Creating ${this.params.analysisType} agent for: ${this.params.targetPath}`;
  }

  async execute(_signal: AbortSignal): Promise<ToolResult> {
    try {
      console.log('🔍 Creating analysis agent:', this.params.analysisType);
      
      const contextState = new ContextState();
      const depth = this.params.depth || 'medium';
      const includePatterns = this.params.includePatterns || ['**/*.ts', '**/*.js', '**/*.json'];
      const excludePatterns = this.params.excludePatterns || ['**/node_modules/**', '**/dist/**'];
      
      // Create specialized analysis prompt based on type
      let systemPrompt = '';
      let toolsNeeded: string[] = ['read_file', 'list_directory', 'grep'];
      let maxTimeMinutes = 8;
      let maxTurns = 20;
      
      switch (this.params.analysisType) {
        case 'codebaseAnalysis':
          systemPrompt = `You are a codebase analysis agent. Analyze the codebase at ${this.params.targetPath} with ${depth} depth. Focus on code structure, patterns, dependencies, and overall architecture.`;
          toolsNeeded = ['read_file', 'list_directory', 'grep', 'ast_find', 'ast_read'];
          maxTurns = 25;
          break;
        case 'performanceAnalysis':
          systemPrompt = `You are a performance analysis agent. Analyze the code at ${this.params.targetPath} for performance bottlenecks, optimization opportunities, and efficiency concerns with ${depth} depth.`;
          toolsNeeded = ['read_file', 'list_directory', 'grep', 'ast_find'];
          maxTimeMinutes = 10;
          break;
        case 'securityAnalysis':
          systemPrompt = `You are a security analysis agent. Examine the code at ${this.params.targetPath} for security vulnerabilities, potential risks, and security best practices with ${depth} depth.`;
          toolsNeeded = ['read_file', 'list_directory', 'grep', 'ast_find'];
          maxTimeMinutes = 12;
          maxTurns = 30;
          break;
        case 'architectureAnalysis':
          systemPrompt = `You are an architecture analysis agent. Evaluate the software architecture at ${this.params.targetPath} including design patterns, modularity, and structural quality with ${depth} depth.`;
          toolsNeeded = ['read_file', 'list_directory', 'grep', 'ast_find', 'ast_read'];
          maxTimeMinutes = 15;
          maxTurns = 35;
          break;
        case 'customAnalysis':
        default:
          systemPrompt = `You are a custom analysis agent. Perform analysis on ${this.params.targetPath} with ${depth} depth.`;
          if (this.params.customInstructions) {
            systemPrompt += `\n\nCustom Instructions: ${this.params.customInstructions}`;
          }
          break;
      }
      
      systemPrompt += `\n\nTarget Path: ${this.params.targetPath}`;
      systemPrompt += `\nInclude Patterns: ${includePatterns.join(', ')}`;
      systemPrompt += `\nExclude Patterns: ${excludePatterns.join(', ')}`;
      systemPrompt += `\nAnalysis Depth: ${depth}`;
      
      // Create and run the analysis agent
      const analysisAgent = await SubAgentScope.create(
        `${this.params.analysisType}-analyzer`,
        this.config,
        { systemPrompt },
        { 
          model: 'gemini-1.5-flash-latest', 
          temp: 0.3, // Lower temperature for more focused analysis
          top_p: 1 
        },
        { 
          max_time_minutes: maxTimeMinutes, 
          max_turns: maxTurns 
        },
        {
          tools: toolsNeeded
        },
        {
          outputs: {
            analysisResult: `Detailed ${this.params.analysisType} findings and recommendations`,
            summary: 'Executive summary of key findings',
            recommendations: 'Specific actionable recommendations',
            metrics: 'Relevant metrics and measurements'
          }
        }
      );
      
      await analysisAgent.runNonInteractive(contextState);
      
      const result = {
        analysisType: this.params.analysisType,
        targetPath: this.params.targetPath,
        depth,
        terminationReason: analysisAgent.output.terminate_reason,
        emittedVars: analysisAgent.output.emitted_vars
      };
      
      const summary = `✅ Analysis agent created for ${this.params.analysisType} with ${depth} depth analysis`;
      
      return {
        llmContent: summary + '\n\nResult:\n' + JSON.stringify(result, null, 2),
        returnDisplay: summary
      };
    } catch (error) {
      const errorMsg = `Failed to create analysis agent: ${error instanceof Error ? error.message : String(error)}`;
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg
      };
    }
  }
}

// Tool Classes

export class RunParallelTool extends BaseDeclarativeTool<RunParallelParams, ToolResult> {
  static readonly Name = 'run_parallel_subagents';

  constructor(private config: Config) {
    super(
      RunParallelTool.Name,
      'SubagentOrchestration',
      'Run multiple specialized subagents in parallel for complex analysis tasks. Use this for directory analysis, code review, or any task that benefits from parallel processing.',
      Kind.Search,
      {
        properties: {
          taskDescription: {
            description: 'Description of the overall task to be decomposed and executed in parallel',
            type: 'string',
          },
          strategy: {
            description: 'Strategy for agent coordination',
            type: 'string',
            enum: ['analysis', 'modification', 'testing', 'custom'],
          },
          maxConcurrent: {
            description: 'Maximum number of concurrent agents (default: 3)',
            type: 'number',
          },
          toolsNeeded: {
            description: 'List of tools that subagents should have access to',
            type: 'array',
            items: { type: 'string' },
          }
        },
        required: ['taskDescription', 'strategy'],
        type: 'object',
      },
    );
  }

  protected createInvocation(
    params: RunParallelParams,
  ): ToolInvocation<RunParallelParams, ToolResult> {
    return new RunParallelToolInvocation(this.config, params);
  }
}

export class DelegateSubagentTool extends BaseDeclarativeTool<DelegateSubagentParams, ToolResult> {
  static readonly Name = 'delegate_to_subagent';

  constructor(private config: Config) {
    super(
      DelegateSubagentTool.Name,
      'SubagentOrchestration',
      'Delegate a specific task to a specialized subagent. Use this when you need focused expertise for file editing, code review, testing, or other specialized tasks.',
      Kind.Execute,
      {
        properties: {
          taskDescription: {
            description: 'Description of the task to delegate to the subagent',
            type: 'string',
          },
          agentType: {
            description: 'Type of specialized agent to delegate to',
            type: 'string',
            enum: ['specialistAgent', 'reviewAgent', 'testAgent', 'customAgent'],
          },
          context: {
            description: 'Additional context or constraints for the subagent',
            type: 'string',
          },
          toolsNeeded: {
            description: 'List of tools the subagent should have access to',
            type: 'array',
            items: { type: 'string' },
          },
          priority: {
            description: 'Priority level for task execution',
            type: 'string',
            enum: ['low', 'medium', 'high', 'critical'],
          }
        },
        required: ['taskDescription', 'agentType'],
        type: 'object',
      },
    );
  }

  protected createInvocation(
    params: DelegateSubagentParams,
  ): ToolInvocation<DelegateSubagentParams, ToolResult> {
    return new DelegateSubagentToolInvocation(this.config, params);
  }
}

export class CreateAnalysisAgentTool extends BaseDeclarativeTool<CreateAnalysisAgentParams, ToolResult> {
  static readonly Name = 'create_analysis_agent';

  constructor(private config: Config) {
    super(
      CreateAnalysisAgentTool.Name,
      'SubagentOrchestration',
      'Create a specialized analysis agent for codebase, performance, security, or architecture analysis. Use this for comprehensive code analysis tasks.',
      Kind.Search,
      {
        properties: {
          analysisType: {
            description: 'Type of analysis to perform',
            type: 'string',
            enum: ['codebaseAnalysis', 'performanceAnalysis', 'securityAnalysis', 'architectureAnalysis', 'customAnalysis'],
          },
          targetPath: {
            description: 'Absolute path to the target directory or file to analyze',
            type: 'string',
          },
          depth: {
            description: 'Depth of analysis to perform',
            type: 'string',
            enum: ['shallow', 'medium', 'deep'],
          },
          includePatterns: {
            description: 'Glob patterns for files to include in analysis',
            type: 'array',
            items: { type: 'string' },
          },
          excludePatterns: {
            description: 'Glob patterns for files to exclude from analysis',
            type: 'array',
            items: { type: 'string' },
          },
          customInstructions: {
            description: 'Custom instructions for the analysis agent',
            type: 'string',
          }
        },
        required: ['analysisType', 'targetPath'],
        type: 'object',
      },
    );
  }

  protected validateToolParamValues(params: CreateAnalysisAgentParams): string | null {
    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(params.targetPath)) {
      const directories = workspaceContext.getDirectories();
      return `Target path must be within one of the workspace directories: ${directories.join(', ')}`;
    }
    return null;
  }

  protected createInvocation(
    params: CreateAnalysisAgentParams,
  ): ToolInvocation<CreateAnalysisAgentParams, ToolResult> {
    return new CreateAnalysisAgentToolInvocation(this.config, params);
  }
}
