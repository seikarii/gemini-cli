/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content } from '@google/genai';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Context type for different interaction scenarios
 */
export enum ContextType {
  USER_PROMPT = 'user_prompt', // Initial user request
  TOOL_CALL = 'tool_call', // After tool usage
  PLANNING = 'planning', // When LLM needs to plan
  BLOCKED = 'blocked', // When LLM seems stuck
}

/**
 * Current turn plan maintained across tool calls
 */
export interface TurnPlan {
  /** Original user objective */
  objective: string;
  /** Current step in the plan */
  currentStep: number;
  /** All planned steps */
  steps: string[];
  /** Tools used so far */
  toolsUsed: string[];
  /** Working directory */
  workingDirectory: string;
  /** Key files being worked on */
  activeFiles: string[];
  /** Current status/progress */
  status: string;
}

/**
 * Enhanced context information for LLM calls
 */
export interface EnhancedContext {
  /** Type of context being provided */
  contextType: ContextType;
  /** Current turn plan if available */
  turnPlan?: TurnPlan;
  /** Workspace information */
  workspaceInfo: WorkspaceInfo;
  /** Tool-specific guidance */
  toolGuidance: string;
  /** Self-reflection prompts if needed */
  selfReflectionPrompts?: string;
  /** Formatted context ready for LLM */
  formattedContext: string;
}

/**
 * Workspace awareness information
 */
export interface WorkspaceInfo {
  /** Current working directory */
  currentDirectory: string;
  /** Directory listing of current location */
  directoryContents: string[];
  /** Git repository info if available */
  gitInfo?: {
    branch: string;
    status: string;
    hasChanges: boolean;
  };
  /** Recent file modifications */
  recentFiles: string[];
}

/**
 * Enhanced Context Service that provides intelligent context based on interaction type
 */
export class EnhancedContextService {
  private currentTurnPlan?: TurnPlan;
  private workspaceCache?: WorkspaceInfo;
  private lastCacheTime = 0;
  private readonly cacheTimeout = 30000; // 30 seconds

  /**
   * Determines what type of context to provide based on conversation history
   */
  determineContextType(
    userMessage: string,
    conversationHistory: Content[],
  ): ContextType {
    // Check if this needs planning first (before other checks)
    if (this.needsPlanning(userMessage, conversationHistory)) {
      return ContextType.PLANNING;
    }

    // Check if this is a fresh user prompt (no recent tool calls)
    const recentHistory = conversationHistory.slice(-20);
    const hasRecentToolCalls = this.hasRecentToolCalls(recentHistory);
    const looksBlocked = this.detectIfBlocked(recentHistory);

    if (looksBlocked) {
      return ContextType.BLOCKED;
    }

    if (!hasRecentToolCalls && this.isUserInitiatedMessage(userMessage)) {
      return ContextType.USER_PROMPT;
    }

    if (hasRecentToolCalls) {
      return ContextType.TOOL_CALL;
    }

    return ContextType.USER_PROMPT;
  }

  /**
   * Generates enhanced context based on the determined type
   */
  async generateEnhancedContext(
    userMessage: string,
    conversationHistory: Content[],
    contextType: ContextType,
  ): Promise<EnhancedContext> {
    const workspaceInfo = await this.getWorkspaceInfo();
    const toolGuidance = this.generateToolGuidance(
      conversationHistory,
      contextType,
    );

    let turnPlan = this.getCurrentTurnPlan();

    // Update or create turn plan based on context type
    if (contextType === ContextType.USER_PROMPT && !turnPlan) {
      turnPlan = this.createNewTurnPlan(userMessage, workspaceInfo);
    } else if (contextType === ContextType.TOOL_CALL && turnPlan) {
      turnPlan = this.updateTurnPlan(turnPlan, conversationHistory);
    }

    const selfReflectionPrompts =
      contextType === ContextType.BLOCKED
        ? this.generateSelfReflectionPrompts()
        : undefined;

    const formattedContext = this.formatEnhancedContext({
      contextType,
      turnPlan,
      workspaceInfo,
      toolGuidance,
      selfReflectionPrompts,
      userMessage,
      conversationHistory,
    });

    return {
      contextType,
      turnPlan,
      workspaceInfo,
      toolGuidance,
      selfReflectionPrompts,
      formattedContext,
    };
  }

  /**
   * Checks if there are recent tool calls in conversation
   */
  private hasRecentToolCalls(recentHistory: Content[]): boolean {
    return recentHistory.some((content) => {
      const text = this.extractTextFromContent(content);
      return (
        text.includes('function_call') ||
        text.includes('tool_call') ||
        text.match(/\[.*Tool.*\]/i) ||
        text.includes('```') || // Often indicates tool output
        content.role === 'function'
      );
    });
  }

  /**
   * Detects if the LLM seems blocked or confused
   */
  private detectIfBlocked(recentHistory: Content[]): boolean {
    const modelMessages = recentHistory
      .filter((c) => c.role === 'model')
      .map((c) => this.extractTextFromContent(c))
      .slice(-3);

    if (modelMessages.length < 2) return false;

    // Check for repetitive patterns
    const hasRepetition = modelMessages.some(
      (msg, i) =>
        i > 0 && this.calculateSimilarity(msg, modelMessages[i - 1]) > 0.7,
    );

    // Check for confusion indicators
    const confusionMarkers = [
      'not sure',
      'unclear',
      'confused',
      "don't understand",
      'could you clarify',
      "i'm having trouble",
      'not working',
      'error',
      'failed',
      'unable to',
    ];

    const hasConfusion = modelMessages.some((msg) =>
      confusionMarkers.some((marker) => msg.toLowerCase().includes(marker)),
    );

    return hasRepetition || hasConfusion;
  }

  /**
   * Checks if message is user-initiated (vs continuation)
   */
  private isUserInitiatedMessage(userMessage: string): boolean {
    const continuationPatterns = [
      'continue',
      'next',
      'go on',
      'proceed',
      'keep going',
      "what's next",
      'then what',
      'after that',
    ];

    return !continuationPatterns.some((pattern) =>
      userMessage.toLowerCase().includes(pattern),
    );
  }

  /**
   * Checks if LLM needs planning help
   */
  private needsPlanning(
    userMessage: string,
    _recentHistory: Content[],
  ): boolean {
    const planningKeywords = [
      'how should i',
      'what should i do',
      'plan',
      'approach',
      'strategy',
      'next steps',
      'how to proceed',
    ];

    return planningKeywords.some((keyword) =>
      userMessage.toLowerCase().includes(keyword),
    );
  }

  /**
   * Gets current workspace information with caching
   */
  private async getWorkspaceInfo(): Promise<WorkspaceInfo> {
    const now = Date.now();
    if (this.workspaceCache && now - this.lastCacheTime < this.cacheTimeout) {
      return this.workspaceCache;
    }

    const currentDirectory = process.cwd();
    let directoryContents: string[] = [];
    let gitInfo;

    try {
      directoryContents = await fs.promises.readdir(currentDirectory);
      directoryContents = directoryContents.slice(0, 20); // Limit to avoid clutter
    } catch (_error) {
      console.warn('Could not read directory contents:', _error);
      directoryContents = []; // Ensure it's always an array
    }

    // Try to get git info
    try {
      const { execSync } = await import('child_process');
      const branch = execSync('git branch --show-current', {
        cwd: currentDirectory,
        encoding: 'utf8',
      }).trim();

      const status = execSync('git status --porcelain', {
        cwd: currentDirectory,
        encoding: 'utf8',
      }).trim();

      gitInfo = {
        branch,
        status: status || 'clean',
        hasChanges: status.length > 0,
      };
    } catch (_error) {
      // Not a git repo or git not available
    }

    // Get recently modified files
    let recentFiles: string[] = [];
    try {
      const files = await fs.promises.readdir(currentDirectory);
      const fileStats = await Promise.all(
        files.slice(0, 50).map(async (file) => {
          try {
            const stat = await fs.promises.stat(
              path.join(currentDirectory, file),
            );
            return { file, mtime: stat.mtime };
          } catch {
            return null;
          }
        }),
      );

      recentFiles = fileStats
        .filter((item) => item !== null && !item.file.startsWith('.'))
        .sort((a, b) => b!.mtime.getTime() - a!.mtime.getTime())
        .slice(0, 20)
        .map((item) => item!.file);
    } catch (_error) {
      console.warn('Could not get recent files:', _error);
    }

    this.workspaceCache = {
      currentDirectory,
      directoryContents: directoryContents.filter(
        (item) => !item.startsWith('.'),
      ),
      gitInfo,
      recentFiles,
    };

    this.lastCacheTime = now;
    return this.workspaceCache;
  }

  /**
   * Generates tool-specific guidance based on recent usage
   */
  private generateToolGuidance(
    conversationHistory: Content[],
    contextType: ContextType,
  ): string {
    const recentHistory = conversationHistory.slice(-10);
    const usedTools = this.extractUsedTools(recentHistory);

    if (contextType === ContextType.USER_PROMPT) {
      return this.getGeneralToolGuidance();
    }

    if (usedTools.length === 0) {
      return this.getGeneralToolGuidance();
    }

    const guidanceParts = [
      '## TOOL USAGE GUIDANCE FOR CURRENT CONTEXT',
      '',
      `Recently used tools: ${usedTools.join(', ')}`,
      '',
    ];

    // Add specific guidance for each used tool
    usedTools.forEach((tool) => {
      const guidance = this.getSpecificToolGuidance(tool);
      if (guidance) {
        guidanceParts.push(`### ${tool}:`);
        guidanceParts.push(guidance);
        guidanceParts.push('');
      }
    });

    return guidanceParts.join('\n');
  }

  /**
   * Extracts tools that were recently used from conversation
   */
  private extractUsedTools(recentHistory: Content[]): string[] {
    const tools = new Set<string>();
    const toolPatterns = [
      /read[_-]file/gi,
      /write[_-]file/gi,
      /edit|replace/gi,
      /shell|terminal/gi,
      /grep|search/gi,
      /ls|list/gi,
      /upsert[_-]code/gi,
      /ast[_-]edit/gi,
      /sequential[_-]thinking/gi,
    ];

    recentHistory.forEach((content) => {
      const text = this.extractTextFromContent(content);
      toolPatterns.forEach((pattern) => {
        if (pattern.test(text)) {
          const match = pattern.source
            .replace(/[gi]/g, '')
            .replace(/[_-]/g, '_');
          tools.add(match);
        }
      });
    });

    return Array.from(tools);
  }

  /**
   * Gets general tool guidance for fresh interactions
   */
  private getGeneralToolGuidance(): string {
    return `
## ESSENTIAL TOOL USAGE REMINDERS

### Path Management:
- Always use ABSOLUTE paths: /full/path/to/file
- Current directory: ${process.cwd()}
- Use 'ls' tool to check directory contents first

### Common Tool Usage:
- **read_file**: Absolute path required
- **write_file**: Absolute path + full content
- **replace**: Need exact matching text with context
- **shell**: Use for file operations, npm commands
- **grep**: Search within files, use absolute paths

### Before Making Changes:
1. Use 'ls' to understand current directory
2. Use 'read_file' to understand existing code
3. Make precise, targeted changes
4. Test changes with appropriate commands
`.trim();
  }

  /**
   * Gets specific guidance for individual tools
   */
  private getSpecificToolGuidance(tool: string): string | null {
    const guides: Record<string, string> = {
      read_file:
        '- Always use absolute paths\n- Check file exists with ls first',
      write_file:
        '- Use absolute paths\n- Provide complete file content\n- Create directories if needed',
      edit: '- Include 3-5 lines of context before/after\n- Exact text matching required\n- Use absolute paths',
      replace:
        '- Same as edit - exact matching with context\n- If failing, try upsert_code_block instead',
      shell:
        '- Use absolute paths in commands\n- Check current directory with pwd\n- Use && for command chaining',
      grep: '- Use absolute paths or relative from current dir\n- Include file patterns for better results',
      upsert_code:
        '- Best for code structure changes\n- Automatically handles context\n- Use for functions/classes',
      sequential_thinking:
        '- Use when stuck or need to plan\n- Break down complex problems\n- Reflect on progress',
    };

    return guides[tool] || null;
  }

  /**
   * Creates a new turn plan for fresh user requests
   */
  private createNewTurnPlan(
    userMessage: string,
    workspaceInfo: WorkspaceInfo,
  ): TurnPlan {
    const steps = this.parseStepsFromMessage(userMessage);

    this.currentTurnPlan = {
      objective: userMessage,
      currentStep: 0,
      steps,
      toolsUsed: [],
      workingDirectory: workspaceInfo.currentDirectory,
      activeFiles: [],
      status: 'Starting',
    };

    return this.currentTurnPlan;
  }

  /**
   * Updates existing turn plan based on recent actions
   */
  private updateTurnPlan(
    turnPlan: TurnPlan,
    conversationHistory: Content[],
  ): TurnPlan {
    const recentActions = this.extractRecentActions(
      conversationHistory.slice(-5),
    );
    const newToolsUsed = this.extractUsedTools(conversationHistory.slice(-10));

    return {
      ...turnPlan,
      toolsUsed: [...new Set([...turnPlan.toolsUsed, ...newToolsUsed])],
      currentStep: Math.min(
        turnPlan.currentStep + 1,
        turnPlan.steps.length - 1,
      ),
      status:
        recentActions.length > 0
          ? `Executed: ${recentActions.join(', ')}`
          : turnPlan.status,
    };
  }

  /**
   * Generates self-reflection prompts for blocked scenarios
   */
  private generateSelfReflectionPrompts(): string {
    return `
## 🤔 SELF-REFLECTION PROMPTS - You seem stuck, consider:

### Use Sequential Thinking:
If you're uncertain about the next steps, use the \`mcp_sequentialthi_sequentialthinking\` tool to:
- Break down the problem step by step
- Analyze what you've tried so far
- Generate alternative approaches
- Plan the next sequence of actions

### Debugging Questions:
- What was I trying to accomplish?
- What tools have I used and what were the results?
- Are there any error messages I should analyze?
- Do I understand the current state of the workspace?
- Should I re-read files to refresh my understanding?

### Recovery Actions:
1. Use 'ls' to check current directory state
2. Use 'read_file' to refresh understanding of key files  
3. Use sequential thinking to plan alternative approaches
4. Break the problem into smaller, manageable steps

**Remember: You can always call sequential thinking to help organize your thoughts!**
`.trim();
  }

  /**
   * Formats the complete enhanced context
   */
  private formatEnhancedContext(params: {
    contextType: ContextType;
    turnPlan?: TurnPlan;
    workspaceInfo: WorkspaceInfo;
    toolGuidance: string;
    selfReflectionPrompts?: string;
    userMessage: string;
    conversationHistory: Content[];
  }): string {
    const sections = [
      '# 🧠 ENHANCED CONTEXT FOR THIS INTERACTION',
      '',
      `**Context Type**: ${params.contextType.toUpperCase()}`,
      `**Working Directory**: \`${params.workspaceInfo.currentDirectory}\``,
      '',
    ];

    // Add turn plan if available
    if (params.turnPlan) {
      sections.push('## 📋 CURRENT TURN PLAN');
      sections.push(`**Objective**: ${params.turnPlan.objective}`);
      sections.push(
        `**Progress**: Step ${params.turnPlan.currentStep + 1}/${params.turnPlan.steps.length}`,
      );
      sections.push(`**Status**: ${params.turnPlan.status}`);
      sections.push(
        `**Tools Used**: ${params.turnPlan.toolsUsed.join(', ') || 'None yet'}`,
      );
      sections.push('');

      sections.push('**Planned Steps**:');
      params.turnPlan.steps.forEach((step, i) => {
        const marker =
          i === params.turnPlan!.currentStep
            ? '👉'
            : i < params.turnPlan!.currentStep
              ? '✅'
              : '⏸️';
        sections.push(`${marker} ${i + 1}. ${step}`);
      });
      sections.push('');
    }

    // Add workspace info
    sections.push('## 📁 WORKSPACE CONTEXT');
    sections.push(
      `**Current Directory**: \`${params.workspaceInfo.currentDirectory}\``,
    );

    if (params.workspaceInfo.directoryContents.length > 0) {
      sections.push('**Directory Contents**:');
      sections.push(
        params.workspaceInfo.directoryContents
          .slice(0, 15)
          .map((item) => `- ${item}`)
          .join('\n'),
      );
      sections.push('');
    }

    if (params.workspaceInfo.gitInfo) {
      sections.push(
        `**Git Info**: Branch \`${params.workspaceInfo.gitInfo.branch}\`, ${params.workspaceInfo.gitInfo.hasChanges ? 'Has changes' : 'Clean'}`,
      );
      sections.push('');
    }

    // Add tool guidance
    sections.push(params.toolGuidance);
    sections.push('');

    // Add self-reflection prompts if needed
    if (params.selfReflectionPrompts) {
      sections.push(params.selfReflectionPrompts);
      sections.push('');
    }

    sections.push('---');
    sections.push(
      '**Use this context to stay focused and make informed decisions!**',
    );
    sections.push('');

    return sections.join('\n');
  }

  /**
   * Helper methods
   */
  private getCurrentTurnPlan(): TurnPlan | undefined {
    return this.currentTurnPlan;
  }

  private parseStepsFromMessage(message: string): string[] {
    // Simple step extraction - could be enhanced with NLP
    if (
      message.includes('step') ||
      message.includes('1.') ||
      message.includes('first')
    ) {
      // Try to extract numbered steps
      const stepMatches = message.match(/\d+\.\s*([^.]+)/g);
      if (stepMatches) {
        return stepMatches.map((match) =>
          match.replace(/^\d+\.\s*/, '').trim(),
        );
      }
    }

    // Default breakdown for common requests
    if (
      message.toLowerCase().includes('fix') ||
      message.toLowerCase().includes('debug')
    ) {
      return [
        'Analyze the issue and identify root cause',
        'Implement the fix',
        'Test the solution',
        'Verify everything works correctly',
      ];
    }

    if (
      message.toLowerCase().includes('create') ||
      message.toLowerCase().includes('build')
    ) {
      return [
        'Plan the structure and requirements',
        'Implement the core functionality',
        'Add tests and documentation',
        'Verify the implementation',
      ];
    }

    // Generic breakdown
    return [
      'Understand the requirement',
      'Plan the approach',
      'Implement the solution',
      'Test and verify',
    ];
  }

  private extractRecentActions(recentHistory: Content[]): string[] {
    const actions: string[] = [];

    recentHistory.forEach((content) => {
      const text = this.extractTextFromContent(content);
      if (text.includes('read_file')) actions.push('file reading');
      if (text.includes('write_file')) actions.push('file creation');
      if (text.includes('edit') || text.includes('replace'))
        actions.push('file editing');
      if (text.includes('shell') || text.includes('terminal'))
        actions.push('shell commands');
      if (text.includes('grep') || text.includes('search'))
        actions.push('searching');
    });

    return [...new Set(actions)];
  }

  private extractTextFromContent(content: Content): string {
    if (!content.parts) return '';
    return content.parts
      .filter((part) => part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
  }

  private calculateSimilarity(str1: string, str2: string): number {
    const words1 = str1.toLowerCase().split(/\s+/);
    const words2 = str2.toLowerCase().split(/\s+/);

    const intersection = words1.filter((word) => words2.includes(word));
    const union = [...new Set([...words1, ...words2])];

    return intersection.length / union.length;
  }

  /**
   * Reset turn plan (for new conversations)
   */
  resetTurnPlan(): void {
    this.currentTurnPlan = undefined;
  }

  /**
   * Clear workspace cache (force refresh)
   */
  clearWorkspaceCache(): void {
    this.workspaceCache = undefined;
    this.lastCacheTime = 0;
  }
}
