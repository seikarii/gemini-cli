/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content } from '@google/genai';
import * as path from 'path';

/**
 * Context about the current task for tool selection
 */
export interface TaskContext {
  /** Files mentioned in recent conversation */
  referencedFiles: string[];
  /** Type of operation being requested */
  operationType:
    | 'read'
    | 'modify'
    | 'create'
    | 'search'
    | 'analyze'
    | 'unknown';
  /** Specific modification type if applicable */
  modificationType?:
    | 'function'
    | 'class'
    | 'variable'
    | 'config'
    | 'structure'
    | 'text';
  /** Whether previous tool calls failed */
  hasRecentFailures: boolean;
  /** Previous tool that failed */
  lastFailedTool?: string;
  /** Error patterns from recent failures */
  recentErrors: string[];
}

/**
 * Tool recommendation with reasoning
 */
export interface ToolRecommendation {
  /** Primary recommended tool */
  primaryTool: string;
  /** Alternative tools in order of preference */
  alternatives: string[];
  /** Reasoning for the recommendation */
  reasoning: string;
  /** Specific guidance for tool usage */
  usage: string;
  /** Warning about potential issues */
  warnings?: string[];
}

/**
 * Dynamic tool selection guidance based on conversation context
 */
export class ToolSelectionGuidance {
  /**
   * Analyzes conversation and generates contextual tool selection guidance
   */
  static generateGuidance(
    userMessage: string,
    conversationHistory: Content[],
  ): string {
    const context = this.analyzeTaskContext(userMessage, conversationHistory);
    const recommendation = this.recommendTools(context);

    return this.formatGuidance(context, recommendation);
  }

  /**
   * Analyzes the current task context from conversation
   */
  private static analyzeTaskContext(
    userMessage: string,
    conversationHistory: Content[],
  ): TaskContext {
    const referencedFiles = this.extractFileReferences(
      userMessage,
      conversationHistory,
    );
    const operationType = this.detectOperationType(userMessage);
    const modificationType = this.detectModificationType(userMessage);
    const { hasRecentFailures, lastFailedTool, recentErrors } =
      this.analyzeRecentFailures(conversationHistory);

    return {
      referencedFiles,
      operationType,
      modificationType,
      hasRecentFailures,
      lastFailedTool,
      recentErrors,
    };
  }

  /**
   * Extracts file references from conversation
   */
  private static extractFileReferences(
    userMessage: string,
    history: Content[],
  ): string[] {
    const files = new Set<string>();

    // Extract from current message
    const filePatterns = [
      /['""`]([^'""`]*\.(ts|js|py|java|cpp|c|h|jsx|tsx|json|md|yml|yaml|xml|txt))['""`]/gi,
      /\b([a-zA-Z0-9_-]+\.(ts|js|py|java|cpp|c|h|jsx|tsx|json|md|yml|yaml|xml|txt))\b/gi,
      /\/[a-zA-Z0-9_.-/]*\.(ts|js|py|java|cpp|c|h|jsx|tsx|json|md|yml|yaml|xml|txt)/gi,
    ];

    const allText =
      userMessage +
      ' ' +
      history
        .slice(-50)
        .map((c) => this.extractTextFromContent(c))
        .join(' ');

    filePatterns.forEach((pattern) => {
      const matches = allText.match(pattern);
      if (matches) {
        matches.forEach((match) => {
          const cleaned = match.replace(/['""`]/g, '');
          if (cleaned.length > 3) files.add(cleaned);
        });
      }
    });

    return Array.from(files);
  }

  /**
   * Detects the type of operation being requested
   */
  private static detectOperationType(
    userMessage: string,
  ): TaskContext['operationType'] {
    const msg = userMessage.toLowerCase();

    if (
      msg.includes('read') ||
      msg.includes('show') ||
      msg.includes('display') ||
      msg.includes('view')
    ) {
      return 'read';
    }
    if (
      msg.includes('create') ||
      msg.includes('new') ||
      msg.includes('add') ||
      msg.includes('generate')
    ) {
      return 'create';
    }
    if (
      msg.includes('update') ||
      msg.includes('modify') ||
      msg.includes('change') ||
      msg.includes('edit') ||
      msg.includes('fix') ||
      msg.includes('refactor') ||
      msg.includes('rename')
    ) {
      return 'modify';
    }
    if (
      msg.includes('search') ||
      msg.includes('find') ||
      msg.includes('look for') ||
      msg.includes('locate')
    ) {
      return 'search';
    }
    if (
      msg.includes('analyze') ||
      msg.includes('review') ||
      msg.includes('check') ||
      msg.includes('examine')
    ) {
      return 'analyze';
    }

    return 'unknown';
  }

  /**
   * Detects the type of modification being requested
   */
  private static detectModificationType(
    userMessage: string,
  ): TaskContext['modificationType'] {
    const msg = userMessage.toLowerCase();

    if (
      msg.includes('function') ||
      msg.includes('method') ||
      msg.includes('procedure')
    ) {
      return 'function';
    }
    if (
      msg.includes('class') ||
      msg.includes('interface') ||
      msg.includes('type') ||
      msg.includes('struct')
    ) {
      return 'class';
    }
    if (
      msg.includes('variable') ||
      msg.includes('constant') ||
      msg.includes('property') ||
      msg.includes('field')
    ) {
      return 'variable';
    }
    if (
      msg.includes('config') ||
      msg.includes('json') ||
      msg.includes('yaml') ||
      msg.includes('xml') ||
      msg.includes('setting') ||
      msg.includes('package.json')
    ) {
      return 'config';
    }
    if (
      msg.includes('architecture') ||
      msg.includes('structure') ||
      msg.includes('organize') ||
      msg.includes('module')
    ) {
      return 'structure';
    }

    return 'text';
  }

  /**
   * Analyzes recent tool failures from conversation
   */
  private static analyzeRecentFailures(history: Content[]): {
    hasRecentFailures: boolean;
    lastFailedTool?: string;
    recentErrors: string[];
  } {
    const recentMessages = history.slice(-50); // Last 50 messages
    const errors: string[] = [];
    let lastFailedTool: string | undefined;

    for (const content of recentMessages) {
      const text = this.extractTextFromContent(content);

      // Look for error patterns
      if (
        text.includes('Error:') ||
        text.includes('Failed:') ||
        text.includes('could not') ||
        text.includes('failed') ||
        text.includes('multiple matches') ||
        text.includes('not found')
      ) {
        errors.push(text.substring(0, 200));

        // Extract failed tool name
        const toolMatch = text.match(
          /tool[:\s]+(\w+)|(\w+)\s+failed|Error.*?(\w+_\w+)|replace.*?failed/i,
        );
        if (toolMatch) {
          lastFailedTool =
            toolMatch[1] || toolMatch[2] || toolMatch[3] || 'replace';
        }
      }

      // Look for specific tool failure patterns
      if (
        text.includes('replace') &&
        (text.includes('multiple matches') ||
          text.includes('not found') ||
          text.includes('failed'))
      ) {
        lastFailedTool = 'replace';
        errors.push('replace tool failed due to matching issues');
      }
    }

    return {
      hasRecentFailures: errors.length > 0,
      lastFailedTool,
      recentErrors: errors,
    };
  }

  /**
   * Recommends tools based on context analysis
   */
  private static recommendTools(context: TaskContext): ToolRecommendation {
    // Handle error recovery cases first
    if (context.hasRecentFailures && context.lastFailedTool === 'replace') {
      const codeFiles = context.referencedFiles.filter((f) =>
        this.isCodeFile(f),
      );
      if (
        codeFiles.length > 0 ||
        context.modificationType === 'function' ||
        context.modificationType === 'class'
      ) {
        return {
          primaryTool: 'upsert_code_block',
          alternatives: ['ast_edit', 'replace'],
          reasoning:
            'Previous replace operation failed, switching to AST-based approach for code files',
          usage:
            'Use upsert_code_block for structural changes or ast_edit for precise modifications',
          warnings: [
            'Ensure you understand the code structure before making changes',
          ],
        };
      } else {
        return {
          primaryTool: 'upsert_code_block',
          alternatives: ['ast_edit'],
          reasoning:
            'Previous replace operation failed, avoiding text-based tools',
          usage:
            'Use AST-based tools for more reliable editing after replace failure',
          warnings: [
            'Replace tool failed previously, avoid text-based matching',
          ],
        };
      }
    }

    // Determine recommendation based on operation and file types
    const codeFiles = context.referencedFiles.filter((f) => this.isCodeFile(f));
    const configFiles = context.referencedFiles.filter((f) =>
      this.isConfigFile(f),
    );

    if (context.operationType === 'modify' && codeFiles.length > 0) {
      if (
        context.modificationType === 'function' ||
        context.modificationType === 'class'
      ) {
        return {
          primaryTool: 'upsert_code_block',
          alternatives: ['ast_edit', 'replace'],
          reasoning: 'Structural code modification detected in code files',
          usage:
            'Use upsert_code_block to replace/insert complete functions or classes',
          warnings: ['Read the file first to understand the current structure'],
        };
      }

      if (context.modificationType === 'variable') {
        return {
          primaryTool: 'ast_edit',
          alternatives: ['upsert_code_block', 'replace'],
          reasoning: 'Variable/expression modification in code files',
          usage:
            'Use ast_edit with precise AST queries to modify variables or expressions',
          warnings: [
            'Ensure the AST query is specific enough to avoid unintended matches',
          ],
        };
      }
    }

    if (context.operationType === 'modify' && configFiles.length > 0) {
      return {
        primaryTool: 'replace',
        alternatives: ['write_file'],
        reasoning: 'Configuration file modification detected',
        usage: 'Use replace for precise text modifications in config files',
        warnings: [
          'Include sufficient context (5+ lines) to avoid ambiguous matches',
        ],
      };
    }

    if (context.operationType === 'create') {
      return {
        primaryTool: 'write_file',
        alternatives: ['upsert_code_block'],
        reasoning: 'New file creation requested',
        usage: 'Use write_file to create new files with complete content',
        warnings: ['Check if the file already exists before creating'],
      };
    }

    // Default fallback
    return {
      primaryTool: codeFiles.length > 0 ? 'upsert_code_block' : 'replace',
      alternatives:
        codeFiles.length > 0 ? ['ast_edit', 'replace'] : ['write_file'],
      reasoning: 'General operation, using safe defaults based on file types',
      usage: 'Choose tool based on the specific nature of your changes',
      warnings: ['Analyze the task carefully before selecting tools'],
    };
  }

  /**
   * Formats the guidance into a structured prompt section
   */
  private static formatGuidance(
    context: TaskContext,
    recommendation: ToolRecommendation,
  ): string {
    const sections = [
      '## CONTEXTUAL TOOL GUIDANCE FOR THIS INTERACTION',
      '',
      '### Current Task Analysis:',
      `- **Operation**: ${context.operationType}`,
      `- **Files involved**: ${context.referencedFiles.length > 0 ? context.referencedFiles.join(', ') : 'none specified'}`,
      `- **Modification type**: ${context.modificationType || 'not specified'}`,
      `- **Recent failures**: ${context.hasRecentFailures ? `Yes (${context.lastFailedTool})` : 'No'}`,
      '',
      '### RECOMMENDED TOOL SELECTION:',
      `- **PRIMARY**: \`${recommendation.primaryTool}\``,
      `- **ALTERNATIVES**: ${recommendation.alternatives.map((t) => `\`${t}\``).join(', ')}`,
      `- **REASONING**: ${recommendation.reasoning}`,
      '',
      '### USAGE GUIDANCE:',
      recommendation.usage,
      '',
    ];

    if (recommendation.warnings && recommendation.warnings.length > 0) {
      sections.push('### ⚠️ WARNINGS:');
      recommendation.warnings.forEach((warning) =>
        sections.push(`- ${warning}`),
      );
      sections.push('');
    }

    if (context.hasRecentFailures && context.recentErrors.length > 0) {
      sections.push('### Recent Error Context:');
      context.recentErrors.slice(-2).forEach((error) => {
        sections.push(`- ${error.substring(0, 150)}...`);
      });
      sections.push('');
    }

    sections.push('### File Type Guidelines:');
    const codeFiles = context.referencedFiles.filter((f) => this.isCodeFile(f));
    const configFiles = context.referencedFiles.filter((f) =>
      this.isConfigFile(f),
    );

    if (codeFiles.length > 0) {
      sections.push(
        `- **Code files** (${codeFiles.join(', ')}): Prefer AST tools (upsert_code_block, ast_edit)`,
      );
    }
    if (configFiles.length > 0) {
      sections.push(
        `- **Config files** (${configFiles.join(', ')}): Use replace with sufficient context`,
      );
    }

    sections.push('');
    sections.push(
      '**IMPORTANT**: This guidance is specific to the current task context. Follow it precisely.',
    );
    sections.push('---');
    sections.push('');

    return sections.join('\n');
  }

  /**
   * Checks if a file is a code file
   */
  private static isCodeFile(filename: string): boolean {
    const codeExtensions = [
      '.ts',
      '.js',
      '.tsx',
      '.jsx',
      '.py',
      '.java',
      '.cpp',
      '.c',
      '.h',
      '.hpp',
      '.cs',
      '.php',
      '.rb',
      '.go',
      '.rs',
    ];
    const ext = path.extname(filename.toLowerCase());
    return codeExtensions.includes(ext);
  }

  /**
   * Checks if a file is a configuration file
   */
  private static isConfigFile(filename: string): boolean {
    const configExtensions = [
      '.json',
      '.yml',
      '.yaml',
      '.xml',
      '.toml',
      '.ini',
      '.env',
    ];
    const configFiles = [
      'package.json',
      'tsconfig.json',
      'webpack.config.js',
      '.gitignore',
      'Dockerfile',
    ];

    const ext = path.extname(filename.toLowerCase());
    const name = path.basename(filename.toLowerCase());

    return configExtensions.includes(ext) || configFiles.includes(name);
  }

  /**
   * Extracts text from Content object
   */
  private static extractTextFromContent(content: Content): string {
    if (!content.parts) return '';

    return content.parts
      .filter((part) => part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
  }
}
