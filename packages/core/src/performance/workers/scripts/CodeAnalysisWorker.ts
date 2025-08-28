/**
 * @fileoverview Code Analysis Worker for handling code analysis tasks in parallel
 * @version 1.0.0
 * @license MIT
 */

import { isMainThread, parentPort } from 'node:worker_threads';
import * as path from 'node:path';

import {
  WorkerMessage,
  WorkerResult,
  CodeAnalysisInput,
  CodeAnalysisOutput,
  WorkerTask,
} from '../WorkerInterfaces.js';

/**
 * Code complexity metrics
 */
interface ComplexityMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  linesOfCode: number;
  functionsCount: number;
  classesCount: number;
  commentsRatio: number;
}

/**
 * Code quality issues
 */
interface QualityIssue {
  type: 'error' | 'warning' | 'info';
  severity: 'high' | 'medium' | 'low';
  message: string;
  line: number;
  column: number;
  rule: string;
  suggestion?: string;
}

/**
 * Code analysis worker class
 */
class CodeAnalysisWorker {
  private readonly supportedLanguages = new Set([
    'javascript', 'typescript', 'python', 'java', 'cpp', 'c', 'csharp', 'go', 'rust'
  ]);

  /**
   * Analyze code based on input
   */
  async analyzeCode(input: CodeAnalysisInput): Promise<CodeAnalysisOutput> {
    const startTime = Date.now();
    
    try {
      const language = input.language || this.detectLanguage(input.filePath);
      
      if (!this.supportedLanguages.has(language)) {
        throw new Error(`Unsupported language: ${language}`);
      }

      // Perform different types of analysis
      const complexity = this.analyzeComplexity(input.code, language);
      const quality = this.analyzeQuality(input.code, language);
      const structure = this.analyzeStructure(input.code, language);
      const dependencies = this.analyzeDependencies(input.code, language);
      const security = this.analyzeSecurityIssues(input.code, language);

      const processingTime = Date.now() - startTime;

      return {
        success: true,
        language,
        complexity,
        quality: {
          issues: quality,
          score: this.calculateQualityScore(quality),
        },
        structure,
        dependencies,
        security: {
          issues: security,
          riskLevel: this.calculateSecurityRisk(security),
        },
        metadata: {
          analysisType: input.analysisType || 'full',
          processingTime,
          linesAnalyzed: input.code.split('\n').length,
          language,
        },
      };
    } catch (error) {
      return {
        success: false,
        language: 'unknown',
        complexity: {
          cyclomaticComplexity: 0,
          cognitiveComplexity: 0,
          linesOfCode: 0,
          functionsCount: 0,
          classesCount: 0,
          commentsRatio: 0,
        },
        quality: {
          issues: [],
          score: 0,
        },
        structure: {},
        dependencies: [],
        security: {
          issues: [],
          riskLevel: 'unknown',
        },
        metadata: {
          analysisType: input.analysisType || 'full',
          processingTime: Date.now() - startTime,
          linesAnalyzed: 0,
          language: 'unknown',
          error: error instanceof Error ? error.message : String(error),
        },
      };
    }
  }

  /**
   * Detect programming language from file path
   */
  private detectLanguage(filePath: string): string {
    const extension = path.extname(filePath).toLowerCase();
    
    const languageMap: Record<string, string> = {
      '.js': 'javascript',
      '.mjs': 'javascript',
      '.jsx': 'javascript',
      '.ts': 'typescript',
      '.tsx': 'typescript',
      '.py': 'python',
      '.java': 'java',
      '.cpp': 'cpp',
      '.cc': 'cpp',
      '.cxx': 'cpp',
      '.c': 'c',
      '.cs': 'csharp',
      '.go': 'go',
      '.rs': 'rust',
    };
    
    return languageMap[extension] || 'unknown';
  }

  /**
   * Analyze code complexity
   */
  private analyzeComplexity(code: string, language: string): ComplexityMetrics {
    const lines = code.split('\n');
    const nonEmptyLines = lines.filter(line => line.trim().length > 0);
    const commentLines = lines.filter(line => this.isCommentLine(line, language));
    
    return {
      cyclomaticComplexity: this.calculateCyclomaticComplexity(code, language),
      cognitiveComplexity: this.calculateCognitiveComplexity(code, language),
      linesOfCode: nonEmptyLines.length,
      functionsCount: this.countFunctions(code, language),
      classesCount: this.countClasses(code, language),
      commentsRatio: commentLines.length / Math.max(lines.length, 1),
    };
  }

  /**
   * Analyze code quality issues
   */
  private analyzeQuality(code: string, language: string): QualityIssue[] {
    const issues: QualityIssue[] = [];
    const lines = code.split('\n');
    
    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      
      // Check for common quality issues
      if (line.length > 120) {
        issues.push({
          type: 'warning',
          severity: 'medium',
          message: 'Line too long (>120 characters)',
          line: lineNumber,
          column: 121,
          rule: 'max-line-length',
          suggestion: 'Break line into multiple lines',
        });
      }
      
      if (this.hasTrailingWhitespace(line)) {
        issues.push({
          type: 'warning',
          severity: 'low',
          message: 'Trailing whitespace',
          line: lineNumber,
          column: line.length,
          rule: 'no-trailing-spaces',
          suggestion: 'Remove trailing whitespace',
        });
      }
      
      if (this.hasTabs(line)) {
        issues.push({
          type: 'warning',
          severity: 'low',
          message: 'Mixed tabs and spaces',
          line: lineNumber,
          column: 1,
          rule: 'no-mixed-tabs-spaces',
          suggestion: 'Use consistent indentation',
        });
      }
      
      // Language-specific checks
      if (language === 'javascript' || language === 'typescript') {
        this.analyzeJavaScriptQuality(line, lineNumber, issues);
      } else if (language === 'python') {
        this.analyzePythonQuality(line, lineNumber, issues);
      }
    });
    
    return issues;
  }

  /**
   * Analyze code structure
   */
  private analyzeStructure(code: string, language: string): Record<string, unknown> {
    const structure: Record<string, unknown> = {};
    
    // Extract imports/includes
    structure.imports = this.extractImports(code, language);
    
    // Extract function signatures
    structure.functions = this.extractFunctions(code, language);
    
    // Extract class definitions
    structure.classes = this.extractClasses(code, language);
    
    // Extract variables/constants
    structure.variables = this.extractVariables(code, language);
    
    return structure;
  }

  /**
   * Analyze dependencies
   */
  private analyzeDependencies(code: string, language: string): Array<{
    name: string;
    type: 'internal' | 'external' | 'standard';
    usage: string[];
  }> {
    const dependencies: Array<{
      name: string;
      type: 'internal' | 'external' | 'standard';
      usage: string[];
    }> = [];
    
    const imports = this.extractImports(code, language);
    
    imports.forEach(importStatement => {
      const name = this.extractDependencyName(importStatement, language);
      if (name) {
        dependencies.push({
          name,
          type: this.classifyDependency(name, language),
          usage: [importStatement],
        });
      }
    });
    
    return dependencies;
  }

  /**
   * Analyze security issues
   */
  private analyzeSecurityIssues(code: string, language: string): Array<{
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    message: string;
    line: number;
    recommendation: string;
  }> {
    const issues: Array<{
      type: string;
      severity: 'critical' | 'high' | 'medium' | 'low';
      message: string;
      line: number;
      recommendation: string;
    }> = [];
    
    const lines = code.split('\n');
    
    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      
      // Check for common security patterns
      if (this.hasHardcodedSecrets(line)) {
        issues.push({
          type: 'hardcoded-secret',
          severity: 'critical',
          message: 'Potential hardcoded secret detected',
          line: lineNumber,
          recommendation: 'Use environment variables or secure configuration',
        });
      }
      
      if (this.hasUnsafeEval(line, language)) {
        issues.push({
          type: 'unsafe-eval',
          severity: 'high',
          message: 'Use of eval() detected',
          line: lineNumber,
          recommendation: 'Avoid eval() for security reasons',
        });
      }
      
      if (this.hasUnsafeFileOperations(line, language)) {
        issues.push({
          type: 'unsafe-file-ops',
          severity: 'medium',
          message: 'Potentially unsafe file operation',
          line: lineNumber,
          recommendation: 'Validate file paths and use safe file operations',
        });
      }
    });
    
    return issues;
  }

  /**
   * Calculate cyclomatic complexity
   */
  private calculateCyclomaticComplexity(code: string, language: string): number {
    let complexity = 1; // Base complexity
    
    // Count decision points based on language
    const decisionKeywords = this.getDecisionKeywords(language);
    
    for (const keyword of decisionKeywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g');
      const matches = code.match(regex);
      if (matches) {
        complexity += matches.length;
      }
    }
    
    return complexity;
  }

  /**
   * Calculate cognitive complexity
   */
  private calculateCognitiveComplexity(code: string, language: string): number {
    let complexity = 0;
    let nestingLevel = 0;
    const lines = code.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Track nesting level
      if (this.isOpeningBlock(trimmed, language)) {
        nestingLevel++;
      }
      
      if (this.isClosingBlock(trimmed, language)) {
        nestingLevel = Math.max(0, nestingLevel - 1);
      }
      
      // Add complexity for control structures
      if (this.isControlStructure(trimmed, language)) {
        complexity += 1 + nestingLevel;
      }
    }
    
    return complexity;
  }

  /**
   * Count functions in code
   */
  private countFunctions(code: string, language: string): number {
    const functionKeywords = this.getFunctionKeywords(language);
    let count = 0;
    
    for (const keyword of functionKeywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g');
      const matches = code.match(regex);
      if (matches) {
        count += matches.length;
      }
    }
    
    return count;
  }

  /**
   * Count classes in code
   */
  private countClasses(code: string, language: string): number {
    const classKeywords = this.getClassKeywords(language);
    let count = 0;
    
    for (const keyword of classKeywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g');
      const matches = code.match(regex);
      if (matches) {
        count += matches.length;
      }
    }
    
    return count;
  }

  /**
   * Check if line is a comment
   */
  private isCommentLine(line: string, language: string): boolean {
    const trimmed = line.trim();
    
    switch (language) {
      case 'javascript':
      case 'typescript':
      case 'java':
      case 'cpp':
      case 'c':
      case 'csharp':
      case 'go':
      case 'rust':
        return trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*');
      case 'python':
        return trimmed.startsWith('#');
      default:
        return false;
    }
  }

  /**
   * Get decision keywords for complexity calculation
   */
  private getDecisionKeywords(language: string): string[] {
    const common = ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'except'];
    
    switch (language) {
      case 'javascript':
      case 'typescript':
        return [...common, 'try', 'finally'];
      case 'python':
        return [...common, 'elif', 'try', 'finally', 'with'];
      case 'java':
      case 'csharp':
        return [...common, 'try', 'finally', 'foreach'];
      default:
        return common;
    }
  }

  /**
   * Get function keywords for counting
   */
  private getFunctionKeywords(language: string): string[] {
    switch (language) {
      case 'javascript':
      case 'typescript':
        return ['function', 'const.*=.*=>', 'let.*=.*=>', 'var.*=.*=>'];
      case 'python':
        return ['def'];
      case 'java':
      case 'csharp':
        return ['public.*[a-zA-Z]+.*\\(', 'private.*[a-zA-Z]+.*\\(', 'protected.*[a-zA-Z]+.*\\('];
      case 'cpp':
      case 'c':
        return ['[a-zA-Z_][a-zA-Z0-9_]*\\s*\\('];
      default:
        return ['function'];
    }
  }

  /**
   * Get class keywords for counting
   */
  private getClassKeywords(language: string): string[] {
    switch (language) {
      case 'javascript':
      case 'typescript':
        return ['class'];
      case 'python':
        return ['class'];
      case 'java':
      case 'csharp':
        return ['class', 'interface', 'enum'];
      case 'cpp':
        return ['class', 'struct'];
      default:
        return ['class'];
    }
  }

  // Helper methods for quality analysis
  private hasTrailingWhitespace(line: string): boolean {
    return /\s+$/.test(line);
  }

  private hasTabs(line: string): boolean {
    return line.includes('\t');
  }

  private analyzeJavaScriptQuality(line: string, lineNumber: number, issues: QualityIssue[]): void {
    if (line.includes('==') && !line.includes('===')) {
      issues.push({
        type: 'warning',
        severity: 'medium',
        message: 'Use strict equality (===) instead of loose equality (==)',
        line: lineNumber,
        column: line.indexOf('==') + 1,
        rule: 'strict-equality',
        suggestion: 'Replace == with ===',
      });
    }
    
    if (line.includes('var ')) {
      issues.push({
        type: 'warning',
        severity: 'low',
        message: 'Prefer const or let over var',
        line: lineNumber,
        column: line.indexOf('var') + 1,
        rule: 'no-var',
        suggestion: 'Use const or let instead of var',
      });
    }
  }

  private analyzePythonQuality(line: string, lineNumber: number, issues: QualityIssue[]): void {
    if (line.includes('import *')) {
      issues.push({
        type: 'warning',
        severity: 'medium',
        message: 'Avoid wildcard imports',
        line: lineNumber,
        column: line.indexOf('*') + 1,
        rule: 'no-wildcard-import',
        suggestion: 'Import specific names instead of using *',
      });
    }
  }

  // Helper methods for structure analysis
  private extractImports(code: string, language: string): string[] {
    const imports: string[] = [];
    const lines = code.split('\n');
    
    lines.forEach(line => {
      const trimmed = line.trim();
      
      switch (language) {
        case 'javascript':
        case 'typescript':
          if (trimmed.startsWith('import ') || trimmed.startsWith('require(')) {
            imports.push(trimmed);
          }
          break;
        case 'python':
          if (trimmed.startsWith('import ') || trimmed.startsWith('from ')) {
            imports.push(trimmed);
          }
          break;
        case 'java':
          if (trimmed.startsWith('import ')) {
            imports.push(trimmed);
          }
          break;
        default:
          break;
      }
    });
    
    return imports;
  }

  private extractFunctions(code: string, _language: string): string[] {
    // Simplified function extraction
    const functions: string[] = [];
    const lines = code.split('\n');
    
    lines.forEach(line => {
      if (/function\s+\w+\s*\(/.test(line) || /def\s+\w+\s*\(/.test(line)) {
        functions.push(line.trim());
      }
    });
    
    return functions;
  }

  private extractClasses(code: string, _language: string): string[] {
    const classes: string[] = [];
    const lines = code.split('\n');
    
    lines.forEach(line => {
      if (/class\s+\w+/.test(line)) {
        classes.push(line.trim());
      }
    });
    
    return classes;
  }

  private extractVariables(_code: string, _language: string): string[] {
    // Simplified variable extraction
    return [];
  }

  // Helper methods for dependency analysis
  private extractDependencyName(importStatement: string, language: string): string | null {
    switch (language) {
      case 'javascript':
      case 'typescript': {
        const jsMatch = importStatement.match(/from\s+['"]([^'"]+)['"]/);
        return jsMatch ? jsMatch[1] : null;
      }
      case 'python': {
        const pyMatch = importStatement.match(/import\s+(\w+)/);
        return pyMatch ? pyMatch[1] : null;
      }
      default:
        return null;
    }
  }

  private classifyDependency(name: string, language: string): 'internal' | 'external' | 'standard' {
    // Simple classification logic
    if (name.startsWith('.')) return 'internal';
    
    const standardLibs = this.getStandardLibraries(language);
    if (standardLibs.includes(name)) return 'standard';
    
    return 'external';
  }

  private getStandardLibraries(language: string): string[] {
    switch (language) {
      case 'javascript':
      case 'typescript':
        return ['fs', 'path', 'util', 'crypto', 'http', 'https'];
      case 'python':
        return ['os', 'sys', 'json', 'datetime', 'collections', 'itertools'];
      default:
        return [];
    }
  }

  // Helper methods for security analysis
  private hasHardcodedSecrets(line: string): boolean {
    const secretPatterns = [
      /password\s*=\s*['"][^'"]+['"]/i,
      /api[_-]?key\s*=\s*['"][^'"]+['"]/i,
      /secret\s*=\s*['"][^'"]+['"]/i,
      /token\s*=\s*['"][^'"]+['"]/i,
    ];
    
    return secretPatterns.some(pattern => pattern.test(line));
  }

  private hasUnsafeEval(line: string, language: string): boolean {
    switch (language) {
      case 'javascript':
      case 'typescript':
        return /\beval\s*\(/.test(line);
      case 'python':
        return /\beval\s*\(/.test(line) || /\bexec\s*\(/.test(line);
      default:
        return false;
    }
  }

  private hasUnsafeFileOperations(line: string, language: string): boolean {
    switch (language) {
      case 'javascript':
      case 'typescript':
        return /fs\.(readFile|writeFile|unlink)/.test(line) && !line.includes('path.join');
      case 'python':
        return /open\s*\(/.test(line) && !line.includes('os.path.join');
      default:
        return false;
    }
  }

  // Helper methods for complexity calculation
  private isOpeningBlock(line: string, _language: string): boolean {
    return line.includes('{') || line.endsWith(':');
  }

  private isClosingBlock(line: string, _language: string): boolean {
    return line.includes('}');
  }

  private isControlStructure(line: string, language: string): boolean {
    const keywords = this.getDecisionKeywords(language);
    return keywords.some(keyword => new RegExp(`\\b${keyword}\\b`).test(line));
  }

  // Helper methods for scoring
  private calculateQualityScore(issues: QualityIssue[]): number {
    if (issues.length === 0) return 100;
    
    let totalPenalty = 0;
    
    issues.forEach(issue => {
      switch (issue.severity) {
        case 'high':
          totalPenalty += 10;
          break;
        case 'medium':
          totalPenalty += 5;
          break;
        case 'low':
          totalPenalty += 2;
          break;
        default:
          totalPenalty += 1;
          break;
      }
    });
    
    return Math.max(0, 100 - totalPenalty);
  }

  private calculateSecurityRisk(issues: Array<{ severity: string }>): 'low' | 'medium' | 'high' | 'critical' {
    if (issues.some(issue => issue.severity === 'critical')) return 'critical';
    if (issues.some(issue => issue.severity === 'high')) return 'high';
    if (issues.some(issue => issue.severity === 'medium')) return 'medium';
    return 'low';
  }
}

// Worker thread execution
if (!isMainThread && parentPort) {
  const worker = new CodeAnalysisWorker();

  parentPort.on('message', async (message: WorkerMessage) => {
    try {
      if (message.type === 'task' && message.payload) {
        const task = message.payload as WorkerTask<CodeAnalysisInput>;
        const result = await worker.analyzeCode(task.input);
        
        const response: WorkerResult<CodeAnalysisOutput> = {
          taskId: task.id,
          success: true,
          result,
          executionTime: Date.now() - (task.metadata?.submittedAt ? Number(task.metadata.submittedAt) : Date.now()),
          workerId: 'code-analyzer',
        };
        
        parentPort?.postMessage({
          type: 'result',
          taskId: task.id,
          payload: response,
        });
      }
    } catch (error) {
      const errorResponse: WorkerResult<CodeAnalysisOutput> = {
        taskId: 'unknown',
        success: false,
        error: {
          message: error instanceof Error ? error.message : String(error),
          code: 'CODE_ANALYSIS_ERROR',
          stack: error instanceof Error ? error.stack : undefined,
        },
        executionTime: 0,
        workerId: 'code-analyzer',
      };
      
      parentPort?.postMessage({
        type: 'error',
        payload: errorResponse,
      });
    }
  });

  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    parentPort?.postMessage({
      type: 'error',
      error: {
        message: error.message,
        stack: error.stack,
      },
    });
    process.exit(1);
  });

  // Signal ready
  parentPort.postMessage({ type: 'ping' });
}

export { CodeAnalysisWorker };
