/**
 * @fileoverview File Processing Worker for handling file operations in parallel
 * @version 1.0.0
 * @license MIT
 */

import { isMainThread, parentPort } from 'node:worker_threads';
import { promises as fs } from 'node:fs';
import { createHash } from 'node:crypto';
import * as path from 'node:path';

import {
  WorkerMessage,
  WorkerResult,
  FileProcessingInput,
  FileProcessingOutput,
  WorkerTask,
} from '../WorkerInterfaces.js';

/**
 * File processing worker class
 */
class FileProcessingWorker {
  private readonly maxFileSize = 100 * 1024 * 1024; // 100MB default

  /**
   * Process file operation based on input
   */
  async processFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    try {
      switch (input.operation) {
        case 'read':
          return await this.readFile(input);
        case 'write':
          return await this.writeFile(input);
        case 'analyze':
          return await this.analyzeFile(input);
        case 'transform':
          return await this.transformFile(input);
        case 'validate':
          return await this.validateFile(input);
        default:
          throw new Error(`Unsupported operation: ${input.operation}`);
      }
    } catch (_error) {
      return {
        success: false,
        metadata: {
          size: 0,
          encoding: 'none',
        },
      };
    }
  }

  /**
   * Read file content
   */
  private async readFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    const stats = await fs.stat(input.filePath);
    const maxSize = input.options?.maxSize || this.maxFileSize;

    if (stats.size > maxSize) {
      return {
        success: false,
        metadata: {
          size: stats.size,
          encoding: 'none',
        },
      };
    }

    const encoding = input.options?.encoding || 'utf8';
    const parseAs = input.options?.parseAs || 'text';

    let content: string;

    if (parseAs === 'binary') {
      const buffer = await fs.readFile(input.filePath);
      content = buffer.toString('base64');
    } else {
      content = await fs.readFile(input.filePath, encoding);
      
      // Parse content based on parseAs option
      if (parseAs === 'json') {
        try {
          const parsed = JSON.parse(content);
          content = JSON.stringify(parsed);
        } catch (_error) {
          return {
            success: false,
            metadata: {
              size: stats.size,
              encoding,
            },
          };
        }
      }
    }

    const checksum = createHash('sha256').update(content).digest('hex');

    return {
      success: true,
      content,
      metadata: {
        size: stats.size,
        encoding,
        mimeType: this.getMimeType(input.filePath),
        checksum,
        lastModified: stats.mtime,
      },
    };
  }

  /**
   * Write file content
   */
  private async writeFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    if (!input.content) {
      return {
        success: false,
        metadata: {
          size: 0,
          encoding: 'none',
        },
      };
    }

    const encoding = input.options?.encoding || 'utf8';
    let contentToWrite = input.content;

    // Handle different content types
    if (input.options?.parseAs === 'json') {
      try {
        const parsed = JSON.parse(input.content);
        contentToWrite = JSON.stringify(parsed, null, 2);
      } catch (_error) {
        return {
          success: false,
          metadata: {
            size: 0,
            encoding,
          },
        };
      }
    }

    // Ensure directory exists
    const dir = path.dirname(input.filePath);
    await fs.mkdir(dir, { recursive: true });

    await fs.writeFile(input.filePath, contentToWrite, encoding);
    
    const stats = await fs.stat(input.filePath);
    const checksum = createHash('sha256').update(contentToWrite).digest('hex');

    return {
      success: true,
      content: contentToWrite,
      metadata: {
        size: stats.size,
        encoding,
        mimeType: this.getMimeType(input.filePath),
        checksum,
        lastModified: stats.mtime,
      },
    };
  }

  /**
   * Analyze file structure and content
   */
  private async analyzeFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    const stats = await fs.stat(input.filePath);
    const extension = path.extname(input.filePath);
    
    const analysis = {
      path: input.filePath,
      name: path.basename(input.filePath),
      extension,
      isDirectory: stats.isDirectory(),
      isFile: stats.isFile(),
      permissions: stats.mode.toString(8),
      language: this.detectLanguage(extension),
      mimeType: this.getMimeType(input.filePath),
    };

    // Content analysis for text files
    let content = '';
    let contentAnalysis = {};

    if (this.isTextFile(extension) && stats.size < this.maxFileSize) {
      try {
        content = await fs.readFile(input.filePath, 'utf-8');
        contentAnalysis = {
          lineCount: content.split('\n').length,
          wordCount: content.split(/\s+/).filter(word => word.length > 0).length,
          charCount: content.length,
          hasContent: content.trim().length > 0,
        };
      } catch (error) {
        contentAnalysis = {
          error: `Content analysis failed: ${error instanceof Error ? error.message : String(error)}`,
        };
      }
    }

    const checksum = content ? createHash('sha256').update(content).digest('hex') : undefined;

    return {
      success: true,
      content: JSON.stringify({ analysis, contentAnalysis }),
      metadata: {
        size: stats.size,
        encoding: 'utf-8',
        mimeType: this.getMimeType(input.filePath),
        checksum,
        lastModified: stats.mtime,
      },
    };
  }

  /**
   * Transform file content
   */
  private async transformFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    const transformations = input.options?.transformations || [];
    
    if (transformations.length === 0) {
      return {
        success: false,
        metadata: {
          size: 0,
          encoding: 'none',
        },
      };
    }

    // Read current content
    const readResult = await this.readFile(input);
    if (!readResult.success || !readResult.content) {
      return readResult;
    }

    let content = readResult.content;

    // Apply transformations
    for (const transformation of transformations) {
      content = this.applyTransformation(content, transformation);
    }

    // Write transformed content back
    const writeInput: FileProcessingInput = {
      ...input,
      operation: 'write',
      content,
    };

    return await this.writeFile(writeInput);
  }

  /**
   * Validate file content
   */
  private async validateFile(input: FileProcessingInput): Promise<FileProcessingOutput> {
    const rules = input.options?.validationRules || [];
    
    if (rules.length === 0) {
      return {
        success: false,
        metadata: {
          size: 0,
          encoding: 'none',
        },
      };
    }

    // Read file content
    const readResult = await this.readFile(input);
    if (!readResult.success || !readResult.content) {
      return readResult;
    }

    const validationResults = [];
    const content = readResult.content;

    // Apply validation rules
    for (const rule of rules) {
      const result = this.applyValidationRule(content, rule, input.filePath);
      validationResults.push(result);
    }

    const allPassed = validationResults.every(result => result.passed);

    return {
      success: allPassed,
      content: JSON.stringify(validationResults),
      metadata: {
        size: readResult.metadata?.size || 0,
        encoding: readResult.metadata?.encoding || 'utf-8',
        mimeType: readResult.metadata?.mimeType,
        checksum: readResult.metadata?.checksum,
        lastModified: readResult.metadata?.lastModified,
      },
    };
  }

  /**
   * Apply content transformation
   */
  private applyTransformation(content: string, transformation: string): string {
    switch (transformation.toLowerCase()) {
      case 'uppercase':
        return content.toUpperCase();
      case 'lowercase':
        return content.toLowerCase();
      case 'trim':
        return content.trim();
      case 'normalize-whitespace':
        return content.replace(/\s+/g, ' ').trim();
      case 'remove-empty-lines':
        return content.split('\n').filter(line => line.trim().length > 0).join('\n');
      case 'sort-lines':
        return content.split('\n').sort().join('\n');
      case 'reverse-lines':
        return content.split('\n').reverse().join('\n');
      default:
        return content;
    }
  }

  /**
   * Apply validation rule
   */
  private applyValidationRule(content: string, rule: string, filePath: string): {
    rule: string;
    passed: boolean;
    message?: string;
  } {
    switch (rule.toLowerCase()) {
      case 'not-empty':
        return {
          rule,
          passed: content.trim().length > 0,
          message: content.trim().length === 0 ? 'File is empty' : undefined,
        };
      
      case 'valid-json':
        try {
          JSON.parse(content);
          return { rule, passed: true };
        } catch (error) {
          return {
            rule,
            passed: false,
            message: `Invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
          };
        }
      
      case 'no-tabs': {
        const hasTabs = content.includes('\t');
        return {
          rule,
          passed: !hasTabs,
          message: hasTabs ? 'File contains tab characters' : undefined,
        };
      }
      
      case 'unix-line-endings': {
        const hasWindowsLineEndings = content.includes('\r\n');
        return {
          rule,
          passed: !hasWindowsLineEndings,
          message: hasWindowsLineEndings ? 'File contains Windows line endings' : undefined,
        };
      }
      
      case 'valid-extension': {
        const extension = path.extname(filePath);
        const hasValidExtension = extension.length > 0;
        return {
          rule,
          passed: hasValidExtension,
          message: hasValidExtension ? undefined : 'File has no extension',
        };
      }
      
      default:
        return {
          rule,
          passed: false,
          message: `Unknown validation rule: ${rule}`,
        };
    }
  }

  /**
   * Get MIME type from file extension
   */
  private getMimeType(filePath: string): string {
    const extension = path.extname(filePath).toLowerCase();
    
    const mimeTypes: Record<string, string> = {
      '.txt': 'text/plain',
      '.md': 'text/markdown',
      '.json': 'application/json',
      '.js': 'application/javascript',
      '.ts': 'application/typescript',
      '.html': 'text/html',
      '.css': 'text/css',
      '.xml': 'application/xml',
      '.yaml': 'application/yaml',
      '.yml': 'application/yaml',
      '.csv': 'text/csv',
      '.log': 'text/plain',
      '.sql': 'application/sql',
      '.py': 'text/x-python',
      '.java': 'text/x-java-source',
      '.cpp': 'text/x-c++src',
      '.c': 'text/x-csrc',
      '.h': 'text/x-chdr',
      '.sh': 'application/x-sh',
      '.bat': 'application/x-msdos-program',
    };
    
    return mimeTypes[extension] || 'application/octet-stream';
  }

  /**
   * Check if file is text-based
   */
  private isTextFile(extension: string): boolean {
    const textExtensions = [
      '.txt', '.md', '.json', '.js', '.ts', '.html', '.css', '.xml', '.yaml', '.yml',
      '.csv', '.log', '.sql', '.py', '.java', '.cpp', '.c', '.h', '.sh', '.bat',
      '.ini', '.conf', '.config', '.properties', '.env',
    ];
    
    return textExtensions.includes(extension.toLowerCase());
  }

  /**
   * Detect programming language from extension
   */
  private detectLanguage(extension: string): string {
    const languageMap: Record<string, string> = {
      '.js': 'javascript',
      '.ts': 'typescript',
      '.py': 'python',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.h': 'c',
      '.cs': 'csharp',
      '.php': 'php',
      '.rb': 'ruby',
      '.go': 'go',
      '.rs': 'rust',
      '.swift': 'swift',
      '.kt': 'kotlin',
      '.scala': 'scala',
      '.sh': 'bash',
      '.bat': 'batch',
      '.ps1': 'powershell',
      '.html': 'html',
      '.css': 'css',
      '.json': 'json',
      '.xml': 'xml',
      '.yaml': 'yaml',
      '.yml': 'yaml',
      '.sql': 'sql',
      '.md': 'markdown',
    };
    
    return languageMap[extension.toLowerCase()] || 'unknown';
  }
}

// Worker thread execution
if (!isMainThread && parentPort) {
  const worker = new FileProcessingWorker();

  parentPort.on('message', async (message: WorkerMessage) => {
    try {
      if (message.type === 'task' && message.payload) {
        const task = message.payload as WorkerTask<FileProcessingInput>;
        const result = await worker.processFile(task.input);
        
        const response: WorkerResult<FileProcessingOutput> = {
          taskId: task.id,
          success: true,
          result,
          executionTime: Date.now() - (task.metadata?.submittedAt ? Number(task.metadata.submittedAt) : Date.now()),
          workerId: 'file-processor',
        };
        
        parentPort?.postMessage({
          type: 'result',
          taskId: task.id,
          payload: response,
        });
      }
    } catch (error) {
      const errorResponse: WorkerResult<FileProcessingOutput> = {
        taskId: 'unknown',
        success: false,
        error: {
          message: error instanceof Error ? error.message : String(error),
          code: 'FILE_PROCESSING_ERROR',
          stack: error instanceof Error ? error.stack : undefined,
        },
        executionTime: 0,
        workerId: 'file-processor',
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

export { FileProcessingWorker };
