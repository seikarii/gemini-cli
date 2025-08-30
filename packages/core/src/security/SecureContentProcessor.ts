/**
 * @fileoverview Secure Content Processor for file operations
 * Provides input validation, content sanitization, and rate limiting for file content processing
 */

import { RateLimiter, RateLimitAlgorithm, IRateLimitConfig, IRateLimitStats } from './RateLimiter.js';
import { InputValidator, ValidationRules, IValidationRule } from './InputValidator.js';
import { AuditLogger, AuditEventType, AuditSeverity } from './AuditLogger.js';

/**
 * Configuration for secure content processing
 */
export interface ISecureContentConfig {
  /** Enable rate limiting for file operations */
  enableRateLimit: boolean;
  /** Enable input validation */
  enableInputValidation: boolean;
  /** Enable content sanitization */
  enableContentSanitization: boolean;
  /** Enable audit logging */
  enableAuditLogging: boolean;
  /** Maximum file size in bytes */
  maxFileSize: number;
  /** Maximum content length for API calls */
  maxContentLength: number;
  /** Rate limit configuration */
  rateLimitConfig: IRateLimitConfig;
}

/**
 * Result of content processing
 */
export interface IContentProcessingResult {
  /** Whether processing was successful */
  success: boolean;
  /** Processed/sanitized content */
  content?: string;
  /** Error message if processing failed */
  error?: string;
  /** Warning messages */
  warnings: string[];
  /** Processing metadata */
  metadata: {
    originalLength: number;
    processedLength: number;
    sanitized: boolean;
    rateLimited: boolean;
    validationApplied: boolean;
  };
}

/**
 * Default configuration for secure content processing
 */
const DEFAULT_CONFIG: ISecureContentConfig = {
  enableRateLimit: true,
  enableInputValidation: true,
  enableContentSanitization: true,
  enableAuditLogging: false,
  maxFileSize: 10 * 1024 * 1024, // 10MB
  maxContentLength: 100000, // 100KB for API calls
  rateLimitConfig: {
    algorithm: RateLimitAlgorithm.TOKEN_BUCKET,
    maxRequests: 30, // 30 requests per minute
    windowMs: 60000, // 1 minute
    enableAdaptive: true,
    minDelay: 100,
    maxDelay: 5000,
    skipSuccessfulRequests: false,
    skipFailedRequests: false,
    burstSize: 10,
    refillRate: 0.5, // 0.5 tokens per second
  },
};

/**
 * Secure content processor for file operations
 */
export class SecureContentProcessor {
  private rateLimiter?: RateLimiter;
  private inputValidator?: InputValidator;
  private auditLogger?: AuditLogger;
  private config: ISecureContentConfig;

  constructor(config: Partial<ISecureContentConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initialize();
  }

  /**
   * Initialize security components
   */
  private initialize(): void {
    // Initialize rate limiter
    if (this.config.enableRateLimit) {
      this.rateLimiter = new RateLimiter(this.config.rateLimitConfig);
    }

    // Initialize input validator
    if (this.config.enableInputValidation) {
      this.inputValidator = new InputValidator();
      this.setupValidationSchemas();
    }

    // Initialize audit logger
    if (this.config.enableAuditLogging) {
      this.auditLogger = new AuditLogger({
        enabled: true,
        logFilePath: '.gemini/logs/security-audit.log',
        maxFileSize: 10 * 1024 * 1024, // 10MB
        maxFiles: 5,
        rotationInterval: 24 * 60 * 60 * 1000, // 24 hours
        minSeverity: AuditSeverity.LOW,
        enableEncryption: false,
        enableIntegrityCheck: true,
        enableAlerting: false,
      });
    }
  }

  /**
   * Setup validation schemas for different content types
   */
  private setupValidationSchemas(): void {
    if (!this.inputValidator) return;

    // File content validation schema
    this.inputValidator.registerSchema({
      name: 'fileContent',
      fields: {
        content: [
          ValidationRules.required(),
          ValidationRules.stringLength(0, this.config.maxContentLength),
        ] as Array<IValidationRule<unknown>>,
        filePath: [
          ValidationRules.required(),
          ValidationRules.filePath(),
        ] as Array<IValidationRule<unknown>>,
        operation: [
          ValidationRules.required(),
          ValidationRules.regex(/^(read|write|edit)$/, 'Invalid operation type'),
        ] as Array<IValidationRule<unknown>>,
      },
    });
  }

  /**
   * Validate content for safety and compatibility
   */
  private validateContentSafety(content: string): { isValid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check for null bytes
    if (content.includes('\0')) {
      errors.push('Content contains null bytes which may cause API errors');
    }

    // Check for excessive escape sequences
    const escapeSequences = content.match(/\\[ntrfvb\\"']/g) || [];
    if (escapeSequences.length > content.length * 0.1) {
      warnings.push('Content has many escape sequences, may need sanitization');
    }

    // Check for control characters - Create regex pattern without literal control chars
    const controlChars: string[] = [];
    for (let i = 0; i < content.length; i++) {
      const charCode = content.charCodeAt(i);
      if ((charCode >= 0 && charCode <= 31) || (charCode >= 127 && charCode <= 159)) {
        controlChars.push(content[i]);
      }
    }
    if (controlChars.length > 0) {
      warnings.push('Content contains control characters that may cause issues');
    }

    // Check for very long lines that might cause API issues
    const lines = content.split('\n');
    const longLines = lines.filter(line => line.length > 10000);
    if (longLines.length > 0) {
      warnings.push('Content contains very long lines that may cause API issues');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  /**
   * Sanitize content to prevent API errors
   */
  private sanitizeContent(content: string): string {
    let sanitized = content;

    // Remove null bytes
    sanitized = sanitized.replace(/\0/g, '');

    // Normalize line endings
    sanitized = sanitized.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // Fix common escape sequence issues
    sanitized = sanitized
      .replace(/\\"/g, '"')  // Fix escaped quotes
      .replace(/\\'/g, "'")  // Fix escaped single quotes
      .replace(/\\n/g, '\n') // Fix literal \n sequences
      .replace(/\\t/g, '\t') // Fix literal \t sequences
      .replace(/\\r/g, '\r') // Fix literal \r sequences
      .replace(/\\\\/g, '\\'); // Fix double backslashes

    // Remove or replace control characters (except common ones like tab, newline)
    // Using character code filtering instead of regex with control characters
    let cleanedContent = '';
    for (let i = 0; i < sanitized.length; i++) {
      const charCode = sanitized.charCodeAt(i);
      const char = sanitized[i];
      // Keep printable chars, newline (10), tab (9), carriage return (13)
      if (charCode >= 32 || charCode === 9 || charCode === 10 || charCode === 13) {
        cleanedContent += char;
      }
    }
    sanitized = cleanedContent;

    // Truncate extremely long lines
    const lines = sanitized.split('\n');
    const processedLines = lines.map(line => {
      if (line.length > 10000) {
        return line.substring(0, 10000) + '... [line truncated for API compatibility]';
      }
      return line;
    });
    sanitized = processedLines.join('\n');

    return sanitized;
  }

  /**
   * Process file content with security measures
   */
  async processContent(
    content: string,
    filePath: string,
    operation: 'read' | 'write' | 'edit'
  ): Promise<IContentProcessingResult> {
    const originalLength = content.length;
    const warnings: string[] = [];
    let processedContent = content;
    let rateLimited = false;
    let validationApplied = false;
    let sanitized = false;

    try {
      // Rate limiting check
      if (this.rateLimiter) {
        const rateLimitKey = `file_${operation}:${filePath}`;
        const rateLimitResult = await this.rateLimiter.checkLimit(rateLimitKey);
        
        if (!rateLimitResult.allowed) {
          return {
            success: false,
            error: `Rate limit exceeded for ${operation} operation. Retry after ${rateLimitResult.retryDelay || 'some time'} ms`,
            warnings,
            metadata: {
              originalLength,
              processedLength: 0,
              sanitized: false,
              rateLimited: true,
              validationApplied: false,
            },
          };
        }
        rateLimited = false;
      }

      // File size check
      if (originalLength > this.config.maxFileSize) {
        return {
          success: false,
          error: `File content too large: ${originalLength} bytes (max: ${this.config.maxFileSize} bytes)`,
          warnings,
          metadata: {
            originalLength,
            processedLength: 0,
            sanitized: false,
            rateLimited,
            validationApplied: false,
          },
        };
      }

      // Input validation
      if (this.inputValidator) {
        const validationResult = await this.inputValidator.validateObject(
          {
            content: processedContent,
            filePath,
            operation,
          },
          'fileContent'
        );

        validationApplied = true;

        if (!validationResult.isValid) {
          return {
            success: false,
            error: `Content validation failed: ${validationResult.errors.join(', ')}`,
            warnings: validationResult.warnings,
            metadata: {
              originalLength,
              processedLength: 0,
              sanitized: false,
              rateLimited,
              validationApplied,
            },
          };
        }

        warnings.push(...validationResult.warnings);
      }

      // Content sanitization
      if (this.config.enableContentSanitization) {
        const sanitizedContent = this.sanitizeContent(processedContent);
        if (sanitizedContent !== processedContent) {
          processedContent = sanitizedContent;
          sanitized = true;
          warnings.push('Content was sanitized to prevent API compatibility issues');
        }
      }

      // Content length check for API calls
      if (processedContent.length > this.config.maxContentLength) {
        processedContent = processedContent.substring(0, this.config.maxContentLength) + 
          '\n\n[Content truncated for API compatibility]';
        warnings.push(`Content truncated to ${this.config.maxContentLength} characters for API compatibility`);
      }

      // Audit logging
      if (this.auditLogger) {
        await this.auditLogger.log({
          type: AuditEventType.DATA_ACCESS,
          severity: AuditSeverity.LOW,
          actor: {
            id: 'system',
            type: 'system',
          },
          target: {
            id: filePath,
            type: 'file',
            path: filePath,
          },
          action: `content_${operation}`,
          description: `Content processed for ${operation} operation`,
          metadata: {
            originalLength,
            processedLength: processedContent.length,
            sanitized,
            warnings: warnings.length,
          },
        });
      }

      return {
        success: true,
        content: processedContent,
        warnings,
        metadata: {
          originalLength,
          processedLength: processedContent.length,
          sanitized,
          rateLimited,
          validationApplied,
        },
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      if (this.auditLogger) {
        await this.auditLogger.log({
          type: AuditEventType.SECURITY_VIOLATION,
          severity: AuditSeverity.HIGH,
          actor: {
            id: 'system',
            type: 'system',
          },
          target: {
            id: filePath,
            type: 'file',
            path: filePath,
          },
          action: 'content_processing_error',
          description: `Error processing content for ${operation} operation`,
          metadata: {
            error: errorMessage,
          },
        });
      }

      return {
        success: false,
        error: `Content processing failed: ${errorMessage}`,
        warnings,
        metadata: {
          originalLength,
          processedLength: 0,
          sanitized: false,
          rateLimited,
          validationApplied,
        },
      };
    }
  }

  /**
   * Get processing statistics
   */
  getStats(): {
    rateLimiterStats?: IRateLimitStats;
    processingCount: number;
  } {
    return {
      rateLimiterStats: this.rateLimiter?.getStats(),
      processingCount: 0, // Could be tracked if needed
    };
  }

  /**
   * Reset rate limiter state for all identifiers
   */
  resetRateLimit(): void {
    // Note: RateLimiter.reset() requires an identifier parameter
    // For now, we'll skip implementation or could reset specific patterns
    // this.rateLimiter?.reset('file_operations'); // if we track a global key
  }
}

/**
 * Create a secure content processor with default configuration
 */
export function createSecureContentProcessor(
  config?: Partial<ISecureContentConfig>
): SecureContentProcessor {
  return new SecureContentProcessor(config);
}

// Export validation rules for external use
export { ValidationRules };
