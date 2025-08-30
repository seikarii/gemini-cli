/**
 * @fileoverview Security integration utilities for file operations
 * Provides easy integration of security measures into existing file processing tools
 */

import { SecureContentProcessor, createSecureContentProcessor } from '../security/SecureContentProcessor.js';

/**
 * Global secure content processor instance
 */
let globalSecureProcessor: SecureContentProcessor | null = null;

/**
 * Configuration for secure file processing
 */
export interface ISecureFileConfig {
  /** Enable security features */
  enabled: boolean;
  /** Enable rate limiting */
  enableRateLimit?: boolean;
  /** Enable input validation */
  enableInputValidation?: boolean;
  /** Enable content sanitization */
  enableContentSanitization?: boolean;
  /** Maximum content length for processing */
  maxContentLength?: number;
}

/**
 * Default secure file configuration
 */
const DEFAULT_SECURE_CONFIG: ISecureFileConfig = {
  enabled: true,
  enableRateLimit: true,
  enableInputValidation: true,
  enableContentSanitization: true,
  maxContentLength: 100000, // 100KB
};

/**
 * Initialize the global secure processor
 */
export function initializeSecureFileProcessing(config: Partial<ISecureFileConfig> = {}): void {
  const finalConfig = { ...DEFAULT_SECURE_CONFIG, ...config };
  
  if (!finalConfig.enabled) {
    globalSecureProcessor = null;
    return;
  }

  globalSecureProcessor = createSecureContentProcessor({
    enableRateLimit: finalConfig.enableRateLimit ?? true,
    enableInputValidation: finalConfig.enableInputValidation ?? true,
    enableContentSanitization: finalConfig.enableContentSanitization ?? true,
    enableAuditLogging: false, // Keep logging minimal for performance
    maxContentLength: finalConfig.maxContentLength ?? 100000,
    maxFileSize: 10 * 1024 * 1024, // 10MB
  });
}

/**
 * Get the global secure processor instance
 */
export function getSecureProcessor(): SecureContentProcessor | null {
  return globalSecureProcessor;
}

/**
 * Process file content for reading operations
 */
export async function secureProcessReadContent(
  content: string,
  filePath: string
): Promise<{ success: boolean; content?: string; error?: string; warnings: string[] }> {
  if (!globalSecureProcessor) {
    // Security disabled, return content as-is
    return { success: true, content, warnings: [] };
  }

  try {
    const result = await globalSecureProcessor.processContent(content, filePath, 'read');
    
    if (!result.success) {
      return {
        success: false,
        error: result.error,
        warnings: result.warnings,
      };
    }

    return {
      success: true,
      content: result.content,
      warnings: result.warnings,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      error: `Security processing failed: ${errorMessage}`,
      warnings: [],
    };
  }
}

/**
 * Process file content for writing operations
 */
export async function secureProcessWriteContent(
  content: string,
  filePath: string
): Promise<{ success: boolean; content?: string; error?: string; warnings: string[] }> {
  if (!globalSecureProcessor) {
    // Security disabled, return content as-is
    return { success: true, content, warnings: [] };
  }

  try {
    const result = await globalSecureProcessor.processContent(content, filePath, 'write');
    
    if (!result.success) {
      return {
        success: false,
        error: result.error,
        warnings: result.warnings,
      };
    }

    return {
      success: true,
      content: result.content,
      warnings: result.warnings,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      error: `Security processing failed: ${errorMessage}`,
      warnings: [],
    };
  }
}

/**
 * Process file content for editing operations
 */
export async function secureProcessEditContent(
  content: string,
  filePath: string
): Promise<{ success: boolean; content?: string; error?: string; warnings: string[] }> {
  if (!globalSecureProcessor) {
    // Security disabled, return content as-is
    return { success: true, content, warnings: [] };
  }

  try {
    const result = await globalSecureProcessor.processContent(content, filePath, 'edit');
    
    if (!result.success) {
      return {
        success: false,
        error: result.error,
        warnings: result.warnings,
      };
    }

    return {
      success: true,
      content: result.content,
      warnings: result.warnings,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      error: `Security processing failed: ${errorMessage}`,
      warnings: [],
    };
  }
}

/**
 * Check if security processing is enabled
 */
export function isSecureProcessingEnabled(): boolean {
  return globalSecureProcessor !== null;
}

/**
 * Get security processing statistics
 */
export function getSecurityStats(): { enabled: boolean; stats?: Record<string, unknown> } {
  if (!globalSecureProcessor) {
    return { enabled: false };
  }

  return {
    enabled: true,
    stats: globalSecureProcessor.getStats(),
  };
}

/**
 * Reset security state (useful for testing)
 */
export function resetSecurityState(): void {
  if (globalSecureProcessor) {
    globalSecureProcessor.resetRateLimit();
  }
}

/**
 * Environment variable based auto-initialization
 * This can be called during module initialization to auto-enable security
 */
export function autoInitializeFromEnvironment(): void {
  const enabled = process.env['GEMINI_SECURE_FILE_PROCESSING'] !== 'false';
  const enableRateLimit = process.env['GEMINI_ENABLE_RATE_LIMIT'] !== 'false';
  const enableValidation = process.env['GEMINI_ENABLE_INPUT_VALIDATION'] !== 'false';
  const enableSanitization = process.env['GEMINI_ENABLE_CONTENT_SANITIZATION'] !== 'false';
  const maxContentLength = process.env['GEMINI_MAX_CONTENT_LENGTH'] 
    ? parseInt(process.env['GEMINI_MAX_CONTENT_LENGTH'], 10)
    : undefined;

  if (enabled) {
    initializeSecureFileProcessing({
      enabled: true,
      enableRateLimit,
      enableInputValidation: enableValidation,
      enableContentSanitization: enableSanitization,
      maxContentLength,
    });
  }
}

// Auto-initialize on module load
autoInitializeFromEnvironment();
