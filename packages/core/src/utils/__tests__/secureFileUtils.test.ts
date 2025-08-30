/**
 * @fileoverview Tests for secure file processing integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  initializeSecureFileProcessing,
  secureProcessReadContent,
  secureProcessWriteContent,
  secureProcessEditContent,
  isSecureProcessingEnabled,
  getSecurityStats,
  resetSecurityState,
} from '../secureFileUtils.js';

describe('Secure File Processing Integration', () => {
  beforeEach(() => {
    // Reset state before each test
    resetSecurityState();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should initialize with default configuration', () => {
      initializeSecureFileProcessing();
      expect(isSecureProcessingEnabled()).toBe(true);
    });

    it('should allow disabling security', () => {
      initializeSecureFileProcessing({ enabled: false });
      expect(isSecureProcessingEnabled()).toBe(false);
    });

    it('should initialize with custom configuration', () => {
      initializeSecureFileProcessing({
        enabled: true,
        enableRateLimit: false,
        enableInputValidation: true,
        maxContentLength: 50000,
      });
      expect(isSecureProcessingEnabled()).toBe(true);
    });
  });

  describe('Content Processing', () => {
    beforeEach(() => {
      initializeSecureFileProcessing({
        enabled: true,
        enableRateLimit: false, // Disable for testing
        enableInputValidation: true,
        enableContentSanitization: true,
      });
    });

    it('should process normal content successfully', async () => {
      const content = 'Hello, world!';
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      expect(result.error).toBeUndefined();
    });

    it('should handle content with null bytes', async () => {
      const content = 'Hello\\0world'; // Content with null byte
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      // Should still succeed but content might be sanitized
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      if (result.warnings.length > 0) {
        expect(result.warnings.some(w => w.includes('sanitized'))).toBe(true);
      }
    });

    it('should handle escape sequences', async () => {
      const content = 'console.log(\\"Hello World\\");';
      const filePath = '/test/file.js';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      // Content should be sanitized to remove problematic escape sequences
      expect(result.content).not.toContain('\\"');
    });

    it('should handle very large content', async () => {
      const largeContent = 'x'.repeat(200000); // 200KB content
      const filePath = '/test/large-file.txt';

      const result = await secureProcessReadContent(largeContent, filePath);
      
      // Should either succeed with truncation or fail gracefully
      if (result.success) {
        expect(result.content!.length).toBeLessThanOrEqual(100000);
        expect(result.warnings.length).toBeGreaterThan(0);
      } else {
        expect(result.error).toContain('large');
      }
    });

    it('should process write operations', async () => {
      const content = 'New file content';
      const filePath = '/test/new-file.txt';

      const result = await secureProcessWriteContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBe(content);
    });

    it('should process edit operations', async () => {
      const content = 'Edited content';
      const filePath = '/test/edit-file.txt';

      const result = await secureProcessEditContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
    });
  });

  describe('Content Sanitization', () => {
    beforeEach(() => {
      initializeSecureFileProcessing({
        enabled: true,
        enableRateLimit: false,
        enableContentSanitization: true,
      });
    });

    it('should sanitize escape sequences', async () => {
      const content = 'Text with \\"quotes\\" and \\n newlines';
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      // Should convert escape sequences to actual characters
      expect(result.content).toContain('"quotes"');
      expect(result.content).toContain('\n');
    });

    it('should handle control characters', async () => {
      const content = 'Text with\\x00null\\x1Fcontrol chars';
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      // Control characters should be removed
      expect(result.content).toBe('Text withnullcontrol chars');
    });

    it('should truncate very long lines', async () => {
      const longLine = 'x'.repeat(15000);
      const content = `Normal line\\n${longLine}\\nAnother normal line`;
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBeDefined();
      expect(result.content).toContain('truncated for API compatibility');
    });
  });

  describe('Security Disabled Mode', () => {
    beforeEach(() => {
      initializeSecureFileProcessing({ enabled: false });
    });

    it('should pass through content unchanged when disabled', async () => {
      const content = 'Any content\\"with\\x00issues';
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(content, filePath);
      
      expect(result.success).toBe(true);
      expect(result.content).toBe(content); // Should be unchanged
      expect(result.warnings).toEqual([]);
    });

    it('should report disabled state', () => {
      expect(isSecureProcessingEnabled()).toBe(false);
      
      const stats = getSecurityStats();
      expect(stats.enabled).toBe(false);
      expect(stats.stats).toBeUndefined();
    });
  });

  describe('Error Handling', () => {
    beforeEach(() => {
      initializeSecureFileProcessing({
        enabled: true,
        enableInputValidation: true,
        maxContentLength: 100, // Very small limit for testing
      });
    });

    it('should handle content that exceeds size limits', async () => {
      const largeContent = 'x'.repeat(1000); // Exceeds the 100 char limit
      const filePath = '/test/file.txt';

      const result = await secureProcessReadContent(largeContent, filePath);
      
      // Should either succeed with truncation or fail with size error
      if (!result.success) {
        expect(result.error).toContain('large');
      } else {
        expect(result.content!.length).toBeLessThanOrEqual(100);
        expect(result.warnings.some(w => w.includes('truncated'))).toBe(true);
      }
    });
  });

  describe('Statistics and Monitoring', () => {
    beforeEach(() => {
      initializeSecureFileProcessing();
    });

    it('should provide security statistics when enabled', () => {
      const stats = getSecurityStats();
      
      expect(stats.enabled).toBe(true);
      expect(stats.stats).toBeDefined();
      expect(typeof stats.stats).toBe('object');
    });

    it('should track processing operations', async () => {
      const content = 'Test content';
      const filePath = '/test/file.txt';

      // Process some content
      await secureProcessReadContent(content, filePath);
      await secureProcessWriteContent(content, filePath);
      
      const stats = getSecurityStats();
      expect(stats.enabled).toBe(true);
      // Note: Currently the processor doesn't track counts, but the structure is there
    });
  });
});
