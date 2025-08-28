import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PluginSandbox, ISecurityPolicy } from '../PluginSandbox';
import { RateLimiter, RateLimitAlgorithm, IRateLimitConfig } from '../RateLimiter';
import { InputValidator } from '../InputValidator';
import { AuditLogger } from '../AuditLogger';
import { EncryptionService } from '../EncryptionService';

describe('Security Integration Tests', () => {
  let rateLimiter: RateLimiter;
  let inputValidator: InputValidator;
  let auditLogger: AuditLogger;
  let encryptionService: EncryptionService;
  let _sandbox: PluginSandbox;

  beforeEach(async () => {
    vi.useFakeTimers();

    // Initialize rate limiter
    const rateLimitConfig: IRateLimitConfig = {
      algorithm: RateLimitAlgorithm.TOKEN_BUCKET,
      maxRequests: 10,
      windowMs: 60000,
      enableAdaptive: false,
      minDelay: 100,
      maxDelay: 5000,
      skipSuccessfulRequests: false,
      skipFailedRequests: false
    };
    rateLimiter = new RateLimiter(rateLimitConfig);

    // Initialize input validator
    inputValidator = new InputValidator();

    // Initialize audit logger
    auditLogger = new AuditLogger({
      logLevel: 'info',
      enableConsoleLogging: false,
      enableFileLogging: false,
      enableIntegrityChecking: true
    });

    // Initialize encryption service
    encryptionService = new EncryptionService();
    await encryptionService.initialize();

    // Initialize plugin sandbox
    sandbox = new PluginSandbox();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('Rate Limiting with Input Validation', () => {
    it('should validate input before applying rate limits', async () => {
      const userId = 'test-user';
      const maliciousInput = '<script>alert("xss")</script>';

      // First validate the input
      const validation = inputValidator.validateInput(maliciousInput, 'html');
      expect(validation.isValid).toBe(false);
      expect(validation.errors.some(e => e.type === 'XSS_DETECTED')).toBe(true);

      // If validation fails, don't even check rate limits
      if (!validation.isValid) {
        await auditLogger.logSecurityEvent({
          type: 'INPUT_VALIDATION_FAILED',
          severity: 'high',
          userId,
          details: { errors: validation.errors }
        });

        const auditEntries = await auditLogger.query({
          eventType: 'INPUT_VALIDATION_FAILED',
          limit: 1
        });
        expect(auditEntries.length).toBe(1);
        expect(auditEntries[0].details.errors).toEqual(validation.errors);
      }
    });

    it('should apply rate limiting after successful input validation', async () => {
      const userId = 'test-user';
      const validInput = 'valid user input';

      // Validate input first
      const validation = inputValidator.validateInput(validInput, 'text');
      expect(validation.isValid).toBe(true);

      if (validation.isValid) {
        // Apply rate limiting
        const rateLimitResult = await rateLimiter.checkLimit(userId);
        expect(rateLimitResult.allowed).toBe(true);

        // Log successful request
        await auditLogger.logSecurityEvent({
          type: 'REQUEST_PROCESSED',
          severity: 'info',
          userId,
          details: { 
            input: validInput,
            rateLimitStatus: rateLimitResult 
          }
        });
      }
    });
  });

  describe('Encryption with Audit Logging', () => {
    it('should encrypt sensitive data and log the operation', async () => {
      const sensitiveData = 'user-password-123';
      const userId = 'test-user';

      // Encrypt the data
      const encrypted = await encryptionService.encrypt(sensitiveData);
      expect(encrypted.ciphertext).toBeDefined();
      expect(encrypted.iv).toBeDefined();
      expect(encrypted.tag).toBeDefined();

      // Log the encryption operation
      await auditLogger.logSecurityEvent({
        type: 'DATA_ENCRYPTED',
        severity: 'info',
        userId,
        details: {
          dataType: 'password',
          encryptionAlgorithm: 'AES-256-GCM',
          keyId: encrypted.keyId
        }
      });

      // Verify the audit log
      const auditEntries = await auditLogger.query({
        eventType: 'DATA_ENCRYPTED',
        userId,
        limit: 1
      });
      expect(auditEntries.length).toBe(1);
      expect(auditEntries[0].details.encryptionAlgorithm).toBe('AES-256-GCM');
    });

    it('should decrypt data and verify integrity', async () => {
      const originalData = 'sensitive-information';
      const userId = 'test-user';

      // Encrypt then decrypt
      const encrypted = await encryptionService.encrypt(originalData);
      const decrypted = await encryptionService.decrypt(encrypted);

      expect(decrypted).toBe(originalData);

      // Log successful decryption
      await auditLogger.logSecurityEvent({
        type: 'DATA_DECRYPTED',
        severity: 'info',
        userId,
        details: {
          keyId: encrypted.keyId,
          success: true
        }
      });
    });
  });

  describe('Plugin Sandbox with Full Security Stack', () => {
    it('should validate plugin input, apply rate limits, and audit execution', async () => {
      const userId = 'plugin-user';
      const pluginCode = `
        module.exports = {
          name: 'test-plugin',
          execute: (input) => ({ result: 'processed: ' + input })
        };
      `;

      // 1. Validate plugin code
      const codeValidation = inputValidator.validateInput(pluginCode, 'javascript');
      expect(codeValidation.isValid).toBe(true);

      // 2. Check rate limits for plugin execution
      const rateLimitResult = await rateLimiter.checkLimit(`plugin:${userId}`);
      expect(rateLimitResult.allowed).toBe(true);

      // 3. Log plugin execution attempt
      await auditLogger.logSecurityEvent({
        type: 'PLUGIN_EXECUTION_START',
        severity: 'info',
        userId,
        details: {
          pluginName: 'test-plugin',
          rateLimitStatus: rateLimitResult
        }
      });

      // 4. Execute plugin in sandbox (would normally load and execute)
      const securityPolicy: ISecurityPolicy = {
        maxExecutionTime: 5000,
        maxMemoryUsage: 50 * 1024 * 1024,
        allowedHosts: [],
        allowedReadPaths: [],
        allowedWritePaths: [],
        allowFileSystem: false,
        allowNetwork: false,
        allowChildProcesses: false,
        apiCallLimit: 10,
        requireSignature: false,
        environmentVariables: {}
      };

      // Mock successful plugin execution
      const executionResult = {
        success: true,
        output: { result: 'processed: test input' },
        errors: [],
        warnings: [],
        duration: 100,
        memoryUsage: { peak: 1024, current: 512 },
        apiCalls: 0,
        securityViolations: [],
        context: {
          id: 'test-plugin-id',
          name: 'test-plugin',
          version: '1.0.0',
          author: 'test-author',
          description: 'Test plugin',
          permissions: [],
          sourceHash: 'mock-hash',
          entryPoint: 'index.js',
          dependencies: {}
        }
      };

      // 5. Log plugin execution completion
      await auditLogger.logSecurityEvent({
        type: 'PLUGIN_EXECUTION_COMPLETE',
        severity: 'info',
        userId,
        details: {
          pluginName: 'test-plugin',
          success: executionResult.success,
          duration: executionResult.duration,
          securityViolations: executionResult.securityViolations
        }
      });

      // Verify audit trail
      const auditEntries = await auditLogger.query({
        userId,
        limit: 10
      });
      expect(auditEntries.length).toBeGreaterThanOrEqual(2);
      
      const startEvent = auditEntries.find(e => e.eventType === 'PLUGIN_EXECUTION_START');
      const completeEvent = auditEntries.find(e => e.eventType === 'PLUGIN_EXECUTION_COMPLETE');
      
      expect(startEvent).toBeDefined();
      expect(completeEvent).toBeDefined();
    });
  });

  describe('Security Event Correlation', () => {
    it('should correlate multiple security events for threat detection', async () => {
      const suspiciousUserId = 'suspicious-user';
      
      // Simulate multiple suspicious activities
      const activities = [
        { type: 'MULTIPLE_FAILED_LOGINS', count: 5 },
        { type: 'RATE_LIMIT_EXCEEDED', count: 3 },
        { type: 'INPUT_VALIDATION_FAILED', count: 10 },
        { type: 'UNAUTHORIZED_ACCESS_ATTEMPT', count: 2 }
      ];

      // Log all activities
      for (const activity of activities) {
        for (let i = 0; i < activity.count; i++) {
          await auditLogger.logSecurityEvent({
            type: activity.type as any,
            severity: 'medium',
            userId: suspiciousUserId,
            details: { attemptNumber: i + 1 }
          });
        }
      }

      // Query for all events from this user
      const userEvents = await auditLogger.query({
        userId: suspiciousUserId,
        limit: 100
      });

      expect(userEvents.length).toBe(20); // Total of all activities

      // Analyze for threat patterns
      const eventsByType = userEvents.reduce((acc, event) => {
        acc[event.eventType] = (acc[event.eventType] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      expect(eventsByType['MULTIPLE_FAILED_LOGINS']).toBe(5);
      expect(eventsByType['INPUT_VALIDATION_FAILED']).toBe(10);

      // This would trigger automated threat detection in a real system
      const threatScore = Object.values(eventsByType).reduce((sum, count) => sum + count, 0);
      expect(threatScore).toBeGreaterThan(15); // High threat score
    });
  });

  describe('Performance under Load', () => {
    it('should handle concurrent security operations efficiently', async () => {
      const startTime = Date.now();
      const operations = [];

      // Create 50 concurrent operations involving all security components
      for (let i = 0; i < 50; i++) {
        const userId = `user-${i}`;
        const operation = async () => {
          // Input validation
          const validation = inputValidator.validateInput(`input-${i}`, 'text');
          
          // Rate limiting
          const rateLimit = await rateLimiter.checkLimit(userId);
          
          // Encryption
          const encrypted = await encryptionService.encrypt(`data-${i}`);
          
          // Audit logging
          await auditLogger.logSecurityEvent({
            type: 'BULK_OPERATION',
            severity: 'info',
            userId,
            details: { 
              operationId: i,
              validationResult: validation.isValid,
              rateLimitAllowed: rateLimit.allowed,
              encryptionKeyId: encrypted.keyId
            }
          });

          return { validation, rateLimit, encrypted };
        };

        operations.push(operation());
      }

      // Wait for all operations to complete
      const results = await Promise.all(operations);
      const endTime = Date.now();

      // Verify all operations completed successfully
      expect(results.length).toBe(50);
      results.forEach((result, index) => {
        expect(result.validation.isValid).toBe(true);
        expect(result.rateLimit.allowed).toBe(true);
        expect(result.encrypted.ciphertext).toBeDefined();
      });

      // Verify reasonable performance (should complete within 5 seconds)
      const duration = endTime - startTime;
      expect(duration).toBeLessThan(5000);

      // Verify audit log captured all operations
      const auditEntries = await auditLogger.query({
        eventType: 'BULK_OPERATION',
        limit: 100
      });
      expect(auditEntries.length).toBe(50);
    });
  });

  describe('Security Configuration Validation', () => {
    it('should validate and enforce consistent security policies', async () => {
      // Test that components work together with compatible configurations
      
      // Rate limiter should respect encryption overhead
      const heavyEncryptionConfig: IRateLimitConfig = {
        algorithm: RateLimitAlgorithm.TOKEN_BUCKET,
        maxRequests: 5, // Lower limit for encryption-heavy operations
        windowMs: 60000,
        enableAdaptive: true,
        minDelay: 200, // Higher delays for crypto operations
        maxDelay: 10000,
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
        costFunction: (operation: string) => operation.includes('encrypt') ? 3 : 1
      };

      const cryptoRateLimiter = new RateLimiter(heavyEncryptionConfig);

      // Test encryption operations with rate limiting
      const userId = 'crypto-user';
      
      for (let i = 0; i < 3; i++) {
        const rateLimitResult = await cryptoRateLimiter.checkLimit(`encrypt:${userId}`);
        if (rateLimitResult.allowed) {
          const encrypted = await encryptionService.encrypt(`sensitive-data-${i}`);
          expect(encrypted.ciphertext).toBeDefined();
          
          // Higher cost operation should consume more rate limit tokens
          expect(rateLimitResult.cost).toBe(3);
        }
      }

      // Should be near or at limit after 3 heavy operations
      const finalCheck = await cryptoRateLimiter.checkLimit(`encrypt:${userId}`);
      expect(finalCheck.currentRequests).toBeGreaterThan(5);
    });
  });
});
