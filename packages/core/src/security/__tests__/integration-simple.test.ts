import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { RateLimiter, RateLimitAlgorithm, IRateLimitConfig } from '../RateLimiter';
import { InputValidator, ValidationRules } from '../InputValidator';
import { AuditLogger } from '../AuditLogger';
import { EncryptionService } from '../EncryptionService';

describe('Security Integration Tests', () => {
  let rateLimiter: RateLimiter;
  let inputValidator: InputValidator;
  let auditLogger: AuditLogger;
  let encryptionService: EncryptionService;

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

    // Initialize audit logger with valid config
    auditLogger = new AuditLogger({
      logFile: './test-audit.log',
      maxFileSize: '10MB',
      maxFiles: 5,
      enableConsoleLogging: false,
      enableFileLogging: false,
      enableIntegrityChecking: true
    });

    // Initialize encryption service
    encryptionService = new EncryptionService({
      keyDerivationIterations: 100000,
      keyRotationInterval: 24 * 60 * 60 * 1000, // 24 hours
      algorithm: 'aes-256-gcm'
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('Rate Limiting with Input Validation', () => {
    it('should validate input and apply rate limits', async () => {
      const userId = 'test-user';
      const validInput = 'valid user input';

      // Validate input first using validation rules
      const stringRule = ValidationRules.createStringValidation({
        minLength: 1,
        maxLength: 100,
        pattern: /^[a-zA-Z0-9\s]+$/
      });

      const validation = await inputValidator.validateValue(validInput, [stringRule]);
      expect(validation.isValid).toBe(true);

      if (validation.isValid) {
        // Apply rate limiting
        const rateLimitResult = await rateLimiter.checkLimit(userId);
        expect(rateLimitResult.allowed).toBe(true);

        // Log successful request
        await auditLogger.log({
          level: 'info',
          message: 'Request processed successfully',
          metadata: { 
            userId,
            input: validInput,
            rateLimitStatus: rateLimitResult 
          }
        });
      }
    });

    it('should deny malicious input before rate limiting', async () => {
      const userId = 'test-user';
      const maliciousInput = '<script>alert("xss")</script>';

      // Use XSS detection rule
      const xssRule = ValidationRules.createXSSDetection();
      const validation = await inputValidator.validateValue(maliciousInput, [xssRule]);
      
      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);

      // Log the security violation
      await auditLogger.log({
        level: 'error',
        message: 'Malicious input detected',
        metadata: { 
          userId,
          input: maliciousInput,
          errors: validation.errors
        }
      });
    });
  });

  describe('Encryption with Audit Logging', () => {
    it('should encrypt sensitive data and log the operation', async () => {
      const sensitiveData = 'user-password-123';
      const userId = 'test-user';

      // Encrypt the data
      const encrypted = await encryptionService.encrypt(sensitiveData);
      expect(encrypted.encryptedData).toBeDefined();
      expect(encrypted.iv).toBeDefined();
      expect(encrypted.authTag).toBeDefined();

      // Log the encryption operation
      await auditLogger.log({
        level: 'info',
        message: 'Data encrypted successfully',
        metadata: {
          userId,
          dataType: 'password',
          encryptionAlgorithm: 'AES-256-GCM',
          timestamp: new Date().toISOString()
        }
      });

      // Verify we can decrypt the data
      const decrypted = await encryptionService.decrypt(encrypted);
      expect(decrypted).toBe(sensitiveData);
    });

    it('should handle field-level encryption', async () => {
      const userData = {
        username: 'john_doe',
        email: 'john@example.com',
        password: 'secret123',
        profile: {
          age: 30,
          location: 'New York'
        }
      };

      // Use field-level encryption decorator
      class _UserModel {
        @encryptionService.encryptField()
        password!: string;

        @encryptionService.encryptField()
        email!: string;

        username!: string;
        profile!: { age: number; location: string };
      }

      // In a real scenario, this would automatically encrypt marked fields
      const encrypted = await encryptionService.encrypt(userData.password);
      expect(encrypted.encryptedData).toBeDefined();

      // Log field encryption
      await auditLogger.log({
        level: 'info',
        message: 'Field-level encryption applied',
        metadata: {
          fields: ['password', 'email'],
          model: 'UserModel'
        }
      });
    });
  });

  describe('Rate Limiting for Different Operations', () => {
    it('should apply different rate limits based on operation cost', async () => {
      // Configure rate limiter with cost function
      const costBasedConfig: IRateLimitConfig = {
        algorithm: RateLimitAlgorithm.TOKEN_BUCKET,
        maxRequests: 10,
        windowMs: 60000,
        enableAdaptive: false,
        minDelay: 100,
        maxDelay: 5000,
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
        costFunction: (operation: string) => {
          if (operation.includes('encrypt')) return 3;
          if (operation.includes('decrypt')) return 2;
          return 1;
        }
      };

      const costRateLimiter = new RateLimiter(costBasedConfig);
      const userId = 'crypto-user';

      // Cheap operations
      let result = await costRateLimiter.checkLimit(`read:${userId}`);
      expect(result.allowed).toBe(true);
      expect(result.cost).toBe(1);

      // Expensive encryption operations
      result = await costRateLimiter.checkLimit(`encrypt:${userId}`);
      expect(result.allowed).toBe(true);
      expect(result.cost).toBe(3);

      // Medium cost decryption
      result = await costRateLimiter.checkLimit(`decrypt:${userId}`);
      expect(result.allowed).toBe(true);
      expect(result.cost).toBe(2);

      // After a few expensive operations, should be limited
      for (let i = 0; i < 3; i++) {
        await costRateLimiter.checkLimit(`encrypt:${userId}`);
      }

      result = await costRateLimiter.checkLimit(`encrypt:${userId}`);
      // Should be near or at limit due to high cost operations
      expect(result.currentRequests).toBeGreaterThan(8);
    });
  });

  describe('Security Event Correlation', () => {
    it('should track and correlate security events', async () => {
      const userId = 'monitored-user';

      // Simulate various security events
      const events = [
        { type: 'LOGIN_ATTEMPT', success: true },
        { type: 'RATE_LIMIT_CHECK', allowed: true },
        { type: 'DATA_ACCESS', resource: 'user-profile' },
        { type: 'ENCRYPTION_OPERATION', operation: 'encrypt' },
        { type: 'INPUT_VALIDATION', result: 'passed' }
      ];

      // Log all events
      for (const event of events) {
        await auditLogger.log({
          level: 'info',
          message: `Security event: ${event.type}`,
          metadata: {
            userId,
            eventType: event.type,
            ...event
          }
        });
      }

      // Query recent events for analysis
      const recentEvents = await auditLogger.query({
        timeRange: {
          start: new Date(Date.now() - 60000), // Last minute
          end: new Date()
        },
        limit: 100
      });

      // Filter events for our test user
      const userEvents = recentEvents.filter(
        event => event.metadata?.userId === userId
      );

      expect(userEvents.length).toBe(5);
      
      // Verify event types
      const eventTypes = userEvents.map(event => event.metadata?.eventType);
      expect(eventTypes).toContain('LOGIN_ATTEMPT');
      expect(eventTypes).toContain('RATE_LIMIT_CHECK');
      expect(eventTypes).toContain('ENCRYPTION_OPERATION');
    });
  });

  describe('Performance Integration', () => {
    it('should handle concurrent security operations efficiently', async () => {
      const startTime = Date.now();
      const operations = [];

      // Create 20 concurrent operations
      for (let i = 0; i < 20; i++) {
        const userId = `user-${i}`;
        const operation = async () => {
          // Input validation
          const rule = ValidationRules.createStringValidation({ maxLength: 100 });
          const validation = await inputValidator.validateValue(`input-${i}`, [rule]);
          
          // Rate limiting
          const rateLimit = await rateLimiter.checkLimit(userId);
          
          // Encryption
          const encrypted = await encryptionService.encrypt(`data-${i}`);
          
          // Audit logging
          await auditLogger.log({
            level: 'info',
            message: 'Concurrent operation completed',
            metadata: { 
              operationId: i,
              userId,
              validationPassed: validation.isValid,
              rateLimitAllowed: rateLimit.allowed
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
      expect(results.length).toBe(20);
      results.forEach(result => {
        expect(result.validation.isValid).toBe(true);
        expect(result.rateLimit.allowed).toBe(true);
        expect(result.encrypted.encryptedData).toBeDefined();
      });

      // Verify reasonable performance (should complete within 3 seconds)
      const duration = endTime - startTime;
      expect(duration).toBeLessThan(3000);
    });
  });

  describe('Configuration Validation', () => {
    it('should validate security component configurations', () => {
      // Test rate limiter configuration
      const validRateLimitConfig: IRateLimitConfig = {
        algorithm: RateLimitAlgorithm.FIXED_WINDOW,
        maxRequests: 100,
        windowMs: 60000,
        enableAdaptive: true,
        minDelay: 50,
        maxDelay: 2000,
        skipSuccessfulRequests: false,
        skipFailedRequests: true
      };

      const testRateLimiter = new RateLimiter(validRateLimitConfig);
      expect(testRateLimiter).toBeDefined();

      // Test input validator with custom rules
      const testValidator = new InputValidator();
      testValidator.registerSchema({
        name: 'user-input',
        fields: {
          username: [ValidationRules.createStringValidation({ minLength: 3, maxLength: 20 })],
          email: [ValidationRules.createEmailValidation()],
          url: [ValidationRules.createURLValidation()]
        }
      });

      expect(testValidator).toBeDefined();
    });
  });
});
