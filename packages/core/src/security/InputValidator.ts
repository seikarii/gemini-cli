/**
 * @fileoverview Input validation system for security and data integrity
 * Provides comprehensive validation for user inputs, API requests, and data structures
 */

/**
 * Validation result interface
 */
export interface IValidationResult {
  /** Whether validation passed */
  isValid: boolean;
  /** Validation errors if any */
  errors: string[];
  /** Warnings that don't prevent validation */
  warnings: string[];
  /** Sanitized/normalized value if applicable */
  sanitizedValue?: unknown;
}

/**
 * Validation rule interface
 */
export interface IValidationRule<T = unknown> {
  /** Rule name/identifier */
  name: string;
  /** Validation function */
  validate: (value: T) => IValidationResult | Promise<IValidationResult>;
  /** Rule description */
  description?: string;
  /** Rule severity */
  severity: 'error' | 'warning';
}

/**
 * Schema validation configuration
 */
export interface IValidationSchema {
  /** Schema name */
  name: string;
  /** Field validations */
  fields: Record<string, IValidationRule[]>;
  /** Optional custom validation */
  customValidation?: (obj: Record<string, unknown>) => IValidationResult | Promise<IValidationResult>;
}

/**
 * Input sanitization options
 */
export interface ISanitizationOptions {
  /** Trim whitespace */
  trim: boolean;
  /** Remove HTML tags */
  stripHtml: boolean;
  /** Escape special characters */
  escapeHtml: boolean;
  /** Convert to lowercase */
  toLowerCase: boolean;
  /** Normalize unicode */
  normalizeUnicode: boolean;
  /** Maximum length */
  maxLength?: number;
}

/**
 * Pre-built validation rules
 */
export class ValidationRules {
  /**
   * String length validation
   */
  static stringLength(min?: number, max?: number): IValidationRule<string> {
    return {
      name: 'stringLength',
      description: `String length must be ${min ? `at least ${min}` : ''}${min && max ? ' and ' : ''}${max ? `at most ${max}` : ''}`,
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const errors: string[] = [];
        
        if (min !== undefined && value.length < min) {
          errors.push(`String must be at least ${min} characters long`);
        }
        
        if (max !== undefined && value.length > max) {
          errors.push(`String must be at most ${max} characters long`);
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings: []
        };
      }
    };
  }

  /**
   * Email validation
   */
  static email(): IValidationRule<string> {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    return {
      name: 'email',
      description: 'Must be a valid email address',
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const isValid = emailRegex.test(value);
        
        return {
          isValid,
          errors: isValid ? [] : ['Invalid email format'],
          warnings: []
        };
      }
    };
  }

  /**
   * URL validation
   */
  static url(protocols: string[] = ['http', 'https']): IValidationRule<string> {
    return {
      name: 'url',
      description: `Must be a valid URL with protocol: ${protocols.join(', ')}`,
      severity: 'error',
      validate: (value: string): IValidationResult => {
        try {
          const url = new URL(value);
          const isValid = protocols.includes(url.protocol.slice(0, -1));
          
          return {
            isValid,
            errors: isValid ? [] : [`URL must use one of these protocols: ${protocols.join(', ')}`],
            warnings: []
          };
        } catch {
          return {
            isValid: false,
            errors: ['Invalid URL format'],
            warnings: []
          };
        }
      }
    };
  }

  /**
   * File path validation
   */
  static filePath(allowedExtensions?: string[]): IValidationRule<string> {
    return {
      name: 'filePath',
      description: 'Must be a valid file path',
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const errors: string[] = [];
        
        // Check for path traversal attempts
        if (value.includes('..') || value.includes('\\..\\') || value.includes('/../')) {
          errors.push('Path traversal detected');
        }
        
        // Check for null bytes
        if (value.includes('\0')) {
          errors.push('Null bytes not allowed in file paths');
        }
        
        // Check file extension if specified
        if (allowedExtensions && allowedExtensions.length > 0) {
          const extension = value.split('.').pop()?.toLowerCase();
          if (!extension || !allowedExtensions.includes(extension)) {
            errors.push(`File extension must be one of: ${allowedExtensions.join(', ')}`);
          }
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings: []
        };
      }
    };
  }

  /**
   * Number range validation
   */
  static numberRange(min?: number, max?: number): IValidationRule<number> {
    return {
      name: 'numberRange',
      description: `Number must be ${min !== undefined ? `at least ${min}` : ''}${min !== undefined && max !== undefined ? ' and ' : ''}${max !== undefined ? `at most ${max}` : ''}`,
      severity: 'error',
      validate: (value: number): IValidationResult => {
        const errors: string[] = [];
        
        if (Number.isNaN(value)) {
          errors.push('Value must be a valid number');
        } else {
          if (min !== undefined && value < min) {
            errors.push(`Number must be at least ${min}`);
          }
          
          if (max !== undefined && value > max) {
            errors.push(`Number must be at most ${max}`);
          }
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings: []
        };
      }
    };
  }

  /**
   * Required field validation
   */
  static required(): IValidationRule {
    return {
      name: 'required',
      description: 'Field is required',
      severity: 'error',
      validate: (value: unknown): IValidationResult => {
        const isEmpty = value === null || 
                       value === undefined || 
                       (typeof value === 'string' && value.trim().length === 0) ||
                       (Array.isArray(value) && value.length === 0);
        
        return {
          isValid: !isEmpty,
          errors: isEmpty ? ['Field is required'] : [],
          warnings: []
        };
      }
    };
  }

  /**
   * Regular expression validation
   */
  static regex(pattern: RegExp, message = 'Invalid format'): IValidationRule<string> {
    return {
      name: 'regex',
      description: `Must match pattern: ${pattern.source}`,
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const isValid = pattern.test(value);
        
        return {
          isValid,
          errors: isValid ? [] : [message],
          warnings: []
        };
      }
    };
  }

  /**
   * SQL injection detection
   */
  static sqlInjection(): IValidationRule<string> {
    const sqlPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/i,
      /'(''|[^'])*'/,
      /;[\s]*(?:DROP|DELETE|UPDATE|INSERT)/i,
      /\bOR\b[\s]+[\w'"]+[\s]*=[\s]*[\w'"]+/i,
      /--[\s]*$/
    ];
    
    return {
      name: 'sqlInjection',
      description: 'Input contains potential SQL injection patterns',
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const warnings: string[] = [];
        const errors: string[] = [];
        
        for (const pattern of sqlPatterns) {
          if (pattern.test(value)) {
            errors.push('Potential SQL injection detected');
            break;
          }
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings
        };
      }
    };
  }

  /**
   * XSS detection
   */
  static xss(): IValidationRule<string> {
    const xssPatterns = [
      /<script[^>]*>.*?<\/script>/gi,
      /javascript:/gi,
      /on\w+[\s]*=/gi,
      /<iframe[^>]*>/gi,
      /<object[^>]*>/gi,
      /<embed[^>]*>/gi,
      /<link[^>]*>/gi,
      /<meta[^>]*>/gi
    ];
    
    return {
      name: 'xss',
      description: 'Input contains potential XSS patterns',
      severity: 'error',
      validate: (value: string): IValidationResult => {
        const errors: string[] = [];
        
        for (const pattern of xssPatterns) {
          if (pattern.test(value)) {
            errors.push('Potential XSS detected');
            break;
          }
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings: []
        };
      }
    };
  }
}

/**
 * Input sanitizer utility
 */
export class InputSanitizer {
  /**
   * Sanitize string input
   */
  static sanitizeString(value: string, options: Partial<ISanitizationOptions> = {}): string {
    const opts: ISanitizationOptions = {
      trim: true,
      stripHtml: false,
      escapeHtml: false,
      toLowerCase: false,
      normalizeUnicode: false,
      ...options
    };

    let result = value;

    // Trim whitespace
    if (opts.trim) {
      result = result.trim();
    }

    // Normalize unicode
    if (opts.normalizeUnicode) {
      result = result.normalize('NFKC');
    }

    // Strip HTML tags
    if (opts.stripHtml) {
      result = result.replace(/<[^>]*>/g, '');
    }

    // Escape HTML
    if (opts.escapeHtml) {
      result = result
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
    }

    // Convert to lowercase
    if (opts.toLowerCase) {
      result = result.toLowerCase();
    }

    // Truncate if max length specified
    if (opts.maxLength && result.length > opts.maxLength) {
      result = result.substring(0, opts.maxLength);
    }

    return result;
  }

  /**
   * Remove null bytes from string
   */
  static removeNullBytes(value: string): string {
    return value.replace(/\0/g, '');
  }

  /**
   * Sanitize file path
   */
  static sanitizeFilePath(value: string): string {
    return value
      .replace(/\.\./g, '') // Remove path traversal
      .replace(/[<>:"|?*]/g, '') // Remove invalid characters
      .replace(/\0/g, ''); // Remove null bytes
  }
}

/**
 * Main input validator class
 */
export class InputValidator {
  private schemas = new Map<string, IValidationSchema>();

  /**
   * Register a validation schema
   */
  registerSchema(schema: IValidationSchema): void {
    this.schemas.set(schema.name, schema);
  }

  /**
   * Validate a single value against rules
   */
  async validateValue<T>(value: T, rules: Array<IValidationRule<T>>): Promise<IValidationResult> {
    const allErrors: string[] = [];
    const allWarnings: string[] = [];
    let sanitizedValue = value;

    for (const rule of rules) {
      const result = await rule.validate(value);
      
      if (rule.severity === 'error') {
        allErrors.push(...result.errors);
      } else {
        allWarnings.push(...result.errors);
      }
      
      allWarnings.push(...result.warnings);
      
      if (result.sanitizedValue !== undefined) {
        sanitizedValue = result.sanitizedValue as T;
      }
    }

    return {
      isValid: allErrors.length === 0,
      errors: allErrors,
      warnings: allWarnings,
      sanitizedValue
    };
  }

  /**
   * Validate an object against a schema
   */
  async validateObject(obj: Record<string, unknown>, schemaName: string): Promise<IValidationResult> {
    const schema = this.schemas.get(schemaName);
    if (!schema) {
      return {
        isValid: false,
        errors: [`Schema '${schemaName}' not found`],
        warnings: []
      };
    }

    const allErrors: string[] = [];
    const allWarnings: string[] = [];
    const sanitizedObj: Record<string, unknown> = { ...obj };

    // Validate each field
    for (const [fieldName, rules] of Object.entries(schema.fields)) {
      const fieldValue = obj[fieldName];
      const result = await this.validateValue(fieldValue, rules);
      
      if (!result.isValid) {
        allErrors.push(...result.errors.map(error => `${fieldName}: ${error}`));
      }
      
      allWarnings.push(...result.warnings.map(warning => `${fieldName}: ${warning}`));
      
      if (result.sanitizedValue !== undefined) {
        sanitizedObj[fieldName] = result.sanitizedValue;
      }
    }

    // Run custom validation if present
    if (schema.customValidation) {
      const customResult = await schema.customValidation(sanitizedObj);
      allErrors.push(...customResult.errors);
      allWarnings.push(...customResult.warnings);
    }

    return {
      isValid: allErrors.length === 0,
      errors: allErrors,
      warnings: allWarnings,
      sanitizedValue: sanitizedObj
    };
  }

  /**
   * Quick validation with common rules
   */
  static async validateBasic(value: unknown, type: 'string' | 'number' | 'email' | 'url' | 'filePath'): Promise<IValidationResult> {
    const validator = new InputValidator();
    
    let rules: Array<IValidationRule<unknown>>;
    
    switch (type) {
      case 'string':
        rules = [ValidationRules.required(), ValidationRules.stringLength(1, 10000)] as Array<IValidationRule<unknown>>;
        break;
      case 'number':
        rules = [ValidationRules.required(), ValidationRules.numberRange()] as Array<IValidationRule<unknown>>;
        break;
      case 'email':
        rules = [ValidationRules.required(), ValidationRules.email()] as Array<IValidationRule<unknown>>;
        break;
      case 'url':
        rules = [ValidationRules.required(), ValidationRules.url()] as Array<IValidationRule<unknown>>;
        break;
      case 'filePath':
        rules = [ValidationRules.required(), ValidationRules.filePath()] as Array<IValidationRule<unknown>>;
        break;
      default:
        rules = [ValidationRules.required()];
    }
    
    return validator.validateValue(value, rules);
  }
}

/**
 * Create commonly used validation schemas
 */
export const CommonSchemas = {
  /**
   * API request validation schema
   */
  apiRequest: (): IValidationSchema => ({
    name: 'apiRequest',
    fields: {
      method: [ValidationRules.required(), ValidationRules.regex(/^(GET|POST|PUT|DELETE|PATCH)$/, 'Invalid HTTP method')] as IValidationRule[],
      url: [ValidationRules.required(), ValidationRules.url()] as IValidationRule[],
      body: [], // Optional
      headers: [] // Optional
    }
  }),

  /**
   * File upload validation schema
   */
  fileUpload: (allowedExtensions: string[]): IValidationSchema => ({
    name: 'fileUpload',
    fields: {
      filename: [ValidationRules.required(), ValidationRules.filePath(allowedExtensions)] as IValidationRule[],
      size: [ValidationRules.required(), ValidationRules.numberRange(1, 100 * 1024 * 1024)] as IValidationRule[], // Max 100MB
      mimetype: [ValidationRules.required()]
    }
  }),

  /**
   * User input validation schema
   */
  userInput: (): IValidationSchema => ({
    name: 'userInput',
    fields: {
      content: [
        ValidationRules.required(),
        ValidationRules.stringLength(1, 50000),
        ValidationRules.xss(),
        ValidationRules.sqlInjection()
      ] as IValidationRule[]
    }
  })
};
