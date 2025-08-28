/**
 * @fileoverview Audit logging system for security monitoring and compliance
 * Provides comprehensive logging of security events, user actions, and system changes
 */

import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { createHash } from 'crypto';
import { join } from 'path';

/**
 * Audit event types
 */
export enum AuditEventType {
  // Authentication events
  LOGIN_ATTEMPT = 'login_attempt',
  LOGIN_SUCCESS = 'login_success',
  LOGIN_FAILURE = 'login_failure',
  LOGOUT = 'logout',
  TOKEN_GENERATED = 'token_generated',
  TOKEN_REVOKED = 'token_revoked',

  // Authorization events
  ACCESS_GRANTED = 'access_granted',
  ACCESS_DENIED = 'access_denied',
  PERMISSION_CHANGED = 'permission_changed',
  ROLE_ASSIGNED = 'role_assigned',
  ROLE_REVOKED = 'role_revoked',

  // Data events
  DATA_ACCESS = 'data_access',
  DATA_MODIFIED = 'data_modified',
  DATA_DELETED = 'data_deleted',
  DATA_EXPORTED = 'data_exported',
  DATA_IMPORTED = 'data_imported',

  // System events
  SYSTEM_START = 'system_start',
  SYSTEM_SHUTDOWN = 'system_shutdown',
  CONFIG_CHANGED = 'config_changed',
  PLUGIN_LOADED = 'plugin_loaded',
  PLUGIN_UNLOADED = 'plugin_unloaded',

  // Security events
  SECURITY_VIOLATION = 'security_violation',
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  VALIDATION_FAILED = 'validation_failed',
  ENCRYPTION_KEY_ROTATED = 'encryption_key_rotated',
  SUSPICIOUS_ACTIVITY = 'suspicious_activity',

  // API events
  API_CALL = 'api_call',
  API_ERROR = 'api_error',
  API_RATE_LIMITED = 'api_rate_limited',

  // File system events
  FILE_CREATED = 'file_created',
  FILE_MODIFIED = 'file_modified',
  FILE_DELETED = 'file_deleted',
  FILE_ACCESSED = 'file_accessed',

  // Custom events
  CUSTOM = 'custom'
}

/**
 * Audit event severity levels
 */
export enum AuditSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

/**
 * Audit event interface
 */
export interface IAuditEvent {
  /** Unique event ID */
  id: string;
  /** Event type */
  type: AuditEventType;
  /** Event severity */
  severity: AuditSeverity;
  /** Timestamp in ISO format */
  timestamp: string;
  /** User or system component that triggered the event */
  actor: {
    id?: string;
    type: 'user' | 'system' | 'api' | 'plugin';
    name?: string;
    ip?: string;
    userAgent?: string;
  };
  /** Target of the action */
  target?: {
    id?: string;
    type: string;
    name?: string;
    path?: string;
  };
  /** Action performed */
  action: string;
  /** Event description */
  description: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
  /** Request/session ID for correlation */
  correlationId?: string;
  /** Event tags for categorization */
  tags?: string[];
  /** Hash of event content for integrity */
  hash?: string;
}

/**
 * Audit logger configuration
 */
export interface IAuditLoggerConfig {
  /** Enable audit logging */
  enabled: boolean;
  /** Log file path */
  logFilePath: string;
  /** Maximum log file size in bytes */
  maxFileSize: number;
  /** Number of log files to keep */
  maxFiles: number;
  /** Log rotation interval in milliseconds */
  rotationInterval: number;
  /** Minimum severity level to log */
  minSeverity: AuditSeverity;
  /** Enable log encryption */
  enableEncryption: boolean;
  /** Encryption key for log files */
  encryptionKey?: string;
  /** Enable log integrity checking */
  enableIntegrityCheck: boolean;
  /** Enable real-time alerting */
  enableAlerting: boolean;
  /** Alert configuration */
  alertConfig?: {
    webhookUrl?: string;
    emailEndpoint?: string;
    criticalEventsOnly: boolean;
  };
}

/**
 * Audit statistics
 */
export interface IAuditStats {
  /** Total events logged */
  totalEvents: number;
  /** Events by type */
  eventsByType: Record<AuditEventType, number>;
  /** Events by severity */
  eventsBySeverity: Record<AuditSeverity, number>;
  /** Events in last 24 hours */
  eventsLast24h: number;
  /** Current log file size */
  currentLogSize: number;
  /** Number of log files */
  logFileCount: number;
  /** Last rotation timestamp */
  lastRotation: string;
}

/**
 * Audit query interface
 */
export interface IAuditQuery {
  /** Start date/time */
  startDate?: Date;
  /** End date/time */
  endDate?: Date;
  /** Event types to include */
  types?: AuditEventType[];
  /** Severity levels to include */
  severities?: AuditSeverity[];
  /** Actor ID filter */
  actorId?: string;
  /** Target ID filter */
  targetId?: string;
  /** Correlation ID filter */
  correlationId?: string;
  /** Text search in description */
  searchText?: string;
  /** Tags to match */
  tags?: string[];
  /** Maximum number of results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Main audit logger class
 */
export class AuditLogger extends EventEmitter {
  private config: IAuditLoggerConfig;
  private stats: IAuditStats;
  private logBuffer: IAuditEvent[] = [];
  private lastRotation = Date.now();
  private rotationTimer?: NodeJS.Timeout;

  constructor(config: IAuditLoggerConfig) {
    super();
    
    this.config = {
      enabled: config.enabled,
      logFilePath: config.logFilePath,
      maxFileSize: config.maxFileSize ?? 10 * 1024 * 1024, // 10MB
      maxFiles: config.maxFiles ?? 10,
      rotationInterval: config.rotationInterval ?? 24 * 60 * 60 * 1000, // 24 hours
      minSeverity: config.minSeverity ?? AuditSeverity.LOW,
      enableEncryption: config.enableEncryption ?? false,
      enableIntegrityCheck: config.enableIntegrityCheck ?? true,
      enableAlerting: config.enableAlerting ?? false,
      encryptionKey: config.encryptionKey,
    };

    this.stats = {
      totalEvents: 0,
      eventsByType: {} as Record<AuditEventType, number>,
      eventsBySeverity: {} as Record<AuditSeverity, number>,
      eventsLast24h: 0,
      currentLogSize: 0,
      logFileCount: 0,
      lastRotation: new Date().toISOString()
    };

    this.initialize();
  }

  /**
   * Initialize audit logger
   */
  private async initialize(): Promise<void> {
    if (!this.config.enabled) {
      return;
    }

    try {
      // Ensure log directory exists
      const logDir = join(this.config.logFilePath, '..');
      await fs.mkdir(logDir, { recursive: true });

      // Start rotation timer
      this.startRotationTimer();

      this.emit('initialized');
    } catch (error) {
      this.emit('error', new Error(`Failed to initialize audit logger: ${error}`));
    }
  }

  /**
   * Log an audit event
   */
  async log(event: Omit<IAuditEvent, 'id' | 'timestamp' | 'hash'>): Promise<void> {
    if (!this.config.enabled) {
      return;
    }

    // Check severity filter
    const severityOrder = [AuditSeverity.LOW, AuditSeverity.MEDIUM, AuditSeverity.HIGH, AuditSeverity.CRITICAL];
    if (severityOrder.indexOf(event.severity) < severityOrder.indexOf(this.config.minSeverity)) {
      return;
    }

    const auditEvent: IAuditEvent = {
      id: this.generateEventId(),
      timestamp: new Date().toISOString(),
      ...event
    };

    // Calculate hash for integrity
    if (this.config.enableIntegrityCheck) {
      auditEvent.hash = this.calculateEventHash(auditEvent);
    }

    // Add to buffer
    this.logBuffer.push(auditEvent);

    // Update statistics
    this.updateStats(auditEvent);

    // Write to log file
    await this.writeToLog(auditEvent);

    // Send alerts if enabled
    if (this.config.enableAlerting) {
      await this.sendAlert(auditEvent);
    }

    this.emit('event-logged', auditEvent);
  }

  /**
   * Log authentication attempt
   */
  async logAuthAttempt(actorId: string, success: boolean, ip?: string, userAgent?: string): Promise<void> {
    await this.log({
      type: success ? AuditEventType.LOGIN_SUCCESS : AuditEventType.LOGIN_FAILURE,
      severity: success ? AuditSeverity.LOW : AuditSeverity.MEDIUM,
      actor: {
        id: actorId,
        type: 'user',
        ip,
        userAgent
      },
      action: success ? 'authenticate' : 'authenticate_failed',
      description: success ? 'User successfully authenticated' : 'User authentication failed'
    });
  }

  /**
   * Log access attempt
   */
  async logAccess(actorId: string, targetId: string, targetType: string, granted: boolean, reason?: string): Promise<void> {
    await this.log({
      type: granted ? AuditEventType.ACCESS_GRANTED : AuditEventType.ACCESS_DENIED,
      severity: granted ? AuditSeverity.LOW : AuditSeverity.MEDIUM,
      actor: {
        id: actorId,
        type: 'user'
      },
      target: {
        id: targetId,
        type: targetType
      },
      action: granted ? 'access_granted' : 'access_denied',
      description: granted ? 'Access granted to resource' : `Access denied to resource${reason ? `: ${reason}` : ''}`,
      metadata: reason ? { reason } : undefined
    });
  }

  /**
   * Log data modification
   */
  async logDataModification(actorId: string, targetId: string, action: string, changes?: Record<string, unknown>): Promise<void> {
    await this.log({
      type: AuditEventType.DATA_MODIFIED,
      severity: AuditSeverity.MEDIUM,
      actor: {
        id: actorId,
        type: 'user'
      },
      target: {
        id: targetId,
        type: 'data'
      },
      action,
      description: `Data modified: ${action}`,
      metadata: changes
    });
  }

  /**
   * Log security violation
   */
  async logSecurityViolation(
    description: string, 
    actorId?: string, 
    severity = AuditSeverity.HIGH,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    await this.log({
      type: AuditEventType.SECURITY_VIOLATION,
      severity,
      actor: {
        id: actorId,
        type: actorId ? 'user' : 'system'
      },
      action: 'security_violation',
      description,
      metadata
    });
  }

  /**
   * Query audit events
   */
  async query(query: IAuditQuery): Promise<IAuditEvent[]> {
    try {
      // This is a simplified implementation
      // In production, you'd want to use a proper database
      const logContent = await fs.readFile(this.config.logFilePath, 'utf-8');
      const events = logContent
        .split('\n')
        .filter(line => line.trim())
        .map(line => {
          try {
            return JSON.parse(line) as IAuditEvent;
          } catch {
            return null;
          }
        })
        .filter((event): event is IAuditEvent => event !== null);

      let filteredEvents = events;

      // Apply filters
      if (query.startDate) {
        filteredEvents = filteredEvents.filter(event => 
          new Date(event.timestamp) >= query.startDate!
        );
      }

      if (query.endDate) {
        filteredEvents = filteredEvents.filter(event => 
          new Date(event.timestamp) <= query.endDate!
        );
      }

      if (query.types && query.types.length > 0) {
        filteredEvents = filteredEvents.filter(event => 
          query.types!.includes(event.type)
        );
      }

      if (query.severities && query.severities.length > 0) {
        filteredEvents = filteredEvents.filter(event => 
          query.severities!.includes(event.severity)
        );
      }

      if (query.actorId) {
        filteredEvents = filteredEvents.filter(event => 
          event.actor.id === query.actorId
        );
      }

      if (query.searchText) {
        const searchLower = query.searchText.toLowerCase();
        filteredEvents = filteredEvents.filter(event => 
          event.description.toLowerCase().includes(searchLower)
        );
      }

      // Apply pagination
      const offset = query.offset || 0;
      const limit = query.limit || 100;
      
      return filteredEvents
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(offset, offset + limit);

    } catch (error) {
      throw new Error(`Failed to query audit logs: ${error}`);
    }
  }

  /**
   * Get audit statistics
   */
  getStats(): IAuditStats {
    return { ...this.stats };
  }

  /**
   * Rotate log files
   */
  async rotateLog(): Promise<void> {
    try {
      const currentTime = Date.now();
      const logFile = this.config.logFilePath;
      const rotatedFile = `${logFile}.${currentTime}`;

      // Move current log to rotated file
      await fs.rename(logFile, rotatedFile);

      // Clean up old log files
      await this.cleanupOldLogs();

      this.lastRotation = currentTime;
      this.stats.lastRotation = new Date().toISOString();
      this.stats.currentLogSize = 0;

      this.emit('log-rotated', { rotatedFile });
    } catch (error) {
      this.emit('error', new Error(`Log rotation failed: ${error}`));
    }
  }

  /**
   * Shutdown audit logger
   */
  async shutdown(): Promise<void> {
    if (this.rotationTimer) {
      clearInterval(this.rotationTimer);
    }

    // Flush any remaining events
    if (this.logBuffer.length > 0) {
      await this.flushBuffer();
    }

    this.emit('shutdown');
  }

  /**
   * Generate unique event ID
   */
  private generateEventId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2);
    return `${timestamp}-${random}`;
  }

  /**
   * Calculate event hash for integrity
   */
  private calculateEventHash(event: Omit<IAuditEvent, 'hash'>): string {
    const content = JSON.stringify(event, Object.keys(event).sort());
    return createHash('sha256').update(content).digest('hex');
  }

  /**
   * Update statistics
   */
  private updateStats(event: IAuditEvent): void {
    this.stats.totalEvents++;
    
    this.stats.eventsByType[event.type] = (this.stats.eventsByType[event.type] || 0) + 1;
    this.stats.eventsBySeverity[event.severity] = (this.stats.eventsBySeverity[event.severity] || 0) + 1;

    // Count events in last 24 hours
    const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
    if (new Date(event.timestamp).getTime() > oneDayAgo) {
      this.stats.eventsLast24h++;
    }
  }

  /**
   * Write event to log file
   */
  private async writeToLog(event: IAuditEvent): Promise<void> {
    try {
      const logLine = JSON.stringify(event) + '\n';
      
      await fs.appendFile(this.config.logFilePath, logLine);
      this.stats.currentLogSize += logLine.length;

      // Check if rotation is needed
      if (this.stats.currentLogSize >= this.config.maxFileSize) {
        await this.rotateLog();
      }
    } catch (error) {
      this.emit('error', new Error(`Failed to write to log: ${error}`));
    }
  }

  /**
   * Send alert for critical events
   */
  private async sendAlert(event: IAuditEvent): Promise<void> {
    if (!this.config.alertConfig) {
      return;
    }

    const shouldAlert = !this.config.alertConfig.criticalEventsOnly || 
                       event.severity === AuditSeverity.CRITICAL;

    if (!shouldAlert) {
      return;
    }

    try {
      // Simplified alert implementation
      // In production, you'd integrate with actual alerting systems
      this.emit('alert', event);
    } catch (error) {
      this.emit('error', new Error(`Failed to send alert: ${error}`));
    }
  }

  /**
   * Start log rotation timer
   */
  private startRotationTimer(): void {
    this.rotationTimer = setInterval(async () => {
      await this.rotateLog();
    }, this.config.rotationInterval);
  }

  /**
   * Clean up old log files
   */
  private async cleanupOldLogs(): Promise<void> {
    try {
      const logDir = join(this.config.logFilePath, '..');
      const files = await fs.readdir(logDir);
      const logFiles = files
        .filter(file => file.startsWith(join(this.config.logFilePath, '..')))
        .sort()
        .reverse();

      if (logFiles.length > this.config.maxFiles) {
        const filesToDelete = logFiles.slice(this.config.maxFiles);
        
        for (const file of filesToDelete) {
          await fs.unlink(join(logDir, file));
        }
      }

      this.stats.logFileCount = Math.min(logFiles.length, this.config.maxFiles);
    } catch (error) {
      this.emit('error', new Error(`Log cleanup failed: ${error}`));
    }
  }

  /**
   * Flush log buffer
   */
  private async flushBuffer(): Promise<void> {
    for (const event of this.logBuffer) {
      await this.writeToLog(event);
    }
    this.logBuffer = [];
  }
}

/**
 * Factory function to create audit logger
 */
export function createAuditLogger(config: IAuditLoggerConfig): AuditLogger {
  return new AuditLogger(config);
}

/**
 * Audit logger middleware for Express.js
 */
export function auditMiddleware(logger: AuditLogger) {
  return (req: {method: string; path: string; ip: string; get: (header: string) => string; query: Record<string, unknown>; headers: Record<string, string>}, _res: unknown, next: () => void): void => {
    const startTime = Date.now();
    
    // Log API call
    logger.log({
      type: AuditEventType.API_CALL,
      severity: AuditSeverity.LOW,
      actor: {
        type: 'api',
        ip: req.ip,
        userAgent: req.get('User-Agent')
      },
      action: `${req.method} ${req.path}`,
      description: `API call: ${req.method} ${req.path}`,
      metadata: {
        method: req.method,
        path: req.path,
        query: req.query,
        startTime
      },
      correlationId: req.headers['x-correlation-id'] || req.headers['x-request-id']
    });

    next();
  };
}
