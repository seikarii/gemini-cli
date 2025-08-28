/**
 * @fileoverview Connection pooling system for efficient HTTP/API request management
 * with adaptive sizing, health monitoring, and intelligent request routing
 */

import { EventEmitter } from 'events';
import { Agent } from 'http';
import { Agent as HttpsAgent } from 'https';
import * as http from 'http';
import * as https from 'https';

/**
 * Connection pool configuration options
 */
export interface IConnectionPoolConfig {
  /** Minimum number of connections to maintain */
  minConnections: number;
  /** Maximum number of connections allowed */
  maxConnections: number;
  /** Maximum time a connection can be idle before being closed (ms) */
  maxIdleTime: number;
  /** Maximum time to wait for a connection (ms) */
  connectionTimeout: number;
  /** Maximum time for a request to complete (ms) */
  requestTimeout: number;
  /** Enable keep-alive on connections */
  keepAlive: boolean;
  /** Keep-alive initial delay (ms) */
  keepAliveInitialDelay: number;
  /** Maximum number of sockets per host */
  maxSocketsPerHost: number;
  /** Maximum number of free sockets per host */
  maxFreeSocketsPerHost: number;
  /** Enable automatic retry on connection failures */
  enableRetry: boolean;
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Base retry delay in milliseconds */
  retryDelay: number;
  /** Enable connection health monitoring */
  enableHealthCheck: boolean;
  /** Health check interval in milliseconds */
  healthCheckInterval: number;
}

/**
 * Connection pool statistics
 */
export interface IConnectionPoolStats {
  /** Total number of active connections */
  activeConnections: number;
  /** Total number of idle connections */
  idleConnections: number;
  /** Total number of pending requests */
  pendingRequests: number;
  /** Average response time in milliseconds */
  averageResponseTime: number;
  /** Total requests processed */
  totalRequests: number;
  /** Total connection errors */
  connectionErrors: number;
  /** Total timeouts */
  timeouts: number;
  /** Pool efficiency (successful requests / total requests) */
  efficiency: number;
  /** Current pool size */
  poolSize: number;
  /** Maximum pool size reached */
  maxPoolSizeReached: number;
}

/**
 * HTTP connection pool for managing persistent connections
 */
export interface IConnectionPool extends EventEmitter {
  /**
   * Initialize the connection pool
   */
  initialize(): Promise<void>;

  /**
   * Make an HTTP request using the pool
   * @param url Target URL
   * @param options Request options
   * @returns Promise resolving to response
   */
  request(url: string, options?: RequestOptions): Promise<PooledResponse>;

  /**
   * Get current pool statistics
   * @returns Pool statistics
   */
  getStats(): IConnectionPoolStats;

  /**
   * Perform health check on all connections
   * @returns Health check results
   */
  healthCheck(): Promise<HealthCheckResult[]>;

  /**
   * Gracefully shutdown the pool
   * @param force Whether to force immediate shutdown
   */
  shutdown(force?: boolean): Promise<void>;

  /**
   * Scale the pool size based on current load
   * @param targetSize Desired pool size
   */
  scale(targetSize: number): Promise<void>;
}

/**
 * Request options for pooled requests
 */
export interface RequestOptions {
  /** HTTP method */
  method?: string;
  /** Request headers */
  headers?: Record<string, string>;
  /** Request body */
  body?: string | Buffer;
  /** Request timeout override */
  timeout?: number;
  /** Maximum retry attempts override */
  maxRetries?: number;
  /** Enable/disable connection reuse for this request */
  reuse?: boolean;
}

/**
 * Pooled response object
 */
export interface PooledResponse {
  /** HTTP status code */
  status: number;
  /** Response headers */
  headers: Record<string, string>;
  /** Response body */
  body: string | Buffer;
  /** Response time in milliseconds */
  responseTime: number;
  /** Whether the response came from a reused connection */
  fromCache: boolean;
  /** Connection metadata */
  connectionInfo: {
    id: string;
    age: number;
    requestCount: number;
  };
}

/**
 * Health check result for a connection
 */
export interface HealthCheckResult {
  /** Connection identifier */
  connectionId: string;
  /** Whether the connection is healthy */
  healthy: boolean;
  /** Response time for health check */
  responseTime: number;
  /** Error message if unhealthy */
  error?: string;
  /** Last successful request timestamp */
  lastSuccess: number;
}

/**
 * Connection metadata
 */
interface ConnectionMetadata {
  id: string;
  agent: Agent | HttpsAgent;
  created: number;
  lastUsed: number;
  requestCount: number;
  errorCount: number;
  isHealthy: boolean;
  host: string;
  port: number;
  protocol: 'http' | 'https';
}

/**
 * Advanced HTTP connection pool implementation
 */
export class HTTPConnectionPool extends EventEmitter implements IConnectionPool {
  private config: IConnectionPoolConfig;
  private connections = new Map<string, ConnectionMetadata>();
  private stats: IConnectionPoolStats;
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private isInitialized = false;
  private pendingRequests = new Map<string, number>();

  constructor(config: Partial<IConnectionPoolConfig> = {}) {
    super();
    this.config = {
      minConnections: 5,
      maxConnections: 50,
      maxIdleTime: 30000, // 30 seconds
      connectionTimeout: 10000, // 10 seconds
      requestTimeout: 30000, // 30 seconds
      keepAlive: true,
      keepAliveInitialDelay: 1000,
      maxSocketsPerHost: 10,
      maxFreeSocketsPerHost: 5,
      enableRetry: true,
      maxRetries: 3,
      retryDelay: 1000,
      enableHealthCheck: true,
      healthCheckInterval: 60000, // 1 minute
      ...config
    };

    this.stats = this.createInitialStats();
  }

  /**
   * Initialize the connection pool
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    // Create minimum number of connections
    for (let i = 0; i < this.config.minConnections; i++) {
      await this.createConnection('default', 'http', 'localhost', 80);
    }

    // Start health check timer if enabled
    if (this.config.enableHealthCheck) {
      this.startHealthCheck();
    }

    this.isInitialized = true;
    this.emit('initialized');
  }

  /**
   * Make an HTTP request using the pool
   */
  async request(url: string, options: RequestOptions = {}): Promise<PooledResponse> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    const urlObj = new URL(url);
    const connectionKey = this.getConnectionKey(urlObj);

    // Update pending requests
    this.pendingRequests.set(connectionKey, (this.pendingRequests.get(connectionKey) || 0) + 1);
    this.stats.pendingRequests = Array.from(this.pendingRequests.values()).reduce((a, b) => a + b, 0);

    try {
      const connection = await this.getConnection(urlObj);
      const response = await this.executeRequest(connection, url, options);
      
      const responseTime = Date.now() - startTime;
      this.updateSuccessStats(responseTime);
      
      return {
        ...response,
        responseTime,
        connectionInfo: {
          id: connection.id,
          age: Date.now() - connection.created,
          requestCount: connection.requestCount
        }
      };
    } catch (error) {
      this.updateErrorStats();
      throw error;
    } finally {
      // Update pending requests
      const pending = this.pendingRequests.get(connectionKey) || 1;
      if (pending <= 1) {
        this.pendingRequests.delete(connectionKey);
      } else {
        this.pendingRequests.set(connectionKey, pending - 1);
      }
      this.stats.pendingRequests = Array.from(this.pendingRequests.values()).reduce((a, b) => a + b, 0);
    }
  }

  /**
   * Get current pool statistics
   */
  getStats(): IConnectionPoolStats {
    this.updateRealTimeStats();
    return { ...this.stats };
  }

  /**
   * Perform health check on all connections
   */
  async healthCheck(): Promise<HealthCheckResult[]> {
    const results: HealthCheckResult[] = [];

    for (const connection of this.connections.values()) {
      const startTime = Date.now();
      
      try {
        // Perform a simple HEAD request to check connection health
        const testUrl = `${connection.protocol}://${connection.host}:${connection.port}/`;
        await this.executeRequest(connection, testUrl, { method: 'HEAD', timeout: 5000 });
        
        results.push({
          connectionId: connection.id,
          healthy: true,
          responseTime: Date.now() - startTime,
          lastSuccess: connection.lastUsed
        });
        
        connection.isHealthy = true;
      } catch (error) {
        results.push({
          connectionId: connection.id,
          healthy: false,
          responseTime: Date.now() - startTime,
          error: error instanceof Error ? error.message : String(error),
          lastSuccess: connection.lastUsed
        });
        
        connection.isHealthy = false;
        connection.errorCount++;
      }
    }

    this.emit('health-check-completed', results);
    return results;
  }

  /**
   * Gracefully shutdown the pool
   */
  async shutdown(force = false): Promise<void> {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }

    if (force) {
      // Immediately destroy all connections
      for (const connection of this.connections.values()) {
        connection.agent.destroy();
      }
      this.connections.clear();
    } else {
      // Wait for pending requests to complete
      while (this.stats.pendingRequests > 0) {
        await this.delay(100);
      }

      // Gracefully close all connections
      for (const connection of this.connections.values()) {
        connection.agent.destroy();
      }
      this.connections.clear();
    }

    this.isInitialized = false;
    this.emit('shutdown');
  }

  /**
   * Scale the pool size based on current load
   */
  async scale(targetSize: number): Promise<void> {
    const currentSize = this.connections.size;
    
    if (targetSize > currentSize) {
      // Scale up
      const connectionsToAdd = Math.min(
        targetSize - currentSize,
        this.config.maxConnections - currentSize
      );
      
      for (let i = 0; i < connectionsToAdd; i++) {
        await this.createConnection('scale-up', 'http', 'localhost', 80);
      }
    } else if (targetSize < currentSize) {
      // Scale down
      const connectionsToRemove = currentSize - targetSize;
      const connections = Array.from(this.connections.values())
        .sort((a, b) => a.lastUsed - b.lastUsed); // Remove least recently used first
      
      for (let i = 0; i < connectionsToRemove && i < connections.length; i++) {
        const connection = connections[i];
        connection.agent.destroy();
        this.connections.delete(connection.id);
      }
    }

    this.emit('scaled', { from: currentSize, to: this.connections.size });
  }

  /**
   * Get or create a connection for the given URL
   */
  private async getConnection(url: URL): Promise<ConnectionMetadata> {
    const connectionKey = this.getConnectionKey(url);
    const existingConnection = this.findAvailableConnection(connectionKey);

    if (existingConnection && this.isConnectionUsable(existingConnection)) {
      existingConnection.lastUsed = Date.now();
      existingConnection.requestCount++;
      return existingConnection;
    }

    // Create new connection if under limit
    if (this.connections.size < this.config.maxConnections) {
      return await this.createConnection(
        connectionKey,
        url.protocol.slice(0, -1) as 'http' | 'https',
        url.hostname,
        url.port ? parseInt(url.port, 10) : (url.protocol === 'https:' ? 443 : 80)
      );
    }

    // Wait for an available connection
    return await this.waitForAvailableConnection(connectionKey);
  }

  /**
   * Create a new connection
   */
  private async createConnection(
    key: string,
    protocol: 'http' | 'https',
    host: string,
    port: number
  ): Promise<ConnectionMetadata> {
    const id = `${key}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const agentOptions = {
      keepAlive: this.config.keepAlive,
      keepAliveMsecs: this.config.keepAliveInitialDelay,
      maxSockets: this.config.maxSocketsPerHost,
      maxFreeSockets: this.config.maxFreeSocketsPerHost,
      timeout: this.config.connectionTimeout,
    };

    const agent = protocol === 'https' 
      ? new HttpsAgent(agentOptions)
      : new Agent(agentOptions);

    const connection: ConnectionMetadata = {
      id,
      agent,
      created: Date.now(),
      lastUsed: Date.now(),
      requestCount: 0,
      errorCount: 0,
      isHealthy: true,
      host,
      port,
      protocol
    };

    this.connections.set(id, connection);
    this.updatePoolSizeStats();
    
    this.emit('connection-created', connection);
    return connection;
  }

  /**
   * Execute HTTP request using the connection
   */
  private async executeRequest(
    connection: ConnectionMetadata,
    url: string,
    options: RequestOptions
  ): Promise<Omit<PooledResponse, 'responseTime' | 'connectionInfo'>> {
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const requestOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        agent: connection.agent,
        timeout: options.timeout || this.config.requestTimeout
      };

      const httpModule = urlObj.protocol === 'https:' ? https : http;
      const request = httpModule.request(requestOptions, (response: http.IncomingMessage) => {
          const chunks: Buffer[] = [];
          
          response.on('data', (chunk: Buffer) => {
            chunks.push(chunk);
          });

          response.on('end', () => {
            const body = Buffer.concat(chunks);
            
            // Convert headers to Record<string, string>
            const headers: Record<string, string> = {};
            for (const [key, value] of Object.entries(response.headers)) {
              if (typeof value === 'string') {
                headers[key] = value;
              } else if (Array.isArray(value)) {
                headers[key] = value.join(', ');
              }
            }
            
            resolve({
              status: response.statusCode || 500,
              headers,
              body,
              fromCache: false
            });
          });
        });

      request.on('error', (error: Error) => {
        connection.errorCount++;
        reject(error);
      });

      request.on('timeout', () => {
        request.destroy();
        reject(new Error('Request timeout'));
      });

      if (options.body) {
        request.write(options.body);
      }

      request.end();
    });
  }

  /**
   * Find an available connection for the given key
   */
  private findAvailableConnection(key: string): ConnectionMetadata | undefined {
    for (const connection of this.connections.values()) {
      if (this.getConnectionKey(new URL(`${connection.protocol}://${connection.host}:${connection.port}`)) === key &&
          this.isConnectionUsable(connection)) {
        return connection;
      }
    }
    return undefined;
  }

  /**
   * Check if a connection is still usable
   */
  private isConnectionUsable(connection: ConnectionMetadata): boolean {
    const now = Date.now();
    const idleTime = now - connection.lastUsed;
    
    return connection.isHealthy &&
           idleTime < this.config.maxIdleTime &&
           connection.errorCount < 3; // Max 3 errors before considering unusable
  }

  /**
   * Wait for an available connection
   */
  private async waitForAvailableConnection(key: string): Promise<ConnectionMetadata> {
    const timeout = this.config.connectionTimeout;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const connection = this.findAvailableConnection(key);
      if (connection) {
        return connection;
      }
      await this.delay(50); // Wait 50ms before retrying
    }

    throw new Error('Connection timeout: No available connections');
  }

  /**
   * Get connection key for grouping connections
   */
  private getConnectionKey(url: URL): string {
    return `${url.protocol}//${url.hostname}:${url.port || (url.protocol === 'https:' ? 443 : 80)}`;
  }

  /**
   * Start health check timer
   */
  private startHealthCheck(): void {
    this.healthCheckTimer = setInterval(async () => {
      try {
        await this.healthCheck();
        await this.cleanupUnhealthyConnections();
      } catch (error) {
        this.emit('health-check-error', error);
      }
    }, this.config.healthCheckInterval);
  }

  /**
   * Clean up unhealthy connections
   */
  private async cleanupUnhealthyConnections(): Promise<void> {
    const connectionsToRemove: string[] = [];

    for (const [id, connection] of this.connections.entries()) {
      if (!this.isConnectionUsable(connection)) {
        connectionsToRemove.push(id);
      }
    }

    for (const id of connectionsToRemove) {
      const connection = this.connections.get(id);
      if (connection) {
        connection.agent.destroy();
        this.connections.delete(id);
        this.emit('connection-removed', { id, reason: 'unhealthy' });
      }
    }

    // Ensure minimum connections
    if (this.connections.size < this.config.minConnections) {
      const connectionsToAdd = this.config.minConnections - this.connections.size;
      for (let i = 0; i < connectionsToAdd; i++) {
        await this.createConnection('maintenance', 'http', 'localhost', 80);
      }
    }
  }

  /**
   * Update statistics for successful requests
   */
  private updateSuccessStats(responseTime: number): void {
    this.stats.totalRequests++;
    
    // Update average response time using running average
    const total = this.stats.averageResponseTime * (this.stats.totalRequests - 1);
    this.stats.averageResponseTime = (total + responseTime) / this.stats.totalRequests;
    
    this.stats.efficiency = this.stats.totalRequests / 
      (this.stats.totalRequests + this.stats.connectionErrors + this.stats.timeouts);
  }

  /**
   * Update statistics for failed requests
   */
  private updateErrorStats(): void {
    this.stats.connectionErrors++;
    this.stats.efficiency = this.stats.totalRequests / 
      (this.stats.totalRequests + this.stats.connectionErrors + this.stats.timeouts);
  }

  /**
   * Update real-time statistics
   */
  private updateRealTimeStats(): void {
    this.stats.activeConnections = Array.from(this.connections.values())
      .filter(c => Date.now() - c.lastUsed < 5000).length; // Active in last 5 seconds
    
    this.stats.idleConnections = this.connections.size - this.stats.activeConnections;
    this.stats.poolSize = this.connections.size;
  }

  /**
   * Update pool size statistics
   */
  private updatePoolSizeStats(): void {
    if (this.connections.size > this.stats.maxPoolSizeReached) {
      this.stats.maxPoolSizeReached = this.connections.size;
    }
  }

  /**
   * Create initial statistics object
   */
  private createInitialStats(): IConnectionPoolStats {
    return {
      activeConnections: 0,
      idleConnections: 0,
      pendingRequests: 0,
      averageResponseTime: 0,
      totalRequests: 0,
      connectionErrors: 0,
      timeouts: 0,
      efficiency: 1.0,
      poolSize: 0,
      maxPoolSizeReached: 0
    };
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Create HTTP connection pool with optimized configuration
 */
export function createConnectionPool(config?: Partial<IConnectionPoolConfig>): HTTPConnectionPool {
  return new HTTPConnectionPool(config);
}

/**
 * Default connection pool configurations for different use cases
 */
export const CONNECTION_POOL_PRESETS = {
  /** High-performance configuration for heavy API usage */
  HIGH_PERFORMANCE: {
    minConnections: 10,
    maxConnections: 100,
    maxIdleTime: 60000, // 1 minute
    connectionTimeout: 5000, // 5 seconds
    requestTimeout: 30000, // 30 seconds
    keepAlive: true,
    maxSocketsPerHost: 20,
    maxFreeSocketsPerHost: 10,
    enableRetry: true,
    maxRetries: 3,
    enableHealthCheck: true,
    healthCheckInterval: 30000 // 30 seconds
  },

  /** Memory-efficient configuration for limited resources */
  MEMORY_EFFICIENT: {
    minConnections: 2,
    maxConnections: 10,
    maxIdleTime: 15000, // 15 seconds
    connectionTimeout: 10000, // 10 seconds
    requestTimeout: 20000, // 20 seconds
    keepAlive: true,
    maxSocketsPerHost: 3,
    maxFreeSocketsPerHost: 1,
    enableRetry: true,
    maxRetries: 2,
    enableHealthCheck: true,
    healthCheckInterval: 120000 // 2 minutes
  },

  /** Balanced configuration for general use */
  BALANCED: {
    minConnections: 5,
    maxConnections: 25,
    maxIdleTime: 30000, // 30 seconds
    connectionTimeout: 8000, // 8 seconds
    requestTimeout: 25000, // 25 seconds
    keepAlive: true,
    maxSocketsPerHost: 8,
    maxFreeSocketsPerHost: 3,
    enableRetry: true,
    maxRetries: 3,
    enableHealthCheck: true,
    healthCheckInterval: 60000 // 1 minute
  }
} satisfies Record<string, Partial<IConnectionPoolConfig>>;
