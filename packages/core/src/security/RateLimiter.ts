/**
 * @fileoverview Rate limiting system for API calls and resource usage control
 * Provides multiple rate limiting algorithms and adaptive throttling mechanisms
 */

import { EventEmitter } from 'events';

/**
 * Rate limiting algorithm types
 */
export enum RateLimitAlgorithm {
  /** Token bucket algorithm - allows bursts up to bucket capacity */
  TOKEN_BUCKET = 'token_bucket',
  /** Fixed window - resets counter at fixed intervals */
  FIXED_WINDOW = 'fixed_window',
  /** Sliding window log - maintains exact request history */
  SLIDING_WINDOW_LOG = 'sliding_window_log',
  /** Sliding window counter - approximates sliding window with less memory */
  SLIDING_WINDOW_COUNTER = 'sliding_window_counter',
  /** Adaptive rate limiting - adjusts based on system load */
  ADAPTIVE = 'adaptive'
}

/**
 * Rate limit configuration
 */
export interface IRateLimitConfig {
  /** Rate limiting algorithm to use */
  algorithm: RateLimitAlgorithm;
  /** Maximum requests per window */
  maxRequests: number;
  /** Time window in milliseconds */
  windowMs: number;
  /** Maximum burst size (for token bucket) */
  burstSize?: number;
  /** Token refill rate per second (for token bucket) */
  refillRate?: number;
  /** Enable adaptive rate limiting */
  enableAdaptive: boolean;
  /** Minimum delay between requests in milliseconds */
  minDelay: number;
  /** Maximum delay between requests in milliseconds */
  maxDelay: number;
  /** Skip successful requests in rate limiting (count only errors) */
  skipSuccessfulRequests: boolean;
  /** Skip failed requests in rate limiting */
  skipFailedRequests: boolean;
  /** Custom key generator function */
  keyGenerator?: (identifier: string) => string;
  /** Custom cost function for different operations */
  costFunction?: (operation: string) => number;
}

/**
 * Rate limit result
 */
export interface IRateLimitResult {
  /** Whether the request is allowed */
  allowed: boolean;
  /** Current request count in window */
  currentRequests: number;
  /** Maximum requests allowed */
  maxRequests: number;
  /** Time until reset in milliseconds */
  resetTime: number;
  /** Suggested retry delay in milliseconds */
  retryDelay?: number;
  /** Rate limit key used */
  key: string;
  /** Cost of the operation */
  cost: number;
}

/**
 * Rate limit statistics
 */
export interface IRateLimitStats {
  /** Total requests processed */
  totalRequests: number;
  /** Total requests allowed */
  allowedRequests: number;
  /** Total requests blocked */
  blockedRequests: number;
  /** Current active rate limits */
  activeRateLimits: number;
  /** Average requests per second */
  avgRequestsPerSecond: number;
  /** Current system load (0-1) */
  systemLoad: number;
  /** Memory usage for rate limiting */
  memoryUsage: number;
}

/**
 * Token bucket rate limiter implementation
 */
class TokenBucketRateLimiter {
  tokens: number;
  private lastRefill: number;
  private readonly maxTokens: number;
  private readonly refillRate: number;

  constructor(maxTokens: number, refillRate: number) {
    this.maxTokens = maxTokens;
    this.refillRate = refillRate;
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  /**
   * Try to consume tokens
   */
  consume(cost: number): boolean {
    this.refill();
    
    if (this.tokens >= cost) {
      this.tokens -= cost;
      return true;
    }
    
    return false;
  }

  /**
   * Get next available time for request
   */
  getNextAvailableTime(cost: number): number {
    this.refill();
    
    if (this.tokens >= cost) {
      return 0;
    }
    
    const tokensNeeded = cost - this.tokens;
    return (tokensNeeded / this.refillRate) * 1000; // Convert to milliseconds
  }

  /**
   * Refill tokens based on elapsed time
   */
  private refill(): void {
    const now = Date.now();
    const timePassed = (now - this.lastRefill) / 1000; // Convert to seconds
    const tokensToAdd = timePassed * this.refillRate;
    
    this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}

/**
 * Advanced rate limiter with multiple algorithms
 */
export class RateLimiter extends EventEmitter {
  private config: IRateLimitConfig;
  private tokenBuckets = new Map<string, TokenBucketRateLimiter>();
  private requestCounts = new Map<string, number>();
  private requestHistory = new Map<string, Array<{timestamp: number; cost: number}>>();
  private lastCleanup = 0;
  private systemLoadSamples: number[] = [];
  private startTime = Date.now();
  private stats: IRateLimitStats = {
    totalRequests: 0,
    allowedRequests: 0,
    blockedRequests: 0,
    activeRateLimits: 0,
    avgRequestsPerSecond: 0,
    systemLoad: 0,
    memoryUsage: 0
  };

  constructor(config: IRateLimitConfig) {
    super();
    this.config = {
      burstSize: config.burstSize || config.maxRequests,
      refillRate: config.refillRate || 1,
      ...config
    };
  }

  /**
   * Check if request is allowed
   */
  async checkLimit(identifier: string, cost = 1): Promise<IRateLimitResult> {
    const key = this.config.keyGenerator ? this.config.keyGenerator(identifier) : identifier;
    const operationCost = this.config.costFunction ? this.config.costFunction(identifier) * cost : cost;
    
    this.stats.totalRequests++;
    
    let result: IRateLimitResult;
    
    switch (this.config.algorithm) {
      case RateLimitAlgorithm.TOKEN_BUCKET:
        result = await this.checkTokenBucket(key, operationCost);
        break;
      case RateLimitAlgorithm.FIXED_WINDOW:
        result = await this.checkFixedWindow(key, operationCost);
        break;
      case RateLimitAlgorithm.SLIDING_WINDOW_LOG:
        result = await this.checkSlidingWindowLog(key, operationCost);
        break;
      default:
        result = await this.checkTokenBucket(key, operationCost);
    }

    if (result.allowed) {
      this.stats.allowedRequests++;
    } else {
      this.stats.blockedRequests++;
    }

    this.updateStats();
    this.emit('rate-limit-check', result);
    return result;
  }

  /**
   * Reset rate limit for specific identifier
   */
  reset(identifier: string): void {
    const key = this.config.keyGenerator ? this.config.keyGenerator(identifier) : identifier;
    this.tokenBuckets.delete(key);
    this.requestCounts.delete(key);
    this.requestHistory.delete(key);
    this.emit('rate-limit-reset', { key });
  }

  /**
   * Get current statistics
   */
  getStats(): IRateLimitStats {
    this.updateStats();
    return { ...this.stats };
  }

  /**
   * Clean up expired entries
   */
  cleanup(): void {
    const now = Date.now();
    
    if (now - this.lastCleanup < 60000) {
      return;
    }

    const expiredKeys: string[] = [];
    
    // Clean up request history
    for (const [key, history] of this.requestHistory.entries()) {
      const validHistory = history.filter(entry => 
        now - entry.timestamp < this.config.windowMs
      );
      
      if (validHistory.length === 0) {
        expiredKeys.push(key);
      } else {
        this.requestHistory.set(key, validHistory);
      }
    }

    // Clean up expired keys
    for (const key of expiredKeys) {
      this.tokenBuckets.delete(key);
      this.requestCounts.delete(key);
      this.requestHistory.delete(key);
    }

    this.lastCleanup = now;
    this.emit('cleanup-completed', { removedKeys: expiredKeys.length });
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<IRateLimitConfig>): void {
    this.config = { ...this.config, ...config };
    this.emit('config-updated', this.config);
  }

  /**
   * Token bucket algorithm implementation
   */
  private async checkTokenBucket(key: string, cost: number): Promise<IRateLimitResult> {
    let bucket = this.tokenBuckets.get(key);
    
    if (!bucket) {
      bucket = new TokenBucketRateLimiter(
        this.config.burstSize || this.config.maxRequests,
        this.config.refillRate || 1
      );
      this.tokenBuckets.set(key, bucket);
    }

    const allowed = bucket.consume(cost);
    const retryDelay = allowed ? undefined : bucket.getNextAvailableTime(cost);

    return {
      allowed,
      currentRequests: (this.config.burstSize || this.config.maxRequests) - bucket.tokens,
      maxRequests: this.config.maxRequests,
      resetTime: Date.now() + (this.config.windowMs || 60000),
      retryDelay,
      key,
      cost
    };
  }

  /**
   * Fixed window algorithm implementation
   */
  private async checkFixedWindow(key: string, cost: number): Promise<IRateLimitResult> {
    const now = Date.now();
    const windowStart = Math.floor(now / this.config.windowMs) * this.config.windowMs;
    const windowKey = `${key}:${windowStart}`;
    
    const count = this.requestCounts.get(windowKey) || 0;
    const allowed = count + cost <= this.config.maxRequests;
    
    if (allowed) {
      this.requestCounts.set(windowKey, count + cost);
    }

    return {
      allowed,
      currentRequests: count,
      maxRequests: this.config.maxRequests,
      resetTime: windowStart + this.config.windowMs,
      retryDelay: allowed ? undefined : (windowStart + this.config.windowMs - now),
      key,
      cost
    };
  }

  /**
   * Sliding window log algorithm implementation
   */
  private async checkSlidingWindowLog(key: string, cost: number): Promise<IRateLimitResult> {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;
    
    let history = this.requestHistory.get(key) || [];
    
    // Remove old entries
    history = history.filter(entry => entry.timestamp > windowStart);
    
    // Calculate current usage
    const currentUsage = history.reduce((sum, entry) => sum + entry.cost, 0);
    const allowed = currentUsage + cost <= this.config.maxRequests;
    
    if (allowed) {
      history.push({ timestamp: now, cost });
      this.requestHistory.set(key, history);
    }

    return {
      allowed,
      currentRequests: currentUsage,
      maxRequests: this.config.maxRequests,
      resetTime: history.length > 0 ? history[0].timestamp + this.config.windowMs : now + this.config.windowMs,
      retryDelay: allowed ? undefined : this.config.minDelay,
      key,
      cost
    };
  }

  /**
   * Calculate current system load
   */
  private calculateSystemLoad(): number {
    const memoryUsage = process.memoryUsage();
    const memoryLoad = memoryUsage.heapUsed / memoryUsage.heapTotal;
    
    // Simple load calculation based on memory and request rate
    const requestRate = this.stats.totalRequests / Math.max(1, (Date.now() - this.startTime) / 1000);
    const requestLoad = Math.min(1, requestRate / (this.config.maxRequests * 10));
    
    const currentLoad = Math.max(memoryLoad, requestLoad);
    
    // Keep sliding window of load samples
    this.systemLoadSamples.push(currentLoad);
    if (this.systemLoadSamples.length > 10) {
      this.systemLoadSamples.shift();
    }
    
    // Return average load
    return this.systemLoadSamples.reduce((sum, load) => sum + load, 0) / this.systemLoadSamples.length;
  }

  /**
   * Update statistics
   */
  private updateStats(): void {
    const now = Date.now();
    const elapsedSeconds = Math.max(1, (now - this.startTime) / 1000);
    
    this.stats.activeRateLimits = this.tokenBuckets.size + this.requestCounts.size + this.requestHistory.size;
    this.stats.avgRequestsPerSecond = this.stats.totalRequests / elapsedSeconds;
    this.stats.systemLoad = this.calculateSystemLoad();
    this.stats.memoryUsage = process.memoryUsage().heapUsed;
  }
}

/**
 * Factory function to create rate limiter
 */
export function createRateLimiter(config: IRateLimitConfig): RateLimiter {
  return new RateLimiter(config);
}

/**
 * Predefined rate limit configurations
 */
export const RATE_LIMIT_PRESETS: Record<string, IRateLimitConfig> = {
  STRICT: {
    algorithm: RateLimitAlgorithm.TOKEN_BUCKET,
    maxRequests: 10,
    windowMs: 60000,
    burstSize: 2,
    refillRate: 0.2,
    enableAdaptive: false,
    minDelay: 1000,
    maxDelay: 30000,
    skipSuccessfulRequests: false,
    skipFailedRequests: false
  },
  MODERATE: {
    algorithm: RateLimitAlgorithm.SLIDING_WINDOW_LOG,
    maxRequests: 100,
    windowMs: 60000,
    burstSize: 20,
    refillRate: 2,
    enableAdaptive: true,
    minDelay: 100,
    maxDelay: 10000,
    skipSuccessfulRequests: false,
    skipFailedRequests: false
  },
  LENIENT: {
    algorithm: RateLimitAlgorithm.FIXED_WINDOW,
    maxRequests: 1000,
    windowMs: 60000,
    burstSize: 100,
    refillRate: 20,
    enableAdaptive: true,
    minDelay: 10,
    maxDelay: 5000,
    skipSuccessfulRequests: true,
    skipFailedRequests: false
  }
};
