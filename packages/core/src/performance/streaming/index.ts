/**
 * @fileoverview Main export file for streaming and advanced performance systems
 * Provides comprehensive streaming, connection pooling, and lazy loading capabilities
 */

// Core streaming interfaces and types
export * from './StreamInterfaces.js';

// Stream processing implementations
export * from './StreamProcessor.js';
export * from './FileStreamProcessor.js';

// Connection pooling system
export * from './ConnectionPool.js';

// Lazy loading and pagination
export * from './LazyLoading.js';

// Re-export commonly used types and interfaces
export type {
  IStreamProcessor,
  IFileStreamProcessor,
  ITextStreamProcessor,
  IJSONStreamProcessor,
  IStreamConfig,
  IStreamResult,
  IStreamChunk,
  IStreamStats,
  StreamMode,
  StreamPriority
} from './StreamInterfaces.js';

export type {
  IConnectionPool,
  IConnectionPoolConfig,
  IConnectionPoolStats,
  RequestOptions,
  PooledResponse
} from './ConnectionPool.js';

export type {
  ILazyLoadingManager,
  IVirtualScrollingManager,
  IPaginationConfig,
  IPageData,
  DataLoader
} from './LazyLoading.js';

// Factory functions for easy instantiation
export {
  createStreamProcessor,
  createOptimizedConfig
} from './StreamProcessor.js';

export {
  createFileStreamProcessor,
  getFileProcessingRecommendations
} from './FileStreamProcessor.js';

export {
  createConnectionPool,
  CONNECTION_POOL_PRESETS
} from './ConnectionPool.js';

export {
  createLazyLoadingManager,
  createVirtualScrollingManager,
  PAGINATION_PRESETS
} from './LazyLoading.js';

// Default configurations
export { DEFAULT_STREAM_CONFIGS } from './StreamInterfaces.js';
