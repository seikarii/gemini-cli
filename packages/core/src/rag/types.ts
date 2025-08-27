/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Core interfaces and types for the RAG (Retrieval-Augmented Generation) system.
 * This module defines the foundational abstractions that enable sophisticated
 * context retrieval and augmentation for reducing LLM hallucinations.
 */

/**
 * Represents a chunk of content that can be retrieved and used for context.
 */
export interface RAGChunk {
  /** Unique identifier for the chunk */
  id: string;
  /** The actual content of the chunk */
  content: string;
  /** Type of content (code, documentation, conversation, etc.) */
  type: ChunkType;
  /** Language or format of the content */
  language?: string;
  /** Source information */
  source: ChunkSource;
  /** Metadata associated with the chunk */
  metadata: ChunkMetadata;
  /** Embedding vector for semantic search */
  embedding?: number[];
  /** Timestamp when chunk was created/indexed */
  timestamp: string;
  /** Hash of the content for change detection */
  contentHash: string;
}

/**
 * Types of content chunks supported by the RAG system.
 */
export enum ChunkType {
  CODE_FUNCTION = 'code_function',
  CODE_CLASS = 'code_class',
  CODE_MODULE = 'code_module',
  CODE_SNIPPET = 'code_snippet',
  DOCUMENTATION = 'documentation',
  CONVERSATION = 'conversation',
  COMMENT = 'comment',
  CONFIG = 'config',
  TEST = 'test',
  ERROR_MESSAGE = 'error_message',
}

/**
 * Source information for a chunk.
 */
export interface ChunkSource {
  /** File path or conversation ID */
  id: string;
  /** Type of source */
  type: 'file' | 'conversation' | 'web' | 'documentation';
  /** Repository or project identifier */
  repository?: string;
  /** Branch or version */
  version?: string;
  /** Line numbers for code chunks */
  startLine?: number;
  endLine?: number;
  /** URL for web sources */
  url?: string;
}

/**
 * Rich metadata for enhanced retrieval and context understanding.
 */
export interface ChunkMetadata {
  /** File-specific metadata */
  file?: {
    path: string;
    extension: string;
    size: number;
    lastModified: string;
  };
  
  /** Code-specific metadata */
  code?: {
    functionName?: string;
    className?: string;
    methodName?: string;
    variables?: string[];
    imports?: string[];
    exports?: string[];
    dependencies?: string[];
    complexity?: number;
    testCoverage?: number;
  };
  
  /** Git-specific metadata */
  git?: {
    author: string;
    commitHash: string;
    commitMessage: string;
    commitDate: string;
    modificationFrequency: number;
  };
  
  /** Project-specific metadata */
  project?: {
    component: string;
    layer: string; // presentation, business, data, etc.
    importance: number; // 0-1 relevance score
    usageFrequency: number;
  };
  
  /** Conversation-specific metadata */
  conversation?: {
    speaker: 'user' | 'assistant';
    turnIndex: number;
    topic?: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
    intent?: string;
  };
  
  /** Quality metrics */
  quality?: {
    readability: number; // 0-1
    complexity: number; // 0-1
    completeness: number; // 0-1
    relevance: number; // 0-1
  };
  
  /** Custom tags and annotations */
  tags?: string[];
  annotations?: Record<string, unknown>;
}

/**
 * Configuration for the RAG system.
 */
export interface RAGConfig {
  /** Whether RAG is enabled */
  enabled: boolean;
  
  /** Vector store configuration */
  vectorStore: {
    provider: 'chroma' | 'pinecone' | 'weaviate' | 'qdrant' | 'memory';
    connectionString?: string;
    apiKey?: string;
    collectionName?: string;
    persistenceDirectory?: string;
  };
  
  /** Embedding configuration */
  embedding: {
    model: string;
    dimension: number;
    batchSize: number;
    cacheSize: number;
    maxRetries: number;
  };
  
  /** Chunking strategy configuration */
  chunking: {
    strategy: 'ast' | 'semantic' | 'fixed' | 'hybrid';
    maxChunkSize: number;
    minChunkSize: number;
    overlapRatio: number;
    respectBoundaries: boolean;
  };
  
  /** Retrieval configuration */
  retrieval: {
    maxResults: number;
    similarityThreshold: number;
    hybridWeights: {
      semantic: number;
      keyword: number;
      graph: number;
      recency: number;
    };
    reRankingEnabled: boolean;
    diversityThreshold: number;
  };
  
  /** Context assembly configuration */
  context: {
    maxTokens: number;
    compressionRatio: number;
    includeDependencies: boolean;
    includeDocumentation: boolean;
    prioritizeRecent: boolean;
    preserveStructure: boolean;
  };
  
  /** Performance configuration */
  performance: {
    enableCaching: boolean;
    cacheExpiration: number; // seconds
    batchProcessing: boolean;
    parallelProcessing: boolean;
    maxConcurrentOperations: number;
  };
  
  /** Debug and monitoring */
  debug: {
    enableLogging: boolean;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
    enableMetrics: boolean;
    enableTracing: boolean;
  };
}

/**
 * Query for retrieving relevant chunks.
 */
export interface RAGQuery {
  /** The search query text */
  text: string;
  /** Type of query for specialized handling */
  type?: QueryType;
  /** Filters to apply during retrieval */
  filters?: QueryFilters;
  /** Maximum number of results to return */
  maxResults?: number;
  /** Include conversation context */
  includeContext?: boolean;
  /** Boost certain types of content */
  boosts?: Record<ChunkType, number>;
}

/**
 * Types of queries supported by the RAG system.
 */
export enum QueryType {
  CODE_GENERATION = 'code_generation',
  CODE_EXPLANATION = 'code_explanation',
  DEBUGGING = 'debugging',
  API_USAGE = 'api_usage',
  BEST_PRACTICES = 'best_practices',
  GENERAL_QUESTION = 'general_question',
  REFACTORING = 'refactoring',
  TESTING = 'testing',
}

/**
 * Filters for narrowing search results.
 */
export interface QueryFilters {
  /** Filter by content type */
  chunkTypes?: ChunkType[];
  /** Filter by programming language */
  languages?: string[];
  /** Filter by file patterns */
  filePatterns?: string[];
  /** Filter by recency */
  timeRange?: {
    start?: string;
    end?: string;
  };
  /** Filter by author */
  authors?: string[];
  /** Filter by project components */
  components?: string[];
  /** Filter by quality threshold */
  minQuality?: number;
  /** Custom metadata filters */
  metadata?: Record<string, unknown>;
}

/**
 * Result of a RAG retrieval operation.
 */
export interface RAGResult {
  /** Retrieved chunks ranked by relevance */
  chunks: ScoredChunk[];
  /** Query that produced these results */
  query: RAGQuery;
  /** Total number of chunks considered */
  totalConsidered: number;
  /** Time taken for retrieval */
  retrievalTimeMs: number;
  /** Metrics about the retrieval operation */
  metrics: RetrievalMetrics;
}

/**
 * A chunk with its relevance score and explanation.
 */
export interface ScoredChunk {
  /** The chunk itself */
  chunk: RAGChunk;
  /** Overall relevance score (0-1) */
  score: number;
  /** Breakdown of score components */
  scoreBreakdown: {
    semantic: number;
    keyword: number;
    graph: number;
    recency: number;
    quality: number;
    custom?: Record<string, number>;
  };
  /** Explanation of why this chunk was retrieved */
  explanation?: string;
  /** Highlighted text showing matches */
  highlights?: string[];
}

/**
 * Metrics about a retrieval operation.
 */
export interface RetrievalMetrics {
  /** Number of chunks retrieved from vector store */
  vectorStoreResults: number;
  /** Number of chunks from keyword search */
  keywordResults: number;
  /** Number of chunks from graph traversal */
  graphResults: number;
  /** Number of chunks after re-ranking */
  reRankedResults: number;
  /** Cache hit rates */
  cacheHits: {
    embedding: number;
    retrieval: number;
  };
  /** Processing times */
  timings: {
    queryProcessing: number;
    vectorSearch: number;
    keywordSearch: number;
    graphSearch: number;
    reRanking: number;
    total: number;
  };
}

/**
 * Enhanced context for LLM consumption.
 */
export interface RAGContext {
  /** The augmented context text */
  content: string;
  /** Source chunks used to build the context */
  sourceChunks: ScoredChunk[];
  /** Token count of the generated context */
  tokenCount: number;
  /** Metadata about context assembly */
  assembly: {
    strategy: string;
    compressionRatio: number;
    dependenciesIncluded: boolean;
    summariesGenerated: number;
  };
  /** Instructions for the LLM */
  instructions?: string;
}

/**
 * Options for context assembly.
 */
export interface ContextAssemblyOptions {
  /** Maximum tokens to use */
  maxTokens: number;
  /** Whether to include dependencies */
  includeDependencies?: boolean;
  /** Whether to include documentation */
  includeDocumentation?: boolean;
  /** Whether to compress older context */
  compressOlder?: boolean;
  /** Custom template for context formatting */
  template?: string;
  /** Priority weights for different content types */
  priorities?: Record<ChunkType, number>;
}

/**
 * Abstract base class for vector stores.
 */
export abstract class RAGVectorStore {
  abstract initialize(): Promise<void>;
  abstract addChunks(chunks: RAGChunk[]): Promise<void>;
  abstract updateChunks(chunks: RAGChunk[]): Promise<void>;
  abstract deleteChunks(chunkIds: string[]): Promise<void>;
  abstract search(
    query: string,
    embedding: number[],
    filters?: QueryFilters,
    limit?: number
  ): Promise<ScoredChunk[]>;
  abstract getChunk(id: string): Promise<RAGChunk | null>;
  abstract listChunks(filters?: QueryFilters): Promise<string[]>;
  abstract getStats(): Promise<VectorStoreStats>;
  abstract close(): Promise<void>;
}

/**
 * Statistics about the vector store.
 */
export interface VectorStoreStats {
  totalChunks: number;
  totalSize: number;
  indexSize: number;
  lastUpdated: string;
  performance: {
    averageQueryTime: number;
    cacheHitRate: number;
  };
}

/**
 * Abstract base class for embedding services.
 */
export abstract class RAGEmbeddingService {
  abstract generateEmbedding(text: string): Promise<number[]>;
  abstract generateEmbeddings(texts: string[]): Promise<number[][]>;
  abstract getEmbeddingDimension(): number;
  abstract getModelInfo(): EmbeddingModelInfo;
}

/**
 * Information about the embedding model.
 */
export interface EmbeddingModelInfo {
  name: string;
  dimension: number;
  maxTokens: number;
  isCodeSpecific: boolean;
  provider: string;
}

/**
 * Error types for the RAG system.
 */
export class RAGError extends Error {
  readonly code: string;
  readonly cause?: Error;

  constructor(
    message: string,
    code: string,
    cause?: Error
  ) {
    super(message);
    this.name = 'RAGError';
    this.code = code;
    this.cause = cause;
  }
}

export class RAGConfigurationError extends RAGError {
  constructor(message: string, cause?: Error) {
    super(message, 'RAG_CONFIGURATION_ERROR', cause);
  }
}

export class RAGVectorStoreError extends RAGError {
  constructor(message: string, cause?: Error) {
    super(message, 'RAG_VECTOR_STORE_ERROR', cause);
  }
}

export class RAGEmbeddingError extends RAGError {
  constructor(message: string, cause?: Error) {
    super(message, 'RAG_EMBEDDING_ERROR', cause);
  }
}

export class RAGRetrievalError extends RAGError {
  constructor(message: string, cause?: Error) {
    super(message, 'RAG_RETRIEVAL_ERROR', cause);
  }
}
