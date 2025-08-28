/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';
import { RAGLogger } from './logger.js';
import {
  RAGConfig,
  RAGQuery,
  RAGChunk,
  RAGVectorStore,
  RAGEmbeddingService,
  IndexingSource,
  IndexingResult,
  QueryEnhancementOptions,
  EnhancedQueryResult,
  RAGMetrics,
  VectorStoreConfig,
} from './types.js';
import { RAGMemoryVectorStore } from './vectorStores/memoryVectorStore.js';
import { RAGGeminiEmbeddingService } from './embeddingServices/geminiEmbeddingService.js';
import { RAGASTChunkingService } from './chunking/astChunkingService.js';

/**
 * Main orchestration service for RAG operations.
 * Coordinates between chunking, embedding, vector storage, and retrieval components.
 */
export class RAGService {
  private vectorStore!: RAGVectorStore;
  private embeddingService!: RAGEmbeddingService;
  private chunkingService!: RAGASTChunkingService;
  private config!: RAGConfig;
  private metrics: RAGMetrics;
  private isInitialized = false;

  constructor(
    private readonly userConfig: Config,
    private readonly logger: RAGLogger,
  ) {
    this.metrics = {
      totalQueries: 0,
      totalChunksIndexed: 0,
      averageRetrievalTime: 0,
      cacheHitRate: 0,
    };
  }

  /**
   * Initialize the RAG system with configuration and services.
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      this.logger.info('RAG service already initialized');
      return;
    }

    this.logger.info('Initializing RAG service...');

    try {
      // Load configuration
      this.config = this.loadRAGConfig();

      // Initialize services
      this.embeddingService = new RAGGeminiEmbeddingService(
        this.userConfig,
        this.config,
        this.logger,
      );

      this.chunkingService = new RAGASTChunkingService(
        this.config,
        this.logger,
      );

      // Use memory vector store as default
      const memoryConfig: VectorStoreConfig = {
        provider: 'memory',
        collection: 'default',
      };

      this.vectorStore = new RAGMemoryVectorStore(
        memoryConfig,
        this.logger,
        this.config.retrieval.hybridWeights,
      );

      await this.vectorStore.initialize();

      this.isInitialized = true;
      this.logger.info('RAG service initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize RAG service', error);
      throw error;
    }
  }

  /**
   * Index content from multiple sources.
   */
  async indexContent(sources: IndexingSource[]): Promise<IndexingResult> {
    this.ensureInitialized();

    this.logger.info(`Starting to index ${sources.length} sources`);

    const result: IndexingResult = {
      totalChunks: 0,
      successfulSources: 0,
      errors: [],
    };

    for (const source of sources) {
      try {
        const chunksCreated = await this.indexSingleSource(source);
        result.totalChunks += chunksCreated;
        result.successfulSources++;
      } catch (error) {
        this.logger.error(`Failed to index source ${source.id}`, error);
        result.errors.push({
          sourceId: source.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    // Update metrics
    this.metrics.totalChunksIndexed += result.totalChunks;

    this.logger.info(
      `Indexing completed: ${result.successfulSources}/${sources.length} sources, ${result.totalChunks} chunks`,
    );

    return result;
  }

  /**
   * Index a single source.
   */
  private async indexSingleSource(source: IndexingSource): Promise<number> {
    this.logger.debug(`Indexing source: ${source.id}`);

    // Chunk the content using the correct method signature
    const chunks = await this.chunkingService.chunkTextContent(
      source.content,
      source.id,
      source.type,
      {
        sourceId: source.id,
        ...source.metadata,
      },
    );

    // Generate embeddings for chunks
    const chunksWithEmbeddings: RAGChunk[] = [];

    for (const chunk of chunks) {
      try {
        const embedding = await this.embeddingService.generateEmbedding(
          chunk.content,
        );
        chunksWithEmbeddings.push({
          ...chunk,
          embedding,
        });
      } catch (error) {
        this.logger.error(
          `Failed to generate embedding for chunk ${chunk.id}`,
          error,
        );
        // Continue with other chunks
      }
    }

    // Store chunks in vector store
    await this.vectorStore.addChunks(chunksWithEmbeddings);

    this.logger.debug(
      `Successfully indexed ${chunksWithEmbeddings.length} chunks from source ${source.id}`,
    );
    return chunksWithEmbeddings.length;
  }

  /**
   * Enhance a query by retrieving relevant context.
   */
  async enhanceQuery(
    ragQuery: RAGQuery,
    options: QueryEnhancementOptions = {},
  ): Promise<EnhancedQueryResult> {
    this.ensureInitialized();

    const startTime = Date.now();
    this.metrics.totalQueries++;

    this.logger.debug(`Enhancing query: ${ragQuery.text}`);

    try {
      // Generate embedding for the query
      const queryEmbedding = await this.embeddingService.generateEmbedding(
        ragQuery.text,
      );

      // Search for relevant chunks
      const scoredChunks = await this.vectorStore.search(
        ragQuery.text,
        queryEmbedding,
        ragQuery.filters,
        ragQuery.maxResults,
      );

      // Extract chunks from scored results
      const chunks = scoredChunks.map((scored) => ({
        ...scored.chunk,
        score: scored.score,
      }));

      // Apply re-ranking if enabled
      const rerankedChunks = this.config.retrieval.reRankingEnabled
        ? this.reRankChunks(chunks, ragQuery.text)
        : chunks;

      // Assemble context from retrieved chunks
      const context = this.assembleContext(rerankedChunks, options);

      const result: EnhancedQueryResult = {
        content: context.content,
        tokenCount: context.tokenCount,
        sourceChunks: chunks,
        metadata: {
          retrievalTime: Date.now() - startTime,
          totalResults: scoredChunks.length,
          queryType: ragQuery.type,
        },
      };

      // Update metrics
      const retrievalTime = Date.now() - startTime;
      this.updateAverageRetrievalTime(retrievalTime);

      this.logger.debug(
        `Query enhanced: ${result.tokenCount} tokens from ${result.sourceChunks.length} chunks`,
      );

      return result;
    } catch (error) {
      this.logger.error('Failed to enhance query', error);
      throw error;
    }
  }

  /**
   * Re-rank chunks based on query relevance and diversity.
   */
  private reRankChunks(
    chunks: Array<RAGChunk & { score?: number }>,
    query: string,
  ): Array<RAGChunk & { score?: number }> {
    const queryWords = query.toLowerCase().split(/\s+/);

    // Enhanced scoring with query-specific factors
    const enhancedChunks = chunks.map((chunk) => {
      let enhancedScore = chunk.score || 0;

      // Boost exact phrase matches
      if (chunk.content.toLowerCase().includes(query.toLowerCase())) {
        enhancedScore += 0.2;
      }

      // Boost chunks with multiple query word matches
      const contentLower = chunk.content.toLowerCase();
      const matchCount = queryWords.filter(
        (word) => contentLower.includes(word) && word.length > 2,
      ).length;
      enhancedScore += (matchCount / queryWords.length) * 0.15;

      // Boost chunks with title/header matches
      const filePath = chunk.metadata?.file?.path;
      if (
        filePath &&
        queryWords.some((word) => filePath.toLowerCase().includes(word))
      ) {
        enhancedScore += 0.1;
      }

      return { ...chunk, score: enhancedScore };
    });

    // Sort by enhanced score and apply diversity filtering
    const sorted = enhancedChunks.sort(
      (a, b) => (b.score || 0) - (a.score || 0),
    );

    // Simple diversity: avoid too many chunks from the same source
    const diversified: Array<RAGChunk & { score?: number }> = [];
    const sourceCount = new Map<string, number>();

    for (const chunk of sorted) {
      const sourcePath = chunk.metadata?.file?.path || 'unknown';
      const currentCount = sourceCount.get(sourcePath) || 0;

      // Allow max 2 chunks per source file in top results
      if (currentCount < 2 || diversified.length < 3) {
        diversified.push(chunk);
        sourceCount.set(sourcePath, currentCount + 1);
      }

      // Stop when we have enough diverse results
      if (diversified.length >= this.config.retrieval.maxResults) {
        break;
      }
    }

    return diversified;
  }

  /**
   * Assemble context from retrieved chunks.
   */
  private assembleContext(
    chunks: Array<RAGChunk & { score?: number }>,
    options: QueryEnhancementOptions,
  ): { content: string; tokenCount: number } {
    const maxTokens = options.maxTokens || this.config.context.maxTokens;
    let content = '';
    let tokenCount = 0;

    // Sort chunks by relevance (score)
    const sortedChunks = chunks.sort((a, b) => (b.score || 0) - (a.score || 0));

    for (const chunk of sortedChunks) {
      const chunkContent = this.formatChunkForContext(chunk, options);
      const chunkTokens = this.estimateTokenCount(chunkContent);

      if (tokenCount + chunkTokens > maxTokens) {
        break;
      }

      content += chunkContent + '\n\n';
      tokenCount += chunkTokens;
    }

    return { content: content.trim(), tokenCount };
  }

  /**
   * Format a chunk for inclusion in context.
   */
  private formatChunkForContext(
    chunk: RAGChunk,
    _options: QueryEnhancementOptions,
  ): string {
    let formatted = `## ${chunk.type} - ${chunk.id}\n`;

    if (chunk.metadata?.file?.path) {
      formatted += `**Source:** ${chunk.metadata.file.path}\n`;
    }

    if (chunk.language) {
      formatted += `**Language:** ${chunk.language}\n`;
    }

    formatted += `\n${chunk.content}`;

    return formatted;
  }

  /**
   * Simple token count estimation.
   */
  private estimateTokenCount(text: string): number {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }

  /**
   * Update average retrieval time.
   */
  private updateAverageRetrievalTime(newTime: number): void {
    const totalQueries = this.metrics.totalQueries;
    const currentAverage = this.metrics.averageRetrievalTime;

    this.metrics.averageRetrievalTime =
      (currentAverage * (totalQueries - 1) + newTime) / totalQueries;
  }

  /**
   * Get system metrics.
   */
  getMetrics(): RAGMetrics {
    return {
      ...this.metrics,
      memoryUsage: process.memoryUsage(),
    };
  }

  /**
   * Shutdown the RAG service.
   */
  async shutdown(): Promise<void> {
    if (!this.isInitialized) {
      return;
    }

    this.logger.info('Shutting down RAG service...');

    try {
      // Note: shutdown methods are optional in the interfaces
      this.isInitialized = false;
      this.logger.info('RAG service shut down successfully');
    } catch (error) {
      this.logger.error('Error during RAG service shutdown', error);
      throw error;
    }
  }

  /**
   * Load RAG configuration from user config.
   */
  private loadRAGConfig(): RAGConfig {
    // Default configuration - in real implementation, this would come from config files
    return {
      enabled: true,
      vectorStore: {
        provider: 'memory',
        connectionString: '',
        apiKey: '',
        collectionName: 'default',
        persistenceDirectory: '',
      },
      embedding: {
        model: 'text-embedding-004',
        dimension: 768,
        batchSize: 10,
        cacheSize: 1000,
        maxRetries: 3,
      },
      chunking: {
        strategy: 'ast',
        maxChunkSize: 1000,
        minChunkSize: 100,
        overlapRatio: 0.1,
        respectBoundaries: true,
      },
      retrieval: {
        maxResults: 10,
        similarityThreshold: 0.7,
        hybridWeights: {
          semantic: 0.7,
          keyword: 0.2,
          graph: 0.05,
          recency: 0.05,
        },
        reRankingEnabled: true,
        diversityThreshold: 0.8,
      },
      context: {
        maxTokens: 4000,
        compressionRatio: 0.8,
        includeDependencies: true,
        includeDocumentation: true,
        prioritizeRecent: true,
        preserveStructure: true,
      },
      performance: {
        enableCaching: true,
        cacheExpiration: 3600,
        batchProcessing: true,
        parallelProcessing: true,
        maxConcurrentOperations: 5,
      },
      debug: {
        enableLogging: true,
        logLevel: 'info',
        enableMetrics: true,
        enableTracing: false,
      },
    };
  }

  /**
   * Ensure the service is initialized.
   */
  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error('RAG service not initialized. Call initialize() first.');
    }
  }
}

// Export for compatibility with example
export { IndexingSource };
