import {
  RAGVectorStore,
  RAGChunk,
  VectorStoreConfig,
  ScoredChunk,
  VectorStoreStats,
  QueryFilters,
  RAGVectorStoreError,
  ChunkType,
  ChunkSource,
  ChunkMetadata,
} from '../types.js';
import { RAGLogger } from '../logger.js';

// Types for Chroma client (to avoid requiring the actual package when not used)
interface ChromaClient {
  getOrCreateCollection(params: {
    name: string;
    metadata?: Record<string, unknown>;
  }): Promise<ChromaCollection>;
  deleteCollection(name: string): Promise<void>;
  heartbeat(): Promise<Record<string, unknown>>;
}

interface ChromaClient {
  getOrCreateCollection(params: { name: string }): Promise<ChromaCollection>;
  heartbeat(): Promise<void>;
}

interface ChromaCollection {
  add(params: {
    ids: string[];
    embeddings: number[][];
    metadatas?: Array<Record<string, unknown>>;
    documents?: string[];
  }): Promise<void>;

  query(params: {
    queryEmbeddings: number[][];
    nResults?: number;
    where?: Record<string, unknown>;
    include?: string[];
  }): Promise<{
    ids: string[][];
    distances: number[][];
    metadatas: Array<Array<Record<string, unknown>>>;
    documents: string[][];
  }>;

  get(params: {
    ids?: string[];
    where?: Record<string, unknown>;
    include?: string[];
  }): Promise<{
    ids: string[];
    embeddings?: number[][];
    metadatas?: Array<Record<string, unknown>>;
    documents?: string[];
  }>;

  update(params: {
    ids: string[];
    embeddings?: number[][];
    metadatas?: Array<Record<string, unknown>>;
    documents?: string[];
  }): Promise<void>;

  delete(params: { ids: string[] }): Promise<void>;
  count(): Promise<number>;
}

/**
 * Chroma Vector Store implementation for RAG system.
 * Provides persistent, scalable vector storage with advanced querying capabilities.
 */
export class RAGChromaVectorStore extends RAGVectorStore {
  private client: ChromaClient | null = null;
  private collection: ChromaCollection | null = null;
  private collectionName: string;
  private initialized = false;
  private connectionRetries = 0;
  private maxRetries = 3;

  constructor(
    config: VectorStoreConfig,
    private logger: RAGLogger,
    private hybridWeights = {
      semantic: 0.7,
      keyword: 0.2,
      graph: 0.05,
      recency: 0.05,
    },
  ) {
    super(config);
    this.collectionName = config.collection || 'gemini_rag_chunks';
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      await this.initializeWithRetry();
      this.initialized = true;
      this.logger.info('ChromaVectorStore initialized successfully', {
        collectionName: this.collectionName,
        connectionString: this.config.connectionString,
      });
    } catch (error) {
      this.logger.error('Failed to initialize ChromaVectorStore', { error });
      throw new RAGVectorStoreError(
        `ChromaVectorStore initialization failed: ${error}`,
      );
    }
  }

  private async initializeWithRetry(): Promise<void> {
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        await this.initializeClient();
        return;
      } catch (error) {
        this.connectionRetries = attempt;
        if (attempt === this.maxRetries) {
          throw error;
        }

        this.logger.warn(
          `Chroma connection attempt ${attempt}/${this.maxRetries} failed, retrying...`,
          { error },
        );
        await this.sleep(Math.pow(2, attempt) * 1000); // Exponential backoff
      }
    }
  }

  private async initializeClient(): Promise<void> {
    try {
      // Dynamic import to handle optional dependency
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const chromaModule = (await this.loadChromaClient()) as any;

      if (!chromaModule) {
        throw new Error(
          'ChromaDB client not available. Install with: npm install chromadb',
        );
      }

      // Initialize Chroma client
      this.client = new chromaModule.ChromaClient({
        path: this.config.connectionString || 'http://localhost:8000',
      });

      // Test connection
      if (this.client) {
        await this.client.heartbeat();

        // Get or create collection with optimized settings
        this.collection = await this.client.getOrCreateCollection({
          name: this.collectionName,
        });
      }
    } catch (error) {
      throw new RAGVectorStoreError(
        `Failed to initialize Chroma client: ${error}`,
      );
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async loadChromaClient(): Promise<any> {
    try {
      return await import('chromadb');
    } catch (_error) {
      this.logger.warn(
        'ChromaDB not installed. Falling back to memory store for development.',
      );
      return null;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async addChunks(chunks: RAGChunk[]): Promise<void> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    try {
      // Validate chunks have embeddings
      const validChunks = chunks.filter(
        (chunk) => chunk.embedding && chunk.embedding.length > 0,
      );
      if (validChunks.length !== chunks.length) {
        this.logger.warn('Some chunks missing embeddings, skipping them', {
          total: chunks.length,
          valid: validChunks.length,
        });
      }

      if (validChunks.length === 0) {
        this.logger.warn('No valid chunks with embeddings to add');
        return;
      }

      // Prepare data for ChromaDB
      const ids = validChunks.map((chunk) => chunk.id);
      const embeddings = validChunks.map((chunk) => chunk.embedding!);
      const documents = validChunks.map((chunk) => chunk.content);
      const metadatas = validChunks.map((chunk) => ({
        ...chunk.metadata,
        type: chunk.type,
        language: chunk.language,
        source: chunk.source,
      }));

      await this.collection.add({
        ids,
        embeddings,
        documents,
        metadatas,
      });

      this.logger.debug('Added chunks to Chroma', {
        count: validChunks.length,
      });
    } catch (error) {
      this.logger.error('Failed to add chunks to ChromaVectorStore', {
        error,
        chunkCount: chunks.length,
      });
      throw error;
    }
  }

  async search(
    _query: string,
    embedding: number[],
    filters?: QueryFilters,
    limit?: number,
  ): Promise<ScoredChunk[]> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    const maxResults = limit || 10;

    try {
      // Build where clause from filters
      const where: Record<string, unknown> = {};
      if (filters?.languages?.length) {
        where['language'] = { $in: filters.languages };
      }
      if (filters?.filePatterns?.length) {
        // Note: ChromaDB where clause limitations - this is simplified
        where['source.filePath'] = { $regex: filters.filePatterns[0] };
      }
      if (filters?.chunkTypes?.length) {
        where['type'] = { $in: filters.chunkTypes };
      }

      const results = await this.collection.query({
        queryEmbeddings: [embedding],
        nResults: maxResults,
        where: Object.keys(where).length > 0 ? where : undefined,
        include: ['documents', 'metadatas', 'distances'],
      });

      // Convert results to ScoredChunk[]
      const scoredChunks: ScoredChunk[] = [];

      if (
        results.documents &&
        results.documents[0] &&
        results.metadatas &&
        results.distances
      ) {
        for (let i = 0; i < results.documents[0].length; i++) {
          const document = results.documents[0][i];
          const metadata = results.metadatas[0][i];
          const distance = results.distances[0][i];
          const id = results.ids[0][i];

          if (document && metadata && typeof distance === 'number') {
            // Convert distance to similarity score (cosine distance -> similarity)
            const score = 1 - distance;

            // Skip results below minimum quality
            if (filters?.minQuality && score < filters.minQuality) {
              continue;
            }

            const chunk: RAGChunk = {
              id,
              content: document,
              type: (metadata.type as ChunkType) || ChunkType.CODE_SNIPPET,
              language: metadata.language as string,
              source: (metadata.source as ChunkSource) || {
                type: 'file',
                filePath: 'unknown',
                range: { start: 0, end: 0 },
              },
              metadata: metadata as ChunkMetadata,
              embedding,
              timestamp: new Date().toISOString(), // TODO: Store actual timestamp
              contentHash: `hash-${id}`, // TODO: Store actual content hash
            };

            const scoredChunk: ScoredChunk = {
              chunk,
              score,
              scoreBreakdown: {
                semantic: score,
                keyword: 0,
                graph: 0,
                recency: 0,
                quality: score,
              },
            };

            scoredChunks.push(scoredChunk);
          }
        }
      }

      this.logger.debug('Performed Chroma search', {
        maxResults,
        found: scoredChunks.length,
        query: _query.substring(0, 100),
      });

      return scoredChunks;
    } catch (error) {
      this.logger.error('ChromaVectorStore search failed', { error });
      throw error;
    }
  }

  async updateChunks(chunks: RAGChunk[]): Promise<void> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    try {
      // TODO: Implement Chroma update
      // For Chroma, this typically involves deleting and re-adding
      const ids = chunks.map((chunk) => chunk.id);
      await this.deleteChunks(ids);
      await this.addChunks(chunks);

      this.logger.debug('Updated chunks in Chroma', { count: chunks.length });
    } catch (error) {
      this.logger.error('Failed to update chunks in ChromaVectorStore', {
        error,
      });
      throw error;
    }
  }

  async deleteChunks(chunkIds: string[]): Promise<void> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    try {
      if (chunkIds.length === 0) {
        this.logger.warn('No chunk IDs provided for deletion');
        return;
      }

      await this.collection.delete({
        ids: chunkIds,
      });

      this.logger.debug('Deleted chunks from Chroma', {
        count: chunkIds.length,
      });
    } catch (error) {
      this.logger.error('Failed to delete chunks from ChromaVectorStore', {
        error,
        chunkIds,
      });
      throw error;
    }
  }

  async getChunk(id: string): Promise<RAGChunk | null> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    try {
      const results = await this.collection.get({
        ids: [id],
        include: ['documents', 'metadatas', 'embeddings'],
      });

      if (results.ids && results.ids.length > 0 && results.documents) {
        const metadata = results.metadatas ? results.metadatas[0] : {};
        const document = results.documents[0];
        const embedding = results.embeddings
          ? results.embeddings[0]
          : undefined;

        if (document) {
          return {
            id,
            content: document,
            type: (metadata.type as ChunkType) || ChunkType.CODE_SNIPPET,
            language: metadata.language as string,
            source: (metadata.source as ChunkSource) || {
              type: 'file',
              filePath: 'unknown',
              range: { start: 0, end: 0 },
            },
            metadata: metadata as ChunkMetadata,
            embedding: embedding as number[],
            timestamp:
              (metadata.timestamp as string) || new Date().toISOString(),
            contentHash: (metadata.contentHash as string) || `hash-${id}`,
          };
        }
      }

      return null;
    } catch (error) {
      this.logger.error('Failed to get chunk from ChromaVectorStore', {
        error,
        id,
      });
      throw error;
    }
  }

  async listChunks(filters?: QueryFilters): Promise<string[]> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    try {
      // Build where clause from filters
      const where: Record<string, unknown> = {};
      if (filters?.languages?.length) {
        where['language'] = { $in: filters.languages };
      }
      if (filters?.chunkTypes?.length) {
        where['type'] = { $in: filters.chunkTypes };
      }
      if (filters?.metadata) {
        Object.assign(where, filters.metadata);
      }

      const results = await this.collection.get({
        where: Object.keys(where).length > 0 ? where : undefined,
        include: ['documents'], // Only need IDs, documents for filtering
      });

      return results.ids || [];
    } catch (error) {
      this.logger.error('Failed to list chunks from ChromaVectorStore', {
        error,
      });
      throw error;
    }
  }

  async getStats(): Promise<VectorStoreStats> {
    if (!this.initialized) {
      throw new Error('ChromaVectorStore not initialized');
    }

    if (!this.collection) {
      throw new Error('ChromaVectorStore collection not available');
    }

    try {
      const count = await this.collection.count();

      return {
        totalChunks: count,
        indexSize: count * 1536 * 4, // Rough estimate: count * embedding_dim * 4 bytes
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      this.logger.error('Failed to get ChromaVectorStore stats', { error });
      throw error;
    }
  }

  async shutdown(): Promise<void> {
    await this.close();
  }

  async close(): Promise<void> {
    try {
      // TODO: Cleanup Chroma connection
      this.client = null;
      this.collection = null;
      this.initialized = false;

      this.logger.info('ChromaVectorStore shutdown complete');
    } catch (error) {
      this.logger.error('Error during ChromaVectorStore shutdown', { error });
    }
  }
}
