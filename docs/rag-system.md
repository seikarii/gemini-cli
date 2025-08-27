# RAG System for Gemini CLI

## Overview

This is a sophisticated Retrieval-Augmented Generation (RAG) system designed to enhance the Gemini CLI's capabilities by providing intelligent context retrieval and reducing LLM hallucinations. The system goes beyond simple keyword matching and leverages advanced techniques for code understanding and conversational context.

## Features

### 🧠 Intelligent Code Understanding
- **AST-based chunking**: Parses code into semantically meaningful units (functions, classes, methods)
- **Multi-language support**: TypeScript, JavaScript, Python, and extensible for more
- **Smart overlapping**: Preserves context at chunk boundaries
- **Hierarchical organization**: Functions → Classes → Files → Modules

### 🔍 Advanced Retrieval
- **Hybrid search**: Combines semantic vector search with keyword-based search (BM25)
- **Re-ranking**: Uses multiple scoring factors for optimal relevance
- **Query expansion**: Automatically improves queries for better retrieval
- **Graph-based relationships**: Understands code dependencies and call graphs

### 📊 Context Optimization
- **Token budget management**: Optimizes context within model limits
- **Dependency resolution**: Auto-includes required imports and dependencies
- **Smart compression**: Intelligently summarizes less relevant context
- **Documentation integration**: Includes relevant API docs and examples

### ⚡ Performance & Scalability
- **Pluggable vector stores**: Memory, Chroma, Pinecone, Weaviate, Qdrant
- **Intelligent caching**: Multi-level caching for embeddings and retrieval
- **Incremental updates**: Efficient handling of codebase changes
- **Batch processing**: Optimized for large codebases

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG System                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Data Ingestion │ │   Retrieval     │ │   Context       │   │
│  │   & Indexing    │ │    Engine       │ │  Augmentation   │   │
│  │                 │ │                 │ │                 │   │
│  │ • AST Chunking  │ │ • Hybrid Search │ │ • Smart Fusion  │   │
│  │ • Embeddings    │ │ • Re-ranking    │ │ • Dependency    │   │
│  │ • Metadata      │ │ • Query Expand  │ │   Resolution    │   │
│  │ • Vector DB     │ │ • Graph Search  │ │ • Summarization │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Enable RAG System

Set environment variables to enable and configure the RAG system:

```bash
export RAG_ENABLED=true
export RAG_VECTOR_STORE=memory  # or chroma, pinecone, etc.
export RAG_EMBEDDING_MODEL=text-embedding-004
export RAG_MAX_TOKENS=8000
```

### 2. Initialize RAG Service

```typescript
import { RAGService } from '@google/gemini-cli-core';
import { Config } from '@google/gemini-cli-core';

const config = new Config(/* your config */);
const ragService = new RAGService(config);
await ragService.initialize();
```

### 3. Index Your Codebase

```typescript
import { ChunkType } from '@google/gemini-cli-core';

// Index code files
const sources = [
  {
    id: 'src/components/Button.tsx',
    type: ChunkType.CODE_CLASS,
    content: fs.readFileSync('src/components/Button.tsx', 'utf-8'),
    metadata: {
      file: {
        path: 'src/components/Button.tsx',
        extension: '.tsx',
      },
    },
  },
  // ... more sources
];

const result = await ragService.indexContent(sources);
console.log(`Indexed ${result.totalChunks} chunks`);
```

### 4. Enhanced Query Processing

```typescript
// Enhance a user query with relevant context
const enhancedContext = await ragService.enhanceQuery(
  {
    text: "How do I create a React component?",
    type: QueryType.CODE_GENERATION,
    maxResults: 5,
  },
  {
    maxTokens: 4000,
    includeDependencies: true,
    includeDocumentation: true,
  }
);

// Use the enhanced context in your LLM prompt
const prompt = `Context: ${enhancedContext.content}\\n\\nUser: ${userQuery}`;
```

## Configuration

The RAG system can be configured through environment variables:

### Core Settings
- `RAG_ENABLED`: Enable/disable RAG system (default: false)
- `RAG_VECTOR_STORE`: Vector store provider (memory, chroma, pinecone, weaviate, qdrant)
- `RAG_EMBEDDING_MODEL`: Embedding model to use
- `RAG_MAX_TOKENS`: Maximum tokens for context (default: 8000)

### Chunking Settings
- `RAG_CHUNKING_STRATEGY`: Chunking strategy (ast, semantic, fixed, hybrid)
- `RAG_MAX_CHUNK_SIZE`: Maximum chunk size in characters (default: 1000)
- `RAG_MIN_CHUNK_SIZE`: Minimum chunk size in characters (default: 100)
- `RAG_OVERLAP_RATIO`: Overlap ratio between chunks (default: 0.1)

### Retrieval Settings
- `RAG_MAX_RESULTS`: Maximum results to retrieve (default: 10)
- `RAG_SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)
- `RAG_SEMANTIC_WEIGHT`: Weight for semantic search (default: 0.7)
- `RAG_KEYWORD_WEIGHT`: Weight for keyword search (default: 0.2)
- `RAG_RERANKING_ENABLED`: Enable re-ranking (default: true)

### Performance Settings
- `RAG_ENABLE_CACHING`: Enable caching (default: true)
- `RAG_CACHE_EXPIRATION`: Cache expiration in seconds (default: 3600)
- `RAG_BATCH_SIZE`: Batch size for processing (default: 100)
- `RAG_MAX_CONCURRENT`: Maximum concurrent operations (default: 5)

## Integration with GeminiChat

The RAG system can be integrated with the existing GeminiChat system to enhance conversations:

```typescript
import { GeminiChat } from '@google/gemini-cli-core';
import { RAGService } from '@google/gemini-cli-core';

class EnhancedGeminiChat extends GeminiChat {
  constructor(
    config: Config,
    contentGenerator: ContentGenerator,
    private ragService: RAGService
  ) {
    super(config, contentGenerator);
  }

  async sendMessage(params: SendMessageParameters, prompt_id: string) {
    // Enhance the message with RAG context
    const ragContext = await this.ragService.enhanceQuery(
      {
        text: params.message,
        type: QueryType.GENERAL_QUESTION,
      },
      {
        maxTokens: 4000,
        includeDependencies: true,
      }
    );

    // Modify the message to include context
    const enhancedParams = {
      ...params,
      message: `Context: ${ragContext.content}\\n\\nUser: ${params.message}`,
    };

    return super.sendMessage(enhancedParams, prompt_id);
  }
}
```

## Vector Store Providers

### Memory (Development)
```bash
export RAG_VECTOR_STORE=memory
```
Simple in-memory vector store for development and testing.

### Chroma (Local)
```bash
export RAG_VECTOR_STORE=chroma
export RAG_CONNECTION_STRING=http://localhost:8000
```
Local Chroma instance for persistent storage.

### Pinecone (Cloud)
```bash
export RAG_VECTOR_STORE=pinecone
export RAG_API_KEY=your_pinecone_api_key
export RAG_CONNECTION_STRING=your_pinecone_environment
```

### Weaviate (Cloud/Local)
```bash
export RAG_VECTOR_STORE=weaviate
export RAG_CONNECTION_STRING=https://your-cluster.weaviate.network
export RAG_API_KEY=your_weaviate_api_key
```

## API Reference

### RAGService

Main service for RAG operations.

```typescript
class RAGService {
  constructor(config: Config, logger?: RAGLogger)
  
  // Initialize the service
  async initialize(): Promise<void>
  
  // Index content for retrieval
  async indexContent(sources: IndexingSource[]): Promise<IndexingResult>
  
  // Retrieve relevant context
  async retrieveContext(query: RAGQuery): Promise<RAGResult>
  
  // Assemble context for LLM
  async assembleContext(result: RAGResult, options: ContextAssemblyOptions): Promise<RAGContext>
  
  // Combined retrieval and assembly
  async enhanceQuery(query: RAGQuery, options: ContextAssemblyOptions): Promise<RAGContext>
  
  // Update/delete chunks
  async updateChunks(chunks: RAGChunk[]): Promise<void>
  async deleteChunks(chunkIds: string[]): Promise<void>
  
  // Get metrics
  getMetrics(): RAGServiceMetrics
  
  // Cleanup
  async shutdown(): Promise<void>
}
```

### Types

Key types for the RAG system:

```typescript
interface RAGQuery {
  text: string;
  type?: QueryType;
  filters?: QueryFilters;
  maxResults?: number;
  includeContext?: boolean;
  boosts?: Record<ChunkType, number>;
}

interface RAGContext {
  content: string;
  sourceChunks: ScoredChunk[];
  tokenCount: number;
  assembly: {
    strategy: string;
    compressionRatio: number;
    dependenciesIncluded: boolean;
    summariesGenerated: number;
  };
}

interface RAGChunk {
  id: string;
  content: string;
  type: ChunkType;
  language?: string;
  source: ChunkSource;
  metadata: ChunkMetadata;
  embedding?: number[];
  timestamp: string;
  contentHash: string;
}
```

## Performance Optimization

### Caching Strategy
- **Embedding cache**: Cache embeddings for frequently accessed content
- **Retrieval cache**: Cache search results for repeated queries
- **Smart invalidation**: Invalidate cache when content changes

### Batch Processing
- **Embedding generation**: Process multiple texts in batches
- **Indexing**: Batch index operations for efficiency
- **Parallel processing**: Concurrent operations where safe

### Memory Management
- **Lazy loading**: Load chunks only when needed
- **Streaming**: Stream large result sets
- **Garbage collection**: Clean up unused resources

## Monitoring and Debugging

### Metrics
The RAG system provides comprehensive metrics:

```typescript
interface RAGServiceMetrics {
  totalQueries: number;
  totalChunksIndexed: number;
  averageRetrievalTime: number;
  cacheHitRate: number;
  errorRate: number;
}
```

### Debugging
Enable debug logging:

```bash
export RAG_ENABLE_LOGGING=true
export RAG_LOG_LEVEL=debug
```

### Health Checks
Monitor system health:

```typescript
const stats = await ragService.getVectorStore().getStats();
console.log('Vector store stats:', stats);
```

## Troubleshooting

### Common Issues

**1. RAG system not initializing**
- Check `RAG_ENABLED=true` is set
- Verify vector store configuration
- Check network connectivity for cloud providers

**2. Poor retrieval quality**
- Adjust similarity threshold: `RAG_SIMILARITY_THRESHOLD`
- Tune hybrid search weights
- Check embedding model compatibility

**3. High memory usage**
- Reduce cache size: `RAG_CACHE_SIZE`
- Enable context compression: `RAG_COMPRESSION_RATIO`
- Use streaming for large datasets

**4. Slow performance**
- Enable caching: `RAG_ENABLE_CACHING=true`
- Increase batch size: `RAG_BATCH_SIZE`
- Use parallel processing: `RAG_PARALLEL_PROCESSING=true`

### Performance Tuning

1. **Vector Store Selection**
   - Use memory store for development
   - Use Chroma for local production
   - Use cloud providers for scale

2. **Chunk Size Optimization**
   - Smaller chunks: Better precision, more storage
   - Larger chunks: Better context, less precise

3. **Embedding Model Selection**
   - Code-specific models for code content
   - General models for mixed content
   - Consider model size vs. quality trade-offs

## Contributing

The RAG system is designed to be extensible. Key extension points:

### Custom Vector Stores
Implement the `RAGVectorStore` abstract class:

```typescript
class CustomVectorStore extends RAGVectorStore {
  async initialize(): Promise<void> { /* implementation */ }
  async addChunks(chunks: RAGChunk[]): Promise<void> { /* implementation */ }
  // ... other methods
}
```

### Custom Chunking Strategies
Extend the chunking system:

```typescript
class CustomChunkingService {
  async chunkContent(content: string, type: ChunkType): Promise<RAGChunk[]> {
    // Custom chunking logic
  }
}
```

### Custom Embeddings
Implement custom embedding services:

```typescript
class CustomEmbeddingService extends RAGEmbeddingService {
  async generateEmbedding(text: string): Promise<number[]> {
    // Custom embedding logic
  }
}
```

## License

This RAG system is part of the Gemini CLI project and follows the same Apache 2.0 license.

## Support

For issues and questions:
1. Check this documentation
2. Review the example code in `rag/example.ts`
3. Enable debug logging for troubleshooting
4. File issues in the main Gemini CLI repository
