import { RAGService } from '../ragService.js';
import { RAGLogger } from '../logger.js';
import {
  ScoredChunk,
  QueryFilters,
  ChunkType,
  RAGChunk,
  RAGQuery,
} from '../types.js';

/**
 * Configuration for RAG context integration
 */
export interface RAGContextConfig {
  /** Enable RAG context enhancement */
  enabled: boolean;
  /** Maximum number of chunks to retrieve */
  maxChunks: number;
  /** Minimum relevance score threshold */
  relevanceThreshold: number;
  /** Maximum total context length in tokens */
  maxContextTokens: number;
  /** Include code snippets in context */
  includeCode: boolean;
  /** Include documentation in context */
  includeDocumentation: boolean;
  /** Include conversation history in context */
  includeConversations: boolean;
  /** Context summarization strategy */
  summarizationStrategy: 'none' | 'extractive' | 'abstractive';
}

/**
 * Enhanced context for conversations
 */
export interface EnhancedContext {
  /** Original user query */
  query: string;
  /** Retrieved relevant chunks */
  relevantChunks: ScoredChunk[];
  /** Formatted context string for the LLM */
  contextString: string;
  /** Metadata about the context */
  metadata: {
    totalChunks: number;
    averageRelevance: number;
    contextLength: number;
    sources: string[];
    chunkTypes: string[];
  };
}

/**
 * Service that integrates RAG capabilities with chat conversations
 * to provide relevant context and enhance response quality.
 */
export class RAGContextService {
  private readonly ragService: RAGService;
  private readonly logger: RAGLogger;
  private readonly config: RAGContextConfig;

  constructor(
    ragService: RAGService,
    logger: RAGLogger,
    config: Partial<RAGContextConfig> = {},
  ) {
    this.ragService = ragService;
    this.logger = logger;
    this.config = {
      enabled: true,
      maxChunks: 10,
      relevanceThreshold: 0.7,
      maxContextTokens: 4000,
      includeCode: true,
      includeDocumentation: true,
      includeConversations: false,
      summarizationStrategy: 'extractive',
      ...config,
    };

    this.logger.info('RAGContextService initialized', { config: this.config });
  }

  /**
   * Enhance a user query with relevant context from the RAG system
   */
  async enhanceQuery(
    query: string,
    _conversationHistory?: string[],
  ): Promise<EnhancedContext> {
    if (!this.config.enabled) {
      return {
        query,
        relevantChunks: [],
        contextString: '',
        metadata: {
          totalChunks: 0,
          averageRelevance: 0,
          contextLength: 0,
          sources: [],
          chunkTypes: [],
        },
      };
    }

    try {
      // Build query filters based on configuration
      const filters = this.buildQueryFilters();

      // Create RAG query
      const ragQuery: RAGQuery = {
        text: query,
        maxResults: this.config.maxChunks,
        filters,
      };

      // Retrieve relevant chunks using RAG service
      const result = await this.ragService.enhanceQuery(ragQuery);
      const chunks: ScoredChunk[] = result.sourceChunks.map((chunk) => ({
        chunk,
        score: 0.8, // Default score since we don't have it from sourceChunks
        scoreBreakdown: {
          semantic: 0.8,
          keyword: 0,
          graph: 0,
          recency: 0,
          quality: 0.8,
        },
      }));

      // Filter by relevance threshold
      const relevantChunks = chunks.filter(
        (chunk: ScoredChunk) => chunk.score >= this.config.relevanceThreshold,
      );

      if (relevantChunks.length === 0) {
        this.logger.debug('No relevant chunks found for query', { query });
        return {
          query,
          relevantChunks: [],
          contextString: '',
          metadata: {
            totalChunks: 0,
            averageRelevance: 0,
            contextLength: 0,
            sources: [],
            chunkTypes: [],
          },
        };
      }

      // Generate enhanced context
      const contextString = await this.formatContext(relevantChunks, query);

      // Calculate metadata
      const metadata = this.calculateMetadata(relevantChunks, contextString);

      this.logger.debug('Query enhanced with RAG context', {
        query: query.substring(0, 100),
        chunksFound: relevantChunks.length,
        averageRelevance: metadata.averageRelevance,
        contextLength: metadata.contextLength,
      });

      return {
        query,
        relevantChunks,
        contextString,
        metadata,
      };
    } catch (error) {
      this.logger.error('Failed to enhance query with RAG context', {
        error,
        query,
      });

      // Return empty context on error to not break the conversation
      return {
        query,
        relevantChunks: [],
        contextString: '',
        metadata: {
          totalChunks: 0,
          averageRelevance: 0,
          contextLength: 0,
          sources: [],
          chunkTypes: [],
        },
      };
    }
  }

  /**
   * Format enhanced context for inclusion in LLM prompts
   */
  async formatForPrompt(enhancedContext: EnhancedContext): Promise<string> {
    if (enhancedContext.contextString.length === 0) {
      return '';
    }

    const contextHeader = `
## Relevant Context

The following information has been retrieved from your codebase and documentation that may be relevant to your query:

`;

    const contextFooter = `

---

Please use the above context to provide a more accurate and helpful response. Reference specific code snippets, functions, or documentation when relevant.
`;

    return contextHeader + enhancedContext.contextString + contextFooter;
  }

  /**
   * Update RAG context configuration
   */
  updateConfig(newConfig: Partial<RAGContextConfig>): void {
    Object.assign(this.config, newConfig);
    this.logger.info('RAGContextService configuration updated', {
      config: this.config,
    });
  }

  /**
   * Get current service statistics
   */
  getStats() {
    return {
      config: this.config,
      enabled: this.config.enabled,
    };
  }

  private buildQueryFilters(): QueryFilters {
    const chunkTypes: ChunkType[] = [];

    if (this.config.includeCode) {
      chunkTypes.push(
        ChunkType.CODE_FUNCTION,
        ChunkType.CODE_CLASS,
        ChunkType.CODE_MODULE,
        ChunkType.CODE_SNIPPET,
      );
    }

    if (this.config.includeDocumentation) {
      chunkTypes.push(ChunkType.DOCUMENTATION, ChunkType.COMMENT);
    }

    if (this.config.includeConversations) {
      chunkTypes.push(ChunkType.CONVERSATION);
    }

    return {
      chunkTypes,
      minQuality: this.config.relevanceThreshold,
    };
  }

  private async formatContext(
    chunks: ScoredChunk[],
    query: string,
  ): Promise<string> {
    let context = '';
    let currentLength = 0;

    // Sort chunks by relevance score (highest first)
    const sortedChunks = chunks.sort((a, b) => b.score - a.score);

    for (const scoredChunk of sortedChunks) {
      const chunk = scoredChunk.chunk;

      // Estimate token count (rough approximation: 1 token ≈ 4 characters)
      const chunkTokens = Math.ceil(chunk.content.length / 4);

      if (currentLength + chunkTokens > this.config.maxContextTokens) {
        break;
      }

      // Format chunk based on type
      const formattedChunk = this.formatChunk(chunk, scoredChunk.score);
      context += formattedChunk + '\n\n';
      currentLength += chunkTokens;
    }

    // Apply summarization if configured
    if (this.config.summarizationStrategy === 'extractive') {
      context = this.extractiveSummarization(context, query);
    }

    return context.trim();
  }

  private formatChunk(chunk: RAGChunk, score: number): string {
    const relevancePercent = Math.round(score * 100);
    const source = chunk.source?.id || 'Unknown source';

    let formattedContent = '';

    // Add source information
    formattedContent += `**Source**: ${source} (Relevance: ${relevancePercent}%)\n`;

    // Add type-specific formatting
    if (chunk.type.toString().startsWith('code_')) {
      formattedContent += `**Type**: ${chunk.type}\n`;
      if (chunk.language) {
        formattedContent += `\`\`\`${chunk.language}\n${chunk.content}\n\`\`\``;
      } else {
        formattedContent += `\`\`\`\n${chunk.content}\n\`\`\``;
      }
    } else if (chunk.type === ChunkType.DOCUMENTATION) {
      formattedContent += `**Documentation**:\n${chunk.content}`;
    } else {
      formattedContent += chunk.content;
    }

    return formattedContent;
  }

  private extractiveSummarization(context: string, query: string): string {
    // Simple extractive summarization - could be enhanced with NLP techniques
    const sentences = context
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 10);
    const queryWords = query.toLowerCase().split(/\s+/);

    // Score sentences based on query word overlap
    const scoredSentences = sentences.map((sentence) => {
      const sentenceWords = sentence.toLowerCase().split(/\s+/);
      const overlap = queryWords.filter((word) =>
        sentenceWords.some((sw) => sw.includes(word)),
      ).length;

      return {
        sentence: sentence.trim(),
        score: overlap / queryWords.length,
      };
    });

    // Return top sentences
    return (
      scoredSentences
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.min(10, Math.ceil(scoredSentences.length * 0.3)))
        .map((item) => item.sentence)
        .join('. ') + '.'
    );
  }

  private calculateMetadata(chunks: ScoredChunk[], contextString: string) {
    const sources = [
      ...new Set(chunks.map((c) => c.chunk.source?.id || 'Unknown')),
    ];
    const chunkTypes = [...new Set(chunks.map((c) => c.chunk.type))];
    const averageRelevance =
      chunks.reduce((sum, c) => sum + c.score, 0) / chunks.length;

    return {
      totalChunks: chunks.length,
      averageRelevance: Math.round(averageRelevance * 10000) / 10000,
      contextLength: contextString.length,
      sources,
      chunkTypes,
    };
  }
}
