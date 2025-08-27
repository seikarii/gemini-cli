/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Example integration showing how to use the RAG system
 * with the existing Gemini CLI infrastructure.
 */

import { RAGService, IndexingSource } from './ragService.js';
import { Config } from '../config/config.js';
import { ChunkType, QueryType, RAGQuery } from './types.js';
import { ConsoleRAGLogger } from './logger.js';

/**
 * Example of how to integrate RAG with the existing chat system.
 * This demonstrates the end-to-end workflow.
 */
export class RAGIntegrationExample {
  private ragService: RAGService;

  constructor(config: Config) {
    const logger = new ConsoleRAGLogger('[RAG-Example]');
    this.ragService = new RAGService(config, logger);
  }

  /**
   * Initialize the RAG system and index some example content.
   */
  async initialize(): Promise<void> {
    console.log('🚀 Initializing RAG system...');
    
    // Enable RAG by setting environment variable
    process.env.RAG_ENABLED = 'true';
    
    await this.ragService.initialize();
    
    // Index some example code content
    await this.indexExampleContent();
  }

  /**
   * Example of indexing code files and conversations.
   */
  private async indexExampleContent(): Promise<void> {
    console.log('📚 Indexing example content...');

    const sources: IndexingSource[] = [
      {
        id: 'geminiChat.ts',
        type: ChunkType.CODE_CLASS,
        content: `
export class GeminiChat {
  private sendPromise: Promise<void> = Promise.resolve();
  private historyCache: { lastHistoryHash: string; curatedHistory: Content[] } = {
    lastHistoryHash: '',
    curatedHistory: [],
  };

  constructor(
    private readonly config: Config,
    private readonly contentGenerator: ContentGenerator,
    private readonly generationConfig: GenerateContentConfig = {},
    private history: Content[] = [],
  ) {
    validateHistory(history);
  }

  async sendMessage(params: SendMessageParameters, prompt_id: string): Promise<GenerateContentResponse> {
    await this.sendPromise;
    const userContent = createUserContent(params.message);
    const requestContents = this.getHistory(true).concat(userContent);
    // ... rest of implementation
  }
}`,
        metadata: {
          file: {
            path: '/media/seikarii/Nvme/gemini-cli/packages/core/src/core/geminiChat.ts',
            extension: '.ts',
          },
        },
      },
      {
        id: 'conversation-1',
        type: ChunkType.CONVERSATION,
        content: `User: How do I create a new TypeScript class?
Assistant: To create a new TypeScript class, you can use the following syntax:

\`\`\`typescript
export class MyClass {
  private property: string;
  
  constructor(property: string) {
    this.property = property;
  }
  
  public getProperty(): string {
    return this.property;
  }
}
\`\`\`

This creates a class with a private property, a constructor, and a public method.`,
        metadata: {
          conversation: {
            speaker: 'assistant',
            topic: 'typescript-classes',
            timestamp: new Date().toISOString(),
          },
        },
      },
      {
        id: 'utils.ts',
        type: ChunkType.CODE_FUNCTION,
        content: `
export function validateHistory(history: Content[]) {
  for (const content of history) {
    if (content.role !== 'user' && content.role !== 'model') {
      throw new Error(\`Role must be user or model, but got \${content.role}.\`);
    }
  }
}

export function extractCuratedHistory(
  comprehensiveHistory: Content[],
  cache?: { lastHistoryHash: string; curatedHistory: Content[] },
): Content[] {
  if (comprehensiveHistory === undefined || comprehensiveHistory.length === 0) {
    return [];
  }
  // ... implementation
}`,
        metadata: {
          file: {
            path: '/media/seikarii/Nvme/gemini-cli/packages/core/src/utils/history.ts',
            extension: '.ts',
          },
        },
      },
    ];

    const result = await this.ragService.indexContent(sources);
    console.log(`✅ Indexed ${result.totalChunks} chunks from ${result.successfulSources} sources`);
    
    if (result.errors.length > 0) {
      console.log('⚠️ Indexing errors:', result.errors);
    }
  }

  /**
   * Example of retrieving context for a user query.
   */
  async enhanceQuery(userQuery: string): Promise<string> {
    console.log(`🔍 Enhancing query: "${userQuery}"`);

    const ragQuery: RAGQuery = {
      text: userQuery,
      type: this.detectQueryType(userQuery),
      maxResults: 5,
      includeContext: true,
      filters: {
        chunkTypes: [
          ChunkType.CODE_CLASS,
          ChunkType.CODE_FUNCTION,
          ChunkType.CONVERSATION,
          ChunkType.DOCUMENTATION,
        ],
      },
    };

    const context = await this.ragService.enhanceQuery(ragQuery, {
      maxTokens: 4000,
      includeDependencies: true,
      includeDocumentation: true,
      compressOlder: true,
    });

    console.log(`📄 Retrieved context: ${context.tokenCount} tokens from ${context.sourceChunks.length} chunks`);

    // Create enhanced prompt
    const enhancedPrompt = this.createEnhancedPrompt(userQuery, context.content);
    return enhancedPrompt;
  }

  /**
   * Simple query type detection.
   */
  private detectQueryType(query: string): QueryType {
    const queryLower = query.toLowerCase();
    
    if (queryLower.includes('create') || queryLower.includes('generate') || queryLower.includes('write')) {
      return QueryType.CODE_GENERATION;
    }
    if (queryLower.includes('explain') || queryLower.includes('what is') || queryLower.includes('how does')) {
      return QueryType.CODE_EXPLANATION;
    }
    if (queryLower.includes('debug') || queryLower.includes('error') || queryLower.includes('fix')) {
      return QueryType.DEBUGGING;
    }
    if (queryLower.includes('api') || queryLower.includes('usage') || queryLower.includes('how to use')) {
      return QueryType.API_USAGE;
    }
    if (queryLower.includes('best practice') || queryLower.includes('recommend')) {
      return QueryType.BEST_PRACTICES;
    }
    
    return QueryType.GENERAL_QUESTION;
  }

  /**
   * Create an enhanced prompt with retrieved context.
   */
  private createEnhancedPrompt(userQuery: string, retrievedContext: string): string {
    return `You are an expert programming assistant with access to relevant context from the codebase and previous conversations.

## Retrieved Context
${retrievedContext}

## User Query
${userQuery}

## Instructions
Based on the retrieved context above, provide a helpful and accurate response. If the context contains relevant code examples or previous explanations, reference them in your answer. If the context doesn't contain sufficient information to answer the query, clearly state that and provide general guidance.

Please ensure your response is:
1. Accurate and based on the provided context
2. Helpful and actionable
3. Includes relevant code examples when appropriate
4. References the source context when relevant

Response:`;
  }

  /**
   * Example workflow combining indexing and retrieval.
   */
  async demonstrateWorkflow(): Promise<void> {
    console.log('\n🎯 RAG System Demonstration\n');

    // Initialize the system
    await this.initialize();

    // Example queries
    const exampleQueries = [
      'How do I create a new chat message in the GeminiChat class?',
      'What is the validateHistory function used for?',
      'How do I create a TypeScript class?',
      'Show me examples of error handling in the codebase',
    ];

    for (const query of exampleQueries) {
      console.log(`\n${'='.repeat(60)}`);
      try {
        const enhancedPrompt = await this.enhanceQuery(query);
        console.log('📝 Enhanced Prompt Preview:');
        console.log(enhancedPrompt.substring(0, 300) + '...\n');
      } catch (error) {
        console.error(`❌ Failed to enhance query: ${error}`);
      }
    }

    // Get metrics
    const metrics = this.ragService.getMetrics();
    console.log('\n📊 RAG System Metrics:');
    console.log(`- Total queries: ${metrics.totalQueries}`);
    console.log(`- Total chunks indexed: ${metrics.totalChunksIndexed}`);
    console.log(`- Average retrieval time: ${metrics.averageRetrievalTime.toFixed(2)}ms`);
    console.log(`- Cache hit rate: ${(metrics.cacheHitRate * 100).toFixed(1)}%`);

    await this.ragService.shutdown();
    console.log('\n✅ RAG system demonstration completed!\n');
  }
}

/**
 * Run the example if this file is executed directly.
 */
if (require.main === module) {
  (async () => {
    try {
      // Create a mock config for demonstration
      const config = {} as Config; // In real usage, pass actual Config instance
      
      const example = new RAGIntegrationExample(config);
      await example.demonstrateWorkflow();
    } catch (error) {
      console.error('❌ Example failed:', error);
      process.exit(1);
    }
  })();
}
