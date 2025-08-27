/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Simple integration test for the RAG system
 * This can be run to verify the RAG system is working correctly.
 */

import { RAGService } from './ragService.js';
import { ConsoleRAGLogger } from './logger.js';
import { ChunkType, QueryType, RAGQuery } from './types.js';
import { Config } from '../config/config.js';

/**
 * Simple test function to verify RAG system functionality
 */
export async function testRAGIntegration(): Promise<boolean> {
  const logger = new ConsoleRAGLogger('[RAG-Test]');
  
  try {
    // Create a mock config
    const mockConfig = {
      getApiKey: () => process.env.GOOGLE_API_KEY || 'test-key',
      getModel: () => 'gemini-1.5-flash',
      isDevelopment: () => true
    } as Config;

    const ragService = new RAGService(mockConfig, logger);
    
    console.log('🔧 Initializing RAG service...');
    await ragService.initialize();
    
    // Test indexing
    console.log('📚 Testing content indexing...');
    const testSources = [
      {
        id: 'test-function',
        type: ChunkType.CODE_FUNCTION,
        content: `
function calculateSum(a: number, b: number): number {
  return a + b;
}

function calculateProduct(a: number, b: number): number {
  return a * b;
}`,
        metadata: {
          file: { path: '/test/math.ts', extension: '.ts' }
        }
      },
      {
        id: 'test-documentation',
        type: ChunkType.DOCUMENTATION,
        content: `
# Math Functions

This module provides basic mathematical operations:

## calculateSum
Adds two numbers together and returns the result.

## calculateProduct
Multiplies two numbers together and returns the result.
`,
        metadata: {
          file: { path: '/test/README.md', extension: '.md' }
        }
      }
    ];

    const indexResult = await ragService.indexContent(testSources);
    console.log(`✅ Indexed ${indexResult.totalChunks} chunks from ${indexResult.successfulSources} sources`);
    
    if (indexResult.errors.length > 0) {
      console.log('⚠️ Indexing errors:', indexResult.errors);
    }

    // Test query enhancement
    console.log('🔍 Testing query enhancement...');
    const testQuery: RAGQuery = {
      text: 'How do I add two numbers together?',
      type: QueryType.CODE_EXPLANATION,
      maxResults: 3,
      includeContext: true,
      filters: {
        chunkTypes: [ChunkType.CODE_FUNCTION, ChunkType.DOCUMENTATION]
      }
    };

    const enhancedResult = await ragService.enhanceQuery(testQuery, {
      maxTokens: 1000,
      includeDependencies: false,
      includeDocumentation: true
    });

    console.log(`📄 Retrieved context: ${enhancedResult.tokenCount} tokens from ${enhancedResult.sourceChunks.length} chunks`);
    console.log('📋 Context preview:', enhancedResult.content.substring(0, 200) + '...');

    // Test metrics
    const metrics = ragService.getMetrics();
    console.log('📊 RAG System Metrics:');
    console.log(`- Total queries: ${metrics.totalQueries}`);
    console.log(`- Total chunks indexed: ${metrics.totalChunksIndexed}`);
    console.log(`- Average retrieval time: ${metrics.averageRetrievalTime.toFixed(2)}ms`);

    // Cleanup
    await ragService.shutdown();
    console.log('✅ RAG system test completed successfully!');
    
    return true;
  } catch (error) {
    console.error('❌ RAG system test failed:', error);
    return false;
  }
}

// Export for use in other tests
export { testRAGIntegration };

// Allow running directly for testing
if (require.main === module) {
  testRAGIntegration()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}
