
import path from 'path';
import { ragService } from '@google/gemini-cli-core/rag/ragService.js';
import { fileDiscoveryService } from '@google/gemini-cli-core/services/fileDiscoveryService.js';
import { ollamaEmbedding } from '@google/gemini-cli-core/rag/embeddingServices/ollamaEmbedding.js';
import { memoryVectorStore } from '@google/gemini-cli-core/rag/vectorStores/memoryVectorStore.js';
import { CodeChunker } from '@google/gemini-cli-core/rag/chunking/codeChunking.js';
import { MarkdownChunker } from '@google/gemini-cli-core/rag/chunking/markdownChunking.js';
import { logger } from '@google/gemini-cli-core/rag/logger.js';

const projectRoot = path.resolve(__dirname, '..');

async function buildIndex() {
  logger.info('Starting RAG index build process...');
  console.time('Total Indexing Time');

  const vectorStore = memoryVectorStore();
  // Using Ollama by default. You can swap this with geminiEmbedding() if you prefer.
  const embeddingService = ollamaEmbedding(); 

  const rag = ragService({
    vectorStore,
    embeddingService,
    chunkers: {
      '.ts': new CodeChunker({ maxChunkSize: 1024 }),
      '.js': new CodeChunker({ maxChunkSize: 1024 }),
      '.md': new MarkdownChunker({ maxChunkSize: 800 }),
    },
  });

  const files = await fileDiscoveryService.discover({
    basePath: projectRoot,
    patterns: ['packages/**/*.ts', 'packages/**/*.js', 'packages/**/*.md'],
    exclusions: ['**/node_modules/**', '**/*.test.ts', '**/*.spec.ts', '**/__tests__/**'],
  });

  logger.info(`Found ${files.length} files to index. This may take a few minutes...`);

  await rag.index(files);

  const stats = await vectorStore.getStats();
  logger.info('Index build complete.');
  logger.info(`Total vectors in store: ${stats.vectorCount}`);

  console.timeEnd('Total Indexing Time');
}

buildIndex().catch((error) => {
  logger.error('Failed to build RAG index:', error);
  process.exit(1);
});
