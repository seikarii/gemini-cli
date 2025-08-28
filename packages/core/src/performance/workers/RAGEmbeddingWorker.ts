/**
 * @fileoverview Enhanced embedding worker that integrates with existing RAG system
 * Uses the mature RAG embedding service instead of simple TF-IDF implementation
 */

import { parentPort } from 'worker_threads';
import type { 
  IWorkerMessage, 
  IEmbeddingWorkerData,
  IEmbeddingWorkerResult 
} from '../interfaces/WorkerInterfaces';
import { RAGGeminiEmbeddingService } from '../../rag/embeddingServices/geminiEmbeddingService';
import { RAGLogger } from '../../rag/logger';
import type { Config } from '@google/gemini-cli-core';
import type { RAGConfig } from '../../rag/types';

interface RAGWorkerInitData {
  config: Config;
  ragConfig: RAGConfig;
}

/**
 * RAG-powered embedding worker that leverages the existing mature embedding infrastructure
 */
class RAGEmbeddingWorker {
  private embeddingService: RAGGeminiEmbeddingService | null = null;
  private logger: RAGLogger | null = null;

  constructor() {
    this.setupMessageHandler();
  }

  private setupMessageHandler(): void {
    if (!parentPort) {
      throw new Error('Worker must be run in worker thread context');
    }

    parentPort.on('message', async (message: IWorkerMessage<IEmbeddingWorkerData>) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        this.sendError(message.id, error);
      }
    });
  }

  private async handleMessage(message: IWorkerMessage<IEmbeddingWorkerData>): Promise<void> {
    const { id, type, data } = message;

    switch (type) {
      case 'init':
        await this.handleInit(id, data as RAGWorkerInitData);
        break;
      case 'embed_text':
        await this.handleEmbedText(id, data);
        break;
      case 'embed_batch':
        await this.handleEmbedBatch(id, data);
        break;
      case 'shutdown':
        await this.handleShutdown(id);
        break;
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  }

  private async handleInit(id: string, initData: RAGWorkerInitData): Promise<void> {
    try {
      // Initialize logger
      this.logger = new RAGLogger(initData.config, initData.ragConfig);
      
      // Initialize RAG embedding service
      this.embeddingService = new RAGGeminiEmbeddingService(
        initData.config,
        initData.ragConfig,
        this.logger
      );

      this.logger.info('RAG embedding worker initialized successfully', {
        model: initData.ragConfig.embedding.model,
        dimension: initData.ragConfig.embedding.dimension
      });

      this.sendResult(id, { 
        success: true,
        message: 'RAG embedding worker initialized'
      });
    } catch (error) {
      this.sendError(id, error, 'Failed to initialize RAG embedding worker');
    }
  }

  private async handleEmbedText(id: string, data: IEmbeddingWorkerData): Promise<void> {
    if (!this.embeddingService) {
      throw new Error('Embedding service not initialized');
    }

    try {
      const { text } = data;
      
      if (!text || text.trim().length === 0) {
        throw new Error('Text input is required and cannot be empty');
      }

      // Use the mature RAG embedding service
      const embedding = await this.embeddingService.generateEmbedding(text);
      
      const result: IEmbeddingWorkerResult = {
        embedding,
        dimension: embedding.length,
        model: this.embeddingService.getModelInfo().name,
        cached: false, // The RAG service handles caching internally
        processingTime: 0 // Could add timing if needed
      };

      this.sendResult(id, result);
    } catch (error) {
      this.sendError(id, error, 'Failed to generate embedding');
    }
  }

  private async handleEmbedBatch(id: string, data: IEmbeddingWorkerData): Promise<void> {
    if (!this.embeddingService) {
      throw new Error('Embedding service not initialized');
    }

    try {
      const { texts } = data;
      
      if (!texts || !Array.isArray(texts) || texts.length === 0) {
        throw new Error('Texts array is required and cannot be empty');
      }

      // Validate all texts
      const validTexts = texts.filter(text => text && text.trim().length > 0);
      if (validTexts.length === 0) {
        throw new Error('All texts are empty or invalid');
      }

      // Use the mature RAG embedding service for batch processing
      const embeddings = await this.embeddingService.generateEmbeddings(validTexts);
      
      const result: IEmbeddingWorkerResult = {
        embeddings,
        dimension: embeddings[0]?.length || 0,
        model: this.embeddingService.getModelInfo().name,
        cached: false, // The RAG service handles caching internally
        processingTime: 0,
        batchSize: embeddings.length
      };

      this.sendResult(id, result);
    } catch (error) {
      this.sendError(id, error, 'Failed to generate batch embeddings');
    }
  }

  private async handleShutdown(id: string): Promise<void> {
    try {
      if (this.logger) {
        this.logger.info('RAG embedding worker shutting down');
      }

      // Clean up resources
      this.embeddingService = null;
      this.logger = null;

      this.sendResult(id, { 
        success: true,
        message: 'RAG embedding worker shutdown complete'
      });

      // Exit process
      process.exit(0);
    } catch (error) {
      this.sendError(id, error, 'Failed to shutdown worker');
    }
  }

  private sendResult(id: string, result: IEmbeddingWorkerResult | {success: boolean; message: string}): void {
    if (!parentPort) return;
    
    parentPort.postMessage({
      id,
      type: 'result',
      data: result,
      timestamp: Date.now()
    });
  }

  private sendError(id: string, error: unknown, context?: string): void {
    if (!parentPort) return;

    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : undefined;
    
    if (this.logger) {
      this.logger.error(`RAG embedding worker error: ${context || errorMessage}`, {
        error: errorMessage,
        stack: errorStack,
        context
      });
    }

    parentPort.postMessage({
      id,
      type: 'error',
      data: {
        message: context ? `${context}: ${errorMessage}` : errorMessage,
        stack: errorStack,
        context
      },
      timestamp: Date.now()
    });
  }
}

// Initialize worker if this is the main module
if (require.main === module) {
  new RAGEmbeddingWorker();
}

export default RAGEmbeddingWorker;
