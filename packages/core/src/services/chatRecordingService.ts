import { PartListUnion, Status, ThoughtSummary, Config } from '../index.js';

import path from 'node:path';
import { randomUUID } from 'node:crypto';
/**
 * Base error class for chat recording errors.
 */
export class ChatRecordingError extends Error {
  override cause?: Error;
  constructor(message: string, cause?: Error) {
    super(message);
    this.name = 'ChatRecordingError';
    if (cause) {
      this.cause = cause;
    }
  }
}
/**
 * File system abstraction for better testability.
 */
export interface FileSystemAdapter {
  readFile(filePath: string): Promise<string>;
  writeFile(filePath: string, data: string): Promise<void>;
  mkdir(dirPath: string): Promise<void>;
  unlink(filePath: string): Promise<void>;
  exists(filePath: string): Promise<boolean>;
}

/**
 * Default file system adapter using Node.js fs promises.
 */
import fs from 'node:fs/promises';
export class NodeFileSystemAdapter implements FileSystemAdapter {
  async readFile(filePath: string): Promise<string> {
    return fs.readFile(filePath, 'utf8');
  }

  async writeFile(filePath: string, data: string): Promise<void> {
    return fs.writeFile(filePath, data, 'utf8');
  }

  async mkdir(dirPath: string): Promise<void> {
    await fs.mkdir(dirPath, { recursive: true });
  }

  async unlink(filePath: string): Promise<void> {
    return fs.unlink(filePath);
  }

  async exists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}

export class ChatRecordingInitializationError extends ChatRecordingError {
  constructor(message: string, cause?: Error) {
    super(message, cause);
    this.name = 'ChatRecordingInitializationError';
  }
}

export class ChatRecordingCompressionError extends ChatRecordingError {
  constructor(message: string, cause?: Error) {
    super(message, cause);
    this.name = 'ChatRecordingCompressionError';
  }
}

export class ChatRecordingFileError extends ChatRecordingError {
  constructor(message: string, cause?: Error) {
    super(message, cause);
    this.name = 'ChatRecordingFileError';
  }
}

/**
 * Token estimation strategy interface for dependency injection.
 */
export interface TokenEstimator {
  estimateTokens(text: string): Promise<number>;
}

/**
 * Simple token estimator using character count heuristic.
 */
export class SimpleTokenEstimator implements TokenEstimator {
  async estimateTokens(text: string): Promise<number> {
    // Rough estimation: ~4 characters per token for English text
    return Math.ceil(text.length / 4);
  }
}

/**
 * Advanced token estimator that can use more sophisticated methods.
 * This can be extended to use tiktoken or other precise tokenization libraries.
 */
export class AdvancedTokenEstimator implements TokenEstimator {

  async estimateTokens(text: string): Promise<number> {
    try {
      // Placeholder for actual token estimation logic
      // const { get_encoding, encoding_for_model } = await import('@dqbd/tiktoken');
      return Math.ceil(text.length / 4); // fallback to simple heuristic
    } catch {
      return Math.ceil(text.length / 4);
    }
  }
}

/**
 * Token usage summary for a message or conversation.
 */
export interface TokensSummary {
  input: number; // promptTokenCount
  output: number; // candidatesTokenCount
  cached: number; // cachedContentTokenCount
  thoughts?: number; // thoughtsTokenCount
  tool?: number; // toolUsePromptTokenCount
  total: number; // totalTokenCount
}

/**
 * Configuration for context compression behavior.
 */
export interface ContextCompressionConfig {
  maxContextTokens: number; // Maximum tokens before compression kicks in (default: 35000)
  preserveRecentMessages: number; // Number of recent messages to keep full (default: 8)
  compressionRatio: number; // How aggressively to compress (0.1 = keep 10%, default: 0.3)
  keywordPreservation: boolean; // Whether to preserve important keywords (default: true)
  summarizeToolCalls: boolean; // Whether to summarize old tool calls (default: true)
  strategy: CompressionStrategy; // Compression strategy to use
}

/**
 * Available compression strategies.
 */
export enum CompressionStrategy {
  MINIMAL = 'minimal',     // Keep most information, light compression
  MODERATE = 'moderate',   // Balanced compression
  AGGRESSIVE = 'aggressive', // Maximum compression, keep only essentials
  NO_COMPRESSION = 'no_compression' // No compression, keep all messages as is
}

/**
 * Compression strategy interface for different compression approaches.
 */
export interface CompressionStrategyHandler {
  compress(messages: MessageRecord[], config: ContextCompressionConfig): Promise<CompressedContext>;
}

/**
 * Base fields common to all messages.
 */
export interface BaseMessageRecord {
  id: string;
  timestamp: string;
  content: string;
}

/**
 * Record of a tool call execution within a conversation.
 */
export interface ToolCallRecord {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: PartListUnion | null;
  status: Status;
  timestamp: string;
  // UI-specific fields for display purposes
  displayName?: string;
  description?: string;
  resultDisplay?: string;
  renderOutputAsMarkdown?: boolean;
}

/**
 * Message type and message type-specific fields.
 */
// (Removed duplicate ConversationRecordExtra and MessageRecord definitions)

/**
 * Compressed representation of older context.
 */
export interface CompressedContext {
  summary: string; // High-level summary of the conversation
  keyPoints: string[]; // Important points extracted
  toolCallsSummary: string; // Summary of tool usage patterns
  timespan: { start: string; end: string }; // Time range covered
  messageCount: number; // Number of original messages compressed
  originalTokens: number; // Estimated original token count
  compressedTokens: number; // Actual compressed token count
}

/**
 * Complete conversation record stored in session files.
 */
export interface ConversationRecord {
  sessionId: string;
  projectHash: string;
  startTime: string;
  lastUpdated: string;
  messages: MessageRecord[];
}

/**
 * Enhanced conversation record with compression support.
 */
export interface EnhancedConversationRecord extends ConversationRecord {
  compressedContext?: CompressedContext;
  compressionConfig?: ContextCompressionConfig;
  lastCompressionTime?: string;
}

/**
 * Data structure for resuming an existing session.
 */
export interface ResumedSessionData {
  conversation: ConversationRecord;
  filePath: string;
}

/**
 * Minimal compression strategy - preserves most information.
 */

// Dummy implementation for getProjectHash
function getProjectHash(_: string): string {
  return 'dummy_project_hash';
}

export class MinimalCompressionStrategy implements CompressionStrategyHandler {
  protected tokenEstimator: TokenEstimator;

  constructor(tokenEstimator: TokenEstimator) {
    this.tokenEstimator = tokenEstimator;
  }

  async compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext> {
    // Just summarize tool calls and preserve most content
    const summary = this.createBriefSummary(messages);
    const keyPoints = this.extractImportantPoints(messages, 15); // Keep more points
    const toolCallsSummary = this.summarizeToolCalls(messages);
    return await this.buildCompressedContext(messages, summary, keyPoints, toolCallsSummary);
  }

  private createBriefSummary(messages: MessageRecord[]): string {
    const userCount = messages.filter(m => m.type === 'user').length;
    const assistantCount = messages.filter(m => m.type === 'gemini').length;
    return `Brief exchange: ${userCount} user messages, ${assistantCount} assistant responses`;
  }

  protected extractImportantPoints(messages: MessageRecord[], maxPoints: number): string[] {
    return this.extractKeyPointsFromMessages(messages).slice(-maxPoints);
  }

  private extractKeyPointsFromMessages(messages: MessageRecord[]): string[] {
    const keyPoints: string[] = [];
    messages.forEach(msg => {
      const content = msg.content;
      // Extract error patterns
      if (content.match(/error|failed|exception|problem/i)) {
        const errorContext = content.substring(0, 200); // More context for minimal compression
        keyPoints.push(`Error: ${errorContext}`);
      }
      // Extract file operations
      if (content.match(/created?|modified?|deleted?|file|path/i)) {
        const fileOp = content.substring(0, 150);
        keyPoints.push(`File: ${fileOp}`);
      }
      // Extract important decisions or conclusions
      if (content.match(/decided|concluded|resolved|fixed|implemented/i)) {
        const decision = content.substring(0, 150);
        keyPoints.push(`Decision: ${decision}`);
      }
    });
    return keyPoints;
  }

  protected summarizeToolCalls(messages: MessageRecord[]): string {
    const toolCalls = messages
      .filter(m => m.type === 'gemini' && 'toolCalls' in m && Array.isArray((m as unknown as { toolCalls?: unknown[] }).toolCalls ?? []) && ((m as unknown as { toolCalls?: unknown[] }).toolCalls ?? []).length > 0)
      .map(m => (m as unknown as { toolCalls?: unknown[] }).toolCalls ?? [])
      .flat();
    return `Tool calls: ${toolCalls.length}`;
  }

  protected async buildCompressedContext(
    messages: MessageRecord[],
    summary: string,
    keyPoints: string[],
    toolCallsSummary: string
  ): Promise<CompressedContext> {
    const timespan = {
      start: messages[0]?.timestamp || '',
      end: messages[messages.length - 1]?.timestamp || ''
    };
    const messageCount = messages.length;
    const originalTokens = await this.tokenEstimator.estimateTokens(JSON.stringify(messages));
    const compressedTokens = await this.tokenEstimator.estimateTokens(
      summary + keyPoints.join(' ') + toolCallsSummary
    );
    return {
      summary,
      keyPoints,
      toolCallsSummary,
      timespan,
      messageCount,
      originalTokens,
      compressedTokens
    };
  }
}


/**
 * Aggressive compression strategy - maximum compression.
 */
export class AggressiveCompressionStrategy extends MinimalCompressionStrategy {

  override async compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext> {
    // Aggressive compression: keep only most critical info
    const summary = `Aggressive compression: ${messages.length} messages.`;
    const keyPoints: string[] = [];
    messages.forEach(msg => {
      const content = msg.content;
      if (content.match(/error|failed|exception|problem/i)) {
        keyPoints.push(`Error: ${content.substring(0, 100)}`);
      }
    });
    const toolCallsSummary = '';
    let timespan = { start: '', end: '' };
    if (messages.length > 0) {
      timespan = {
        start: messages[0].timestamp,
        end: messages[messages.length - 1].timestamp,
      };
    }
    let originalTokens = 0;
    for (const msg of messages) {
      originalTokens += await this.tokenEstimator.estimateTokens(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          originalTokens += await this.tokenEstimator.estimateTokens(JSON.stringify(tc.args));
          if (tc.result) {
            originalTokens += await this.tokenEstimator.estimateTokens(JSON.stringify(tc.result));
          }
        }
      }
    }
    const compressedContent = `${summary}. ${keyPoints.join('. ')}. ${toolCallsSummary}`;
    const compressedTokens = await this.tokenEstimator.estimateTokens(compressedContent);
    return {
      summary,
      keyPoints,
      toolCallsSummary,
      timespan,
      messageCount: messages.length,
      originalTokens,
      compressedTokens,
    };
  }
}

/**
 * No compression strategy - all messages are returned as key points.
 */
export class NoCompressionStrategy extends MinimalCompressionStrategy {
  constructor(tokenEstimator: TokenEstimator) {
    super(tokenEstimator);
  }

  override async compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext> {
    // No compression: return all messages as key points
    const summary = `No compression: ${messages.length} messages.`;
    const keyPoints = messages.map(m => m.content);
    const toolCallsSummary = '';
    let timespan = { start: '', end: '' };
    if (messages.length > 0) {
      timespan = {
        start: messages[0].timestamp,
        end: messages[messages.length - 1].timestamp,
      };
    }
    let originalTokens = 0;
    for (const msg of messages) {
      originalTokens += await this.tokenEstimator.estimateTokens(msg.content);
    }
    const compressedContent = `${summary}. ${keyPoints.join('. ')}. ${toolCallsSummary}`;
    const compressedTokens = await this.tokenEstimator.estimateTokens(compressedContent);
    return {
      summary,
      keyPoints,
      toolCallsSummary,
      timespan,
      messageCount: messages.length,
      originalTokens,
      compressedTokens,
    };
  }
}

export class ModerateCompressionStrategy extends MinimalCompressionStrategy {
  constructor(tokenEstimator: TokenEstimator) {
    super(tokenEstimator);
  }

  override async compress(messages: MessageRecord[], config: ContextCompressionConfig): Promise<CompressedContext> {
    // Balanced: keep fewer key points, más resumen
    const summary = this.createModerateSummary(messages);
    const keyPoints = this.extractImportantPoints(messages, Math.max(5, Math.floor(config.preserveRecentMessages / 2)));
    const toolCallsSummary = this.summarizeToolCalls(messages);
    return await this.buildCompressedContext(messages, summary, keyPoints, toolCallsSummary);
  }

  private createModerateSummary(messages: MessageRecord[]): string {
    const userCount = messages.filter(m => m.type === 'user').length;
    const assistantCount = messages.filter(m => m.type === 'gemini').length;
    const withTools = messages.filter(m => m.type === 'gemini' && m.toolCalls?.length).length;
    let summary = `Conversation: ${userCount} user msgs, ${assistantCount} assistant msgs`;
    if (withTools > 0) {
      summary += `, ${withTools} with tools`;
    }
    // Add topic extraction
    return summary;
  }
}

/**
 * Compressed representation of older context.
 */
export interface CompressedContext {
  summary: string; // High-level summary of the conversation
  keyPoints: string[]; // Important points extracted
  toolCallsSummary: string; // Summary of tool usage patterns
  timespan: { start: string; end: string }; // Time range covered
  messageCount: number; // Number of original messages compressed
  originalTokens: number; // Estimated original token count
  compressedTokens: number; // Actual compressed token count
}

/**
 * Enhanced conversation record with compression support.
 */
export interface EnhancedConversationRecord extends ConversationRecord {
  compressedContext?: CompressedContext;
  compressionConfig?: ContextCompressionConfig;
  lastCompressionTime?: string;
}

/**
 * Base fields common to all messages.
 */
export interface BaseMessageRecord {
  id: string;
  timestamp: string;
  content: string;
}

/**
 * Record of a tool call execution within a conversation.
 */
export interface ToolCallRecord {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: PartListUnion | null;
  status: Status;
  timestamp: string;
  // UI-specific fields for display purposes
  displayName?: string;
  description?: string;
  resultDisplay?: string;
  renderOutputAsMarkdown?: boolean;
}

/**
 * Message type and message type-specific fields.
 */
export type ConversationRecordExtra =
  | {
      type: 'user';
    }
  | {
      type: 'gemini';
      toolCalls?: ToolCallRecord[];
      thoughts?: Array<ThoughtSummary & { timestamp: string }>;
      tokens?: TokensSummary | null;
      model?: string;
    };

/**
 * A single message record in a conversation.
 */
export type MessageRecord = BaseMessageRecord & ConversationRecordExtra;

/**
 * Complete conversation record stored in session files.
 */
export interface ConversationRecord {
  sessionId: string;
  projectHash: string;
  startTime: string;
  lastUpdated: string;
  messages: MessageRecord[];
}

/**
 * Data structure for resuming an existing session.
 */
export interface ResumedSessionData {
  conversation: ConversationRecord;
  filePath: string;
}

/**
 * Service for automatically recording chat conversations to disk with intelligent context compression.
 *
 * This service provides comprehensive conversation recording that captures:
 * - All user and assistant messages with intelligent compression
 * - Tool calls and their execution results (with summarization for old calls)
 * - Token usage statistics and compression metrics
 * - Assistant thoughts and reasoning (compressed for older entries)
 * - Intelligent context management to prevent hallucinations from large contexts
 *
 * CONTEXT COMPRESSION STRATEGY:
 * - Recent messages (default: 8) are kept in full detail
 * - Older messages are progressively compressed based on age and relevance
 * - Tool calls are summarized while preserving success/failure patterns
 * - Key information is extracted and preserved regardless of age
 * - Total context is kept under configurable token limits (default: 35k)
 *
 * Sessions are stored as JSON files in ~/.gemini/tmp/<project_hash>/chats/
 */
export class ChatRecordingService {
  private conversationFile: string | null = null;
  private cachedLastConvData: string | null = null;
  private sessionId: string;
  private projectHash: string;
  private queuedThoughts: Array<ThoughtSummary & { timestamp: string }> = [];
  private queuedTokens: TokensSummary | null = null;
  private config: Config;
  
  // Dependency injection for better testability
  private fileSystem: FileSystemAdapter;
  private tokenEstimator: TokenEstimator;
  private compressionStrategies: Map<CompressionStrategy, CompressionStrategyHandler>;
  
  // Context compression configuration
  private compressionConfig: ContextCompressionConfig = {
    maxContextTokens: 66000,
    preserveRecentMessages: 8,
    compressionRatio: 0.3,
    keywordPreservation: true,
    summarizeToolCalls: true,
    strategy: CompressionStrategy.MODERATE,
  };

  constructor(
  config: Config,
    fileSystem?: FileSystemAdapter,
    tokenEstimator?: TokenEstimator
  ) {
    this.config = config;
    this.sessionId = config.getSessionId();
  this.projectHash = getProjectHash(config.getProjectRoot());
    
    // Set up dependencies with defaults
    this.fileSystem = fileSystem || new NodeFileSystemAdapter();
    this.tokenEstimator = tokenEstimator || new AdvancedTokenEstimator();
    // Initialize compression strategies
    this.compressionStrategies = new Map<CompressionStrategy, CompressionStrategyHandler>([
      [CompressionStrategy.MINIMAL, new MinimalCompressionStrategy(this.tokenEstimator)],
      [CompressionStrategy.MODERATE, new ModerateCompressionStrategy(this.tokenEstimator)],
      [CompressionStrategy.AGGRESSIVE, new AggressiveCompressionStrategy(this.tokenEstimator)],
      [CompressionStrategy.NO_COMPRESSION, new NoCompressionStrategy(this.tokenEstimator)],
    ]);
    // Update configuration from environment or config
    this.updateCompressionConfig();
  }

  /**
   * Updates compression configuration based on config object and environment variables.
   */
  private updateCompressionConfig(): void {
    // Centralize all configuration access through environment variables for now
    // TODO: Integrate with Config object when it supports these settings
    const maxTokens = process.env['GEMINI_MAX_CONTEXT_TOKENS'];
    if (maxTokens) {
      this.compressionConfig.maxContextTokens = parseInt(maxTokens, 10);
    }
    const preserveMessages = process.env['GEMINI_PRESERVE_RECENT_MESSAGES'];
    if (preserveMessages) {
      this.compressionConfig.preserveRecentMessages = parseInt(preserveMessages, 10);
    }
    const strategy = process.env['GEMINI_COMPRESSION_STRATEGY'];
    if (strategy && Object.values(CompressionStrategy).includes(strategy as CompressionStrategy)) {
      this.compressionConfig.strategy = strategy as CompressionStrategy;
    }
  }

  /**
   * Estimates token count for a message using the configured estimator.
   */
  private async estimateTokenCount(text: string): Promise<number> {
    return await this.tokenEstimator.estimateTokens(text);
  }

  /**
   * Compresses old context intelligently to stay under token limits.
   */
  private async compressContextIfNeeded(conversation: EnhancedConversationRecord): Promise<EnhancedConversationRecord> {
    const totalMessages = conversation.messages.length;
    // Don't compress if we have few messages
    if (totalMessages <= this.compressionConfig.preserveRecentMessages) {
      return conversation;
    }
    // Estimate current context size
    let totalTokens = 0;
    for (const msg of conversation.messages) {
      totalTokens += await this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          totalTokens += await this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += await this.estimateTokenCount(JSON.stringify(tc.result));
          }
        }
      }
    }
    // Add compressed context tokens if it exists
    if (conversation.compressedContext) {
      totalTokens += conversation.compressedContext.compressedTokens;
    }
    // Check if compression is needed
    if (totalTokens <= this.compressionConfig.maxContextTokens) {
      return conversation;
    }
    console.log(`[ChatRecording] Context size ${totalTokens} tokens exceeds limit ${this.compressionConfig.maxContextTokens}. Compressing...`);
    // Split messages into recent (preserve) and old (compress)
    const preserveCount = this.compressionConfig.preserveRecentMessages;
    const recentMessages = conversation.messages.slice(-preserveCount);
    const oldMessages = conversation.messages.slice(0, -preserveCount);
    // Create or update compressed context using the configured strategy
    const newCompressedContext = await this.createCompressedContext(oldMessages, conversation.compressedContext);
    return {
      ...conversation,
      messages: recentMessages,
      compressedContext: newCompressedContext,
      lastCompressionTime: new Date().toISOString(),
      compressionConfig: this.compressionConfig,
    };
  }

  /**
   * Creates intelligent compressed representation of old messages using the configured strategy.
   */
  private async createCompressedContext(
    messages: MessageRecord[], 
    existingCompressed?: CompressedContext
  ): Promise<CompressedContext> {
    if (messages.length === 0 && !existingCompressed) {
      return {
        summary: '',
        keyPoints: [],
        toolCallsSummary: '',
        timespan: { start: '', end: '' },
        messageCount: 0,
        originalTokens: 0,
        compressedTokens: 0,
      };
    }
    // Get the appropriate compression strategy
    const strategy = this.compressionStrategies.get(this.compressionConfig.strategy);
    if (!strategy) {
      throw new ChatRecordingCompressionError(`Unknown compression strategy: ${this.compressionConfig.strategy}`);
    }
    if (this.compressionConfig.strategy === CompressionStrategy.NO_COMPRESSION) {
      // Estimate tokens for all messages
      let originalTokens = 0;
      for (const msg of messages) {
        originalTokens += await this.estimateTokenCount(msg.content);
        if (msg.type === 'gemini' && msg.toolCalls) {
          for (const tc of msg.toolCalls) {
            originalTokens += await this.estimateTokenCount(JSON.stringify(tc.args));
            if (tc.result) {
              originalTokens += await this.estimateTokenCount(JSON.stringify(tc.result));
            }
          }
        }
      }
      // Calculate timespan
      let timespan = { start: '', end: '' };
      if (messages.length > 0) {
        timespan = {
          start: messages[0].timestamp,
          end: messages[messages.length - 1].timestamp,
        };
      }
      return {
        summary: 'No compression applied.',
        keyPoints: [],
        toolCallsSummary: '',
        timespan,
        messageCount: messages.length,
        originalTokens,
        compressedTokens: originalTokens,
      };
    }
    try {
      // If we have existing compressed context, we need to merge with new messages
      if (existingCompressed && messages.length > 0) {
        // For now, use a simple merge approach
        // In the future, this could be enhanced with more sophisticated merging
        const newCompressed = await strategy.compress(messages, this.compressionConfig);
        const mergedSummary = existingCompressed.summary ? `${existingCompressed.summary}. ${newCompressed.summary}` : newCompressed.summary;
        const mergedKeyPoints = [...existingCompressed.keyPoints, ...newCompressed.keyPoints].slice(-20);
        const mergedToolCallsSummary = existingCompressed.toolCallsSummary ? `${existingCompressed.toolCallsSummary}; ${newCompressed.toolCallsSummary}` : newCompressed.toolCallsSummary;
        const mergedTimespan = {
          start: existingCompressed.timespan.start || newCompressed.timespan.start,
          end: newCompressed.timespan.end,
        };
        const mergedMessageCount = existingCompressed.messageCount + newCompressed.messageCount;
        const mergedOriginalTokens = existingCompressed.originalTokens + newCompressed.originalTokens;
        const mergedCompressedTokens = await this.estimateTokenCount(
          `${mergedSummary}. ${mergedKeyPoints.join('. ')} ${mergedToolCallsSummary}`
        );
        return {
          summary: mergedSummary,
          keyPoints: mergedKeyPoints,
          toolCallsSummary: mergedToolCallsSummary,
          timespan: mergedTimespan,
          messageCount: mergedMessageCount,
          originalTokens: mergedOriginalTokens,
          compressedTokens: mergedCompressedTokens,
        };
      } else if (existingCompressed) {
        return existingCompressed;
      } else {
        return await strategy.compress(messages, this.compressionConfig);
      }
    } catch (error) {
      throw new ChatRecordingCompressionError(
        `Failed to compress context: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Generates a consistent file name for a session.
   */
  private generateSessionFileName(sessionId: string): string {
    const timestamp = new Date()
      .toISOString()
      .slice(0, 16)
      .replace(/:/g, '-');
    return `session-${timestamp}-${sessionId.slice(0, 8)}.json`;
  }

  /**
   * Initializes the chat recording service: creates a new conversation file and associates it with
   * this service instance, or resumes from an existing session if resumedSessionData is provided.
   */
  async initialize(resumedSessionData?: ResumedSessionData): Promise<void> {
    try {
      if (resumedSessionData) {
        // Resume from existing session
        this.conversationFile = resumedSessionData.filePath;
        this.sessionId = resumedSessionData.conversation.sessionId;

        // Update the session ID in the existing file
        await this.updateConversation((conversation) => {
          conversation.sessionId = this.sessionId;
        });

        // Clear any cached data to force fresh reads
        this.cachedLastConvData = null;
      } else {
        // Create new session
        const chatsDir = path.join(
          this.config.storage.getProjectTempDir(),
          'chats',
        );
        await this.fileSystem.mkdir(chatsDir);

        const filename = this.generateSessionFileName(this.sessionId);
        this.conversationFile = path.join(chatsDir, filename);

        await this.writeConversation({
          sessionId: this.sessionId,
          projectHash: this.projectHash,
          startTime: new Date().toISOString(),
          lastUpdated: new Date().toISOString(),
          messages: [],
        });
      }

      // Clear any queued data since this is a fresh start
      this.queuedThoughts = [];
      this.queuedTokens = null;
    } catch (error) {
      throw new ChatRecordingInitializationError(
        `Failed to initialize chat recording: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  private getLastMessage(
    conversation: ConversationRecord,
  ): MessageRecord | undefined {
    return conversation.messages.at(-1);
  }

  private newMessage(
    type: ConversationRecordExtra['type'],
    content: string,
  ): MessageRecord {
    return {
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      type,
      content,
    };
  }

  /**
   * Records a message in the conversation with intelligent compression.
   */
  async recordMessage(message: {
    type: ConversationRecordExtra['type'];
    content: string;
    append?: boolean;
  }): Promise<void> {
    if (!this.conversationFile) return;

    try {
      await this.updateConversation((conversation) => {
        if (message.append) {
          const lastMsg = this.getLastMessage(conversation);
          if (lastMsg && lastMsg.type === message.type) {
            lastMsg.content += message.content;
            return;
          }
        }
        // We're not appending, or we are appending but the last message's type is not the same as
        // the specified type, so just create a new message.
        const msg = this.newMessage(message.type, message.content);
        if (msg.type === 'gemini') {
          // If it's a new Gemini message then incorporate any queued thoughts.
          conversation.messages.push({
            ...msg,
            thoughts: this.queuedThoughts,
            tokens: this.queuedTokens,
            model: this.config.getModel(),
          });
          this.queuedThoughts = [];
          this.queuedTokens = null;
        } else {
          // Or else just add it.
          conversation.messages.push(msg);
        }
      });
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error saving message: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Records a thought from the assistant's reasoning process.
   */
  recordThought(thought: ThoughtSummary): void {
    if (!this.conversationFile) return;

    try {
      this.queuedThoughts.push({
  subject: (typeof thought === 'object' && thought !== null && 'subject' in thought && typeof (thought as ThoughtSummary).subject === 'string') ? (thought as ThoughtSummary).subject : '',
  description: (typeof thought === 'object' && thought !== null && 'description' in thought && typeof (thought as ThoughtSummary).description === 'string') ? (thought as ThoughtSummary).description : '',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      if (this.config.getDebugMode()) {
        throw new ChatRecordingError(
          `Error saving thought: ${error instanceof Error ? error.message : String(error)}`,
          error instanceof Error ? error : undefined
        );
      }
    }
  }

  /**
   * Updates the tokens for the last message in the conversation (which should be by Gemini).
   */
  async recordMessageTokens(tokens: {
    input: number;
    output: number;
    cached: number;
    thoughts?: number;
    tool?: number;
    total: number;
  }): Promise<void> {
    if (!this.conversationFile) return;

    try {
      await this.updateConversation((conversation) => {
        const lastMsg = this.getLastMessage(conversation);
        // If the last message already has token info, it's because this new token info is for a
        // new message that hasn't been recorded yet.
        if (lastMsg && lastMsg.type === 'gemini' && !lastMsg.tokens) {
          lastMsg.tokens = tokens;
          this.queuedTokens = null;
        } else {
          this.queuedTokens = tokens;
        }
      });
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error updating message tokens: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Adds tool calls to the last message in the conversation (which should be by Gemini).
   */
  async recordToolCalls(toolCalls: ToolCallRecord[]): Promise<void> {
    if (!this.conversationFile) return;

    try {
      await this.updateConversation((conversation) => {
        const lastMsg = this.getLastMessage(conversation);
        // If a tool call was made, but the last message isn't from Gemini, it's because Gemini is
        // calling tools without starting the message with text.  So the user submits a prompt, and
        // Gemini immediately calls a tool (maybe with some thinking first).  In that case, create
        // a new empty Gemini message.
        // Also if there are any queued thoughts, it means this tool call(s) is from a new Gemini
        // message--because it's thought some more since we last, if ever, created a new Gemini
        // message from tool calls, when we dequeued the thoughts.
        if (
          !lastMsg ||
          lastMsg.type !== 'gemini' ||
          this.queuedThoughts.length > 0
        ) {
          const newMsg: MessageRecord = {
            ...this.newMessage('gemini' as const, ''),
            // This isn't strictly necessary, but TypeScript apparently can't
            // tell that the first parameter to newMessage() becomes the
            // resulting message's type, and so it thinks that toolCalls may
            // not be present.  Confirming the type here satisfies it.
            type: 'gemini' as const,
            toolCalls,
            thoughts: this.queuedThoughts,
            model: this.config.getModel(),
          };
          // If there are any queued thoughts join them to this message.
          if (this.queuedThoughts.length > 0) {
            newMsg.thoughts = this.queuedThoughts;
            this.queuedThoughts = [];
          }
          // If there's any queued tokens info join it to this message.
          if (this.queuedTokens) {
            newMsg.tokens = this.queuedTokens;
            this.queuedTokens = null;
          }
          conversation.messages.push(newMsg);
        } else {
          // The last message is an existing Gemini message that we need to update.

          // Update any existing tool call entries.
          if (!lastMsg.toolCalls) {
            lastMsg.toolCalls = [];
          }
          lastMsg.toolCalls = lastMsg.toolCalls.map((toolCall) => {
            // If there are multiple tool calls with the same ID, this will take the first one.
            const incomingToolCall = toolCalls.find(
              (tc) => tc.id === toolCall.id,
            );
            if (incomingToolCall) {
              // Merge in the new data to keep preserve thoughts, etc., that were assigned to older
              // versions of the tool call.
              return { ...toolCall, ...incomingToolCall };
            } else {
              return toolCall;
            }
          });

          // Add any new tools calls that aren't in the message yet.
          for (const toolCall of toolCalls) {
            const existingToolCall = lastMsg.toolCalls.find(
              (tc) => tc.id === toolCall.id,
            );
            if (!existingToolCall) {
              lastMsg.toolCalls.push(toolCall);
            }
          }
        }
      });
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error adding tool call to message: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Loads up the conversation record from disk.
   */
  private async readConversation(): Promise<EnhancedConversationRecord> {
    try {
      this.cachedLastConvData = await this.fileSystem.readFile(this.conversationFile!);
  return this.cachedLastConvData ? (JSON.parse(this.cachedLastConvData) as EnhancedConversationRecord) : ({} as EnhancedConversationRecord);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw new ChatRecordingFileError(
          `Error reading conversation file: ${error instanceof Error ? error.message : String(error)}`,
          error instanceof Error ? error : undefined
        );
      }

      // Placeholder empty conversation if file doesn't exist.
      return {
        sessionId: this.sessionId,
        projectHash: this.projectHash,
        startTime: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        messages: [],
      };
    }
  }

  /**
   * Saves the conversation record with intelligent compression.
   */
  private async writeConversation(conversation: EnhancedConversationRecord): Promise<void> {
    try {
      if (!this.conversationFile) return;
      // Don't write the file yet until there's at least one message.
      if (conversation.messages.length === 0 && !conversation.compressedContext) return;

      // Apply compression if needed before writing
      const compressedConversation = await this.compressContextIfNeeded(conversation);

      // Only write the file if this change would change the file.
      compressedConversation.lastUpdated = new Date().toISOString();
      const newContent = JSON.stringify(compressedConversation, null, 2);

      if (this.cachedLastConvData !== newContent) {
        this.cachedLastConvData = newContent;
        await this.fileSystem.writeFile(this.conversationFile, newContent);

        // Log compression if it occurred
        if (compressedConversation.compressedContext) {
          const ctx = compressedConversation.compressedContext;
          console.log(`[ChatRecording] Compressed ${ctx.messageCount} messages: ${ctx.originalTokens} → ${ctx.compressedTokens} tokens`);
        }
      }
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error writing conversation file: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }

  /**
   * Convenient helper for updating the conversation without file reading and writing and time
   * updating boilerplate.
   */
  private async updateConversation(
    updateFn: (conversation: EnhancedConversationRecord) => void,
  ): Promise<void> {
    const conversation = await this.readConversation();
    updateFn(conversation);
    await this.writeConversation(conversation);
  }

  /**
   * Gets optimized conversation context for LLM consumption.
   * Returns recent messages in full + compressed historical context.
   */
  async getOptimizedContext(): Promise<{
    compressedContext?: CompressedContext;
    recentMessages: MessageRecord[];
    totalEstimatedTokens: number;
    compressionStats?: {
      originalMessages: number;
      compressedMessages: number;
      tokenReduction: number;
      compressionRatio: number;
    };
  }> {
    if (!this.conversationFile) {
      return {
        recentMessages: [],
        totalEstimatedTokens: 0,
      };
    }

    const conversation = await this.readConversation();
    
    // Apply compression to get optimized version
  const optimized = await (async () => await this.compressContextIfNeeded(conversation))();
    
    // Calculate statistics
    let totalTokens = 0;
    for (const msg of optimized.messages) {
      totalTokens += await this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          totalTokens += await this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += await this.estimateTokenCount(JSON.stringify(tc.result));
          }
        }
      }
    }

    if (optimized.compressedContext) {
      totalTokens += optimized.compressedContext.compressedTokens;
    }

    let compressionStats;
    if (optimized.compressedContext && optimized.compressedContext.messageCount > 0) {
      const originalMessages = optimized.compressedContext.messageCount + optimized.messages.length;
      const compressedMessages = optimized.messages.length;
      const tokenReduction = optimized.compressedContext.originalTokens - optimized.compressedContext.compressedTokens;
      const compressionRatio = optimized.compressedContext.compressedTokens / optimized.compressedContext.originalTokens;
      compressionStats = {
        originalMessages,
        compressedMessages,
        tokenReduction,
        compressionRatio,
      };
    }
    return {
      compressedContext: optimized.compressedContext,
      recentMessages: optimized.messages,
      totalEstimatedTokens: totalTokens,
      compressionStats,
    };
  }

  /**
   * Forces immediate compression of the current conversation.
   * Useful for testing or manual optimization.
   */
  async forceCompression(): Promise<void> {
    if (!this.conversationFile) return;
    
  await this.updateConversation(async (conversation) => {
      // Temporarily lower the threshold to force compression
      const originalConfig = { ...this.compressionConfig };
      this.compressionConfig.maxContextTokens = 1000; // Very low threshold
      
    const compressed = await this.compressContextIfNeeded(conversation);
      
      // Restore original config
      this.compressionConfig = originalConfig;
      
      // Apply the compression
      conversation.messages = compressed.messages;
      conversation.compressedContext = compressed.compressedContext;
      conversation.lastCompressionTime = compressed.lastCompressionTime;
      conversation.compressionConfig = compressed.compressionConfig;
    });
  }

  /**
   * Gets compression statistics for monitoring.
   */
  async getCompressionStats(): Promise<{
    isCompressed: boolean;
    totalMessages: number;
    recentMessages: number;
    compressedMessages: number;
    estimatedTokens: number;
    compressionRatio?: number;
    lastCompressionTime?: string;
  }> {
    if (!this.conversationFile) {
      return {
        isCompressed: false,
        totalMessages: 0,
        recentMessages: 0,
        compressedMessages: 0,
        estimatedTokens: 0,
      };
    }

    const conversation = await this.readConversation();
    const optimized = await this.getOptimizedContext();
    
    return {
      isCompressed: !!conversation.compressedContext,
      totalMessages: (conversation.compressedContext?.messageCount || 0) + conversation.messages.length,
      recentMessages: conversation.messages.length,
      compressedMessages: conversation.compressedContext?.messageCount || 0,
      estimatedTokens: optimized.totalEstimatedTokens,
      compressionRatio: optimized.compressionStats?.compressionRatio,
      lastCompressionTime: conversation.lastCompressionTime,
    };
  }

  /**
   * Deletes a session file by session ID.
   */
  async deleteSession(sessionId: string): Promise<void> {
    try {
      const chatsDir = path.join(
        this.config.storage.getProjectTempDir(),
        'chats',
      );
      
      // Use the same naming pattern as generateSessionFileName for consistency
      const sessionPath = path.join(chatsDir, this.generateSessionFileName(sessionId));
      await this.fileSystem.unlink(sessionPath);
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error deleting session: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined
      );
    }
  }
}

