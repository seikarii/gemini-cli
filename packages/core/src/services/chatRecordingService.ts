/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { type Config } from '../config/config.js';
import { type Status } from '../core/coreToolScheduler.js';
import { type ThoughtSummary } from '../core/turn.js';
import { getProjectHash } from '../utils/paths.js';
import path from 'node:path';
import fs from 'node:fs';
import { randomUUID } from 'node:crypto';
import { PartListUnion } from '@google/genai';

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
  
  // Context compression configuration
  private compressionConfig: ContextCompressionConfig = {
    maxContextTokens: 35000,
    preserveRecentMessages: 8,
    compressionRatio: 0.3,
    keywordPreservation: true,
    summarizeToolCalls: true,
  };

  constructor(config: Config) {
    this.config = config;
    this.sessionId = config.getSessionId();
    this.projectHash = getProjectHash(config.getProjectRoot());
    
    // Allow configuration override via environment or config
    this.updateCompressionConfig();
  }

  /**
   * Updates compression configuration based on environment variables or config settings.
   */
  private updateCompressionConfig(): void {
    // Allow runtime configuration
    const envMaxTokens = process.env['GEMINI_MAX_CONTEXT_TOKENS'];
    if (envMaxTokens) {
      this.compressionConfig.maxContextTokens = parseInt(envMaxTokens, 10);
    }
    
    const envPreserveMessages = process.env['GEMINI_PRESERVE_RECENT_MESSAGES'];
    if (envPreserveMessages) {
      this.compressionConfig.preserveRecentMessages = parseInt(envPreserveMessages, 10);
    }
  }

  /**
   * Estimates token count for a message (rough approximation).
   */
  private estimateTokenCount(text: string): number {
    // Rough estimation: ~4 characters per token for English text
    return Math.ceil(text.length / 4);
  }

  /**
   * Compresses old context intelligently to stay under token limits.
   */
  private compressContextIfNeeded(conversation: EnhancedConversationRecord): EnhancedConversationRecord {
    const totalMessages = conversation.messages.length;
    
    // Don't compress if we have few messages
    if (totalMessages <= this.compressionConfig.preserveRecentMessages) {
      return conversation;
    }

    // Estimate current context size
    let totalTokens = 0;
    conversation.messages.forEach(msg => {
      totalTokens += this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        msg.toolCalls.forEach(tc => {
          totalTokens += this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += this.estimateTokenCount(JSON.stringify(tc.result));
          }
        });
      }
    });

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

    // Create or update compressed context
    const newCompressedContext = this.createCompressedContext(oldMessages, conversation.compressedContext);

    return {
      ...conversation,
      messages: recentMessages,
      compressedContext: newCompressedContext,
      lastCompressionTime: new Date().toISOString(),
      compressionConfig: this.compressionConfig,
    };
  }

  /**
   * Creates intelligent compressed representation of old messages.
   */
  private createCompressedContext(
    messages: MessageRecord[], 
    existingCompressed?: CompressedContext
  ): CompressedContext {
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

    // Merge with existing compressed context if it exists
    let existingSummary = '';
    let existingKeyPoints: string[] = [];
    let existingToolSummary = '';
    let totalOriginalTokens = 0;

    if (existingCompressed) {
      existingSummary = existingCompressed.summary;
      existingKeyPoints = existingCompressed.keyPoints;
      existingToolSummary = existingCompressed.toolCallsSummary;
      totalOriginalTokens = existingCompressed.originalTokens;
    }

    // Calculate original tokens for new messages
    let newOriginalTokens = 0;
    messages.forEach(msg => {
      newOriginalTokens += this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        msg.toolCalls.forEach(tc => {
          newOriginalTokens += this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            newOriginalTokens += this.estimateTokenCount(JSON.stringify(tc.result));
          }
        });
      }
    });

    // Extract key information
    const keyPoints = this.extractKeyPoints(messages);
    const toolCallsSummary = this.summarizeToolCalls(messages);
    const conversationSummary = this.createConversationSummary(messages);

    // Merge with existing data
    const mergedKeyPoints = [...existingKeyPoints, ...keyPoints].slice(-20); // Keep last 20 key points
    const mergedToolSummary = existingToolSummary ? 
      `${existingToolSummary}; ${toolCallsSummary}` : toolCallsSummary;
    const mergedSummary = existingSummary ? 
      `${existingSummary}. ${conversationSummary}` : conversationSummary;

    // Calculate time span
    const timespan = this.calculateTimespan(messages, existingCompressed);

    // Estimate compressed tokens
    const compressedContent = `${mergedSummary} ${mergedKeyPoints.join('. ')} ${mergedToolSummary}`;
    const compressedTokens = this.estimateTokenCount(compressedContent);

    return {
      summary: mergedSummary,
      keyPoints: mergedKeyPoints,
      toolCallsSummary: mergedToolSummary,
      timespan,
      messageCount: messages.length + (existingCompressed?.messageCount || 0),
      originalTokens: totalOriginalTokens + newOriginalTokens,
      compressedTokens,
    };
  }

  /**
   * Extracts key points from messages using heuristics.
   */
  private extractKeyPoints(messages: MessageRecord[]): string[] {
    const keyPoints: string[] = [];
    
    messages.forEach(msg => {
      // Extract important patterns
      const content = msg.content;
      
      // Look for error patterns
      if (content.match(/error|failed|exception|problem/i)) {
        const errorContext = content.substring(0, 150);
        keyPoints.push(`Error context: ${errorContext}`);
      }
      
      // Look for file operations
      if (content.match(/created?|modified?|deleted?|file|path/i)) {
        const fileOp = content.substring(0, 100);
        keyPoints.push(`File operation: ${fileOp}`);
      }
      
      // Look for configuration or setup
      if (content.match(/config|setup|install|initialize/i)) {
        const setupContext = content.substring(0, 120);
        keyPoints.push(`Setup: ${setupContext}`);
      }
      
      // Look for code changes
      if (content.match(/function|class|interface|import|export/i)) {
        const codeContext = content.substring(0, 100);
        keyPoints.push(`Code: ${codeContext}`);
      }
    });
    
    return keyPoints.slice(-10); // Keep last 10 key points from this batch
  }

  /**
   * Creates a summary of tool call patterns and outcomes.
   */
  private summarizeToolCalls(messages: MessageRecord[]): string {
    const toolStats: Record<string, { count: number; success: number; errors: number }> = {};
    
    messages.forEach(msg => {
      if (msg.type === 'gemini' && msg.toolCalls) {
        msg.toolCalls.forEach(tc => {
          if (!toolStats[tc.name]) {
            toolStats[tc.name] = { count: 0, success: 0, errors: 0 };
          }
          toolStats[tc.name].count++;
          
          if (tc.status === 'success') {
            toolStats[tc.name].success++;
          } else if (tc.status === 'error') {
            toolStats[tc.name].errors++;
          }
        });
      }
    });
    
    const summary = Object.entries(toolStats)
      .map(([tool, stats]) => `${tool}(${stats.count}, ${stats.success}✓, ${stats.errors}✗)`)
      .join(', ');
      
    return summary ? `Tools used: ${summary}` : '';
  }

  /**
   * Creates a high-level summary of the conversation flow.
   */
  private createConversationSummary(messages: MessageRecord[]): string {
    if (messages.length === 0) return '';
    
    const userMessages = messages.filter(m => m.type === 'user').length;
    const geminiMessages = messages.filter(m => m.type === 'gemini').length;
    const withTools = messages.filter(m => m.type === 'gemini' && m.toolCalls?.length).length;
    
    let summary = `Conversation: ${userMessages} user msgs, ${geminiMessages} assistant msgs`;
    if (withTools > 0) {
      summary += `, ${withTools} with tools`;
    }
    
    // Try to extract main topics
    const allContent = messages.map(m => m.content).join(' ');
    const topics: string[] = [];
    
    // Simple topic extraction heuristics
    if (allContent.match(/error|bug|fix|problem/i)) topics.push('debugging');
    if (allContent.match(/file|create|write|read/i)) topics.push('file operations');
    if (allContent.match(/code|function|class|typescript/i)) topics.push('coding');
    if (allContent.match(/test|testing|spec/i)) topics.push('testing');
    if (allContent.match(/config|setup|install/i)) topics.push('configuration');
    
    if (topics.length > 0) {
      summary += `. Topics: ${topics.join(', ')}`;
    }
    
    return summary;
  }

  /**
   * Calculates the time span covered by messages.
   */
  private calculateTimespan(messages: MessageRecord[], existingCompressed?: CompressedContext): { start: string; end: string } {
    if (messages.length === 0) {
      return existingCompressed?.timespan || { start: '', end: '' };
    }
    
    const firstTime = messages[0].timestamp;
    const lastTime = messages[messages.length - 1].timestamp;
    
    let start = firstTime;
    const end = lastTime;
    
    if (existingCompressed?.timespan.start) {
      start = existingCompressed.timespan.start < firstTime ? 
        existingCompressed.timespan.start : firstTime;
    }
    
    return { start, end };
  }

  /**
   * Initializes the chat recording service: creates a new conversation file and associates it with
   * this service instance, or resumes from an existing session if resumedSessionData is provided.
   */
  initialize(resumedSessionData?: ResumedSessionData): void {
    try {
      if (resumedSessionData) {
        // Resume from existing session
        this.conversationFile = resumedSessionData.filePath;
        this.sessionId = resumedSessionData.conversation.sessionId;

        // Update the session ID in the existing file
        this.updateConversation((conversation) => {
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
        fs.mkdirSync(chatsDir, { recursive: true });

        const timestamp = new Date()
          .toISOString()
          .slice(0, 16)
          .replace(/:/g, '-');
        const filename = `session-${timestamp}-${this.sessionId.slice(
          0,
          8,
        )}.json`;
        this.conversationFile = path.join(chatsDir, filename);

        this.writeConversation({
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
      console.error('Error initializing chat recording service:', error);
      throw error;
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
  recordMessage(message: {
    type: ConversationRecordExtra['type'];
    content: string;
    append?: boolean;
  }): void {
    if (!this.conversationFile) return;

    try {
      this.updateConversation((conversation) => {
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
      console.error('Error saving message:', error);
      throw error;
    }
  }

  /**
   * Records a thought from the assistant's reasoning process.
   */
  recordThought(thought: ThoughtSummary): void {
    if (!this.conversationFile) return;

    try {
      this.queuedThoughts.push({
        ...thought,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      if (this.config.getDebugMode()) {
        console.error('Error saving thought:', error);
        throw error;
      }
    }
  }

  /**
   * Updates the tokens for the last message in the conversation (which should be by Gemini).
   */
  recordMessageTokens(tokens: {
    input: number;
    output: number;
    cached: number;
    thoughts?: number;
    tool?: number;
    total: number;
  }): void {
    if (!this.conversationFile) return;

    try {
      this.updateConversation((conversation) => {
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
      console.error('Error updating message tokens:', error);
      throw error;
    }
  }

  /**
   * Adds tool calls to the last message in the conversation (which should be by Gemini).
   */
  recordToolCalls(toolCalls: ToolCallRecord[]): void {
    if (!this.conversationFile) return;

    try {
      this.updateConversation((conversation) => {
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
      console.error('Error adding tool call to message:', error);
      throw error;
    }
  }

  /**
   * Loads up the conversation record from disk.
   */
  private readConversation(): EnhancedConversationRecord {
    try {
      this.cachedLastConvData = fs.readFileSync(this.conversationFile!, 'utf8');
      return JSON.parse(this.cachedLastConvData);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        console.error('Error reading conversation file:', error);
        throw error;
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
  private writeConversation(conversation: EnhancedConversationRecord): void {
    try {
      if (!this.conversationFile) return;
      // Don't write the file yet until there's at least one message.
      if (conversation.messages.length === 0 && !conversation.compressedContext) return;

      // Apply compression if needed before writing
      const compressedConversation = this.compressContextIfNeeded(conversation);

      // Only write the file if this change would change the file.
      const newContent = JSON.stringify(compressedConversation, null, 2);
      if (this.cachedLastConvData !== newContent) {
        compressedConversation.lastUpdated = new Date().toISOString();
        const finalContent = JSON.stringify(compressedConversation, null, 2);
        this.cachedLastConvData = finalContent;
        fs.writeFileSync(this.conversationFile, finalContent);
        
        // Log compression if it occurred
        if (compressedConversation.compressedContext) {
          const ctx = compressedConversation.compressedContext;
          console.log(`[ChatRecording] Compressed ${ctx.messageCount} messages: ${ctx.originalTokens} → ${ctx.compressedTokens} tokens`);
        }
      }
    } catch (error) {
      console.error('Error writing conversation file:', error);
      throw error;
    }
  }

  /**
   * Convenient helper for updating the conversation without file reading and writing and time
   * updating boilerplate.
   */
  private updateConversation(
    updateFn: (conversation: EnhancedConversationRecord) => void,
  ) {
    const conversation = this.readConversation();
    updateFn(conversation);
    this.writeConversation(conversation);
  }

  /**
   * Gets optimized conversation context for LLM consumption.
   * Returns recent messages in full + compressed historical context.
   */
  getOptimizedContext(): {
    compressedContext?: CompressedContext;
    recentMessages: MessageRecord[];
    totalEstimatedTokens: number;
    compressionStats?: {
      originalMessages: number;
      compressedMessages: number;
      tokenReduction: number;
      compressionRatio: number;
    };
  } {
    if (!this.conversationFile) {
      return {
        recentMessages: [],
        totalEstimatedTokens: 0,
      };
    }

    const conversation = this.readConversation();
    
    // Apply compression to get optimized version
    const optimized = this.compressContextIfNeeded(conversation);
    
    // Calculate statistics
    let totalTokens = 0;
    optimized.messages.forEach(msg => {
      totalTokens += this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        msg.toolCalls.forEach(tc => {
          totalTokens += this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += this.estimateTokenCount(JSON.stringify(tc.result));
          }
        });
      }
    });

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
  forceCompression(): void {
    if (!this.conversationFile) return;
    
    this.updateConversation((conversation) => {
      // Temporarily lower the threshold to force compression
      const originalConfig = { ...this.compressionConfig };
      this.compressionConfig.maxContextTokens = 1000; // Very low threshold
      
      const compressed = this.compressContextIfNeeded(conversation);
      
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
  getCompressionStats(): {
    isCompressed: boolean;
    totalMessages: number;
    recentMessages: number;
    compressedMessages: number;
    estimatedTokens: number;
    compressionRatio?: number;
    lastCompressionTime?: string;
  } {
    if (!this.conversationFile) {
      return {
        isCompressed: false,
        totalMessages: 0,
        recentMessages: 0,
        compressedMessages: 0,
        estimatedTokens: 0,
      };
    }

    const conversation = this.readConversation();
    const optimized = this.getOptimizedContext();
    
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
  deleteSession(sessionId: string): void {
    try {
      const chatsDir = path.join(
        this.config.storage.getProjectTempDir(),
        'chats',
      );
      const sessionPath = path.join(chatsDir, `${sessionId}.json`);
      fs.unlinkSync(sessionPath);
    } catch (error) {
      console.error('Error deleting session:', error);
      throw error;
    }
  }
}
