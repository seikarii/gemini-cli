import { PartListUnion, Status, ThoughtSummary, Config } from '../index.js';
import { Content } from '@google/genai';
import { ContextManager, DualContextConfig, ContextType } from './contextManager.js';

import path from 'node:path';
import { randomUUID } from 'node:crypto';
/**
 * Base error class for chat recording errors.
 */
export class ChatRecordingError extends Error {
  cause?: Error;
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
 * Advanced token estimator with precise Gemini tokenization patterns.
 * This implementation provides more accurate token estimation for Gemini models
 * by considering different text patterns and tokenization rules.
 */
export class AdvancedTokenEstimator implements TokenEstimator {
  async estimateTokens(text: string): Promise<number> {
    try {
      return this.estimateTokensSync(text);
    } catch (error) {
      // Fallback to simple heuristic if anything fails
      console.warn('Token estimation failed, using fallback:', error);
      return Math.ceil(text.length / 4);
    }
  }

  /**
   * Synchronous token estimation for better performance
   */
  estimateTokensSync(text: string): number {
    if (!text || text.length === 0) {
      return 0;
    }

    let tokens = 0;

    // Split text into words, punctuation, and special characters
    const words = text
      .split(/(\s+|[^\w\s]+)/)
      .filter((part) => part.length > 0);

    for (const word of words) {
      if (word.trim().length === 0) {
        // Whitespace - typically 1 token per 4 spaces, but minimum 1
        tokens += Math.max(1, Math.ceil(word.length / 4));
        continue;
      }

      // Check for common patterns that affect tokenization

      // Numbers (often 1-2 tokens regardless of length)

      if (/^\d+$/.test(word)) {
        tokens += word.length > 6 ? 2 : 1;
        continue;
      }

      // URLs and email addresses (typically 3-5 tokens)

      if (/^(https?:\/\/|www\.|\S+@\S+\.\S+)$/.test(word)) {
        tokens += this.estimateUrlTokens(word);
        continue;
      }

      // Code patterns (underscores, dots, slashes)
      if (/[_.//\\]/.test(word)) {
        tokens += this.estimateCodeTokens(word);
        continue;
      }

      // Punctuation marks
      if (/^[^\w\s]$/.test(word)) {
        tokens += 1;
        continue;
      }

      // Regular words - estimate based on length and complexity
      tokens += this.estimateWordTokens(word);
    }

    // Add base tokens for message structure (if this is a complete message)
    if (text.includes('"role"') || text.includes('"content"')) {
      tokens += 4; // Base tokens for JSON structure
    }

    return Math.max(1, tokens);
  }

  private estimateWordTokens(word: string): number {
    const length = word.length;

    // Very short words (1-3 chars) - typically 1 token
    if (length <= 3) {
      return 1;
    }

    // Medium words (4-8 chars) - typically 1 token
    if (length <= 8) {
      return 1;
    }

    // Longer words - may be split into multiple tokens
    // English words: ~4 chars per token on average
    // Technical terms: ~3 chars per token
    const isTechnical = /[A-Z]{2,}|[_.//\\]/.test(word);
    const avgCharsPerToken = isTechnical ? 3 : 4;

    return Math.ceil(length / avgCharsPerToken);
  }

  private estimateCodeTokens(word: string): number {
    // Code identifiers with dots, underscores, slashes
    const parts = word.split(/([_.//\\])/);
    let tokens = 0;

    for (const part of parts) {
      if (part.length === 0) continue;

      if (/[_.//\\]/.test(part)) {
        tokens += 1; // Separators are usually 1 token
      } else {
        tokens += this.estimateWordTokens(part);
      }
    }

    return tokens;
  }

  private estimateUrlTokens(url: string): number {
    // URLs typically tokenize as: protocol + domain + path + query
    const parts = url.split(/[:/.?&=#]/);
    let tokens = 0;

    for (const part of parts) {
      if (part.length > 0) {
        tokens += this.estimateWordTokens(part);
      }
    }

    // Add tokens for separators
    const separatorCount = (url.match(/[:/.?&=#]/g) || []).length;
    tokens += Math.ceil(separatorCount * 0.5);

    return Math.min(tokens, 8); // Cap at 8 tokens for very long URLs
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
  preserveRecentMessages: number; // Number of recent USER messages to keep full (assistant messages are preserved separately)
  compressionRatio: number; // How aggressively to compress (0.1 = keep 10%, default: 0.3)
  keywordPreservation: boolean; // Whether to preserve important keywords (default: true)
  summarizeToolCalls: boolean; // Whether to summarize old tool calls (default: true)
  strategy: CompressionStrategy; // Compression strategy to use
  neverCompressUserMessages: boolean; // Always preserve ALL user messages, never compress them (default: true)
  importanceThreshold: number; // Minimum importance score (0.0-1.0) for message preservation (default: 0.7)
}

/**
 * Available compression strategies.
 */
export enum CompressionStrategy {
  MINIMAL = 'minimal', // Keep most information, light compression
  MODERATE = 'moderate', // Balanced compression
  AGGRESSIVE = 'aggressive', // Maximum compression, keep only essentials
  INTELLIGENT = 'intelligent', // Advanced NLP-based compression
  NO_COMPRESSION = 'no_compression', // No compression, keep all messages as is
}

/**
 * Compression strategy interface for different compression approaches.
 */
export interface CompressionStrategyHandler {
  compress(
    messages: MessageRecord[],
    config: ContextCompressionConfig,
  ): Promise<CompressedContext>;
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

  async compress(
    messages: MessageRecord[],
    _config: ContextCompressionConfig,
  ): Promise<CompressedContext> {
    // Just summarize tool calls and preserve most content
    const summary = this.createBriefSummary(messages);
    const keyPoints = this.extractImportantPoints(messages, 15); // Keep more points
    const toolCallsSummary = this.summarizeToolCalls(messages);
    return await this.buildCompressedContext(
      messages,
      summary,
      keyPoints,
      toolCallsSummary,
    );
  }

  private createBriefSummary(messages: MessageRecord[]): string {
    const userCount = messages.filter((m) => m.type === 'user').length;
    const assistantCount = messages.filter((m) => m.type === 'gemini').length;
    return `Brief exchange: ${userCount} user messages, ${assistantCount} assistant responses`;
  }

  protected extractImportantPoints(
    messages: MessageRecord[],
    maxPoints: number,
  ): string[] {
    return this.extractKeyPointsFromMessages(messages).slice(-maxPoints);
  }

  protected extractKeyPointsFromMessages(messages: MessageRecord[]): string[] {
    const keyPoints: string[] = [];

    // Extract traditional pattern-based points
    messages.forEach((msg) => {
      const content = msg.content;
      // Extract error patterns
      if (content.match(/error|failed|exception|problem/i)) {
        const errorContext = content.substring(0, 200);
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

    // Add advanced NLP-based extraction
    const advancedPoints = this.extractAdvancedKeyPoints(messages);
    keyPoints.push(...advancedPoints);

    return keyPoints;
  }

  /**
   * Advanced key point extraction using NLP techniques
   */
  private extractAdvancedKeyPoints(messages: MessageRecord[]): string[] {
    const keyPoints: string[] = [];

    // Combine all message content for analysis
    const allContent = messages.map((m) => m.content).join(' ');

    // Extract keywords using TF-IDF inspired approach
    const keywords = this.extractKeywords(allContent);
    keywords.slice(0, 5).forEach((keyword) => {
      keyPoints.push(`Keyword: ${keyword}`);
    });

    // Extract important phrases (quoted text, capitalized phrases)
    const phrases = this.extractImportantPhrases(allContent);
    phrases.slice(0, 3).forEach((phrase) => {
      keyPoints.push(`Phrase: ${phrase}`);
    });

    // Extract action items and tasks
    const actions = this.extractActionItems(allContent);
    actions.slice(0, 3).forEach((action) => {
      keyPoints.push(`Action: ${action}`);
    });

    return keyPoints;
  }

  /**
   * Extract keywords using frequency and importance analysis
   */
  protected extractKeywords(text: string): string[] {
    // Simple TF-IDF inspired approach
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter((word) => word.length > 3); // Filter short words

    const wordFreq = new Map<string, number>();
    const totalWords = words.length;

    // Calculate word frequencies
    words.forEach((word) => {
      wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
    });

    // Score words based on frequency and length (longer technical terms get higher scores)
    const scoredWords = Array.from(wordFreq.entries())
      .map(([word, freq]) => ({
        word,
        score:
          (freq / totalWords) *
          Math.log(word.length) *
          (word.match(/[A-Z]/) ? 1.5 : 1), // Boost technical terms
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map((item) => item.word);

    return scoredWords;
  }

  /**
   * Extract important phrases from text
   */
  protected extractImportantPhrases(text: string): string[] {
    const phrases: string[] = [];

    // Extract quoted text
    const quotes = text.match(/"([^"]+)"/g);
    if (quotes) {
      phrases.push(...quotes.map((q) => q.slice(1, -1)));
    }

    // Extract capitalized phrases (potential proper nouns, titles)
    const capsPhrases = text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g);
    if (capsPhrases) {
      phrases.push(...capsPhrases);
    }

    // Extract technical terms (words with numbers, underscores, dots)
    const technicalTerms = text.match(/\b\w*[\d_]\w*\b/g);
    if (technicalTerms) {
      phrases.push(...technicalTerms);
    }

    return phrases.filter((phrase) => phrase.length > 5).slice(0, 5);
  }

  /**
   * Extract action items and tasks from text
   */
  protected extractActionItems(text: string): string[] {
    const actions: string[] = [];

    // Look for imperative verbs followed by objects
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 10);

    sentences.forEach((sentence) => {
      const trimmed = sentence.trim();
      // Check for action patterns
      if (
        trimmed.match(
          /^(create|implement|fix|update|add|remove|modify|refactor|optimize)/i,
        )
      ) {
        actions.push(trimmed);
      }
      // Check for TODO/FIXME patterns
      if (trimmed.match(/(todo|fixme|note|important)/i)) {
        actions.push(trimmed);
      }
    });

    return actions.slice(0, 5);
  }

  protected summarizeToolCalls(messages: MessageRecord[]): string {
    const toolCalls = messages
      .filter(
        (m) =>
          m.type === 'gemini' &&
          'toolCalls' in m &&
          Array.isArray(
            (m as unknown as { toolCalls?: unknown[] }).toolCalls ?? [],
          ) &&
          ((m as unknown as { toolCalls?: unknown[] }).toolCalls ?? []).length >
            0,
      )
      .map((m) => (m as unknown as { toolCalls?: unknown[] }).toolCalls ?? [])
      .flat();
    return `Tool calls: ${toolCalls.length}`;
  }

  protected async buildCompressedContext(
    messages: MessageRecord[],
    summary: string,
    keyPoints: string[],
    toolCallsSummary: string,
  ): Promise<CompressedContext> {
    const timespan = {
      start: messages[0]?.timestamp || '',
      end: messages[messages.length - 1]?.timestamp || '',
    };
    const messageCount = messages.length;
    const originalTokens = await this.tokenEstimator.estimateTokens(
      JSON.stringify(messages),
    );
    const compressedTokens = await this.tokenEstimator.estimateTokens(
      summary + keyPoints.join(' ') + toolCallsSummary,
    );
    return {
      summary,
      keyPoints,
      toolCallsSummary,
      timespan,
      messageCount,
      originalTokens,
      compressedTokens,
    };
  }

  /**
   * Checks if two strings are similar based on simple text comparison
   */
  protected areSimilar(
    text1: string,
    text2: string,
    threshold: number = 0.7,
  ): boolean {
    if (text1 === text2) return true;

    // Simple similarity based on common words
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    const similarity = intersection.size / union.size;
    return similarity >= threshold;
  }
}

/**
 * Aggressive compression strategy - maximum compression.
 */
export class AggressiveCompressionStrategy extends MinimalCompressionStrategy {
  override async compress(
    messages: MessageRecord[],
    _config: ContextCompressionConfig,
  ): Promise<CompressedContext> {
    // Aggressive compression: keep only most critical info
    const summary = `Aggressive compression: ${messages.length} messages.`;
    const keyPoints: string[] = [];
    messages.forEach((msg) => {
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
          originalTokens += await this.tokenEstimator.estimateTokens(
            JSON.stringify(tc.args),
          );
          if (tc.result) {
            originalTokens += await this.tokenEstimator.estimateTokens(
              JSON.stringify(tc.result),
            );
          }
        }
      }
    }
    const compressedContent = `${summary}. ${keyPoints.join('. ')}. ${toolCallsSummary}`;
    const compressedTokens =
      await this.tokenEstimator.estimateTokens(compressedContent);
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

  override async compress(
    messages: MessageRecord[],
    _config: ContextCompressionConfig,
  ): Promise<CompressedContext> {
    // No compression: return all messages as key points
    const summary = `No compression: ${messages.length} messages.`;
    const keyPoints = messages.map((m) => m.content);
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
    const compressedTokens =
      await this.tokenEstimator.estimateTokens(compressedContent);
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

  override async compress(
    messages: MessageRecord[],
    config: ContextCompressionConfig,
  ): Promise<CompressedContext> {
    // Balanced: keep fewer key points, más resumen
    const summary = this.createModerateSummary(messages);
    const keyPoints = this.extractImportantPoints(
      messages,
      Math.max(5, Math.floor(config.preserveRecentMessages / 2)),
    );
    const toolCallsSummary = this.summarizeToolCalls(messages);
    return await this.buildCompressedContext(
      messages,
      summary,
      keyPoints,
      toolCallsSummary,
    );
  }

  protected createModerateSummary(messages: MessageRecord[]): string {
    const userCount = messages.filter((m) => m.type === 'user').length;
    const assistantCount = messages.filter((m) => m.type === 'gemini').length;
    const withTools = messages.filter(
      (m) => m.type === 'gemini' && m.toolCalls?.length,
    ).length;

    let summary = `Conversation: ${userCount} user msgs, ${assistantCount} assistant msgs`;
    if (withTools > 0) {
      summary += `, ${withTools} with tools`;
    }

    // Add intelligent topic extraction
    const topics = this.extractConversationTopics(messages);
    if (topics.length > 0) {
      summary += `. Topics: ${topics.slice(0, 3).join(', ')}`;
    }

    // Add sentiment analysis
    const sentiment = this.analyzeConversationSentiment(messages);
    if (sentiment !== 'neutral') {
      summary += `. Tone: ${sentiment}`;
    }

    return summary;
  }

  /**
   * Extracts main topics from conversation using keyword analysis
   */
  protected extractConversationTopics(messages: MessageRecord[]): string[] {
    const allContent = messages.map((m) => m.content).join(' ');
    const keywords = this.extractKeywords(allContent);

    // Group related keywords into topics
    const topics = new Map<string, string[]>();

    keywords.forEach((keyword) => {
      // Simple topic categorization
      if (keyword.match(/error|bug|fix|issue|problem/)) {
        const topic = topics.get('issues') || [];
        topic.push(keyword);
        topics.set('issues', topic);
      } else if (keyword.match(/file|code|function|class|module/)) {
        const topic = topics.get('development') || [];
        topic.push(keyword);
        topics.set('development', topic);
      } else if (keyword.match(/test|testing|spec|validation/)) {
        const topic = topics.get('testing') || [];
        topic.push(keyword);
        topics.set('testing', topic);
      } else {
        const topic = topics.get('general') || [];
        topic.push(keyword);
        topics.set('general', topic);
      }
    });

    // Return top topics
    return Array.from(topics.keys()).slice(0, 3);
  }

  /**
   * Analyzes overall sentiment of the conversation
   */
  protected analyzeConversationSentiment(messages: MessageRecord[]): string {
    const allContent = messages
      .map((m) => m.content)
      .join(' ')
      .toLowerCase();

    const positiveWords = [
      'success',
      'completed',
      'working',
      'good',
      'excellent',
      'great',
      'perfect',
    ];
    const negativeWords = [
      'error',
      'failed',
      'problem',
      'issue',
      'bug',
      'broken',
      'wrong',
    ];

    let positiveScore = 0;
    let negativeScore = 0;

    positiveWords.forEach((word) => {
      const matches = (allContent.match(new RegExp(word, 'g')) || []).length;
      positiveScore += matches;
    });

    negativeWords.forEach((word) => {
      const matches = (allContent.match(new RegExp(word, 'g')) || []).length;
      negativeScore += matches;
    });

    if (positiveScore > negativeScore * 1.5) return 'positive';
    if (negativeScore > positiveScore * 1.5) return 'negative';
    return 'neutral';
  }
}

/**
 * Intelligent compression strategy using advanced NLP techniques
 */
export class IntelligentCompressionStrategy extends ModerateCompressionStrategy {
  constructor(tokenEstimator: TokenEstimator) {
    super(tokenEstimator);
  }

  override async compress(
    messages: MessageRecord[],
    config: ContextCompressionConfig,
  ): Promise<CompressedContext> {
    // Use advanced NLP techniques for intelligent compression
    const summary = this.createIntelligentSummary(messages);
    const keyPoints = this.extractIntelligentKeyPoints(messages, config);
    const toolCallsSummary = this.summarizeToolCallsIntelligently(messages);
    return await this.buildCompressedContext(
      messages,
      summary,
      keyPoints,
      toolCallsSummary,
    );
  }

  private createIntelligentSummary(messages: MessageRecord[]): string {
    const basicSummary = this.createModerateSummary(messages);

    // Enhance with conversation flow analysis
    const conversationFlow = this.analyzeConversationFlow(messages);
    const topics = this.extractConversationTopics(messages);
    const sentiment = this.analyzeConversationSentiment(messages);

    let enhancedSummary = basicSummary;

    if (conversationFlow.length > 0) {
      enhancedSummary += `. Flow: ${conversationFlow.join(' → ')}`;
    }

    if (topics.length > 0) {
      enhancedSummary += `. Focus: ${topics.slice(0, 2).join(', ')}`;
    }

    if (sentiment !== 'neutral') {
      enhancedSummary += `. Overall: ${sentiment}`;
    }

    return enhancedSummary;
  }

  private extractIntelligentKeyPoints(
    messages: MessageRecord[],
    config: ContextCompressionConfig,
  ): string[] {
    const allContent = messages.map((m) => m.content).join(' ');

    // Use advanced NLP techniques
    const keywords = this.extractKeywords(allContent);
    const phrases = this.extractImportantPhrases(allContent);
    const actions = this.extractActionItems(allContent);

    // Combine and prioritize key points
    const intelligentPoints: string[] = [];

    // Add top keywords
    keywords.slice(0, 3).forEach((keyword) => {
      intelligentPoints.push(`Keyword: ${keyword}`);
    });

    // Add important phrases
    phrases.slice(0, 2).forEach((phrase) => {
      intelligentPoints.push(`Phrase: "${phrase}"`);
    });

    // Add action items
    actions.slice(0, 2).forEach((action) => {
      intelligentPoints.push(`Action: ${action}`);
    });

    // Add traditional pattern-based points
    const traditionalPoints = this.extractKeyPointsFromMessages(messages);
    intelligentPoints.push(...traditionalPoints.slice(0, 5));

    // Remove duplicates and limit based on config
    const uniquePoints = this.deduplicateKeyPoints(intelligentPoints);
    const maxPoints = Math.max(
      8,
      Math.floor(config.preserveRecentMessages / 1.5),
    );

    return uniquePoints.slice(0, maxPoints);
  }

  private analyzeConversationFlow(messages: MessageRecord[]): string[] {
    const flow: string[] = [];
    let currentPhase = '';

    messages.forEach((_msg, _index) => {
      const content = _msg.content.toLowerCase();

      // Detect phase changes
      if (content.includes('error') || content.includes('problem')) {
        if (currentPhase !== 'problem-solving') {
          currentPhase = 'problem-solving';
          flow.push('Problem');
        }
      } else if (content.includes('implement') || content.includes('create')) {
        if (currentPhase !== 'implementation') {
          currentPhase = 'implementation';
          flow.push('Implementation');
        }
      } else if (content.includes('test') || content.includes('verify')) {
        if (currentPhase !== 'testing') {
          currentPhase = 'testing';
          flow.push('Testing');
        }
      } else if (content.includes('complete') || content.includes('done')) {
        if (currentPhase !== 'completion') {
          currentPhase = 'completion';
          flow.push('Completion');
        }
      }
    });

    return flow.slice(0, 4); // Limit to 4 phases
  }

  private deduplicateKeyPoints(points: string[]): string[] {
    const unique: string[] = [];

    for (const point of points) {
      const isDuplicate = unique.some((existing) =>
        this.areSimilar(point, existing, 0.6),
      );
      if (!isDuplicate) {
        unique.push(point);
      }
    }

    return unique;
  }

  private summarizeToolCallsIntelligently(messages: MessageRecord[]): string {
    const toolCalls = messages
      .filter(
        (m) =>
          m.type === 'gemini' &&
          'toolCalls' in m &&
          Array.isArray(
            (m as unknown as { toolCalls?: unknown[] }).toolCalls ?? [],
          ) &&
          ((m as unknown as { toolCalls?: unknown[] }).toolCalls ?? []).length >
            0,
      )
      .map((m) => (m as unknown as { toolCalls?: unknown[] }).toolCalls ?? [])
      .flat();

    if (toolCalls.length === 0) {
      return '';
    }

    // Group tool calls by type
    const toolUsage = new Map<string, number>();
    toolCalls.forEach((call: unknown) => {
      const toolCall = call as { name?: string };
      const toolName = toolCall.name || 'unknown';
      toolUsage.set(toolName, (toolUsage.get(toolName) || 0) + 1);
    });

    // Create intelligent summary
    const topTools = Array.from(toolUsage.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3);

    let summary = `Tools used: ${toolCalls.length} calls`;
    if (topTools.length > 0) {
      const toolList = topTools
        .map(([name, count]) => `${name}(${count})`)
        .join(', ');
      summary += ` (${toolList})`;
    }

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
 * - ALL user messages (prompts) are ALWAYS preserved in full detail
 * - Recent assistant messages (default: 5-10) are kept in full detail
 * - Older assistant messages are progressively compressed based on age and relevance
 * - Tool calls are summarized while preserving success/failure patterns
 * - Key information is extracted and preserved regardless of age
 * - Total context is kept under configurable token limits (default: 66k)
 *
 * This ensures that the AI never forgets the user's original mission or context,
 * while still managing conversation length efficiently.
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
  private compressionStrategies: Map<
    CompressionStrategy,
    CompressionStrategyHandler
  >;

  // Context compression configuration
  private compressionConfig: ContextCompressionConfig = {
    maxContextTokens: 240000, // Increased from 66k to 240k for better context preservation
    preserveRecentMessages: 100, // Always preserve ALL user messages up to this count
    compressionRatio: 0.3,
    keywordPreservation: true,
    summarizeToolCalls: true,
    strategy: CompressionStrategy.NO_COMPRESSION,
    neverCompressUserMessages: true, // CRITICAL: User messages are NEVER compressed
    importanceThreshold: 0.7, // High threshold for preserving important messages
  };

  // Dual-context manager for intelligent routing
  private contextManager: ContextManager;

  constructor(
    config: Config,
    fileSystem?: FileSystemAdapter,
    tokenEstimator?: TokenEstimator,
  ) {
    this.config = config;
    this.sessionId = config.getSessionId();
    this.projectHash = getProjectHash(config.getProjectRoot());

    // Set up dependencies with defaults
    this.fileSystem = fileSystem || new NodeFileSystemAdapter();
    this.tokenEstimator = tokenEstimator || new AdvancedTokenEstimator();
    // Initialize compression strategies
    this.compressionStrategies = new Map<
      CompressionStrategy,
      CompressionStrategyHandler
    >([
      [
        CompressionStrategy.MINIMAL,
        new MinimalCompressionStrategy(this.tokenEstimator),
      ],
      [
        CompressionStrategy.MODERATE,
        new ModerateCompressionStrategy(this.tokenEstimator),
      ],
      [
        CompressionStrategy.AGGRESSIVE,
        new AggressiveCompressionStrategy(this.tokenEstimator),
      ],
      [
        CompressionStrategy.NO_COMPRESSION,
        new NoCompressionStrategy(this.tokenEstimator),
      ],
      [
        CompressionStrategy.INTELLIGENT,
        new IntelligentCompressionStrategy(this.tokenEstimator),
      ],
    ]);
    // Update configuration from environment or config
    this.updateCompressionConfig();

    // Initialize dual-context manager
    this.contextManager = new ContextManager(
      this.createDualContextConfig(),
      this.tokenEstimator
    );
  }

  /**
   * Creates dual-context configuration from current settings
   */
  private createDualContextConfig(): DualContextConfig {
    const chatCompression = this.config.getChatCompression();

    return {
      promptContextTokens: chatCompression?.promptContextTokens ?? 1000000, // 1M tokens for prompts
      toolContextTokens: chatCompression?.toolContextTokens ?? 28000, // 28K tokens for tool execution
      enableDualContext: chatCompression?.enableDualContext ?? true,
      promptModel: 'gemini-2.5-pro',
      toolModel: 'gemini-2.5-flash-lite',
    };
  }

  /**
   * Updates compression configuration based on config object and environment variables.
   */
  private updateCompressionConfig(): void {
    // Get chat compression settings from Config object
    const chatCompression = this.config.getChatCompression();

    // Apply settings from Config object first
    if (chatCompression) {
      // Map contextPercentageThreshold to maxContextTokens if provided
      // Use a reasonable default context size and apply the percentage
      if (chatCompression.contextPercentageThreshold !== undefined) {
        // Assuming default context is 128k tokens, apply percentage threshold
        const defaultContextTokens = 128000;
        this.compressionConfig.maxContextTokens = Math.floor(
          defaultContextTokens *
            (chatCompression.contextPercentageThreshold / 100),
        );
      }

      // Handle dual-context settings
      if (chatCompression.enableDualContext) {
        // Update context manager with new dual-context config
        this.contextManager.updateConfig(this.createDualContextConfig());
      }
    }

    // Fallback to environment variables for backward compatibility
    const maxTokens = process.env['GEMINI_MAX_CONTEXT_TOKENS'];
    if (maxTokens) {
      this.compressionConfig.maxContextTokens = parseInt(maxTokens, 10);
    }

    const preserveMessages = process.env['GEMINI_PRESERVE_RECENT_MESSAGES'];
    if (preserveMessages) {
      this.compressionConfig.preserveRecentMessages = parseInt(
        preserveMessages,
        10,
      );
    }

    const strategy = process.env['GEMINI_COMPRESSION_STRATEGY'];
    if (
      strategy &&
      Object.values(CompressionStrategy).includes(
        strategy as CompressionStrategy,
      )
    ) {
      this.compressionConfig.strategy = strategy as CompressionStrategy;
    }

    // Handle dual-context environment variables
    const promptTokens = process.env['GEMINI_PROMPT_CONTEXT_TOKENS'];
    const toolTokens = process.env['GEMINI_TOOL_CONTEXT_TOKENS'];
    const enableDual = process.env['GEMINI_ENABLE_DUAL_CONTEXT'];

    if (promptTokens || toolTokens || enableDual) {
      const dualConfig = this.createDualContextConfig();
      if (promptTokens) dualConfig.promptContextTokens = parseInt(promptTokens, 10);
      if (toolTokens) dualConfig.toolContextTokens = parseInt(toolTokens, 10);
      if (enableDual) dualConfig.enableDualContext = enableDual === 'true';

      this.contextManager.updateConfig(dualConfig);
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
  private async compressContextIfNeeded(
    conversation: EnhancedConversationRecord,
  ): Promise<EnhancedConversationRecord> {
    const userMessages = conversation.messages.filter(
      (msg) => msg.type === 'user',
    );
    const totalUserMessages = userMessages.length;

    // CRITICAL: Never compress if neverCompressUserMessages is true
    if (this.compressionConfig.neverCompressUserMessages) {
      // Estimate current context size including ALL user messages
      let totalTokens = 0;
      for (const msg of conversation.messages) {
        totalTokens += await this.estimateTokenCount(msg.content);
        if (msg.type === 'gemini' && msg.toolCalls) {
          for (const tc of msg.toolCalls) {
            totalTokens += await this.estimateTokenCount(
              JSON.stringify(tc.args),
            );
            if (tc.result) {
              totalTokens += await this.estimateTokenCount(
                JSON.stringify(tc.result),
              );
            }
          }
        }
      }
      // Add compressed context tokens if it exists
      if (conversation.compressedContext) {
        totalTokens += conversation.compressedContext.compressedTokens;
      }

      // Only compress if we exceed the limit significantly (give more headroom for user messages)
      if (totalTokens <= this.compressionConfig.maxContextTokens * 1.2) {
        return conversation;
      }
    } else {
      // Original logic for when user messages can be compressed
      if (totalUserMessages <= this.compressionConfig.preserveRecentMessages) {
        return conversation;
      }
    }

    // Estimate current context size
    let totalTokens = 0;
    for (const msg of conversation.messages) {
      totalTokens += await this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          totalTokens += await this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += await this.estimateTokenCount(
              JSON.stringify(tc.result),
            );
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

    console.log(
      `[ChatRecording] Context size ${totalTokens} tokens exceeds limit ${this.compressionConfig.maxContextTokens}. Compressing...`,
    );

    // Separate user and assistant messages
    const allUserMessages = conversation.messages.filter(
      (msg) => msg.type === 'user',
    );
    const allAssistantMessages = conversation.messages.filter(
      (msg) => msg.type === 'gemini',
    );

    // CRITICAL: Always preserve ALL user messages (they contain the mission/context)
    // Only compress old assistant messages
    const preserveAssistantCount = Math.max(
      5,
      Math.floor(this.compressionConfig.preserveRecentMessages / 2),
    );
    const recentAssistantMessages = allAssistantMessages.slice(
      -preserveAssistantCount,
    );
    const oldAssistantMessages = allAssistantMessages.slice(
      0,
      -preserveAssistantCount,
    );

    // Evaluate importance of assistant messages to preserve critical ones
    const importantAssistantMessages =
      await this.filterImportantMessages(oldAssistantMessages);

    // Combine: ALL user messages + recent assistant messages + important old messages
    const messagesToKeep = [
      ...allUserMessages,
      ...recentAssistantMessages,
      ...importantAssistantMessages,
    ];

    // Create compressed context from remaining old assistant messages
    const remainingOldMessages = oldAssistantMessages.filter(
      (msg) => !importantAssistantMessages.includes(msg),
    );
    const newCompressedContext = await this.createCompressedContext(
      remainingOldMessages,
      conversation.compressedContext,
    );

    return {
      ...conversation,
      messages: messagesToKeep,
      compressedContext: newCompressedContext,
      lastCompressionTime: new Date().toISOString(),
      compressionConfig: this.compressionConfig,
    };
  }

  /**
   * Filters messages based on importance score to preserve critical information.
   */
  private async filterImportantMessages(
    messages: MessageRecord[],
  ): Promise<MessageRecord[]> {
    if (messages.length === 0) return [];

    const importantMessages: MessageRecord[] = [];

    for (const message of messages) {
      const importanceScore = await this.calculateMessageImportance(message);
      if (importanceScore >= this.compressionConfig.importanceThreshold) {
        importantMessages.push(message);
      }
    }

    return importantMessages;
  }

  /**
   * Calculates importance score for a message (0.0-1.0).
   * Higher scores indicate messages that should be preserved during compression.
   */
  private async calculateMessageImportance(
    message: MessageRecord,
  ): Promise<number> {
    let score = 0.0;
    const content = message.content.toLowerCase();

    // High importance indicators
    const highImportanceKeywords = [
      'error',
      'failed',
      'failure',
      'exception',
      'crash',
      'bug',
      'critical',
      'important',
      'key',
      'summary',
      'conclusion',
      'result',
      'outcome',
      'decision',
      'requirement',
      'specification',
      'architecture',
      'security',
      'performance',
      'breaking change',
    ];

    for (const keyword of highImportanceKeywords) {
      if (content.includes(keyword)) {
        score += 0.2;
      }
    }

    // Medium importance indicators
    const mediumImportanceKeywords = [
      'success',
      'completed',
      'done',
      'fixed',
      'resolved',
      'created',
      'modified',
      'updated',
      'changed',
      'test',
      'function',
      'class',
      'method',
      'api',
    ];

    for (const keyword of mediumImportanceKeywords) {
      if (content.includes(keyword)) {
        score += 0.1;
      }
    }

    // Tool calls are generally important
    if (
      message.type === 'gemini' &&
      message.toolCalls &&
      message.toolCalls.length > 0
    ) {
      score += 0.3;
    }

    // Recent messages are more important (recency bias)
    const messageAge = Date.now() - new Date(message.timestamp).getTime();
    const hoursOld = messageAge / (1000 * 60 * 60);
    if (hoursOld < 1) score += 0.2;
    else if (hoursOld < 24) score += 0.1;

    // Cap at 1.0 and ensure minimum score for very recent messages
    return Math.min(1.0, Math.max(score, hoursOld < 0.1 ? 0.5 : 0));
  }

  /**
   * Creates intelligent compressed representation of old messages using the configured strategy.
   */
  private async createCompressedContext(
    messages: MessageRecord[],
    existingCompressed?: CompressedContext,
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
    const strategy = this.compressionStrategies.get(
      this.compressionConfig.strategy,
    );
    if (!strategy) {
      throw new ChatRecordingCompressionError(
        `Unknown compression strategy: ${this.compressionConfig.strategy}`,
      );
    }
    if (
      this.compressionConfig.strategy === CompressionStrategy.NO_COMPRESSION
    ) {
      // Estimate tokens for all messages
      let originalTokens = 0;
      for (const msg of messages) {
        originalTokens += await this.estimateTokenCount(msg.content);
        if (msg.type === 'gemini' && msg.toolCalls) {
          for (const tc of msg.toolCalls) {
            originalTokens += await this.estimateTokenCount(
              JSON.stringify(tc.args),
            );
            if (tc.result) {
              originalTokens += await this.estimateTokenCount(
                JSON.stringify(tc.result),
              );
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
        // Use intelligent merging instead of simple concatenation
        return await this.mergeCompressedContexts(existingCompressed, messages);
      } else if (existingCompressed) {
        return existingCompressed;
      } else {
        return await strategy.compress(messages, this.compressionConfig);
      }
    } catch (error) {
      throw new ChatRecordingCompressionError(
        `Failed to compress context: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Intelligently merges existing compressed context with new messages
   */
  private async mergeCompressedContexts(
    existingCompressed: CompressedContext,
    newMessages: MessageRecord[],
  ): Promise<CompressedContext> {
    // Get the compression strategy
    const strategy = this.compressionStrategies.get(
      this.compressionConfig.strategy,
    );
    if (!strategy) {
      throw new ChatRecordingCompressionError(
        `Unknown compression strategy: ${this.compressionConfig.strategy}`,
      );
    }

    // Compress new messages
    const newCompressed = await strategy.compress(
      newMessages,
      this.compressionConfig,
    );

    // Intelligent merging logic
    const mergedSummary = this.mergeSummaries(
      existingCompressed.summary,
      newCompressed.summary,
    );
    const mergedKeyPoints = this.mergeKeyPoints(
      existingCompressed.keyPoints,
      newCompressed.keyPoints,
    );
    const mergedToolCallsSummary = this.mergeToolCallsSummaries(
      existingCompressed.toolCallsSummary,
      newCompressed.toolCallsSummary,
    );

    // Update timespan to cover both periods
    const mergedTimespan = {
      start: existingCompressed.timespan.start || newCompressed.timespan.start,
      end: newCompressed.timespan.end,
    };

    // Calculate totals
    const mergedMessageCount =
      existingCompressed.messageCount + newCompressed.messageCount;
    const mergedOriginalTokens =
      existingCompressed.originalTokens + newCompressed.originalTokens;

    // Estimate compressed tokens for the merged content
    const mergedContent = `${mergedSummary}. ${mergedKeyPoints.join('. ')}. ${mergedToolCallsSummary}`;
    const mergedCompressedTokens = await this.estimateTokenCount(mergedContent);

    return {
      summary: mergedSummary,
      keyPoints: mergedKeyPoints,
      toolCallsSummary: mergedToolCallsSummary,
      timespan: mergedTimespan,
      messageCount: mergedMessageCount,
      originalTokens: mergedOriginalTokens,
      compressedTokens: mergedCompressedTokens,
    };
  }

  /**
   * Intelligently merges two summaries
   */
  private mergeSummaries(existingSummary: string, newSummary: string): string {
    if (!existingSummary) return newSummary;
    if (!newSummary) return existingSummary;

    // If summaries are very similar, keep the newer one
    if (this.areSimilar(existingSummary, newSummary, 0.8)) {
      return newSummary;
    }

    // Combine summaries intelligently
    const combined = `${existingSummary}. ${newSummary}`;

    // If combined is too long, create a more concise version
    if (combined.length > 300) {
      return this.createConciseSummary(combined);
    }

    return combined;
  }

  /**
   * Merges key points while removing duplicates and prioritizing importance
   */
  private mergeKeyPoints(
    existingPoints: string[],
    newPoints: string[],
  ): string[] {
    const allPoints = [...existingPoints, ...newPoints];

    // Remove duplicates based on similarity
    const uniquePoints: string[] = [];
    for (const point of allPoints) {
      const isDuplicate = uniquePoints.some((existing) =>
        this.areSimilar(point, existing, 0.7),
      );
      if (!isDuplicate) {
        uniquePoints.push(point);
      }
    }

    // Prioritize points: errors first, then decisions, then others
    const prioritized = uniquePoints.sort((a, b) => {
      const getPriority = (point: string) => {
        if (point.toLowerCase().includes('error')) return 3;
        if (point.toLowerCase().includes('decision')) return 2;
        if (point.toLowerCase().includes('action')) return 1;
        return 0;
      };
      return getPriority(b) - getPriority(a);
    });

    // Keep only the most important points (max 20)
    return prioritized.slice(0, 20);
  }

  /**
   * Merges tool call summaries intelligently
   */
  private mergeToolCallsSummaries(
    existing: string,
    newSummary: string,
  ): string {
    if (!existing) return newSummary;
    if (!newSummary) return existing;

    // Parse tool call counts from summaries
    const existingCount = this.extractToolCallCount(existing);
    const newCount = this.extractToolCallCount(newSummary);

    const totalCount = existingCount + newCount;
    return `Tool calls: ${totalCount}`;
  }

  /**
   * Extracts tool call count from summary string
   */
  private extractToolCallCount(summary: string): number {
    const match = summary.match(/Tool calls: (\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  }

  /**
   * Creates a concise summary from a long combined summary
   */
  private createConciseSummary(longSummary: string): string {
    // Extract the most important parts
    const sentences = longSummary
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 10);

    // Prioritize sentences with key information
    const prioritized = sentences
      .map((sentence) => ({
        text: sentence.trim(),
        score: this.scoreSentenceImportance(sentence),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .map((item) => item.text);

    return prioritized.join('. ') + '.';
  }

  /**
   * Scores sentence importance for summary creation
   */
  private scoreSentenceImportance(sentence: string): number {
    let score = 0;
    const lowerSentence = sentence.toLowerCase();

    // Boost score for important keywords
    if (lowerSentence.includes('error') || lowerSentence.includes('failed'))
      score += 3;
    if (
      lowerSentence.includes('decision') ||
      lowerSentence.includes('conclusion')
    )
      score += 2;
    if (
      lowerSentence.includes('implemented') ||
      lowerSentence.includes('completed')
    )
      score += 2;
    if (lowerSentence.includes('tool')) score += 1;

    // Boost score for longer sentences (likely more detailed)
    score += Math.min(sentence.length / 50, 2);

    return score;
  }

  /**
   * Checks if two strings are similar based on simple text comparison
   */
  protected areSimilar(
    text1: string,
    text2: string,
    threshold: number = 0.7,
  ): boolean {
    if (text1 === text2) return true;

    // Simple similarity based on common words
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    const similarity = intersection.size / union.size;
    return similarity >= threshold;
  }

  /**
   * Generates a consistent file name for a session.
   */
  private generateSessionFileName(sessionId: string): string {
    const timestamp = new Date().toISOString().slice(0, 16).replace(/:/g, '-');
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
        error instanceof Error ? error : undefined,
      );
    }
  }

  private getLastMessage(
    conversation: ConversationRecord,
  ): MessageRecord | undefined {
    return conversation.messages[conversation.messages.length - 1];
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
        error instanceof Error ? error : undefined,
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
        subject:
          typeof thought === 'object' &&
          thought !== null &&
          'subject' in thought &&
          typeof (thought as ThoughtSummary).subject === 'string'
            ? (thought as ThoughtSummary).subject
            : '',
        description:
          typeof thought === 'object' &&
          thought !== null &&
          'description' in thought &&
          typeof (thought as ThoughtSummary).description === 'string'
            ? (thought as ThoughtSummary).description
            : '',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      if (this.config.getDebugMode()) {
        throw new ChatRecordingError(
          `Error saving thought: ${error instanceof Error ? error.message : String(error)}`,
          error instanceof Error ? error : undefined,
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
        error instanceof Error ? error : undefined,
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
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Loads up the conversation record from disk.
   */
  private async readConversation(): Promise<EnhancedConversationRecord> {
    try {
      this.cachedLastConvData = await this.fileSystem.readFile(
        this.conversationFile!,
      );
      return this.cachedLastConvData
        ? (JSON.parse(this.cachedLastConvData) as EnhancedConversationRecord)
        : ({} as EnhancedConversationRecord);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw new ChatRecordingFileError(
          `Error reading conversation file: ${error instanceof Error ? error.message : String(error)}`,
          error instanceof Error ? error : undefined,
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
  private async writeConversation(
    conversation: EnhancedConversationRecord,
  ): Promise<void> {
    try {
      if (!this.conversationFile) return;
      // Don't write the file yet until there's at least one message.
      if (conversation.messages.length === 0 && !conversation.compressedContext)
        return;

      // Apply compression if needed before writing
      const compressedConversation =
        await this.compressContextIfNeeded(conversation);

      // Only write the file if this change would change the file.
      compressedConversation.lastUpdated = new Date().toISOString();
      const newContent = JSON.stringify(compressedConversation, null, 2);

      if (this.cachedLastConvData !== newContent) {
        this.cachedLastConvData = newContent;
        await this.fileSystem.writeFile(this.conversationFile, newContent);

        // Log compression if it occurred
        if (compressedConversation.compressedContext) {
          const ctx = compressedConversation.compressedContext;
          console.log(
            `[ChatRecording] Compressed ${ctx.messageCount} messages: ${ctx.originalTokens} → ${ctx.compressedTokens} tokens`,
          );
        }
      }
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error writing conversation file: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined,
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
    const optimized = await (async () =>
      await this.compressContextIfNeeded(conversation))();

    // Calculate statistics
    let totalTokens = 0;
    for (const msg of optimized.messages) {
      totalTokens += await this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          totalTokens += await this.estimateTokenCount(JSON.stringify(tc.args));
          if (tc.result) {
            totalTokens += await this.estimateTokenCount(
              JSON.stringify(tc.result),
            );
          }
        }
      }
    }

    if (optimized.compressedContext) {
      totalTokens += optimized.compressedContext.compressedTokens;
    }

    let compressionStats;
    if (
      optimized.compressedContext &&
      optimized.compressedContext.messageCount > 0
    ) {
      const originalMessages =
        optimized.compressedContext.messageCount + optimized.messages.length;
      const compressedMessages = optimized.messages.length;
      const tokenReduction =
        optimized.compressedContext.originalTokens -
        optimized.compressedContext.compressedTokens;
      const compressionRatio =
        optimized.compressedContext.compressedTokens /
        optimized.compressedContext.originalTokens;
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
   * Gets optimized conversational history specifically designed for PromptContextManager integration.
   * This method provides flexible token budget management and returns history in Content[] format.
   *
   * @param fullHistory - Complete conversation history to optimize (Content[] format)
   * @param targetTokenCount - Maximum number of tokens to allocate for history
   * @param includeSystemInfo - Whether to include compression stats in the result (default: false)
   * @returns Optimized history formatted for integration with PromptContextManager
   */
  async getOptimizedHistoryForPrompt(
    fullHistory: Content[],
    targetTokenCount: number,
    includeSystemInfo: boolean = false,
  ): Promise<{
    contents: Content[];
    estimatedTokens: number;
    compressionLevel: CompressionStrategy;
    metaInfo: {
      compressionApplied: boolean;
      originalMessageCount: number;
      finalMessageCount: number;
      compressionStats?: {
        originalMessages: number;
        compressedMessages: number;
        tokenReduction: number;
        compressionRatio: number;
      };
    };
  }> {
    if (!fullHistory || fullHistory.length === 0) {
      return {
        contents: [],
        estimatedTokens: 0,
        compressionLevel: CompressionStrategy.NO_COMPRESSION,
        metaInfo: {
          compressionApplied: false,
          originalMessageCount: 0,
          finalMessageCount: 0,
        },
      };
    }

    // Convert Content[] to MessageRecord[] for compression processing
    const messageRecords: MessageRecord[] = fullHistory.map(
      (content, index) => ({
        id: `msg_${index}_${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: content.role === 'user' ? 'user' : 'gemini',
        content: this.extractTextFromContent(content),
      }),
    );

    // Calculate current token count
    let originalTokens = 0;
    for (const record of messageRecords) {
      originalTokens += await this.estimateTokenCount(record.content);
    }

    // If already under budget, return as-is
    if (originalTokens <= targetTokenCount) {
      return {
        contents: fullHistory,
        estimatedTokens: originalTokens,
        compressionLevel: CompressionStrategy.NO_COMPRESSION,
        metaInfo: {
          compressionApplied: false,
          originalMessageCount: fullHistory.length,
          finalMessageCount: fullHistory.length,
        },
      };
    }

    // Apply compression with custom token budget
    const originalConfig = { ...this.compressionConfig };
    // Apply compression by creating a temporary conversation record
    const tempConversation: EnhancedConversationRecord = {
      sessionId: 'temp_optimization',
      projectHash: 'temp_project',
      startTime: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
      messages: messageRecords,
    };

    const optimized = await this.compressContextIfNeeded(tempConversation);

    // Restore original config
    this.compressionConfig = originalConfig;

    // Convert messages back to Content[] format
    const optimizedContents: Content[] = optimized.messages.map(
      (msg: MessageRecord) => ({
        role: msg.type === 'user' ? 'user' : 'model',
        parts: [{ text: msg.content }],
      }),
    );

    // Calculate final tokens
    let finalTokens = 0;
    for (const msg of optimized.messages) {
      finalTokens += await this.estimateTokenCount(msg.content);
    }

    if (optimized.compressedContext) {
      finalTokens += optimized.compressedContext.compressedTokens;
    }

    const compressionApplied = !!optimized.compressedContext;
    let compressionStats;
    if (compressionApplied && includeSystemInfo) {
      const ctx = optimized.compressedContext!;
      const originalMessages = ctx.messageCount + optimized.messages.length;
      const compressedMessages = optimized.messages.length;
      const tokenReduction = ctx.originalTokens - ctx.compressedTokens;
      const compressionRatio = ctx.compressedTokens / ctx.originalTokens;
      compressionStats = {
        originalMessages,
        compressedMessages,
        tokenReduction,
        compressionRatio,
      };
    }

    return {
      contents: optimizedContents,
      estimatedTokens: finalTokens,
      compressionLevel: this.compressionConfig.strategy,
      metaInfo: {
        compressionApplied,
        originalMessageCount: fullHistory.length,
        finalMessageCount: optimizedContents.length,
        compressionStats,
      },
    };
  }

  /**
   * Helper method to extract text content from Content objects
   */
  private extractTextFromContent(content: Content): string {
    if (!content.parts) return '';

    return content.parts
      .filter((part) => part.text)
      .map((part) => part.text)
      .join(' ')
      .trim();
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
      totalMessages:
        (conversation.compressedContext?.messageCount || 0) +
        conversation.messages.length,
      recentMessages: conversation.messages.length,
      compressedMessages: conversation.compressedContext?.messageCount || 0,
      estimatedTokens: optimized.totalEstimatedTokens,
      compressionRatio: optimized.compressionStats?.compressionRatio,
      lastCompressionTime: conversation.lastCompressionTime,
    };
  }

  /**
   * Gets the current total token count for the conversation.
   * This includes all messages and any compressed context.
   */
  async getCurrentTokenCount(): Promise<number> {
    if (!this.conversationFile) {
      return 0;
    }

    const optimized = await this.getOptimizedContext();
    return optimized.totalEstimatedTokens;
  }

  /**
   * Gets the token count for the last message in the conversation.
   * Returns 0 if there are no messages.
   */
  async getLastMessageTokenCount(): Promise<number> {
    if (!this.conversationFile) {
      return 0;
    }

    const conversation = await this.readConversation();
    const lastMessage = conversation.messages[conversation.messages.length - 1];

    if (!lastMessage) {
      return 0;
    }

    let tokenCount = await this.estimateTokenCount(lastMessage.content);

    // Add tokens from tool calls if present
    if (lastMessage.type === 'gemini' && lastMessage.toolCalls) {
      for (const tc of lastMessage.toolCalls) {
        tokenCount += await this.estimateTokenCount(JSON.stringify(tc.args));
        if (tc.result) {
          tokenCount += await this.estimateTokenCount(
            JSON.stringify(tc.result),
          );
        }
      }
    }

    return tokenCount;
  }

  /**
   * Gets a comprehensive token summary for the current conversation.
   */
  async getTokenSummary(): Promise<{
    total: number;
    messages: number;
    toolCalls: number;
    compressed: number;
    lastMessage: number;
    messageCount: number;
  }> {
    if (!this.conversationFile) {
      return {
        total: 0,
        messages: 0,
        toolCalls: 0,
        compressed: 0,
        lastMessage: 0,
        messageCount: 0,
      };
    }

    const conversation = await this.readConversation();
    const optimized = await this.getOptimizedContext();

    let messageTokens = 0;
    let toolCallTokens = 0;

    // Calculate tokens for all messages
    for (const msg of conversation.messages) {
      messageTokens += await this.estimateTokenCount(msg.content);
      if (msg.type === 'gemini' && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          toolCallTokens += await this.estimateTokenCount(
            JSON.stringify(tc.args),
          );
          if (tc.result) {
            toolCallTokens += await this.estimateTokenCount(
              JSON.stringify(tc.result),
            );
          }
        }
      }
    }

    const lastMessageTokens = await this.getLastMessageTokenCount();
    const compressedTokens = optimized.compressedContext?.compressedTokens || 0;

    return {
      total: optimized.totalEstimatedTokens,
      messages: messageTokens,
      toolCalls: toolCallTokens,
      compressed: compressedTokens,
      lastMessage: lastMessageTokens,
      messageCount: conversation.messages.length,
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
      const sessionPath = path.join(
        chatsDir,
        this.generateSessionFileName(sessionId),
      );
      await this.fileSystem.unlink(sessionPath);
    } catch (error) {
      throw new ChatRecordingFileError(
        `Error deleting session: ${error instanceof Error ? error.message : String(error)}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Determines the appropriate context type for a given operation and content.
   * This is used by other services to route operations to the correct context.
   */
  determineContextType(operation: string, content: string): ContextType {
    return this.contextManager.determineContextType(operation, content);
  }

  /**
   * Gets the appropriate model for a given context type.
   * This enables intelligent model selection based on operation type.
   */
  getModelForContext(contextType: ContextType): string {
    return this.contextManager.getModelForContext(contextType);
  }

  /**
   * Checks if content fits within the specified context limits.
   * Useful for pre-flight checks before processing.
   */
  async canFitInContext(
    content: string,
    contextType: ContextType
  ): Promise<boolean> {
    return await this.contextManager.canFitInContext(content, contextType);
  }

  /**
   * Gets usage statistics for all contexts.
   * Useful for monitoring and debugging.
   */
  getContextUsageStats() {
    return this.contextManager.getAllUsageStats();
  }

  /**
   * Updates usage statistics for a context after processing.
   * Should be called after message processing to maintain accurate stats.
   */
  async updateContextUsage(
    contextType: ContextType,
    content: string,
    messageCount: number
  ): Promise<void> {
    await this.contextManager.updateUsageStats(
      contextType,
      content,
      messageCount
    );
  }
}
