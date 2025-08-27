import { PartListUnion, Status, ThoughtSummary, Config } from '../index.js';
/**
 * Base error class for chat recording errors.
 */
export declare class ChatRecordingError extends Error {
    cause?: Error;
    constructor(message: string, cause?: Error);
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
export declare class NodeFileSystemAdapter implements FileSystemAdapter {
    readFile(filePath: string): Promise<string>;
    writeFile(filePath: string, data: string): Promise<void>;
    mkdir(dirPath: string): Promise<void>;
    unlink(filePath: string): Promise<void>;
    exists(filePath: string): Promise<boolean>;
}
export declare class ChatRecordingInitializationError extends ChatRecordingError {
    constructor(message: string, cause?: Error);
}
export declare class ChatRecordingCompressionError extends ChatRecordingError {
    constructor(message: string, cause?: Error);
}
export declare class ChatRecordingFileError extends ChatRecordingError {
    constructor(message: string, cause?: Error);
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
export declare class SimpleTokenEstimator implements TokenEstimator {
    estimateTokens(text: string): Promise<number>;
}
/**
 * Advanced token estimator with precise Gemini tokenization patterns.
 * This implementation provides more accurate token estimation for Gemini models
 * by considering different text patterns and tokenization rules.
 */
export declare class AdvancedTokenEstimator implements TokenEstimator {
    estimateTokens(text: string): Promise<number>;
    /**
     * Synchronous token estimation for better performance
     */
    estimateTokensSync(text: string): number;
    private estimateWordTokens;
    private estimateCodeTokens;
    private estimateUrlTokens;
}
/**
 * Token usage summary for a message or conversation.
 */
export interface TokensSummary {
    input: number;
    output: number;
    cached: number;
    thoughts?: number;
    tool?: number;
    total: number;
}
/**
 * Configuration for context compression behavior.
 */
export interface ContextCompressionConfig {
    maxContextTokens: number;
    preserveRecentMessages: number;
    compressionRatio: number;
    keywordPreservation: boolean;
    summarizeToolCalls: boolean;
    strategy: CompressionStrategy;
}
/**
 * Available compression strategies.
 */
export declare enum CompressionStrategy {
    MINIMAL = "minimal",// Keep most information, light compression
    MODERATE = "moderate",// Balanced compression
    AGGRESSIVE = "aggressive",// Maximum compression, keep only essentials
    INTELLIGENT = "intelligent",// Advanced NLP-based compression
    NO_COMPRESSION = "no_compression"
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
    displayName?: string;
    description?: string;
    resultDisplay?: string;
    renderOutputAsMarkdown?: boolean;
}
/**
 * Message type and message type-specific fields.
 */
/**
 * Compressed representation of older context.
 */
export interface CompressedContext {
    summary: string;
    keyPoints: string[];
    toolCallsSummary: string;
    timespan: {
        start: string;
        end: string;
    };
    messageCount: number;
    originalTokens: number;
    compressedTokens: number;
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
export declare class MinimalCompressionStrategy implements CompressionStrategyHandler {
    protected tokenEstimator: TokenEstimator;
    constructor(tokenEstimator: TokenEstimator);
    compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext>;
    private createBriefSummary;
    protected extractImportantPoints(messages: MessageRecord[], maxPoints: number): string[];
    protected extractKeyPointsFromMessages(messages: MessageRecord[]): string[];
    /**
     * Advanced key point extraction using NLP techniques
     */
    private extractAdvancedKeyPoints;
    /**
     * Extract keywords using frequency and importance analysis
     */
    protected extractKeywords(text: string): string[];
    /**
     * Extract important phrases from text
     */
    protected extractImportantPhrases(text: string): string[];
    /**
     * Extract action items and tasks from text
     */
    protected extractActionItems(text: string): string[];
    protected summarizeToolCalls(messages: MessageRecord[]): string;
    protected buildCompressedContext(messages: MessageRecord[], summary: string, keyPoints: string[], toolCallsSummary: string): Promise<CompressedContext>;
    /**
     * Checks if two strings are similar based on simple text comparison
     */
    protected areSimilar(text1: string, text2: string, threshold?: number): boolean;
}
/**
 * Aggressive compression strategy - maximum compression.
 */
export declare class AggressiveCompressionStrategy extends MinimalCompressionStrategy {
    compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext>;
}
/**
 * No compression strategy - all messages are returned as key points.
 */
export declare class NoCompressionStrategy extends MinimalCompressionStrategy {
    constructor(tokenEstimator: TokenEstimator);
    compress(messages: MessageRecord[], _config: ContextCompressionConfig): Promise<CompressedContext>;
}
export declare class ModerateCompressionStrategy extends MinimalCompressionStrategy {
    constructor(tokenEstimator: TokenEstimator);
    compress(messages: MessageRecord[], config: ContextCompressionConfig): Promise<CompressedContext>;
    protected createModerateSummary(messages: MessageRecord[]): string;
    /**
     * Extracts main topics from conversation using keyword analysis
     */
    protected extractConversationTopics(messages: MessageRecord[]): string[];
    /**
     * Analyzes overall sentiment of the conversation
     */
    protected analyzeConversationSentiment(messages: MessageRecord[]): string;
}
/**
 * Intelligent compression strategy using advanced NLP techniques
 */
export declare class IntelligentCompressionStrategy extends ModerateCompressionStrategy {
    constructor(tokenEstimator: TokenEstimator);
    compress(messages: MessageRecord[], config: ContextCompressionConfig): Promise<CompressedContext>;
    private createIntelligentSummary;
    private extractIntelligentKeyPoints;
    private analyzeConversationFlow;
    private deduplicateKeyPoints;
    private summarizeToolCallsIntelligently;
}
/**
 * Compressed representation of older context.
 */
export interface CompressedContext {
    summary: string;
    keyPoints: string[];
    toolCallsSummary: string;
    timespan: {
        start: string;
        end: string;
    };
    messageCount: number;
    originalTokens: number;
    compressedTokens: number;
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
    displayName?: string;
    description?: string;
    resultDisplay?: string;
    renderOutputAsMarkdown?: boolean;
}
/**
 * Message type and message type-specific fields.
 */
export type ConversationRecordExtra = {
    type: 'user';
} | {
    type: 'gemini';
    toolCalls?: ToolCallRecord[];
    thoughts?: Array<ThoughtSummary & {
        timestamp: string;
    }>;
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
export declare class ChatRecordingService {
    private conversationFile;
    private cachedLastConvData;
    private sessionId;
    private projectHash;
    private queuedThoughts;
    private queuedTokens;
    private config;
    private fileSystem;
    private tokenEstimator;
    private compressionStrategies;
    private compressionConfig;
    constructor(config: Config, fileSystem?: FileSystemAdapter, tokenEstimator?: TokenEstimator);
    /**
     * Updates compression configuration based on config object and environment variables.
     */
    private updateCompressionConfig;
    /**
     * Estimates token count for a message using the configured estimator.
     */
    private estimateTokenCount;
    /**
     * Compresses old context intelligently to stay under token limits.
     */
    private compressContextIfNeeded;
    /**
     * Creates intelligent compressed representation of old messages using the configured strategy.
     */
    private createCompressedContext;
    /**
     * Intelligently merges existing compressed context with new messages
     */
    private mergeCompressedContexts;
    /**
     * Intelligently merges two summaries
     */
    private mergeSummaries;
    /**
     * Merges key points while removing duplicates and prioritizing importance
     */
    private mergeKeyPoints;
    /**
     * Merges tool call summaries intelligently
     */
    private mergeToolCallsSummaries;
    /**
     * Extracts tool call count from summary string
     */
    private extractToolCallCount;
    /**
     * Creates a concise summary from a long combined summary
     */
    private createConciseSummary;
    /**
     * Scores sentence importance for summary creation
     */
    private scoreSentenceImportance;
    /**
     * Checks if two strings are similar based on simple text comparison
     */
    protected areSimilar(text1: string, text2: string, threshold?: number): boolean;
    /**
     * Generates a consistent file name for a session.
     */
    private generateSessionFileName;
    /**
     * Initializes the chat recording service: creates a new conversation file and associates it with
     * this service instance, or resumes from an existing session if resumedSessionData is provided.
     */
    initialize(resumedSessionData?: ResumedSessionData): Promise<void>;
    private getLastMessage;
    private newMessage;
    /**
     * Records a message in the conversation with intelligent compression.
     */
    recordMessage(message: {
        type: ConversationRecordExtra['type'];
        content: string;
        append?: boolean;
    }): Promise<void>;
    /**
     * Records a thought from the assistant's reasoning process.
     */
    recordThought(thought: ThoughtSummary): void;
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
    }): Promise<void>;
    /**
     * Adds tool calls to the last message in the conversation (which should be by Gemini).
     */
    recordToolCalls(toolCalls: ToolCallRecord[]): Promise<void>;
    /**
     * Loads up the conversation record from disk.
     */
    private readConversation;
    /**
     * Saves the conversation record with intelligent compression.
     */
    private writeConversation;
    /**
     * Convenient helper for updating the conversation without file reading and writing and time
     * updating boilerplate.
     */
    private updateConversation;
    /**
     * Gets optimized conversation context for LLM consumption.
     * Returns recent messages in full + compressed historical context.
     */
    getOptimizedContext(): Promise<{
        compressedContext?: CompressedContext;
        recentMessages: MessageRecord[];
        totalEstimatedTokens: number;
        compressionStats?: {
            originalMessages: number;
            compressedMessages: number;
            tokenReduction: number;
            compressionRatio: number;
        };
    }>;
    /**
     * Forces immediate compression of the current conversation.
     * Useful for testing or manual optimization.
     */
    forceCompression(): Promise<void>;
    /**
     * Gets compression statistics for monitoring.
     */
    getCompressionStats(): Promise<{
        isCompressed: boolean;
        totalMessages: number;
        recentMessages: number;
        compressedMessages: number;
        estimatedTokens: number;
        compressionRatio?: number;
        lastCompressionTime?: string;
    }>;
    /**
     * Deletes a session file by session ID.
     */
    deleteSession(sessionId: string): Promise<void>;
}
