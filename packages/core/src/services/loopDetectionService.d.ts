/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { ServerGeminiStreamEvent } from '../core/turn.js';
import { LoopType } from '../telemetry/types.js';
import { Config } from '../config/config.js';
/**
 * Loop detection result with confidence and suggested actions
 */
export interface LoopDetectionResult {
    isLoop: boolean;
    confidence: number;
    loopType: LoopType;
    suggestedActions: LoopBreakAction[];
    reasoning?: string;
    affectedFiles?: string[];
}
/**
 * Suggested actions for breaking out of loops
 */
export declare enum LoopBreakAction {
    RESTORE_CHECKPOINT = "restore_checkpoint",
    CLEAR_CONTEXT = "clear_context",
    COMPRESS_HISTORY = "compress_history",
    CHANGE_STRATEGY = "change_strategy",
    INCREASE_TEMPERATURE = "increase_temperature",
    REQUEST_USER_INPUT = "request_user_input",
    RESET_FILE_STATE = "reset_file_state",
    SKIP_PROBLEMATIC_OPERATION = "skip_operation"
}
/**
 * Enhanced service for detecting and preventing infinite loops in AI responses.
 *
 * Features:
 * - Semantic content similarity detection beyond simple hashing
 * - Advanced tool call pattern recognition (alternating, non-consecutive)
 * - File system state tracking for operation loops
 * - Confidence-based escalation strategies
 * - Automatic loop breaking suggestions
 * - Visual feedback integration for user awareness
 * - Defensive programming patterns following Crisalida conventions
 */
export declare class LoopDetectionService {
    private readonly config;
    private promptId;
    private lastToolCallKey;
    private toolCallRepetitionCount;
    private streamContentHistory;
    private contentStats;
    private lastContentIndex;
    private loopDetected;
    private inCodeBlock;
    private temperatureOverride;
    private toolCallHistory;
    private fileOperationHistory;
    private semanticContentChunks;
    private recentChunkHashes;
    private recentContentEvents;
    private lastLoopConfidence;
    private consecutiveHighConfidenceChecks;
    private lastFailedToolCallName;
    private consecutiveFailedToolCallsCount;
    private eventSequenceNumber;
    private lastResetSequenceNumber;
    private turnsInCurrentPrompt;
    private _llmCheckInterval;
    private _lastCheckTurn;
    private pendingBreakActions;
    get llmCheckInterval(): number;
    set llmCheckInterval(v: number);
    get lastCheckTurn(): number;
    set lastCheckTurn(v: number);
    private confidenceListener?;
    private actionSuggestionListener?;
    private thinkingListener?;
    constructor(config: Config);
    /**
     * Returns a temporary temperature override if an automatic recovery is attempted.
     * The caller should use this value for the next generation call and then clear it.
     */
    getTemperatureOverride(): number | undefined;
    /**
     * Clears the temporary temperature override.
     */
    clearTemperatureOverride(): void;
    /**
     * Set callback for confidence level updates
     */
    setConfidenceListener(listener: (confidence: number, reasoning?: string) => void): void;
    /**
     * Set callback for action suggestions
     */
    setActionSuggestionListener(listener: (actions: LoopBreakAction[], reasoning: string) => void): void;
    /**
     * Set callback for thinking state updates
     */
    setThinkingListener(listener: (isThinking: boolean) => void): void;
    /**
     * Process user-initiated loop break command
     */
    handleUserLoopCommand(signal: AbortSignal): Promise<LoopDetectionResult>;
    /**
     * Track file system operations for state-based loop detection
     */
    trackFileOperation(filePath: string, operation: string, contentHash: string): void;
    private checkFileStateLoop;
    private getToolCallKey;
    /**
     * Enhanced loop detection with comprehensive analysis
     */
    addAndCheck(event: ServerGeminiStreamEvent): boolean;
    /**
     * New method to track the result of a tool call (success or failure).
     * This method should be called *after* a tool has been executed.
     */
    trackToolCallResult(toolCall: {
        name: string;
        args: object;
    }, isSuccess: boolean): boolean;
    private handleDetectedLoop;
    private trackToolCall;
    private checkAdvancedToolCallPatterns;
    private _checkAlternatingPattern;
    private _checkNonConsecutivePattern;
    private checkToolCallLoop;
    /**
     * Enhanced content loop detection with semantic analysis
     */
    private checkEnhancedContentLoop;
    private trackSemanticContent;
    private _analyzeSemanticContentLoop;
    /**
     * Detect if similar pairs form a repetitive pattern
     */
    private detectRepetitivePattern;
    /**
     * Turn management with enhanced LLM checking
     */
    turnStarted(signal: AbortSignal): Promise<boolean>;
    private shouldPerformLLMCheck;
    private performComprehensiveLoopCheck;
    private mapLoopTypeFromString;
    /**
     * Update confidence and notify listeners
     */
    private updateConfidence;
    /**
     * Generate adaptive break actions based on loop type and confidence
     */
    private generateAdaptiveBreakActions;
    private suggestLoopBreakActions;
    /**
     * Get pending break actions (for CLI integration)
     */
    getPendingBreakActions(): LoopBreakAction[];
    /**
     * Clear pending break actions after they've been handled
     */
    clearPendingBreakActions(): void;
    /**
     * Get current loop confidence level
     */
    getCurrentConfidence(): number;
    private truncateAndUpdate;
    private analyzeContentChunksForLoop;
    private logLoopDetected;
    private notifyActionSuggestions;
    /**
     * Reset all loop detection state
     */
    reset(promptId: string): void;
    private resetToolCallCount;
    private resetContentTracking;
    private resetLlmCheckTracking;
    private resetAdvancedTracking;
}
