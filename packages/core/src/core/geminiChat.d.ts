/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { GenerateContentResponse, Content, GenerateContentConfig, SendMessageParameters, Tool } from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { Config } from '../config/config.js';
import { StructuredError } from './turn.js';
/**
 * Custom error to signal that a stream completed without valid content,
 * which should trigger a retry.
 */
export declare class EmptyStreamError extends Error {
    constructor(message: string);
}
/**
 * Chat session that enables sending messages to the model with previous
 * conversation context.
 *
 * @remarks
 * The session maintains all the turns between user and model.
 */
export declare class GeminiChat {
    private readonly config;
    private readonly contentGenerator;
    private readonly generationConfig;
    private history;
    private sendPromise;
    private historyCache;
    private retryCount;
    private lastPerformanceCheck;
    constructor(config: Config, contentGenerator: ContentGenerator, generationConfig?: GenerateContentConfig, history?: Content[]);
    /**
     * Handles falling back to Flash model when persistent 429 errors occur for OAuth users.
     * Uses a fallback handler if provided by the config; otherwise, returns null.
     * Enhanced with better retry tracking and performance monitoring.
     */
    private handleFlashFallback;
    setSystemInstruction(sysInstr: string): void;
    /**
     * Sends a message to the model and returns the response.
     * Enhanced with improved retry logic and performance monitoring.
     *
     * @remarks
     * This method will wait for the previous message to be processed before
     * sending the next message.
     *
     * @see {@link Chat#sendMessageStream} for streaming method.
     * @param params - parameters for sending messages within a chat session.
     * @returns The model's response.
     *
     * @example
     * ```ts
     * const chat = ai.chats.create({model: 'gemini-2.0-flash'});
     * const response = await chat.sendMessage({
     *   message: 'Why is the sky blue?'
     * });
     * console.log(response.text);
     * ```
     */
    sendMessage(params: SendMessageParameters, prompt_id: string): Promise<GenerateContentResponse>;
    /**
     * Optimized method to process response and update history.
     * Separated for better maintainability and error handling.
     */
    private processResponseAndUpdateHistory;
    /**
     * Sends a message to the model and returns the response in chunks.
     *
     * @remarks
     * This method will wait for the previous message to be processed before
     * sending the next message.
     *
     * @see {@link Chat#sendMessage} for non-streaming method.
     * @param params - parameters for sending the message.
     * @return The model's response.
     *
     * @example
     * ```ts
     * const chat = ai.chats.create({model: 'gemini-2.0-flash'});
     * const response = await chat.sendMessageStream({
     *   message: 'Why is the sky blue?'
     * });
     * for await (const chunk of response) {
     *   console.log(chunk.text);
     * }
     * ```
     */
    sendMessageStream(params: SendMessageParameters, prompt_id: string): Promise<AsyncGenerator<GenerateContentResponse>>;
    private makeApiCallAndProcessStream;
    /**
     * Returns the chat history with optimized cloning strategy.
     *
     * @remarks
     * The history is a list of contents alternating between user and model.
     *
     * There are two types of history:
     * - The `curated history` contains only the valid turns between user and
     * model, which will be included in the subsequent requests sent to the model.
     * - The `comprehensive history` contains all turns, including invalid or
     *   empty model outputs, providing a complete record of the history.
     *
     * The history is updated after receiving the response from the model,
     * for streaming response, it means receiving the last chunk of the response.
     *
     * The `comprehensive history` is returned by default. To get the `curated
     * history`, set the `curated` parameter to `true`.
     *
     * @param curated - whether to return the curated history or the comprehensive
     *     history.
     * @return History contents alternating between user and model for the entire
     *     chat session.
     */
    getHistory(curated?: boolean): Content[];
    /**
     * Clears the chat history and resets performance counters.
     */
    clearHistory(): void;
    /**
     * Adds a new entry to the chat history and invalidates cache.
     */
    addHistory(content: Content): void;
    /**
     * Sets the entire history and invalidates cache.
     */
    setHistory(history: Content[]): void;
    setTools(tools: Tool[]): void;
    maybeIncludeSchemaDepthContext(error: StructuredError): Promise<void>;
    private processStreamResponse;
    private recordHistory;
    private hasTextContent;
    private isThoughtContent;
}
/** Visible for Testing */
export declare function isSchemaDepthError(errorMessage: string): boolean;
export declare function isInvalidArgumentError(errorMessage: string): boolean;
