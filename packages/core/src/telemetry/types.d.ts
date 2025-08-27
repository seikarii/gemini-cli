/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { GenerateContentResponseUsageMetadata } from '@google/genai';
import { Config } from '../config/config.js';
import { CompletedToolCall } from '../core/coreToolScheduler.js';
import { DiffStat } from '../tools/tools.js';
import { ToolCallDecision } from './tool-call-decision.js';
import { FileOperation } from './metrics.js';
export { ToolCallDecision };
import { ToolRegistry } from '../tools/tool-registry.js';
export interface BaseTelemetryEvent {
    'event.name': string;
    /** Current timestamp in ISO 8601 format */
    'event.timestamp': string;
}
type CommonFields = keyof BaseTelemetryEvent;
export declare class StartSessionEvent implements BaseTelemetryEvent {
    'event.name': 'cli_config';
    'event.timestamp': string;
    model: string;
    embedding_model: string;
    sandbox_enabled: boolean;
    core_tools_enabled: string;
    approval_mode: string;
    api_key_enabled: boolean;
    vertex_ai_enabled: boolean;
    debug_enabled: boolean;
    mcp_servers: string;
    telemetry_enabled: boolean;
    telemetry_log_user_prompts_enabled: boolean;
    file_filtering_respect_git_ignore: boolean;
    mcp_servers_count: number;
    mcp_tools_count?: number;
    mcp_tools?: string;
    constructor(config: Config, toolRegistry?: ToolRegistry);
}
export declare class EndSessionEvent implements BaseTelemetryEvent {
    'event.name': 'end_session';
    'event.timestamp': string;
    session_id?: string;
    constructor(config?: Config);
}
export declare class UserPromptEvent implements BaseTelemetryEvent {
    'event.name': 'user_prompt';
    'event.timestamp': string;
    prompt_length: number;
    prompt_id: string;
    auth_type?: string;
    prompt?: string;
    constructor(prompt_length: number, prompt_Id: string, auth_type?: string, prompt?: string);
}
export declare class ToolCallEvent implements BaseTelemetryEvent {
    'event.name': 'tool_call';
    'event.timestamp': string;
    function_name: string;
    function_args: Record<string, unknown>;
    duration_ms: number;
    success: boolean;
    decision?: ToolCallDecision;
    error?: string;
    error_type?: string;
    prompt_id: string;
    tool_type: 'native' | 'mcp';
    metadata?: {
        [key: string]: any;
    };
    constructor(call: CompletedToolCall);
}
export declare class ApiRequestEvent implements BaseTelemetryEvent {
    'event.name': 'api_request';
    'event.timestamp': string;
    model: string;
    prompt_id: string;
    request_text?: string;
    constructor(model: string, prompt_id: string, request_text?: string);
}
export declare class ApiErrorEvent implements BaseTelemetryEvent {
    'event.name': 'api_error';
    'event.timestamp': string;
    model: string;
    error: string;
    error_type?: string;
    status_code?: number | string;
    duration_ms: number;
    prompt_id: string;
    auth_type?: string;
    constructor(model: string, error: string, duration_ms: number, prompt_id: string, auth_type?: string, error_type?: string, status_code?: number | string);
}
export declare class ApiResponseEvent implements BaseTelemetryEvent {
    'event.name': 'api_response';
    'event.timestamp': string;
    model: string;
    status_code?: number | string;
    duration_ms: number;
    error?: string;
    input_token_count: number;
    output_token_count: number;
    cached_content_token_count: number;
    thoughts_token_count: number;
    tool_token_count: number;
    total_token_count: number;
    response_text?: string;
    prompt_id: string;
    auth_type?: string;
    constructor(model: string, duration_ms: number, prompt_id: string, auth_type?: string, usage_data?: GenerateContentResponseUsageMetadata, response_text?: string, error?: string);
}
export declare class FlashFallbackEvent implements BaseTelemetryEvent {
    'event.name': 'flash_fallback';
    'event.timestamp': string;
    auth_type: string;
    constructor(auth_type: string);
}
export declare enum LoopType {
    CONSECUTIVE_IDENTICAL_TOOL_CALLS = "consecutive_identical_tool_calls",
    CHANTING_IDENTICAL_SENTENCES = "chanting_identical_sentences",
    LLM_DETECTED_LOOP = "llm_detected_loop",
    ALTERNATING_TOOL_PATTERN = "alternating_tool_pattern",
    NON_CONSECUTIVE_TOOL_PATTERN = "non_consecutive_tool_pattern",
    FILE_STATE_LOOP = "file_state_loop",
    SEMANTIC_CONTENT_LOOP = "semantic_content_loop",
    CONSECUTIVE_FAILED_TOOL_CALLS = "consecutive_failed_tool_calls"
}
/**
 * Enhanced loop detection event with additional metadata
 */
export declare class LoopDetectedEvent {
    readonly loopType: LoopType;
    readonly promptId: string;
    readonly confidence?: number | undefined;
    readonly affectedFiles?: string[] | undefined;
    readonly toolsInvolved?: string[] | undefined;
    readonly reasoning?: string | undefined;
    constructor(loopType: LoopType, promptId: string, confidence?: number | undefined, affectedFiles?: string[] | undefined, toolsInvolved?: string[] | undefined, reasoning?: string | undefined);
}
/**
 * File system state for tracking modification patterns
 */
export interface FileSystemState {
    filePath: string;
    contentHash: string;
    timestamp: number;
    operation: string;
    size?: number;
}
export declare class NextSpeakerCheckEvent implements BaseTelemetryEvent {
    'event.name': 'next_speaker_check';
    'event.timestamp': string;
    prompt_id: string;
    finish_reason: string;
    result: string;
    constructor(prompt_id: string, finish_reason: string, result: string);
}
export interface SlashCommandEvent extends BaseTelemetryEvent {
    'event.name': 'slash_command';
    'event.timestamp': string;
    command: string;
    subcommand?: string;
    status?: SlashCommandStatus;
}
export declare function makeSlashCommandEvent({ command, subcommand, status, }: Omit<SlashCommandEvent, CommonFields>): SlashCommandEvent;
export declare enum SlashCommandStatus {
    SUCCESS = "success",
    ERROR = "error"
}
export interface ChatCompressionEvent extends BaseTelemetryEvent {
    'event.name': 'chat_compression';
    'event.timestamp': string;
    tokens_before: number;
    tokens_after: number;
}
export declare function makeChatCompressionEvent({ tokens_before, tokens_after, }: Omit<ChatCompressionEvent, CommonFields>): ChatCompressionEvent;
export declare class MalformedJsonResponseEvent implements BaseTelemetryEvent {
    'event.name': 'malformed_json_response';
    'event.timestamp': string;
    model: string;
    constructor(model: string);
}
export declare enum IdeConnectionType {
    START = "start",
    SESSION = "session"
}
export declare class IdeConnectionEvent {
    'event.name': 'ide_connection';
    'event.timestamp': string;
    connection_type: IdeConnectionType;
    constructor(connection_type: IdeConnectionType);
}
export declare class KittySequenceOverflowEvent {
    'event.name': 'kitty_sequence_overflow';
    'event.timestamp': string;
    sequence_length: number;
    truncated_sequence: string;
    constructor(sequence_length: number, truncated_sequence: string);
}
export declare class FileOperationEvent implements BaseTelemetryEvent {
    'event.name': 'file_operation';
    'event.timestamp': string;
    tool_name: string;
    operation: FileOperation;
    lines?: number;
    mimetype?: string;
    extension?: string;
    diff_stat?: DiffStat;
    programming_language?: string;
    constructor(tool_name: string, operation: FileOperation, lines?: number, mimetype?: string, extension?: string, diff_stat?: DiffStat, programming_language?: string);
}
export type TelemetryEvent = StartSessionEvent | EndSessionEvent | UserPromptEvent | ToolCallEvent | ApiRequestEvent | ApiErrorEvent | ApiResponseEvent | FlashFallbackEvent | LoopDetectedEvent | NextSpeakerCheckEvent | KittySequenceOverflowEvent | MalformedJsonResponseEvent | IdeConnectionEvent | SlashCommandEvent | FileOperationEvent;
