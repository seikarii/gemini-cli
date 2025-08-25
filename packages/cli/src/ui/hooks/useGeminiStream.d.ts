/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Config, GeminiClient, EditorType, ThoughtSummary } from '@google/gemini-cli-core';
import { GeminiAgent } from '../../../../mew-upgrade/src/agent/gemini-agent.js';
import { StreamingState, HistoryItem, HistoryItemWithoutId, SlashCommandProcessorResult } from '../types.js';
import { UseHistoryManagerReturn } from './useHistoryManager.js';
/**
 * Manages the Gemini stream, including user input, command processing,
 * API interaction, and tool call lifecycle.
 */
export declare const useGeminiStream: (agent: GeminiAgent, geminiClient: GeminiClient, history: HistoryItem[], addItem: UseHistoryManagerReturn["addItem"], config: Config, onDebugMessage: (message: string) => void, handleSlashCommand: (cmd: PartListUnion) => Promise<SlashCommandProcessorResult | false>, shellModeActive: boolean, getPreferredEditor: () => EditorType | undefined, onAuthError: () => void, performMemoryRefresh: () => Promise<void>, modelSwitchedFromQuotaError: boolean, setModelSwitchedFromQuotaError: React.Dispatch<React.SetStateAction<boolean>>, onEditorClose: () => void, onCancelSubmit: () => void) => {
    streamingState: StreamingState;
    submitQuery: (query: PartListUnion, options?: {
        isContinuation: boolean;
    }, prompt_id?: string) => Promise<void>;
    initError: string | null;
    pendingHistoryItems: HistoryItemWithoutId[];
    thought: ThoughtSummary | null;
    cancelOngoingRequest: () => void;
};
