/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Config } from '@google/gemini-cli-core';
/**
 * Agent is an opaque runtime instance supplied by mew-upgrade. We keep the
 * parameter intentionally loose to avoid cross-package type coupling.
 */
type AgentLike = unknown;
/**
 * Optimized non-interactive CLI runner with improved performance and resource management
 */
export declare function runNonInteractive(agent: AgentLike, config: Config, input: string, prompt_id: string): Promise<void>;
export {};
