/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Config } from '@google/gemini-cli-core';
import { GeminiAgent } from '@google/gemini-cli-mew-upgrade/agent/gemini-agent.js';
export declare function runNonInteractive(agent: GeminiAgent, config: Config, input: string, prompt_id: string): Promise<void>;
