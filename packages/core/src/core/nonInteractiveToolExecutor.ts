/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ToolCallRequestInfo, ToolCallResponseInfo, Config } from '../index.js';
import { CoreToolScheduler } from './coreToolScheduler.js';
import { LoopDetectionService } from '../services/loopDetectionService.js';

/**
 * Executes a single tool call non-interactively by leveraging the CoreToolScheduler.
 */
export async function executeToolCall(
  config: Config,
  toolCallRequest: ToolCallRequestInfo,
  abortSignal: AbortSignal,
): Promise<ToolCallResponseInfo> {
  return new Promise<ToolCallResponseInfo>((resolve, reject) => {
    new CoreToolScheduler({
      config,
      getPreferredEditor: async () => undefined,
      onEditorClose: () => {},
      onAllToolCallsComplete: async (completedToolCalls) => {
        resolve(completedToolCalls[0].response);
      },
      loopDetectionService: new LoopDetectionService(config),
    })
      .schedule(toolCallRequest, abortSignal)
      .catch(reject);
  });
}
