/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LoopDetectionService } from '../services/loopDetectionService.js';
import {
  ToolCallRequestInfo,
  ToolCallResponseInfo,
  ToolConfirmationOutcome,
  ToolCallConfirmationDetails,
  ToolResult,
  ToolResultDisplay,
  ToolRegistry,
  ApprovalMode,
  EditorType,
  Config,
  logToolCall,
  ToolCallEvent,
  ToolConfirmationPayload,
  ToolErrorType,
  AnyDeclarativeTool,
  AnyToolInvocation,
  ActionScriptRequestInfo,
} from '../index.js';
import { ActionSystem } from './action-system.js';
import { ActionScript, ScriptNode } from './action-script.js';
import { ActionPriority } from './action-system.js';
import { Part, PartListUnion } from '@google/genai';
import { getResponseTextFromParts } from '../utils/generateContentResponseUtilities.js';
import {
  isModifiableDeclarativeTool,
  ModifyContext,
  modifyWithEditor,
} from '../tools/modifiable-tool.js';
import * as Diff from 'diff';

export type ValidatingToolCall = {
  status: 'validating';
  request: ToolCallRequestInfo;
  tool: AnyDeclarativeTool;
  invocation: AnyToolInvocation;
  startTime?: number;
  outcome?: ToolConfirmationOutcome;
};

export type ScheduledToolCall = {
  status: 'scheduled';
  request: ToolCallRequestInfo;
  tool: AnyDeclarativeTool;
  invocation: AnyToolInvocation;
  startTime?: number;
  outcome?: ToolConfirmationOutcome;
};

export type ErroredToolCall = {
  status: 'error';
  request: ToolCallRequestInfo;
  response: ToolCallResponseInfo;
  tool?: AnyDeclarativeTool;
  durationMs?: number;
  outcome?: ToolConfirmationOutcome;
};

export type SuccessfulToolCall = {
  status: 'success';
  request: ToolCallRequestInfo;
  tool: AnyDeclarativeTool;
  response: ToolCallResponseInfo;
  invocation: AnyToolInvocation;
  durationMs?: number;
  outcome?: ToolConfirmationOutcome;
};

export type ExecutingToolCall = {
  status: 'executing';
  request: ToolCallRequestInfo;
  tool: AnyDeclarativeTool;
  invocation: AnyToolInvocation;
  liveOutput?: string;
  startTime?: number;
  outcome?: ToolConfirmationOutcome;
};

export type CancelledToolCall = {
  status: 'cancelled';
  request: ToolCallRequestInfo;
  response: ToolCallResponseInfo;
  tool: AnyDeclarativeTool;
  invocation: AnyToolInvocation;
  durationMs?: number;
  outcome?: ToolConfirmationOutcome;
};

export type WaitingToolCall = {
  status: 'awaiting_approval';
  request: ToolCallRequestInfo;
  tool: AnyDeclarativeTool;
  invocation: AnyToolInvocation;
  confirmationDetails: ToolCallConfirmationDetails;
  startTime?: number;
  outcome?: ToolConfirmationOutcome;
};

export type Status = ToolCall['status'];

export type ToolCall =
  | ValidatingToolCall
  | ScheduledToolCall
  | ErroredToolCall
  | SuccessfulToolCall
  | ExecutingToolCall
  | CancelledToolCall
  | WaitingToolCall;

export type CompletedToolCall =
  | SuccessfulToolCall
  | CancelledToolCall
  | ErroredToolCall;

export type ConfirmHandler = (
  toolCall: WaitingToolCall,
) => Promise<ToolConfirmationOutcome>;

export type OutputUpdateHandler = (
  toolCallId: string,
  outputChunk: string,
) => void;

export type AllToolCallsCompleteHandler = (
  completedToolCalls: CompletedToolCall[],
) => Promise<void>;

export type ToolCallsUpdateHandler = (toolCalls: ToolCall[]) => void;

/**
 * Formats tool output for a Gemini FunctionResponse.
 */
function createFunctionResponsePart(
  callId: string,
  toolName: string,
  output: string,
): Part {
  return {
    functionResponse: {
      id: callId,
      name: toolName,
      response: { output },
    },
  };
}

export function convertToFunctionResponse(
  toolName: string,
  callId: string,
  llmContent: PartListUnion,
): Part[] {
  const contentToProcess =
    Array.isArray(llmContent) && llmContent.length === 1
      ? llmContent[0]
      : llmContent;

  if (typeof contentToProcess === 'string') {
    return [createFunctionResponsePart(callId, toolName, contentToProcess)];
  }

  if (Array.isArray(contentToProcess)) {
    const functionResponse = createFunctionResponsePart(
      callId,
      toolName,
      'Tool execution succeeded.',
    );
    return [functionResponse, ...toParts(contentToProcess)];
  }

  // After this point, contentToProcess is a single Part object.
  if (contentToProcess.functionResponse) {
    if (contentToProcess.functionResponse.response?.['content']) {
      const stringifiedOutput =
        getResponseTextFromParts(
          contentToProcess.functionResponse.response['content'] as Part[],
        ) || '';
      return [createFunctionResponsePart(callId, toolName, stringifiedOutput)];
    }
    // It's a functionResponse that we should pass through as is.
    return [contentToProcess];
  }

  if (contentToProcess.inlineData || contentToProcess.fileData) {
    const mimeType =
      contentToProcess.inlineData?.mimeType ||
      contentToProcess.fileData?.mimeType ||
      'unknown';
    const functionResponse = createFunctionResponsePart(
      callId,
      toolName,
      `Binary content of type ${mimeType} was processed.`,
    );
    return [functionResponse, contentToProcess];
  }

  if (contentToProcess.text !== undefined) {
    return [
      createFunctionResponsePart(callId, toolName, contentToProcess.text),
    ];
  }

  // Default case for other kinds of parts.
  return [
    createFunctionResponsePart(callId, toolName, 'Tool execution succeeded.'),
  ];
}

function toParts(input: PartListUnion): Part[] {
  const parts: Part[] = [];
  for (const part of Array.isArray(input) ? input : [input]) {
    if (typeof part === 'string') {
      parts.push({ text: part });
    } else if (part) {
      parts.push(part);
    }
  }
  return parts;
}

const createErrorResponse = (
  request: ToolCallRequestInfo,
  error: Error,
  errorType: ToolErrorType | undefined,
): ToolCallResponseInfo => ({
  callId: request.callId,
  error,
  responseParts: [
    {
      functionResponse: {
        id: request.callId,
        name: request.name,
        response: { error: error.message },
      },
    },
  ],
  resultDisplay: error.message,
  errorType,
});

interface CoreToolSchedulerOptions {
  config: Config;
  outputUpdateHandler?: OutputUpdateHandler;
  onAllToolCallsComplete?: AllToolCallsCompleteHandler;
  onToolCallsUpdate?: ToolCallsUpdateHandler;
  getPreferredEditor: () => EditorType | undefined;
  onEditorClose: () => void;
  // Optional - if not provided a default LoopDetectionService will be constructed
  loopDetectionService?: LoopDetectionService;
}

export class CoreToolScheduler {
  private toolRegistry: ToolRegistry;
  private toolCalls: ToolCall[] = [];
  private outputUpdateHandler?: OutputUpdateHandler;
  private onAllToolCallsComplete?: AllToolCallsCompleteHandler;
  private onToolCallsUpdate?: ToolCallsUpdateHandler;
  private getPreferredEditor: () => EditorType | undefined;
  private config: Config;
  private onEditorClose: () => void;
  private isFinalizingToolCalls = false;
  private isScheduling = false;
  private requestQueue: Array<{
    request: ToolCallRequestInfo | ToolCallRequestInfo[];
    signal: AbortSignal;
    resolve: () => void;
    reject: (reason?: Error) => void;
  }> = [];
  private loopDetectionService: LoopDetectionService;

  constructor(options: CoreToolSchedulerOptions) {
    this.config = options.config;
    this.toolRegistry = options.config.getToolRegistry();
    this.outputUpdateHandler = options.outputUpdateHandler;
    this.onAllToolCallsComplete = options.onAllToolCallsComplete;
    this.onToolCallsUpdate = options.onToolCallsUpdate;
    this.getPreferredEditor = options.getPreferredEditor;
    this.onEditorClose = options.onEditorClose;
    // If caller didn't supply a LoopDetectionService, create a default instance tied to our config.
    this.loopDetectionService =
      options.loopDetectionService ?? new LoopDetectionService(this.config);
  }

  /**
   * Executes a tool with retry logic for transient errors
   */
  private async executeToolWithRetry(
    invocation: AnyToolInvocation,
    signal: AbortSignal,
    liveOutputCallback?: (output: string) => void,
    maxRetries: number = 3,
    baseDelayMs: number = 1000,
  ): Promise<ToolResult> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const result = await invocation.execute(signal, liveOutputCallback);

        // If successful or error is not retryable, return immediately
        if (!result.error || !this.isRetryableError(result.error.type)) {
          return result;
        }

        // If it's a retryable error but we've exhausted retries, return the error
        if (attempt === maxRetries) {
          return result;
        }

        lastError = new Error(result.error.message);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // If it's not a retryable error or we've exhausted retries, rethrow
        if (
          !this.isRetryableErrorFromException(lastError) ||
          attempt === maxRetries
        ) {
          throw lastError;
        }
      }

      // Wait before retrying with exponential backoff
      if (attempt < maxRetries) {
        const delay = baseDelayMs * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    // This should never be reached, but just in case
    throw lastError || new Error('Tool execution failed after retries');
  }

  /**
   * Determines if an error type is retryable
   */
  private isRetryableError(errorType?: ToolErrorType): boolean {
    if (!errorType) return false;

    const retryableErrors: ToolErrorType[] = [
      ToolErrorType.RETRY_TRANSIENT_FAILURE,
      ToolErrorType.SHELL_COMMAND_TIMEOUT,
      ToolErrorType.SHELL_COMMAND_RESOURCE_EXHAUSTED,
      // Add other transient errors as needed
    ];

    return retryableErrors.includes(errorType);
  }

  /**
   * Determines if an exception is retryable based on error properties
   */
  private isRetryableErrorFromException(error: Error): boolean {
    // Check for common transient error patterns
    const errorMessage = error.message.toLowerCase();
    const retryablePatterns = [
      'temporarily unavailable',
      'resource temporarily unavailable',
      'too many files open',
      'eagain',
      'emfile',
      'timeout',
      'connection reset',
      'network error',
    ];

    return retryablePatterns.some((pattern) => errorMessage.includes(pattern));
  }

  private setStatusInternal(
    targetCallId: string,
    status: 'success',
    response: ToolCallResponseInfo,
  ): void;
  private setStatusInternal(
    targetCallId: string,
    status: 'awaiting_approval',
    confirmationDetails: ToolCallConfirmationDetails,
  ): void;
  private setStatusInternal(
    targetCallId: string,
    status: 'error',
    response: ToolCallResponseInfo,
  ): void;
  private setStatusInternal(
    targetCallId: string,
    status: 'cancelled',
    reason: string,
  ): void;
  private setStatusInternal(
    targetCallId: string,
    status: 'executing' | 'scheduled' | 'validating',
  ): void;
  private setStatusInternal(
    targetCallId: string,
    newStatus: Status,
    auxiliaryData?: unknown,
  ): void {
    this.toolCalls = this.toolCalls.map((currentCall) => {
      if (
        currentCall.request.callId !== targetCallId ||
        currentCall.status === 'success' ||
        currentCall.status === 'error' ||
        currentCall.status === 'cancelled'
      ) {
        return currentCall;
      }

      // currentCall is a non-terminal state here and should have startTime and tool.
      const existingStartTime = currentCall.startTime;
      const toolInstance = currentCall.tool;
      const invocation = currentCall.invocation;

      const outcome = currentCall.outcome;

      switch (newStatus) {
        case 'success': {
          const durationMs = existingStartTime
            ? Date.now() - existingStartTime
            : undefined;
          return {
            request: currentCall.request,
            tool: toolInstance,
            invocation,
            status: 'success',
            response: auxiliaryData as ToolCallResponseInfo,
            durationMs,
            outcome,
          } as SuccessfulToolCall;
        }
        case 'error': {
          const durationMs = existingStartTime
            ? Date.now() - existingStartTime
            : undefined;
          return {
            request: currentCall.request,
            status: 'error',
            tool: toolInstance,
            response: auxiliaryData as ToolCallResponseInfo,
            durationMs,
            outcome,
          } as ErroredToolCall;
        }
        case 'awaiting_approval':
          return {
            request: currentCall.request,
            tool: toolInstance,
            status: 'awaiting_approval',
            confirmationDetails: auxiliaryData as ToolCallConfirmationDetails,
            startTime: existingStartTime,
            outcome,
            invocation,
          } as WaitingToolCall;
        case 'scheduled':
          return {
            request: currentCall.request,
            tool: toolInstance,
            status: 'scheduled',
            startTime: existingStartTime,
            outcome,
            invocation,
          } as ScheduledToolCall;
        case 'cancelled': {
          const durationMs = existingStartTime
            ? Date.now() - existingStartTime
            : undefined;

          // Preserve diff for cancelled edit operations
          let resultDisplay: ToolResultDisplay | undefined = undefined;
          if (currentCall.status === 'awaiting_approval') {
            const waitingCall = currentCall as WaitingToolCall;
            if (waitingCall.confirmationDetails.type === 'edit') {
              resultDisplay = {
                fileDiff: waitingCall.confirmationDetails.fileDiff,
                fileName: waitingCall.confirmationDetails.fileName,
                originalContent:
                  waitingCall.confirmationDetails.originalContent,
                newContent: waitingCall.confirmationDetails.newContent,
              };
            }
          }

          return {
            request: currentCall.request,
            tool: toolInstance,
            invocation,
            status: 'cancelled',
            response: {
              callId: currentCall.request.callId,
              responseParts: [
                {
                  functionResponse: {
                    id: currentCall.request.callId,
                    name: currentCall.request.name,
                    response: {
                      error: `[Operation Cancelled] Reason: ${auxiliaryData}`,
                    },
                  },
                },
              ],
              resultDisplay,
              error: undefined,
              errorType: undefined,
            },
            durationMs,
            outcome,
          } as CancelledToolCall;
        }
        case 'validating':
          return {
            request: currentCall.request,
            tool: toolInstance,
            status: 'validating',
            startTime: existingStartTime,
            outcome,
            invocation,
          } as ValidatingToolCall;
        case 'executing':
          return {
            request: currentCall.request,
            tool: toolInstance,
            status: 'executing',
            startTime: existingStartTime,
            outcome,
            invocation,
          } as ExecutingToolCall;
        default: {
          const exhaustiveCheck: never = newStatus;
          return exhaustiveCheck;
        }
      }
    });
    this.notifyToolCallsUpdate();
    this.checkAndNotifyCompletion();
  }

  private setArgsInternal(targetCallId: string, args: unknown): void {
    this.toolCalls = this.toolCalls.map((call) => {
      // We should never be asked to set args on an ErroredToolCall, but
      // we guard for the case anyways.
      if (call.request.callId !== targetCallId || call.status === 'error') {
        return call;
      }

      const invocationOrError = this.buildInvocation(
        call.tool,
        args as Record<string, unknown>,
      );
      if (invocationOrError instanceof Error) {
        const response = createErrorResponse(
          call.request,
          invocationOrError,
          ToolErrorType.INVALID_TOOL_PARAMS,
        );
        return {
          request: { ...call.request, args: args as Record<string, unknown> },
          status: 'error',
          tool: call.tool,
          response,
        } as ErroredToolCall;
      }

      return {
        ...call,
        request: { ...call.request, args: args as Record<string, unknown> },
        invocation: invocationOrError,
      };
    });
  }

  private isRunning(): boolean {
    return (
      this.isFinalizingToolCalls ||
      this.toolCalls.some(
        (call) =>
          call.status === 'executing' || call.status === 'awaiting_approval',
      )
    );
  }

  private buildInvocation(
    tool: AnyDeclarativeTool,
    args: object,
  ): AnyToolInvocation | Error {
    try {
      return tool.build(args);
    } catch (_e) {
      if (_e instanceof Error) {
        return _e;
      }
      return new Error(String(_e));
    }
  }

  schedule(
    request: ToolCallRequestInfo | ToolCallRequestInfo[],
    signal: AbortSignal,
  ): Promise<void> {
    if (this.isRunning() || this.isScheduling) {
      return new Promise((resolve, reject) => {
        const abortHandler = () => {
          // Find and remove the request from the queue
          const index = this.requestQueue.findIndex(
            (item) => item.request === request,
          );
          if (index > -1) {
            this.requestQueue.splice(index, 1);
            reject(new Error('Tool call cancelled while in queue.'));
          }
        };

        signal.addEventListener('abort', abortHandler, { once: true });

        this.requestQueue.push({
          request,
          signal,
          resolve: () => {
            signal.removeEventListener('abort', abortHandler);
            resolve();
          },
          reject: (reason?: Error) => {
            signal.removeEventListener('abort', abortHandler);
            reject(reason);
          },
        });
      });
    }
    return this._schedule(request, signal);
  }

  /**
   * Executes an Action Script using the ActionSystem
   */
  async executeActionScript(
    actionScriptRequest: ActionScriptRequestInfo,
    _signal: AbortSignal,
  ): Promise<ToolCallResponseInfo[]> {
    const actionSystem = new ActionSystem({
      maxConcurrentActions: 5,
      maxQueueSize: 100,
      enableActionHistory: true,
      maxHistorySize: 1000,
      autoCleanupCompleted: true,
      cleanupInterval: 300,
      enablePriorityScheduling: true,
      defaultTimeout: 300,
      adaptivePriority: true,
    });

    try {
      // Register available tools with the ActionSystem
      this.registerToolsWithActionSystem(actionSystem);

      // Convert Action Script to Action System actions
      const actions = this.convertActionScriptToActions(
        actionScriptRequest.script,
      );

      // Execute all actions
      const results: ToolCallResponseInfo[] = [];
      for (const action of actions) {
        const actionId = actionSystem.createAction(
          action.toolName,
          action.parameters,
          action.priority,
        );

        // Wait for the action to complete
        // Note: This is a simplified implementation. In a real scenario,
        // you'd want to use the ActionSystem's event system or polling
        await new Promise((resolve) => setTimeout(resolve, 100)); // Temporary delay

        // For now, we'll create a mock result
        const toolResult = await this.convertActionResultToToolResult(
          actionId,
          actionScriptRequest.callId,
        );
        results.push(toolResult);
      }

      return results;
    } catch (error) {
      throw new Error(`Action Script execution failed: ${error}`);
    }
  }

  private async _schedule(
    request: ToolCallRequestInfo | ToolCallRequestInfo[],
    signal: AbortSignal,
  ): Promise<void> {
    this.isScheduling = true;
    try {
      if (this.isRunning()) {
        throw new Error(
          'Cannot schedule new tool calls while other tool calls are actively running (executing or awaiting approval).',
        );
      }
      const requestsToProcess = Array.isArray(request) ? request : [request];

      const newToolCalls: ToolCall[] = requestsToProcess.map(
        (reqInfo): ToolCall => {
          const toolInstance = this.toolRegistry.getTool(reqInfo.name);
          if (!toolInstance) {
            return {
              status: 'error',
              request: reqInfo,
              response: createErrorResponse(
                reqInfo,
                new Error(`Tool "${reqInfo.name}" not found in registry.`),
                ToolErrorType.TOOL_NOT_REGISTERED,
              ),
              durationMs: 0,
            };
          }

          const invocationOrError = this.buildInvocation(
            toolInstance,
            reqInfo.args,
          );
          if (invocationOrError instanceof Error) {
            return {
              status: 'error',
              request: reqInfo,
              tool: toolInstance,
              response: createErrorResponse(
                reqInfo,
                invocationOrError,
                ToolErrorType.INVALID_TOOL_PARAMS,
              ),
              durationMs: 0,
            };
          }

          return {
            status: 'validating',
            request: reqInfo,
            tool: toolInstance,
            invocation: invocationOrError,
            startTime: Date.now(),
          };
        },
      );

      this.toolCalls = this.toolCalls.concat(newToolCalls);
      this.notifyToolCallsUpdate();

      for (const toolCall of newToolCalls) {
        if (toolCall.status !== 'validating') {
          continue;
        }

        const { request: reqInfo, invocation } = toolCall;

        try {
          if (signal.aborted) {
            this.setStatusInternal(
              reqInfo.callId,
              'cancelled',
              'Tool call cancelled by user.',
            );
            continue;
          }
          if (this.config.getApprovalMode() === ApprovalMode.YOLO) {
            this.setToolCallOutcome(
              reqInfo.callId,
              ToolConfirmationOutcome.ProceedAlways,
            );
            this.setStatusInternal(reqInfo.callId, 'scheduled');
          } else {
            const confirmationDetails =
              await invocation.shouldConfirmExecute(signal);

            if (confirmationDetails) {
              // Allow IDE to resolve confirmation
              if (
                confirmationDetails.type === 'edit' &&
                confirmationDetails.ideConfirmation
              ) {
                confirmationDetails.ideConfirmation.then((resolution) => {
                  if (resolution.status === 'accepted') {
                    this.handleConfirmationResponse(
                      reqInfo.callId,
                      confirmationDetails.onConfirm,
                      ToolConfirmationOutcome.ProceedOnce,
                      signal,
                    );
                  } else {
                    this.handleConfirmationResponse(
                      reqInfo.callId,
                      confirmationDetails.onConfirm,
                      ToolConfirmationOutcome.Cancel,
                      signal,
                    );
                  }
                });
              }

              const originalOnConfirm = confirmationDetails.onConfirm;
              const wrappedConfirmationDetails: ToolCallConfirmationDetails = {
                ...confirmationDetails,
                onConfirm: (
                  outcome: ToolConfirmationOutcome,
                  payload?: ToolConfirmationPayload,
                ) =>
                  this.handleConfirmationResponse(
                    reqInfo.callId,
                    originalOnConfirm,
                    outcome,
                    signal,
                    payload,
                  ),
              };
              this.setStatusInternal(
                reqInfo.callId,
                'awaiting_approval',
                wrappedConfirmationDetails,
              );
            } else {
              this.setToolCallOutcome(
                reqInfo.callId,
                ToolConfirmationOutcome.ProceedAlways,
              );
              this.setStatusInternal(reqInfo.callId, 'scheduled');
            }
          }
        } catch (error) {
          this.setStatusInternal(
            reqInfo.callId,
            'error',
            createErrorResponse(
              reqInfo,
              error instanceof Error ? error : new Error(String(error)),
              ToolErrorType.UNHANDLED_EXCEPTION,
            ),
          );
        }
      }
      this.attemptExecutionOfScheduledCalls(signal);
      void this.checkAndNotifyCompletion();
    } finally {
      this.isScheduling = false;
    }
  }

  async handleConfirmationResponse(
    callId: string,
    originalOnConfirm: (outcome: ToolConfirmationOutcome) => Promise<void>,
    outcome: ToolConfirmationOutcome,
    signal: AbortSignal,
    payload?: ToolConfirmationPayload,
  ): Promise<void> {
    const toolCall = this.toolCalls.find(
      (c) => c.request.callId === callId && c.status === 'awaiting_approval',
    );

    if (toolCall && toolCall.status === 'awaiting_approval') {
      await originalOnConfirm(outcome);
    }

    if (outcome === ToolConfirmationOutcome.ProceedAlways) {
      await this.autoApproveCompatiblePendingTools(signal, callId);
    }

    this.setToolCallOutcome(callId, outcome);

    if (outcome === ToolConfirmationOutcome.Cancel || signal.aborted) {
      this.setStatusInternal(
        callId,
        'cancelled',
        'User did not allow tool call',
      );
    } else if (outcome === ToolConfirmationOutcome.ModifyWithEditor) {
      const waitingToolCall = toolCall as WaitingToolCall;
      if (isModifiableDeclarativeTool(waitingToolCall.tool)) {
        const modifyContext = waitingToolCall.tool.getModifyContext(signal);
        const editorType = this.getPreferredEditor();
        if (!editorType) {
          return;
        }

        this.setStatusInternal(callId, 'awaiting_approval', {
          ...waitingToolCall.confirmationDetails,
          isModifying: true,
        } as ToolCallConfirmationDetails);

        const { updatedParams, updatedDiff } = await modifyWithEditor<
          typeof waitingToolCall.request.args
        >(
          waitingToolCall.request.args,
          modifyContext as ModifyContext<typeof waitingToolCall.request.args>,
          editorType,
          signal,
          this.onEditorClose,
        );
        this.setArgsInternal(callId, updatedParams);
        this.setStatusInternal(callId, 'awaiting_approval', {
          ...waitingToolCall.confirmationDetails,
          fileDiff: updatedDiff,
          isModifying: false,
        } as ToolCallConfirmationDetails);
      }
    } else {
      // If the client provided new content, apply it before scheduling.
      if (payload?.newContent && toolCall) {
        await this._applyInlineModify(
          toolCall as WaitingToolCall,
          payload,
          signal,
        );
      }
      this.setStatusInternal(callId, 'scheduled');
    }
    this.attemptExecutionOfScheduledCalls(signal);
  }

  /**
   * Applies user-provided content changes to a tool call that is awaiting confirmation.
   * This method updates the tool's arguments and refreshes the confirmation prompt with a new diff
   * before the tool is scheduled for execution.
   * @private
   */
  private async _applyInlineModify(
    toolCall: WaitingToolCall,
    payload: ToolConfirmationPayload,
    signal: AbortSignal,
  ): Promise<void> {
    if (
      toolCall.confirmationDetails.type !== 'edit' ||
      !isModifiableDeclarativeTool(toolCall.tool)
    ) {
      return;
    }

    const modifyContext = toolCall.tool.getModifyContext(signal);
    const currentContent = await modifyContext.getCurrentContent(
      toolCall.request.args,
    );

    const updatedParams = modifyContext.createUpdatedParams(
      currentContent,
      payload.newContent,
      toolCall.request.args,
    );
    const updatedDiff = Diff.createPatch(
      modifyContext.getFilePath(toolCall.request.args),
      currentContent,
      payload.newContent,
      'Current',
      'Proposed',
    );

    this.setArgsInternal(toolCall.request.callId, updatedParams);
    this.setStatusInternal(toolCall.request.callId, 'awaiting_approval', {
      ...toolCall.confirmationDetails,
      fileDiff: updatedDiff,
    });
  }

  private attemptExecutionOfScheduledCalls(signal: AbortSignal): void {
    const allCallsFinalOrScheduled = this.toolCalls.every(
      (call) =>
        call.status === 'scheduled' ||
        call.status === 'cancelled' ||
        call.status === 'success' ||
        call.status === 'error',
    );

    if (allCallsFinalOrScheduled) {
      const callsToExecute = this.toolCalls.filter(
        (call) => call.status === 'scheduled',
      );

      callsToExecute.forEach((toolCall) => {
        if (toolCall.status !== 'scheduled') return;

        const scheduledCall = toolCall;
        const { callId, name: toolName } = scheduledCall.request;
        const invocation = scheduledCall.invocation;
        this.setStatusInternal(callId, 'executing');

        const liveOutputCallback =
          scheduledCall.tool.canUpdateOutput && this.outputUpdateHandler
            ? (outputChunk: string) => {
                if (this.outputUpdateHandler) {
                  this.outputUpdateHandler(callId, outputChunk);
                }
                this.toolCalls = this.toolCalls.map((tc) =>
                  tc.request.callId === callId && tc.status === 'executing'
                    ? { ...tc, liveOutput: outputChunk }
                    : tc,
                );
                this.notifyToolCallsUpdate();
              }
            : undefined;

        this.executeToolWithRetry(invocation, signal, liveOutputCallback)
          .then(async (toolResult: ToolResult) => {
            if (signal.aborted) {
              this.setStatusInternal(
                callId,
                'cancelled',
                'User cancelled tool execution.',
              );
              return;
            }

            if (toolResult.error === undefined) {
              const response = convertToFunctionResponse(
                toolName,
                callId,
                toolResult.llmContent,
              );
              const successResponse: ToolCallResponseInfo = {
                callId,
                responseParts: response,
                resultDisplay: toolResult.returnDisplay,
                error: undefined,
                errorType: undefined,
              };
              this.setStatusInternal(callId, 'success', successResponse);
              this.loopDetectionService.trackToolCallResult(
                scheduledCall.request,
                true,
              );
            } else {
              // It is a failure
              const error = new Error(toolResult.error.message);
              const errorResponse = createErrorResponse(
                scheduledCall.request,
                error,
                toolResult.error.type,
              );
              this.setStatusInternal(callId, 'error', errorResponse);
              this.loopDetectionService.trackToolCallResult(
                scheduledCall.request,
                false,
              );
            }
          })
          .catch((executionError: Error) => {
            this.setStatusInternal(
              callId,
              'error',
              createErrorResponse(
                scheduledCall.request,
                executionError instanceof Error
                  ? executionError
                  : new Error(String(executionError)),
                ToolErrorType.UNHANDLED_EXCEPTION,
              ),
            );
            this.loopDetectionService.trackToolCallResult(
              scheduledCall.request,
              false,
            );
          });
      });
    }
  }

  private async checkAndNotifyCompletion(): Promise<void> {
    const allCallsAreTerminal = this.toolCalls.every(
      (call) =>
        call.status === 'success' ||
        call.status === 'error' ||
        call.status === 'cancelled',
    );

    if (this.toolCalls.length > 0 && allCallsAreTerminal) {
      const completedCalls = [...this.toolCalls] as CompletedToolCall[];
      this.toolCalls = [];

      for (const call of completedCalls) {
        logToolCall(this.config, new ToolCallEvent(call));
      }

      if (this.onAllToolCallsComplete) {
        this.isFinalizingToolCalls = true;
        await this.onAllToolCallsComplete(completedCalls);
        this.isFinalizingToolCalls = false;
      }
      this.notifyToolCallsUpdate();
      // After completion, process the next item in the queue.
      if (this.requestQueue.length > 0) {
        const next = this.requestQueue.shift()!;
        this._schedule(next.request, next.signal)
          .then(next.resolve)
          .catch(next.reject);
      }
    }
  }

  private notifyToolCallsUpdate(): void {
    if (this.onToolCallsUpdate) {
      this.onToolCallsUpdate([...this.toolCalls]);
    }
  }

  private setToolCallOutcome(callId: string, outcome: ToolConfirmationOutcome) {
    this.toolCalls = this.toolCalls.map((call) => {
      if (call.request.callId !== callId) return call;
      return {
        ...call,
        outcome,
      };
    });
  }

  private async autoApproveCompatiblePendingTools(
    signal: AbortSignal,
    triggeringCallId: string,
  ): Promise<void> {
    const pendingTools = this.toolCalls.filter(
      (call) =>
        call.status === 'awaiting_approval' &&
        call.request.callId !== triggeringCallId,
    ) as WaitingToolCall[];

    for (const pendingTool of pendingTools) {
      try {
        const stillNeedsConfirmation =
          await pendingTool.invocation.shouldConfirmExecute(signal);

        if (!stillNeedsConfirmation) {
          this.setToolCallOutcome(
            pendingTool.request.callId,
            ToolConfirmationOutcome.ProceedAlways,
          );
          this.setStatusInternal(pendingTool.request.callId, 'scheduled');
        }
      } catch (error) {
        console.error(
          `Error checking confirmation for tool ${pendingTool.request.callId}:`,
          error,
        );
      }
    }
  }

  /**
   * Registers available tools with the ActionSystem
   */
  private registerToolsWithActionSystem(actionSystem: ActionSystem): void {
    // Get all available tools from the registry
    const toolDeclarations = this.toolRegistry.getFunctionDeclarations();

    for (const toolDecl of toolDeclarations) {
      if (!toolDecl.name) continue;

      // Create a wrapper function that uses the tool registry
      const toolWrapper = async (...args: unknown[]): Promise<unknown> => {
        try {
          const tool = this.toolRegistry.getTool(toolDecl.name!);
          if (!tool) {
            throw new Error(`Tool ${toolDecl.name!} not found in registry`);
          }

          // Create a tool invocation
          const invocation = this.buildInvocation(tool, { args });
          if (invocation instanceof Error) {
            throw invocation;
          }

          // Execute the tool
          const result = await invocation.execute(new AbortController().signal);
          return result;
        } catch (error) {
          throw new Error(`Tool execution failed: ${error}`);
        }
      };

      actionSystem.registerTool(toolDecl.name, toolWrapper);
    }
  }

  /**
   * Converts an Action Script to Action System actions
   */
  private convertActionScriptToActions(script: ActionScript): Array<{
    toolName: string;
    parameters: Record<string, unknown>;
    priority: ActionPriority;
  }> {
    const actions: Array<{
      toolName: string;
      parameters: Record<string, unknown>;
      priority: ActionPriority;
    }> = [];

    // Convert the root node of the script
    this.convertScriptNodeToActions(script.rootNode, actions);

    return actions;
  }

  /**
   * Recursively converts script nodes to actions
   */
  private convertScriptNodeToActions(
    node: ScriptNode,
    actions: Array<{
      toolName: string;
      parameters: Record<string, unknown>;
      priority: ActionPriority;
    }>,
  ): void {
    switch (node.type) {
      case 'action':
        actions.push({
          toolName: node.toolName,
          parameters: node.parameters,
          priority: node.priority || ActionPriority.NORMAL,
        });
        break;

      case 'sequence':
      case 'parallel':
        for (const childNode of node.nodes) {
          this.convertScriptNodeToActions(childNode, actions);
        }
        break;

      case 'condition':
        // For conditions, we execute the thenNode by default
        // In a more sophisticated implementation, we'd evaluate the condition
        this.convertScriptNodeToActions(node.thenNode, actions);
        break;

      case 'loop':
        // For loops, we execute the body once for simplicity
        // In a more sophisticated implementation, we'd handle iteration
        this.convertScriptNodeToActions(node.body, actions);
        break;

      case 'variable':
        // Variables don't create actions
        break;

      default:
        console.warn(
          `Unknown script node type: ${(node as ScriptNode & { type: string }).type}`,
        );
        break;
    }
  }

  /**
   * Converts an Action result to ToolCallResponseInfo format
   */
  private async convertActionResultToToolResult(
    actionResult: unknown,
    callId: string,
  ): Promise<ToolCallResponseInfo> {
    // This is a simplified conversion
    // In a real implementation, you'd need to map Action results to the expected format
    return {
      callId,
      responseParts: [
        {
          functionResponse: {
            id: callId,
            name: 'actionScript',
            response: {
              result: actionResult,
            },
          },
        },
      ],
      resultDisplay: undefined,
      error: undefined,
      errorType: undefined,
    };
  }
}
