/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Config,
  ToolCallRequestInfo,
  executeToolCall,
  shutdownTelemetry,
  isTelemetrySdkInitialized,
  GeminiEventType,
  parseAndFormatApiError,
} from '@google/gemini-cli-core';
import { Content, Part } from '@google/genai';
// Avoid importing GeminiAgent type here to prevent cross-package type coupling.
// Use a relaxed runtime type for the agent parameter.
import { ConsolePatcher } from './ui/utils/ConsolePatcher.js';
import { handleAtCommand } from './ui/hooks/atCommandProcessor.js';

/*
 * OPTIMIZED NON-INTERACTIVE CLI MODULE
 *
 * Performance optimizations implemented:
 * 1. OutputBuffer: Efficient stdout buffering for large outputs (64KB buffer, 100ms flush interval)
 * 2. ToolCallCache: LRU cache for read-only tool calls (50 items, 5min TTL)
 * 3. Optimized session management: Reasonable maxTurns defaults for non-interactive use
 * 4. Improved error handling and resource cleanup
 * 5. Chunked memory ingestion for large files (32KB chunks)
 * 6. Asynchronous telemetry shutdown to avoid blocking
 * 7. Performance tracking and debug logging
 * 8. Early abort signal checking to reduce unnecessary work
 */

/**
 * Optimized output buffer to improve stdout write performance for large outputs
 */
class OutputBuffer {
  private buffer: string[] = [];
  private bufferSize = 0;
  private readonly maxBufferSize: number;
  private readonly flushInterval: number;
  private flushTimer?: NodeJS.Timeout;

  constructor(maxBufferSize = 64 * 1024, flushInterval = 100) {
    // 64KB buffer, 100ms flush interval
    this.maxBufferSize = maxBufferSize;
    this.flushInterval = flushInterval;
  }

  write(data: string): void {
    this.buffer.push(data);
    this.bufferSize += data.length;

    // Auto-flush if buffer is getting large
    if (this.bufferSize >= this.maxBufferSize) {
      this.flush();
    } else if (!this.flushTimer) {
      // Schedule flush to ensure timely output
      this.flushTimer = setTimeout(() => this.flush(), this.flushInterval);
    }
  }

  flush(): void {
    if (this.buffer.length > 0) {
      const output = this.buffer.join('');
      process.stdout.write(output);
      this.buffer = [];
      this.bufferSize = 0;
    }

    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = undefined;
    }
  }

  cleanup(): void {
    this.flush();
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = undefined;
    }
  }
}

/**
 * Simple LRU cache for tool call results to avoid redundant operations
 */
class ToolCallCache {
  private cache = new Map<string, { result: unknown; timestamp: number }>();
  private readonly maxSize: number;
  private readonly ttl: number; // Time to live in ms

  constructor(maxSize = 50, ttl = 5 * 60 * 1000) {
    // 50 items, 5 minutes TTL
    this.maxSize = maxSize;
    this.ttl = ttl;
  }

  private generateKey(requestInfo: ToolCallRequestInfo): string {
    return `${requestInfo.name}:${JSON.stringify(requestInfo.args)}`;
  }

  get(requestInfo: ToolCallRequestInfo): unknown | null {
    const key = this.generateKey(requestInfo);
    const entry = this.cache.get(key);

    if (!entry) return null;

    // Check if entry has expired
    if (Date.now() - entry.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.result;
  }

  set(requestInfo: ToolCallRequestInfo, result: unknown): void {
    const key = this.generateKey(requestInfo);

    // Remove oldest entries if cache is full
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
      }
    }

    this.cache.set(key, { result, timestamp: Date.now() });
  }

  clear(): void {
    this.cache.clear();
  }
}

/**
 * Agent is an opaque runtime instance supplied by mew-upgrade. We keep the
 * parameter intentionally loose to avoid cross-package type coupling.
 */
type AgentLike = unknown;

/**
 * Optimized non-interactive CLI runner with improved performance and resource management
 */
export async function runNonInteractive(
  agent: AgentLike,
  config: Config,
  input: string,
  prompt_id: string,
): Promise<void> {
  const consolePatcher = new ConsolePatcher({
    stderr: true,
    debugMode: config.getDebugMode(),
  });

  // Initialize optimized output buffer for better stdout performance
  const outputBuffer = new OutputBuffer();

  // Initialize tool call cache for better performance with repeated operations
  const toolCache = new ToolCallCache();

  // Performance tracking
  const startTime = Date.now();
  let totalToolCalls = 0;

  try {
    consolePatcher.patch();

    // Handle EPIPE errors when the output is piped to a command that closes early.
    process.stdout.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EPIPE') {
        // Exit gracefully if the pipe is closed.
        outputBuffer.cleanup();
        process.exit(0);
      }
    });

    const geminiClient = config.getGeminiClient();
    const abortController = new AbortController();

    // Optimize session turn limits for non-interactive use
    const maxTurns = config.getMaxSessionTurns();
    const effectiveMaxTurns = maxTurns >= 0 ? Math.min(maxTurns, 50) : 20; // Reasonable default for non-interactive

    // Process @ commands with optimized error handling
    const atCommandResult = await handleAtCommand({
      query: input,
      config,
      addItem: (_item, _timestamp) => 0,
      onDebugMessage: () => {},
      messageId: Date.now(),
      signal: abortController.signal,
    });

    const { processedQuery, shouldProceed } = atCommandResult;

    if (!shouldProceed || !processedQuery) {
      // An error occurred during @include processing (e.g., file not found).
      // The error message is already logged by handleAtCommand.
      console.error('Exiting due to an error processing the @ command.');
      process.exit(1);
    }

    let currentMessages: Content[] = [
      { role: 'user', parts: processedQuery as Part[] },
    ];

    let turnCount = 0;
    let hasMoreTurns = true;

    // Optimized main processing loop
    while (hasMoreTurns && turnCount < effectiveMaxTurns) {
      turnCount++;

      // Check for abort signal early
      if (abortController.signal.aborted) {
        outputBuffer.write('\nOperation cancelled.\n');
        outputBuffer.flush();
        return;
      }

      const toolCallRequests: ToolCallRequestInfo[] = [];

      try {
        const responseStream = geminiClient.sendMessageStream(
          currentMessages[0]?.parts || [],
          abortController.signal,
          prompt_id,
        );

        // Process streaming response with optimized buffering
        for await (const event of responseStream) {
          if (abortController.signal.aborted) {
            outputBuffer.write('\nOperation cancelled.\n');
            outputBuffer.flush();
            return;
          }

          if (event.type === GeminiEventType.Content) {
            outputBuffer.write(event.value);
          } else if (event.type === GeminiEventType.ToolCallRequest) {
            toolCallRequests.push(event.value);
          }
        }

        // Flush accumulated content before processing tool calls
        outputBuffer.flush();
      } catch (streamError) {
        outputBuffer.flush();
        throw streamError;
      }

      // Process tool calls with optimized execution
      if (toolCallRequests.length > 0) {
        totalToolCalls += toolCallRequests.length;
        const toolResponseParts = await processToolCallsOptimized(
          config,
          agent,
          toolCallRequests,
          abortController.signal,
          toolCache,
        );

        if (toolResponseParts.length > 0) {
          currentMessages = [{ role: 'user', parts: toolResponseParts }];
        } else {
          hasMoreTurns = false;
        }
      } else {
        hasMoreTurns = false;
      }
    }

    // Handle session limit reached
    if (turnCount >= effectiveMaxTurns && hasMoreTurns) {
      const warningMsg =
        `\nReached max session turns (${effectiveMaxTurns}) for this session. ` +
        'Increase the number of turns by specifying maxSessionTurns in settings.json.\n';
      outputBuffer.write(warningMsg);
    }

    // Ensure final newline and flush
    outputBuffer.write('\n');
    outputBuffer.flush();

    // Performance logging in debug mode
    if (config.getDebugMode()) {
      const duration = Date.now() - startTime;
      console.error(
        `[DEBUG] Session completed: ${turnCount} turns, ${totalToolCalls} tool calls, ${duration}ms`,
      );
    }
  } catch (error) {
    outputBuffer.flush();
    console.error(
      parseAndFormatApiError(
        error,
        config.getContentGeneratorConfig()?.authType,
      ),
    );
    process.exit(1);
  } finally {
    outputBuffer.cleanup();
    consolePatcher.cleanup();

    // Optimized telemetry shutdown - don't block the process
    if (isTelemetrySdkInitialized()) {
      // Run telemetry shutdown asynchronously to avoid blocking
      setImmediate(async () => {
        try {
          await Promise.race([
            shutdownTelemetry(config),
            new Promise((_, reject) =>
              setTimeout(
                () => reject(new Error('Telemetry shutdown timeout')),
                1000,
              ),
            ),
          ]);
        } catch (telemetryError) {
          if (config.getDebugMode()) {
            console.error('[DEBUG] Telemetry shutdown error:', telemetryError);
          }
        }
      });
    }
  }
}

/**
 * Optimized tool call processing with better error handling and memory management
 */
async function processToolCallsOptimized(
  config: Config,
  agent: AgentLike,
  toolCallRequests: ToolCallRequestInfo[],
  signal: AbortSignal,
  cache?: ToolCallCache,
): Promise<Part[]> {
  const toolResponseParts: Part[] = [];
  const errors: string[] = [];

  // Process tool calls sequentially to maintain order and avoid race conditions
  // Note: Parallel execution could be implemented for independent tools in the future
  for (const requestInfo of toolCallRequests) {
    if (signal.aborted) {
      break;
    }

    try {
      // Check cache for read-only operations that can be safely cached
      const isCacheable = isCacheableToolCall(requestInfo);
      let toolResponse;

      if (isCacheable && cache) {
        const cachedResult = cache.get(requestInfo);
        if (cachedResult) {
          // Cast cached result back to expected type
          toolResponse = cachedResult as Awaited<
            ReturnType<typeof executeToolCall>
          >;
          if (config.getDebugMode()) {
            console.error(
              `[DEBUG] Using cached result for tool ${requestInfo.name}`,
            );
          }
        }
      }

      // Execute tool call if not cached
      if (!toolResponse) {
        toolResponse = await executeToolCall(config, requestInfo, signal);

        // Cache successful read-only operations
        if (isCacheable && cache && !toolResponse.error) {
          cache.set(requestInfo, toolResponse);
        }
      }

      if (toolResponse.error) {
        const errorMsg = `Error executing tool ${requestInfo.name}: ${
          toolResponse.resultDisplay || toolResponse.error.message
        }`;
        errors.push(errorMsg);
        console.error(errorMsg);
      }

      if (toolResponse.responseParts) {
        // Optimized memory ingestion for read_file tool
        if (requestInfo.name === 'read_file') {
          await ingestFileContentToMemory(agent, toolResponse.responseParts);
        }
        toolResponseParts.push(...toolResponse.responseParts);
      }
    } catch (toolError) {
      const errorMsg = `Unexpected error in tool ${requestInfo.name}: ${toolError}`;
      errors.push(errorMsg);
      console.error(errorMsg);

      // Continue processing other tools unless it's a critical error
      if (signal.aborted) {
        break;
      }
    }
  }

  // Log summary if there were errors and debug mode is enabled
  if (errors.length > 0 && config.getDebugMode()) {
    console.error(
      `[DEBUG] Tool execution completed with ${errors.length} errors`,
    );
  }

  return toolResponseParts;
}

/**
 * Determines if a tool call can be safely cached
 */
function isCacheableToolCall(requestInfo: ToolCallRequestInfo): boolean {
  // Only cache read-only operations that don't modify state
  const cacheableTools = new Set([
    'read_file',
    'list_directory',
    'file_search',
    'grep_search',
    'semantic_search',
  ]);

  return cacheableTools.has(requestInfo.name);
}

/**
 * Optimized memory ingestion with better error handling and memory management
 */
async function ingestFileContentToMemory(
  agent: AgentLike,
  responseParts: Part[],
): Promise<void> {
  try {
    const fileContent = responseParts
      .map((part) => (part as Part).text)
      .filter(Boolean) // Remove empty/null parts
      .join('');

    if (!fileContent) {
      return; // No content to ingest
    }

    // Runtime-guarded whisper: agent is an opaque runtime instance and may be absent.
    type AgentRuntime = { whisper?: (data: string, kind?: string) => void };
    const runtimeAgent = agent as AgentRuntime | undefined;

    if (runtimeAgent && typeof runtimeAgent.whisper === 'function') {
      // Chunk large content to avoid memory issues
      const chunkSize = 32 * 1024; // 32KB chunks
      if (fileContent.length > chunkSize) {
        for (let i = 0; i < fileContent.length; i += chunkSize) {
          const chunk = fileContent.slice(i, i + chunkSize);
          runtimeAgent.whisper(chunk, 'semantic');
        }
      } else {
        runtimeAgent.whisper(fileContent, 'semantic');
      }
    }
  } catch (memoryError) {
    // Non-critical error - log but don't fail the entire operation
    console.error(
      `Warning: Failed to ingest content to memory: ${memoryError}`,
    );
  }
}
