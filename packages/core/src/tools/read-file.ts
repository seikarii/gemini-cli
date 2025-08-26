/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';
import fs from 'fs/promises';
import { makeRelative, shortenPath } from '../utils/paths.js';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  ToolInvocation,
  ToolLocation,
  ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';

import { PartUnion } from '@google/genai';
import {
  processSingleFileContent,
  getSpecificMimeType,
} from '../utils/fileUtils.js';
import { Config } from '../config/config.js';
import { FileOperation } from '../telemetry/metrics.js';
import { getProgrammingLanguage } from '../telemetry/telemetry-utils.js';
import { logFileOperation } from '../telemetry/loggers.js';
import { FileOperationEvent } from '../telemetry/types.js';
import fetch from 'node-fetch';

/**
 * Parameters for the ReadFile tool
 */
export interface ReadFileToolParams {
  /**
   * The absolute path to the file to read
   */
  absolute_path: string;

  /**
   * The line number to start reading from (optional)
   */
  offset?: number;

  /**
   * The number of lines to read (optional)
   */
  limit?: number;
}

class ReadFileToolInvocation extends BaseToolInvocation<
  ReadFileToolParams,
  ToolResult
> {
  constructor(
    private config: Config,
    params: ReadFileToolParams,
  ) {
    super(params);
  }

  getDescription(): string {
    const relativePath = makeRelative(
      this.params.absolute_path,
      this.config.getTargetDir(),
    );
    return shortenPath(relativePath);
  }

  override toolLocations(): ToolLocation[] {
    return [{ path: this.params.absolute_path, line: this.params.offset }];
  }

  private async openInMewWindow(filePath: string): Promise<void> {
    try {
      const portFilePath = path.join(this.config.getTargetDir(), '.gemini', 'mew_port.txt');
      console.log(`[read_file] Looking for mew_port.txt at: ${portFilePath}`);
      let port = 3000;
      try {
        const portStr = await fs.readFile(portFilePath, 'utf8');
        const parsedPort = parseInt(portStr, 10);
        if (!isNaN(parsedPort)) {
          port = parsedPort;
        }
        console.log(`[read_file] Found Mew server on port: ${port}`);
      } catch (e) {
        console.log(`[read_file] Could not read mew_port.txt, defaulting to port: ${port}`);
      }

      const url = `http://localhost:${port}/api/mew/set-active-file`;
      const body = JSON.stringify({ filePath });
      console.log(`[read_file] Sending POST to ${url} with body: ${body}`);

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: body,
  }).then(async (response: any) => {
        console.info(`[read_file] Received response status: ${response.status}`);
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`[read_file] Server error: ${errorText}`);
        }
      }).catch((err: any) => {
        console.error('[read_file] Failed to update Mew Window (fetch error):', err);
      });
    } catch (error) {
      console.error('[read_file] Error sending file to Mew Window:', error);
    }
  }

  async execute(): Promise<ToolResult> {
    // Fire and forget to update the Mew window
    this.openInMewWindow(this.params.absolute_path);

    const result = await processSingleFileContent(
      this.params.absolute_path,
      this.config.getTargetDir(),
      this.config.getFileSystemService(),
      this.params.offset,
      this.params.limit,
    );

    if (result.error) {
      return {
        llmContent: result.llmContent,
        returnDisplay: result.returnDisplay || 'Error reading file',
        error: {
          message: result.error,
          type: result.errorType,
        },
      };
    }

    let llmContent: PartUnion;
    const MAX_LLM_CONTENT_SIZE = 1024 * 1024; // 1MB limit for LLM content

    if (result.isTruncated) {
      const [start, end] = result.linesShown!;
      const total = result.originalLineCount!;
      const nextOffset = this.params.offset
        ? this.params.offset + end - start + 1
        : end;
      llmContent = `
IMPORTANT: The file content has been truncated.
Status: Showing lines ${start}-${end} of ${total} total lines.
Action: To read more of the file, you can use the 'offset' and 'limit' parameters in a subsequent 'read_file' call. For example, to read the next section of the file, use offset: ${nextOffset}.

--- FILE CONTENT (truncated) ---
${result.llmContent}`;
    } else {
      llmContent = result.llmContent || '';
    }

    // Allow llmContent to be either a string (text) or an object with expected parts
    const mimetype = getSpecificMimeType(this.params.absolute_path);
    const programming_language = getProgrammingLanguage({
      absolute_path: this.params.absolute_path,
    });

    const isString = typeof llmContent === 'string';

    function isPartUnion(value: unknown): value is PartUnion {
      if (!value || typeof value !== 'object') return false;
      const v = value as Record<string, unknown>;
      return 'text' in v || 'inlineData' in v;
    }

    const isPartObject = isPartUnion(llmContent);

    if (!isString && !isPartObject) {
      const errorMsg = `File content is not in a supported format. Mime type: ${mimetype}`;
      return {
        llmContent: errorMsg,
        returnDisplay: `❌ Error: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.READ_CONTENT_FAILURE,
        },
      };
    }

    if (isString && (llmContent as string).length > MAX_LLM_CONTENT_SIZE) {
      const errorMsg = `File content exceeds maximum allowed size for LLM (${MAX_LLM_CONTENT_SIZE} bytes). Actual size: ${(llmContent as string).length} bytes.`;
      return {
        llmContent: errorMsg,
        returnDisplay: `❌ Error: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_TOO_LARGE,
        },
      };
    }

    const lines = isString ? (result.llmContent as string).split('\n').length : undefined;
    logFileOperation(
      this.config,
      new FileOperationEvent(
        ReadFileTool.Name,
        FileOperation.READ,
        lines,
        mimetype,
        path.extname(this.params.absolute_path),
        undefined,
        programming_language,
      ),
    );

    // If the PartUnion is an object containing text, many existing tests expect a plain string.
    // Unwrap { text: string } -> string for backward compatibility, but keep inlineData objects as-is.
    let finalLlmContent: PartUnion = llmContent;
    if (!isString && isPartObject) {
      const obj = llmContent as Record<string, unknown>;
      if ('text' in obj && typeof obj['text'] === 'string') {
        finalLlmContent = obj['text'] as string;
      }
    }

    return {
      llmContent: finalLlmContent,
      returnDisplay: result.returnDisplay || '',
    };
  }
}

/**
 * Implementation of the ReadFile tool logic
 */
export class ReadFileTool extends BaseDeclarativeTool<
  ReadFileToolParams,
  ToolResult
> {
  static readonly Name: string = 'read_file';

  constructor(private config: Config) {
    super(
      ReadFileTool.Name,
      'ReadFile',
      `Reads and returns the content of a specified file. If the file is large, the content will be truncated. The tool's response will clearly indicate if truncation has occurred and will provide details on how to read more of the file using the 'offset' and 'limit' parameters. Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, it can read specific line ranges.`,
      Kind.Read,
      {
        properties: {
          absolute_path: {
            description:
              "The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path.",
            type: 'string',
          },
          offset: {
            description:
              "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.",
            type: 'number',
          },
          limit: {
            description:
              "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit).",
            type: 'number',
          },
        },
        required: ['absolute_path'],
        type: 'object',
      },
    );
  }

  protected override validateToolParamValues(
    params: ReadFileToolParams,
  ): string | null {
    const filePath = params.absolute_path;
    if (params.absolute_path.trim() === '') {
      return "The 'absolute_path' parameter must be non-empty.";
    }

    if (!path.isAbsolute(filePath)) {
      return `File path must be absolute, but was relative: ${filePath}. You must provide an absolute path.`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(filePath)) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(', ')}`;
    }
    if (params.offset !== undefined && params.offset < 0) {
      return 'Offset must be a non-negative number';
    }
    if (params.limit !== undefined && params.limit <= 0) {
      return 'Limit must be a positive number';
    }

    const fileService = this.config.getFileService();
    if (fileService.shouldGeminiIgnoreFile(params.absolute_path)) {
      return `File path '${filePath}' is ignored by .geminiignore pattern(s).`;
    }

    return null;
  }

  protected createInvocation(
    params: ReadFileToolParams,
  ): ToolInvocation<ReadFileToolParams, ToolResult> {
    return new ReadFileToolInvocation(this.config, params);
  }
}
