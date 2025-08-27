/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'fs';
import path from 'path';
import * as Diff from 'diff';
import { Config, ApprovalMode } from '../config/config.js';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  FileDiff,
  Kind,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolEditConfirmationDetails,
  ToolInvocation,
  ToolLocation,
  ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { getErrorMessage, isNodeError } from '../utils/errors.js';
import {
  ensureCorrectEdit,
  ensureCorrectFileContent,
} from '../utils/editCorrector.js';
import { DEFAULT_DIFF_OPTIONS, getDiffStat } from './diffOptions.js';
import { ModifiableDeclarativeTool, ModifyContext } from './modifiable-tool.js';
import { getSpecificMimeType } from '../utils/fileUtils.js';
import { FileOperation } from '../telemetry/metrics.js';
import { IDEConnectionStatus } from '../ide/ide-client.js';
import { getProgrammingLanguage } from '../telemetry/telemetry-utils.js';
import { logFileOperation } from '../telemetry/loggers.js';
import { FileOperationEvent } from '../telemetry/types.js';
import { resetEditCorrectorCaches } from '../utils/editCorrector.js';

/**
 * Parameters for the WriteFile tool
 */
export interface WriteFileToolParams {
  /**
   * The absolute path to the file to write to
   */
  file_path: string;

  /**
   * The content to write to the file
   */
  content: string;

  /**
   * The mode of writing. 'overwrite' will replace the entire file, 'append' will add to the end.
   * Defaults to 'overwrite'.
   */
  mode?: 'overwrite' | 'append';

  /**
   * If true, bypasses the content correction logic (ensureCorrectEdit).
   * Use with caution, primarily for simple appends to config files.
   */
  skip_correction?: boolean;

  /**
   * Whether the proposed content was modified by the user.
   */
  modified_by_user?: boolean;

  /**
   * Initially proposed content.
   */
  ai_proposed_content?: string;
}

interface GetCorrectedFileContentResult {
  originalContent: string;
  correctedContent: string;
  fileExists: boolean;
  error?: { message: string; code?: string };
}

export async function getCorrectedFileContent(
  config: Config,
  filePath: string,
  proposedContent: string,
  abortSignal: AbortSignal,
): Promise<GetCorrectedFileContentResult> {
  let originalContent = '';
  let fileExists = false;
  let correctedContent = proposedContent;

  try {
    const readResult = await config
      .getFileSystemService()
      .readTextFile(filePath);
    if (readResult.success) {
      originalContent = readResult.data!;
      fileExists = true; // File exists and was read
    } else {
      if (readResult.errorCode !== 'ENOENT') {
        throw new Error(readResult.error);
      }
      fileExists = false;
      originalContent = '';
    }
  } catch (err) {
    if (isNodeError(err) && err.code === 'ENOENT') {
      fileExists = false;
      originalContent = '';
    } else {
      // File exists but could not be read (permissions, etc.)
      fileExists = true; // Mark as existing but problematic
      originalContent = ''; // Can't use its content
      const error = {
        message: getErrorMessage(err),
        code: isNodeError(err) ? err.code : undefined,
      };
      // Return early as we can't proceed with content correction meaningfully
      return { originalContent, correctedContent, fileExists, error };
    }
  }

  // If readError is set, we have returned.
  // So, file was either read successfully (fileExists=true, originalContent set)
  // or it was ENOENT (fileExists=false, originalContent='').

  if (fileExists) {
    // This implies originalContent is available
    const { params: correctedParams } = await ensureCorrectEdit(
      filePath,
      originalContent,
      {
        old_string: originalContent, // Treat entire current content as old_string
        new_string: proposedContent,
        file_path: filePath,
      },
      config.getGeminiClient(),
      abortSignal,
    );
    correctedContent = correctedParams.new_string;

    // Safety guard: some correction flows (LLM or heuristics) may incorrectly
    // return an empty string which would silently truncate the target file.
    // If the caller proposed non-empty content, prefer the original proposed
    // content instead of an empty corrected result to avoid data loss.
    if (
      (correctedContent === undefined || correctedContent === '') &&
      proposedContent &&
      proposedContent.length > 0
    ) {
      if (config.getDebugMode && config.getDebugMode()) {
        console.warn(
          `ensureCorrectEdit returned empty corrected content for ${filePath}; falling back to proposed content to avoid truncation.`,
        );
      }
      correctedContent = proposedContent;
    }
  } else {
    // This implies new file (ENOENT)
    correctedContent = await ensureCorrectFileContent(
      proposedContent,
      config.getGeminiClient(),
      abortSignal,
    );
  }
  return { originalContent, correctedContent, fileExists };
}

class WriteFileToolInvocation extends BaseToolInvocation<
  WriteFileToolParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: WriteFileToolParams,
  ) {
    super(params);
  }

  override toolLocations(): ToolLocation[] {
    return [{ path: this.params.file_path }];
  }

  override getDescription(): string {
    const relativePath = makeRelative(
      this.params.file_path,
      this.config.getTargetDir(),
    );
    return `Writing to ${shortenPath(relativePath)}`;
  }

  override async shouldConfirmExecute(
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    if (this.config.getApprovalMode() === ApprovalMode.AUTO_EDIT) {
      return false;
    }

    const correctedContentResult = await getCorrectedFileContent(
      this.config,
      this.params.file_path,
      this.params.content,
      abortSignal,
    );

    if (correctedContentResult.error) {
      // If file exists but couldn't be read, we can't show a diff for confirmation.
      return false;
    }

    const { originalContent, correctedContent } = correctedContentResult;
    const relativePath = makeRelative(
      this.params.file_path,
      this.config.getTargetDir(),
    );
    const fileName = path.basename(this.params.file_path);

    const fileDiff = Diff.createPatch(
      fileName,
      originalContent, // Original content (empty if new file or unreadable)
      correctedContent, // Content after potential correction
      'Current',
      'Proposed',
      DEFAULT_DIFF_OPTIONS,
    );

    const ideClient = this.config.getIdeClient();
    const ideConfirmation =
      this.config.getIdeMode() &&
      ideClient.getConnectionStatus().status === IDEConnectionStatus.Connected
        ? ideClient.openDiff(this.params.file_path, correctedContent)
        : undefined;

    const confirmationDetails: ToolEditConfirmationDetails = {
      type: 'edit',
      title: `Confirm Write: ${shortenPath(relativePath)}`,
      fileName,
      filePath: this.params.file_path,
      fileDiff,
      originalContent,
      newContent: correctedContent,
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlways) {
          this.config.setApprovalMode(ApprovalMode.AUTO_EDIT);
        }

        if (ideConfirmation) {
          const result = await ideConfirmation;
          if (result.status === 'accepted' && result.content) {
            this.params.content = result.content;
          }
        }
      },
      ideConfirmation,
    };
    return confirmationDetails;
  }

  async execute(abortSignal: AbortSignal): Promise<ToolResult> {
    let {
      file_path,
      content,
      ai_proposed_content,
      modified_by_user,
      mode = 'overwrite',
      skip_correction,
    } = this.params;

  if (mode === 'append' && !(await this.config.getFileSystemService().exists(file_path))) {
      // If appending to a non-existent file, it's the same as overwriting an empty file.
      mode = 'overwrite';
    }

    // Safeguard against accidental file wipe
    if (
      mode === 'overwrite' &&
      !content
    ) {
      const fileInfo = await this.config.getFileSystemService().getFileInfo(file_path);
      if (fileInfo.success && fileInfo.data && fileInfo.data.size > 0) {
        const errorMsg = `Attempted to overwrite a non-empty file with empty content. Operation aborted to prevent data loss.`;
        return {
          llmContent: errorMsg,
          returnDisplay: errorMsg,
          error: {
            message: errorMsg,
            type: ToolErrorType.FILE_WRITE_FAILURE,
          },
        };
      }
    }

    let finalContent = content;
    if (mode === 'append') {
      const readResult = await this.config
        .getFileSystemService()
        .readTextFile(file_path);
      if (!readResult.success) {
        throw new Error(readResult.error);
      }
      const existingContent = readResult.data!;
      finalContent = existingContent + '\n' + content;
    }

    let fileContent = finalContent;
    let originalContent = '';
    let fileExists = false;
    let isNewFile = false;
    let correctedContentResult: GetCorrectedFileContentResult | undefined =
      undefined;

    if (!skip_correction) {
      correctedContentResult = await getCorrectedFileContent(
        this.config,
        file_path,
        finalContent, // Use finalContent which includes appended text if necessary
        abortSignal,
      );

      if (correctedContentResult.error) {
        const errDetails = correctedContentResult.error;
        const errorMsg = errDetails.code
          ? `Error checking existing file '${file_path}': ${errDetails.message} (${errDetails.code})`
          : `Error checking existing file: ${errDetails.message}`;
        return {
          llmContent: errorMsg,
          returnDisplay: errorMsg,
          error: {
            message: errorMsg,
            type: ToolErrorType.FILE_WRITE_FAILURE,
          },
        };
      }

      originalContent = correctedContentResult.originalContent;
      fileContent = correctedContentResult.correctedContent;
      fileExists = correctedContentResult.fileExists;
      isNewFile =
        !fileExists ||
        (correctedContentResult.error !== undefined &&
          !correctedContentResult.fileExists);
    } else {
      // If skipping correction, assume fileContent is finalContent
      // and determine fileExists/isNewFile based on simple fs.existsSync
      fileExists = await this.config.getFileSystemService().exists(file_path);
      isNewFile = !fileExists;
      if (fileExists) {
        const readResult = await this.config.getFileSystemService().readTextFile(file_path);
        if (readResult.success) {
          originalContent = readResult.data!;
        } else {
          originalContent = '';
        }
      } else {
        originalContent = '';
      }
    }

    try {
      const dirName = path.dirname(file_path);
      if (!(await this.config.getFileSystemService().exists(dirName))) {
        const createResult = await this.config.getFileSystemService().createDirectory(dirName, { recursive: true });
        if (!createResult.success) {
          throw new Error(createResult.error);
        }
      }
      // Create a backup of the existing file to allow restore on failure.
      const backupPath = `${file_path}.backup`;
      try {
        if (await this.config.getFileSystemService().exists(file_path)) {
          const copyResult = await this.config.getFileSystemService().copyFile(file_path, backupPath);
          if (!copyResult.success) {
            throw new Error(copyResult.error);
          }
        }
      } catch (_e) {
        // ignore backup creation failures; proceed to write but log in debug
        if (this.config.getDebugMode()) console.debug('Failed to create backup before write', _e);
      }

      await this.config
        .getFileSystemService()
        .writeTextFile(file_path, fileContent);

      // Verify the write by reading file back. If mismatch, retry once and restore from backup on persistent failure.
      try {
        const verify = await this.config.getFileSystemService().readTextFile(file_path);
        if (!verify.success || verify.data !== fileContent) {
          // retry write once
          await this.config.getFileSystemService().writeTextFile(file_path, fileContent);
          const verify2 = await this.config.getFileSystemService().readTextFile(file_path);
          if (!verify2.success || verify2.data !== fileContent) {
            // restore from backup if available
            try {
              if (await this.config.getFileSystemService().exists(backupPath)) {
                const restoreResult = await this.config.getFileSystemService().copyFile(backupPath, file_path);
                if (!restoreResult.success) {
                  throw new Error(restoreResult.error);
                }
              }
            } catch (_restoreErr) {
              if (this.config.getDebugMode()) console.error('Failed to restore backup after failed write verification', _restoreErr);
            }
            const errMsg = `File write verification failed for ${file_path}. Restored from backup if available.`;
            return {
              llmContent: errMsg,
              returnDisplay: errMsg,
              error: {
                message: errMsg,
                type: ToolErrorType.FILE_WRITE_FAILURE,
              },
            };
          }
        }
      } catch (verifyErr) {
        if (this.config.getDebugMode()) console.error('Error during write verification:', verifyErr);
      }
      resetEditCorrectorCaches();

      // Generate diff for display result
      const fileName = path.basename(file_path);
      // If there was a readError, originalContent in correctedContentResult is '',
      // but for the diff, we want to show the original content as it was before the write if possible.
      // However, if it was unreadable, currentContentForDiff will be empty.
      const currentContentForDiff = correctedContentResult?.error
        ? '' // Or some indicator of unreadable content
        : originalContent;

      const fileDiff = Diff.createPatch(
        fileName,
        currentContentForDiff,
        fileContent,
        'Original',
        'Written',
        DEFAULT_DIFF_OPTIONS,
      );

      const originallyProposedContent = ai_proposed_content || content;
      const diffStat = getDiffStat(
        fileName,
        currentContentForDiff,
        originallyProposedContent,
        content,
      );

      const llmSuccessMessageParts = [
        isNewFile
          ? `Successfully created and wrote to new file: ${file_path}.`
          : `Successfully overwrote file: ${file_path}.`,
      ];
      if (modified_by_user) {
        llmSuccessMessageParts.push(
          `User modified the \`content\` to be: ${content}`,
        );
      }

      const displayResult: FileDiff = {
        fileDiff,
        fileName,
        originalContent:
          correctedContentResult?.originalContent || originalContent,
        newContent: correctedContentResult?.correctedContent || fileContent,
        diffStat,
      };

      const lines = fileContent.split('\n').length;
      const mimetype = getSpecificMimeType(file_path);
      const extension = path.extname(file_path); // Get extension
      const programming_language = getProgrammingLanguage({ file_path });
      if (isNewFile) {
        logFileOperation(
          this.config,
          new FileOperationEvent(
            WriteFileTool.Name,
            FileOperation.CREATE,
            lines,
            mimetype,
            extension,
            diffStat,
            programming_language,
          ),
        );
      } else {
        logFileOperation(
          this.config,
          new FileOperationEvent(
            WriteFileTool.Name,
            FileOperation.UPDATE,
            lines,
            mimetype,
            extension,
            diffStat,
            programming_language,
          ),
        );
      }

      return {
        llmContent: llmSuccessMessageParts.join(' '),
        returnDisplay: displayResult,
      };
    } catch (error) {
      // Capture detailed error information for debugging
      let errorMsg: string;
      let errorType = ToolErrorType.FILE_WRITE_FAILURE;

      if (isNodeError(error)) {
        // Handle specific Node.js errors with their error codes
        errorMsg = `Error writing to file '${file_path}': ${error.message} (${error.code})`;

        // Log specific error types for better debugging
        if (error.code === 'EACCES') {
          errorMsg = `Permission denied writing to file: ${file_path} (${error.code})`;
          errorType = ToolErrorType.PERMISSION_DENIED;
        } else if (error.code === 'ENOSPC') {
          errorMsg = `No space left on device: ${file_path} (${error.code})`;
          errorType = ToolErrorType.NO_SPACE_LEFT;
        } else if (error.code === 'EISDIR') {
          errorMsg = `Target is a directory, not a file: ${file_path} (${error.code})`;
          errorType = ToolErrorType.TARGET_IS_DIRECTORY;
        }

        // Include stack trace in debug mode for better troubleshooting
        if (this.config.getDebugMode() && error.stack) {
          console.error('Write file error stack:', error.stack);
        }
      } else if (error instanceof Error) {
        errorMsg = `Error writing to file: ${error.message}`;
      } else {
        errorMsg = `Error writing to file: ${String(error)}`;
      }

      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg,
        error: {
          message: errorMsg,
          type: errorType,
        },
      };
    }
  }
}

/**
 * Implementation of the WriteFile tool logic
 */
export class WriteFileTool
  extends BaseDeclarativeTool<WriteFileToolParams, ToolResult>
  implements ModifiableDeclarativeTool<WriteFileToolParams>
{
  static readonly Name: string = 'write_file';

  constructor(private readonly config: Config) {
    super(
      WriteFileTool.Name,
      'WriteFile',
      `Writes content to a specified file in the local filesystem.

      The user has the ability to modify \`content\`. If modified, this will be stated in the response.`,
      Kind.Edit,
      {
        properties: {
          file_path: {
            description:
              "The absolute path to the file to write to (e.g., '/home/user/project/file.txt'). Relative paths are not supported.",
            type: 'string',
          },
          content: {
            description: 'The content to write to the file.',
            type: 'string',
          },
          mode: {
            description:
              'The mode of writing. `overwrite` will replace the entire file, `append` will add to the end. Defaults to `overwrite`.',
            type: 'string',
            enum: ['overwrite', 'append'],
          },
          skip_correction: {
            description:
              'If true, bypasses the content correction logic (ensureCorrectEdit). Use with caution, primarily for simple appends to config files.',
            type: 'boolean',
          },
        },
        required: ['file_path', 'content'],
        type: 'object',
      },
    );
  }

  protected override validateToolParamValues(
    params: WriteFileToolParams,
  ): string | null {
    const filePath = params.file_path;

    if (!filePath) {
      return `Missing or empty "file_path"`;
    }

    if (!path.isAbsolute(filePath)) {
      return `File path must be absolute: ${filePath}`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(filePath)) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(
        ', ',
      )}`;
    }

    try {
      if (fs.existsSync(filePath)) {
        const stats = fs.lstatSync(filePath);
        if (stats.isDirectory()) {
          return `Path is a directory, not a file: ${filePath}`;
        }
      }
    } catch (statError: unknown) {
      return `Error accessing path properties for validation: ${filePath}. Reason: ${
        statError instanceof Error ? statError.message : String(statError)
      }`;
    }

    return null;
  }

  protected createInvocation(
    params: WriteFileToolParams,
  ): ToolInvocation<WriteFileToolParams, ToolResult> {
    return new WriteFileToolInvocation(this.config, params);
  }

  getModifyContext(
    abortSignal: AbortSignal,
  ): ModifyContext<WriteFileToolParams> {
    return {
      getFilePath: (params: WriteFileToolParams) => params.file_path,
      getCurrentContent: async (params: WriteFileToolParams) => {
        const correctedContentResult = await getCorrectedFileContent(
          this.config,
          params.file_path,
          params.content,
          abortSignal,
        );
        return correctedContentResult.originalContent;
      },
      getProposedContent: async (params: WriteFileToolParams) => {
        const correctedContentResult = await getCorrectedFileContent(
          this.config,
          params.file_path,
          params.content,
          abortSignal,
        );
        return correctedContentResult.correctedContent;
      },
      createUpdatedParams: (
        _oldContent: string,
        modifiedProposedContent: string,
        originalParams: WriteFileToolParams,
      ) => {
        const content = originalParams.content;
        return {
          ...originalParams,
          ai_proposed_content: content,
          content: modifiedProposedContent,
          modified_by_user: true,
        };
      },
    };
  }
}
