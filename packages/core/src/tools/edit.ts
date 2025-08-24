/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import * as Diff from 'diff';
import {
  BaseDeclarativeTool,
  Kind,
  ToolCallConfirmationDetails,
  ToolConfirmationOutcome,
  ToolEditConfirmationDetails,
  ToolInvocation,
  ToolLocation,
  ToolResult,
  ToolResultDisplay,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { makeRelative, shortenPath } from '../utils/paths.js';
import { isNodeError } from '../utils/errors.js';
import { Config, ApprovalMode } from '../config/config.js';
import { ensureCorrectEdit } from '../utils/editCorrector.js';
import { DEFAULT_DIFF_OPTIONS, getDiffStat } from './diffOptions.js';
import { ReadFileTool } from './read-file.js';
import { ModifiableDeclarativeTool, ModifyContext } from './modifiable-tool.js';
import { IDEConnectionStatus } from '../ide/ide-client.js';
import { FileOperation } from '../telemetry/metrics.js';
import { logFileOperation } from '../telemetry/loggers.js';
import { FileOperationEvent } from '../telemetry/types.js';
import { getProgrammingLanguage } from '../telemetry/telemetry-utils.js';
import { getSpecificMimeType } from '../utils/fileUtils.js';
import { resetEditCorrectorCaches } from '../utils/editCorrector.js';

export function applyReplacement(
  currentContent: string | null,
  oldString: string,
  newString: string,
  isNewFile: boolean,
): string {
  if (isNewFile) {
    return newString;
  }
  if (currentContent === null) {
    // Should not happen if not a new file, but defensively return empty or newString if oldString is also empty
    return oldString === '' ? newString : '';
  }
  // If oldString is empty and it's not a new file, do not modify the content.
  if (oldString === '' && !isNewFile) {
    return currentContent;
  }
  return currentContent.replaceAll(oldString, newString);
}

/**
 * Applies a range-based edit to content using line/column coordinates
 */
export function applyRangeEdit(
  currentContent: string,
  startLine: number,
  startColumn: number,
  endLine: number,
  endColumn: number,
  newContent: string,
): string {
  const lines = currentContent.split('\n');
  
  // Validate range bounds
  if (startLine < 0 || startLine >= lines.length) {
    throw new Error(`Start line ${startLine} is out of bounds (0-${lines.length - 1})`);
  }
  if (endLine < 0 || endLine >= lines.length) {
    throw new Error(`End line ${endLine} is out of bounds (0-${lines.length - 1})`);
  }
  if (startLine > endLine) {
    throw new Error(`Start line ${startLine} cannot be greater than end line ${endLine}`);
  }
  if (startLine === endLine && startColumn > endColumn) {
    throw new Error(`Start column ${startColumn} cannot be greater than end column ${endColumn} on the same line`);
  }

  // Extract parts: before range, after range
  const beforeLines = lines.slice(0, startLine);
  const afterLines = lines.slice(endLine + 1);
  
  // Handle the start and end lines
  const startLineContent = lines[startLine] || '';
  const endLineContent = lines[endLine] || '';
  
  const beforeRange = startLineContent.substring(0, startColumn);
  const afterRange = endLineContent.substring(endColumn);
  
  // Combine the result
  const newLines = [
    ...beforeLines,
    beforeRange + newContent + afterRange,
    ...afterLines
  ];
  
  return newLines.join('\n');
}

/**
 * Parameters for the Edit tool
 */
export interface EditToolParams {
  /**
   * The absolute path to the file to modify
   */
  file_path: string;

  /**
   * The text to replace (for string-based editing)
   */
  old_string?: string;

  /**
   * The text to replace it with (for string-based editing)
   */
  new_string?: string;

  /**
   * Number of replacements expected. Defaults to 1 if not specified.
   * Use when you want to replace multiple occurrences.
   */
  expected_replacements?: number;

  /**
   * Start line for range-based editing (0-indexed)
   */
  start_line?: number;

  /**
   * Start column for range-based editing (0-indexed)
   */
  start_column?: number;

  /**
   * End line for range-based editing (0-indexed)
   */
  end_line?: number;

  /**
   * End column for range-based editing (0-indexed)
   */
  end_column?: number;

  /**
   * New content to insert (for range-based editing)
   */
  new_content?: string;

  /**
   * Whether the edit was modified manually by the user.
   */
  modified_by_user?: boolean;

  /**
   * Initially proposed string.
   */
  ai_proposed_string?: string;
}

interface CalculatedEdit {
  currentContent: string | null;
  newContent: string;
  occurrences: number;
  error?: { display: string; raw: string; type: ToolErrorType };
  isNewFile: boolean;
  isRangeEdit: boolean;
}

class EditToolInvocation implements ToolInvocation<EditToolParams, ToolResult> {
  constructor(
    private readonly config: Config,
    public params: EditToolParams,
  ) {}

  toolLocations(): ToolLocation[] {
    return [{ path: this.params.file_path }];
  }

  /**
   * Determines if this is a range-based edit or string-based edit
   */
  private isRangeEdit(params: EditToolParams): boolean {
    return (
      params.start_line !== undefined &&
      params.start_column !== undefined &&
      params.end_line !== undefined &&
      params.end_column !== undefined &&
      params.new_content !== undefined
    );
  }

  /**
   * Validates range edit parameters
   */
  private validateRangeParams(params: EditToolParams): string | null {
    if (!this.isRangeEdit(params)) {
      return null; // Not a range edit, no validation needed
    }

    const { start_line, start_column, end_line, end_column } = params;

    if (start_line! < 0 || start_column! < 0 || end_line! < 0 || end_column! < 0) {
      return 'Line and column numbers must be non-negative';
    }

    if (start_line! > end_line!) {
      return 'Start line cannot be greater than end line';
    }

    if (start_line === end_line && start_column! > end_column!) {
      return 'Start column cannot be greater than end column on the same line';
    }

    return null;
  }

  /**
   * Calculates the potential outcome of an edit operation.
   * @param params Parameters for the edit operation
   * @returns An object describing the potential edit outcome
   * @throws File system errors if reading the file fails unexpectedly (e.g., permissions)
   */
  private async calculateEdit(
    params: EditToolParams,
    abortSignal: AbortSignal,
  ): Promise<CalculatedEdit> {
    const isRangeEdit = this.isRangeEdit(params);
    const expectedReplacements = params.expected_replacements ?? 1;
    let currentContent: string | null = null;
    let fileExists = false;
    let isNewFile = false;
    let finalNewString = params.new_string || '';
    let finalOldString = params.old_string || '';
    let occurrences = 0;
    let error:
      | { display: string; raw: string; type: ToolErrorType }
      | undefined = undefined;

    // Validate range parameters if this is a range edit
    if (isRangeEdit) {
      const rangeError = this.validateRangeParams(params);
      if (rangeError) {
        error = {
          display: rangeError,
          raw: `Range validation error: ${rangeError}`,
          type: ToolErrorType.EDIT_PREPARATION_FAILURE,
        };
        return {
          currentContent: null,
          newContent: '',
          occurrences: 0,
          error,
          isNewFile: false,
          isRangeEdit: true,
        };
      }
    }

    try {
      const res = await this.config
        .getFileSystemService()
        .readTextFile(params.file_path);
      if (res.success) {
        currentContent = res.data!;
        // Normalize line endings to LF for consistent processing.
        currentContent = currentContent.replace(/\r\n/g, '\n');
        fileExists = true;
      } else {
        if (res.errorCode !== 'ENOENT') {
          throw new Error(res.error);
        }
        fileExists = false;
      }
    } catch (err: unknown) {
      if (!isNodeError(err) || err.code !== 'ENOENT') {
        // Rethrow unexpected FS errors (permissions, etc.)
        throw err;
      }
      fileExists = false;
    }

    // Handle file creation logic
    if (!fileExists) {
      if (isRangeEdit) {
        error = {
          display: 'File not found. Cannot apply range edit to non-existent file.',
          raw: `File not found: ${params.file_path}`,
          type: ToolErrorType.FILE_NOT_FOUND,
        };
      } else if (params.old_string === '') {
        // Creating a new file with string-based edit
        isNewFile = true;
      } else {
        // Trying to edit a nonexistent file (and old_string is not empty)
        error = {
          display: `File not found. Cannot apply edit. Use an empty old_string to create a new file.`,
          raw: `File not found: ${params.file_path}`,
          type: ToolErrorType.FILE_NOT_FOUND,
        };
      }
    } else if (currentContent !== null) {
      // File exists, handle editing
      if (isRangeEdit) {
        // Range-based editing
        try {
          const lines = currentContent.split('\n');
          const { start_line, start_column, end_line, end_column } = params;
          
          // Additional runtime validation against actual file content
          if (start_line! >= lines.length || end_line! >= lines.length) {
            error = {
              display: `Line numbers out of bounds. File has ${lines.length} lines.`,
              raw: `Line numbers out of bounds for file: ${params.file_path}`,
              type: ToolErrorType.EDIT_PREPARATION_FAILURE,
            };
          } else if (start_column! > lines[start_line!].length || end_column! > lines[end_line!].length) {
            error = {
              display: `Column numbers out of bounds for specified lines.`,
              raw: `Column numbers out of bounds for file: ${params.file_path}`,
              type: ToolErrorType.EDIT_PREPARATION_FAILURE,
            };
          } else {
            occurrences = 1; // Range edits always have exactly 1 "occurrence"
          }
        } catch (rangeError) {
          error = {
            display: `Range edit validation failed: ${rangeError instanceof Error ? rangeError.message : String(rangeError)}`,
            raw: `Range edit validation failed for file: ${params.file_path}`,
            type: ToolErrorType.EDIT_PREPARATION_FAILURE,
          };
        }
      } else {
        // String-based editing (existing logic)
        if (params.old_string === '') {
          // Error: Trying to create a file that already exists
          error = {
            display: `Failed to edit. Attempted to create a file that already exists.`,
            raw: `File already exists, cannot create: ${params.file_path}`,
            type: ToolErrorType.ATTEMPT_TO_CREATE_EXISTING_FILE,
          };
        } else {
          const correctedEdit = await ensureCorrectEdit(
            params.file_path,
            currentContent,
            params,
            this.config.getGeminiClient(),
            abortSignal,
          );
          finalOldString = correctedEdit.params.old_string;
          finalNewString = correctedEdit.params.new_string;
          occurrences = correctedEdit.occurrences;

          if (occurrences === 0) {
            error = {
              display: `Failed to edit, could not find the string to replace.`,
              raw: `Failed to edit, 0 occurrences found for old_string in ${params.file_path}. No edits made. The exact text in old_string was not found. Ensure you're not escaping content incorrectly and check whitespace, indentation, and context. Use ${ReadFileTool.Name} tool to verify.`,
              type: ToolErrorType.EDIT_NO_OCCURRENCE_FOUND,
            };
          } else if (occurrences !== expectedReplacements) {
            const occurrenceTerm =
              expectedReplacements === 1 ? 'occurrence' : 'occurrences';

            error = {
              display: `Failed to edit, expected ${expectedReplacements} ${occurrenceTerm} but found ${occurrences}.`,
              raw: `Failed to edit, Expected ${expectedReplacements} ${occurrenceTerm} but found ${occurrences} for old_string in file: ${params.file_path}`,
              type: ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
            };
          } else if (finalOldString === finalNewString) {
            error = {
              display: `No changes to apply. The old_string and new_string are identical.`,
              raw: `No changes to apply. The old_string and new_string are identical in file: ${params.file_path}`,
              type: ToolErrorType.EDIT_NO_CHANGE,
            };
          }
        }
      }
    } else {
      // Should not happen if fileExists and no exception was thrown, but defensively:
      error = {
        display: `Failed to read content of file.`,
        raw: `Failed to read content of existing file: ${params.file_path}`,
        type: ToolErrorType.READ_CONTENT_FAILURE,
      };
    }

    // Calculate new content
    let newContent = currentContent ?? '';
    if (!error) {
      if (isRangeEdit && currentContent !== null) {
        try {
          newContent = applyRangeEdit(
            currentContent,
            params.start_line!,
            params.start_column!,
            params.end_line!,
            params.end_column!,
            params.new_content!,
          );
        } catch (rangeError) {
          error = {
            display: `Range edit failed: ${rangeError instanceof Error ? rangeError.message : String(rangeError)}`,
            raw: `Range edit failed for file: ${params.file_path}`,
            type: ToolErrorType.EDIT_PREPARATION_FAILURE,
          };
          newContent = currentContent;
        }
      } else if (!isRangeEdit) {
        newContent = applyReplacement(
          currentContent,
          finalOldString,
          finalNewString,
          isNewFile,
        );
      }
    }

    // Check if content actually changed
    if (!error && fileExists && currentContent === newContent) {
      error = {
        display:
          'No changes to apply. The new content is identical to the current content.',
        raw: `No changes to apply. The new content is identical to the current content in file: ${params.file_path}`,
        type: ToolErrorType.EDIT_NO_CHANGE,
      };
    }

    return {
      currentContent,
      newContent,
      occurrences,
      error,
      isNewFile,
      isRangeEdit,
    };
  }

  /**
   * Handles the confirmation prompt for the Edit tool in the CLI.
   * It needs to calculate the diff to show the user.
   */
  async shouldConfirmExecute(
    abortSignal: AbortSignal,
  ): Promise<ToolCallConfirmationDetails | false> {
    if (this.config.getApprovalMode() === ApprovalMode.AUTO_EDIT) {
      return false;
    }

    let editData: CalculatedEdit;
    try {
      editData = await this.calculateEdit(this.params, abortSignal);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.log(`Error preparing edit: ${errorMsg}`);
      return false;
    }

    if (editData.error) {
      console.log(`Error: ${editData.error.display}`);
      return false;
    }

    const fileName = path.basename(this.params.file_path);
    const fileDiff = Diff.createPatch(
      fileName,
      editData.currentContent ?? '',
      editData.newContent,
      'Current',
      'Proposed',
      DEFAULT_DIFF_OPTIONS,
    );
    const ideClient = this.config.getIdeClient();
    const ideConfirmation =
      this.config.getIdeMode() &&
      ideClient?.getConnectionStatus().status === IDEConnectionStatus.Connected
        ? ideClient.openDiff(this.params.file_path, editData.newContent)
        : undefined;

    const confirmationDetails: ToolEditConfirmationDetails = {
      type: 'edit',
      title: `Confirm Edit: ${shortenPath(makeRelative(this.params.file_path, this.config.getTargetDir()))}`,
      fileName,
      filePath: this.params.file_path,
      fileDiff,
      originalContent: editData.currentContent ?? '',
      newContent: editData.newContent,
      onConfirm: async (outcome: ToolConfirmationOutcome) => {
        if (outcome === ToolConfirmationOutcome.ProceedAlways) {
          this.config.setApprovalMode(ApprovalMode.AUTO_EDIT);
        }

        if (ideConfirmation) {
          const result = await ideConfirmation;
          if (result.status === 'accepted' && result.content) {
            // Update params based on edit type
            if (editData.isRangeEdit) {
              this.params.new_content = result.content;
            } else {
              this.params.old_string = editData.currentContent ?? '';
              this.params.new_string = result.content;
            }
          }
        }
      },
      ideConfirmation,
    };
    return confirmationDetails;
  }

  getDescription(): string {
    const relativePath = makeRelative(
      this.params.file_path,
      this.config.getTargetDir(),
    );

    if (this.isRangeEdit(this.params)) {
      return `Range edit ${shortenPath(relativePath)} (${this.params.start_line}:${this.params.start_column}-${this.params.end_line}:${this.params.end_column})`;
    }

    if (this.params.old_string === '') {
      return `Create ${shortenPath(relativePath)}`;
    }

    const oldStringSnippet =
      (this.params.old_string || '').split('\n')[0].substring(0, 30) +
      ((this.params.old_string || '').length > 30 ? '...' : '');
    const newStringSnippet =
      (this.params.new_string || '').split('\n')[0].substring(0, 30) +
      ((this.params.new_string || '').length > 30 ? '...' : '');

    if (this.params.old_string === this.params.new_string) {
      return `No file changes to ${shortenPath(relativePath)}`;
    }
    return `${shortenPath(relativePath)}: ${oldStringSnippet} => ${newStringSnippet}`;
  }

  /**
   * Executes the edit operation with the given parameters.
   * @param params Parameters for the edit operation
   * @returns Result of the edit operation
   */
  async execute(signal: AbortSignal): Promise<ToolResult> {
    let editData: CalculatedEdit;
    try {
      editData = await this.calculateEdit(this.params, signal);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error preparing edit: ${errorMsg}`,
        returnDisplay: `Error preparing edit: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.EDIT_PREPARATION_FAILURE,
        },
      };
    }

    if (editData.error) {
      return {
        llmContent: editData.error.raw,
        returnDisplay: `Error: ${editData.error.display}`,
        error: {
          message: editData.error.raw,
          type: editData.error.type,
        },
      };
    }

    try {
      this.ensureParentDirectoriesExist(this.params.file_path);
      const writeResult = await this.config
        .getFileSystemService()
        .writeTextFile(this.params.file_path, editData.newContent);
      if (!writeResult.success) {
        throw new Error(writeResult.error);
      }
      resetEditCorrectorCaches();

      let displayResult: ToolResultDisplay;
      const fileName = path.basename(this.params.file_path);
      const originallyProposedContent =
        this.params.ai_proposed_string || 
        this.params.new_string || 
        this.params.new_content || 
        '';
      const diffStat = getDiffStat(
        fileName,
        editData.currentContent ?? '',
        originallyProposedContent,
        editData.newContent,
      );

      if (editData.isNewFile) {
        displayResult = `Created ${shortenPath(makeRelative(this.params.file_path, this.config.getTargetDir()))}`;
      } else {
        // Generate diff for display, even though core logic doesn't technically need it
        // The CLI wrapper will use this part of the ToolResult
        const fileDiff = Diff.createPatch(
          fileName,
          editData.currentContent ?? '', // Should not be null here if not isNewFile
          editData.newContent,
          'Current',
          'Proposed',
          DEFAULT_DIFF_OPTIONS,
        );
        displayResult = {
          fileDiff,
          fileName,
          originalContent: editData.currentContent ?? '',
          newContent: editData.newContent,
          diffStat,
        };
      }

      const editType = editData.isRangeEdit ? 'range' : 'string';
      const llmSuccessMessageParts = [
        editData.isNewFile
          ? `Created new file: ${this.params.file_path} with provided content.`
          : `Successfully modified file: ${this.params.file_path} using ${editType} edit (${editData.occurrences} replacements).`,
      ];
      if (this.params.modified_by_user) {
        const modifiedContent = editData.isRangeEdit ? this.params.new_content : this.params.new_string;
        llmSuccessMessageParts.push(
          `User modified the content to be: ${modifiedContent}.`,
        );
      }

      const lines = editData.newContent.split('\n').length;
      const mimetype = getSpecificMimeType(this.params.file_path);
      const extension = path.extname(this.params.file_path);
      const programming_language = getProgrammingLanguage({
        file_path: this.params.file_path,
      });

      logFileOperation(
        this.config,
        new FileOperationEvent(
          EditTool.Name,
          editData.isNewFile ? FileOperation.CREATE : FileOperation.UPDATE,
          lines,
          mimetype,
          extension,
          diffStat,
          programming_language,
        ),
      );

      return {
        llmContent: llmSuccessMessageParts.join(' '),
        returnDisplay: displayResult,
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error executing edit: ${errorMsg}`,
        returnDisplay: `Error writing file: ${errorMsg}`,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_WRITE_FAILURE,
        },
      };
    }
  }

  /**
   * Creates parent directories if they don't exist
   */
  private ensureParentDirectoriesExist(filePath: string): void {
    const dirName = path.dirname(filePath);
    if (!fs.existsSync(dirName)) {
      fs.mkdirSync(dirName, { recursive: true });
    }
  }
}

/**
 * Implementation of the Edit tool logic
 */
export class EditTool
  extends BaseDeclarativeTool<EditToolParams, ToolResult>
  implements ModifiableDeclarativeTool<EditToolParams>
{
  static readonly Name = 'replace';
  constructor(private readonly config: Config) {
    super(
      EditTool.Name,
      'Edit',
      `Replaces text within a file using either string-based replacement or precise range-based editing.

**String-based editing (legacy mode):**
Replaces a single occurrence by default, but can replace multiple occurrences when \`expected_replacements\` is specified. This mode requires providing significant context around the change to ensure precise targeting.

**Range-based editing (robust mode):**
Allows precise specification of what to delete and insert using line/column coordinates. This mode is more robust as it doesn't depend on exact string matching.

Always use the ${ReadFileTool.Name} tool to examine the file's current content before attempting any edit.

The user has the ability to modify the content. If modified, this will be stated in the response.

**For string-based editing:**
1. \`file_path\` MUST be an absolute path.
2. \`old_string\` MUST be the exact literal text to replace (including all whitespace, indentation, newlines, etc.).
3. \`new_string\` MUST be the exact literal text to replace \`old_string\` with.
4. NEVER escape \`old_string\` or \`new_string\`.

**For range-based editing:**
1. \`file_path\` MUST be an absolute path.
2. \`start_line\`, \`start_column\`, \`end_line\`, \`end_column\` specify the range to delete (0-indexed).
3. \`new_content\` is the content to insert at the start position.

**Multiple replacements (string mode only):** Set \`expected_replacements\` to the number of occurrences you want to replace.`,
      Kind.Edit,
      {
        properties: {
          file_path: {
            description:
              "The absolute path to the file to modify. Must start with '/'.",
            type: 'string',
          },
          old_string: {
            description:
              'The exact literal text to replace (string-based editing). Include context for precise targeting.',
            type: 'string',
          },
          new_string: {
            description:
              'The exact literal text to replace `old_string` with (string-based editing).',
            type: 'string',
          },
          expected_replacements: {
            type: 'number',
            description:
              'Number of replacements expected for string-based editing. Defaults to 1.',
            minimum: 1,
          },
          start_line: {
            type: 'number',
            description:
              'Start line for range-based editing (0-indexed).',
            minimum: 0,
          },
          start_column: {
            type: 'number',
            description:
              'Start column for range-based editing (0-indexed).',
            minimum: 0,
          },
          end_line: {
            type: 'number',
            description:
              'End line for range-based editing (0-indexed).',
            minimum: 0,
          },
          end_column: {
            type: 'number',
            description:
              'End column for range-based editing (0-indexed).',
            minimum: 0,
          },
          new_content: {
            type: 'string',
            description:
              'New content to insert for range-based editing.',
          },
        },
        required: ['file_path'],
        type: 'object',
      },
    );
  }

  /**
   * Validates the parameters for the Edit tool
   * @param params Parameters to validate
   * @returns Error message string or null if valid
   */
  protected override validateToolParamValues(
    params: EditToolParams,
  ): string | null {
    if (!params.file_path) {
      return "The 'file_path' parameter must be non-empty.";
    }

    if (!path.isAbsolute(params.file_path)) {
      return `File path must be absolute: ${params.file_path}`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    if (!workspaceContext.isPathWithinWorkspace(params.file_path)) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(', ')}`;
    }

    // Check if this is range-based or string-based editing
    const hasRangeParams = 
      params.start_line !== undefined ||
      params.start_column !== undefined ||
      params.end_line !== undefined ||
      params.end_column !== undefined ||
      params.new_content !== undefined;

    const hasStringParams = 
      params.old_string !== undefined ||
      params.new_string !== undefined;

    if (hasRangeParams && hasStringParams) {
      return 'Cannot mix range-based and string-based editing parameters. Use either (start_line, start_column, end_line, end_column, new_content) or (old_string, new_string).';
    }

    if (hasRangeParams) {
      // Validate range parameters
      if (
        params.start_line === undefined ||
        params.start_column === undefined ||
        params.end_line === undefined ||
        params.end_column === undefined ||
        params.new_content === undefined
      ) {
        return 'Range-based editing requires all of: start_line, start_column, end_line, end_column, new_content';
      }
    } else {
      // Validate string parameters
      if (params.old_string === undefined || params.new_string === undefined) {
        return 'String-based editing requires both old_string and new_string';
      }
    }

    return null;
  }

  getModifyContext(
    abortSignal: AbortSignal,
  ): ModifyContext<EditToolParams> {
    return {
      getFilePath: (params: EditToolParams) => params.file_path,
      getCurrentContent: async (params: EditToolParams): Promise<string> => {
        try {
          const res = await this.config
            .getFileSystemService()
            .readTextFile(params.file_path);
          if (res.success) {
            return res.data ?? '';
          }
          return '';
        } catch (err) {
          if (!isNodeError(err) || err.code !== 'ENOENT') throw err;
          return '';
        }
      },
      getProposedContent: async (params: EditToolParams): Promise<string> => {
        try {
          const res = await this.config
            .getFileSystemService()
            .readTextFile(params.file_path);
          let currentContent: string;
          if (res.success) {
            currentContent = res.data ?? '';
          } else {
            currentContent = '';
          }

          // Determine edit type and apply appropriate transformation
          if (
            params.start_line !== undefined &&
            params.start_column !== undefined &&
            params.end_line !== undefined &&
            params.end_column !== undefined &&
            params.new_content !== undefined
          ) {
            // Range-based edit
            return applyRangeEdit(
              currentContent,
              params.start_line,
              params.start_column,
              params.end_line,
              params.end_column,
              params.new_content,
            );
          } else {
            // String-based edit
            return applyReplacement(
              currentContent,
              params.old_string || '',
              params.new_string || '',
              params.old_string === '' && currentContent === '',
            );
          }
        } catch (err) {
          if (!isNodeError(err) || err.code !== 'ENOENT') throw err;
          return '';
        }
      },
      createUpdatedParams: (
        oldContent: string,
        modifiedProposedContent: string,
        originalParams: EditToolParams,
      ): EditToolParams => {
        const isRangeEdit = 
          originalParams.start_line !== undefined &&
          originalParams.start_column !== undefined &&
          originalParams.end_line !== undefined &&
          originalParams.end_column !== undefined;

        if (isRangeEdit) {
          return {
            ...originalParams,
            ai_proposed_string: originalParams.new_content,
            new_content: modifiedProposedContent,
            modified_by_user: true,
          };
        } else {
          return {
            ...originalParams,
            ai_proposed_string: originalParams.new_string,
            old_string: oldContent,
            new_string: modifiedProposedContent,
            modified_by_user: true,
          };
        }
      },
    };
  }

  createInvocation(params: EditToolParams): ToolInvocation<EditToolParams, ToolResult> {
    return new EditToolInvocation(this.config, params);
  }
}
