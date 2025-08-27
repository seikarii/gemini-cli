/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { EditorType, openDiff } from '../utils/editor.js';
import * as Diff from 'diff';
import path from 'path';
import { promises as fsp } from 'fs';
import os from 'os';
import { DEFAULT_DIFF_OPTIONS } from './diffOptions.js';
import { isNodeError } from '../utils/errors.js';
import { AnyDeclarativeTool, DeclarativeTool, ToolResult } from './tools.js';

/**
 * A declarative tool that supports a modify operation.
 */
export interface ModifiableDeclarativeTool<TParams extends object>
  extends DeclarativeTool<TParams, ToolResult> {
  getModifyContext(abortSignal: AbortSignal): ModifyContext<TParams>;
}

export interface ModifyContext<ToolParams> {
  getFilePath: (params: ToolParams) => string;

  getCurrentContent: (params: ToolParams) => Promise<string>;

  getProposedContent: (params: ToolParams) => Promise<string>;
  createUpdatedParams: (
    oldContent: string,
    newContent: string,
    originalParams: ToolParams,
  ) => ToolParams;
}

export interface ModifyResult<T> {
  updatedParams: T;
  updatedDiff: string;
}

/**
 * Type guard to check if a declarative tool is modifiable.
 */
export function isModifiableDeclarativeTool(
  tool: AnyDeclarativeTool,
): tool is ModifiableDeclarativeTool<object> {
  return 'getModifyContext' in tool;
}

async function createTempFilesForModify(
  currentContent: string,
  proposedContent: string,
  file_path: string,
): Promise<{ oldPath: string; newPath: string }> {
  const tempDir = os.tmpdir();
  const diffDir = path.join(tempDir, 'gemini-cli-tool-modify-diffs');

  // Ensure the diff directory exists (recursive handles existing case).
  try {
    await fsp.mkdir(diffDir, { recursive: true });
  } catch (err: unknown) {
    // Re-throw unless it's a benign EEXIST race or similar.
    if (!isNodeError(err) || (err as NodeJS.ErrnoException).code !== 'EEXIST')
      throw err;
  }

  const ext = path.extname(file_path);
  const fileName = path.basename(file_path, ext);
  const timestamp = Date.now();
  const tempOldPath = path.join(
    diffDir,
    `gemini-cli-modify-${fileName}-old-${timestamp}${ext}`,
  );
  const tempNewPath = path.join(
    diffDir,
    `gemini-cli-modify-${fileName}-new-${timestamp}${ext}`,
  );

  await fsp.writeFile(tempOldPath, currentContent, 'utf8');
  await fsp.writeFile(tempNewPath, proposedContent, 'utf8');

  return { oldPath: tempOldPath, newPath: tempNewPath };
}

async function getUpdatedParams<ToolParams>(
  tmpOldPath: string,
  tempNewPath: string,
  originalParams: ToolParams,
  modifyContext: ModifyContext<ToolParams>,
): Promise<{ updatedParams: ToolParams; updatedDiff: string }> {
  let oldContent = '';
  let newContent = '';

  try {
    oldContent = await fsp.readFile(tmpOldPath, 'utf8');
  } catch (err: unknown) {
    if (!isNodeError(err) || (err as NodeJS.ErrnoException).code !== 'ENOENT')
      throw err;
    oldContent = '';
  }

  try {
    newContent = await fsp.readFile(tempNewPath, 'utf8');
  } catch (err: unknown) {
    if (!isNodeError(err) || (err as NodeJS.ErrnoException).code !== 'ENOENT')
      throw err;
    newContent = '';
  }

  const updatedParams = modifyContext.createUpdatedParams(
    oldContent,
    newContent,
    originalParams,
  );
  const updatedDiff = Diff.createPatch(
    path.basename(modifyContext.getFilePath(originalParams)),
    oldContent,
    newContent,
    'Current',
    'Proposed',
    DEFAULT_DIFF_OPTIONS,
  );

  return { updatedParams, updatedDiff };
}

async function deleteTempFiles(
  oldPath: string,
  newPath: string,
): Promise<void> {
  try {
    await fsp.unlink(oldPath);
  } catch (err: unknown) {
    console.error(`Error deleting temp diff file: ${oldPath}`, err);
  }

  try {
    await fsp.unlink(newPath);
  } catch (err: unknown) {
    console.error(`Error deleting temp diff file: ${newPath}`, err);
  }
}

/**
 * Triggers an external editor for the user to modify the proposed content,
 * and returns the updated tool parameters and the diff after the user has modified the proposed content.
 */
export async function modifyWithEditor<ToolParams>(
  originalParams: ToolParams,
  modifyContext: ModifyContext<ToolParams>,
  editorType: EditorType,
  _abortSignal: AbortSignal,
  onEditorClose: () => void,
): Promise<ModifyResult<ToolParams>> {
  const currentContent = await modifyContext.getCurrentContent(originalParams);
  const proposedContent =
    await modifyContext.getProposedContent(originalParams);

  const { oldPath, newPath } = await createTempFilesForModify(
    currentContent,
    proposedContent,
    modifyContext.getFilePath(originalParams),
  );

  try {
    await openDiff(oldPath, newPath, editorType, onEditorClose);
    const result = await getUpdatedParams(
      oldPath,
      newPath,
      originalParams,
      modifyContext,
    );

    return result;
  } finally {
    await deleteTempFiles(oldPath, newPath);
  }
}
