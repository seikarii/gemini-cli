/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { exec, spawn } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

export type EditorType =
  | 'vscode'
  | 'vscodium'
  | 'windsurf'
  | 'cursor'
  | 'vim'
  | 'neovim'
  | 'zed'
  | 'emacs';

function isValidEditorType(editor: string): editor is EditorType {
  return [
    'vscode',
    'vscodium',
    'windsurf',
    'cursor',
    'vim',
    'neovim',
    'zed',
    'emacs',
  ].includes(editor);
}

interface DiffCommand {
  command: string;
  args: string[];
}

async function commandExists(cmd: string): Promise<boolean> {
  try {
    // The `exec` command will throw an error if the command is not found.
    // We don't need to inspect stdout, just catch the error.
    await execPromise(
      process.platform === 'win32' ? `where.exe ${cmd}` : `command -v ${cmd}`
    );
    return true;
  } catch {
    return false;
  }
}

/**
 * Editor command configurations for different platforms.
 * Each editor can have multiple possible command names, listed in order of preference.
 */
const editorCommands: Record<
  EditorType,
  { win32: string[]; default: string[] }
> = {
  vscode: { win32: ['code.cmd'], default: ['code'] },
  vscodium: { win32: ['codium.cmd'], default: ['codium'] },
  windsurf: { win32: ['windsurf'], default: ['windsurf'] },
  cursor: { win32: ['cursor'], default: ['cursor'] },
  vim: { win32: ['vim'], default: ['vim'] },
  neovim: { win32: ['nvim'], default: ['nvim'] },
  zed: { win32: ['zed'], default: ['zed', 'zeditor'] },
  emacs: { win32: ['emacs.exe'], default: ['emacs'] },
};

export async function checkHasEditorType(editor: EditorType): Promise<boolean> {
  const commandConfig = editorCommands[editor];
  const commands =
    process.platform === 'win32' ? commandConfig.win32 : commandConfig.default;
  for (const cmd of commands) {
    if (await commandExists(cmd)) {
      return true;
    }
  }
  return false;
}

export function allowEditorTypeInSandbox(editor: EditorType): boolean {
  const notUsingSandbox = !process.env['SANDBOX'];
  if (['vscode', 'vscodium', 'windsurf', 'cursor', 'zed'].includes(editor)) {
    return notUsingSandbox;
  }
  // For terminal-based editors like vim and emacs, allow in sandbox.
  return true;
}

/**
 * Check if the editor is valid and can be used.
 * Returns false if preferred editor is not set / invalid / not available / not allowed in sandbox.
 */
export async function isEditorAvailable(
  editor: string | undefined
): Promise<boolean> {
  if (editor && isValidEditorType(editor)) {
    return (
      (await checkHasEditorType(editor)) && allowEditorTypeInSandbox(editor)
    );
  }
  return false;
}

/**
 * Get the diff command for a specific editor.
 */
export async function getDiffCommand(
  oldPath: string,
  newPath: string,
  editor: EditorType
): Promise<DiffCommand | null> {
  if (!isValidEditorType(editor)) {
    return null;
  }
  const commandConfig = editorCommands[editor];
  const commands =
    process.platform === 'win32' ? commandConfig.win32 : commandConfig.default;

  const preferredCommands = commands.slice(0, -1);
  let foundCommand: string | undefined;
  for (const cmd of preferredCommands) {
    if (await commandExists(cmd)) {
      foundCommand = cmd;
      break;
    }
  }
  const command = foundCommand || commands[commands.length - 1];

  switch (editor) {
    case 'vscode':
    case 'vscodium':
    case 'windsurf':
    case 'cursor':
    case 'zed':
      return { command, args: ['--wait', '--diff', oldPath, newPath] };
    case 'vim':
    case 'neovim':
      return {
        command,
        args: [
          '-d',
          // skip viminfo file to avoid E138 errors
          '-i',
          'NONE',
          // make the left window read-only and the right window editable
          '-c',
          'wincmd h | set readonly | wincmd l',
          // set up colors for diffs
          '-c',
          'highlight DiffAdd cterm=bold ctermbg=22 guibg=#005f00 | highlight DiffChange cterm=bold ctermbg=24 guibg=#005f87 | highlight DiffText ctermbg=21 guibg=#0000af | highlight DiffDelete ctermbg=52 guibg=#5f0000',
          // Show helpful messages
          '-c',
          'set showtabline=2 | set tabline=[Instructions]\ :wqa(save\ &\ quit)\ \|\ i/esc(toggle\ edit\ mode)',
          '-c',
          'wincmd h | setlocal statusline=OLD\ FILE',
          '-c',
          'wincmd l | setlocal statusline=%#StatusBold#NEW\ FILE\ :wqa(save\ &\ quit)\ \|\ i/esc(toggle\ edit\ mode)',
          // Auto close all windows when one is closed
          '-c',
          'autocmd BufWritePost * wqa',
          oldPath,
          newPath,
        ],
      };
    case 'emacs':
      return {
        command: 'emacs',
        args: ['--eval', `(ediff "${oldPath}" "${newPath}")`],
      };
    default:
      return null;
  }
}

/**
 * Opens a diff tool to compare two files.
 * Terminal-based editors by default blocks parent process until the editor exits.
 * GUI-based editors require args such as "--wait" to block parent process.
 */
export async function openDiff(
  oldPath: string,
  newPath: string,
  editor: EditorType,
  onEditorClose: () => void
): Promise<void> {
  const diffCommand = await getDiffCommand(oldPath, newPath, editor);
  if (!diffCommand) {
    console.error('No diff tool available. Install a supported editor.');
    return;
  }

  try {
    switch (editor) {
      case 'vscode':
      case 'vscodium':
      case 'windsurf':
      case 'cursor':
      case 'zed':
        // Use spawn for GUI-based editors to avoid blocking the entire process
        return new Promise((resolve, reject) => {
          const childProcess = spawn(diffCommand.command, diffCommand.args, {
            stdio: 'inherit',
            shell: true,
          });

          childProcess.on('close', (code) => {
            if (code === 0) {
              resolve();
            } else {
              reject(new Error(`${editor} exited with code ${code}`));
            }
          });

          childProcess.on('error', (error) => {
            reject(error);
          });
        });

      case 'vim':
      case 'emacs':
      case 'neovim': {
        // Use spawn for terminal-based editors and await completion
        return new Promise((resolve) => {
          const args = diffCommand.args;
          const child = spawn(diffCommand.command, args, {
            stdio: 'inherit',
            shell: false,
          });

          child.on('close', (code) => {
            try {
              if (code !== 0) {
                console.error(`${editor} exited with code ${code}`);
              }
            } catch (e) {
              console.error('Error in onEditorClose callback:', e);
            } finally {
              try {
                onEditorClose();
              } catch (e) {
                console.debug('onEditorClose callback threw:', e);
              }
              resolve();
            }
          });

          child.on('error', (err) => {
            console.error('Failed to spawn editor process:', err);
            try {
              onEditorClose();
            } catch (e) {
              console.debug('onEditorClose callback threw:', e);
            }
            resolve();
          });
        });
      }

      default:
        throw new Error(`Unsupported editor: ${editor}`);
    }
  } catch (_error) {
    console.error(_error);
  }
}
