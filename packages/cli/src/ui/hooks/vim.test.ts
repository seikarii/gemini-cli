/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, type MockInstance } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import React from 'react';
import { useVim } from './vim.js';
import type { TextBuffer } from '../components/shared/text-buffer.js';
import { textBufferReducer } from '../components/shared/text-buffer.js';
import type { Key } from './useKeypress.js';

// Mock the VimModeContext
import type { VimMode } from './vim.js';

const mockVimContext = {
  vimEnabled: true,
  vimMode: 'NORMAL' as VimMode,
  toggleVimEnabled: vi.fn(),
  setVimMode: vi.fn(),
};

vi.mock('../contexts/VimModeContext.js', () => ({
  useVimMode: () => mockVimContext,
  VimModeProvider: ({ children }: { children: React.ReactNode }) => children,
}));

// Test constants
const makeKey = (seq: string, name?: string) => ({
  sequence: seq,
  name: name || seq,
  ctrl: false,
  meta: false,
  shift: false,
  paste: false,
});

const TEST_SEQUENCES = {
  ESCAPE: makeKey('\u001b', 'escape'),
  LEFT: makeKey('h'),
  RIGHT: makeKey('l'),
  UP: makeKey('k'),
  DOWN: makeKey('j'),
  INSERT: makeKey('i'),
  APPEND: makeKey('a'),
  DELETE_CHAR: makeKey('x'),
  DELETE: makeKey('d'),
  CHANGE: makeKey('c'),
  WORD_FORWARD: makeKey('w'),
  WORD_BACKWARD: makeKey('b'),
  WORD_END: makeKey('e'),
  LINE_START: makeKey('0'),
  LINE_END: makeKey('$'),
  REPEAT: makeKey('.'),
} as const;

describe('useVim hook', () => {
  let mockBuffer: Partial<TextBuffer>;
  let mockHandleFinalSubmit: MockInstance<(value: string) => void>;

  const createMockBuffer = (
    text = 'hello world',
    cursor: [number, number] = [0, 5],
  ) => {
    const cursorState = { pos: cursor };
    const lines = text.split('\n');

    return {
      lines,
      get cursor() {
        return cursorState.pos;
      },
      set cursor(newPos: [number, number]) {
        cursorState.pos = newPos;
      },
      text,
      move: vi.fn().mockImplementation((direction: string) => {
        let [row, col] = cursorState.pos;
        const _line = lines[row] || '';
        if (direction === 'left') {
          col = Math.max(0, col - 1);
        } else if (direction === 'right') {
          col = Math.min(_line.length, col + 1);
        } else if (direction === 'home') {
          col = 0;
        } else if (direction === 'end') {
          col = _line.length;
        }
        cursorState.pos = [row, col];
      }),
      del: vi.fn(),
      moveToOffset: vi.fn(),
      insert: vi.fn(),
      newline: vi.fn(),
      replaceRangeByOffset: vi.fn(),
      handleInput: vi.fn(),
      setText: vi.fn(),
      // Vim-specific methods
      vimDeleteWordForward: vi.fn(),
      vimDeleteWordBackward: vi.fn(),
      vimDeleteWordEnd: vi.fn(),
      vimChangeWordForward: vi.fn(),
      vimChangeWordBackward: vi.fn(),
      vimChangeWordEnd: vi.fn(),
      vimDeleteLine: vi.fn(),
      vimChangeLine: vi.fn(),
      vimDeleteToEndOfLine: vi.fn(),
      vimChangeToEndOfLine: vi.fn(),
      vimChangeMovement: vi.fn(),
      vimMoveLeft: vi.fn(),
      vimMoveRight: vi.fn(),
      vimMoveUp: vi.fn(),
      vimMoveDown: vi.fn(),
      vimMoveWordForward: vi.fn(),
      vimMoveWordBackward: vi.fn(),
      vimMoveWordEnd: vi.fn(),
      vimDeleteChar: vi.fn(),
      vimInsertAtCursor: vi.fn(),
      vimAppendAtCursor: vi.fn().mockImplementation(() => {
        // Append moves cursor right (vim 'a' behavior - position after current char)
        const [row, col] = cursorState.pos;
        const _line = lines[row] || '';
        // In vim, 'a' moves cursor to position after current character
        // This allows inserting at the end of the line
        cursorState.pos = [row, col + 1];
      }),
      vimOpenLineBelow: vi.fn(),
      vimOpenLineAbove: vi.fn(),
      vimAppendAtLineEnd: vi.fn(),
      vimInsertAtLineStart: vi.fn(),
      vimMoveToLineStart: vi.fn(),
      vimMoveToLineEnd: vi.fn(),
      vimMoveToFirstNonWhitespace: vi.fn(),
      vimMoveToFirstLine: vi.fn(),
      vimMoveToLastLine: vi.fn(),
      vimMoveToLine: vi.fn(),
      vimEscapeInsertMode: vi.fn().mockImplementation(() => {
        // Escape moves cursor left unless at beginning of line
        const [row, col] = cursorState.pos;
        if (col > 0) {
          cursorState.pos = [row, col - 1];
        }
      }),
    };
  };

  const _createMockSettings = (vimMode = true) => ({
    getValue: vi.fn().mockReturnValue(vimMode),
    setValue: vi.fn(),
    merged: { vimMode },
  });

  const renderVimHook = (buffer?: Partial<TextBuffer>) =>
    renderHook(() =>
      useVim((buffer || mockBuffer) as TextBuffer, mockHandleFinalSubmit as unknown as (value: string) => void),
    );

  // Test helper to create initial TextBufferState objects with required viewportWidth
  const makeInitialState = (overrides: Record<string, unknown> = {}) => ({
    viewportWidth: 80,
    lines: ['hello world'],
    cursorRow: 0,
    cursorCol: 0,
    preferredCol: null,
    undoStack: [],
    redoStack: [],
    clipboard: null,
    selectionAnchor: null,
    ...overrides,
  });

  // Helper to ensure test states passed into textBufferReducer include required fields.
  const ensureState = (s: Record<string, unknown>) => ({
    viewportWidth: 80,
    lines: ['hello world'],
    cursorRow: 0,
    cursorCol: 0,
    preferredCol: null,
    undoStack: [],
    redoStack: [],
    clipboard: null,
    selectionAnchor: null,
    ...s,
  });

  const exitInsertMode = (result: {
    current: {
      handleInput: (input: Key) => void;
    };
  }) => {
    act(() => {
      result.current.handleInput({ sequence: '\u001b', name: 'escape', ctrl: false, meta: false, shift: false, paste: false });
    });
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockHandleFinalSubmit = vi.fn();
    mockBuffer = createMockBuffer();
    // Reset mock context to default state
    mockVimContext.vimEnabled = true;
    mockVimContext.vimMode = 'NORMAL';
    mockVimContext.toggleVimEnabled.mockClear();
    mockVimContext.setVimMode.mockClear();
  });

  describe('Mode switching', () => {
    it('should start in NORMAL mode', () => {
      const { result } = renderVimHook();
      expect(result.current.mode).toBe('NORMAL');
    });

    it('should switch to INSERT mode with i command', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(TEST_SEQUENCES.INSERT);
      });

      expect(result.current.mode).toBe('INSERT');
      expect(mockVimContext.setVimMode).toHaveBeenCalledWith('INSERT');
    });

    it('should switch back to NORMAL mode with Escape', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(TEST_SEQUENCES.INSERT);
      });
      expect(result.current.mode).toBe('INSERT');

      exitInsertMode(result);
      expect(result.current.mode).toBe('NORMAL');
    });

    it('should properly handle escape followed immediately by a command', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 6]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(result.current.mode).toBe('INSERT');

      vi.clearAllMocks();

      exitInsertMode(result);
      expect(result.current.mode).toBe('NORMAL');

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveWordBackward).toHaveBeenCalledWith(1);
    });
  });

  describe('Navigation commands', () => {
    it('should handle h (left movement)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimMoveLeft).toHaveBeenCalledWith(1);
    });

    it('should handle l (right movement)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimMoveRight).toHaveBeenCalledWith(1);
    });

    it('should handle j (down movement)', () => {
      const testBuffer = createMockBuffer('first line\nsecond line');
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveDown).toHaveBeenCalledWith(1);
    });

    it('should handle k (up movement)', () => {
      const testBuffer = createMockBuffer('first line\nsecond line');
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveUp).toHaveBeenCalledWith(1);
    });

    it('should handle 0 (move to start of line)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimMoveToLineStart).toHaveBeenCalled();
    });

    it('should handle $ (move to end of line)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimMoveToLineEnd).toHaveBeenCalled();
    });
  });

  describe('Mode switching commands', () => {
    it('should handle a (append after cursor)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimAppendAtCursor).toHaveBeenCalled();
      expect(result.current.mode).toBe('INSERT');
    });

    it('should handle A (append at end of line)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimAppendAtLineEnd).toHaveBeenCalled();
      expect(result.current.mode).toBe('INSERT');
    });

    it('should handle o (open line below)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimOpenLineBelow).toHaveBeenCalled();
      expect(result.current.mode).toBe('INSERT');
    });

    it('should handle O (open line above)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimOpenLineAbove).toHaveBeenCalled();
      expect(result.current.mode).toBe('INSERT');
    });
  });

  describe('Edit commands', () => {
    it('should handle x (delete character)', () => {
      const { result } = renderVimHook();
      vi.clearAllMocks();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.vimDeleteChar).toHaveBeenCalledWith(1);
    });

    it('should move cursor left when deleting last character on line (vim behavior)', () => {
      const testBuffer = createMockBuffer('hello', [0, 4]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);
    });

    it('should handle first d key (sets pending state)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.replaceRangeByOffset).not.toHaveBeenCalled();
    });
  });

  describe('Count handling', () => {
    it('should handle count input and return to count 0 after command', () => {
      const { result } = renderVimHook();

      act(() => {
        const handled = result.current.handleInput(makeKey(''));
        expect(handled).toBe(true);
      });

      act(() => {
        const handled = result.current.handleInput(makeKey(''));
        expect(handled).toBe(true);
      });

      expect(mockBuffer.vimMoveLeft).toHaveBeenCalledWith(3);
    });

    it('should only delete 1 character with x command when no count is specified', () => {
      const testBuffer = createMockBuffer();
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);
    });
  });

  describe('Word movement', () => {
    it('should properly initialize vim hook with word movement support', () => {
      const testBuffer = createMockBuffer('cat elephant mouse', [0, 0]);
      const { result } = renderVimHook(testBuffer);

      expect(result.current.vimModeEnabled).toBe(true);
      expect(result.current.mode).toBe('NORMAL');
      expect(result.current.handleInput).toBeDefined();
    });

    it('should support vim mode and basic operations across multiple lines', () => {
      const testBuffer = createMockBuffer(
        'first line word\nsecond line word',
        [0, 11],
      );
      const { result } = renderVimHook(testBuffer);

      expect(result.current.vimModeEnabled).toBe(true);
      expect(result.current.mode).toBe('NORMAL');
      expect(result.current.handleInput).toBeDefined();
      expect(testBuffer.replaceRangeByOffset).toBeDefined();
      expect(testBuffer.moveToOffset).toBeDefined();
    });

    it('should handle w (next word)', () => {
      const testBuffer = createMockBuffer('hello world test');
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveWordForward).toHaveBeenCalledWith(1);
    });

    it('should handle b (previous word)', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 6]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveWordBackward).toHaveBeenCalledWith(1);
    });

    it('should handle e (end of word)', () => {
      const testBuffer = createMockBuffer('hello world test');
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveWordEnd).toHaveBeenCalledWith(1);
    });

    it('should handle w when cursor is on the last word', () => {
      const testBuffer = createMockBuffer('hello world', [0, 8]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveWordForward).toHaveBeenCalledWith(1);
    });

    it('should handle first c key (sets pending change state)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(result.current.mode).toBe('NORMAL');
      expect(mockBuffer.del).not.toHaveBeenCalled();
    });

    it('should clear pending state on invalid command sequence (df)', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.replaceRangeByOffset).not.toHaveBeenCalled();
      expect(mockBuffer.del).not.toHaveBeenCalled();
    });

    it('should clear pending state with Escape in NORMAL mode', () => {
      const { result } = renderVimHook();

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      exitInsertMode(result);

      expect(mockBuffer.replaceRangeByOffset).not.toHaveBeenCalled();
    });
  });

  describe('Disabled vim mode', () => {
    it('should not respond to vim commands when disabled', () => {
      mockVimContext.vimEnabled = false;
      const { result } = renderVimHook(mockBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(mockBuffer.move).not.toHaveBeenCalled();
    });
  });

  // These tests are no longer applicable at the hook level

  describe('Command repeat system', () => {
    it('should repeat x command from current cursor position', () => {
      const testBuffer = createMockBuffer('abcd\nefgh\nijkl', [0, 1]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);

      testBuffer.cursor = [1, 2];

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);
    });

    it('should repeat dd command from current position', () => {
      const testBuffer = createMockBuffer('line1\nline2\nline3', [1, 0]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteLine).toHaveBeenCalledTimes(1);

      testBuffer.cursor = [0, 0];

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimDeleteLine).toHaveBeenCalledTimes(2);
    });

    it('should repeat ce command from current position', () => {
      const testBuffer = createMockBuffer('word', [0, 0]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimChangeWordEnd).toHaveBeenCalledTimes(1);

      // Exit INSERT mode to complete the command
      exitInsertMode(result);

      testBuffer.cursor = [0, 2];

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimChangeWordEnd).toHaveBeenCalledTimes(2);
    });

    it('should repeat cc command from current position', () => {
      const testBuffer = createMockBuffer('line1\nline2\nline3', [1, 2]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimChangeLine).toHaveBeenCalledTimes(1);

      // Exit INSERT mode to complete the command
      exitInsertMode(result);

      testBuffer.cursor = [0, 1];

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimChangeLine).toHaveBeenCalledTimes(2);
    });

    it('should repeat cw command from current position', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 6]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimChangeWordForward).toHaveBeenCalledTimes(1);

      // Exit INSERT mode to complete the command
      exitInsertMode(result);

      testBuffer.cursor = [0, 0];

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimChangeWordForward).toHaveBeenCalledTimes(2);
    });

    it('should repeat D command from current position', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 6]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteToEndOfLine).toHaveBeenCalledTimes(1);

      testBuffer.cursor = [0, 2];
      vi.clearAllMocks(); // Clear all mocks instead of just one method

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimDeleteToEndOfLine).toHaveBeenCalledTimes(1);
    });

    it('should repeat C command from current position', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 6]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimChangeToEndOfLine).toHaveBeenCalledTimes(1);

      // Exit INSERT mode to complete the command
      exitInsertMode(result);

      testBuffer.cursor = [0, 2];

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimChangeToEndOfLine).toHaveBeenCalledTimes(2);
    });

    it('should repeat command after cursor movement', () => {
      const testBuffer = createMockBuffer('test text', [0, 0]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);

      testBuffer.cursor = [0, 2];

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(testBuffer.vimDeleteChar).toHaveBeenCalledWith(1);
    });

    it('should move cursor to the correct position after exiting INSERT mode with "a"', () => {
      const testBuffer = createMockBuffer('hello world', [0, 10]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });
      expect(result.current.mode).toBe('INSERT');
      expect(testBuffer.cursor).toEqual([0, 11]);

      exitInsertMode(result);
      expect(result.current.mode).toBe('NORMAL');
      expect(testBuffer.cursor).toEqual([0, 10]);
    });
  });

  describe('Special characters and edge cases', () => {
    it('should handle ^ (move to first non-whitespace character)', () => {
      const testBuffer = createMockBuffer('   hello world', [0, 5]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveToFirstNonWhitespace).toHaveBeenCalled();
    });

    it('should handle G without count (go to last line)', () => {
      const testBuffer = createMockBuffer('line1\nline2\nline3', [0, 0]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveToLastLine).toHaveBeenCalled();
    });

    it('should handle gg (go to first line)', () => {
      const testBuffer = createMockBuffer('line1\nline2\nline3', [2, 0]);
      const { result } = renderVimHook(testBuffer);

      // First 'g' sets pending state
      act(() => {
        result.current.handleInput(makeKey(''));
      });

      // Second 'g' executes the command
      act(() => {
        result.current.handleInput(makeKey(''));
      });

      expect(testBuffer.vimMoveToFirstLine).toHaveBeenCalled();
    });

    it('should handle count with movement commands', () => {
      const testBuffer = createMockBuffer('hello world test', [0, 0]);
      const { result } = renderVimHook(testBuffer);

      act(() => {
        result.current.handleInput(makeKey(''));
      });

      act(() => {
        result.current.handleInput(TEST_SEQUENCES.WORD_FORWARD);
      });

      expect(testBuffer.vimMoveWordForward).toHaveBeenCalledWith(3);
    });
  });

  describe('Vim word operations', () => {
    describe('dw (delete word forward)', () => {
      it('should delete from cursor to start of next word', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordForward).toHaveBeenCalledWith(1);
      });

      it('should actually delete the complete word including trailing space', () => {
        // This test uses the real text-buffer reducer instead of mocks
        const initialState = makeInitialState({
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 0,
        });

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_forward',
          payload: { count: 1 },
        });

        // Should delete "hello " (word + space), leaving "world test"
        expect(result.lines).toEqual(['world test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });

      it('should delete word from middle of word correctly', () => {
        const initialState = makeInitialState({
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 2, // cursor on 'l' in "hello"
        });

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_forward',
          payload: { count: 1 },
        });

        // Should delete "llo " (rest of word + space), leaving "he world test"
        expect(result.lines).toEqual(['heworld test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(2);
      });

      it('should handle dw at end of line', () => {
        const initialState = makeInitialState({
          lines: ['hello world'],
          cursorRow: 0,
          cursorCol: 6, // cursor on 'w' in "world"
        });

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_forward',
          payload: { count: 1 },
        });

        // Should delete "world" (no trailing space at end), leaving "hello "
        expect(result.lines).toEqual(['hello ']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });

      it('should delete multiple words with count', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordForward).toHaveBeenCalledWith(2);
      });

      it('should record command for repeat with dot', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Execute dw
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        vi.clearAllMocks();

        // Execute dot repeat
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordForward).toHaveBeenCalledWith(1);
      });
    });

    describe('de (delete word end)', () => {
      it('should delete from cursor to end of current word', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 1]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordEnd).toHaveBeenCalledWith(1);
      });

      it('should handle count with de', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordEnd).toHaveBeenCalledWith(3);
      });
    });

    describe('cw (change word forward)', () => {
      it('should change from cursor to start of next word and enter INSERT mode', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordForward).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
        expect(mockVimContext.setVimMode).toHaveBeenCalledWith('INSERT');
      });

      it('should handle count with cw', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordForward).toHaveBeenCalledWith(2);
        expect(result.current.mode).toBe('INSERT');
      });

      it('should be repeatable with dot', () => {
        const testBuffer = createMockBuffer('hello world test more', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Execute cw
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Exit INSERT mode
        exitInsertMode(result);

        vi.clearAllMocks();
        mockVimContext.setVimMode.mockClear();

        // Execute dot repeat
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordForward).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
      });
    });

    describe('ce (change word end)', () => {
      it('should change from cursor to end of word and enter INSERT mode', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 1]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordEnd).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
      });

      it('should handle count with ce', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordEnd).toHaveBeenCalledWith(2);
        expect(result.current.mode).toBe('INSERT');
      });
    });

    describe('cc (change line)', () => {
      it('should change entire line and enter INSERT mode', () => {
        const testBuffer = createMockBuffer('hello world\nsecond line', [0, 5]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeLine).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
      });

      it('should change multiple lines with count', () => {
        const testBuffer = createMockBuffer(
          'line1\nline2\nline3\nline4',
          [1, 0],
        );
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeLine).toHaveBeenCalledWith(3);
        expect(result.current.mode).toBe('INSERT');
      });

      it('should be repeatable with dot', () => {
        const testBuffer = createMockBuffer('line1\nline2\nline3', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Execute cc
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Exit INSERT mode
        exitInsertMode(result);

        vi.clearAllMocks();
        mockVimContext.setVimMode.mockClear();

        // Execute dot repeat
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeLine).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
      });
    });

    describe('db (delete word backward)', () => {
      it('should delete from cursor to start of previous word', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 11]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordBackward).toHaveBeenCalledWith(1);
      });

      it('should handle count with db', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 18]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordBackward).toHaveBeenCalledWith(2);
      });
    });

    describe('cb (change word backward)', () => {
      it('should change from cursor to start of previous word and enter INSERT mode', () => {
        const testBuffer = createMockBuffer('hello world test', [0, 11]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordBackward).toHaveBeenCalledWith(1);
        expect(result.current.mode).toBe('INSERT');
      });

      it('should handle count with cb', () => {
        const testBuffer = createMockBuffer('one two three four', [0, 18]);
        const { result } = renderVimHook(testBuffer);

        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeWordBackward).toHaveBeenCalledWith(3);
        expect(result.current.mode).toBe('INSERT');
      });
    });

    describe('Pending state handling', () => {
      it('should clear pending delete state after dw', () => {
        const testBuffer = createMockBuffer('hello world', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Press 'd' to enter pending delete state
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Complete with 'w'
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Next 'd' should start a new pending state, not continue the previous one
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // This should trigger dd (delete line), not an error
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteLine).toHaveBeenCalledWith(1);
      });

      it('should clear pending change state after cw', () => {
        const testBuffer = createMockBuffer('hello world', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Execute cw
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Exit INSERT mode
        exitInsertMode(result);

        // Next 'c' should start a new pending state
        act(() => {
          result.current.handleInput(makeKey(''));
        });
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimChangeLine).toHaveBeenCalledWith(1);
      });

      it('should clear pending state with escape', () => {
        const testBuffer = createMockBuffer('hello world', [0, 0]);
        const { result } = renderVimHook(testBuffer);

        // Enter pending delete state
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        // Press escape to clear pending state
        act(() => {
          result.current.handleInput(makeKey('001b', 'escape'));
        });

        // Now 'w' should just move cursor, not delete
        act(() => {
          result.current.handleInput(makeKey(''));
        });

        expect(testBuffer.vimDeleteWordForward).not.toHaveBeenCalled();
        // w should move to next word after clearing pending state
        expect(testBuffer.vimMoveWordForward).toHaveBeenCalledWith(1);
      });
    });

    describe('NORMAL mode escape behavior', () => {
      it('should pass escape through when no pending operator is active', () => {
        mockVimContext.vimMode = 'NORMAL';
        const { result } = renderVimHook();

        const handled = result.current.handleInput(makeKey('001b', 'escape'));

        expect(handled).toBe(false);
      });

      it('should handle escape and clear pending operator', () => {
        mockVimContext.vimMode = 'NORMAL';
        const { result } = renderVimHook();

        act(() => {
          result.current.handleInput(makeKey(''));
        });

        let handled: boolean | undefined;
        act(() => {
          handled = result.current.handleInput(makeKey('001b', 'escape'));
        });

        expect(handled).toBe(true);
      });
    });
  });

  describe('Shell command pass-through', () => {
    it('should pass through ctrl+r in INSERT mode', () => {
      mockVimContext.vimMode = 'INSERT';
      const { result } = renderVimHook();

      const handled = result.current.handleInput({ name: 'r', ctrl: true, meta: false, shift: false, paste: false, sequence: 'r' });

      expect(handled).toBe(false);
    });

    it('should pass through ! in INSERT mode when buffer is empty', () => {
      mockVimContext.vimMode = 'INSERT';
      const emptyBuffer = createMockBuffer('');
      const { result } = renderVimHook(emptyBuffer);

      const handled = result.current.handleInput(makeKey(''));

      expect(handled).toBe(false);
    });

    it('should handle ! as input in INSERT mode when buffer is not empty', () => {
      mockVimContext.vimMode = 'INSERT';
      const nonEmptyBuffer = createMockBuffer('not empty');
      const { result } = renderVimHook(nonEmptyBuffer);
      const key = { sequence: '!', name: '!', ctrl: false, meta: false, shift: false, paste: false };

      act(() => {
        result.current.handleInput(key);
      });

      expect(nonEmptyBuffer.handleInput).toHaveBeenCalledWith(
        expect.objectContaining(key),
      );
    });
  });

  // Line operations (dd, cc) are tested in text-buffer.test.ts

  describe('Reducer-based integration tests', () => {
    describe('de (delete word end)', () => {
      it('should delete from cursor to end of current word', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 1, // cursor on 'e' in "hello"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_end',
          payload: { count: 1 },
        });

        // Should delete "ello" (from cursor to end of word), leaving "h world test"
        expect(result.lines).toEqual(['h world test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(1);
      });

      it('should delete multiple word ends with count', () => {
        const initialState = {
          lines: ['hello world test more'],
          cursorRow: 0,
          cursorCol: 1, // cursor on 'e' in "hello"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_end',
          payload: { count: 2 },
        });

        // Should delete "ello world" (to end of second word), leaving "h test more"
        expect(result.lines).toEqual(['h test more']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(1);
      });
    });

    describe('db (delete word backward)', () => {
      it('should delete from cursor to start of previous word', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 11, // cursor on 't' in "test"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_backward',
          payload: { count: 1 },
        });

        // Should delete "world" (previous word only), leaving "hello  test"
        expect(result.lines).toEqual(['hello  test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });

      it('should delete multiple words backward with count', () => {
        const initialState = {
          lines: ['hello world test more'],
          cursorRow: 0,
          cursorCol: 17, // cursor on 'm' in "more"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_word_backward',
          payload: { count: 2 },
        });

        // Should delete "world test " (two words backward), leaving "hello more"
        expect(result.lines).toEqual(['hello more']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });
    });

    describe('cw (change word forward)', () => {
      it('should delete from cursor to start of next word', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 0, // cursor on 'h' in "hello"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_word_forward',
          payload: { count: 1 },
        });

        // Should delete "hello " (word + space), leaving "world test"
        expect(result.lines).toEqual(['world test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });

      it('should change multiple words with count', () => {
        const initialState = {
          lines: ['hello world test more'],
          cursorRow: 0,
          cursorCol: 0,
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_word_forward',
          payload: { count: 2 },
        });

        // Should delete "hello world " (two words), leaving "test more"
        expect(result.lines).toEqual(['test more']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });
    });

    describe('ce (change word end)', () => {
      it('should change from cursor to end of current word', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 1, // cursor on 'e' in "hello"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_word_end',
          payload: { count: 1 },
        });

        // Should delete "ello" (from cursor to end of word), leaving "h world test"
        expect(result.lines).toEqual(['h world test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(1);
      });

      it('should change multiple word ends with count', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 1, // cursor on 'e' in "hello"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_word_end',
          payload: { count: 2 },
        });

        // Should delete "ello world" (to end of second word), leaving "h test"
        expect(result.lines).toEqual(['h test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(1);
      });
    });

    describe('cb (change word backward)', () => {
      it('should change from cursor to start of previous word', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 11, // cursor on 't' in "test"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_word_backward',
          payload: { count: 1 },
        });

        // Should delete "world" (previous word only), leaving "hello  test"
        expect(result.lines).toEqual(['hello  test']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });
    });

    describe('cc (change line)', () => {
      it('should clear the line and place cursor at the start', () => {
        const initialState = {
          lines: ['  hello world'],
          cursorRow: 0,
          cursorCol: 5, // cursor on 'o'
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_change_line',
          payload: { count: 1 },
        });

        expect(result.lines).toEqual(['']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });
    });

    describe('dd (delete line)', () => {
      it('should delete the current line', () => {
        const initialState = {
          lines: ['line1', 'line2', 'line3'],
          cursorRow: 1,
          cursorCol: 2,
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_line',
          payload: { count: 1 },
        });

        expect(result.lines).toEqual(['line1', 'line3']);
        expect(result.cursorRow).toBe(1);
        expect(result.cursorCol).toBe(0);
      });

      it('should delete multiple lines with count', () => {
        const initialState = {
          lines: ['line1', 'line2', 'line3', 'line4'],
          cursorRow: 1,
          cursorCol: 2,
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
        };

        const result = textBufferReducer(ensureState(initialState), {
          type: 'vim_delete_line',
          payload: { count: 2 },
        });

        // Should delete lines 1 and 2
        expect(result.lines).toEqual(['line1', 'line4']);
        expect(result.cursorRow).toBe(1);
        expect(result.cursorCol).toBe(0);
      });

      it('should handle deleting last line', () => {
        const initialState = {
          lines: ['only line'],
          cursorRow: 0,
          cursorCol: 3,
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
          viewportWidth: 80,
        };

        const result = textBufferReducer(initialState, {
          type: 'vim_delete_line',
          payload: { count: 1 },
        });

        // Should leave an empty line when deleting the only line
        expect(result.lines).toEqual(['']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });
    });

    describe('D (delete to end of line)', () => {
      it('should delete from cursor to end of line', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 6, // cursor on 'w' in "world"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
          viewportWidth: 80,
        };

        const result = textBufferReducer(initialState, {
          type: 'vim_delete_to_end_of_line',
        });

        // Should delete "world test", leaving "hello "
        expect(result.lines).toEqual(['hello ']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });

      it('should handle D at end of line', () => {
        const initialState = {
          lines: ['hello world'],
          cursorRow: 0,
          cursorCol: 11, // cursor at end
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
          viewportWidth: 80,
        };

        const result = textBufferReducer(initialState, {
          type: 'vim_delete_to_end_of_line',
        });

        // Should not change anything when at end of line
        expect(result.lines).toEqual(['hello world']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(11);
      });
    });

    describe('C (change to end of line)', () => {
      it('should change from cursor to end of line', () => {
        const initialState = {
          lines: ['hello world test'],
          cursorRow: 0,
          cursorCol: 6, // cursor on 'w' in "world"
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
          viewportWidth: 80,
        };

        const result = textBufferReducer(initialState, {
          type: 'vim_change_to_end_of_line',
        });

        // Should delete "world test", leaving "hello "
        expect(result.lines).toEqual(['hello ']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(6);
      });

      it('should handle C at beginning of line', () => {
        const initialState = {
          lines: ['hello world'],
          cursorRow: 0,
          cursorCol: 0,
          preferredCol: null,
          undoStack: [],
          redoStack: [],
          clipboard: null,
          selectionAnchor: null,
          viewportWidth: 80,
        };

        const result = textBufferReducer(initialState, {
          type: 'vim_change_to_end_of_line',
        });

        // Should delete entire line content
        expect(result.lines).toEqual(['']);
        expect(result.cursorRow).toBe(0);
        expect(result.cursorCol).toBe(0);
      });
    });
  });
});
