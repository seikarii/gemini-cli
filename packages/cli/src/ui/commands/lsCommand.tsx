/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { SlashCommand, CommandContext, CommandKind } from './types.js';
import { MessageType } from '../types.js';
import { executeToolCall } from '@google/gemini-cli-core';

export const lsCommand: SlashCommand = {
  name: 'ls',
  description: 'List files in the current or specified directory',
  kind: CommandKind.BUILT_IN,
  action: async (context: CommandContext, args: string) => {
    const {
      ui: { addItem },
      services: { config },
    } = context;

    if (!config) {
      addItem(
        {
          type: MessageType.ERROR,
          text: 'Configuration is not available.',
        },
        Date.now(),
      );
      return;
    }

    const path = args.trim() || '.';
    const toolCallRequest = {
      callId: `ls-${Date.now()}`,
      name: 'list_directory',
      args: { path },
      isClientInitiated: true,
      prompt_id: `ls-prompt-${Date.now()}`,
    };

    try {
      const result = await executeToolCall(
        config,
        toolCallRequest,
        new AbortController().signal,
      );
      if (result.error) {
        addItem(
          {
            type: MessageType.ERROR,
            text: `Error listing directory: ${result.error.message}`,
          },
          Date.now(),
        );
      } else {
        addItem(
          {
            type: MessageType.INFO,
            text: result.responseParts.map((part) => part.text || '').join(''),
          },
          Date.now(),
        );
      }
    } catch (error) {
      addItem(
        {
          type: MessageType.ERROR,
          text: `Error executing ls command: ${error instanceof Error ? error.message : String(error)}`,
        },
        Date.now(),
      );
    }
  },
};
