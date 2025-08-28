/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { SlashCommand, CommandKind } from './types.js';
import type { Config } from '@google/gemini-cli-core';

interface GeminiAgentClass {
  new (config: Config): {
    start: () => Promise<void>;
  };
}

export const mewCommand: SlashCommand = {
  kind: CommandKind.BUILT_IN,
  name: 'mew',
  description: 'Launches the Mew agent window.',
  action: async (context) => {
    if (!context.services.config) {
      context.ui.addItem(
        {
          type: 'error',
          text: 'Config not available, cannot start Mew agent.',
        },
        Date.now(),
      );
      return {
        type: 'message',
        messageType: 'error',
        content: 'Config not available, cannot start Mew agent.',
      };
    }

    context.ui.addItem(
      {
        type: 'info',
        text: 'Starting Mew agent...',
      },
      Date.now(),
    );

    try {
      const { GeminiAgent } = await import(
        '@google/gemini-cli-mew-upgrade/agent/gemini-agent.js'
      );
      const Agent = GeminiAgent as GeminiAgentClass;
      const agent = new Agent(context.services.config);
      await agent.start();

      const open = (await import('open')).default;
      open('http://localhost:3000');

      return {
        type: 'message',
        messageType: 'info',
        content: 'Mew agent started at http://localhost:3000',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      context.ui.addItem(
        {
          type: 'error',
          text: `Failed to start Mew agent: ${errorMessage}`,
        },
        Date.now(),
      );
      return {
        type: 'message',
        messageType: 'error',
        content: `Failed to start Mew agent: ${errorMessage}`,
      };
    }
  },
};
