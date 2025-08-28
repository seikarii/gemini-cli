/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { ToolSelectionGuidance } from '../services/toolSelectionGuidance.js';
import { Content } from '@google/genai';

describe('ToolSelectionGuidance', () => {
  it('should recommend AST tools for code file modifications', () => {
    const userMessage =
      'Update the handleSubmit function in src/components/Form.tsx';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'I need to modify some React components' }],
      },
      {
        role: 'model',
        parts: [
          { text: 'Sure, I can help with React component modifications.' },
        ],
      },
    ];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('CONTEXTUAL TOOL GUIDANCE');
    expect(guidance).toContain('modify');
    expect(guidance).toContain('function');
    expect(guidance).toContain('upsert_code_block');
    expect(guidance).toContain('Form.tsx');
  });

  it('should recommend replace tool for config file changes', () => {
    const userMessage = 'Update the port setting in package.json to 3001';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('replace');
    expect(guidance).toContain('package.json');
    expect(guidance).toContain('config');
  });

  it('should detect recent tool failures and recommend alternatives', () => {
    const userMessage = 'Fix the login function in auth.ts';
    const conversationHistory: Content[] = [
      {
        role: 'user',
        parts: [{ text: 'Update the login function using replace tool' }],
      },
      {
        role: 'model',
        parts: [
          { text: 'Error: replace tool failed due to multiple matches found' },
        ],
      },
    ];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('Previous replace operation failed');
    expect(guidance).toContain('upsert_code_block');
    expect(guidance).toContain('Recent Error Context');
  });

  it('should recommend write_file for new file creation', () => {
    const userMessage =
      'Create a new UserService.ts file with basic CRUD operations';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('create');
    expect(guidance).toContain('write_file');
    expect(guidance).toContain('New file creation');
  });

  it('should handle complex scenarios with multiple file types', () => {
    const userMessage =
      'Update both config.json and UserService.ts to add new validation';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('config.json');
    expect(guidance).toContain('UserService.ts');
    expect(guidance).toContain('Code files');
    expect(guidance).toContain('Config files');
  });

  it('should extract file references from quoted strings', () => {
    const userMessage =
      'Please modify "src/utils/helper.js" and `config/settings.yml`';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('helper.js');
    expect(guidance).toContain('settings.yml');
  });

  it('should provide usage guidance for different operation types', () => {
    const userMessage =
      'Analyze the performance issues in the database connection code';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('analyze');
    expect(guidance).toContain('USAGE GUIDANCE');
  });

  it('should include warnings for potentially problematic scenarios', () => {
    const userMessage =
      'Change all variable names in the entire codebase from oldName to newName';
    const conversationHistory: Content[] = [];

    const guidance = ToolSelectionGuidance.generateGuidance(
      userMessage,
      conversationHistory,
    );

    expect(guidance).toContain('WARNINGS');
  });
});
