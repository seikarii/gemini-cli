/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { ActionScriptParser, ActionScriptBuilder, ActionNode } from './action-script.js';
import { ActionPriority } from './action-system.js';

describe('ActionScriptParser', () => {
  const parser = new ActionScriptParser();

  describe('parse', () => {
    it('should parse a valid action script', () => {
      const scriptJson = JSON.stringify({
        id: 'test_script',
        rootNode: {
          type: 'action',
          toolName: 'read_file',
          parameters: { file_path: 'test.txt' },
          priority: 'normal'
        }
      });

      const result = parser.parse(scriptJson);

      expect(result.id).toBe('test_script');
      expect(result.rootNode.type).toBe('action');
      expect((result.rootNode as ActionNode).toolName).toBe('read_file');
    });

    it('should throw error for invalid JSON', () => {
      const invalidJson = '{ invalid json }';

      expect(() => parser.parse(invalidJson)).toThrow();
    });

    it('should throw error for missing id', () => {
      const scriptJson = JSON.stringify({
        rootNode: {
          type: 'action',
          toolName: 'read_file'
        }
      });

      expect(() => parser.parse(scriptJson)).toThrow('Invalid script: missing id or rootNode');
    });

    it('should throw error for missing rootNode', () => {
      const scriptJson = JSON.stringify({
        id: 'test_script'
      });

      expect(() => parser.parse(scriptJson)).toThrow('Invalid script: missing id or rootNode');
    });
  });

  describe('validateNode', () => {
    it('should validate action node', () => {
      const actionNode = {
        type: 'action',
        toolName: 'read_file',
        parameters: { file_path: 'test.txt' }
      };

      expect(() => parser['validateNode'](actionNode)).not.toThrow();
    });

    it('should throw error for action node without toolName', () => {
      const invalidActionNode = {
        type: 'action',
        parameters: { file_path: 'test.txt' }
      };

      expect(() => parser['validateNode'](invalidActionNode)).toThrow('Invalid action node: missing toolName');
    });

    it('should validate sequence node', () => {
      const sequenceNode = {
        type: 'sequence',
        nodes: [
          {
            type: 'action',
            toolName: 'read_file',
            parameters: { file_path: 'test.txt' }
          }
        ]
      };

      expect(() => parser['validateNode'](sequenceNode)).not.toThrow();
    });

    it('should throw error for sequence node without nodes array', () => {
      const invalidSequenceNode = {
        type: 'sequence'
      };

      expect(() => parser['validateNode'](invalidSequenceNode)).toThrow('Invalid sequence node: nodes must be an array');
    });

    it('should validate condition node', () => {
      const conditionNode = {
        type: 'condition',
        condition: 'true',
        thenNode: {
          type: 'action',
          toolName: 'read_file',
          parameters: { file_path: 'test.txt' }
        }
      };

      expect(() => parser['validateNode'](conditionNode)).not.toThrow();
    });

    it('should throw error for condition node without condition', () => {
      const invalidConditionNode = {
        type: 'condition',
        thenNode: {
          type: 'action',
          toolName: 'read_file'
        }
      };

      expect(() => parser['validateNode'](invalidConditionNode)).toThrow('Invalid condition node: missing condition or thenNode');
    });

    it('should throw error for condition node without thenNode', () => {
      const invalidConditionNode = {
        type: 'condition',
        condition: 'true'
      };

      expect(() => parser['validateNode'](invalidConditionNode)).toThrow('Invalid condition node: missing condition or thenNode');
    });
  });
});

describe('ActionScriptBuilder', () => {
  describe('action', () => {
    it('should create an action node', () => {
      const action = ActionScriptBuilder.action(
        'read_file',
        { file_path: 'test.txt' },
        ActionPriority.NORMAL,
        'Test action'
      );

      expect(action.type).toBe('action');
      expect(action.toolName).toBe('read_file');
      expect(action.parameters).toEqual({ file_path: 'test.txt' });
      expect(action.priority).toBe('normal');
      expect(action.description).toBe('Test action');
    });
  });

  describe('sequence', () => {
    it('should create a sequence node', () => {
      const action1 = ActionScriptBuilder.action('read_file', { file_path: 'test1.txt' });
      const action2 = ActionScriptBuilder.action('read_file', { file_path: 'test2.txt' });

      const sequence = ActionScriptBuilder.sequence([action1, action2], 'Test sequence');

      expect(sequence.type).toBe('sequence');
      expect(sequence.description).toBe('Test sequence');
      expect(sequence.nodes).toHaveLength(2);
      expect((sequence.nodes[0] as ActionNode).toolName).toBe('read_file');
      expect((sequence.nodes[1] as ActionNode).toolName).toBe('read_file');
    });
  });

  describe('parallel', () => {
    it('should create a parallel node', () => {
      const action1 = ActionScriptBuilder.action('read_file', { file_path: 'test1.txt' });
      const action2 = ActionScriptBuilder.action('read_file', { file_path: 'test2.txt' });

      const parallel = ActionScriptBuilder.parallel([action1, action2], 3, 'Test parallel');

      expect(parallel.type).toBe('parallel');
      expect(parallel.description).toBe('Test parallel');
      expect(parallel.maxConcurrency).toBe(3);
      expect(parallel.nodes).toHaveLength(2);
    });
  });

  describe('condition', () => {
    it('should create a condition node', () => {
      const thenAction = ActionScriptBuilder.action('read_file', { file_path: 'exists.txt' });
      const elseAction = ActionScriptBuilder.action('run_shell_command', { command: 'echo "not found"' });

      const condition = ActionScriptBuilder.condition(
        'file_exists("test.txt")',
        thenAction,
        elseAction,
        'Conditional execution'
      );

      expect(condition.type).toBe('condition');
      expect(condition.condition).toBe('file_exists("test.txt")');
      expect((condition.thenNode as ActionNode).toolName).toBe('read_file');
      expect((condition.elseNode as ActionNode).toolName).toBe('run_shell_command');
      expect(condition.description).toBe('Conditional execution');
    });
  });

  describe('build', () => {
    it('should build a complete action script', () => {
      const action1 = ActionScriptBuilder.action('list_dir', { path: '.' }, ActionPriority.NORMAL, 'List directory');
      const action2 = ActionScriptBuilder.action('read_file', { file_path: 'package.json' }, ActionPriority.HIGH, 'Read package.json');
      const parallel = ActionScriptBuilder.parallel([action2], undefined, 'Parallel processing');
      const sequence = ActionScriptBuilder.sequence([action1, parallel], 'Complete test script');

      const scriptBuilder = new ActionScriptBuilder('Test Script', 'A complete test script');
      scriptBuilder.setRoot(sequence);
      const script = scriptBuilder.build();

      expect(script.id).toMatch(/^script_/);
      expect(script.name).toBe('Test Script');
      expect(script.description).toBe('A complete test script');
      expect(script.rootNode.type).toBe('sequence');
      if (script.rootNode.type === 'sequence') {
        expect(script.rootNode.nodes).toHaveLength(2);
        expect(script.rootNode.nodes[0].type).toBe('action');
        expect(script.rootNode.nodes[1].type).toBe('parallel');
        if (script.rootNode.nodes[1].type === 'parallel') {
          expect(script.rootNode.nodes[1].nodes).toHaveLength(1);
        }
      }
    });
  });
});
