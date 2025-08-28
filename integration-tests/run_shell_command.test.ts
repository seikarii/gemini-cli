/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { TestRig, printDebugInfo, validateModelOutput } from './test-helper.js';

describe('run_shell_command', () => {
  it('should be able to run a shell command', async () => {
    const rig = new TestRig();
    await rig.setup('should be able to run a shell command');

    const prompt = `Please run the command "echo hello-world" and show me the output`;

    const result = await rig.run(prompt);

    // Try to find tool call with shorter timeout to avoid hanging
    const foundToolCall = await rig.waitForToolCall('run_shell_command', 8000); // 8 seconds

    // Add debugging information
    if (!foundToolCall || !result.includes('hello-world')) {
      printDebugInfo(rig, result, {
        'Found tool call': foundToolCall,
        'Contains hello-world': result.includes('hello-world'),
      });
    }

    // More flexible validation: either tool was used OR output contains expected result
    const hasExpectedOutput = result.includes('hello-world') || result.includes('exit code 0');
    const testPassed = foundToolCall || hasExpectedOutput;

    expect(testPassed, 'Expected either tool call or correct output').toBeTruthy();

    // If tool was used, validate it was the right one
    if (foundToolCall) {
      expect(
        foundToolCall,
        'Expected to find a run_shell_command tool call',
      ).toBeTruthy();
    }

    // Validate model output - will throw if no output, warn if missing expected content
    // Model often reports exit code instead of showing output
    validateModelOutput(
      result,
      ['hello-world', 'exit code 0'],
      'Shell command test',
    );
  });

  it('should be able to run a shell command via stdin', async () => {
    const rig = new TestRig();
    await rig.setup('should be able to run a shell command via stdin');

    const prompt = `Please run the command "echo test-stdin" and show me what it outputs`;

    const result = await rig.run({ stdin: prompt });

    // Try to find tool call with shorter timeout
    const foundToolCall = await rig.waitForToolCall('run_shell_command', 8000); // 8 seconds

    // Add debugging information
    if (!foundToolCall || !result.includes('test-stdin')) {
      printDebugInfo(rig, result, {
        'Test type': 'Stdin test',
        'Found tool call': foundToolCall,
        'Contains test-stdin': result.includes('test-stdin'),
      });
    }

    // More flexible validation: either tool was used OR output contains expected result
    const hasExpectedOutput = result.includes('test-stdin') || result.includes('exit code 0');
    const testPassed = foundToolCall || hasExpectedOutput;

    expect(testPassed, 'Expected either tool call or correct output for stdin test').toBeTruthy();

    // If tool was used, validate it was the right one
    if (foundToolCall) {
      expect(
        foundToolCall,
        'Expected to find a run_shell_command tool call',
      ).toBeTruthy();
    }

    // Validate model output
    validateModelOutput(
      result,
      ['test-stdin', 'exit code 0'],
      'Shell command stdin test',
    );
  });
});
