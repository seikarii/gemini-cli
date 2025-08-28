/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Config } from '../config/config.js';
import { LoopDetectionService } from './loopDetectionService.js';
import {
  GeminiEventType,
  ServerGeminiToolCallRequestEvent,
} from '../core/turn.js';

vi.mock('../telemetry/loggers.js', () => ({
  logLoopDetected: vi.fn(),
}));

describe('LoopDetectionService Minimal Test', () => {
  let mockConfig: Config;
  let service: LoopDetectionService;

  beforeEach(() => {
    mockConfig = {
      getTelemetryEnabled: () => true,
      getDebugMode: () => false,
    } as unknown as Config;
    service = new LoopDetectionService(mockConfig);
    vi.clearAllMocks();
  });

  const createToolCallRequestEvent = (
    name: string,
    args: Record<string, unknown>,
  ): ServerGeminiToolCallRequestEvent => ({
    type: GeminiEventType.ToolCallRequest,
    value: {
      name,
      args,
      callId: 'test-id',
      isClientInitiated: false,
      prompt_id: 'test-prompt-id',
    },
  });

  it('should create service without hanging', () => {
    console.log('Starting minimal test...');
    expect(service).toBeDefined();
    console.log('Test completed successfully');
  });

  it('should call addAndCheck without hanging', () => {
    console.log('Starting addAndCheck test...');
    const event = createToolCallRequestEvent('testTool', { param: 'value' });
    console.log('Created event');
    const result = service.addAndCheck(event);
    console.log('Called addAndCheck, result:', result);
    expect(result).toBe(false);
    console.log('addAndCheck test completed successfully');
  });
});
