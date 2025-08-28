/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { validateAuthMethod } from './auth.js';

describe('validateAuthMethod', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = {};
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should return null for LOGIN_WITH_GOOGLE', () => {
    expect(validateAuthMethod(AuthType.LOGIN_WITH_GOOGLE)).toBeNull();
  });

  it('should return null for CLOUD_SHELL', () => {
    expect(validateAuthMethod(AuthType.CLOUD_SHELL)).toBeNull();
  });

  describe('USE_GEMINI', () => {
    it('should return null if GEMINI_API_KEY is set', () => {
      const env: NodeJS.ProcessEnv = {};
      env['GEMINI_API_KEY'] = 'test-key';
      expect(validateAuthMethod(AuthType.USE_GEMINI, env)).toBeNull();
    });

    it('should return an error message if GEMINI_API_KEY is not set', () => {
      const env = {} as NodeJS.ProcessEnv;
      expect(validateAuthMethod(AuthType.USE_GEMINI, env)).toBe(
        'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!',
      );
    });
  });

  describe('USE_VERTEX_AI', () => {
    it('should return null if GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set', () => {
      const env: NodeJS.ProcessEnv = {};
      env['GOOGLE_CLOUD_PROJECT'] = 'test-project';
      env['GOOGLE_CLOUD_LOCATION'] = 'test-location';
      expect(validateAuthMethod(AuthType.USE_VERTEX_AI, env)).toBeNull();
    });

    it('should return null if GOOGLE_API_KEY is set', () => {
      const env: NodeJS.ProcessEnv = {};
      env['GOOGLE_API_KEY'] = 'test-api-key';
      expect(validateAuthMethod(AuthType.USE_VERTEX_AI, env)).toBeNull();
    });

    it('should return an error message if no required environment variables are set', () => {
      const env = {} as NodeJS.ProcessEnv;
      expect(validateAuthMethod(AuthType.USE_VERTEX_AI, env)).toBe(
        `When using Vertex AI, you must specify either:
• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
• GOOGLE_API_KEY environment variable (if using express mode).
Update your environment and try again (no reload needed if using .env)!`,
      );
    });
  });

  it('should return an error message for an invalid auth method', () => {
    expect(validateAuthMethod('invalid-method')).toBe(
      'Invalid auth method selected.',
    );
  });
});
