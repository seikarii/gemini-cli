/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { z } from 'zod';
import {
  ENV_GOOGLE_CLOUD_PROJECT,
  ENV_GOOGLE_CLOUD_LOCATION,
  ENV_GEMINI_API_KEY,
  ENV_GOOGLE_API_KEY,
  ERROR_GOOGLE_CLOUD_PROJECT_NOT_SET,
  ERROR_GEMINI_API_KEY_NOT_FOUND,
  ERROR_VERTEX_AI_CONFIG_MISSING,
  ERROR_INVALID_AUTH_METHOD,
} from './constants.js';

// Validation schemas
const AuthMethodSchema = z.enum([
  AuthType.LOGIN_WITH_GOOGLE,
  AuthType.CLOUD_SHELL,
  AuthType.LOGIN_WITH_GOOGLE_GCA,
  AuthType.USE_GEMINI,
  AuthType.USE_VERTEX_AI,
]);

const EnvironmentVariablesSchema = z.object({
  [ENV_GOOGLE_CLOUD_PROJECT]: z.string().min(1).optional(),
  [ENV_GOOGLE_CLOUD_LOCATION]: z.string().min(1).optional(),
  [ENV_GEMINI_API_KEY]: z.string().min(1).optional(),
  [ENV_GOOGLE_API_KEY]: z.string().min(1).optional(),
}).passthrough();

const GcpProjectIdSchema = z.string()
  .min(1, 'Project ID cannot be empty')
  .max(30, 'Project ID too long')
  .regex(/^[a-z][a-z0-9-]*[a-z0-9]$/, 'Invalid GCP project ID format');

const GcpLocationSchema = z.string()
  .min(1, 'Location cannot be empty')
  .regex(/^[a-z0-9-]+$/, 'Invalid GCP location format');

const ApiKeySchema = z.string()
  .min(1, 'API key cannot be empty')
  .regex(/^[A-Za-z0-9_-]*$/, 'Invalid API key format');

/**
 * Validates environment variables for a specific authentication method.
 */
function validateEnvironmentForAuthMethod(
  authMethod: AuthType,
  env: Record<string, string | undefined>,
): { success: true } | { success: false; error: string } {
  try {
    // First validate that the auth method is valid
    AuthMethodSchema.parse(authMethod);

    // Validate environment variables structure
    EnvironmentVariablesSchema.parse(env);

    switch (authMethod) {
      case AuthType.LOGIN_WITH_GOOGLE:
      case AuthType.CLOUD_SHELL:
        // These methods don't require additional environment variables
        return { success: true };

      case AuthType.LOGIN_WITH_GOOGLE_GCA:
        if (!env[ENV_GOOGLE_CLOUD_PROJECT]) {
          return { success: false, error: ERROR_GOOGLE_CLOUD_PROJECT_NOT_SET };
        }
        // Validate project ID format
        GcpProjectIdSchema.parse(env[ENV_GOOGLE_CLOUD_PROJECT]);
        return { success: true };

      case AuthType.USE_GEMINI:
        if (!env[ENV_GEMINI_API_KEY]) {
          return { success: false, error: ERROR_GEMINI_API_KEY_NOT_FOUND };
        }
        // Validate API key format
        ApiKeySchema.parse(env[ENV_GEMINI_API_KEY]);
        return { success: true };

      case AuthType.USE_VERTEX_AI: {
        const hasVertexConfig = env[ENV_GOOGLE_CLOUD_PROJECT] && env[ENV_GOOGLE_CLOUD_LOCATION];
        const hasApiKey = env[ENV_GOOGLE_API_KEY];

        if (!hasVertexConfig && !hasApiKey) {
          return { success: false, error: ERROR_VERTEX_AI_CONFIG_MISSING };
        }

        // Validate formats if present
        if (hasVertexConfig) {
          GcpProjectIdSchema.parse(env[ENV_GOOGLE_CLOUD_PROJECT]);
          GcpLocationSchema.parse(env[ENV_GOOGLE_CLOUD_LOCATION]);
        }
        if (hasApiKey) {
          ApiKeySchema.parse(env[ENV_GOOGLE_API_KEY]);
        }
        return { success: true };
      }

      default:
        return { success: false, error: ERROR_INVALID_AUTH_METHOD };
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      // For invalid auth methods, return the original error message
      if (authMethod && !Object.values(AuthType).includes(authMethod as AuthType)) {
        return { success: false, error: ERROR_INVALID_AUTH_METHOD };
      }
      // For other validation errors, return a generic validation error
      return {
        success: false,
        error: `Validation error: ${error.errors.map(e => e.message).join(', ')}`,
      };
    }
    return { success: false, error: ERROR_INVALID_AUTH_METHOD };
  }
}

export const validateAuthMethod = (authMethod: string, env: NodeJS.ProcessEnv = process.env): string | null => {
  const result = validateEnvironmentForAuthMethod(authMethod as AuthType, env);
  return result.success ? null : (result as { success: false; error: string }).error;
};
