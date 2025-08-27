/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';

export const validateAuthMethod = (authMethod: string, env: NodeJS.ProcessEnv = process.env): string | null => {
  if (
    authMethod === AuthType.LOGIN_WITH_GOOGLE ||
    authMethod === AuthType.CLOUD_SHELL
  ) {
    return null;
  }

  if (authMethod === AuthType.LOGIN_WITH_GOOGLE_GCA) {
    if (!env['GOOGLE_CLOUD_PROJECT']) {
      return `[Error] GOOGLE_CLOUD_PROJECT is not set.
Please set it using:
  export GOOGLE_CLOUD_PROJECT=<your-project-id>
and try again.`;
    }
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!env['GEMINI_API_KEY']) {
      return 'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!env['GOOGLE_CLOUD_PROJECT'] &&
      !!env['GOOGLE_CLOUD_LOCATION'];
    const hasGoogleApiKey = !!env['GOOGLE_API_KEY'];
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return `When using Vertex AI, you must specify either:
• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
• GOOGLE_API_KEY environment variable (if using express mode).
Update your environment and try again (no reload needed if using .env)!`;
    }
    return null;
  }

  return 'Invalid auth method selected.';
};
