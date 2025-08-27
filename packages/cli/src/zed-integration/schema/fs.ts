/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';

export type WriteTextFileRequest = z.infer<typeof writeTextFileRequestSchema>;

export type ReadTextFileRequest = z.infer<typeof readTextFileRequestSchema>;

export type WriteTextFileResponse = z.infer<typeof writeTextFileResponseSchema>;

export type ReadTextFileResponse = z.infer<typeof readTextFileResponseSchema>;

export type FileSystemCapability = z.infer<typeof fileSystemCapabilitySchema>;

export const writeTextFileRequestSchema = z.object({
  content: z.string(),
  path: z.string(),
  sessionId: z.string(),
});

export const readTextFileRequestSchema = z.object({
  limit: z.number().optional().nullable(),
  line: z.number().optional().nullable(),
  path: z.string(),
  sessionId: z.string(),
});

export const writeTextFileResponseSchema = z.null();

export const readTextFileResponseSchema = z.object({
  content: z.string(),
});

export const fileSystemCapabilitySchema = z.object({
  readTextFile: z.boolean(),
  writeTextFile: z.boolean(),
});
