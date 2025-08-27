/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { UserTierId } from '@google/gemini-cli-core';
/**
 * Get human-readable license display text based on auth type and user tier.
 * @param selectedAuthType - The authentication type selected by the user
 * @param userTier - Optional user tier information from the server
 * @returns Human-readable license information
 */
export declare function getLicenseDisplay(selectedAuthType: string, userTier?: UserTierId): string;
