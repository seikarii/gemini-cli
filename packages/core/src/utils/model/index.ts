/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Re-export model-related utilities
// Note: Both generateContentResponseUtilities and partUtils export similar functions
// This creates a unified interface while maintaining backward compatibility

// Export from generateContentResponseUtilities
export {
  getResponseText,
  getFunctionCalls,
  getFunctionCallsAsJson,
  getFunctionCallsFromParts,
  getFunctionCallsFromPartsAsJson,
  getResponseTextFromParts,
  getStructuredResponse,
  getStructuredResponseFromParts,
} from '../generateContentResponseUtilities.js';

// Export from partUtils
export {
  partToString,
  getResponseText as getResponseTextFromPartsUtil,
  getFunctionCalls as getFunctionCallsFromPartsUtil,
} from '../partUtils.js';
