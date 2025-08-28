/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { tokenLimit } from '@google/gemini-cli-core';
import { useSessionStats } from '../contexts/SessionContext.js';
import { computeSessionStats } from '../utils/computeStats.js';

interface ConversationTokenDisplayProps {
  model: string;
  compact?: boolean;
}

export const ConversationTokenDisplay: React.FC<
  ConversationTokenDisplayProps
> = ({ model, compact = false }) => {
  const { stats, getCurrentTokenCount, getLastMessageTokenCount } =
    useSessionStats();
  const computedStats = computeSessionStats(stats.metrics);
  const { totalPromptTokens } = computedStats;
  const { lastPromptTokenCount, promptCount } = stats;

  // State for async token data
  const [currentTokenCount, setCurrentTokenCount] = useState<number | null>(
    null,
  );
  const [lastMessageTokenCount, setLastMessageTokenCount] = useState<
    number | null
  >(null);

  // Refs for debouncing and mounted state
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);
  const lastUpdateRef = useRef<number>(0);

  // Debounced token loading function with improved error handling and caching
  const loadTokens = useCallback(async () => {
    if (!getCurrentTokenCount || !getLastMessageTokenCount) return;

    // Avoid excessive updates (max once per 500ms)
    const now = Date.now();
    if (now - lastUpdateRef.current < 500) return;

    // Clear any existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Debounce the token loading to prevent excessive API calls
    debounceTimerRef.current = setTimeout(async () => {
      if (!isMountedRef.current) return;

      try {
        // Load both token counts in parallel with timeout
        const tokenPromises = [
          Promise.race([
            getCurrentTokenCount(),
            new Promise<number>((_, reject) => 
              setTimeout(() => reject(new Error('Timeout')), 2000)
            )
          ]),
          Promise.race([
            getLastMessageTokenCount(),
            new Promise<number>((_, reject) => 
              setTimeout(() => reject(new Error('Timeout')), 2000)
            )
          ])
        ];

        const [currentTokens, lastMessageTokens] = await Promise.all(tokenPromises);

        if (isMountedRef.current) {
          setCurrentTokenCount(currentTokens);
          setLastMessageTokenCount(lastMessageTokens);
          lastUpdateRef.current = Date.now();
        }
      } catch (_error) {
        // Fallback to sync data on error without spamming console
        if (isMountedRef.current) {
          setCurrentTokenCount(totalPromptTokens);
          setLastMessageTokenCount(lastPromptTokenCount);
        }
      }
    }, 200); // Reduced debounce delay for better responsiveness
  }, [getCurrentTokenCount, getLastMessageTokenCount, totalPromptTokens, lastPromptTokenCount]);

  // Load tokens on mount
  useEffect(() => {
    loadTokens();
  }, [loadTokens]);

  // Reload tokens when prompt count changes (new messages)
  useEffect(() => {
    loadTokens();
  }, [promptCount, loadTokens]);

  // Cleanup on unmount
  useEffect(
    () => () => {
      isMountedRef.current = false;
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    },
    [],
  );

  const maxTokens = tokenLimit(model);
  const usagePercentage = (totalPromptTokens / maxTokens) * 100;

  // Use async token data when available, fallback to sync data
  const displayTokenCount = currentTokenCount ?? totalPromptTokens;
  const displayLastMessageTokens =
    lastMessageTokenCount ?? lastPromptTokenCount;

  // Determine color based on usage level
  const getStatusColor = (percentage: number) => {
    if (percentage >= 90) return theme.status.error; // Danger zone
    if (percentage >= 75) return theme.status.warning; // Warning zone
    return theme.text.secondary; // Normal zone
  };

  const getStatusIcon = (percentage: number) => {
    if (percentage >= 90) return '🔴'; // Red circle for danger
    if (percentage >= 75) return '🟡'; // Yellow circle for warning
    return '🟢'; // Green circle for normal
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  if (compact) {
    return (
      <Box>
        <Text color={getStatusColor(usagePercentage)}>
          {getStatusIcon(usagePercentage)} {formatNumber(displayTokenCount)}/
          {formatNumber(maxTokens)}
        </Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {/* Main token counter */}
      <Box>
        <Text color={theme.text.secondary}>Context: </Text>
        <Text color={getStatusColor(usagePercentage)}>
          {getStatusIcon(usagePercentage)} {formatNumber(displayTokenCount)}/
          {formatNumber(maxTokens)}
        </Text>
        <Text color={theme.text.secondary}>
          {' '}
          ({usagePercentage.toFixed(1)}%)
        </Text>
      </Box>

      {/* Session details */}
      <Box>
        <Text color={theme.text.secondary}>Session: {promptCount} prompts</Text>
        {displayLastMessageTokens > 0 && (
          <Text color={theme.text.secondary}>
            {' '}
            • Last: {formatNumber(displayLastMessageTokens)} tokens
          </Text>
        )}
      </Box>

      {/* Warning for high usage */}
      {usagePercentage >= 75 && (
        <Box>
          <Text color={getStatusColor(usagePercentage)}>
            ⚠️ High context usage - includes conversation history & loaded files
          </Text>
        </Box>
      )}
    </Box>
  );
};
