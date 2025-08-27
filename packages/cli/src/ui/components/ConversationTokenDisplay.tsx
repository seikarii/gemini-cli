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
  const [isLoadingTokens, setIsLoadingTokens] = useState(false);

  // Refs for debouncing and mounted state
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);

  // Debounced token loading function
  const loadTokens = useCallback(async () => {
    if (!getCurrentTokenCount || !getLastMessageTokenCount) return;

    // Clear any existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Debounce the token loading to prevent excessive API calls
    debounceTimerRef.current = setTimeout(async () => {
      if (!isMountedRef.current) return;

      try {
        setIsLoadingTokens(true);

        // Load both token counts in parallel
        const [currentTokens, lastMessageTokens] = await Promise.all([
          getCurrentTokenCount(),
          getLastMessageTokenCount(),
        ]);

        if (isMountedRef.current) {
          setCurrentTokenCount(currentTokens);
          setLastMessageTokenCount(lastMessageTokens);
        }
      } catch (error) {
        // Silently handle errors to avoid disrupting the UI
        console.warn('Failed to load token counts:', error);
      } finally {
        if (isMountedRef.current) {
          setIsLoadingTokens(false);
        }
      }
    }, 300); // 300ms debounce delay
  }, [getCurrentTokenCount, getLastMessageTokenCount]);

  // Load tokens on mount
  useEffect(() => {
    loadTokens();
  }, [loadTokens]);

  // Reload tokens when prompt count changes (new messages)
  useEffect(() => {
    loadTokens();
  }, [promptCount, loadTokens]);

  // Cleanup on unmount
  useEffect(() => () => {
      isMountedRef.current = false;
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    }, []);

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
