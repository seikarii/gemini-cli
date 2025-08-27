/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { useSessionStats } from '../contexts/SessionContext.js';
import { computeSessionStats } from '../utils/computeStats.js';
import { formatDuration } from '../utils/formatters.js';

interface RealtimeStatsPanelProps {
  compact?: boolean;
  showCacheStats?: boolean;
}

export const RealtimeStatsPanel: React.FC<RealtimeStatsPanelProps> = ({
  compact = false,
  showCacheStats = true,
}) => {
  const { stats } = useSessionStats();
  const computedStats = computeSessionStats(stats.metrics);
  const sessionDuration = Date.now() - stats.sessionStartTime.getTime();

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const getTokensPerMinute = () => {
    const minutes = sessionDuration / (1000 * 60);
    if (minutes < 1) return 0;
    return Math.round(computedStats.totalPromptTokens / minutes);
  };

  if (compact) {
    return (
      <Box>
        <Text color={theme.text.secondary}>
          📊 {formatDuration(sessionDuration)} • {stats.promptCount}p •{' '}
          {formatNumber(computedStats.totalPromptTokens)}ctx
          {getTokensPerMinute() > 0 && ` • ${getTokensPerMinute()}ctx/min`}
        </Text>
      </Box>
    );
  }

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.border.default}
      padding={1}
    >
      <Text bold color={theme.text.accent}>
        📊 Session Stats
      </Text>

      <Box marginTop={1}>
        <Box width={20}>
          <Text color={theme.text.secondary}>Duration:</Text>
        </Box>
        <Text>{formatDuration(sessionDuration)}</Text>
      </Box>

      <Box>
        <Box width={20}>
          <Text color={theme.text.secondary}>Prompts:</Text>
        </Box>
        <Text>{stats.promptCount}</Text>
      </Box>

      <Box>
        <Box width={20}>
          <Text color={theme.text.secondary}>Total Context:</Text>
        </Box>
        <Text>{formatNumber(computedStats.totalPromptTokens)}</Text>
      </Box>

      {getTokensPerMinute() > 0 && (
        <Box>
          <Box width={20}>
            <Text color={theme.text.secondary}>Context/Min:</Text>
          </Box>
          <Text>{getTokensPerMinute()}</Text>
        </Box>
      )}

      {showCacheStats && computedStats.cacheEfficiency > 0 && (
        <Box>
          <Box width={20}>
            <Text color={theme.text.secondary}>Cache Efficiency:</Text>
          </Box>
          <Text
            color={
              computedStats.cacheEfficiency > 50
                ? theme.status.success
                : theme.text.primary
            }
          >
            {computedStats.cacheEfficiency.toFixed(1)}%
          </Text>
        </Box>
      )}

      {computedStats.totalDecisions > 0 && (
        <Box>
          <Box width={20}>
            <Text color={theme.text.secondary}>Decisions:</Text>
          </Box>
          <Text>{computedStats.totalDecisions}</Text>
        </Box>
      )}
    </Box>
  );
};
