/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Box, Text } from 'ink';
import { Colors } from '../colors.js';
import { tokenLimit } from '@google/gemini-cli-core';
import { useSessionStats } from '../contexts/SessionContext.js';
import { computeSessionStats } from '../utils/computeStats.js';

interface ContextUsageDisplayProps {
  promptTokenCount: number;
  model: string;
  showDetails?: boolean;
}

export const ContextUsageDisplay = ({
  model,
  showDetails = false,
}: ContextUsageDisplayProps) => {
  const { stats } = useSessionStats();
  const computedStats = computeSessionStats(stats.metrics);

  const maxTokens = tokenLimit(model);
  const percentage = (computedStats.totalPromptTokens / maxTokens) * 100;
  const remainingPercentage = 100 - percentage;

  const getStatusColor = (pct: number) => {
    if (pct >= 90) return Colors.AccentRed;
    if (pct >= 75) return Colors.AccentYellow;
    return Colors.Gray;
  };

  const getStatusIndicator = (pct: number) => {
    if (pct >= 90) return '🔴';
    if (pct >= 75) return '🟡';
    return '🟢';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  if (showDetails) {
    return (
      <Box flexDirection="column">
        <Box>
          <Text color={getStatusColor(percentage)}>
            {getStatusIndicator(percentage)} {remainingPercentage.toFixed(0)}%
            context left
          </Text>
        </Box>
        <Box>
          <Text color={Colors.Gray}>
            Used: {formatNumber(computedStats.totalPromptTokens)} | Remaining:{' '}
            {formatNumber(
              Math.max(0, maxTokens - computedStats.totalPromptTokens),
            )}
          </Text>
        </Box>
        {percentage >= 75 && (
          <Box>
            <Text color={getStatusColor(percentage)}>
              ⚠️ High usage - {percentage.toFixed(1)}% of context used
            </Text>
          </Box>
        )}
      </Box>
    );
  }

  return (
    <Text color={getStatusColor(percentage)}>
      ({remainingPercentage.toFixed(0)}% context left)
    </Text>
  );
};
