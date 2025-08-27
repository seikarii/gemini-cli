/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { tokenLimit } from '@google/gemini-cli-core';
import { useSessionStats } from '../contexts/SessionContext.js';
import { computeSessionStats } from '../utils/computeStats.js';

interface TokenPreviewProps {
  messageLength: number;
  model: string;
  estimatedTokens?: number;
}

export const TokenPreview: React.FC<TokenPreviewProps> = ({
  messageLength,
  model,
  estimatedTokens,
}) => {
  const { stats } = useSessionStats();
  const computedStats = computeSessionStats(stats.metrics);

  const maxTokens = tokenLimit(model);
  const currentTokens = computedStats.totalPromptTokens;

  // Rough estimation: ~4 characters per token
  const estimatedMessageTokens =
    estimatedTokens || Math.ceil(messageLength / 4);
  const projectedTotal = currentTokens + estimatedMessageTokens;
  const projectedPercentage = (projectedTotal / maxTokens) * 100;

  const getStatusColor = (percentage: number) => {
    if (percentage >= 95) return theme.status.error;
    if (percentage >= 85) return theme.status.warning;
    return theme.text.secondary;
  };

  const getStatusIcon = (percentage: number) => {
    if (percentage >= 95) return '🚫';
    if (percentage >= 85) return '⚠️';
    return 'ℹ️';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const willExceedLimit = projectedTotal > maxTokens;

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.border.default}
      padding={1}
    >
      <Text bold color={theme.text.accent}>
        🔮 Context Preview
      </Text>

      <Box marginTop={1}>
        <Box width={25}>
          <Text color={theme.text.secondary}>Current Context:</Text>
        </Box>
        <Text>
          {formatNumber(currentTokens)} tokens (
          {((currentTokens / maxTokens) * 100).toFixed(1)}%)
        </Text>
      </Box>

      <Box>
        <Box width={25}>
          <Text color={theme.text.secondary}>This Message:</Text>
        </Box>
        <Text color={theme.text.accent}>
          ~{formatNumber(estimatedMessageTokens)} tokens
        </Text>
      </Box>

      <Box>
        <Box width={25}>
          <Text color={theme.text.secondary}>After Sending:</Text>
        </Box>
        <Text color={getStatusColor(projectedPercentage)}>
          {formatNumber(projectedTotal)} tokens (
          {projectedPercentage.toFixed(1)}%)
        </Text>
      </Box>

      {willExceedLimit && (
        <Box>
          <Text color={theme.status.error}>
            {getStatusIcon(100)} This message will exceed the context limit!
          </Text>
        </Box>
      )}

      {projectedPercentage >= 85 && !willExceedLimit && (
        <Box>
          <Text color={getStatusColor(projectedPercentage)}>
            {getStatusIcon(projectedPercentage)} High context usage after
            sending
          </Text>
        </Box>
      )}

      <Box marginTop={1}>
        <Text color={theme.text.secondary}>
          💡 Tip: Consider clearing context if usage gets too high
        </Text>
      </Box>
    </Box>
  );
};
