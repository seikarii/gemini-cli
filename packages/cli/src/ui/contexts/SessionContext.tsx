/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useState,
  useMemo,
  useEffect,
  useRef,
} from 'react';

import {
  uiTelemetryService,
  SessionMetrics,
  ModelMetrics,
  sessionId,
  ChatRecordingService,
} from '@google/gemini-cli-core';

// --- Interface Definitions ---

export type { SessionMetrics, ModelMetrics };

export interface SessionStatsState {
  sessionId: string;
  sessionStartTime: Date;
  metrics: SessionMetrics;
  lastPromptTokenCount: number;
  promptCount: number;
  chatRecordingService?: ChatRecordingService;
}

export interface ComputedSessionStats {
  totalApiTime: number;
  totalToolTime: number;
  agentActiveTime: number;
  apiTimePercent: number;
  toolTimePercent: number;
  cacheEfficiency: number;
  totalDecisions: number;
  successRate: number;
  agreementRate: number;
  totalCachedTokens: number;
  totalPromptTokens: number;
  totalLinesAdded: number;
  totalLinesRemoved: number;
}

// Defines the final "value" of our context, including the state
// and the functions to update it.
interface SessionStatsContextValue {
  stats: SessionStatsState;
  startNewPrompt: () => void;
  getPromptCount: () => number;
  getCurrentTokenCount: () => Promise<number>;
  getLastMessageTokenCount: () => Promise<number>;
  setChatRecordingService: (service: ChatRecordingService) => void;
}

// --- Context Definition ---

const SessionStatsContext = createContext<SessionStatsContextValue | undefined>(
  undefined,
);

// --- Provider Component ---

export const SessionStatsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [stats, setStats] = useState<SessionStatsState>({
    sessionId,
    sessionStartTime: new Date(),
    metrics: uiTelemetryService.getMetrics(),
    lastPromptTokenCount: 0,
    promptCount: 0,
  });

  // Use ref to avoid dependency issues in useEffect
  const chatRecordingServiceRef = useRef<ChatRecordingService | null>(null);

  useEffect(() => {
    const handleUpdate = async ({
      metrics,
      lastPromptTokenCount,
    }: {
      metrics: SessionMetrics;
      lastPromptTokenCount: number;
    }) => {
      let tokenCount = lastPromptTokenCount;
      
      // If ChatRecordingService is available, use it for token count
      const chatRecordingService = chatRecordingServiceRef.current;
      if (chatRecordingService) {
        try {
          tokenCount = await chatRecordingService.getLastMessageTokenCount();
        } catch (error) {
          console.warn('Failed to get token count from ChatRecordingService, using telemetry fallback:', error);
          tokenCount = lastPromptTokenCount;
        }
      }

      setStats((prevState) => ({
        ...prevState,
        metrics,
        lastPromptTokenCount: tokenCount,
      }));
    };

    uiTelemetryService.on('update', handleUpdate);
    // Set initial state
    handleUpdate({
      metrics: uiTelemetryService.getMetrics(),
      lastPromptTokenCount: uiTelemetryService.getLastPromptTokenCount(),
    });

    return () => {
      uiTelemetryService.off('update', handleUpdate);
    };
  }, []); // Remove stats.chatRecordingService from dependencies

  const startNewPrompt = useCallback(() => {
    setStats((prevState) => ({
      ...prevState,
      promptCount: prevState.promptCount + 1,
    }));
  }, []);

  const getPromptCount = useCallback(
    () => stats.promptCount,
    [stats.promptCount],
  );

  const getCurrentTokenCount = useCallback(async (): Promise<number> => {
    const chatRecordingService = chatRecordingServiceRef.current;
    if (chatRecordingService) {
      try {
        return await chatRecordingService.getCurrentTokenCount();
      } catch (error) {
        console.warn('Failed to get token count from ChatRecordingService:', error);
      }
    }
    // Fallback to uiTelemetryService
    return uiTelemetryService.getLastPromptTokenCount();
  }, []);

  const getLastMessageTokenCount = useCallback(async (): Promise<number> => {
    const chatRecordingService = chatRecordingServiceRef.current;
    if (chatRecordingService) {
      try {
        return await chatRecordingService.getLastMessageTokenCount();
      } catch (error) {
        console.warn('Failed to get last message token count from ChatRecordingService:', error);
      }
    }
    // Fallback to uiTelemetryService
    return uiTelemetryService.getLastPromptTokenCount();
  }, []);

  const setChatRecordingService = useCallback((service: ChatRecordingService) => {
    chatRecordingServiceRef.current = service;
    setStats(prev => ({
      ...prev,
      chatRecordingService: service,
    }));
  }, []);

  const value = useMemo(
    () => ({
      stats,
      startNewPrompt,
      getPromptCount,
      getCurrentTokenCount,
      getLastMessageTokenCount,
      setChatRecordingService,
    }),
    [stats, startNewPrompt, getPromptCount, getCurrentTokenCount, getLastMessageTokenCount, setChatRecordingService],
  );

  return (
    <SessionStatsContext.Provider value={value}>
      {children}
    </SessionStatsContext.Provider>
  );
};

// --- Consumer Hook ---

export const useSessionStats = () => {
  const context = useContext(SessionStatsContext);
  if (context === undefined) {
    throw new Error(
      'useSessionStats must be used within a SessionStatsProvider',
    );
  }
  return context;
};
