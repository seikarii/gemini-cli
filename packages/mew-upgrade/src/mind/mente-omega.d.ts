import type { ContentGenerator } from '@google/gemini-cli-core';
import type { Persistable } from '../persistence/persistence-service.js';
interface MemoryNode {
    id: string;
    embedding?: number[];
    content: unknown;
    metadata: Record<string, unknown>;
    timestamp: number;
    importance: number;
    valence: number;
    arousal: number;
    kind: string;
    associatedNodes: string[];
    accessCount: number;
    lastAccessed: number;
    tags: string[];
    confidence: number;
}
interface CognitiveState {
    attention: number;
    focus: string[];
    emotionalState: {
        valence: number;
        arousal: number;
        dominance: number;
    };
    workingMemory: MemoryNode[];
    currentGoals: string[];
    inhibition: number;
    curiosity: number;
}
interface ProcessingContext {
    userRequest: string;
    sessionId: string;
    timestamp: number;
    priority: number;
    expectedResponseTime: number;
    contextWindow: MemoryNode[];
    emotionalContext: {
        userMood: string;
        systemMood: string;
        rapport: number;
    };
}
interface MemoryQueryOptions {
    limit?: number;
    threshold?: number;
    includeMetadata?: boolean;
    temporalWindow?: {
        start: number;
        end: number;
    };
    importanceFilter?: {
        min: number;
        max: number;
    };
    kindFilter?: string[];
    tagFilter?: string[];
}
interface InsightMetrics {
    coherence: number;
    novelty: number;
    relevance: number;
    confidence: number;
    complexity: number;
    emotionalResonance: number;
}
declare class EntityMemory {
    private nodes;
    private associationGraph;
    private tagIndex;
    private temporalIndex;
    private importanceThreshold;
    constructor(importanceThreshold?: number);
    generateId(): string;
    ingest(data: unknown, valence?: number, arousal?: number, kind?: string, importance?: number, tags?: string[], associations?: string[]): Promise<string>;
    recall(query: string, options?: MemoryQueryOptions): MemoryNode[];
    private calculateRelevanceScore;
    private updateAssociations;
    private updateTagIndex;
    private updateTemporalIndex;
    private pruneMemories;
    private cleanupIndices;
    getAssociatedNodes(nodeId: string, depth?: number): MemoryNode[];
    getNodesByTag(tag: string): MemoryNode[];
    getAllNodes(): MemoryNode[];
    exportState(): object;
    importState(state: any): void;
}
export declare class MenteOmega implements Persistable {
    memory: EntityMemory;
    private menteAlfa;
    private cognitiveState;
    private processingHistory;
    private maxHistorySize;
    constructor(menteAlfa?: ContentGenerator, memoryThreshold?: number);
    process(userRequest: string, context?: Partial<ProcessingContext>): Promise<string>;
    private analyzeRequest;
    private updateCognitiveState;
    private enhanceWithAssociations;
    private generateInsights;
    private calculateSimilarity;
    private prepareContextForAlfa;
    private calculateResponseLength;
    private storeInteraction;
    private postProcessCognitiveUpdate;
    private addToProcessingHistory;
    getAgentHistorySummary(): string;
    processIncomingData(incomingData: unknown, context: unknown, currentMission: string, agentHistorySummary: string): Promise<void>;
    private calculateDataImportance;
    private extractEmotionalContext;
    getCognitiveState(): CognitiveState;
    getInsightAnalysis(query: string): InsightMetrics;
    getMemoryStats(): object;
    exportState(): object;
    importState(state: unknown): void;
}
export {};
