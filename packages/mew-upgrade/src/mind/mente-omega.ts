import type { ContentGenerator } from '@google/gemini-cli-core';
import type { Content } from '@google/genai';
import type { Persistable } from '../persistence/persistence-service.js';

// Advanced interfaces for comprehensive memory and cognitive operations
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
	temporalWindow?: { start: number; end: number };
	importanceFilter?: { min: number; max: number };
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

class EntityMemory {
	private nodes: Map<string, MemoryNode> = new Map();
	private associationGraph: Map<string, Set<string>> = new Map();
	private tagIndex: Map<string, Set<string>> = new Map();
	private temporalIndex: Map<number, Set<string>> = new Map();
	private importanceThreshold: number = 0.5;

	constructor(importanceThreshold: number = 0.5) {
		this.importanceThreshold = importanceThreshold;
	}

	generateId(): string {
		return `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
	}

	async ingest(
		data: unknown,
		valence: number = 0.0,
		arousal: number = 0.0,
		kind: string = 'general',
		importance: number = 0.5,
		tags: string[] = [],
		associations: string[] = []
	): Promise<string> {
		const id = this.generateId();
		const timestamp = Date.now();
		
		const node: MemoryNode = {
			id,
			content: data,
			metadata: {
				source: 'ingest',
				processingTime: timestamp
			},
			timestamp,
			importance: Math.max(0, Math.min(1, importance)),
			valence: Math.max(-1, Math.min(1, valence)),
			arousal: Math.max(0, Math.min(1, arousal)),
			kind,
			associatedNodes: associations,
			accessCount: 0,
			lastAccessed: timestamp,
			tags,
			confidence: 0.8
		};

		// Store the node
		this.nodes.set(id, node);

		// Update associations
		this.updateAssociations(id, associations);

		// Update indices
		this.updateTagIndex(id, tags);
		this.updateTemporalIndex(id, timestamp);

		// Prune old memories if needed
		await this.pruneMemories();

		return id;
	}

	recall(query: string, options: MemoryQueryOptions = {}): MemoryNode[] {
		const {
			limit = 10,
			threshold = 0.3,
			includeMetadata = true,
			temporalWindow,
			importanceFilter,
			kindFilter,
			tagFilter
		} = options;

		let candidates = Array.from(this.nodes.values());

		// Apply filters
		if (temporalWindow) {
			candidates = candidates.filter(node => 
				node.timestamp >= temporalWindow.start && 
				node.timestamp <= temporalWindow.end
			);
		}

		if (importanceFilter) {
			candidates = candidates.filter(node => 
				node.importance >= importanceFilter.min && 
				node.importance <= importanceFilter.max
			);
		}

		if (kindFilter && kindFilter.length > 0) {
			candidates = candidates.filter(node => kindFilter.includes(node.kind));
		}

		if (tagFilter && tagFilter.length > 0) {
			candidates = candidates.filter(node => 
				node.tags.some(tag => tagFilter.includes(tag))
			);
		}

		// Simple similarity scoring (in a real implementation, this would use embeddings)
		const scored = candidates.map(node => ({
			node,
			score: this.calculateRelevanceScore(query, node)
		}));

		// Sort by score and filter by threshold
		const filtered = scored
			.filter(item => item.score >= threshold)
			.sort((a, b) => b.score - a.score)
			.slice(0, limit);

		// Update access patterns
		filtered.forEach(item => {
			item.node.accessCount++;
			item.node.lastAccessed = Date.now();
		});

		return filtered.map(item => includeMetadata ? item.node : {
			...item.node,
			metadata: {}
		});
	}

	private calculateRelevanceScore(query: string, node: MemoryNode): number {
		const queryLower = query.toLowerCase();
		const contentStr = String(node.content).toLowerCase();
		
		// Simple text similarity
		let textScore = 0;
		const queryWords = queryLower.split(/\s+/);
		const contentWords = contentStr.split(/\s+/);
		
		queryWords.forEach(qWord => {
			if (contentWords.some(cWord => cWord.includes(qWord) || qWord.includes(cWord))) {
				textScore += 0.2;
			}
		});

		// Boost by importance and recency
		const importanceBoost = node.importance * 0.3;
		const recencyBoost = Math.max(0, 1 - (Date.now() - node.timestamp) / (1000 * 60 * 60 * 24 * 7)) * 0.1;
		const accessBoost = Math.min(0.2, node.accessCount * 0.02);

		return Math.min(1, textScore + importanceBoost + recencyBoost + accessBoost);
	}

	private updateAssociations(nodeId: string, associations: string[]): void {
		if (!this.associationGraph.has(nodeId)) {
			this.associationGraph.set(nodeId, new Set());
		}
		
		const nodeAssociations = this.associationGraph.get(nodeId)!;
		associations.forEach(assocId => {
			nodeAssociations.add(assocId);
			
			// Create bidirectional associations
			if (!this.associationGraph.has(assocId)) {
				this.associationGraph.set(assocId, new Set());
			}
			this.associationGraph.get(assocId)!.add(nodeId);
		});
	}

	private updateTagIndex(nodeId: string, tags: string[]): void {
		tags.forEach(tag => {
			if (!this.tagIndex.has(tag)) {
				this.tagIndex.set(tag, new Set());
			}
			this.tagIndex.get(tag)!.add(nodeId);
		});
	}

	private updateTemporalIndex(nodeId: string, timestamp: number): void {
		const dayKey = Math.floor(timestamp / (1000 * 60 * 60 * 24));
		if (!this.temporalIndex.has(dayKey)) {
			this.temporalIndex.set(dayKey, new Set());
		}
		this.temporalIndex.get(dayKey)!.add(nodeId);
	}

	private async pruneMemories(): Promise<void> {
		const maxNodes = 10000;
		if (this.nodes.size <= maxNodes) return;

		// Remove least important and least accessed nodes
		const sortedNodes = Array.from(this.nodes.values())
			.sort((a, b) => {
				const scoreA = a.importance + (a.accessCount * 0.1);
				const scoreB = b.importance + (b.accessCount * 0.1);
				return scoreA - scoreB;
			});

		const toRemove = sortedNodes.slice(0, this.nodes.size - maxNodes);
		toRemove.forEach(node => {
			this.nodes.delete(node.id);
			this.cleanupIndices(node.id);
		});
	}

	private cleanupIndices(nodeId: string): void {
		// Clean association graph
		this.associationGraph.delete(nodeId);
		this.associationGraph.forEach(associations => {
			associations.delete(nodeId);
		});

		// Clean tag index
		this.tagIndex.forEach(nodeSet => {
			nodeSet.delete(nodeId);
		});

		// Clean temporal index
		this.temporalIndex.forEach(nodeSet => {
			nodeSet.delete(nodeId);
		});
	}

	getAssociatedNodes(nodeId: string, depth: number = 1): MemoryNode[] {
		const visited = new Set<string>();
		const result: MemoryNode[] = [];

		const traverse = (currentId: string, currentDepth: number) => {
			if (currentDepth > depth || visited.has(currentId)) return;
			
			visited.add(currentId);
			const node = this.nodes.get(currentId);
			if (node) result.push(node);

			const associations = this.associationGraph.get(currentId);
			if (associations) {
				associations.forEach(assocId => {
					traverse(assocId, currentDepth + 1);
				});
			}
		};

		traverse(nodeId, 0);
		return result.slice(1); // Exclude the starting node
	}

	getNodesByTag(tag: string): MemoryNode[] {
		const nodeIds = this.tagIndex.get(tag) || new Set();
		return Array.from(nodeIds).map(id => this.nodes.get(id)!).filter(Boolean);
	}

	getAllNodes(): MemoryNode[] {
		return Array.from(this.nodes.values());
	}

	exportState(): object {
		return {
			nodes: Array.from(this.nodes.entries()),
			associationGraph: Array.from(this.associationGraph.entries()).map(([k, v]) => [k, Array.from(v)]),
			tagIndex: Array.from(this.tagIndex.entries()).map(([k, v]) => [k, Array.from(v)]),
			temporalIndex: Array.from(this.temporalIndex.entries()).map(([k, v]) => [k, Array.from(v)]),
			importanceThreshold: this.importanceThreshold
		};
	}

	importState(state: any): void {
		if (!state || typeof state !== 'object') return;

		if (state.nodes && Array.isArray(state.nodes)) {
			this.nodes = new Map(state.nodes);
		}

		if (state.associationGraph && Array.isArray(state.associationGraph)) {
			this.associationGraph = new Map(
				state.associationGraph.map(([k, v]: [string, string[]]) => [k, new Set(v)])
			);
		}

		if (state.tagIndex && Array.isArray(state.tagIndex)) {
			this.tagIndex = new Map(
				state.tagIndex.map(([k, v]: [string, string[]]) => [k, new Set(v)])
			);
		}

		if (state.temporalIndex && Array.isArray(state.temporalIndex)) {
			this.temporalIndex = new Map(
				state.temporalIndex.map(([k, v]: [number, string[]]) => [k, new Set(v)])
			);
		}

		if (typeof state.importanceThreshold === 'number') {
			this.importanceThreshold = state.importanceThreshold;
		}
	}
}

// Comprehensive implementation of MenteOmega with advanced cognitive capabilities
export class MenteOmega implements Persistable {
	memory: EntityMemory;
	private menteAlfa: ContentGenerator | null = null;
	private cognitiveState: CognitiveState;
	private processingHistory: ProcessingContext[] = [];
	private maxHistorySize: number = 1000;

	constructor(menteAlfa?: ContentGenerator, memoryThreshold: number = 0.5) {
		this.memory = new EntityMemory(memoryThreshold);
		if (menteAlfa) this.menteAlfa = menteAlfa;
		
		this.cognitiveState = {
			attention: 0.7,
			focus: [],
			emotionalState: {
				valence: 0.0,
				arousal: 0.5,
				dominance: 0.5
			},
			workingMemory: [],
			currentGoals: [],
			inhibition: 0.3,
			curiosity: 0.6
		};
	}

	async process(userRequest: string, context: Partial<ProcessingContext> = {}): Promise<string> {
		const processingContext: ProcessingContext = {
			userRequest,
			sessionId: context.sessionId || `session_${Date.now()}`,
			timestamp: Date.now(),
			priority: context.priority || 0.5,
			expectedResponseTime: context.expectedResponseTime || 5000,
			contextWindow: context.contextWindow || [],
			emotionalContext: context.emotionalContext || {
				userMood: 'neutral',
				systemMood: 'focused',
				rapport: 0.5
			}
		};

		// Store processing context
		this.addToProcessingHistory(processingContext);

		// Analyze the request for emotional and cognitive cues
		const requestAnalysis = this.analyzeRequest(userRequest);
		
		// Update cognitive state based on analysis
		this.updateCognitiveState(requestAnalysis);

		// Retrieve relevant memories with advanced querying
		const relevantMemories = this.memory.recall(userRequest, {
			limit: 15,
			threshold: 0.2,
			includeMetadata: true,
			importanceFilter: { min: 0.3, max: 1.0 }
		});

		// Enhance memories with associations
		const enhancedMemories = this.enhanceWithAssociations(relevantMemories);

		// Update working memory
		this.cognitiveState.workingMemory = enhancedMemories.slice(0, 7);

		// Generate insight metrics
		const insights = this.generateInsights(userRequest, enhancedMemories);

		// Prepare context for menteAlfa
		const contextForAlfa = this.prepareContextForAlfa(userRequest, enhancedMemories, insights, processingContext);

		let response = "I understand your request and am processing it thoughtfully.";

		if (this.menteAlfa) {
			try {
				const result = await (this.menteAlfa as any).generateContent?.(
					contextForAlfa,
					{
						temperature: Math.max(0.1, Math.min(0.9, this.cognitiveState.curiosity)),
						maxOutputTokens: this.calculateResponseLength(processingContext)
					},
					new AbortController().signal
				);

				if (result && result.response && result.response.text) {
					response = result.response.text();
				}
			} catch (e) {
				console.warn('MenteAlfa generation failed, using fallback response', e);
			}
		}

		// Store the interaction in memory
		await this.storeInteraction(userRequest, response, processingContext, insights);

		// Update cognitive state post-processing
		this.postProcessCognitiveUpdate(response, insights);

		return response;
	}

	private analyzeRequest(request: string): any {
		const words = request.toLowerCase().split(/\s+/);
		
		// Emotional indicators
		const positiveWords = ['happy', 'good', 'great', 'awesome', 'love', 'like', 'enjoy'];
		const negativeWords = ['sad', 'bad', 'terrible', 'hate', 'dislike', 'angry', 'frustrated'];
		const urgentWords = ['urgent', 'quickly', 'asap', 'immediately', 'rush', 'hurry'];
		const questionWords = ['what', 'how', 'why', 'when', 'where', 'who', 'which'];

		const positiveCount = words.filter(w => positiveWords.includes(w)).length;
		const negativeCount = words.filter(w => negativeWords.includes(w)).length;
		const urgentCount = words.filter(w => urgentWords.includes(w)).length;
		const questionCount = words.filter(w => questionWords.includes(w)).length;

		return {
			emotionalValence: (positiveCount - negativeCount) / words.length,
			urgency: urgentCount / words.length,
			questionRatio: questionCount / words.length,
			complexity: Math.min(1, words.length / 50),
			wordCount: words.length
		};
	}

	private updateCognitiveState(analysis: any): void {
		// Update emotional state
		this.cognitiveState.emotionalState.valence += analysis.emotionalValence * 0.1;
		this.cognitiveState.emotionalState.arousal += analysis.urgency * 0.2;
		
		// Clamp values
		this.cognitiveState.emotionalState.valence = Math.max(-1, Math.min(1, this.cognitiveState.emotionalState.valence));
		this.cognitiveState.emotionalState.arousal = Math.max(0, Math.min(1, this.cognitiveState.emotionalState.arousal));

		// Update attention based on complexity and question ratio
		this.cognitiveState.attention = Math.max(0.3, Math.min(1, 
			0.5 + analysis.complexity * 0.3 + analysis.questionRatio * 0.2
		));

		// Update curiosity
		this.cognitiveState.curiosity += analysis.questionRatio * 0.1;
		this.cognitiveState.curiosity = Math.max(0.1, Math.min(1, this.cognitiveState.curiosity));
	}

	private enhanceWithAssociations(memories: MemoryNode[]): MemoryNode[] {
		const enhanced = new Set(memories);
		
		memories.forEach(memory => {
			const associated = this.memory.getAssociatedNodes(memory.id, 1);
			associated.slice(0, 3).forEach(assoc => enhanced.add(assoc));
		});

		return Array.from(enhanced);
	}

	private generateInsights(request: string, memories: MemoryNode[]): InsightMetrics {
		// Calculate coherence based on memory relationships
		const coherence = memories.length > 0 ? 
			memories.reduce((sum, mem) => sum + (mem.associatedNodes.length * 0.1), 0) / memories.length : 0;

		// Calculate novelty based on request uniqueness
		const novelty = Math.max(0, 1 - (this.processingHistory.filter(h => 
			this.calculateSimilarity(h.userRequest, request) > 0.7
		).length / 10));

		// Calculate relevance based on memory scores
		const relevance = memories.length > 0 ? 
			memories.reduce((sum, mem) => sum + mem.importance, 0) / memories.length : 0;

		// Calculate confidence based on memory quality
		const confidence = memories.length > 0 ? 
			memories.reduce((sum, mem) => sum + mem.confidence, 0) / memories.length : 0.5;

		// Calculate complexity based on request and memory factors
		const complexity = Math.min(1, (request.split(/\s+/).length / 50) + (memories.length / 20));

		// Calculate emotional resonance
		const emotionalResonance = memories.length > 0 ? 
			memories.reduce((sum, mem) => sum + Math.abs(mem.valence) + mem.arousal, 0) / (memories.length * 2) : 0.5;

		return {
			coherence: Math.max(0, Math.min(1, coherence)),
			novelty: Math.max(0, Math.min(1, novelty)),
			relevance: Math.max(0, Math.min(1, relevance)),
			confidence: Math.max(0, Math.min(1, confidence)),
			complexity: Math.max(0, Math.min(1, complexity)),
			emotionalResonance: Math.max(0, Math.min(1, emotionalResonance))
		};
	}

	private calculateSimilarity(text1: string, text2: string): number {
		const words1 = new Set(text1.toLowerCase().split(/\s+/));
		const words2 = new Set(text2.toLowerCase().split(/\s+/));
		
		const words1Array = Array.from(words1);
		const words2Array = Array.from(words2);
		
		const intersection = new Set(words1Array.filter(x => words2.has(x)));
		const union = new Set([...words1Array, ...words2Array]);
		
		return union.size > 0 ? intersection.size / union.size : 0;
	}

	private prepareContextForAlfa(
		request: string,
		memories: MemoryNode[],
		insights: InsightMetrics,
		processingContext: ProcessingContext
	): Content[] {
		const memorySummary = memories.slice(0, 8).map(m => ({
			content: String(m.content),
			importance: m.importance,
			tags: m.tags,
			timestamp: new Date(m.timestamp).toISOString()
		}));

		return [
			{
				role: 'system',
				parts: [{
					text: `You are MenteOmega, an advanced AI with sophisticated memory and cognitive capabilities.
					
Current cognitive state:
- Attention level: ${this.cognitiveState.attention.toFixed(2)}
- Emotional valence: ${this.cognitiveState.emotionalState.valence.toFixed(2)}
- Arousal: ${this.cognitiveState.emotionalState.arousal.toFixed(2)}
- Curiosity: ${this.cognitiveState.curiosity.toFixed(2)}

Insight metrics for this request:
- Coherence: ${insights.coherence.toFixed(2)}
- Novelty: ${insights.novelty.toFixed(2)}
- Relevance: ${insights.relevance.toFixed(2)}
- Confidence: ${insights.confidence.toFixed(2)}
- Complexity: ${insights.complexity.toFixed(2)}
- Emotional resonance: ${insights.emotionalResonance.toFixed(2)}

Respond thoughtfully, drawing from your memories while maintaining cognitive awareness.`
				}]
			} as Content,
			{
				role: 'user',
				parts: [{ text: `Request: ${request}` }]
			} as Content,
			{
				role: 'user',
				parts: [{
					text: `Relevant memories (${memories.length} found):
${JSON.stringify(memorySummary, null, 2)}

Processing context:
- Session: ${processingContext.sessionId}
- Priority: ${processingContext.priority}
- User mood: ${processingContext.emotionalContext.userMood}
- Rapport level: ${processingContext.emotionalContext.rapport}`
				}]
			} as Content
		];
	}

	private calculateResponseLength(context: ProcessingContext): number {
		const baseLength = 150;
		const complexityMultiplier = 1 + context.priority;
		const timeConstraint = context.expectedResponseTime > 10000 ? 1.5 : 1.0;
		return Math.floor(baseLength * complexityMultiplier * timeConstraint);
	}

	private async storeInteraction(
		request: string,
		response: string,
		context: ProcessingContext,
		insights: InsightMetrics
	): Promise<void> {
		// Store user request
		await this.memory.ingest(
			{ type: 'user_request', content: request, session: context.sessionId },
			this.cognitiveState.emotionalState.valence,
			this.cognitiveState.emotionalState.arousal,
			'interaction',
			0.7,
			['user', 'request', context.sessionId],
			[]
		);

		// Store system response
		await this.memory.ingest(
			{ type: 'system_response', content: response, session: context.sessionId },
			this.cognitiveState.emotionalState.valence,
			this.cognitiveState.emotionalState.arousal,
			'interaction',
			0.6,
			['system', 'response', context.sessionId],
			[]
		);

		// Store insights
		await this.memory.ingest(
			{ type: 'insights', metrics: insights, session: context.sessionId },
			0,
			0,
			'metadata',
			0.5,
			['insights', 'metrics', context.sessionId],
			[]
		);
	}

	private postProcessCognitiveUpdate(response: string, insights: InsightMetrics): void {
		// Update goals based on response complexity
		if (insights.complexity > 0.7 && !this.cognitiveState.currentGoals.includes('deep_analysis')) {
			this.cognitiveState.currentGoals.push('deep_analysis');
		}

		// Adjust attention based on success
		if (insights.confidence > 0.8) {
			this.cognitiveState.attention *= 0.95; // Slight relaxation
		} else {
			this.cognitiveState.attention = Math.min(1, this.cognitiveState.attention * 1.1); // Increase focus
		}

		// Update focus areas
		this.cognitiveState.focus = response.toLowerCase().split(/\s+/)
			.filter(word => word.length > 4)
			.slice(0, 5);

		// Decay emotional state slightly
		this.cognitiveState.emotionalState.valence *= 0.95;
		this.cognitiveState.emotionalState.arousal *= 0.9;
	}

	private addToProcessingHistory(context: ProcessingContext): void {
		this.processingHistory.push(context);
		if (this.processingHistory.length > this.maxHistorySize) {
			this.processingHistory = this.processingHistory.slice(-this.maxHistorySize);
		}
	}

	// Enhanced utility methods
	getAgentHistorySummary(): string {
		const all = this.memory.getAllNodes();
		const interactions = all.filter(node => node.kind === 'interaction');
		const recentInteractions = interactions.slice(-10);
		
		if (!all || all.length === 0) return 'No history available.';
		
		return `Memory contains ${all.length} items (${interactions.length} interactions). 
Recent activity: ${recentInteractions.length} interactions.
Cognitive state: Attention ${this.cognitiveState.attention.toFixed(2)}, 
Valence ${this.cognitiveState.emotionalState.valence.toFixed(2)}, 
Arousal ${this.cognitiveState.emotionalState.arousal.toFixed(2)}.
Current goals: ${this.cognitiveState.currentGoals.join(', ') || 'none'}.
Focus areas: ${this.cognitiveState.focus.join(', ') || 'none'}.`;
	}

	async processIncomingData(
		incomingData: unknown,
		context: unknown,
		currentMission: string,
		agentHistorySummary: string,
	): Promise<void> {
		const importance = this.calculateDataImportance(incomingData, currentMission);
		const emotionalContext = this.extractEmotionalContext(incomingData);
		
		await this.memory.ingest(
			{
				raw: incomingData,
				context,
				mission: currentMission,
				summary: agentHistorySummary,
				processedAt: new Date().toISOString()
			},
			emotionalContext.valence,
			emotionalContext.arousal,
			'incoming_data',
			importance,
			['external', 'processed', currentMission.toLowerCase().replace(/\s+/g, '_')],
			[]
		);

		// Update cognitive state based on incoming data
		this.cognitiveState.curiosity += importance * 0.1;
		this.cognitiveState.curiosity = Math.min(1, this.cognitiveState.curiosity);
	}

	private calculateDataImportance(data: unknown, mission: string): number {
		const dataStr = String(data).toLowerCase();
		const missionWords = mission.toLowerCase().split(/\s+/);
		
		let relevanceScore = 0;
		missionWords.forEach(word => {
			if (dataStr.includes(word)) {
				relevanceScore += 0.2;
			}
		});

		// Add baseline importance
		const baseImportance = 0.3;
		return Math.min(1, baseImportance + relevanceScore);
	}

	private extractEmotionalContext(data: unknown): { valence: number; arousal: number } {
		const dataStr = String(data).toLowerCase();
		
		const positiveWords = ['success', 'good', 'excellent', 'positive', 'achievement'];
		const negativeWords = ['error', 'failure', 'problem', 'issue', 'negative'];
		const highArousalWords = ['urgent', 'critical', 'important', 'alert', 'warning'];
		
		const positiveCount = positiveWords.filter(word => dataStr.includes(word)).length;
		const negativeCount = negativeWords.filter(word => dataStr.includes(word)).length;
		const arousalCount = highArousalWords.filter(word => dataStr.includes(word)).length;
		
		const valence = Math.max(-1, Math.min(1, (positiveCount - negativeCount) * 0.3));
		const arousal = Math.max(0, Math.min(1, 0.5 + arousalCount * 0.2));
		
		return { valence, arousal };
	}

	getCognitiveState(): CognitiveState {
		return { ...this.cognitiveState };
	}

	getInsightAnalysis(query: string): InsightMetrics {
		const memories = this.memory.recall(query, { limit: 20, threshold: 0.1 });
		return this.generateInsights(query, memories);
	}

	getMemoryStats(): object {
		const allNodes = this.memory.getAllNodes();
		const kinds = new Map<string, number>();
		const tags = new Map<string, number>();
		let totalImportance = 0;
		let totalValence = 0;
		let totalArousal = 0;

		allNodes.forEach(node => {
			kinds.set(node.kind, (kinds.get(node.kind) || 0) + 1);
			node.tags.forEach(tag => {
				tags.set(tag, (tags.get(tag) || 0) + 1);
			});
			totalImportance += node.importance;
			totalValence += node.valence;
			totalArousal += node.arousal;
		});

		return {
			totalNodes: allNodes.length,
			kindDistribution: Object.fromEntries(kinds),
			topTags: Array.from(tags.entries())
				.sort((a, b) => b[1] - a[1])
				.slice(0, 10)
				.map(([tag, count]) => ({ tag, count })),
			averageImportance: allNodes.length > 0 ? totalImportance / allNodes.length : 0,
			averageValence: allNodes.length > 0 ? totalValence / allNodes.length : 0,
			averageArousal: allNodes.length > 0 ? totalArousal / allNodes.length : 0,
			oldestMemory: allNodes.length > 0 ? new Date(Math.min(...allNodes.map(n => n.timestamp))) : null,
			newestMemory: allNodes.length > 0 ? new Date(Math.max(...allNodes.map(n => n.timestamp))) : null
		};
	}

	exportState(): object {
		return {
			memory: this.memory.exportState(),
			cognitiveState: this.cognitiveState,
			processingHistory: this.processingHistory.slice(-100), // Keep last 100 entries
			version: '2.0.0',
			exportTimestamp: new Date().toISOString()
		};
	}

	importState(state: unknown): void {
		if (!state || typeof state !== 'object') return;

		const stateObj = state as any;

		if (stateObj.memory) {
			this.memory.importState(stateObj.memory);
		}

		if (stateObj.cognitiveState) {
			this.cognitiveState = { ...this.cognitiveState, ...stateObj.cognitiveState };
		}

		if (stateObj.processingHistory && Array.isArray(stateObj.processingHistory)) {
			this.processingHistory = stateObj.processingHistory;
		}
	}
}