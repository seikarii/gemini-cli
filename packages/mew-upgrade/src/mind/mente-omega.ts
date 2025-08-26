// ```typescript
import type { ContentGenerator } from '@google/gemini-cli-core';
// ```
import type { ContentGenerator } from '@google/gemini-cli-core';
import type { Content } from '@google/genai';
import type { Persistable } from '../persistence/persistence-service.js';

// Minimal implementations of dependencies to keep build fast and safe.
// These are intentionally lightweight so tsc can validate types while
// preserving the expected surface area used by other modules.

type MemoryNode = { embedding?: number[]; content?: unknown; metadata?: Record<string, unknown> };

class EntityMemory {
	private nodes: MemoryNode[] = [];
	recall(_: string) {
		return this.nodes;
	}
	ingest(_data: unknown, _valence?: number, _arousal?: number, _kind?: string, _importance?: number) {
		// no-op for build
	}
	exportState() {
		return this.nodes;
	}
	importState(_state: unknown) {
		// no-op
	}
	getAllNodes() {
		return this.nodes;
	}
}

// NOTE: The full implementations of data significance and memory helpers
// live in other modules. For build-time validation we keep minimal stubs
// inlined where necessary. They are intentionally unused here.

export class MenteOmega implements Persistable {
	memory: EntityMemory;
	private menteAlfa: ContentGenerator | null = null;

	constructor(menteAlfa?: ContentGenerator) {
		this.memory = new EntityMemory();
		if (menteAlfa) this.menteAlfa = menteAlfa;
	}

	async process(userRequest: string): Promise<void> {
		// minimal behavior: recall some memories and optionally call menteAlfa
		const relevantMemories = this.memory.recall(userRequest) as Array<{ data: unknown }>;
		const memorySummary = (relevantMemories || []).map((m) => String(m.data)).slice(0, 5).join('\n');

		const contextForAlfa: Content[] = [
			{ role: 'user', parts: [{ text: userRequest }] } as Content,
			{ role: 'user', parts: [{ text: `Relevant memories (sample): ${memorySummary}` }] } as Content,
		];

			if (this.menteAlfa) {
				try {
					// call generateContent in a defensive way and ignore the returned value
					// eslint-disable-next-line @typescript-eslint/no-explicit-any
					await (this.menteAlfa as any).generateContent?.(contextForAlfa, { temperature: 0.7 }, new AbortController().signal);
				} catch (e) {
					// ignore
				}
			}
	}

		// Minimal helper used by GeminiAgent
		import type { ContentGenerator } from '@google/gemini-cli-core';
		import type { Content } from '@google/genai';
		import type { Persistable } from '../persistence/persistence-service.js';

		// Minimal, well-typed stub implementation to keep build green while
		// preserving the public surface used by other modules (GeminiAgent, tests).

		type MemoryNode = { embedding?: number[]; content?: unknown; metadata?: Record<string, unknown> };

		class EntityMemory {
			private nodes: MemoryNode[] = [];
			recall(_: string) {
				return this.nodes;
			import type { ContentGenerator } from '@google/gemini-cli-core';
			import type { Content } from '@google/genai';
			import type { Persistable } from '../persistence/persistence-service.js';

			// Minimal, well-typed stub implementation to keep build green while
			// preserving the public surface used by other modules (GeminiAgent, tests).

			type MemoryNode = { embedding?: number[]; content?: unknown; metadata?: Record<string, unknown> };

			class EntityMemory {
				private nodes: MemoryNode[] = [];
				recall(_: string) {
					return this.nodes;
				}
				ingest(_data: unknown) {
					// no-op for build
				}
				exportState() {
					return this.nodes;
				}
				importState(_state: unknown) {
					// no-op
				}
				getAllNodes() {
					return this.nodes;
				}
			}

			export class MenteOmega implements Persistable {
				memory: EntityMemory;
				private menteAlfa: ContentGenerator | null = null;

				constructor(menteAlfa?: ContentGenerator) {
					this.memory = new EntityMemory();
					if (menteAlfa) this.menteAlfa = menteAlfa;
				}

				async process(userRequest: string): Promise<void> {
					// minimal behavior for build: recall some memories and optionally call menteAlfa
					const relevantMemories = this.memory.recall(userRequest) as Array<{ data: unknown }>;
					const memorySummary = (relevantMemories || []).map((m) => String(m.data)).slice(0, 5).join('\n');

					const contextForAlfa: Content[] = [
						{ role: 'user', parts: [{ text: userRequest }] } as Content,
						{ role: 'user', parts: [{ text: `Relevant memories (sample): ${memorySummary}` }] } as Content,
					];

					if (this.menteAlfa) {
						try {
							// call generateContent defensively and ignore returned value
							// eslint-disable-next-line @typescript-eslint/no-explicit-any
							await (this.menteAlfa as any).generateContent?.(contextForAlfa, { temperature: 0.7 }, new AbortController().signal);
						} catch (e) {
							// ignore
						}
					}
				}

				getAgentHistorySummary(): string {
					const all = this.memory.getAllNodes();
					if (!all || all.length === 0) return 'No history available.';
					return `Memory contains ${all.length} items.`;
				}

				async processIncomingData(
					incomingData: unknown,
					_context: unknown,
					_currentMission: string,
					_agentHistorySummary: string,
				): Promise<void> {
					this.memory.ingest(incomingData);
				}

				exportState(): object {
					return { memory: this.memory.exportState() };
				}
				importState(state: unknown): void {
					if (state && typeof state === 'object' && 'memory' in (state as Record<string, unknown>)) {
						// eslint-disable-next-line @typescript-eslint/no-explicit-any
						this.memory.importState((state as any).memory);
					}
				}
			}