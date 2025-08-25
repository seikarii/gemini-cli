/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// SPDX-License-Identifier: MIT
//
// Implements MenteOmega, the analytical mind.
// This class orchestrates the memory and action systems to process requests.

import { MentalLaby } from './mental-laby.js';
import { EntityMemory } from './entity-memory.js';
import { ActionSystem, PlanDeAccion } from '../core/action-system.js';
import type { ContentGenerator } from '@google/gemini-cli-core';
import type { Content } from '@google/genai';
import type { Persistable } from '../persistence/persistence-service.js';

/**
 * MenteOmega is the analytical, experience-driven counterpart to the creative LLM (MenteAlfa).
 * It uses memory to ground and refine plans.
 */
export class MenteOmega implements Persistable {
  memory: EntityMemory; // Changed to EntityMemory
  private actions: ActionSystem;
  private menteAlfa: any; // The connection to Gemini (MenteAlfa) - use any to match different runtime client shapes

  constructor(menteAlfa: ContentGenerator) {
    const mentalLaby = new MentalLaby(); // Create MentalLaby instance
    this.memory = new EntityMemory("mente_omega_entity", mentalLaby); // Initialize EntityMemory
    this.actions = new ActionSystem();
    this.menteAlfa = menteAlfa;
  }

  /**
   * The main processing loop for the agent's mind.
   * @param userRequest The initial request from the user.
   */
  async process(userRequest: string): Promise<void> {
    console.log(`MenteOmega: Processing request: "${userRequest}"`);

    // 1. Recall relevant information from memory.
    const relevantMemories = this.memory.recall(userRequest) as Array<{ data: unknown }>;
    console.log(`MenteOmega: Recalled ${relevantMemories.length} relevant memories.`);

    // 2. Build a context for the creative mind (the LLM).
    const memorySummary = relevantMemories
      .map((m) => JSON.stringify(m.data))
      .slice(0, 5)
      .join('\n');

    const contextForAlfa: Content[] = [
      { role: 'user', parts: [{ text: userRequest }] } as Content,
      { role: 'user', parts: [{ text: `Relevant memories (sample): ${memorySummary}` }] } as Content,
    ];

    // 3. Get a plan from the LLM (MenteAlfa).
    const plan = await this.getPlanFromMenteAlfa(contextForAlfa);
    console.log(`MenteOmega: Received plan "${plan.justification}" from MenteAlfa.`);

    // 4. Submit the plan to the ActionSystem for execution.
    this.actions.submitPlan(plan);

    // 5. Store the result of this interaction as a new memory.
    this.memory.ingest({ request: userRequest, planId: plan.id, timestamp: Date.now() });
  }

  private async getPlanFromMenteAlfa(context: Content[], signal?: AbortSignal): Promise<PlanDeAccion> {
    try {
      // Call the underlying client-style generateContent(contents, generationConfig, abortSignal?) which many callers use
      const result = await (this.menteAlfa as any).generateContent(
        context,
        { temperature: 0.7 },
        signal ?? new AbortController().signal,
      );

      // Try to extract a text response safely
      let textResponse = '';
      try {
        // result.response.text() is a common pattern; guard it
        if (result && typeof result === 'object' && 'response' in result && typeof (result as any).response?.text === 'function') {
          textResponse = (result as any).response.text();
        } else if (typeof String(result) === 'string') {
          textResponse = String(result);
        }
      } catch (e) {
        textResponse = '';
      }

      const planId = `llm_plan_${Date.now()}`;
      const justification = (textResponse || 'No response from LLM').slice(0, 200);

      const steps = [
        {
          id: 'llm_step_1',
          tool: 'run_shell_command',
          params: { command: `echo "LLM: ${justification}"` },
        },
      ];

      return { id: planId, justification, steps };
    } catch (error) {
      console.error('Error getting plan from MenteAlfa:', error);
      return {
        id: `fallback_plan_${Date.now()}`,
        justification: 'LLM call failed, using fallback plan.',
        steps: [
          { id: 'fallback_step_1', tool: 'run_shell_command', params: { command: 'echo "Error: Could not get plan from MenteAlfa."' } },
        ],
      };
    }
  }

  // --- Persistence ---

  exportState(): object {
    return { memory: this.memory.exportState() };
  }

  importState(state: unknown): void {
    if (state && typeof state === 'object' && 'memory' in (state as Record<string, unknown>)) {
      // Accept the persisted shape as-is; mental-laby will validate on import
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.memory.importState((state as any).memory);
    }
  }
}
