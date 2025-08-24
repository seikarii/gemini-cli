/**
 * @file Implements the UnifiedPersistence facade, providing a simple top-level API
 * for the agent to save and load its entire state.
 */

import { PersistenceService, Persistable } from './persistence-service.js';

// A placeholder for the main agent class.
// It will need to expose its persistable components.
interface Agent {
  getPersistableComponents(): Record<string, Persistable>;
}

/**
 * The UnifiedPersistence class provides a simple facade for the entire persistence system.
 * The main agent interacts with this class directly.
 */
export class UnifiedPersistence {
  private service: PersistenceService;

  constructor(basePath: string) {
    // The base path where all state files will be stored.
    // e.g., '/home/user/.gemini/agent_state/'
    this.service = new PersistenceService(basePath);
  }

  /**
   * Backs up the entire state of the agent.
   * It iterates through the agent's components and saves each one.
   * @param agent The main agent instance.
   */
  public async backup(agent: Agent): Promise<void> {
    console.log("--- Starting Agent State Backup ---");
    const components = agent.getPersistableComponents();
    for (const key in components) {
      await this.service.save(components[key], key);
    }
    console.log("--- Agent State Backup Complete ---");
  }

  /**
   * Restores the entire state of the agent from storage.
   * @param agent The main agent instance.
   */
  public async restore(agent: Agent): Promise<void> {
    console.log("--- Starting Agent State Restore ---");
    const components = agent.getPersistableComponents();
    for (const key in components) {
      await this.service.load(components[key], key);
    }
    console.log("--- Agent State Restore Complete ---");
  }
}
