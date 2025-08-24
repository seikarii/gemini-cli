/**
 * @file Implements the PersistenceService, which orchestrates the serialization
 * and storage of the agent's state.
 */

import { StateSerializer } from './state-serializer.js';
import { StorageBackend, LocalStorageBackend } from './storage-manager.js';

// Defines the structure for any component that can have its state persisted.
export interface Persistable {
  exportState(): object;
  importState(state: object): void;
}

export class PersistenceService {
  private storage: StorageBackend;
  private serializer: StateSerializer;
  private basePath: string;

  constructor(basePath: string, storage?: StorageBackend, serializer?: StateSerializer) {
    this.basePath = basePath;
    this.storage = storage || new LocalStorageBackend();
    this.serializer = serializer || new StateSerializer();
    console.log(`PersistenceService initialized at base path: ${this.basePath}`);
  }

  /**
   * Saves the state of a persistable component.
   * @param component The component to save (e.g., the Brain, Memory).
   * @param key A unique key for the state, used as the filename.
   */
  public async save(component: Persistable, key: string): Promise<void> {
    try {
      const state = component.exportState();
      const serializedData = this.serializer.serialize(state);
      const filePath = `${this.basePath}/${key}.json.gz`; // Example file path
      
      await this.storage.write(filePath, serializedData);
      console.log(`State for '${key}' saved successfully to ${filePath}`);

    } catch (error) {
      console.error(`Failed to save state for key '${key}'`, error);
      throw error;
    }
  }

  /**
   * Loads the state for a persistable component.
   * @param component The component to load state into.
   * @param key The unique key for the state to load.
   * @returns True if the state was loaded successfully, false otherwise.
   */
  public async load(component: Persistable, key: string): Promise<boolean> {
    try {
      const filePath = `${this.basePath}/${key}.json.gz`;

      if (!await this.storage.exists(filePath)) {
        console.log(`No saved state found for key '${key}' at ${filePath}.`);
        return false;
      }

      const serializedData = await this.storage.read(filePath);
      const state = this.serializer.deserialize(serializedData);
      
      component.importState(state);
      console.log(`State for '${key}' loaded successfully from ${filePath}`);
      return true;

    } catch (error) {
      console.error(`Failed to load state for key '${key}'`, error);
      throw error;
    }
  }
}
