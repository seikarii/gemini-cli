/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Implements the StateSerializer, responsible for converting in-memory state
 * to a storable format (and back). This is the lowest layer of the persistence
 * architecture.
 */
export declare class StateSerializer {
    /**
     * Serializes a state object into a compressed JSON string.
     * @param state The agent's state object (e.g., the MentalLaby).
     * @returns A base64 encoded, gzipped JSON string.
     */
    serialize(state: object): string;
    /**
     * Deserializes a compressed string back into a state object.
     * @param data The base64 encoded, gzipped JSON string.
     * @returns The deserialized state object.
     */
    deserialize(data: string): object;
}
