# CRITICAL UPGRADE STRATEGY: THE GOLEM'S MIND

## 1. The Overarching Goal: The Growing Loop

Our objective is to build a "Golem": an autonomous agent (Mew) that learns and improves itself and its own tools. This requires a recursive improvement loop:
1.  The current agent (Gemini) is used to build the more advanced agent (Mew).
2.  The architecture of Mew is used as a blueprint to upgrade Gemini.
3.  The upgraded Gemini becomes a better tool to build Mew.
4.  Eventually, Mew will be capable of upgrading Gemini directly.

## 2. The Core Architecture: A Dual-Mind System

We will model the agent's brain on a human cognitive pattern: a creative mind nurtured by an experiential, analytical mind.

*   **MenteAlfa (The Creative Mind):** The core LLM (Gemini) will serve this role. Its function is to provide the initial, creative, and sometimes unbounded plans and ideas.

*   **MenteOmega (The Analytical Mind):** This is the component we must build. Its function is to ground, validate, refine, and correct the plans from MenteAlfa, using a deep, persistent memory of the project.

## 3. The Immediate Priority: Building the Learning Loop (`BrainFallback`)

The most critical and foundational piece to implement first is the subconscious learning loop, based on Mew's `BrainFallback` system.

*   **Mechanism:** Implement a system of "Films" (procedural memory) where sequences of successful actions are stored and reinforced.
*   **Learning Loop:** Implement `learn_from_outcome` logic. After every action sequence, the "Film" is either strengthened (on success) or weakened (on failure).
*   **Memory Integration:** Every action's input and output must be recorded as an "experience" in the EVA Memory system to create a feedback loop that enables real learning. This solves the primary problem of the agent getting "worse" over time.

## 4. The Long-Term Vision: The Attention Mechanism

The ultimate force multiplier is to develop an "attention" mechanism within MenteOmega.

*   **Goal:** To move beyond passing massive, raw context (e.g., 20 files) to the LLM.
*   **Function:** MenteOmega will use its vast experience stored in EVA Memory to analyze a new problem and construct a high-quality, "pre-gestated" prompt for MenteAlfa (the LLM). This prompt will contain only the most relevant code snippets, summaries of past similar problems, and lessons from previous errors.
*   **Outcome:** This will allow the LLM's powerful creative capabilities to be applied to a much richer, denser, and more relevant context, leading to an exponential increase in performance and insight.

## 5. Foundational Requirement: Persistence Architecture
 
To ensure the agent's memory and learning are not lost on restart, a robust persistence layer is required. We will adopt the layered architecture from Mew
 as a blueprint.

**Directive for the Engineer:**
The persistence system should be implemented in TypeScript and consist of four distinct layers. The following is a description of each layer's
      responsibility, based on the Python reference implementation.

  **The Serialization Layer (`StateSerializer`):**
*   **Responsibility:** Convert in-memory objects (specifically the future `MentalLaby`, `Film`, and `Node` data structures) into a storable format,
      and back.
 *   **Implementation:** Use JSON as the primary format. Implement a custom serializer that can handle complex data types. Include `gzip` compression
      for the final output.
*   **API:** Should expose `serialize(state: object): string` and `deserialize(data: string): object`.

**The Storage Layer (`StorageManager`):**
*   **Responsibility:** Abstract the physical storage medium.
*   **Implementation:** Create a `StorageBackend` interface. The initial concrete implementation will be a `LocalStorageBackend` that reads from and
      writes to the local filesystem. It must handle directory creation and atomic writes (e.g., write to a temp file then rename) to prevent data corruption.
 *   **API:** Should expose `write(path: string, data: string): Promise<void>` and `read(path: string): Promise<string>`.

.  **The Service Layer (`PersistenceService`):**
 *   **Responsibility:** Orchestrate the saving and loading of the agent's state. It knows *what* components of the agent need to be persisted.
*   **Implementation:** This service will be called by the high-level API. It will get the state from the relevant components (e.g., by calling
`brain.export_snapshot()`), use the `StateSerializer` to pack it, and the `StorageManager` to save it.
*   **API:** Should expose `saveAgentState(agent: Agent): Promise<void>` and `loadAgentState(agent: Agent): Promise<void>`.

4.  **The Facade (`UnifiedPersistence`):**
*   **Responsibility:** Provide a simple, top-level API for the main agent loop.
*   **Implementation:** This class will instantiate the `PersistenceService` and expose a minimal set of methods for the agent to use.
 *   **API:** Should expose `backup(agent: Agent): Promise<void>` and `restore(agent: Agent): Promise<void>`.

