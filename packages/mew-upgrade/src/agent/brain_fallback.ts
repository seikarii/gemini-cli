/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file Data structures for BrainFallback, translated from Python.
 */

export interface FilmNode {
  id: string;
  action: string;
  params?: Record<string, any>;
  cost_energy?: number;
  expected_reward?: number;
  last_outcome?: number;
  ts_last?: number; // Timestamp in seconds (Python time.time())
  meta?: Record<string, any>;
  usage_count?: number;
}

export interface CognitiveAlarm {
  name: string;
  trigger_after_s?: number;
  last_reset_ts?: number; // Timestamp in seconds
  condition?: (ctx: Record<string, any>) => boolean;
  on_fire?: (ctx: Record<any, any>) => void; // Changed to any for now, will refine later
  priority?: number;
  meta?: Record<string, any>;
}

export interface FilmEdge {
  src: string;
  dst: string;
  condition: (ctx: Record<string, any>) => boolean;
  priority?: number;
  meta?: Record<string, any>;
}

export interface Film {
  id: string;
  nodes?: Record<string, FilmNode>;
  edges?: FilmEdge[];
  entry?: string;
  fitness?: number;
  usage?: number;
  epic_score?: number;
  alarms?: CognitiveAlarm[];
  tags?: string[];
  last_run_ts?: number; // Timestamp in seconds
}

import { v4 as uuidv4 } from 'uuid';

// --- Utils ---

export function gen_id(prefix: string = "node"): string {
  return `${prefix}_${uuidv4().substring(0, 8)}`;
}

export function _safe_float(val: any, fallback: number = 0.0): number {
  try {
    if (val === null || val === undefined) {
      return fallback;
    }
    if (typeof val === 'number') {
      return val;
    }
    return parseFloat(String(val));
  } catch (e) {
    console.debug("_safe_float failed for", val, e);
    return fallback;
  }
}

// TODO: Define AdamConfig interface
interface AdamConfig { [key: string]: any; }

// TODO: Define ToolRegistry and ChaoticCognitiveCore properly. Keep only
// the types we currently use to avoid unused-declaration TypeScript errors.
type ToolRegistry = { [key: string]: any };
type ChaoticCognitiveCore = { [key: string]: any };

// Subsystems placeholder types
type EmotionalEvaluator = { [key: string]: any };
type ExecutivePFC = { [key: string]: any };
type BasalGanglia = { [key: string]: any };
type Cerebellum = { [key: string]: any };
type PatternRecognizer = { [key: string]: any };
type FrequencyRegulator = { [key: string]: any };
type SimpleActuator = { [key: string]: any };

// Default config (placeholder)
const DEFAULT_ADAM_CONFIG: AdamConfig = {};

export class BrainFallback {
  private _lock: any; // TODO: Implement proper mutex/lock if needed
  public entity_id: string;
  public config: AdamConfig;

  public patterns: PatternRecognizer;
  public exec: ExecutivePFC;
  public limbic: EmotionalEvaluator;
  public habits: BasalGanglia;
  public cerebellum: Cerebellum;
  public freq: FrequencyRegulator;

  public films: Record<string, Film> = {};
  public current: [string, string] | null = null;
  public recall_fn: ((cue: any) => [any, string[]]); // TODO: Refine types
  public ingest_fn: ((...args: any[]) => any); // TODO: Refine types
  public emit_event: ((eventType: string, data: Record<string, any>) => void); // TODO: Refine types
  public tool_registry: ToolRegistry | null;
  private _hooks: ((ctx: Record<string, any>) => void)[] = []; // TODO: Refine types
  private _trace: Record<string, any>[] = []; // TODO: Refine types
  public actuator: SimpleActuator;

  public chaotic_core: ChaoticCognitiveCore | null = null;

  public eva_runtime: any | null = null;
  public eva_memory_store: Record<string, any> = {};
  public eva_experience_store: Record<string, any> = {};
  public eva_phases: Record<string, Record<string, any>> = {};
  public eva_phase: string = "default";

  private _eva_buffer: Record<string, any>[] = [];
  private _eva_flush_interval: number = 30.0;
  private _last_eva_flush_ts: number = Date.now() / 1000;

  constructor(
    entity_id: string,
    get_embedding: ((x: any) => any) | null = null, // TODO: Refine types
    recall_fn: ((cue: any) => [any, string[]]) | null = null, // TODO: Refine types
    ingest_fn: ((...args: any[]) => any) | null = null, // TODO: Refine types
    emit_event: ((eventType: string, data: Record<string, any>) => void) | null = null, // TODO: Refine types
    tool_registry: ToolRegistry | null = null,
    config: AdamConfig | null = null,
  ) {
    // TODO: Implement proper mutex/lock if needed
    this.entity_id = entity_id;
    this.config = config || DEFAULT_ADAM_CONFIG;

    // Callables with safe defaults
    this.recall_fn = recall_fn || ((x: any) => { // TODO: Refine types
      const dim = (this.config as any).EMBEDDING_DIM || 16;
      return [new Array(dim).fill(0.0), []];
    });

    this.ingest_fn = ingest_fn || ((...args: any[]) => { // TODO: Refine types
      console.debug("ingest noop called with", args);
      return null;
    });

    this.emit_event = emit_event || ((eventType: string, data: Record<string, any>) => { // TODO: Refine types
      console.debug("emit_event noop:", eventType, data);
    });

    // TODO: Implement ToolRegistry instantiation if null
    this.tool_registry = tool_registry;

  // TODO: Implement PatternRecognizer, ExecutivePFC, EmotionalEvaluator, BasalGanglia, Cerebellum, FrequencyRegulator, SimpleActuator
    this.patterns = {} as PatternRecognizer; // Placeholder
    this.exec = {} as ExecutivePFC; // Placeholder
    this.limbic = {} as EmotionalEvaluator; // Placeholder
    this.habits = {} as BasalGanglia; // Placeholder
    this.cerebellum = {} as Cerebellum; // Placeholder
    this.freq = {} as FrequencyRegulator; // Placeholder
    this.actuator = {} as SimpleActuator; // Placeholder

    // Chaotic Cognitive Core (optional)
    if ((this.config as any).ENABLE_CHAOTIC_CORE) {
      try {
        this.chaotic_core = {} as ChaoticCognitiveCore; // Placeholder
        console.info("ChaoticCognitiveCore initialized.");
      } catch (e) {
        console.error("Failed to initialize ChaoticCognitiveCore", e);
      }
    }

  // EVA placeholders
  this._eva_flush_interval = (this.config as any).BRAIN_EVA_FLUSH_S || 30.0;
  this._last_eva_flush_ts = Date.now() / 1000; // Convert to seconds

  // Prevent "declared but never used" TypeScript errors for private placeholders
  void this._lock;
  void this._eva_buffer;
  void this._eva_flush_interval;
  void this._last_eva_flush_ts;
  }

  async step(context: Record<string, any>): Promise<Record<string, any>> {
    // TODO: Implement lock handling
    // Recall relevant experiences from memory
    try {
      const recalled_experiences = this.recall_fn(context);
      if (recalled_experiences) {
        context["recalled_experiences"] = recalled_experiences;
      }
    } catch (e) {
      console.error("Failed to recall experiences from memory", e);
    }

    // TODO: Implement patterns.match, limbic.evaluate, freq.compute_hz, freq.update_parallel
    const pat_id = ""; // Placeholder
    const score = 0; // Placeholder
    let val = 0; // Placeholder
    let arousal = 0; // Placeholder
    let dop = 0; // Placeholder
    let hz = 0; // Placeholder

    // Chaotic Cognitive Core integration
    if (this.chaotic_core) {
      try {
        const chaotic_result = await (this.chaotic_core as any).think(context); // Cast to any for now
        if (chaotic_result && chaotic_result.solution) {
          // TODO: Implement _create_film_from_chaotic_solution
          // this._create_film_from_chaotic_solution(chaotic_result.solution, context);
        }
      } catch (e) {
        console.error("ChaoticCognitiveCore.think() failed", e);
      }
    }

    // TODO: Implement _select_film
    const film_id: string | null = null; // Placeholder

    // potential long-running execution outside lock
    if (film_id) {
      const long_term = this.films[film_id].fitness;
      // TODO: Implement exec.allow_action
      if (!(this.exec as any).allow_action(arousal, long_term, context)) {
        // inhibition path
        try {
          this.emit_event(
            "SELF_RECALL",
            {
              "entity_id": this.entity_id,
              "why": "inhibition_block",
              "film": film_id,
            },
          );
        } catch (e) {
          console.debug("emit_event failed on inhibition (non-fatal)", e);
        }
        try {
          this.ingest_fn(
            {"conscious_call": "INHIBITION_BLOCK", "film": film_id},
            0.0, // valence
            arousal,
            "conscious",
          ); // kind
        } catch (e) {
          console.debug("ingest_fn failed on inhibition (non-fatal)", e);
        }
        // TODO: Implement lock handling
        this._trace.push(
          {
            "mode": "INHIBIT",
            "film": film_id,
            "hz": hz,
            "context": structuredClone(context), // Using structuredClone for deep copy
          },
        );
        for (const hook of this._hooks) {
          try {
            hook(context);
          } catch (e) {
            console.error("hook failed during inhibition", e);
          }
        }
        return {"mode": "INHIBIT", "film": film_id, "hz": hz};
      }
    }

    let exec_info: Record<string, any> = {};
    if (film_id) {
      // TODO: Implement _advance_film
      exec_info = await (this as any)._advance_film(film_id, context); // Placeholder
    }

    // TODO: Implement lock handling
    // TODO: Implement _check_alarms
    // TODO: Implement config.CRISIS_THREAT_THRESHOLD, config.HIGH_AROUSAL_THRESHOLD, config.PROGRESS_THRESHOLD_HIGH
  if ((context['threat'] || 0.0) > ((this.config as any).CRISIS_THREAT_THRESHOLD || 0.9) || arousal > ((this.config as any).HIGH_AROUSAL_THRESHOLD || 0.9)) {
      try {
        this.emit_event(
          "CRISIS",
          {
            "entity_id": this.entity_id,
            "film": film_id,
            "ctx": {
        "threat": context['threat'],
              "arousal": arousal,
              "valence": val,
            },
          },
        );
      } catch (e) {
        console.debug("emit_event CRISIS failed (non-fatal)", e);
      }
    } else if (
    (context['progress'] || 0.0) > ((this.config as any).PROGRESS_THRESHOLD_HIGH || 0.8) &&
      dop > ((this.config as any).PROGRESS_THRESHOLD_HIGH || 0.8) &&
      Math.random() < 0.1
    ) {
      try {
        this.emit_event(
          "INSIGHT",
          {
            "entity_id": this.entity_id,
            "film": film_id,
            "pattern": pat_id,
          },
        );
      } catch (e) {
        console.debug("emit_event INSIGHT failed (non-fatal)", e);
      }
    }

    const MAX_TRACE_LENGTH = 200; // Define MAX_TRACE_LENGTH
    const trace_snapshot = {
      "mode": film_id ? "RUN" : "IDLE",
      "film": film_id,
      "node": (this.current && this.current[1]) || null,
      "hz": hz,
      "parallel": (this.freq as any).parallel_thoughts, // Placeholder
      "affect": [val, arousal, dop],
      "pattern": [pat_id, score],
      "exec": exec_info,
      "context": structuredClone(context), // Using structuredClone for deep copy
      "timestamp": Date.now() / 1000, // Convert to seconds
    };
    this._trace.push(trace_snapshot);
    if (this._trace.length > MAX_TRACE_LENGTH) {
      this._trace = this._trace.slice(-MAX_TRACE_LENGTH);
    }
    for (const hook of this._hooks) {
      try {
        hook(trace_snapshot);
      } catch (e) {
        console.error("hook raised", e);
      }
    }

    // buffer EVA record non-blocking
    // TODO: Implement _buffer_eva
    // TODO: Implement _maybe_flush_eva_async
    // asyncio.create_task(this._maybe_flush_eva_async())

    return trace_snapshot;
  }
}
