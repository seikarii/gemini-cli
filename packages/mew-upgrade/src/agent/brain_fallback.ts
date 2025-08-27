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
  should_fire?: (ctx: Record<string, any>) => boolean;
  fire?: (ctx: Record<string, any>) => void;
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

export function gen_id(prefix: string = 'node'): string {
  return `${prefix}_${uuidv4().substring(0, 8)}`;
}

// --- Constants ---
const _EPS = 1e-8;

// --- Utility functions ---
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
    console.debug('_safe_float failed for', val, e);
    return fallback;
  }
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function cos_sim(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) return 0.0;
  const min_len = Math.min(a.length, b.length);
  let dot = 0.0;
  let norm_a = 0.0;
  let norm_b = 0.0;

  for (let i = 0; i < min_len; i++) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  const denom = Math.sqrt(norm_a) * Math.sqrt(norm_b);
  return denom > _EPS ? dot / denom : 0.0;
}

// TODO: Define AdamConfig interface
interface AdamConfig {
  [key: string]: any;
}

// TODO: Define ToolRegistry and ChaoticCognitiveCore properly. Keep only
// the types we currently use to avoid unused-declaration TypeScript errors.
type ToolRegistry = { [key: string]: any };
type ChaoticCognitiveCore = { [key: string]: any };

// Subsystems placeholder types
export class EmotionalEvaluator {
  affect_fn: ((ctx: Record<string, any>) => [number, number, number]) | null;

  constructor(
    affect_fn:
      | ((ctx: Record<string, any>) => [number, number, number])
      | null = null,
  ) {
    this.affect_fn = affect_fn;
  }

  evaluate(ctx: Record<string, any>): [number, number, number] {
    if (this.affect_fn) {
      try {
        return this.affect_fn(ctx);
      } catch (e) {
        console.error('affect_fn failed; fallback heuristics', e);
      }
    }

    const threat = _safe_float(ctx['threat'], 0.0);
    const opportunity = _safe_float(ctx['opportunity'], 0.0);
    const progress = _safe_float(ctx['progress'], 0.0);

    let valence = clamp(
      0.4 * opportunity - 0.6 * threat + 0.3 * progress,
      -1.0,
      1.0,
    );
    let arousal = clamp(0.6 * threat + 0.5 * opportunity, 0.0, 1.0);
    let dopamine = clamp(0.5 * progress + 0.3 * opportunity, 0.0, 1.0);

    const detailed = ctx['detailed_physiological_state'];
    if (typeof detailed === 'object' && detailed !== null) {
      const feedback = detailed.feedback_modifiers || {};
      valence = clamp(valence + _safe_float(feedback.valence, 0.0), -1.0, 1.0);
      arousal = clamp(arousal + _safe_float(feedback.arousal, 0.0), 0.0, 1.0);
      dopamine = clamp(
        dopamine + _safe_float(feedback.dopamine, 0.0),
        0.0,
        1.0,
      );
    } else {
      const phys = ctx['physiological_state'];
      if (typeof phys === 'string') {
        const som = this._calculate_somatic_modifier(phys);
        valence = clamp(valence + _safe_float(som['valence'], 0.0), -1.0, 1.0);
        arousal = clamp(arousal + _safe_float(som['arousal'], 0.0), 0.0, 1.0);
        dopamine = clamp(
          dopamine + _safe_float(som['dopamine'], 0.0),
          0.0,
          1.0,
        );
      }
    }
    return [valence, arousal, dopamine];
  }

  private _calculate_somatic_modifier(
    physiological_state: string,
  ): Record<string, number> {
    const mapping: Record<string, Record<string, number>> = {
      critical: { valence: -0.4, dopamine: -0.3, arousal: 0.1 },
      stressed: { valence: -0.2, dopamine: -0.15, arousal: 0.2 },
      optimal: { valence: 0.15, dopamine: 0.1, arousal: 0.0 },
      healthy: { valence: 0.15, dopamine: 0.1, arousal: 0.0 },
    };
    return (
      mapping[physiological_state] || {
        valence: 0.0,
        dopamine: 0.0,
        arousal: 0.0,
      }
    );
  }
}
export class ExecutivePFC {
  config: AdamConfig;
  working_memory: Record<string, any>[] = [];
  max_wm: number = 7;
  inhibition_level: number = 0.2;

  constructor(config: AdamConfig = DEFAULT_ADAM_CONFIG) {
    this.config = config;
  }

  push_wm(item: Record<string, any>): void {
    this.working_memory.push(item);
    const limit = _safe_float(
      this.config['PFC_MAX_WORKING_MEMORY'],
      this.max_wm,
    );
    if (this.working_memory.length > limit) {
      this.working_memory.shift(); // Remove the oldest item
    }
  }

  allow_action(
    urge_score: number,
    long_term_gain: number,
    context: Record<string, any> | null = null,
  ): boolean {
    let base_inhibition = _safe_float(
      this.config['PFC_INHIBITION_LEVEL'],
      this.inhibition_level,
    );

    if (context) {
      const detailed = context['detailed_physiological_state'] || {};
      if (typeof detailed === 'object' && detailed !== null) {
        const stress_level = _safe_float(detailed.stress_level, 0.0);
        const fatigue_level = _safe_float(detailed.fatigue_level, 0.0);
        const energy_level = _safe_float(detailed.energy_level, 0.7);
        const arousal = _safe_float(context['arousal'], 0.5);

        if (stress_level > 0.7) {
          base_inhibition *=
            arousal > 0.6 ? 1.0 - stress_level * 0.3 : 1.0 + stress_level * 0.4;
        }
        if (fatigue_level > 0.6) {
          base_inhibition *= 1.0 - fatigue_level * 0.3;
        }
        if (energy_level < 0.3) {
          base_inhibition *= 1.0 - (0.3 - energy_level);
        }
      }
    }

    const gate =
      long_term_gain - base_inhibition * Math.max(0.0, 0.6 - urge_score);
    return gate >= 0.0;
  }
}
export class BasalGanglia {
  habit_scores: Record<string, number> = {};

  constructor() {
    // habit_scores is initialized as an empty object by default
  }

  pick(candidates: string[]): string {
    if (!candidates || candidates.length === 0) {
      return '';
    }
    const weights = candidates.map((c) =>
      Math.exp(this.habit_scores[c] || 0.0),
    );
    const sum_weights = weights.reduce((sum, w) => sum + w, 0) + _EPS;
    const probs = weights.map((w) => w / sum_weights);

    // Implement random.choices equivalent
    const rand = Math.random();
    let cumulative_prob = 0.0;
    for (let i = 0; i < candidates.length; i++) {
      cumulative_prob += probs[i];
      if (rand < cumulative_prob) {
        return candidates[i];
      }
    }
    return candidates[candidates.length - 1]; // Fallback in case of floating point inaccuracies
  }
}
export class Cerebellum {
  micro_adjust(node: FilmNode, ctx: Record<string, any>): void {
    const err = _safe_float(ctx['error'], 0.0);
    node.cost_energy = clamp(
      (node.cost_energy || 0.0) + 0.05 * err,
      0.001,
      1.0,
    );
    node.usage_count = (node.usage_count || 0) + 1;
  }
}
export class PatternRecognizer {
  get_embedding: (x: any) => number[];
  config: AdamConfig;
  templates: Record<string, number[]> = {};
  threshold_new: number = 0.75;

  constructor(
    get_embedding: (x: any) => number[],
    config: AdamConfig = DEFAULT_ADAM_CONFIG,
  ) {
    this.get_embedding = get_embedding;
    this.config = config;
  }

  match(datum: any): [string, number] {
    const emb = this.get_embedding(datum);
    if (!emb || emb.length === 0) {
      const pid = `pat_${Object.keys(this.templates).length}`;
      this.templates[pid] = emb; // Store empty/invalid embedding
      return [pid, 0.0];
    }
    if (Object.keys(this.templates).length === 0) {
      const pid = `pat_${Object.keys(this.templates).length}`;
      this.templates[pid] = emb;
      return [pid, 1.0];
    }

    let best_id: string | null = null;
    let best_s = -1.0;
    for (const pid in this.templates) {
      const temb = this.templates[pid];
      const s = cos_sim(emb, temb);
      if (s > best_s) {
        best_id = pid;
        best_s = s;
      }
    }

    const patternThresholdNew = _safe_float(
      this.config['PATTERN_THRESHOLD_NEW'],
      this.threshold_new,
    );

    if (best_s < patternThresholdNew) {
      const pid = `pat_${Object.keys(this.templates).length}`;
      this.templates[pid] = emb;
      return [pid, Math.max(best_s, 0.0)];
    }

    // online update (conservative)
    const min_len = Math.min(emb.length, this.templates[best_id!].length);
    const updated: number[] = [];
    for (let i = 0; i < min_len; i++) {
      updated.push(0.9 * this.templates[best_id!][i] + 0.1 * emb[i]);
    }
    if (emb.length > min_len) {
      updated.push(...emb.slice(min_len));
    } else if (this.templates[best_id!].length > min_len) {
      updated.push(...this.templates[best_id!].slice(min_len));
    }
    this.templates[best_id!] = updated;
    return [best_id!, best_s];
  }
}
export class FrequencyRegulator {
  config: AdamConfig;
  parallel_thoughts: number = 1;
  last_tick_ts: number = Date.now() / 1000;
  base_hz: number = 1.0;

  constructor(config: AdamConfig = DEFAULT_ADAM_CONFIG) {
    this.config = config;
    try {
      this.base_hz = _safe_float(this.config['BASE_HZ'], 1.0);
    } catch (e) {
      this.base_hz = 1.0;
    }
  }

  compute_hz(arousal: number, threat: number, safe: number): number {
    const base = _safe_float(this.config['BASE_HZ'], this.base_hz);
    const k = base + 40.0 * arousal + 30.0 * threat - 20.0 * safe;
    const min_hz = _safe_float(this.config['MIN_HZ'], 0.1);
    const max_hz = _safe_float(this.config['MAX_HZ'], 240.0);
    return clamp(k, min_hz, max_hz);
  }

  update_parallel(arousal: number, safe: number): void {
    const safe_threshold_high = _safe_float(
      this.config['SAFE_THRESHOLD_HIGH'],
      0.8,
    );
    const safe_threshold_medium = _safe_float(
      this.config['SAFE_THRESHOLD_MEDIUM'],
      0.6,
    );

    if (safe > safe_threshold_high && arousal < 0.3) {
      this.parallel_thoughts = 3;
    } else if (safe > safe_threshold_medium && arousal < 0.4) {
      this.parallel_thoughts = 2;
    } else {
      this.parallel_thoughts = 1;
    }
  }
}
export interface ToolCallResult {
  success: boolean;
  execution_time?: number;
  output?: string;
  command?: string;
  error_message?: string;
}

export class SimpleActuator {
  tool_registry: ToolRegistry | null;

  constructor(tool_registry: ToolRegistry | null) {
    this.tool_registry = tool_registry;
  }

  async execute_action(
    action: string,
    params: Record<string, any>,
    context: Record<string, any>,
  ): Promise<Record<string, any>> {
    console.info(`Executing act: ${action}`);
    try {
      if (['epiphany', 'miracle', 'transcend'].includes(action)) {
        const res: Record<string, any> = {
          progress: 1.0,
          valence: 1.0,
          opportunity: 1.0,
          threat: 0.0,
          tool_output: `Divine act '${action}' manifested.`,
          tool_command: action,
          divine_signature: 'Ω',
          info: 'Reality rewritten by divine will.',
        };
        if (_safe_float(context['energy_balance'], 1.0) < 0.2) {
          res['progress'] = _safe_float(res['progress'], 0.0) * 0.5;
          res['valence'] = _safe_float(res['valence'], 0.0) * 0.5;
          res['info'] = String(res['info']) + ' (low energy: partial effect)';
        }
        return res;
      }

      if (!this.tool_registry) {
        return {
          progress: -0.1,
          valence: -0.2,
          threat: 0.5,
          info: 'No tool registry available.',
        };
      }

      // Assuming tool_registry.execute_tool is an async method
      const tool_result: ToolCallResult = await (
        this.tool_registry as any
      ).execute_tool(action, params); // Pass params as a single object

      if (tool_result.success) {
        const exec_time_val = _safe_float(tool_result.execution_time, 0.0);
        const progress_val = clamp(0.2 + exec_time_val / 8.0, 0.0, 1.0);
        const mapped: Record<string, any> = {
          progress: progress_val,
          valence: 0.2,
          opportunity:
            action.includes('create') || action.includes('explore') ? 0.2 : 0.0,
          threat: 0.0,
          tool_output: tool_result.output || '',
          tool_command: tool_result.command || action,
          divine_signature: action.includes('create') ? 'Φ' : 'Ψ',
          info: `Tool '${action}' executed successfully.`,
        };
        if (_safe_float(context['energy_balance'], 1.0) < 0.2) {
          for (const k in mapped) {
            if (typeof mapped[k] === 'number') {
              mapped[k] = _safe_float(mapped[k], 0.0) * 0.5;
            }
          }
          mapped['info'] += ' (low energy mode)';
        }
        if (_safe_float(context['threat'], 0.0) > 0.7) {
          for (const k in mapped) {
            if (typeof mapped[k] === 'number') {
              mapped[k] = _safe_float(mapped[k], 0.0) * 0.7;
            }
          }
          mapped['info'] += ' (threat detected)';
        } else if (_safe_float(context['safe'], 0.5) > 0.8) {
          for (const k in mapped) {
            if (typeof mapped[k] === 'number') {
              mapped[k] = _safe_float(mapped[k], 0.0) * 1.2;
            }
          }
          mapped['info'] += ' (safe environment)';
        }
        return mapped;
      } else {
        return {
          progress: -0.1,
          valence: -0.5,
          threat: 0.8,
          error_message: tool_result.error_message || 'failed',
          tool_command: tool_result.command || action,
          divine_signature: 'Δ',
          info: 'Tool failed.',
        };
      }
    } catch (e: any) {
      console.error(`Unexpected error executing '${action}'`, e);
      return {
        progress: -0.1,
        valence: -0.3,
        threat: 0.5,
        error_message: String(e),
        tool_command: action,
        divine_signature: '∇',
        info: 'Runtime error.',
      };
    }
  }
}

// Default config (placeholder)
const DEFAULT_ADAM_CONFIG: AdamConfig = {};

export class BrainFallback {
  entity_id: string;
  config: AdamConfig;
  recall_fn: (cue: any) => [any, string[]];
  ingest_fn: (...args: any[]) => any;
  emit_event: (eventType: string, data: Record<string, any>) => void;
  tool_registry: ToolRegistry | null;
  patterns: PatternRecognizer;
  exec: ExecutivePFC;
  limbic: EmotionalEvaluator;
  habits: BasalGanglia;
  cerebellum: Cerebellum;
  freq: FrequencyRegulator;
  actuator: SimpleActuator;
  films: Record<string, Film>;
  current: [string, string] | null;
  _hooks: ((data: any) => void)[];
  _trace: Record<string, any>[];
  chaotic_core: ChaoticCognitiveCore | null = null;
  eva_runtime: any;
  eva_memory_store: Record<string, any>;
  eva_experience_store: Record<string, any>;
  eva_phases: Record<string, any>;
  eva_phase: string;
  _eva_buffer: Record<string, any>[];
  _eva_flush_interval: number;
  _last_eva_flush_ts: number;

  constructor(
    entity_id: string,
    get_embedding: ((x: any) => number[]) | null = null,
    recall_fn: ((cue: any) => [any, string[]]) | null = null,
    ingest_fn: ((...args: any[]) => any) | null = null,
    emit_event:
      | ((eventType: string, data: Record<string, any>) => void)
      | null = null,
    tool_registry: ToolRegistry | null = null,
    config: AdamConfig | null = null,
  ) {
    // TODO: Implement proper mutex/lock if needed
    this.entity_id = entity_id;
    this.config = config || DEFAULT_ADAM_CONFIG;

    // Callables with safe defaults
    this.recall_fn =
      recall_fn ||
      ((x: any) => {
        const dim = _safe_float((this.config as any).EMBEDDING_DIM, 16);
        return [new Array(dim).fill(0.0), []];
      });

    this.ingest_fn =
      ingest_fn ||
      ((...args: any[]) => {
        console.debug('ingest noop called with', args);
        return null;
      });

    this.emit_event =
      emit_event ||
      ((eventType: string, data: Record<string, any>) => {
        console.debug('emit_event noop:', eventType, data);
      });

    this.tool_registry = tool_registry;

    // Subsystem Instantiation
    this.patterns = new PatternRecognizer(
      get_embedding ||
        ((x: any) =>
          new Array(_safe_float((this.config as any).EMBEDDING_DIM, 16)).fill(
            0.0,
          )),
      this.config,
    );
    this.exec = new ExecutivePFC(this.config);
    this.limbic = new EmotionalEvaluator();
    this.habits = new BasalGanglia();
    this.cerebellum = new Cerebellum();
    this.freq = new FrequencyRegulator(this.config);
    this.actuator = new SimpleActuator(this.tool_registry);

    this.films = {};
    this.current = null;
    this._hooks = [];
    this._trace = [];

    // Chaotic Cognitive Core (optional)
    if ((this.config as any).ENABLE_CHAOTIC_CORE) {
      try {
        // TODO: Implement ChaoticCognitiveCore properly
        this.chaotic_core = {} as ChaoticCognitiveCore; // Placeholder
        console.info('ChaoticCognitiveCore initialized.');
      } catch (e) {
        console.error('Failed to initialize ChaoticCognitiveCore', e);
      }
    }

    // EVA placeholders
    this.eva_runtime = null;
    this.eva_memory_store = {};
    this.eva_experience_store = {};
    this.eva_phases = {};
    this.eva_phase = 'default';

    // Buffered EVA writes
    this._eva_buffer = [];
    this._eva_flush_interval = _safe_float(
      (this.config as any).BRAIN_EVA_FLUSH_S,
      30.0,
    );
    this._last_eva_flush_ts = Date.now() / 1000;
  }
  async step(context: Record<string, any>): Promise<Record<string, any>> {
    // Recall relevant experiences from memory
    try {
      const recalled_experiences = this.recall_fn(context);
      if (recalled_experiences) {
        context['recalled_experiences'] = recalled_experiences;
      }
    } catch (e) {
      console.error('Failed to recall experiences from memory', e);
    }

    const [pat_id, score] = this.patterns.match(context['sensory'] || context);
    context['pattern_id'] = pat_id;
    context['pattern_match'] = score;

    const [val, arousal, dop] = this.limbic.evaluate(context);
    context['valence'] = val;
    context['arousal'] = arousal;
    context['dopamine'] = dop;

    const hz = this.freq.compute_hz(
      arousal,
      _safe_float(context['threat'], 0.0),
      _safe_float(context['safe'], 0.0),
    );
    this.freq.update_parallel(arousal, _safe_float(context['safe'], 0.0));
    context['tick_hz'] = hz;
    context['parallel_thoughts'] = this.freq.parallel_thoughts;

    // Chaotic Cognitive Core integration
    if (this.chaotic_core) {
      try {
        const chaotic_result = await (this.chaotic_core as any).think({
          problem_spec: context,
        });
        if (chaotic_result && chaotic_result.solution) {
          // TODO: Implement _create_film_from_chaotic_solution
          // this._create_film_from_chaotic_solution(chaotic_result.solution, context);
        }
      } catch (e) {
        console.error('ChaoticCognitiveCore.think() failed', e);
      }
    }

    const film_id = this._select_film(context);

    // potential long-running execution outside lock
    if (film_id) {
      const film = this.films[film_id];
      const long_term_gain = film ? film.fitness : 0.0;

      if (!this.exec.allow_action(arousal, long_term_gain || 0.0, context)) {
        // inhibition path
        try {
          this.emit_event('SELF_RECALL', {
            entity_id: this.entity_id,
            why: 'inhibition_block',
            film: film_id,
          });
        } catch (e) {
          console.debug('emit_event failed on inhibition (non-fatal)', e);
        }
        try {
          this.ingest_fn(
            { conscious_call: 'INHIBITION_BLOCK', film: film_id },
            0.0, // valence
            arousal,
            'conscious',
          );
        } catch (e) {
          console.debug('ingest_fn failed on inhibition (non-fatal)', e);
        }
        this._trace.push({
          mode: 'INHIBIT',
          film: film_id,
          hz: hz,
          context: structuredClone(context),
        });
        for (const hook of this._hooks) {
          try {
            hook(context);
          } catch (e) {
            console.error('hook failed during inhibition', e);
          }
        }
        return { mode: 'INHIBIT', film: film_id, hz: hz };
      }
    }

    let exec_info: Record<string, any> = {};
    if (film_id) {
      exec_info = await this._advance_film(film_id, context);
    }

    this._check_alarms(film_id, context);

    const crisis_threat_threshold = _safe_float(
      (this.config as any).CRISIS_THREAT_THRESHOLD,
      0.9,
    );
    const high_arousal_threshold = _safe_float(
      (this.config as any).HIGH_AROUSAL_THRESHOLD,
      0.9,
    );
    const progress_threshold_high = _safe_float(
      (this.config as any).PROGRESS_THRESHOLD_HIGH,
      0.8,
    );

    if (
      _safe_float(context['threat'], 0.0) > crisis_threat_threshold ||
      arousal > high_arousal_threshold
    ) {
      try {
        this.emit_event('CRISIS', {
          entity_id: this.entity_id,
          film: film_id,
          ctx: {
            threat: context['threat'],
            arousal: arousal,
            valence: val,
          },
        });
      } catch (e) {
        console.debug('emit_event CRISIS failed (non-fatal)', e);
      }
    } else if (
      _safe_float(context['progress'], 0.0) > progress_threshold_high &&
      dop > progress_threshold_high &&
      Math.random() < 0.1
    ) {
      try {
        this.emit_event('INSIGHT', {
          entity_id: this.entity_id,
          film: film_id,
          pattern: pat_id,
        });
      } catch (e) {
        console.debug('emit_event INSIGHT failed (non-fatal)', e);
      }
    }

    const MAX_TRACE_LENGTH = 200; // Define MAX_TRACE_LENGTH
    const trace_snapshot = {
      mode: film_id ? 'RUN' : 'IDLE',
      film: film_id,
      node: (this.current && this.current[1]) || null,
      hz: hz,
      parallel: this.freq.parallel_thoughts,
      affect: [val, arousal, dop],
      pattern: [pat_id, score],
      exec: exec_info,
      context: structuredClone(context),
      timestamp: Date.now() / 1000,
    };
    this._trace.push(trace_snapshot);
    if (this._trace.length > MAX_TRACE_LENGTH) {
      this._trace = this._trace.slice(-MAX_TRACE_LENGTH);
    }
    for (const hook of this._hooks) {
      try {
        hook(trace_snapshot);
      } catch (e) {
        console.error('hook raised', e);
      }
    }

    // buffer EVA record non-blocking
    this._buffer_eva({
      entity_id: this.entity_id,
      event_type: 'brain_step',
      payload: trace_snapshot,
    });
    // try flush in background
    // TODO: Implement async task creation
    // asyncio.create_task(this._maybe_flush_eva_async());

    return trace_snapshot;
  }

  _select_film(context: Record<string, any>): string | null {
    // Simple implementation - returns the film with highest fitness
    let best_film: string | null = null;
    let best_fitness = -Infinity;

    for (const film_id in this.films) {
      const film = this.films[film_id];
      if ((film.fitness || 0.0) > best_fitness) {
        best_fitness = film.fitness || 0.0;
        best_film = film_id;
      }
    }

    if (best_film) {
      this.current = [best_film, this.films[best_film].entry || ''];
    }

    return best_film;
  }

  async _advance_film(
    film_id: string,
    ctx: Record<string, any>,
  ): Promise<Record<string, any>> {
    // TODO: Implement lock handling
    const f = this.films[film_id];
    if (!f) {
      return {};
    }
    if (this.current === null) {
      return {};
    }
    let node_id = this.current[1];
    let node = f.nodes ? f.nodes[node_id] : undefined;
    if (node === undefined) {
      // fallback gracefully
      node_id =
        f.entry ||
        (f.nodes && Object.keys(f.nodes).length > 0
          ? Object.keys(f.nodes)[0]
          : '');
      node = f.nodes ? f.nodes[node_id] : undefined;
    }
    if (node === undefined) {
      return {};
    }

    let action_result: Record<string, any> = {};
    try {
      this.cerebellum.micro_adjust(node, ctx);
      action_result = await this.actuator.execute_action(
        node.action,
        node.params || {},
        ctx,
      );
    } catch (e) {
      console.error('_advance_film execution failed', e);
      action_result = { progress: 0.0, valence: 0.0 };
    }

    // TODO: Implement lock handling
    for (const key in action_result) {
      const value = action_result[key];
      if (typeof value === 'number') {
        ctx[key] = _safe_float(ctx[key], 0.0) + value;
      } else {
        ctx[key] = value;
      }
      if (['progress', 'opportunity', 'threat', 'safe'].includes(key)) {
        ctx[key] = clamp(ctx[key], 0.0, 1.0);
      } else if (key === 'valence') {
        ctx[key] = clamp(ctx[key], -1.0, 1.0);
      }
    }
    try {
      this.ingest_fn(
        { film: film_id, node: node_id, act: node.action },
        _safe_float(ctx['valence'], 0.0),
        _safe_float(ctx['arousal'], 0.0),
        'habit',
      );
    } catch (e) {
      console.debug('ingest_fn failed when ingesting habit (non-fatal)', e);
    }

    const next_nodes = (f.edges || [])
      .filter((e) => e.src === node_id && e.condition(ctx))
      .map((e) => e.dst);
    if (next_nodes.length > 0) {
      this.current = [film_id, next_nodes[0]];
    } else {
      this.learn_from_outcome(
        film_id,
        _safe_float(ctx['progress'], 0.0) +
          _safe_float(ctx['opportunity'], 0.0),
        node.cost_energy,
      );
      this.current = [film_id, f.entry || '']; // Fallback to empty string if entry is undefined
    }
    node.ts_last = Date.now() / 1000;
    node.last_outcome = _safe_float(ctx['progress'], 0.0);

    return {
      action: node.action,
      cost: node.cost_energy,
      node_id: node_id,
      action_result: action_result,
    };
  }

  _check_alarms(film_id: string | null, ctx: Record<string, any>): void {
    // TODO: Implement lock handling
    if (!film_id) {
      return;
    }
    const f = this.films[film_id];
    if (!f || !f.alarms) {
      return;
    }
    for (const al of f.alarms) {
      try {
        if (al.should_fire && al.should_fire(ctx)) {
          try {
            this.emit_event('SELF_RECALL', {
              entity_id: this.entity_id,
              alarm: al.name,
              film: film_id,
            });
          } catch (e) {
            console.debug('emit_event failed in alarm handling', e);
          }
          if (al.fire) {
            al.fire(ctx);
          }
          try {
            this.ingest_fn(
              {
                conscious_call: 'ALARM',
                name: al.name,
                film: film_id,
              },
              0.0,
              _safe_float(ctx['arousal'], 0.0),
              'conscious',
            );
          } catch (e) {
            console.debug('ingest_fn failed in alarm handling', e);
          }
        }
      } catch (e) {
        console.error('alarm handling failed (continuing)', e);
      }
    }
  }

  _buffer_eva(item: Record<string, any>): void {
    // TODO: Implement lock handling
    this._eva_buffer.push(item);
  }

  async _maybe_flush_eva_async(): Promise<void> {
    const now = Date.now() / 1000;
    // TODO: Implement lock handling
    if (now - this._last_eva_flush_ts < this._eva_flush_interval) {
      return;
    }
    const buf = [...this._eva_buffer];
    this._eva_buffer = [];
    this._last_eva_flush_ts = now;

    if (!buf.length) {
      return;
    }

    for (const item of buf) {
      try {
        // Best-effort: if EVAMemoryMixin present, delegate
        // TODO: Implement actual EVA integration logic
        console.warn('EVA flush not fully implemented yet.', item);
      } catch (e) {
        console.error('flush_eva failed for item (non-fatal)', e);
      }
    }
  }

  learn_from_outcome(
    film_id: string,
    reward: number,
    cost: number = 0.0,
  ): void {
    // TODO: Implement lock handling
    if (!(film_id in this.films)) {
      return;
    }

    const film = this.films[film_id];

    // Calculate impact score (high reward or extreme negative reward = high impact)
    const impact_score = Math.abs(reward);

    // ONE-SHOT LEARNING: For high-impact experiences
    if (impact_score > 0.8) {
      // High impact threshold
      // Create instant high-fitness learning
      const fitness_boost =
        reward > 0 ? impact_score * 2.0 : -impact_score * 1.5;
      film.fitness = (film.fitness || 0.0) + fitness_boost;

      // Create new Film based on this high-impact experience if positive
      if (reward > 0.8) {
        // TODO: Implement _create_one_shot_film
        // this._create_one_shot_film(film_id, reward, impact_score);
      }

      console.info(
        `One-shot learning applied to film ${film_id} ` +
          `(impact: ${impact_score.toFixed(3)}, fitness boost: ${fitness_boost.toFixed(3)})`,
      );
    } else {
      // Standard gradual learning
      const learning_rate = 0.1;
      const fitness_delta = learning_rate * (reward - cost);
      film.fitness = (film.fitness || 0.0) + fitness_delta;
    }

    // Update film usage and timing
    film.usage = (film.usage || 0) + 1;
    film.last_run_ts = Date.now() / 1000;

    // Update habit scores in basal ganglia
    if (film.nodes) {
      for (const node_id in film.nodes) {
        const node: FilmNode = film.nodes[node_id];
        const action = node.action;
        if (action in this.habits.habit_scores) {
          this.habits.habit_scores[action] += reward * 0.1;
        } else {
          this.habits.habit_scores[action] = reward * 0.1;
        }
      }
    }
  }

  _create_film_from_chaotic_solution(
    solution: Record<string, any>,
    context: Record<string, any>,
  ): void {
    // TODO: Implement lock handling
    try {
      const confidence = _safe_float(solution['confidence'], 0.5);
      const chaoticCoreConfidenceThreshold = _safe_float(
        (this.config as any)['CHAOTIC_CORE_CONFIDENCE_THRESHOLD'],
        0.7,
      );

      if (confidence < chaoticCoreConfidenceThreshold) {
        return;
      }

      const film_id = `chaotic_film_${gen_id()}`;
      const action_name = `chaotic_${solution['type'] || 'emergent'}`;
      const node_id = `${film_id}_n0`;

      const node: FilmNode = {
        id: node_id,
        action: action_name,
        params: solution['details'] || {},
        expected_reward: confidence,
      };

      const film: Film = {
        id: film_id,
        nodes: { [node_id]: node },
        entry: node_id,
        fitness: confidence * 1.5, // Give a higher initial fitness
        tags: ['chaotic', `confidence_${confidence.toFixed(2)}`],
      };

      this.films[film_id] = film;
      console.info(`Created new film from chaotic solution: ${film_id}`);
    } catch (e) {
      console.error('Failed to create film from chaotic solution', e);
    }
  }

  _create_one_shot_film(
    source_film_id: string,
    reward: number,
    impact_score: number,
  ): string {
    const film_id = `oneshot_${gen_id()}`;
    const source_film = this.films[source_film_id];

    if (!source_film || !source_film.nodes) {
      return '';
    }

    // Create a simplified version of the source film
    const entry_node_id =
      source_film.entry || Object.keys(source_film.nodes)[0];
    const source_node = source_film.nodes[entry_node_id];

    if (!source_node) {
      return '';
    }

    const node: FilmNode = {
      id: `${film_id}_n0`,
      action: source_node.action,
      params: { ...source_node.params },
      expected_reward: reward,
      cost_energy: (source_node.cost_energy || 0.0) * 0.8, // Slightly more efficient
    };

    const film: Film = {
      id: film_id,
      nodes: { [`${film_id}_n0`]: node },
      entry: `${film_id}_n0`,
      fitness: reward * 1.5, // High initial fitness
      tags: ['one_shot', `impact_${impact_score.toFixed(2)}`],
    };

    this.films[film_id] = film;
    console.info(
      `Created one-shot film: ${film_id} (reward: ${reward.toFixed(3)})`,
    );
    return film_id;
  }

  forget_unused_films(
    usage_threshold: number = 3,
    fitness_threshold: number = -0.5,
    age_threshold_days: number = 7.0,
  ): string[] {
    const now = Date.now() / 1000;
    const age_threshold_s = age_threshold_days * 24 * 3600;
    const forgotten: string[] = [];

    for (const film_id in this.films) {
      const film = this.films[film_id];
      const usage = film.usage || 0;
      const fitness = film.fitness || 0.0;
      const age = film.last_run_ts
        ? now - film.last_run_ts
        : age_threshold_s + 1;

      if (
        usage < usage_threshold &&
        fitness < fitness_threshold &&
        age > age_threshold_s
      ) {
        delete this.films[film_id];
        forgotten.push(film_id);
      }
    }

    console.info(`Forgot ${forgotten.length} unused films`);
    return forgotten;
  }

  generate_complex_film(
    base_actions: string[],
    context: Record<string, any>,
  ): string {
    const film_id = `complex_${gen_id()}`;
    const nodes: Record<string, FilmNode> = {};
    const edges: FilmEdge[] = [];

    // Create nodes for each action
    base_actions.forEach((action, index) => {
      const node_id = `${film_id}_n${index}`;
      nodes[node_id] = {
        id: node_id,
        action: action,
        params: {},
        expected_reward: 0.5,
      };

      // Create sequential edges
      if (index > 0) {
        const prev_node_id = `${film_id}_n${index - 1}`;
        edges.push({
          src: prev_node_id,
          dst: node_id,
          condition: (ctx: Record<string, any>) =>
            _safe_float(ctx['progress'], 0.0) > 0.1,
        });
      }
    });

    const film: Film = {
      id: film_id,
      nodes: nodes,
      edges: edges,
      entry: `${film_id}_n0`,
      fitness: 0.3,
      tags: ['complex', `actions_${base_actions.length}`],
    };

    this.films[film_id] = film;
    console.info(
      `Generated complex film: ${film_id} with ${base_actions.length} actions`,
    );
    return film_id;
  }

  get_trace(last_n: number = 20): Record<string, any>[] {
    return this._trace.slice(-last_n);
  }

  get_current_state(): any {
    return {
      entity_id: this.entity_id,
      current: this.current,
      films_count: Object.keys(this.films).length,
      working_memory_size: this.exec.working_memory.length,
      pattern_templates: Object.keys(this.patterns.templates).length,
      habit_scores: { ...this.habits.habit_scores },
      eva_buffer_size: this._eva_buffer.length,
    };
  }

  get_learning_statistics(): Record<string, any> {
    if (Object.keys(this.films).length === 0) {
      return {
        total_films: 0,
        average_fitness: 0.0,
        one_shot_films: 0,
        complex_films: 0,
        fitness_distribution: {},
      };
    }

    const allFilms = Object.values(this.films);
    const total_films = allFilms.length;
    const total_fitness = allFilms.reduce(
      (sum, f) => sum + (f.fitness || 0.0),
      0,
    );
    const average_fitness = total_fitness / total_films;

    const one_shot_films = allFilms.filter(
      (f) => f.tags && f.tags.includes('one_shot'),
    ).length;
    const complex_films = allFilms.filter(
      (f) => f.tags && f.tags.includes('complex'),
    ).length;

    const fitness_distribution = {
      excellent: 0,
      good: 0,
      average: 0,
      poor: 0,
      terrible: 0,
    };
    for (const film of allFilms) {
      if ((film.fitness || 0.0) > 0.8) {
        fitness_distribution.excellent += 1;
      } else if ((film.fitness || 0.0) > 0.4) {
        fitness_distribution.good += 1;
      } else if ((film.fitness || 0.0) > 0.0) {
        fitness_distribution.average += 1;
      } else if ((film.fitness || 0.0) > -0.5) {
        fitness_distribution.poor += 1;
      } else {
        fitness_distribution.terrible += 1;
      }
    }
    return {
      total_films,
      average_fitness,
      one_shot_films,
      complex_films,
      fitness_distribution,
    };
  }

  get_film_stats(): Record<string, any> {
    // TODO: Implement lock handling
    const film_usage: Record<string, number> = {};
    const epic_scores: Record<string, number> = {};
    const fitness: Record<string, number> = {};
    const tags: Record<string, string[]> = {};

    for (const film_id in this.films) {
      const film = this.films[film_id];
      film_usage[film_id] = film.usage || 0;
      epic_scores[film_id] = film.epic_score || 0.0;
      fitness[film_id] = film.fitness || 0.0;
      tags[film_id] = film.tags || [];
    }
    return { film_usage, epic_scores, fitness, tags };
  }
}
