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
    ts_last?: number;
    meta?: Record<string, any>;
    usage_count?: number;
}
export interface CognitiveAlarm {
    name: string;
    trigger_after_s?: number;
    last_reset_ts?: number;
    condition?: (ctx: Record<string, any>) => boolean;
    on_fire?: (ctx: Record<any, any>) => void;
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
    last_run_ts?: number;
}
export declare function gen_id(prefix?: string): string;
export declare function _safe_float(val: any, fallback?: number): number;
export declare function clamp(value: number, min: number, max: number): number;
export declare function cos_sim(a: number[], b: number[]): number;
interface AdamConfig {
    [key: string]: any;
}
type ToolRegistry = {
    [key: string]: any;
};
type ChaoticCognitiveCore = {
    [key: string]: any;
};
export declare class EmotionalEvaluator {
    affect_fn: ((ctx: Record<string, any>) => [number, number, number]) | null;
    constructor(affect_fn?: ((ctx: Record<string, any>) => [number, number, number]) | null);
    evaluate(ctx: Record<string, any>): [number, number, number];
    private _calculate_somatic_modifier;
}
export declare class ExecutivePFC {
    config: AdamConfig;
    working_memory: Record<string, any>[];
    max_wm: number;
    inhibition_level: number;
    constructor(config?: AdamConfig);
    push_wm(item: Record<string, any>): void;
    allow_action(urge_score: number, long_term_gain: number, context?: Record<string, any> | null): boolean;
}
export declare class BasalGanglia {
    habit_scores: Record<string, number>;
    constructor();
    pick(candidates: string[]): string;
}
export declare class Cerebellum {
    micro_adjust(node: FilmNode, ctx: Record<string, any>): void;
}
export declare class PatternRecognizer {
    get_embedding: (x: any) => number[];
    config: AdamConfig;
    templates: Record<string, number[]>;
    threshold_new: number;
    constructor(get_embedding: (x: any) => number[], config?: AdamConfig);
    match(datum: any): [string, number];
}
export declare class FrequencyRegulator {
    config: AdamConfig;
    parallel_thoughts: number;
    last_tick_ts: number;
    base_hz: number;
    constructor(config?: AdamConfig);
    compute_hz(arousal: number, threat: number, safe: number): number;
    update_parallel(arousal: number, safe: number): void;
}
export interface ToolCallResult {
    success: boolean;
    execution_time?: number;
    output?: string;
    command?: string;
    error_message?: string;
}
export declare class SimpleActuator {
    tool_registry: ToolRegistry | null;
    constructor(tool_registry: ToolRegistry | null);
    execute_action(action: string, params: Record<string, any>, context: Record<string, any>): Promise<Record<string, any>>;
}
export declare class BrainFallback {
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
    chaotic_core: ChaoticCognitiveCore | null;
    eva_runtime: any;
    eva_memory_store: Record<string, any>;
    eva_experience_store: Record<string, any>;
    eva_phases: Record<string, any>;
    eva_phase: string;
    _eva_buffer: Record<string, any>[];
    _eva_flush_interval: number;
    _last_eva_flush_ts: number;
    constructor(entity_id: string, get_embedding?: ((x: any) => number[]) | null, recall_fn?: ((cue: any) => [any, string[]]) | null, ingest_fn?: ((...args: any[]) => any) | null, emit_event?: ((eventType: string, data: Record<string, any>) => void) | null, tool_registry?: ToolRegistry | null, config?: AdamConfig | null);
    step(context: Record<string, any>): Promise<Record<string, any>>;
    _select_film(context: Record<string, any>): string | null;
    _advance_film(film_id: string, ctx: Record<string, any>): Promise<Record<string, any>>;
    _check_alarms(film_id: string | null, ctx: Record<string, any>): void;
    _buffer_eva(item: Record<string, any>): void;
    _maybe_flush_eva_async(): Promise<void>;
    learn_from_outcome(film_id: string, reward: number, cost?: number): void;
    _create_film_from_chaotic_solution(solution: Record<string, any>, context: Record<string, any>): void;
    _create_one_shot_film(source_film_id: string, reward: number, impact_score: number): string;
    forget_unused_films(usage_threshold?: number, fitness_threshold?: number, age_threshold_days?: number): string[];
    generate_complex_film(base_actions: string[], context: Record<string, any>): string;
    get_trace(last_n?: number): Record<string, any>[];
    get_current_state(): any;
    get_learning_statistics(): Record<string, any>;
    get_film_stats(): Record<string, any>;
}
export {};
