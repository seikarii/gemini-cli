/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file Implements the ActionSystem, the agent's "body".
 * It takes plans from the mind and executes them by calling tools.
 */
export interface Action {
    tool: string;
    params: Record<string, any>;
    id: string;
}
export interface PlanDeAccion {
    id: string;
    justification: string;
    steps: Action[];
}
/**
 * The ActionSystem is responsible for executing plans.
 * It maintains a queue of actions and calls the appropriate tools.
 */
export declare class ActionSystem {
    private actionQueue;
    private isRunning;
    private toolRegistry;
    constructor();
    /**
     * Receives a plan and adds its steps to the execution queue.
     * @param plan The plan to execute.
     */
    submitPlan(plan: PlanDeAccion): void;
    private run;
}
