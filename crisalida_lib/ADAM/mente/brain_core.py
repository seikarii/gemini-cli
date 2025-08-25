"""
Brain Core - Fallback System (definitive)
========================================

Professionalized, hardened and future-proofed version of the BrainFallback
subsystem.

Highlights:
- Thread-safety (RLock) for mutable state.
- Optional numpy-accelerated embeddings with graceful fallback.
- Async-aware EVA recording (best-effort, non-blocking).
- Clearer typing and safe coercions.
- Persistence helpers for simple film snapshots.
- Hook system and diagnostics API for observability.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, cast

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # graceful fallback; many environments run without numpy

from crisalida_lib.ADAM.config import (
    AdamConfig,
)  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    EVAMemoryManager,
)  # [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
from crisalida_lib.ASTRAL_TOOLS.base import (
    ToolRegistry,
)  # [`crisalida_lib.ASTRAL_TOOLS.base.ToolRegistry`](crisalida_lib/ASTRAL_TOOLS/base.py)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _safe_float(val: Any, fallback: float = 0.0) -> float:
    try:
        return float(cast("int | float | str | bytes", val))
    except Exception:
        return float(fallback)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return float(lo)


def gen_id(prefix: str = "node") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def cos_sim(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity with optional numpy acceleration and defensive guards.
    Returns value in [-1.0, 1.0].
    """
    try:
        if np is not None:
            va = np.asarray(a, dtype=float)
            vb = np.asarray(b, dtype=float)
            if va.size == 0 or vb.size == 0:
                return 0.0
            na = np.linalg.norm(va) + 1e-8
            nb = np.linalg.norm(vb) + 1e-8
            return float(float(np.dot(va, vb) / (na * nb)))
        # fallback pure-python
        na = math.sqrt(sum(float(x) * float(x) for x in a)) + 1e-8
        nb = math.sqrt(sum(float(x) * float(x) for x in b)) + 1e-8
        denom = na * nb
        if denom == 0:
            return 0.0
        # zip stops at shortest so use strict-safe behaviour
        dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=False))
        return float(dot / denom)
    except Exception as exc:
        logger.debug("cos_sim error: %s", exc)
        return 0.0


# --- Core dataclasses (unchanged shape but explicit typing) ----------
@dataclass
class FilmNode:
    id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    cost_energy: float = 0.01
    expected_reward: float = 0.0
    last_outcome: float = 0.0
    ts_last: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0


@dataclass
class FilmEdge:
    src: str
    dst: str
    condition: Callable[[dict[str, Any]], bool]
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Film:
    id: str
    nodes: dict[str, FilmNode] = field(default_factory=dict)
    edges: list[FilmEdge] = field(default_factory=list)
    entry: str = ""
    fitness: float = 0.0
    usage: int = 0
    epic_score: float = 0.0
    alarms: list[Any] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    last_run_ts: float = 0.0


@dataclass
class CognitiveAlarm:
    name: str
    trigger_after_s: float = 30.0
    last_reset_ts: float = field(default_factory=time.time)
    condition: Callable[[dict[str, Any]], bool] | None = None
    on_fire: Callable[[dict[str, Any]], None] | None = None
    priority: int = 5
    meta: dict[str, Any] = field(default_factory=dict)

    def should_fire(self, ctx: dict[str, Any]) -> bool:
        try:
            t = time.time()
            timed = (t - self.last_reset_ts) >= float(self.trigger_after_s)
            cond = True if self.condition is None else bool(self.condition(ctx))
            return timed and cond
        except Exception:
            return False

    def fire(self, ctx: dict[str, Any]) -> None:
        try:
            if self.on_fire:
                self.on_fire(ctx)
        finally:
            self.last_reset_ts = time.time()


# --- Evaluators / Subsystems (kept small and testable) --------------
@dataclass
class EmotionalEvaluator:
    affect_fn: Callable[[dict[str, Any]], tuple[float, float, float]] | None = None

    def evaluate(self, ctx: dict[str, Any]) -> tuple[float, float, float]:
        if self.affect_fn:
            try:
                return self.affect_fn(ctx)
            except Exception:
                logger.exception("affect_fn failed; falling back to heuristics")

        threat = _safe_float(ctx.get("threat", 0.0), 0.0)
        opportunity = _safe_float(ctx.get("opportunity", 0.0), 0.0)
        progress = _safe_float(ctx.get("progress", 0.0), 0.0)

        valence = clamp(0.4 * opportunity - 0.6 * threat + 0.3 * progress, -1.0, 1.0)
        arousal = clamp(0.6 * threat + 0.5 * opportunity, 0.0, 1.0)
        dopamine = clamp(0.5 * progress + 0.3 * opportunity, 0.0, 1.0)

        # Support both detailed_physiological_state (dict) and simple physiological_state
        detailed = ctx.get("detailed_physiological_state")
        if isinstance(detailed, dict):
            feedback = detailed.get("feedback_modifiers", {})
            valence = clamp(
                valence + _safe_float(feedback.get("valence", 0.0)), -1.0, 1.0
            )
            arousal = clamp(
                arousal + _safe_float(feedback.get("arousal", 0.0)), 0.0, 1.0
            )
            dopamine = clamp(
                dopamine + _safe_float(feedback.get("dopamine", 0.0)), 0.0, 1.0
            )
        else:
            phys = ctx.get("physiological_state")
            if isinstance(phys, str):
                som = self._calculate_somatic_modifier(phys)
                valence = clamp(
                    valence + _safe_float(som.get("valence", 0.0)), -1.0, 1.0
                )
                arousal = clamp(
                    arousal + _safe_float(som.get("arousal", 0.0)), 0.0, 1.0
                )
                dopamine = clamp(
                    dopamine + _safe_float(som.get("dopamine", 0.0)), 0.0, 1.0
                )
        return valence, arousal, dopamine

    def _calculate_somatic_modifier(self, physiological_state: str) -> dict[str, float]:
        mapping = {
            "critical": {"valence": -0.4, "dopamine": -0.3, "arousal": 0.1},
            "stressed": {"valence": -0.2, "dopamine": -0.15, "arousal": 0.2},
            "optimal": {"valence": 0.15, "dopamine": 0.1, "arousal": 0.0},
            "healthy": {"valence": 0.15, "dopamine": 0.1, "arousal": 0.0},
            "homeostasis": {"valence": 0.05, "dopamine": 0.0, "arousal": -0.05},
        }
        return mapping.get(
            physiological_state, {"valence": 0.0, "dopamine": 0.0, "arousal": 0.0}
        )


@dataclass
class ExecutivePFC:
    config: AdamConfig
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    max_wm: int = 7
    inhibition_level: float = 0.2

    def push_wm(self, item: dict[str, Any]) -> None:
        self.working_memory.append(item)
        limit = getattr(self.config, "PFC_MAX_WORKING_MEMORY", self.max_wm)
        if len(self.working_memory) > int(limit):
            self.working_memory.pop(0)

    def allow_action(
        self,
        urge_score: float,
        long_term_gain: float,
        context: dict[str, Any] | None = None,
    ) -> bool:
        base_inhibition = float(
            getattr(self.config, "PFC_INHIBITION_LEVEL", self.inhibition_level)
        )
        if context:
            detailed = context.get("detailed_physiological_state", {})
            stress_level = _safe_float(
                (
                    detailed.get("stress_level", 0.0)
                    if isinstance(detailed, dict)
                    else 0.0
                ),
                0.0,
            )
            fatigue_level = _safe_float(
                (
                    detailed.get("fatigue_level", 0.0)
                    if isinstance(detailed, dict)
                    else 0.0
                ),
                0.0,
            )
            energy_level = _safe_float(
                (
                    detailed.get("energy_level", 0.7)
                    if isinstance(detailed, dict)
                    else 0.7
                ),
                0.7,
            )
            if stress_level > 0.7:
                base_inhibition *= (
                    (1.0 - stress_level * 0.3)
                    if context.get("arousal", 0.5) > 0.6
                    else (1.0 + stress_level * 0.4)
                )
            if fatigue_level > 0.6:
                base_inhibition *= 1.0 - fatigue_level * 0.3
            if energy_level < 0.3:
                base_inhibition *= 1.0 - (0.3 - energy_level)
        gate = float(long_term_gain) - (
            base_inhibition * max(0.0, 0.6 - float(urge_score))
        )
        return gate >= 0.0


@dataclass
class BasalGanglia:
    habit_scores: dict[str, float] = field(default_factory=dict)

    def pick(self, candidates: list[str]) -> str:
        if not candidates:
            return ""
        weights = [math.exp(self.habit_scores.get(c, 0.0)) for c in candidates]
        s = sum(weights) + 1e-8
        probs = [w / s for w in weights]
        return random.choices(candidates, probs)[0]


@dataclass
class Cerebellum:
    def micro_adjust(self, node: FilmNode, ctx: dict[str, Any]) -> None:
        err = float(_safe_float(ctx.get("error", 0.0), 0.0))
        node.cost_energy = clamp(node.cost_energy + 0.05 * err, 0.001, 1.0)
        node.usage_count += 1


@dataclass
class PatternRecognizer:
    get_embedding: Callable[[Any], list[float]]
    config: AdamConfig
    templates: dict[str, list[float]] = field(default_factory=dict)
    threshold_new: float = 0.75

    def match(self, datum: Any) -> tuple[str, float]:
        emb = self.get_embedding(datum)
        if not emb:
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, 1.0
        if not self.templates:
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, 1.0
        best_id: str | None = None
        best_s = -1.0
        for pid, temb in self.templates.items():
            try:
                s = cos_sim(emb, temb)
            except Exception:
                s = 0.0
            if s > best_s:
                best_id, best_s = pid, s
        if best_id is None or best_s < float(
            getattr(self.config, "PATTERN_THRESHOLD_NEW", self.threshold_new)
        ):
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, best_s if best_s >= 0.0 else 0.0
        # online update (conservative)
        min_len = min(len(emb), len(self.templates[best_id]))
        updated = [
            0.9 * self.templates[best_id][i] + 0.1 * emb[i] for i in range(min_len)
        ]
        if len(emb) > min_len:
            updated.extend(emb[min_len:])
        elif len(self.templates[best_id]) > min_len:
            updated.extend(self.templates[best_id][min_len:])
        self.templates[best_id] = updated
        return best_id, best_s


@dataclass
class FrequencyRegulator:
    config: AdamConfig
    parallel_thoughts: int = 1
    last_tick_ts: float = field(default_factory=time.time)

    def compute_hz(self, arousal: float, threat: float, safe: float) -> float:
        base = float(getattr(self.config, "BASE_HZ", 1.0))
        k = base + 40.0 * float(arousal) + 30.0 * float(threat) - 20.0 * float(safe)
        return clamp(
            k,
            float(getattr(self.config, "MIN_HZ", 0.1)),
            float(getattr(self.config, "MAX_HZ", 240.0)),
        )

    def update_parallel(self, arousal: float, safe: float) -> None:
        if (
            safe > float(getattr(self.config, "SAFE_THRESHOLD_HIGH", 0.8))
            and arousal < 0.3
        ):
            self.parallel_thoughts = 3
        elif (
            safe > float(getattr(self.config, "SAFE_THRESHOLD_MEDIUM", 0.6))
            and arousal < 0.4
        ):
            self.parallel_thoughts = 2
        else:
            self.parallel_thoughts = 1


class SimpleActuator:
    """Executes tools and produces normalized act results."""

    def __init__(self, tool_registry: ToolRegistry):
        self.logger = logging.getLogger(__name__)
        self.tool_registry = tool_registry

    async def execute_action(
        self, action: str, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        self.logger.info("Executing act: %s", action)
        try:
            if action in {"epiphany", "miracle", "transcend"}:
                res = {
                    "progress": 1.0,
                    "valence": 1.0,
                    "opportunity": 1.0,
                    "threat": 0.0,
                    "tool_output": f"Divine act '{action}' manifested.",
                    "tool_command": action,
                    "divine_signature": "Ω",
                    "info": "Reality rewritten by divine will.",
                }
                if context.get("energy_balance", 1.0) < 0.2:
                    res["progress"] *= 0.5
                    res["valence"] *= 0.5
                    res["info"] += " (low energy: partial effect)"
                return res

            tool_result = await self.tool_registry.execute_tool(action, **params)
            if getattr(tool_result, "success", False):
                exec_time_val = _safe_float(
                    getattr(tool_result, "execution_time", 0.0), 0.0
                )
                progress_val = clamp(0.2 + float(exec_time_val) / 8.0, 0.0, 1.0)
                mapped = {
                    "progress": progress_val,
                    "valence": 0.2,
                    "opportunity": (
                        0.2 if ("create" in action or "explore" in action) else 0.0
                    ),
                    "threat": 0.0,
                    "tool_output": getattr(tool_result, "output", ""),
                    "tool_command": getattr(tool_result, "command", action),
                    "divine_signature": "Φ" if "create" in action else "Ψ",
                    "info": f"Tool '{action}' executed successfully.",
                }
                if context.get("energy_balance", 1.0) < 0.2:
                    for k in list(mapped.keys()):
                        if isinstance(mapped[k], (int, float)):
                            mapped[k] = float(mapped[k]) * 0.5
                    mapped["info"] += " (low energy mode)"
                if context.get("threat", 0.0) > 0.7:
                    for k in list(mapped.keys()):
                        if isinstance(mapped[k], (int, float)):
                            mapped[k] = float(mapped[k]) * 0.7
                    mapped["info"] += " (threat detected)"
                elif context.get("safe", 0.5) > 0.8:
                    for k in list(mapped.keys()):
                        if isinstance(mapped[k], (int, float)):
                            mapped[k] = float(mapped[k]) * 1.2
                    mapped["info"] += " (safe environment)"
                return mapped
            else:
                self.logger.error(
                    "Tool '%s' failed: %s",
                    action,
                    getattr(tool_result, "error_message", "unknown"),
                )
                return {
                    "progress": -0.1,
                    "valence": -0.5,
                    "threat": 0.8,
                    "error_message": getattr(tool_result, "error_message", "failed"),
                    "tool_command": getattr(tool_result, "command", action),
                    "divine_signature": "Δ",
                    "info": "Tool failed.",
                }
        except Exception as e:
            self.logger.exception("Unexpected error executing '%s'", action)
            return {
                "progress": -0.1,
                "valence": -0.3,
                "threat": 0.5,
                "error_message": str(e),
                "tool_command": action,
                "divine_signature": "∇",
                "info": "Runtime error.",
            }


# --- BrainFallback (main class) ------------------------------------
class BrainFallback:
    """
    Robust fallback subconscious engine.

    Usage:
      bf = BrainFallback(entity_id, config, eva_manager=..., get_embedding=..., tool_registry=...)
      await bf.step(context)
    """

    def __init__(
        self,
        entity_id: str,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        get_embedding: Callable[[Any], list[float]] | None = None,
        recall_fn: Callable[[Any], tuple[list[float], list[str]]] | None = None,
        ingest_fn: Callable[..., Any] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._lock = RLock()
        self.entity_id = entity_id
        self.config = config
        self.eva_manager = eva_manager

        # sensible defaults
        fallback_embed = (lambda x: [0.0]) if get_embedding is None else get_embedding
        self.patterns = PatternRecognizer(
            get_embedding=fallback_embed, config=self.config
        )
        self.exec = ExecutivePFC(config=self.config)
        self.limbic = EmotionalEvaluator()
        self.habits = BasalGanglia()
        self.cerebellum = Cerebellum()
        self.freq = FrequencyRegulator(config=self.config)
        self.films: dict[str, Film] = {}
        self.current: tuple[str, str] | None = None
        self.recall_fn = recall_fn
        self.ingest_fn = ingest_fn
        self.emit_event = emit_event
        self.tool_registry = tool_registry or ToolRegistry()
        self._hooks: list[Callable[[dict[str, Any]], None]] = []
        self._trace: list[dict[str, Any]] = []
        self.actuator = SimpleActuator(tool_registry=self.tool_registry)
        # lightweight local EVA buffer to avoid thrashing external backend
        self._local_eva_buffer: list[dict[str, Any]] = []
        self._last_eva_flush = 0.0

        logger.info("BrainFallback initialized for '%s'.", self.entity_id)

    # Public API -----------------------------------------------------
    def add_hook(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._hooks.append(fn)

    def remove_hook(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            try:
                self._hooks.remove(fn)
            except ValueError:
                pass

    async def step(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Single cognitive step. Returns trace snapshot.
        """
        with self._lock:
            # pattern matching and emotion evaluation
            pat_id, score = self.patterns.match(context.get("sensory", context))
            context["pattern_id"] = pat_id
            context["pattern_match"] = score
            val, arousal, dop = self.limbic.evaluate(context)
            context.update({"valence": val, "arousal": arousal, "dopamine": dop})
            hz = self.freq.compute_hz(
                arousal=arousal,
                threat=float(context.get("threat", 0.0)),
                safe=float(context.get("safe", 0.0)),
            )
            self.freq.update_parallel(
                arousal=arousal, safe=float(context.get("safe", 0.0))
            )
            context["tick_hz"] = hz
            context["parallel_thoughts"] = self.freq.parallel_thoughts

            film_id = self._select_film(context)
        # release lock for potentially long-running operations
        if film_id:
            long_term = self.films[film_id].fitness
            allowed = self.exec.allow_action(
                urge_score=arousal, long_term_gain=long_term, context=context
            )
            if not allowed:
                self._record_inhibition(film_id, context, arousal)
                return {"mode": "INHIBIT", "film": film_id, "hz": hz}

        exec_info: dict[str, Any] = {}
        if film_id:
            exec_info = await self._advance_film(film_id, context)

        with self._lock:
            self._check_alarms(film_id, context)
            if context.get("threat", 0.0) > float(
                getattr(self.config, "CRISIS_THREAT_THRESHOLD", 0.9)
            ) or arousal > float(getattr(self.config, "HIGH_AROUSAL_THRESHOLD", 0.9)):
                if self.emit_event:
                    self.emit_event(
                        "CRISIS",
                        {
                            "entity_id": self.entity_id,
                            "film": film_id,
                            "ctx": {
                                "threat": context.get("threat"),
                                "arousal": arousal,
                                "valence": val,
                            },
                        },
                    )
            elif (
                context.get("progress", 0.0)
                > float(getattr(self.config, "PROGRESS_THRESHOLD_HIGH", 0.8))
                and dop > float(getattr(self.config, "PROGRESS_THRESHOLD_HIGH", 0.8))
                and random.random() < 0.1
            ):
                if self.emit_event:
                    self.emit_event(
                        "INSIGHT",
                        {
                            "entity_id": self.entity_id,
                            "film": film_id,
                            "pattern": pat_id,
                        },
                    )

            trace_snapshot = {
                "mode": "RUN" if film_id else "IDLE",
                "film": film_id,
                "node": (self.current[1] if self.current is not None else None),
                "hz": hz,
                "parallel": self.freq.parallel_thoughts,
                "affect": (val, arousal, dop),
                "pattern": (pat_id, score),
                "exec": exec_info,
                "context": copy.deepcopy(context),
                "timestamp": time.time(),
            }
            self._trace.append(trace_snapshot)
            self._trace = self._trace[-200:]
            for hook in list(self._hooks):
                try:
                    hook(trace_snapshot)
                except Exception:
                    logger.exception("hook raised")

            # best-effort EVA record (buffered + async)
            self._buffer_eva(
                {
                    "entity_id": self.entity_id,
                    "event_type": "brain_step",
                    "payload": trace_snapshot,
                }
            )
            self._maybe_flush_eva()

        return trace_snapshot

    # Internal methods ------------------------------------------------
    async def _advance_film(self, film_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            f = self.films.get(film_id)
            if not f:
                return {}
            # determine node
            if self.current is None:
                node_id = f.entry or (next(iter(f.nodes)) if f.nodes else "")
            else:
                node_id = self.current[1]
            node = f.nodes.get(node_id)
            if node is None:
                # fallback to entry
                node_id = f.entry or (next(iter(f.nodes)) if f.nodes else "")
                node = f.nodes.get(node_id)
            # defensive: node may still be None
            if node is None:
                return {}
        # perform micro-adjust then execute (no lock)
        try:
            self.cerebellum.micro_adjust(node, ctx)
            action_result = await self.actuator.execute_action(
                node.action, node.params, ctx
            )
        except Exception:
            logger.exception("_advance_film execution failed")
            action_result = {"progress": 0.0, "valence": 0.0}

        with self._lock:
            for key, value in action_result.items():
                if isinstance(value, (int, float)):
                    ctx[key] = float(ctx.get(key, 0.0)) + float(value)
                else:
                    ctx[key] = value
                if key in {"progress", "opportunity", "threat", "safe"}:
                    ctx[key] = clamp(ctx[key], 0.0, 1.0)
                elif key == "valence":
                    ctx[key] = clamp(ctx[key], -1.0, 1.0)
            if self.ingest_fn:
                try:
                    self.ingest_fn(
                        {"film": film_id, "node": node_id, "act": node.action},
                        valence=ctx.get("valence", 0.0),
                        arousal=ctx.get("arousal", 0.0),
                        kind="habit",
                    )
                except Exception:
                    logger.debug("ingest_fn failed (non-fatal)")
            next_nodes = [
                e.dst for e in f.edges if e.src == node_id and e.condition(ctx)
            ]
            if next_nodes:
                self.current = (film_id, next_nodes[0])
            else:
                # learn and reset to entry
                self.learn_from_outcome(
                    film_id,
                    reward=ctx.get("progress", 0.0) + ctx.get("opportunity", 0.0),
                    cost=node.cost_energy,
                )
                self.current = (film_id, f.entry)
            node.ts_last = time.time()
            node.last_outcome = ctx.get("progress", 0.0)
        return {
            "action": node.action if node else None,
            "cost": node.cost_energy if node else 0.0,
            "node_id": node_id,
            "action_result": action_result,
        }

    def _check_alarms(self, film_id: str | None, ctx: dict[str, Any]) -> None:
        if not film_id:
            return
        f = self.films.get(film_id)
        if not f:
            return
        for al in f.alarms:
            try:
                if al.should_fire(ctx):
                    if self.emit_event:
                        self.emit_event(
                            "SELF_RECALL",
                            {
                                "entity_id": self.entity_id,
                                "alarm": al.name,
                                "film": film_id,
                            },
                        )
                    al.fire(ctx)
                    if self.ingest_fn:
                        self.ingest_fn(
                            {
                                "conscious_call": "ALARM",
                                "name": al.name,
                                "film": film_id,
                            },
                            valence=0.0,
                            arousal=ctx.get("arousal", 0.0),
                            kind="conscious",
                        )
            except Exception:
                logger.exception("alarm handling failed")

    def _select_film(self, context: dict[str, Any]) -> str | None:
        with self._lock:
            if not self.films:
                return None
            # highest fitness, prefer non-empty entry
            selected, film = max(self.films.items(), key=lambda kv: kv[1].fitness)
            entry = (
                film.entry
                if film.entry
                else (next(iter(film.nodes)) if film.nodes else "")
            )
            self.current = (selected, entry)
            return selected

    def propose_film_from_pattern(
        self, label: str, actions: list[str], tags: list[str] | None = None
    ) -> str:
        with self._lock:
            fid = f"film_{len(self.films)}"
            nodes: dict[str, FilmNode] = {}
            for i, act in enumerate(actions):
                nid = f"{fid}_n{i}"
                nodes[nid] = FilmNode(id=nid, action=act)
            entry = f"{fid}_n0" if nodes else ""
            film = Film(id=fid, nodes=nodes, entry=entry, tags=tags or [])
            # linear edges
            edges: list[FilmEdge] = []
            keys = list(nodes.keys())
            for i in range(len(keys) - 1):
                src, dst = keys[i], keys[i + 1]
                edges.append(FilmEdge(src=src, dst=dst, condition=lambda ctx: True))
            film.edges = edges
            self._register_film_alarms(film, actions)
            self.films[fid] = film
            return fid

    def _register_film_alarms(self, film: Film, actions: list[str]) -> None:
        # minimal non-intrusive alarm
        try:
            if not getattr(film, "alarms", None):
                film.alarms = []

            class _SimpleAlarm:
                def __init__(self, name: str):
                    self.name = name

                def should_fire(self, ctx: dict[str, Any]) -> bool:
                    return False

                def fire(self, ctx: dict[str, Any]) -> None:
                    return None

            film.alarms.append(_SimpleAlarm(name=f"timeout_{film.id}"))
        except Exception:
            logger.debug("register_film_alarms failed (non-fatal)")

    def learn_from_outcome(
        self, film_id: str, reward: float, cost: float = 0.0
    ) -> None:
        with self._lock:
            film = self.films.get(film_id)
            if not film:
                return
            impact_score = abs(reward)
            if impact_score > 0.8:
                fitness_boost = (
                    (impact_score * 2.0) if reward > 0 else (-impact_score * 1.5)
                )
                film.fitness += fitness_boost
                if reward > 0.8:
                    self._create_one_shot_film(film_id, reward, impact_score)
                logger.info(
                    "One-shot learning applied to %s (impact: %.3f)",
                    film_id,
                    impact_score,
                )
            else:
                learning_rate = 0.1
                fitness_delta = learning_rate * (reward - cost)
                film.fitness += fitness_delta
            film.usage += 1
            film.last_run_ts = time.time()
            for node in film.nodes.values():
                self.habits.habit_scores[node.action] = (
                    self.habits.habit_scores.get(node.action, 0.0) + reward * 0.1
                )

    def _create_one_shot_film(
        self, source_film_id: str, reward: float, impact_score: float
    ) -> str:
        with self._lock:
            source = self.films.get(source_film_id)
            if not source:
                return ""
            successful_actions = [n.action for n in source.nodes.values()]
            variant_actions = successful_actions.copy()
            if len(variant_actions) > 1:
                variant_actions.insert(-1, "optimize")
            new_id = self.propose_film_from_pattern(
                label=f"oneshot_{source_film_id}_{int(time.time())}",
                actions=variant_actions,
                tags=["one_shot", "high_impact"],
            )
            if new_id:
                self.films[new_id].fitness = reward * 0.8
                logger.info("One-shot film created: %s", new_id)
            return new_id

    def forget_unused_films(
        self,
        usage_threshold: int = 3,
        fitness_threshold: float = -0.5,
        age_threshold_days: float = 7.0,
    ) -> list[str]:
        with self._lock:
            now = time.time()
            age_seconds = age_threshold_days * 24 * 60 * 60
            removed: list[str] = []
            for fid, film in list(self.films.items()):
                should_forget = False
                if film.usage < usage_threshold:
                    film_age = (
                        (now - film.last_run_ts)
                        if film.last_run_ts > 0
                        else age_seconds + 1.0
                    )
                    if film_age > age_seconds:
                        should_forget = True
                if film.fitness < fitness_threshold:
                    should_forget = True
                if "one_shot" in film.tags or "critical" in film.tags:
                    should_forget = False
                if should_forget:
                    removed.append(fid)
                    del self.films[fid]
                    logger.info("Forgetting film %s", fid)
            return removed

    def generate_complex_film(
        self, base_actions: list[str], context: dict[str, Any]
    ) -> str:
        with self._lock:
            fid = f"complex_film_{len(self.films)}_{int(time.time())}"
            nodes: dict[str, FilmNode] = {}
            for i, action in enumerate(base_actions):
                nid = f"{fid}_n{i}"
                nodes[nid] = FilmNode(
                    id=nid, action=action, cost_energy=0.02, expected_reward=0.1
                )
            # decision/alt/loop nodes
            decision = f"{fid}_decision"
            nodes[decision] = FilmNode(
                id=decision, action="evaluate_situation", cost_energy=0.01
            )
            alt = f"{fid}_alt"
            nodes[alt] = FilmNode(
                id=alt,
                action="alternative_approach",
                cost_energy=0.03,
                expected_reward=0.15,
            )
            loop = f"{fid}_loop"
            nodes[loop] = FilmNode(
                id=loop, action="reassess_and_continue", cost_energy=0.02
            )
            film = Film(
                id=fid,
                nodes=nodes,
                entry=f"{fid}_n0",
                fitness=0.1,
                tags=["complex", "branching"],
            )
            edges: list[FilmEdge] = []
            keys = [k for k in nodes.keys() if k.startswith(f"{fid}_n")]
            for i in range(len(keys) - 1):
                edges.append(
                    FilmEdge(
                        src=keys[i],
                        dst=keys[i + 1],
                        condition=lambda ctx: ctx.get("progress", 0.5) > 0.3,
                    )
                )
            if len(keys) > 1:
                edges.append(
                    FilmEdge(
                        src=keys[-2],
                        dst=decision,
                        condition=lambda ctx: ctx.get("threat", 0.0) > 0.4
                        or ctx.get("progress", 0.5) < 0.2,
                    )
                )
            edges.append(
                FilmEdge(
                    src=decision,
                    dst=alt,
                    condition=lambda ctx: ctx.get("threat", 0.0) > 0.6,
                )
            )
            edges.append(
                FilmEdge(
                    src=decision,
                    dst=keys[-1],
                    condition=lambda ctx: ctx.get("threat", 0.0) <= 0.6,
                )
            )
            edges.append(
                FilmEdge(
                    src=alt,
                    dst=loop,
                    condition=lambda ctx: ctx.get("progress", 0.5) < 0.7,
                )
            )
            edges.append(
                FilmEdge(
                    src=loop,
                    dst=keys[0],
                    condition=lambda ctx: ctx.get("retry_count", 0) < 2,
                )
            )
            film.edges = edges
            self._register_film_alarms(
                film, base_actions + ["evaluate_situation", "alternative_approach"]
            )
            self.films[fid] = film
            logger.info("Complex film created: %s", fid)
            return fid

    # Diagnostics & state helpers ---------------------------------------
    def get_trace(self, last_n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._trace[-int(last_n) :])

    def get_film_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_films": len(self.films),
                "film_usage": {fid: f.usage for fid, f in self.films.items()},
                "epic_scores": {fid: f.epic_score for fid, f in self.films.items()},
                "fitness": {fid: f.fitness for fid, f in self.films.items()},
                "tags": {fid: f.tags for fid, f in self.films.items()},
            }

    def get_current_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_film": self.current[0] if self.current else None,
                "current_node": self.current[1] if self.current else None,
                "parallel_thoughts": self.freq.parallel_thoughts,
                "tick_hz": self.freq.compute_hz(0.5, 0.0, 0.5),
                "trace": self.get_trace(8),
                "film_stats": self.get_film_stats(),
            }

    # --- EVA buffer + persistence (best-effort, async-aware) ----------
    def _buffer_eva(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._local_eva_buffer.append(item)

    def _maybe_flush_eva(self) -> None:
        now = time.time()
        flush_interval = float(getattr(self.config, "BRAIN_EVA_FLUSH_S", 30.0))
        if (now - self._last_eva_flush) < flush_interval:
            return
        buf: list[dict[str, Any]] = []
        with self._lock:
            if not self._local_eva_buffer:
                return
            buf = self._local_eva_buffer[:]
            self._local_eva_buffer = []
            self._last_eva_flush = now
        if self.eva_manager:
            for item in buf:
                try:
                    # schedule async but don't block
                    res = self.eva_manager.record_experience(
                        entity_id=item.get("entity_id", self.entity_id),
                        event_type=item.get("event_type", "brain_event"),
                        data=item.get("payload", {}),
                    )
                    if asyncio.iscoroutine(res):
                        asyncio.ensure_future(res)
                except Exception:
                    logger.exception("flush_eva failed for item")

    def save_films_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "films": {
                    fid: {
                        "entry": f.entry,
                        "fitness": f.fitness,
                        "usage": f.usage,
                        "tags": f.tags,
                        "nodes": {
                            nid: {"action": n.action, "cost": n.cost_energy}
                            for nid, n in f.nodes.items()
                        },
                    }
                    for fid, f in self.films.items()
                },
                "timestamp": time.time(),
            }

    def load_films_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            try:
                films = snapshot.get("films", {})
                for fid, meta in films.items():
                    nodes = {}
                    for nid, nd in meta.get("nodes", {}).items():
                        nodes[nid] = FilmNode(
                            id=nid,
                            action=nd.get("action", ""),
                            cost_energy=_safe_float(nd.get("cost", 0.01), 0.01),
                        )
                    film = Film(
                        id=fid,
                        nodes=nodes,
                        entry=meta.get("entry", ""),
                        fitness=_safe_float(meta.get("fitness", 0.0), 0.0),
                        usage=int(meta.get("usage", 0)),
                    )
                    film.tags = list(meta.get("tags", []))
                    self.films[fid] = film
            except Exception:
                logger.exception("load_films_snapshot failed")
