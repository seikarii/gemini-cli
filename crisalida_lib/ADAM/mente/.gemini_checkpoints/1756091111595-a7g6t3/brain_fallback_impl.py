"""
brain_fallback.py v3.0 — Cerebro/Cerebelo "fallback" definitivo
Robustez, trazabilidad, adaptabilidad, meta-cognición y hooks.
Integración total con UniverseMindOrchestrator, ConsciousUniverse, EventBus y memoria d=256.
Optimizado para auto-regulación, aprendizaje, crisis, insight y meta-adaptación.
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
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import (
    DEFAULT_ADAM_CONFIG,
    AdamConfig,
    EVAConfig,
)  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    QualiaState,
)  # [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.QualiaState`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
from crisalida_lib.ASTRAL_TOOLS.base import (
    ToolCallResult,
    ToolRegistry,
)  # [`crisalida_lib.ASTRAL_TOOLS.base.ToolRegistry`](crisalida_lib/ASTRAL_TOOLS/base.py)
from crisalida_lib.EVA.eva_memory_mixin import (
    EVAMemoryMixin,
)  # [`crisalida_lib.EVA.eva_memory_mixin.EVAMemoryMixin`](crisalida_lib/EVA/eva_memory_mixin.py)
from crisalida_lib.LOGOS.caos import ChaoticCognitiveCore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Defensive numpy import pattern used across repo
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # many runtime environments may not have numpy installed

# --- Constants ---
MAX_TRACE_LENGTH = 200
DEFAULT_MAX_EXPERIENCES = 100000
DEFAULT_RETENTION_POLICY = "dynamic"
_EPS = 1e-8


# --------------------------
# Utils
# --------------------------
def cos_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors with numpy fallback."""
    try:
        if np is not None:
            va = np.asarray(a, dtype=float)
            vb = np.asarray(b, dtype=float)
            na = float(np.linalg.norm(va)) + _EPS
            nb = float(np.linalg.norm(vb)) + _EPS
            return float(float(np.dot(va, vb) / (na * nb)))
        # pure-python fallback
        na = math.sqrt(sum(float(x) * float(x) for x in a)) + _EPS
        nb = math.sqrt(sum(float(x) * float(x) for x in b)) + _EPS
        dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=False))
        return float(dot / (na * nb)) if na * nb > 0 else 0.0
    except Exception:
        logger.exception("cos_sim fallback encountered an error")
        return 0.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value between lo and hi."""
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return float(lo)


def gen_id(prefix: str = "node") -> str:
    """Generate a concise unique id."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _safe_float(val: Any, fallback: float = 0.0) -> float:
    try:
        if val is None:
            return float(fallback)
        if isinstance(val, (int, float)):
            return float(val)
        return float(val)
    except Exception:
        try:
            return float(str(val))
        except Exception:
            logger.debug("_safe_float failed for %r", val)
            return float(fallback)


# --------------------------
# Data Structures
# --------------------------
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
    alarms: list[CognitiveAlarm] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    last_run_ts: float = 0.0


# --------------------------
# Subsystems (kept concise)
# --------------------------
@dataclass
class EmotionalEvaluator:
    affect_fn: Callable[[dict[str, Any]], tuple[float, float, float]] | None = None

    def evaluate(self, ctx: dict[str, Any]) -> tuple[float, float, float]:
        if self.affect_fn:
            try:
                return self.affect_fn(ctx)
            except Exception:
                logger.exception("affect_fn failed; fallback heuristics")

        threat = _safe_float(ctx.get("threat", 0.0))
        opportunity = _safe_float(ctx.get("opportunity", 0.0))
        progress = _safe_float(ctx.get("progress", 0.0))

        valence = clamp(0.4 * opportunity - 0.6 * threat + 0.3 * progress, -1.0, 1.0)
        arousal = clamp(0.6 * threat + 0.5 * opportunity, 0.0, 1.0)
        dopamine = clamp(0.5 * progress + 0.3 * opportunity, 0.0, 1.0)

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
        }
        return mapping.get(
            physiological_state, {"valence": 0.0, "dopamine": 0.0, "arousal": 0.0}
        )


@dataclass
class ExecutivePFC:
    config: AdamConfig = field(default_factory=lambda: DEFAULT_ADAM_CONFIG)
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    max_wm: int = 7
    inhibition_level: float = 0.2

    def push_wm(self, item: dict[str, Any]) -> None:
        self.working_memory.append(item)
        limit = int(getattr(self.config, "PFC_MAX_WORKING_MEMORY", self.max_wm))
        if len(self.working_memory) > limit:
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
            if isinstance(detailed, dict):
                stress_level = _safe_float(detailed.get("stress_level", 0.0))
                fatigue_level = _safe_float(detailed.get("fatigue_level", 0.0))
                energy_level = _safe_float(detailed.get("energy_level", 0.7))
                arousal = _safe_float(context.get("arousal", 0.5))
                if stress_level > 0.7:
                    base_inhibition *= (
                        (1.0 - stress_level * 0.3)
                        if arousal > 0.6
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
        s = sum(weights) + _EPS
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
    config: AdamConfig = field(default_factory=lambda: DEFAULT_ADAM_CONFIG)
    templates: dict[str, list[float]] = field(default_factory=dict)
    threshold_new: float = 0.75

    def match(self, datum: Any) -> tuple[str, float]:
        emb = self.get_embedding(datum)
        if not emb:
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, 0.0
        if not self.templates:
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, 1.0
        best_id: str | None = None
        best_s = -1.0
        for pid, temb in self.templates.items():
            s = cos_sim(emb, temb)
            if s > best_s:
                best_id, best_s = pid, s
        if best_s < float(
            getattr(self.config, "PATTERN_THRESHOLD_NEW", self.threshold_new)
        ):
            pid = f"pat_{len(self.templates)}"
            self.templates[pid] = emb
            return pid, max(best_s, 0.0)
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
    config: AdamConfig = field(default_factory=lambda: DEFAULT_ADAM_CONFIG)
    parallel_thoughts: int = 1
    last_tick_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        try:
            self.base_hz = float(getattr(self.config, "BASE_HZ", 1.0))
        except Exception:
            self.base_hz = 1.0

    def compute_hz(self, arousal: float, threat: float, safe: float) -> float:
        base = float(getattr(self.config, "BASE_HZ", self.base_hz))
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
    def __init__(self, tool_registry: ToolRegistry | None):
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
                    res["progress"] = _safe_float(res.get("progress", 0.0)) * 0.5
                    res["valence"] = _safe_float(res.get("valence", 0.0)) * 0.5
                    res["info"] = (
                        str(res.get("info", "")) + " (low energy: partial effect)"
                    )
                return res

            if not self.tool_registry:
                return {
                    "progress": -0.1,
                    "valence": -0.2,
                    "threat": 0.5,
                    "info": "No tool registry available.",
                }

            tool_result: ToolCallResult = await self.tool_registry.execute_tool(
                action, **params
            )
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
                            mapped[k] = _safe_float(mapped.get(k, 0.0)) * 0.5
                    mapped["info"] += " (low energy mode)"
                if context.get("threat", 0.0) > 0.7:
                    for k in list(mapped.keys()):
                        if isinstance(mapped[k], (int, float)):
                            mapped[k] = _safe_float(mapped.get(k, 0.0)) * 0.7
                    mapped["info"] += " (threat detected)"
                elif context.get("safe", 0.5) > 0.8:
                    for k in list(mapped.keys()):
                        if isinstance(mapped[k], (int, float)):
                            mapped[k] = _safe_float(mapped.get(k, 0.0)) * 1.2
                    mapped["info"] += " (safe environment)"
                return mapped
            else:
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


# --------------------------
# BrainFallback (core)
# --------------------------
class BrainFallback:
    """
    Robust fallback subconscious engine.
    """

    def __init__(
        self,
        entity_id: str,
        get_embedding: Callable[[Any], Any] | None = None,
        recall_fn: Callable[[Any], tuple[Any, list[str]]] | None = None,
        ingest_fn: Callable[..., Any] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        tool_registry: ToolRegistry | None = None,
        config: AdamConfig | None = None,
    ) -> None:
        self._lock = RLock()
        self.entity_id = entity_id
        self.config = config or DEFAULT_ADAM_CONFIG

        # callables with safe defaults
        if get_embedding is None:

            def _default_get_embedding(x: Any) -> list[float]:
                dim = int(getattr(self.config, "EMBEDDING_DIM", 16))
                if np is not None:
                    return list(np.zeros(dim).tolist())
                return [0.0] * dim

            get_embedding = _default_get_embedding

        if recall_fn is None:

            def _default_recall(x: Any) -> tuple[list[float], list[str]]:
                dim = int(getattr(self.config, "EMBEDDING_DIM", 16))
                vec = list(np.zeros(dim).tolist()) if np is not None else [0.0] * dim
                return vec, []

            recall_fn = _default_recall

        if ingest_fn is None:

            def _ingest_noop(*a, **k):
                logger.debug("ingest noop called with %s %s", a, k)
                return None

            ingest_fn = _ingest_noop

        if emit_event is None:

            def _emit_noop(e: str, c: dict[str, Any]) -> None:
                logger.debug("emit_event noop: %s %s", e, c)
                return None

            emit_event = _emit_noop

        if tool_registry is None:
            try:
                tool_registry = ToolRegistry()
            except Exception:
                tool_registry = None

        self.patterns = PatternRecognizer(
            get_embedding=get_embedding, config=self.config
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
        self.tool_registry = tool_registry
        self._hooks: list[Callable[[dict[str, Any]], None]] = []
        self._trace: list[dict[str, Any]] = []
        self.actuator = SimpleActuator(tool_registry=self.tool_registry)

        # EVA placeholders (kept for backward compatibility, mixin will override in EVABrainFallback)
        self.eva_runtime: Any | None = None
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self.eva_phase: str = "default"

        # Buffered EVA writes to avoid excessive sync calls; flushed periodically
        self._eva_buffer: list[dict[str, Any]] = []
        self._eva_flush_interval = float(
            getattr(self.config, "BRAIN_EVA_FLUSH_S", 30.0)
        )
        self._last_eva_flush_ts = time.time()

    # --------------------------
    # Public runtime API
    # --------------------------
    async def step(self, context: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
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

        # potential long-running execution outside lock
        if film_id:
            long_term = self.films[film_id].fitness
            if not self.exec.allow_action(
                urge_score=arousal, long_term_gain=long_term, context=context
            ):
                # inhibition path
                try:
                    self.emit_event(
                        "SELF_RECALL",
                        {
                            "entity_id": self.entity_id,
                            "why": "inhibition_block",
                            "film": film_id,
                        },
                    )
                except Exception:
                    logger.debug("emit_event failed on inhibition (non-fatal)")
                try:
                    self.ingest_fn(
                        {"conscious_call": "INHIBITION_BLOCK", "film": film_id},
                        valence=0.0,
                        arousal=arousal,
                        kind="conscious",
                    )
                except Exception:
                    logger.debug("ingest_fn failed on inhibition (non-fatal)")
                with self._lock:
                    self._trace.append(
                        {
                            "mode": "INHIBIT",
                            "film": film_id,
                            "hz": hz,
                            "context": copy.deepcopy(context),
                        }
                    )
                    for hook in list(self._hooks):
                        try:
                            hook(context)
                        except Exception:
                            logger.exception("hook failed during inhibition")
                return {"mode": "INHIBIT", "film": film_id, "hz": hz}

        exec_info: dict[str, Any] = {}
        if film_id:
            exec_info = await self._advance_film(film_id, context)

        with self._lock:
            self._check_alarms(film_id, context)
            if context.get("threat", 0.0) > float(
                getattr(self.config, "CRISIS_THREAT_THRESHOLD", 0.9)
            ) or arousal > float(getattr(self.config, "HIGH_AROUSAL_THRESHOLD", 0.9)):
                try:
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
                except Exception:
                    logger.debug("emit_event CRISIS failed (non-fatal)")
            elif (
                context.get("progress", 0.0)
                > float(getattr(self.config, "PROGRESS_THRESHOLD_HIGH", 0.8))
                and dop > float(getattr(self.config, "PROGRESS_THRESHOLD_HIGH", 0.8))
                and random.random() < 0.1
            ):
                try:
                    self.emit_event(
                        "INSIGHT",
                        {
                            "entity_id": self.entity_id,
                            "film": film_id,
                            "pattern": pat_id,
                        },
                    )
                except Exception:
                    logger.debug("emit_event INSIGHT failed (non-fatal)")

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
            if len(self._trace) > MAX_TRACE_LENGTH:
                self._trace = self._trace[-MAX_TRACE_LENGTH:]
            for hook in list(self._hooks):
                try:
                    hook(trace_snapshot)
                except Exception:
                    logger.exception("hook raised")

            # buffer EVA record non-blocking
            self._buffer_eva(
                {
                    "entity_id": self.entity_id,
                    "event_type": "brain_step",
                    "payload": trace_snapshot,
                }
            )
            # try flush in background
            asyncio.create_task(self._maybe_flush_eva_async())

        return trace_snapshot

    # --------------------------
    # Internal execution
    # --------------------------
    async def _advance_film(self, film_id: str, ctx: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            f = self.films.get(film_id)
            if not f:
                return {}
            if self.current is None:
                return {}
            node_id = self.current[1]
            node = f.nodes.get(node_id)
            if node is None:
                # fallback gracefully
                node_id = f.entry or next(iter(f.nodes), "")
                node = f.nodes.get(node_id)
            # defensive snapshot
        if node is None:
            return {}
        # micro adjust and execute outside lock
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
            try:
                self.ingest_fn(
                    {"film": film_id, "node": node_id, "act": node.action},
                    valence=ctx.get("valence", 0.0),
                    arousal=ctx.get("arousal", 0.0),
                    kind="habit",
                )
            except Exception:
                logger.debug("ingest_fn failed when ingesting habit (non-fatal)")
            next_nodes = [
                e.dst for e in f.edges if e.src == node_id and e.condition(ctx)
            ]
            if next_nodes:
                self.current = (film_id, next_nodes[0])
            else:
                self.learn_from_outcome(
                    film_id,
                    reward=ctx.get("progress", 0.0) + ctx.get("opportunity", 0.0),
                    cost=node.cost_energy,
                )
                self.current = (film_id, f.entry)
            node.ts_last = time.time()
            node.last_outcome = ctx.get("progress", 0.0)
        return {
            "action": node.action,
            "cost": node.cost_energy,
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
                    try:
                        self.emit_event(
                            "SELF_RECALL",
                            {
                                "entity_id": self.entity_id,
                                "alarm": al.name,
                                "film": film_id,
                            },
                        )
                    except Exception:
                        logger.debug("emit_event failed in alarm handling")
                    al.fire(ctx)
                    try:
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
                        logger.debug("ingest_fn failed in alarm handling")
            except Exception:
                logger.exception("alarm handling failed (continuing)")

    # --------------------------
    # Trace / stats
    # --------------------------
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
                "tick_hz": getattr(self.freq, "base_hz", 1.0),
                "trace": self.get_trace(8),
                "film_stats": self.get_film_stats(),
            }

    # --------------------------
    # EVA buffering + flush helpers
    # --------------------------
    def _buffer_eva(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._eva_buffer.append(item)

    async def _maybe_flush_eva_async(self) -> None:
        now = time.time()
        with self._lock:
            if (now - self._last_eva_flush_ts) < self._eva_flush_interval:
                return
            buf = list(self._eva_buffer)
            self._eva_buffer.clear()
            self._last_eva_flush_ts = now
        if not buf:
            return
        # Best-effort: if EVAMemoryMixin present, delegate
        for item in buf:
            try:
                # non-blocking: EVAMemoryMixin.record_experience may be sync or async
                res = None
                if hasattr(self, "record_experience"):
                    res = self.record_experience(
                        entity_id=item.get("entity_id", self.entity_id),
                        event_type=item.get("event_type", "brain_event"),
                        data=item.get("payload", {}),
                    )
                elif hasattr(self, "eva_runtime") and getattr(
                    self.eva_runtime, "divine_compiler", None
                ):
                    # legacy fallback: compile small intention and store in local memory store
                    intention = {
                        "intention_type": item.get("event_type", "brain_event").upper(),
                        "experience": item.get("payload"),
                    }
                    try:
                        bc = self._compile_intention_to_bytecode(intention)
                        key = bc.get("bytecode_id", f"bc_{int(time.time()*1000)}")
                        self.eva_memory_store[key] = bc
                        res = key
                    except Exception:
                        logger.exception("legacy eva store failed (non-fatal)")
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception:
                logger.exception("flush_eva failed for item (non-fatal)")

    # --------------------------
    # Learning / film management (kept from existing with minor hardening)
    # --------------------------
    def learn_from_outcome(
        self, film_id: str, reward: float, cost: float = 0.0
    ) -> None:
        """
        ENHANCED: Learn from outcome with one-shot learning for high-impact experiences.

        According to problem statement: implement one-shot learning for high-impact experiences
        that can create a Film instantaneously with higher initial fitness.
        """
        if film_id not in self.films:
            return

        film = self.films[film_id]

        # Calculate impact score (high reward or extreme negative reward = high impact)
        impact_score = abs(reward)

        # ONE-SHOT LEARNING: For high-impact experiences
        if impact_score > 0.8:  # High impact threshold
            # Create instant high-fitness learning
            fitness_boost = impact_score * 2.0 if reward > 0 else -impact_score * 1.5
            film.fitness += fitness_boost

            # Create new Film based on this high-impact experience if positive
            if reward > 0.8:
                self._create_one_shot_film(film_id, reward, impact_score)

            logger.info(
                f"One-shot learning applied to film {film_id} "
                f"(impact: {impact_score:.3f}, fitness boost: {fitness_boost:.3f})"
            )
        else:
            # Standard gradual learning
            learning_rate = 0.1
            fitness_delta = learning_rate * (reward - cost)
            film.fitness += fitness_delta

        # Update film usage and timing
        film.usage += 1
        film.last_run_ts = time.time()

        # Update habit scores in basal ganglia
        for node in film.nodes.values():
            action = node.action
            if action in self.habits.habit_scores:
                self.habits.habit_scores[action] += reward * 0.1
            else:
                self.habits.habit_scores[action] = reward * 0.1

    def _create_one_shot_film(
        self, source_film_id: str, reward: float, impact_score: float
    ) -> str:
        """
        Create a new Film based on high-impact experience for one-shot learning.
        """
        source_film = self.films[source_film_id]

        # Extract successful action sequence
        successful_actions = [node.action for node in source_film.nodes.values()]

        # Create variant with slight modifications for exploration
        variant_actions = successful_actions.copy()
        if len(variant_actions) > 1:
            # Add an optimization step
            variant_actions.insert(-1, "optimize")

        # Create new film with high initial fitness
        new_film_id = self.propose_film_from_pattern(
            label=f"oneshot_{source_film_id}_{int(time.time())}",
            actions=variant_actions,
            tags=["one_shot", "high_impact", f"impact_{impact_score:.1f}"],
        )

        # Set high initial fitness based on the successful experience
        if new_film_id in self.films:
            self.films[new_film_id].fitness = reward * 0.8  # High initial fitness
            logger.info(
                f"One-shot film created: {new_film_id} with initial fitness {reward * 0.8:.3f}"
            )

        return new_film_id

    def forget_unused_films(
        self,
        usage_threshold: int = 3,
        fitness_threshold: float = -0.5,
        age_threshold_days: float = 7.0,
    ) -> list[str]:
        """
        ENHANCED: Implement forgetting mechanism for unused Films.

        According to problem statement: implement "olvido" or "decadencia" for Films
        not utilized or with persistent negative fitness.
        """
        current_time = time.time()
        age_threshold_seconds = age_threshold_days * 24 * 60 * 60

        films_to_forget = []

        for film_id, film in list(self.films.items()):
            should_forget = False
            reason = ""

            # Check usage-based forgetting
            if film.usage < usage_threshold:
                film_age = (
                    current_time - film.last_run_ts
                    if film.last_run_ts > 0
                    else age_threshold_seconds
                )
                if film_age > age_threshold_seconds:
                    should_forget = True
                    reason = f"unused (usage: {film.usage}, age: {film_age / 86400:.1f} days)"

            # Check fitness-based forgetting
            if film.fitness < fitness_threshold:
                should_forget = True
                reason = f"negative fitness ({film.fitness:.3f})"

            # Don't forget one-shot learned films or critical films
            if "one_shot" in film.tags or "critical" in film.tags:
                should_forget = False

            if should_forget:
                films_to_forget.append(film_id)
                logger.info(f"Forgetting film {film_id}: {reason}")
                del self.films[film_id]

        return films_to_forget

    def generate_complex_film(
        self, base_actions: list[str], context: dict[str, Any]
    ) -> str:
        """
        ENHANCED: Generate complex Films with bifurcations or conditional loops.

        According to problem statement: DualMind should generate Films with bifurcations
        or conditional loops, not just linear sequences.
        """
        film_id = f"complex_film_{len(self.films)}_{int(time.time())}"

        # Create nodes for base actions
        nodes = {}
        for i, action in enumerate(base_actions):
            node_id = f"{film_id}_n{i}"
            nodes[node_id] = FilmNode(
                id=node_id, action=action, cost_energy=0.02, expected_reward=0.1
            )

        # Add conditional branch nodes
        decision_node_id = f"{film_id}_decision"
        nodes[decision_node_id] = FilmNode(
            id=decision_node_id,
            action="evaluate_situation",
            cost_energy=0.01,
            expected_reward=0.0,
        )

        # Add alternative action nodes
        alt_action_id = f"{film_id}_alt"
        nodes[alt_action_id] = FilmNode(
            id=alt_action_id,
            action="alternative_approach",
            cost_energy=0.03,
            expected_reward=0.15,
        )

        # Add loop-back node
        loop_node_id = f"{film_id}_loop"
        nodes[loop_node_id] = FilmNode(
            id=loop_node_id,
            action="reassess_and_continue",
            cost_energy=0.02,
            expected_reward=0.05,
        )

        # Create complex film with branching structure
        film = Film(
            id=film_id,
            nodes=nodes,
            entry=f"{film_id}_n0",
            fitness=0.1,
            tags=["complex", "branching", "conditional"],
        )

        # Create complex edge structure with conditionals
        edges = []

        # Linear progression through base actions
        for i in range(len(base_actions) - 1):
            src = f"{film_id}_n{i}"
            dst = f"{film_id}_n{i + 1}"
            edges.append(
                FilmEdge(
                    src=src,
                    dst=dst,
                    condition=lambda ctx: ctx.get("progress", 0.5) > 0.3,
                )
            )

        # Branch to decision point
        if len(base_actions) > 1:
            edges.append(
                FilmEdge(
                    src=f"{film_id}_n{len(base_actions) - 2}",
                    dst=decision_node_id,
                    condition=lambda ctx: ctx.get("threat", 0.0) > 0.4
                    or ctx.get("progress", 0.5) < 0.2,
                )
            )

        # Conditional branches from decision
        edges.append(
            FilmEdge(
                src=decision_node_id,
                dst=alt_action_id,
                condition=lambda ctx: ctx.get("threat", 0.0)
                > 0.6,  # High threat = alternative approach
            )
        )

        edges.append(
            FilmEdge(
                src=decision_node_id,
                dst=f"{film_id}_n{len(base_actions) - 1}",
                condition=lambda ctx: ctx.get("threat", 0.0)
                <= 0.6,  # Low threat = continue normal path
            )
        )

        # Loop back condition
        edges.append(
            FilmEdge(
                src=alt_action_id,
                dst=loop_node_id,
                condition=lambda ctx: ctx.get("progress", 0.5)
                < 0.7,  # If still not successful, loop
            )
        )

        edges.append(
            FilmEdge(
                src=loop_node_id,
                dst=f"{film_id}_n0",  # Loop back to beginning
                condition=lambda ctx: ctx.get("retry_count", 0) < 2,  # Max 2 retries
            )
        )

        film.edges = edges

        # Register alarms for complex film
        self._register_film_alarms(
            film, base_actions + ["evaluate_situation", "alternative_approach"]
        )

        self.films[film_id] = film
        logger.info(
            f"Complex film created: {film_id} with {len(nodes)} nodes and {len(edges)} edges"
        )

        return film_id

    def get_learning_statistics(self) -> dict[str, Any]:
        """Get statistics about learning and film management."""
        if not self.films:
            return {
                "total_films": 0,
                "average_fitness": 0.0,
                "one_shot_films": 0,
                "complex_films": 0,
                "fitness_distribution": {},
            }

        total_films = len(self.films)
        total_fitness = sum(f.fitness for f in self.films.values())
        average_fitness = total_fitness / total_films

        one_shot_films = len([f for f in self.films.values() if "one_shot" in f.tags])
        complex_films = len([f for f in self.films.values() if "complex" in f.tags])

        # Fitness distribution
        fitness_ranges = {
            "excellent": 0,
            "good": 0,
            "average": 0,
            "poor": 0,
            "terrible": 0,
        }
        for film in self.films.values():
            if film.fitness > 0.8:
                fitness_ranges["excellent"] += 1
            elif film.fitness > 0.4:
                fitness_ranges["good"] += 1
            elif film.fitness > 0.0:
                fitness_ranges["average"] += 1
            elif film.fitness > -0.5:
                fitness_ranges["poor"] += 1
            else:
                fitness_ranges["terrible"] += 1

        return {
            "total_films": total_films,
            "average_fitness": average_fitness,
            "one_shot_films": one_shot_films,
            "complex_films": complex_films,
            "fitness_distribution": fitness_ranges,
            "habit_scores": len(self.habits.habit_scores),
        }


class EVABrainFallback(BrainFallback, EVAMemoryMixin):
    """
    EVABrainFallback - Cerebro/Cerebelo extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar experiencias cognitivas como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.

    Refactorizado para usar EVAMemoryMixin para centralizar la lógica de memoria EVA.
    """

    def __init__(
        self,
        entity_id: str,
        get_embedding: Callable[[Any], Any],
        recall_fn: Callable[[Any], tuple[Any, list[str]]],
        ingest_fn: Callable[..., Any],
        emit_event: Callable[[str, dict[str, Any]], None],
        tool_registry: ToolRegistry,
        config: AdamConfig | None = None,
        phase: str = "default",
        eva_config: Any = None,
    ):
        super().__init__(
            entity_id,
            get_embedding,
            recall_fn,
            ingest_fn,
            emit_event,
            tool_registry,
            config,
        )
        self._init_eva_memory(eva_phase=phase)

        # Brain-specific EVA configuration
        self.benchmark_log = []
        self.eva_config = eva_config or EVAConfig()
        self.max_experiences = getattr(
            self.eva_config, "EVA_MEMORY_MAX_EXPERIENCES", 100000
        )
        self.retention_policy = getattr(
            self.eva_config, "EVA_MEMORY_RETENTION_POLICY", "dynamic"
        )

    def _compile_intention_to_bytecode(
        self, intention: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compila la intención usando el compilador divino específico para cerebro.
        """
        runtime = getattr(self, "eva_runtime", None)
        if runtime is None or not getattr(runtime, "divine_compiler", None):
            return super()._compile_intention_to_bytecode(intention)
        try:
            return runtime.divine_compiler.compile_intention(intention)
        except Exception:
            return super()._compile_intention_to_bytecode(intention)

    def eva_ingest_brain_experience(
        self,
        context: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Ingesta una experiencia cerebral usando EVAMemoryMixin con lógica específica del cerebro.
        """
        import time

        context = context or self.get_current_state()

        # Build brain-specific qualia state if not provided
        if qualia_state is None:
            qualia_state = QualiaState(
                emotional_valence=context.get("valence", 0.7),
                cognitive_complexity=context.get("arousal", 0.8),
                consciousness_density=context.get("tick_hz", 0.7),
                narrative_importance=context.get("film_stats", {}).get(
                    "average_fitness", 0.8
                ),
                energy_level=context.get("energy", 1.0),
            )

        # Prepare brain-specific experience data
        experience_data = {
            "entity_id": self.entity_id,
            "context": context,
            "film_stats": self.get_film_stats(),
            "trace": self.get_trace(10),
            "timestamp": time.time(),
            "experience_id": f"eva_brain_{self.entity_id}_{int(time.time())}",
        }

        # Use centralized EVA memory ingest
        experience_id = self.eva_ingest_experience(
            intention_type="ARCHIVE_BRAIN_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

        # Add brain-specific benchmarking
        self.benchmark_log.append(
            {
                "operation": "ingest",
                "experience_id": experience_id,
                "phase": phase or self.eva_phase,
                "timestamp": experience_data["timestamp"],
            }
        )

        # Auto-cleanup memory based on brain's retention policy
        self._auto_cleanup_eva_memory()

        return experience_id

    def eva_recall_brain_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Recall una experiencia cerebral usando EVAMemoryMixin con benchmarking específico.
        """
        # Use centralized recall
        result = self.eva_recall_experience(cue, phase)

        # Add brain-specific benchmarking if successful
        if "error" not in result:
            self.benchmark_log.append(
                {
                    "operation": "recall",
                    "experience_id": result.get("experience_id"),
                    "phase": result.get("phase"),
                    "timestamp": result.get("timestamp"),
                }
            )

        return result

    def add_brain_experience_phase(
        self, experience_id: str, phase: str, context: dict, qualia_state: QualiaState
    ) -> bool:
        """
        Añade una fase alternativa para una experiencia cerebral usando EVAMemoryMixin.
        """
        import time

        experience_data = {
            "entity_id": self.entity_id,
            "context": context,
            "film_stats": self.get_film_stats(),
            "trace": self.get_trace(10),
            "timestamp": time.time(),
        }

        # Use centralized add_experience_phase from mixin
        return self.add_experience_phase(
            experience_id=experience_id,
            phase=phase,
            intention_type="ARCHIVE_BRAIN_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
        )

    def get_benchmark_log(self) -> list:
        """Devuelve el log de benchmarking específico del cerebro."""
        return self.benchmark_log

    def optimize_eva_memory(self):
        """Optimiza la memoria EVA basándose en límites específicos del cerebro."""
        if len(self.eva_memory_store) > self.max_experiences:
            sorted_exps = sorted(
                self.eva_memory_store.items(),
                key=lambda x: getattr(x[1], "timestamp", 0),
            )
            for exp_id, _ in sorted_exps[
                : len(self.eva_memory_store) - self.max_experiences
            ]:
                del self.eva_memory_store[exp_id]

    def _auto_cleanup_eva_memory(self):
        """Limpieza automática de memoria basada en política de retención."""
        if self.retention_policy == "dynamic":
            self.optimize_eva_memory()

    def get_eva_api(self) -> dict:
        """
        Devuelve un diccionario con la API de EVA específica para cerebro,
        combinando métodos del mixin con métodos específicos de cerebro.
        """
        base_api = super().get_eva_api()
        brain_api = {
            "eva_ingest_brain_experience": self.eva_ingest_brain_experience,
            "eva_recall_brain_experience": self.eva_recall_brain_experience,
            "add_brain_experience_phase": self.add_brain_experience_phase,
            "get_benchmark_log": self.get_benchmark_log,
            "optimize_eva_memory": self.optimize_eva_memory,
        }
        return {**base_api, **brain_api}
