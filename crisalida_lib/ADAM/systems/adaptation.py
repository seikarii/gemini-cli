"""
Adaptive Tuner for Adam's consciousness configuration (definitive)
=================================================================

- Hardened numeric backend (numpy preferred, pure-Python fallback).
- Defensive TYPE_CHECKING imports to avoid circular dependencies.
- Async-aware EVA persistence (best-effort) with coroutine detection.
- Smoothing of metrics (EMA), bounds/clamping, and safe parameter updates.
- Small public API additions: simulate_tuning_run, benchmark, export/import history.
- Preserves public API of AdaptiveTuner and EVAAdaptiveTuner while filling placeholders.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import (
    AdamConfig,
)  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)

# Defensive numeric backend pattern used across repo
try:
    import numpy as _np  # type: ignore

    np = _np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

    class _np_fallback:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0

        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return math.sqrt(variance)

        @staticmethod
        def array(lst):
            return list(lst)

    np = _np_fallback()  # type: ignore

# Avoid heavy runtime imports to reduce circular import pressure
if TYPE_CHECKING:
    from crisalida_lib.EDEN.living_symbol import (
        LivingSymbolRuntime,
    )  # [`crisalida_lib.EDEN.living_symbol.LivingSymbolRuntime`](crisalida_lib/EDEN/living_symbol.py)
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        RealityBytecode,
    )  # [`crisalida_lib.EVA.core_types`](crisalida_lib/EVA/core_types.py)
    from crisalida_lib.EVA.typequalia import (
        QualiaState,
    )  # [`crisalida_lib.EVA.typequalia.QualiaState`](crisalida_lib/EVA/typequalia.py)
else:
    LivingSymbolRuntime = Any
    EVAExperience = Any
    RealityBytecode = Any
    QualiaState = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Utilities ---
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _ema(prev: float, value: float, alpha: float = 0.2) -> float:
    if prev is None:
        return value
    return (1 - alpha) * prev + alpha * value


@dataclass
class PerformanceEvent:
    """Represents a performance event in the system."""

    timestamp: datetime
    event_type: str  # CRISIS, INSIGHT, SUCCESS, FAILURE, etc.
    context: dict[str, Any]
    severity: float = 0.5  # 0.0 to 1.0
    outcome: str | None = None


class AdaptiveTuner:
    """
    Adaptive configuration tuner that monitors system performance and
    slowly adjusts configuration parameters to optimize Adam's behavior.

    Public methods:
      - record_event
      - should_tune
      - tune_configuration
      - get_tuning_history
      - get_current_metrics
      - reset_history
      - simulate_tuning_run (convenience for tests)
      - benchmark (simulate synthetic event stream)
    """

    def __init__(self, config: AdamConfig, cycle_interval: int | None = None) -> None:
        self.config = config
        self.cycle_interval = int(
            cycle_interval or getattr(config, "ADAPTIVE_TUNER_CYCLE_INTERVAL", 100)
        )
        self.history_window = int(
            getattr(config, "ADAPTIVE_TUNER_HISTORY_WINDOW", 1000)
        )
        self.min_confidence = float(
            getattr(config, "ADAPTIVE_TUNER_MIN_CONFIDENCE", 0.5)
        )
        self.adjustment_rate = float(
            getattr(config, "ADAPTIVE_TUNER_ADJUSTMENT_RATE", 0.05)
        )

        self.event_history: deque[PerformanceEvent] = deque(maxlen=self.history_window)
        self.cycle_count = 0
        self.last_tuning_cycle = 0
        self.tuning_history: list[dict[str, Any]] = []

        # EMA-smoothed metrics (start neutral)
        self.crisis_frequency = 0.0
        self.insight_frequency = 0.0
        self.success_rate = 0.5
        self.alfa_failure_rate = 0.0
        self.omega_failure_rate = 0.0

        logger.info(
            "AdaptiveTuner initialized (cycle_interval=%s, history_window=%s)",
            self.cycle_interval,
            self.history_window,
        )

    def record_event(
        self,
        event_type: str,
        context: dict[str, Any],
        severity: float = 0.5,
        outcome: str | None = None,
    ) -> None:
        event = PerformanceEvent(
            datetime.utcnow(),
            event_type,
            dict(context or {}),
            float(severity or 0.0),
            outcome,
        )
        self.event_history.append(event)
        # Update smoothed metrics incrementally to avoid expensive full scans every event
        self._update_performance_metrics(incremental=True)

    def should_tune(self) -> bool:
        self.cycle_count += 1
        return (self.cycle_count - self.last_tuning_cycle) >= self.cycle_interval

    def tune_configuration(self) -> dict[str, Any]:
        if not self.should_tune():
            return {
                "status": "not_ready",
                "cycles_until_next": self.cycle_interval
                - (self.cycle_count - self.last_tuning_cycle),
            }

        logger.info("AdaptiveTuner: running tuning cycle %s", self.cycle_count)
        analysis = self._analyze_performance_trends()
        adjustments = self._calculate_adjustments(analysis)
        applied_adjustments = self._apply_adjustments(adjustments)

        tuning_record = {
            "cycle": self.cycle_count,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis,
            "proposed_adjustments": adjustments,
            "applied_adjustments": applied_adjustments,
            "confidence": analysis.get("confidence", 0.0),
        }
        self.tuning_history.append(tuning_record)
        self.last_tuning_cycle = self.cycle_count
        logger.info(
            "AdaptiveTuner: tuning complete; applied %d adjustments",
            len(applied_adjustments),
        )
        return tuning_record

    def _update_performance_metrics(self, incremental: bool = False) -> None:
        """
        Update performance metrics.

        If incremental==True attempt to update EMAs using the last event only.
        Otherwise perform a robust windowed computation over recent events.
        """
        if not self.event_history:
            return

        alpha = 0.15  # EMA weight
        if incremental:
            e = self.event_history[-1]
            # binary indicators for event types
            self.crisis_frequency = _ema(
                self.crisis_frequency, 1.0 if e.event_type == "CRISIS" else 0.0, alpha
            )
            self.insight_frequency = _ema(
                self.insight_frequency, 1.0 if e.event_type == "INSIGHT" else 0.0, alpha
            )
            succ = 1.0 if e.event_type in ("SUCCESS", "EXITO_TOTAL") else 0.0
            fail = (
                1.0
                if e.event_type in ("FAILURE", "FALLO_CRITICO", "FALLO_MENOR")
                else 0.0
            )
            # update success_rate only when outcome is observable
            if succ or fail:
                self.success_rate = _ema(
                    self.success_rate, succ / max(1.0, succ + fail), alpha
                )
            # failure type breakdown
            if fail:
                mind_type = e.context.get("mind_type")
                if mind_type == "alfa":
                    self.alfa_failure_rate = _ema(self.alfa_failure_rate, 1.0, alpha)
                elif mind_type == "omega":
                    self.omega_failure_rate = _ema(self.omega_failure_rate, 1.0, alpha)
        else:
            # robust windowed computation (last N)
            N = min(len(self.event_history), 200)
            recent = list(self.event_history)[-N:]
            crises = [1 for ev in recent if ev.event_type == "CRISIS"]
            insights = [1 for ev in recent if ev.event_type == "INSIGHT"]
            successes = [
                ev for ev in recent if ev.event_type in ("SUCCESS", "EXITO_TOTAL")
            ]
            failures = [
                ev
                for ev in recent
                if ev.event_type in ("FAILURE", "FALLO_CRITICO", "FALLO_MENOR")
            ]
            self.crisis_frequency = _clamp01(len(crises) / len(recent))
            self.insight_frequency = _clamp01(len(insights) / len(recent))
            total_outcomes = len(successes) + len(failures)
            if total_outcomes > 0:
                self.success_rate = float(len(successes)) / float(total_outcomes)
            alfa_failures = [
                ev for ev in failures if ev.context.get("mind_type") == "alfa"
            ]
            omega_failures = [
                ev for ev in failures if ev.context.get("mind_type") == "omega"
            ]
            if failures:
                self.alfa_failure_rate = float(len(alfa_failures)) / float(
                    len(failures)
                )
                self.omega_failure_rate = float(len(omega_failures)) / float(
                    len(failures)
                )

    def _analyze_performance_trends(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        metrics = {
            "crisis_frequency": self.crisis_frequency,
            "insight_frequency": self.insight_frequency,
            "success_rate": self.success_rate,
            "alfa_failure_rate": self.alfa_failure_rate,
            "omega_failure_rate": self.omega_failure_rate,
        }
        analysis = {"issues": issues, "metrics": metrics, "confidence": 0.0}

        crisis_threshold = float(
            getattr(AdamConfig, "CRISIS_FREQUENCY_THRESHOLD", 0.15)
        )
        if self.crisis_frequency > crisis_threshold:
            issues.append(
                {
                    "type": "high_crisis_frequency",
                    "severity": min(1.0, self.crisis_frequency * 2.0),
                    "description": f"Crisis frequency too high: {self.crisis_frequency:.2%}",
                }
            )

        if self.success_rate < 0.6:
            issues.append(
                {
                    "type": "low_success_rate",
                    "severity": min(1.0, 1.0 - self.success_rate),
                    "description": f"Success rate too low: {self.success_rate:.2%}",
                }
            )

        alfa_thresh = float(getattr(AdamConfig, "ALFA_FAILURE_RATE_THRESHOLD", 0.6))
        if self.alfa_failure_rate > alfa_thresh:
            issues.append(
                {
                    "type": "alfa_high_failure_rate",
                    "severity": min(1.0, self.alfa_failure_rate - 0.5),
                    "description": f"Alfa failure rate high: {self.alfa_failure_rate:.2%}",
                }
            )

        omega_thresh = float(getattr(AdamConfig, "OMEGA_FAILURE_RATE_THRESHOLD", 0.6))
        if self.omega_failure_rate > omega_thresh:
            issues.append(
                {
                    "type": "omega_high_failure_rate",
                    "severity": min(1.0, self.omega_failure_rate - 0.5),
                    "description": f"Omega failure rate high: {self.omega_failure_rate:.2%}",
                }
            )

        if self.insight_frequency < 0.05:
            issues.append(
                {
                    "type": "low_insight_frequency",
                    "severity": min(1.0, 0.05 - self.insight_frequency),
                    "description": f"Insight frequency too low: {self.insight_frequency:.2%}",
                }
            )

        # confidence grows with data availability
        analysis["confidence"] = _clamp01(
            min(1.0, len(self.event_history) / max(1.0, float(self.history_window)))
        )
        return analysis

    def _calculate_adjustments(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        adjustments: list[dict[str, Any]] = []
        if analysis.get("confidence", 0.0) < self.min_confidence:
            logger.debug(
                "AdaptiveTuner: low confidence (%.2f) — skipping adjustments",
                analysis.get("confidence", 0.0),
            )
            return adjustments

        for issue in analysis.get("issues", []):
            it = issue.get("type")
            severity = float(issue.get("severity", 0.0))
            if it == "high_crisis_frequency":
                adjustments.append(
                    {
                        "parameter": "CRISIS_AROUSAL_THRESHOLD",
                        "adjustment": severity * self.adjustment_rate,
                        "reason": issue.get("description"),
                    }
                )
                adjustments.append(
                    {
                        "parameter": "CRISIS_THREAT_THRESHOLD",
                        "adjustment": severity * self.adjustment_rate,
                        "reason": issue.get("description"),
                    }
                )
            elif it == "low_success_rate":
                adjustments.append(
                    {
                        "parameter": "OMEGA_MIND_RIGOR",
                        "adjustment": severity * self.adjustment_rate * 0.5,
                        "reason": issue.get("description"),
                    }
                )
                adjustments.append(
                    {
                        "parameter": "JUDGMENT_BASE_LEARNING_RATE",
                        "adjustment": severity * self.adjustment_rate * 0.3,
                        "reason": issue.get("description"),
                    }
                )
            elif it == "alfa_high_failure_rate":
                adjustments.append(
                    {
                        "parameter": "ALFA_MIND_AUDACITY",
                        "adjustment": -severity * self.adjustment_rate,
                        "reason": issue.get("description"),
                    }
                )
                adjustments.append(
                    {
                        "parameter": "ALFA_MIND_TOLERANCE_RISK",
                        "adjustment": -severity * self.adjustment_rate * 0.5,
                        "reason": issue.get("description"),
                    }
                )
            elif it == "omega_high_failure_rate":
                adjustments.append(
                    {
                        "parameter": "OMEGA_MIND_RIGOR",
                        "adjustment": -severity * self.adjustment_rate * 0.3,
                        "reason": issue.get("description"),
                    }
                )
                adjustments.append(
                    {
                        "parameter": "OMEGA_MIND_CAUTION",
                        "adjustment": -severity * self.adjustment_rate * 0.2,
                        "reason": issue.get("description"),
                    }
                )
            elif it == "low_insight_frequency":
                adjustments.append(
                    {
                        "parameter": "ALFA_MIND_CREATIVITY",
                        "adjustment": severity * self.adjustment_rate,
                        "reason": issue.get("description"),
                    }
                )
                adjustments.append(
                    {
                        "parameter": "OPPORTUNITY_THRESHOLD_MEDIUM",
                        "adjustment": -severity * self.adjustment_rate * 0.3,
                        "reason": issue.get("description"),
                    }
                )
        return adjustments

    def _apply_adjustments(
        self, adjustments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for adj in adjustments:
            param = adj.get("parameter")
            delta = float(adj.get("adjustment", 0.0))
            reason = adj.get("reason", "")
            try:
                current = self.config.get_parameter(param)
            except Exception:
                current = None
            if current is None:
                logger.warning("AdaptiveTuner: parameter %s not found; skipping", param)
                continue
            try:
                new_value = float(current) + delta
            except Exception:
                logger.warning(
                    "AdaptiveTuner: invalid current value for %s (%r); skipping",
                    param,
                    current,
                )
                continue

            # parameter-specific bounds
            if param.endswith("_THRESHOLD") or "MIND_" in param:
                new_value = _clamp01(new_value)
            elif param == "JUDGMENT_BASE_LEARNING_RATE":
                new_value = max(0.01, min(0.5, new_value))
            elif param.endswith("_HZ"):
                new_value = max(1.0, new_value)

            success = False
            try:
                success = self.config.update_parameter(param, new_value)
            except Exception:
                logger.exception("AdaptiveTuner: exception updating param %s", param)

            if success:
                applied.append(
                    {
                        "parameter": param,
                        "old_value": current,
                        "new_value": new_value,
                        "adjustment": delta,
                        "reason": reason,
                    }
                )
                logger.info(
                    "AdaptiveTuner: applied %s: %.4f -> %.4f (%s)",
                    param,
                    float(current),
                    float(new_value),
                    reason,
                )
            else:
                logger.warning("AdaptiveTuner: failed to apply adjustment to %s", param)
        return applied

    def get_tuning_history(self, last_n: int = 10) -> list[dict[str, Any]]:
        return list(self.tuning_history[-int(last_n) :])

    def get_current_metrics(self) -> dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "last_tuning_cycle": self.last_tuning_cycle,
            "cycles_until_next_tuning": max(
                0, self.cycle_interval - (self.cycle_count - self.last_tuning_cycle)
            ),
            "event_history_size": len(self.event_history),
            "crisis_frequency": self.crisis_frequency,
            "insight_frequency": self.insight_frequency,
            "success_rate": self.success_rate,
            "alfa_failure_rate": self.alfa_failure_rate,
            "omega_failure_rate": self.omega_failure_rate,
            "total_tuning_sessions": len(self.tuning_history),
        }

    def reset_history(self) -> None:
        self.event_history.clear()
        self.tuning_history.clear()
        self.cycle_count = 0
        self.last_tuning_cycle = 0
        self.crisis_frequency = 0.0
        self.insight_frequency = 0.0
        self.success_rate = 0.5
        self.alfa_failure_rate = 0.0
        self.omega_failure_rate = 0.0
        logger.info("AdaptiveTuner: history reset")

    # --- Convenience / test helpers ---
    def simulate_tuning_run(
        self, synthetic_events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Run a simulated tuning session using synthetic events (for tests/benchmarks).
        Each item in synthetic_events: {"event_type":"...", "context":{...}, "severity":0.1}
        """
        self.reset_history()
        for e in synthetic_events:
            self.record_event(
                e.get("event_type", "UNKNOWN"),
                e.get("context", {}),
                severity=float(e.get("severity", 0.5)),
            )
        # force a full metrics recompute and run tuning
        self._update_performance_metrics(incremental=False)
        return self.tune_configuration()

    def benchmark(
        self, iterations: int = 100, crash_rate: float = 0.05
    ) -> dict[str, Any]:
        """
        Quick benchmark: generate synthetic stream and run several tuning cycles.
        Returns a summary dict.
        """
        import random

        summary = {"iterations": iterations, "applied_total": 0, "tuning_runs": 0}
        for i in range(iterations):
            ev_type = "SUCCESS" if random.random() > crash_rate else "FAILURE"
            mind = "alfa" if random.random() < 0.5 else "omega"
            self.record_event(ev_type, {"mind_type": mind}, severity=0.5)
            if self.should_tune():
                summary["tuning_runs"] += 1
                tr = self.tune_configuration()
                summary["applied_total"] += len(tr.get("applied_adjustments", []))
        return summary

    def export_history_json(self) -> str:
        try:
            return json.dumps(self.tuning_history, default=str)
        except Exception:
            logger.exception("AdaptiveTuner: export_history_json failed")
            return "[]"

    def import_history_json(self, payload: str) -> bool:
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                self.tuning_history = data
                return True
            return False
        except Exception:
            logger.exception("AdaptiveTuner: import_history_json failed")
            return False


class EVAAdaptiveTuner(AdaptiveTuner):
    """
    EVA-integrated AdaptiveTuner with best-effort RealityBytecode compilation
    and experience storage. Async-aware and non-fatal on EVA failures.
    """

    def __init__(
        self,
        config: AdamConfig,
        phase: str = "default",
        cycle_interval: int | None = None,
    ):
        super().__init__(config, cycle_interval)
        self.eva_phase = phase
        self.eva_runtime: LivingSymbolRuntime | None = None
        try:
            # lazy create minimal runtime if available
            from crisalida_lib.EDEN.living_symbol import (
                LivingSymbolRuntime as _LSR,  # type: ignore
            )

            self.eva_runtime = _LSR()  # may raise; guard below
        except Exception:
            self.eva_runtime = None

        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []

    def _compile_intention_maybe_async(self, intention: dict[str, Any]) -> list[Any]:
        """
        Helper: compile intention via eager coroutine detection.
        Returns compiled bytecode list (possibly empty).
        """
        _eva = getattr(self, "eva_runtime", None)
        bytecode = []
        try:
            compiler = getattr(_eva, "divine_compiler", None) if _eva else None
            if compiler is None:
                return []
            compile_fn = getattr(compiler, "compile_intention", None)
            if compile_fn is None:
                return []
            if inspect.iscoroutinefunction(compile_fn):
                # If there's a running loop schedule it, otherwise run synchronously
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(compile_fn(intention))
                    # do not await — return placeholder to indicate scheduled work
                    return []
                except RuntimeError:
                    # no running loop
                    return asyncio.run(compile_fn(intention))
            else:
                return compile_fn(intention)
        except Exception:
            logger.exception("EVAAdaptiveTuner: _compile_intention_maybe_async failed")
            return bytecode

    def eva_ingest_tuning_experience(
        self,
        tuning_record: dict[str, Any],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=float(
                tuning_record.get("analysis", {})
                .get("metrics", {})
                .get("success_rate", 0.5)
            ),
            cognitive_complexity=float(
                tuning_record.get("analysis", {})
                .get("metrics", {})
                .get("insight_frequency", 0.5)
            ),
            consciousness_density=float(
                tuning_record.get("analysis", {})
                .get("metrics", {})
                .get("crisis_frequency", 0.5)
            ),
            narrative_importance=float(tuning_record.get("confidence", 0.5)),
            energy_level=1.0,
        )
        # deterministic id based on record content and time
        experience_id = f"eva_tuning_{hashlib.sha1(json.dumps(tuning_record, sort_keys=True).encode('utf-8')).hexdigest()[:10]}_{int(time.time())}"
        experience_data = {
            "tuning_record": tuning_record,
            "metrics": tuning_record.get("analysis", {}).get("metrics", {}),
            "proposed_adjustments": tuning_record.get("proposed_adjustments", []),
            "applied_adjustments": tuning_record.get("applied_adjustments", []),
            "confidence": tuning_record.get("confidence", 0.5),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_TUNING_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self._compile_intention_maybe_async(intention) or []
        # Best-effort construct RealityBytecode if type is available; otherwise store raw dict
        try:
            rb = RealityBytecode(bytecode_id=experience_id, instructions=bytecode, qualia_state=qualia_state, phase=phase, timestamp=experience_data["timestamp"])  # type: ignore
        except Exception:
            rb = experience_data  # permissive fallback
        self.eva_memory_store[experience_id] = rb
        self.eva_experience_store[experience_id] = rb
        self.eva_phases.setdefault(phase, {})[experience_id] = rb
        for hook in list(self._environment_hooks):
            try:
                hook(rb)
            except Exception:
                logger.warning(
                    "EVAAdaptiveTuner: environment hook failed during ingest",
                    exc_info=True,
                )
        return experience_id

    def eva_recall_tuning_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        phase = phase or self.eva_phase
        rb = self.eva_phases.get(phase, {}).get(cue) or self.eva_memory_store.get(cue)
        if not rb:
            return {"error": "No bytecode found for EVA tuning experience"}
        manifestations: list[Any] = []
        quantum_field = (
            getattr(self.eva_runtime, "quantum_field", None)
            if self.eva_runtime
            else None
        )
        if quantum_field and isinstance(getattr(rb, "instructions", None), list):
            for instr in rb.instructions:  # type: ignore
                try:
                    if self.eva_runtime and getattr(
                        self.eva_runtime, "execute_instruction", None
                    ):
                        symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)  # type: ignore
                    else:
                        symbol_manifest = None
                    if symbol_manifest:
                        manifestations.append(symbol_manifest)
                        for hook in list(self._environment_hooks):
                            try:
                                hook(symbol_manifest)
                            except Exception:
                                logger.warning(
                                    "EVAAdaptiveTuner: manifestation hook failed",
                                    exc_info=True,
                                )
                except Exception:
                    logger.exception("EVAAdaptiveTuner: executing instruction failed")
        # Construct a permissive EVAExperience-like dict for callers
        response = {
            "experience_id": getattr(rb, "bytecode_id", cue),
            "manifestations": [
                getattr(m, "to_dict", lambda: m)() for m in manifestations
            ],
            "phase": getattr(rb, "phase", phase),
            "qualia_state": (
                getattr(rb, "qualia_state", {}).to_dict()
                if hasattr(getattr(rb, "qualia_state", {}), "to_dict")
                else {}
            ),
            "timestamp": getattr(rb, "timestamp", time.time()),
        }
        # cache the EVAExperience if type exists
        try:
            eva_exp = EVAExperience(experience_id=response["experience_id"], bytecode=rb, manifestations=manifestations, phase=response["phase"], qualia_state=response["qualia_state"], timestamp=response["timestamp"])  # type: ignore
            self.eva_experience_store[response["experience_id"]] = eva_exp
        except Exception:
            self.eva_experience_store[response["experience_id"]] = response
        return response

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        tuning_record: dict[str, Any],
        qualia_state: QualiaState,
    ) -> None:
        experience_data = {
            "tuning_record": tuning_record,
            "metrics": tuning_record.get("analysis", {}).get("metrics", {}),
            "proposed_adjustments": tuning_record.get("proposed_adjustments", []),
            "applied_adjustments": tuning_record.get("applied_adjustments", []),
            "confidence": tuning_record.get("confidence", 0.5),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_TUNING_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self._compile_intention_maybe_async(intention) or []
        try:
            rb = RealityBytecode(bytecode_id=experience_id, instructions=bytecode, qualia_state=qualia_state, phase=phase, timestamp=experience_data["timestamp"])  # type: ignore
            self.eva_phases.setdefault(phase, {})[experience_id] = rb
        except Exception:
            self.eva_phases.setdefault(phase, {})[experience_id] = experience_data

    def set_memory_phase(self, phase: str) -> None:
        self.eva_phase = phase
        for hook in list(self._environment_hooks):
            try:
                hook({"phase_changed": phase})
            except Exception:
                logger.warning("EVAAdaptiveTuner: phase hook failed", exc_info=True)

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list[str]:
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]) -> None:
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        return {
            "eva_ingest_tuning_experience": self.eva_ingest_tuning_experience,
            "eva_recall_tuning_experience": self.eva_recall_tuning_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
