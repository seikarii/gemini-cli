"""
Consciousness Metrics System (definitive, professionalized)
==========================================================

- Defensive numeric backend (numpy preferred, pure-Python fallback).
- TYPE_CHECKING guarded imports to avoid cycles.
- Robust baseline handling, EMA smoothing, rolling-window analyses.
- Async-aware EVA persistence (best-effort): schedules coroutine if needed.
- Clear public API: collect_report, reset_baseline, export/import history, simulate run.
- Reasonable defaults and tunables for evolution and production use.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Defensive numeric backend
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore
    HAS_NUMPY = False

# TYPE_CHECKING imports to avoid circular deps at runtime
if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import (
        AdamConfig,
    )  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,
    )  # [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    from crisalida_lib.EVA.typequalia import (
        QualiaState,
    )  # [`crisalida_lib.EVA.typequalia.QualiaState`](crisalida_lib/EVA/typequalia.py)
else:
    AdamConfig = Any
    EVAMemoryManager = Any
    QualiaState = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Helpers ---
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _ema(prev: float | None, value: float, alpha: float = 0.2) -> float:
    if prev is None:
        return float(value)
    return float((1 - alpha) * prev + alpha * value)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    try:
        if HAS_NUMPY:
            return float(np.mean(values))
        return float(sum(values) / len(values))
    except Exception:
        return 0.0


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    try:
        if HAS_NUMPY:
            return float(np.std(values))
        mean = sum(values) / len(values)
        return float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))
    except Exception:
        return 0.0


# --- Data containers ---


class MetricsHistory:
    """Stores historical data for trend analysis and provides windowed views."""

    def __init__(self, max_len: int = 2000) -> None:
        self.max_len = int(max_len)
        self.history: deque[dict[str, Any]] = deque(maxlen=self.max_len)

    def add_measurement(
        self,
        timestamp: float,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.history.append(
            {
                "timestamp": float(timestamp),
                "metrics": dict(metrics or {}),
                "metadata": metadata or {},
            }
        )

    def get_recent_measurements(self, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        return list(self.history)[-int(min(count, len(self.history))) :]

    def export_json(self) -> str:
        try:
            return json.dumps(list(self.history), default=str)
        except Exception:
            logger.exception("MetricsHistory: export_json failed")
            return "[]"

    def import_json(self, payload: str) -> bool:
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                self.history = deque(data, maxlen=self.max_len)
                return True
            return False
        except Exception:
            logger.exception("MetricsHistory: import_json failed")
            return False

    def clear(self) -> None:
        self.history.clear()


class TrendAnalyzer:
    """Analyzes trends using simple moving averages and optional linear slope."""

    def analyze_trends(
        self, measurements: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        if len(measurements) < 3:
            return {"positive_trends": [], "negative_trends": []}

        # Build per-key series
        series: dict[str, list[float]] = {}
        for m in measurements:
            for k, v in m.get("metrics", {}).items():
                series.setdefault(k, []).append(float(v))

        positive: list[str] = []
        negative: list[str] = []
        for k, vals in series.items():
            if len(vals) < 3:
                continue
            recent = vals[-3:]
            avg_all = _safe_mean(vals)
            avg_recent = _safe_mean(recent)
            if avg_recent > avg_all * 1.08:
                positive.append(k)
            elif avg_recent < avg_all * 0.92:
                negative.append(k)
            else:
                # small slope check
                try:
                    slope = (recent[-1] - recent[0]) / max(1.0, len(recent) - 1)
                    if slope > 1e-6:
                        positive.append(k)
                    elif slope < -1e-6:
                        negative.append(k)
                except Exception:
                    pass
        return {"positive_trends": positive, "negative_trends": negative}


class ConsciousnessAnomalyDetector:
    """Detects anomalies in the stream using rolling z-score with robust fallback."""

    def detect_anomalies(
        self,
        current_metrics: dict[str, float],
        metrics_history: MetricsHistory,
        window: int = 120,
    ) -> dict[str, list[str]]:
        anomalies: list[str] = []
        hist = metrics_history.get_recent_measurements(window)
        if len(hist) < 10:
            return {"detected_anomalies": []}

        # build histories
        per_key: dict[str, list[float]] = {}
        for h in hist:
            for k, v in h.get("metrics", {}).items():
                per_key.setdefault(k, []).append(float(v))

        for k, cur in current_metrics.items():
            vals = per_key.get(k, [])
            if not vals or len(vals) < 6:
                continue
            mean = _safe_mean(vals)
            std = _safe_std(vals)
            if std and abs(cur - mean) > 3.0 * std:
                anomalies.append(
                    f"Anomaly in {k}: value={cur:.3f} mean={mean:.3f} std={std:.3f}"
                )
            # also detect sustained drift > 20% over window
            try:
                if mean > 1e-9 and abs(cur - mean) / mean > 0.2:
                    anomalies.append(
                        f"Drift in {k}: current is {((cur/mean)-1)*100:.1f}% from historical mean"
                    )
            except Exception:
                pass

        return {"detected_anomalies": anomalies}


@dataclass
class ComprehensiveConsciousnessMetricsReport:
    timestamp: float
    metrics: dict[str, Any]
    baseline_comparison: dict[str, Any]
    trend_analysis: dict[str, Any]
    anomaly_analysis: dict[str, Any]
    alerts: list[str]
    recommendations: list[str]


# --- Main system ---


class ConsciousnessMetrics:
    """Complete consciousness metrics system with EVA persistence and utilities."""

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
        consciousness_core: Any | None = None,
        history_len: int = 2000,
    ) -> None:
        self.config = config
        self.eva_manager = eva_manager
        self.entity_id = str(entity_id)
        self.consciousness_core = consciousness_core
        self.metrics_history = MetricsHistory(max_len=history_len)
        self.baseline_metrics: dict[str, float] | None = None
        self.alert_thresholds = self._set_comprehensive_alert_thresholds()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = ConsciousnessAnomalyDetector()
        # EMA smoothing for key aggregates
        self.ema_state: dict[str, float] = {}
        # Tunables
        self.ema_alpha = float(getattr(self.config, "CONSCIOUSNESS_EMA_ALPHA", 0.15))
        self.min_history_for_baseline = int(
            getattr(self.config, "CONSCIOUSNESS_MIN_HISTORY_BASELINE", 20)
        )

        logger.info("ConsciousnessMetrics initialized (entity_id=%s)", self.entity_id)

    def _set_comprehensive_alert_thresholds(self) -> dict[str, float]:
        return {
            "coherence_level_critical": 0.3,
            "coherence_level_warning": 0.5,
            "emotional_stability_critical": 0.2,
            "emotional_stability_warning": 0.4,
            "cognitive_clarity_critical": 0.3,
            "cognitive_clarity_warning": 0.5,
        }

    def _capture_snapshot(self) -> dict[str, float]:
        """Safely capture a metrics snapshot from the consciousness_core (best-effort)."""
        snapshot = {}
        try:
            if self.consciousness_core and hasattr(
                self.consciousness_core, "current_qualia"
            ):
                q = self.consciousness_core.current_qualia
                # Prefer explicit API if present
                if hasattr(q, "_capture_current_state"):
                    snapshot = dict(q._capture_current_state() or {})
                elif hasattr(q, "get_state"):
                    snapshot = dict(q.get_state() or {})
                else:
                    # Best-effort introspection
                    snapshot = {
                        k: getattr(q, k)
                        for k in ("coherence", "cognitive_clarity", "emotional_valence")
                        if hasattr(q, k)
                    }
        except Exception:
            logger.exception("Failed to capture consciousness snapshot (non-fatal)")
            snapshot = {}

        # Normalize numeric keys into expected metric names
        metrics = {
            "coherence_level": float(
                snapshot.get("coherence", snapshot.get("coherence_level", 0.0)) or 0.0
            ),
            "complexity_index": float(
                snapshot.get("complexity_index", snapshot.get("cognitive_clarity", 0.0))
                or 0.0
            ),
            "emotional_stability": float(
                1.0 - abs(float(snapshot.get("emotional_valence", 0.0) or 0.0))
            ),
            "cognitive_clarity": float(snapshot.get("cognitive_clarity", 0.0) or 0.0),
        }
        # clamp sensible ranges
        for k in list(metrics.keys()):
            metrics[k] = _clamp01(metrics[k])

        return metrics

    def collect_comprehensive_consciousness_metrics(
        self,
    ) -> ComprehensiveConsciousnessMetricsReport:
        timestamp = time.time()
        current_metrics = self._capture_snapshot()

        # update EMA state
        for k, v in current_metrics.items():
            prev = self.ema_state.get(k)
            self.ema_state[k] = _ema(prev, float(v), alpha=self.ema_alpha)

        # add to history
        self.metrics_history.add_measurement(
            timestamp, current_metrics, {"source": "consciousness_core"}
        )

        # initialize baseline after minimum history is reached
        if (
            self.baseline_metrics is None
            and len(self.metrics_history.history) >= self.min_history_for_baseline
        ):
            # baseline = median of first N measurements (robust)
            recent = self.metrics_history.get_recent_measurements(
                self.min_history_for_baseline
            )
            baseline: dict[str, float] = {}
            keys = set()
            for m in recent:
                keys.update(m.get("metrics", {}).keys())
            for k in keys:
                vals = [float(m["metrics"].get(k, 0.0)) for m in recent]
                if vals:
                    try:
                        if HAS_NUMPY:
                            baseline[k] = float(np.median(vals))
                        else:
                            sorted_v = sorted(vals)
                            mid = len(sorted_v) // 2
                            baseline[k] = float(
                                sorted_v[mid]
                                if len(sorted_v) % 2 == 1
                                else (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
                            )
                    except Exception:
                        baseline[k] = _safe_mean(vals)
                else:
                    baseline[k] = 0.0
            self.baseline_metrics = baseline
            logger.info("Baseline metrics established: %s", self.baseline_metrics)

        trend_analysis = self.trend_analyzer.analyze_trends(
            self.metrics_history.get_recent_measurements(100)
        )
        anomaly_analysis = self.anomaly_detector.detect_anomalies(
            current_metrics, self.metrics_history
        )
        alerts = self._generate_comprehensive_alerts(current_metrics)
        recommendations = self._generate_improvement_recommendations(
            current_metrics, trend_analysis, anomaly_analysis
        )
        baseline_comparison = self._compare_to_baseline(current_metrics)

        report = ComprehensiveConsciousnessMetricsReport(
            timestamp=timestamp,
            metrics=current_metrics,
            baseline_comparison=baseline_comparison,
            trend_analysis=trend_analysis,
            anomaly_analysis=anomaly_analysis,
            alerts=alerts,
            recommendations=recommendations,
        )

        # EVA persistence (best-effort; async-aware)
        if self.eva_manager is not None:
            try:
                record_fn = getattr(self.eva_manager, "record_experience", None)
                if callable(record_fn):
                    payload = {
                        "timestamp": report.timestamp,
                        "metrics": report.metrics,
                        "alerts": report.alerts,
                        "recommendations": report.recommendations,
                        "trend_analysis": report.trend_analysis,
                        "anomaly_analysis": report.anomaly_analysis,
                    }
                    if inspect.iscoroutinefunction(record_fn):
                        try:
                            # schedule background task if loop is running
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                record_fn(
                                    entity_id=self.entity_id,
                                    event_type="consciousness_metrics_report",
                                    data=payload,
                                )
                            )
                        except RuntimeError:
                            # no running loop; run synchronously
                            try:
                                asyncio.run(
                                    record_fn(
                                        entity_id=self.entity_id,
                                        event_type="consciousness_metrics_report",
                                        data=payload,
                                    )
                                )
                            except Exception:
                                logger.debug("EVA async record failed silently")
                    else:
                        try:
                            record_fn(
                                entity_id=self.entity_id,
                                event_type="consciousness_metrics_report",
                                data=payload,
                            )
                        except Exception:
                            logger.debug("EVA sync record failed silently")
            except Exception:
                logger.exception(
                    "EVA integration failed while recording consciousness metrics (non-fatal)"
                )

        return report

    def _generate_comprehensive_alerts(self, metrics: dict[str, float]) -> list[str]:
        alerts: list[str] = []
        # Use mapping to evaluate critical/warning thresholds
        mapping = {
            "coherence_level": ("coherence_level_critical", "coherence_level_warning"),
            "emotional_stability": (
                "emotional_stability_critical",
                "emotional_stability_warning",
            ),
            "cognitive_clarity": (
                "cognitive_clarity_critical",
                "cognitive_clarity_warning",
            ),
        }
        for metric, th_keys in mapping.items():
            crit_key, warn_key = th_keys
            crit = self.alert_thresholds.get(crit_key)
            warn = self.alert_thresholds.get(warn_key)
            val = float(metrics.get(metric, 0.0))
            if crit is not None and val < crit:
                alerts.append(
                    f"CRITICAL: {metric} below critical threshold ({val:.2f} < {crit:.2f})"
                )
            elif warn is not None and val < warn:
                alerts.append(
                    f"WARNING: {metric} below warning threshold ({val:.2f} < {warn:.2f})"
                )
        return alerts

    def _compare_to_baseline(self, current_metrics: dict[str, float]) -> dict[str, Any]:
        if not self.baseline_metrics:
            return {}
        comparison: dict[str, Any] = {}
        for k, v in current_metrics.items():
            base = float(self.baseline_metrics.get(k, 0.0))
            if base:
                ratio = v / base
                comparison[k] = {
                    "current": v,
                    "baseline": base,
                    "ratio": ratio,
                    "percent_of_baseline": ratio * 100.0,
                }
            else:
                comparison[k] = {
                    "current": v,
                    "baseline": base,
                    "note": "baseline zero",
                }
        return comparison

    def _generate_improvement_recommendations(
        self,
        metrics: dict[str, float],
        trends: dict[str, Any],
        anomalies: dict[str, Any],
    ) -> list[str]:
        recs: list[str] = []
        if trends.get("negative_trends"):
            recs.append(
                f"Address negative trends: {', '.join(trends['negative_trends'])}"
            )
        if anomalies.get("detected_anomalies"):
            recs.append(
                f"Investigate anomalies: {', '.join(anomalies['detected_anomalies'])}"
            )
        if metrics.get("coherence_level", 0.0) < 0.4:
            recs.append(
                "Increase coherence interventions: reduce conflicting inputs and allow consolidation cycles (dream/sleep)."
            )
        return recs

    # --- Utilities & public helpers ---

    def reset_baseline(self) -> None:
        self.baseline_metrics = None
        logger.info("ConsciousnessMetrics: baseline reset")

    def export_history_json(self) -> str:
        return self.metrics_history.export_json()

    def import_history_json(self, payload: str) -> bool:
        return self.metrics_history.import_json(payload)

    def simulate_tuning_run(
        self, synthetic_snapshots: list[dict[str, Any]]
    ) -> ComprehensiveConsciousnessMetricsReport:
        """
        Helper for tests: feed a sequence of synthetic snapshots (dicts with metric keys)
        and return the last generated report.
        """
        self.metrics_history.clear()
        self.baseline_metrics = None
        last_report: ComprehensiveConsciousnessMetricsReport | None = None
        for snap in synthetic_snapshots:
            # create a fake consciousness_core wrapper for capture
            class _Fake:
                current_qualia = None

            fake = _Fake()
            # create a minimal structure expected by _capture_snapshot
            fake.current_qualia = type(
                "Q", (), {"_capture_current_state": lambda self, s=snap: s}
            )()
            self.consciousness_core = fake
            last_report = self.collect_comprehensive_consciousness_metrics()
        if last_report is None:
            # produce an empty default
            last_report = ComprehensiveConsciousnessMetricsReport(
                time.time(), {}, {}, {}, {}, [], []
            )
        return last_report

    def get_current_ema_state(self) -> dict[str, float]:
        return dict(self.ema_state)

    def get_current_metrics_snapshot(self) -> dict[str, Any]:
        recent = self.metrics_history.get_recent_measurements(1)
        return recent[0] if recent else {}
