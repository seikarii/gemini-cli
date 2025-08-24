"""
Dialectical Processor System (professionalized)
===============================================

- Defensive numeric backend (numpy optional).
- TYPE_CHECKING guarded imports to avoid circular deps at runtime.
- Async-aware EVA persistence (best-effort scheduling when coroutine functions are exposed).
- Robust baseline handling, trend bookkeeping, export/import utilities.
- Clear public API: process_perception, reset_state, export/import memory, simulate_perception.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, cast

# Defensive numeric backend pattern
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False

# TYPE_CHECKING guarded imports to avoid circular runtime deps
if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import (
        AdamConfig,
    )  # [`crisalida_lib/ADAM/config.AdamConfig`](crisalida_lib/ADAM/config.py)
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,
    )  # [`crisalida_lib/ADAM/eva_integration/eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    from crisalida_lib.EARTH.core_types.data_models import (
        PatternRecognitionResult,
        PerceptionData,
    )  # [`crisalida_lib/EARTH/core_types/data_models.PatternRecognitionResult`](crisalida_lib/EARTH/core_types/data_models.py) [`crisalida_lib/EARTH/core_types/data_models.PerceptionData`](crisalida_lib/EARTH/core_types/data_models.py)
    from crisalida_lib.EARTH.core_types.enums import (
        QliphothNode,
        SephirotNode,
    )  # [`crisalida_lib/EARTH/core_types/enums.SephirotNode`](crisalida_lib/EARTH/core_types/enums.py) [`crisalida_lib/EARTH/core_types/enums.QliphothNode`](crisalida_lib/EARTH/core_types/enums.py)
    from crisalida_lib.EVA.eva_memory_mixin import (
        EVAMemoryMixin,
    )  # [`crisalida_lib/EVA/eva_memory_mixin.EVAMemoryMixin`](crisalida_lib/EVA/eva_memory_mixin.py)
else:
    AdamConfig = Any
    EVAMemoryManager = Any
    SephirotNode = Any
    QliphothNode = Any
    PatternRecognitionResult = Any
    PerceptionData = Any
    EVAMemoryMixin = object

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Helpers ---


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _safe_mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    try:
        if HAS_NUMPY:
            return float(np.mean(vals))
        return float(sum(vals) / len(vals))
    except Exception:
        return 0.0


# --- Dialectical Processor ---


class ProcesadorDialectico(EVAMemoryMixin):  # type: ignore[misc]
    """Procesador dialéctico avanzado que analiza percepciones y mantiene memoria de patrones."""

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        # core deps
        self.config = config
        self.eva_manager = eva_manager
        self.entity_id = str(entity_id)

        # Initialize EVA memory mixin (best-effort)
        try:
            # mixin may implement _init_eva_memory; call defensively
            cast(Any, super())._init_eva_memory(
                eva_runtime=(eva_manager.eva_runtime if eva_manager else None)
            )
        except Exception:
            # ignore if mixin not available at runtime
            pass

        # Node activations
        self.sephirot_activation: dict[SephirotNode, float] = dict.fromkeys(
            SephirotNode if SephirotNode is not Any else [], 0.0
        )
        self.qliphoth_activation: dict[QliphothNode, float] = dict.fromkeys(
            QliphothNode if QliphothNode is not Any else [], 0.0
        )

        # Connection graphs (lazy init)
        self.sephirot_connections: dict[Any, dict[Any, float]] = (
            self._initialize_sephirot_connections()
        )
        self.qliphoth_connections: dict[Any, dict[Any, float]] = (
            self._initialize_qliphoth_connections()
        )

        # Pattern memory and templates
        self.pattern_memory: dict[str, dict[str, Any]] = {}
        self.pattern_templates: dict[str, dict[str, Any]] = (
            self._initialize_pattern_templates()
        )

        # thresholds and bookkeeping
        self.activation_threshold: float = float(
            getattr(config, "DIALECTICAL_ACTIVATION_THRESHOLD", 0.3)
        )
        self.resonance_threshold: float = float(
            getattr(config, "DIALECTICAL_RESONANCE_THRESHOLD", 0.5)
        )
        self.processing_history: deque = deque(
            maxlen=int(getattr(config, "DIALECTICAL_HISTORY_LEN", 1000))
        )

        logger.info("ProcesadorDialectico initialized (entity_id=%s)", self.entity_id)

    # --- Initialization helpers (kept defensive & minimal) ---

    def _initialize_sephirot_connections(self) -> dict[Any, dict[Any, float]]:
        """Return a pre-wired Sephirot connection map; tolerant to missing enums."""
        try:
            return self._initialize_sephirot_connections_impl()
        except Exception:
            # fallback: small neutral graph
            return {}

    def _initialize_sephirot_connections_impl(self) -> dict[Any, dict[Any, float]]:
        connections: dict[Any, dict[Any, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        try:
            connections[SephirotNode.KETHER][SephirotNode.TIPHERETH] = 0.8
            connections[SephirotNode.TIPHERETH][SephirotNode.YESOD] = 0.8
            connections[SephirotNode.YESOD][SephirotNode.MALKUTH] = 0.8
            connections[SephirotNode.CHOKMAH][SephirotNode.CHESED] = 0.7
            connections[SephirotNode.BINAH][SephirotNode.GEBURAH] = 0.7
            # bidirectional
            for s, ts in list(connections.items()):
                for t, v in ts.items():
                    connections[t][s] = v
        except Exception:
            logger.debug("Sephirot connection init partial/fallback", exc_info=True)
        return dict(connections)

    def _initialize_qliphoth_connections(self) -> dict[Any, dict[Any, float]]:
        try:
            return self._initialize_qliphoth_connections_impl()
        except Exception:
            return {}

    def _initialize_qliphoth_connections_impl(self) -> dict[Any, dict[Any, float]]:
        connections: dict[Any, dict[Any, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        try:
            connections[QliphothNode.THAUMIEL][QliphothNode.SAMAEL] = 0.7
            connections[QliphothNode.GOLACHAB][QliphothNode.LILITH] = 0.6
            for s, ts in list(connections.items()):
                for t, v in ts.items():
                    connections[t][s] = v * 0.8
        except Exception:
            logger.debug("Qliphoth connection init partial/fallback", exc_info=True)
        return dict(connections)

    def _initialize_pattern_templates(self) -> dict[str, dict[str, Any]]:
        # re-use a stable set of templates; can be extended by import or EVA updates
        return {
            "threat_pattern": {
                "sephirot_signature": {},
                "qliphoth_signature": {},
                "emotional_markers": {"fear": 0.7},
            },
            "creative_pattern": {
                "sephirot_signature": {},
                "qliphoth_signature": {},
                "emotional_markers": {"excitement": 0.6},
            },
        }

    # --- Public API ---

    def reset_state(self, preserve_memory: bool = True) -> None:
        """Reset transient activations and optionally keep learned pattern memory."""
        for k in list(self.sephirot_activation.keys()):
            self.sephirot_activation[k] = 0.0
        for k in list(self.qliphoth_activation.keys()):
            self.qliphoth_activation[k] = 0.0
        self.processing_history.clear()
        if not preserve_memory:
            self.pattern_memory.clear()
        logger.info(
            "ProcesadorDialectico state reset (preserve_memory=%s)",
            bool(preserve_memory),
        )

    def export_pattern_memory_json(self) -> str:
        try:
            return json.dumps(self.pattern_memory, default=str)
        except Exception:
            logger.exception("export_pattern_memory_json failed")
            return "{}"

    def import_pattern_memory_json(self, payload: str) -> bool:
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                self.pattern_memory.update(data)
                return True
            return False
        except Exception:
            logger.exception("import_pattern_memory_json failed")
            return False

    async def process_perception(
        self, perception_data: PerceptionData
    ) -> PatternRecognitionResult:
        """
        Primary entry point to process perception data.

        This method is async to accommodate future expansions (LLM calls, external tools).
        """
        # 1) Activation
        try:
            self._activate_nodes_from_data(perception_data)
        except Exception:
            logger.exception("Activation failed (recoverable)")

        # 2) Propagation
        try:
            self._propagate_activations()
        except Exception:
            logger.exception("Propagation failed (recoverable)")

        # 3) Recognition
        try:
            pattern_result = self._recognize_patterns(perception_data)
        except Exception:
            logger.exception("Recognition failed; returning safe default")
            pattern_result = PatternRecognitionResult(  # type: ignore[call-arg]
                pattern_type="unknown",
                confidence=0.0,
                complexity=0.0,
                archetypes_activated=[],
                semantic_features={},
                emotional_valence=0.0,
                temporal_coherence=0.5,
            )

        # 4) Synthesis & memory update
        try:
            synthesis = self._perform_dialectical_synthesis()
            self._update_pattern_memory(pattern_result, perception_data)
        except Exception:
            logger.exception("Synthesis/memory update failed (non-fatal)")

        # 5) Persist summary to EVA (best-effort, async-aware)
        try:
            await self._maybe_record_eva_event(
                {
                    "perception_summary": str(perception_data.raw_data)[:200],
                    "pattern_type": getattr(pattern_result, "pattern_type", "unknown"),
                    "confidence": getattr(pattern_result, "confidence", 0.0),
                }
            )
        except Exception:
            logger.debug("EVA recording attempted and failed (non-fatal)")

        # 6) Append history & return
        try:
            self.processing_history.append(
                {
                    "timestamp": getattr(perception_data, "timestamp", time.time()),
                    "pattern": getattr(pattern_result, "pattern_type", "unknown"),
                    "confidence": getattr(pattern_result, "confidence", 0.0),
                }
            )
        except Exception:
            pass

        return pattern_result

    # --- Core algorithmic helpers (kept compact and robust) ---

    def _activate_nodes_from_data(self, perception_data: PerceptionData) -> None:
        raw = getattr(perception_data, "raw_data", {}) or {}
        novelty = float(raw.get("novelty", 0.0) or 0.0)
        threat = float(raw.get("threat_level", 0.0) or 0.0)
        complexity = float(raw.get("complexity", 0.0) or 0.0)
        harmony = float(raw.get("harmony", 0.0) or 0.0)
        chaos = float(raw.get("chaos_level", 0.0) or 0.0)

        # Positive activations
        try:
            if novelty > 0.5 and SephirotNode is not Any:
                self.sephirot_activation[SephirotNode.CHOKMAH] = _clamp01(
                    self.sephirot_activation.get(SephirotNode.CHOKMAH, 0.0)
                    + novelty * 0.7
                )
            if harmony > 0.5 and SephirotNode is not Any:
                self.sephirot_activation[SephirotNode.TIPHERETH] = _clamp01(
                    self.sephirot_activation.get(SephirotNode.TIPHERETH, 0.0)
                    + harmony * 0.8
                )
            if complexity > 0.7 and SephirotNode is not Any:
                self.sephirot_activation[SephirotNode.BINAH] = _clamp01(
                    self.sephirot_activation.get(SephirotNode.BINAH, 0.0)
                    + complexity * 0.6
                )
        except Exception:
            logger.debug("Sephirot activation partial failed", exc_info=True)

        # Negative / shadow activations
        try:
            if threat > 0.3 and QliphothNode is not Any:
                self.qliphoth_activation[QliphothNode.GOLACHAB] = _clamp01(
                    self.qliphoth_activation.get(QliphothNode.GOLACHAB, 0.0)
                    + threat * 0.8
                )
                self.qliphoth_activation[QliphothNode.SAMAEL] = _clamp01(
                    self.qliphoth_activation.get(QliphothNode.SAMAEL, 0.0)
                    + threat * 0.6
                )
            if chaos > 0.5 and QliphothNode is not Any:
                self.qliphoth_activation[QliphothNode.THAUMIEL] = _clamp01(
                    self.qliphoth_activation.get(QliphothNode.THAUMIEL, 0.0)
                    + chaos * 0.7
                )
        except Exception:
            logger.debug("Qliphoth activation partial failed", exc_info=True)

        self._normalize_activations()

    def _propagate_activations(self) -> None:
        new_seph = dict(self.sephirot_activation)
        for src, act in list(self.sephirot_activation.items()):
            if act > self.activation_threshold and src in self.sephirot_connections:
                for tgt, strength in self.sephirot_connections[src].items():
                    new_seph[tgt] = _clamp01(
                        new_seph.get(tgt, 0.0) + act * strength * 0.3
                    )
        new_qlip = dict(self.qliphoth_activation)
        for src, act in list(self.qliphoth_activation.items()):
            if src in self.qliphoth_connections:
                for tgt, strength in self.qliphoth_connections[src].items():
                    new_qlip[tgt] = _clamp01(
                        new_qlip.get(tgt, 0.0) + act * strength * 0.4
                    )
        self.sephirot_activation = new_seph
        self.qliphoth_activation = new_qlip

    def _normalize_activations(self) -> None:
        # clamp all known members
        try:
            if SephirotNode is not Any:
                for n in SephirotNode:
                    self.sephirot_activation[n] = _clamp01(
                        self.sephirot_activation.get(n, 0.0)
                    )
            if QliphothNode is not Any:
                for n in QliphothNode:
                    self.qliphoth_activation[n] = _clamp01(
                        self.qliphoth_activation.get(n, 0.0)
                    )
        except Exception:
            logger.debug("Normalization fallback used", exc_info=True)

    def _calculate_pattern_confidence(self, template: dict[str, Any]) -> float:
        seph_match = 0.0
        seph_sig = template.get("sephirot_signature", {}) or {}
        if seph_sig and SephirotNode is not Any:
            for node, expected in seph_sig.items():
                actual = self.sephirot_activation.get(node, 0.0)
                seph_match += 1.0 - abs(float(expected) - float(actual))
            seph_match /= max(1, len(seph_sig))
        qlip_match = 0.0
        qliph_sig = template.get("qliphoth_signature", {}) or {}
        if qliph_sig and QliphothNode is not Any:
            for node, expected in qliph_sig.items():
                actual = self.qliphoth_activation.get(node, 0.0)
                qlip_match += 1.0 - abs(float(expected) - float(actual))
            qlip_match /= max(1, len(qliph_sig))
        if seph_sig and qliph_sig:
            return float(_clamp01(seph_match * 0.6 + qlip_match * 0.4))
        if seph_sig:
            return float(_clamp01(seph_match))
        return float(_clamp01(qlip_match))

    def _calculate_perception_complexity(self) -> float:
        total_seph = _safe_mean(list(self.sephirot_activation.values()) or [0.0])
        total_qlip = _safe_mean(list(self.qliphoth_activation.values()) or [0.0])
        diversity = len(
            [
                v
                for v in list(self.sephirot_activation.values())
                + list(self.qliphoth_activation.values())
                if v > 0.1
            ]
        )
        raw = (total_seph + total_qlip) / 2.0
        diversity_factor = min(1.0, float(diversity) / 15.0)
        return float(_clamp01(raw * 0.7 + diversity_factor * 0.3))

    def _identify_activated_archetypes(self) -> list[str]:
        archetypes: list[str] = []
        try:
            if self.sephirot_activation.get(SephirotNode.KETHER, 0.0) > 0.7:
                archetypes.append("Divine_Crown")
            if self.qliphoth_activation.get(QliphothNode.LILITH, 0.0) > 0.6:
                archetypes.append("Night_Queen")
        except Exception:
            pass
        return archetypes

    def _extract_semantic_features(
        self, perception_data: PerceptionData
    ) -> dict[str, float]:
        raw = getattr(perception_data, "raw_data", {}) or {}
        features: dict[str, float] = {
            "spatial_coherence": float(raw.get("spatial_coherence", 0.5) or 0.5),
            "temporal_continuity": float(raw.get("temporal_continuity", 0.5) or 0.5),
            "causal_connectivity": float(raw.get("causal_connectivity", 0.5) or 0.5),
            "symbolic_density": float(raw.get("symbolic_density", 0.3) or 0.3),
        }
        # derived
        try:
            features["light_balance"] = float(
                _safe_mean(list(self.sephirot_activation.values()) or [0.0])
            )
            features["shadow_balance"] = float(
                _safe_mean(list(self.qliphoth_activation.values()) or [0.0])
            )
            features["dialectical_tension"] = abs(
                features["light_balance"] - features["shadow_balance"]
            )
        except Exception:
            pass
        return features

    def _calculate_emotional_valence(self) -> float:
        try:
            pos = (
                self.sephirot_activation.get(SephirotNode.CHESED, 0.0) * 0.8
                + self.sephirot_activation.get(SephirotNode.TIPHERETH, 0.0) * 0.7
                + self.sephirot_activation.get(SephirotNode.NETZACH, 0.0) * 0.6
            ) / 3.0
            neg = (
                self.qliphoth_activation.get(QliphothNode.GOLACHAB, 0.0) * 0.8
                + self.qliphoth_activation.get(QliphothNode.SAMAEL, 0.0) * 0.7
                + self.qliphoth_activation.get(QliphothNode.LILITH, 0.0) * 0.6
            ) / 3.0
            return float(pos - neg)
        except Exception:
            return 0.0

    def _calculate_temporal_coherence(self) -> float:
        if len(self.processing_history) < 2:
            return 0.5
        current_total = sum(
            list(self.sephirot_activation.values())
            + list(self.qliphoth_activation.values())
        )
        prevs = [
            entry.get("sephirot_state", {})
            for entry in list(self.processing_history)[-5:]
        ]
        if not prevs:
            return 0.5
        coherence_sum = 0.0
        for prev_state in prevs:
            prev_total = (
                sum(prev_state.values()) if isinstance(prev_state, dict) else 0.0
            )
            coherence_sum += 1.0 - min(1.0, abs(current_total - prev_total) / 10.0)
        return float(coherence_sum / len(prevs))

    def _recognize_patterns(
        self, perception_data: PerceptionData
    ) -> PatternRecognitionResult:
        best = None
        best_conf = 0.0
        for name, tmpl in self.pattern_templates.items():
            conf = self._calculate_pattern_confidence(tmpl)
            if conf > best_conf:
                best_conf = conf
                best = name
        complexity = self._calculate_perception_complexity()
        archetypes = self._identify_activated_archetypes()
        semantic_features = self._extract_semantic_features(perception_data)
        emotional_valence = self._calculate_emotional_valence()
        temporal_coherence = self._calculate_temporal_coherence()
        return PatternRecognitionResult(  # type: ignore[call-arg]
            pattern_type=best or "unknown",
            confidence=float(best_conf),
            complexity=float(complexity),
            archetypes_activated=list(archetypes),
            semantic_features=dict(semantic_features),
            emotional_valence=float(emotional_valence),
            temporal_coherence=float(temporal_coherence),
        )

    def _perform_dialectical_synthesis(self) -> dict[str, Any]:
        synthesis: dict[str, Any] = {}
        try:
            pairs = self._get_dialectical_pairs()
            for seph, qlips in pairs.items():
                seph_act = float(self.sephirot_activation.get(seph, 0.0))
                qlip_act = float(
                    _safe_mean([self.qliphoth_activation.get(q, 0.0) for q in qlips])
                    or 0.0
                )
                tension = abs(seph_act - qlip_act)
                synthesis_value = (seph_act + qlip_act) / 2.0
                synthesis[f"{getattr(seph, 'name', str(seph))}_synthesis"] = {
                    "tension": tension,
                    "synthesis": synthesis_value,
                    "dominant": ("sephirot" if seph_act > qlip_act else "qliphoth"),
                }
        except Exception:
            logger.debug("Dialectical synthesis failed", exc_info=True)
        return synthesis

    def _get_dialectical_pairs(self) -> dict[Any, list[Any]]:
        # stable mapping; tolerant to missing enums
        try:
            return {
                SephirotNode.KETHER: [QliphothNode.THAUMIEL],
                SephirotNode.CHOKMAH: [QliphothNode.GHAGIEL],
                SephirotNode.BINAH: [QliphothNode.SATHARIEL],
                SephirotNode.CHESED: [QliphothNode.GAMCHICOTH],
                SephirotNode.GEBURAH: [QliphothNode.GOLACHAB],
                SephirotNode.TIPHERETH: [QliphothNode.THAGIRION],
                SephirotNode.NETZACH: [QliphothNode.AREB_ZARAK],
                SephirotNode.HOD: [QliphothNode.SAMAEL],
                SephirotNode.YESOD: [QliphothNode.GAMALIEL],
                SephirotNode.MALKUTH: [QliphothNode.LILITH],
            }
        except Exception:
            return {}

    def _update_pattern_memory(
        self, pattern_result: PatternRecognitionResult, perception_data: PerceptionData
    ) -> None:
        try:
            key = f"{getattr(pattern_result, 'pattern_type', 'unknown')}_{float(getattr(pattern_result, 'confidence', 0.0)):.2f}"
            entry = self.pattern_memory.setdefault(
                key,
                {
                    "pattern_type": getattr(pattern_result, "pattern_type", "unknown"),
                    "occurrences": 0,
                    "average_confidence": 0.0,
                    "contexts": [],
                    "last_seen": None,
                },
            )
            entry["occurrences"] += 1
            entry["average_confidence"] = (
                (entry["average_confidence"] * (entry["occurrences"] - 1))
                + getattr(pattern_result, "confidence", 0.0)
            ) / entry["occurrences"]
            entry["last_seen"] = getattr(perception_data, "timestamp", time.time())
            if len(entry["contexts"]) < 10:
                entry["contexts"].append(
                    {
                        "timestamp": getattr(perception_data, "timestamp", time.time()),
                        "source_type": getattr(
                            perception_data, "source_type", "unknown"
                        ),
                        "archetypes": getattr(
                            pattern_result, "archetypes_activated", []
                        ),
                    }
                )
        except Exception:
            logger.debug("Update pattern memory failed", exc_info=True)

    # --- EVA persistence (async-aware, best-effort) ---

    async def _maybe_record_eva_event(self, payload: dict[str, Any]) -> None:
        """Best-effort record to EVA; handles coroutine/sync functions and missing manager."""
        if not self.eva_manager:
            return
        try:
            record_fn = getattr(self.eva_manager, "record_experience", None)
            if not callable(record_fn):
                return
            event = {
                "entity_id": self.entity_id,
                "event_type": "dialectical_processor_event",
                "data": payload,
                "timestamp": time.time(),
            }
            if inspect.iscoroutinefunction(record_fn):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(record_fn(**event))
                except RuntimeError:
                    # no running loop; run synchronously
                    try:
                        asyncio.run(record_fn(**event))
                    except Exception:
                        logger.debug("EVA async record failed (no loop)", exc_info=True)
            else:
                try:
                    record_fn(**event)
                except Exception:
                    logger.debug("EVA sync record failed", exc_info=True)
        except Exception:
            logger.exception("EVA recording failed in _maybe_record_eva_event")

    # --- Utilities for tests / simulation ---

    def snapshot_state(self) -> dict[str, Any]:
        """Return a compact snapshot of dialectical state (useful for tests)."""
        return {
            "sephirot_activation": {
                str(k): float(v) for k, v in self.sephirot_activation.items()
            },
            "qliphoth_activation": {
                str(k): float(v) for k, v in self.qliphoth_activation.items()
            },
            "pattern_memory_size": len(self.pattern_memory),
            "processing_history_len": len(self.processing_history),
        }

    def simulate_perception(
        self, synthetic: dict[str, Any]
    ) -> PatternRecognitionResult:
        """
        Synchronous convenience wrapper for tests: build a PerceptionData-like object
        and process it via the async entrypoint.
        """

        class _PD:
            raw_data = synthetic
            timestamp = time.time()
            source_type = "simulated"

        coro = self.process_perception(_PD())  # type: ignore[arg-type]
        try:
            return asyncio.get_event_loop().run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
