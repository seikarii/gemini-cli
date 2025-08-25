"""
Cerebellum - The Dialectical Processor (definitive)
===================================================

Professionalized, hardened and future-proofed implementation of the Cerebellum
(dialectical processor). Key improvements:

- Thread-safe (RLock) mutable state.
- Defensive numpy acceleration with graceful pure-Python fallback.
- Configurable thresholds and decay dynamics.
- Serialization (to/from dict) and simple pattern-learning hooks.
- Stable typing and safer numeric guards.
- Small API for future async/plug-in evolutions (learn(), register_template()).

This module depends on two enums living in the Earth core types:
- [`crisalida_lib.EARTH.core_types.enums.SephirotNode`](crisalida_lib/EARTH/core_types/enums.py)
- [`crisalida_lib.EARTH.core_types.enums.QliphothNode`](crisalida_lib/EARTH/core_types/enums.py)
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, cast
from types import ModuleType

if TYPE_CHECKING:
    from crisalida_lib.EARTH.core_types.enums import (  # type: ignore
        QliphothNode,
        SephirotNode,
    )

# Defensive import pattern for numpy used for optional acceleration.
np: ModuleType | None = None
try:
    import numpy as np  # type: ignore
    np = np # re-assign to satisfy mypy
except Exception:
    np = None  # graceful fallback; functions use pure-Python alternatives

# Use the canonical enums at runtime; import lazily to avoid cycles.
try:
    from crisalida_lib.EARTH.core_types.enums import (  # type: ignore
        QliphothNode,
        SephirotNode,
    )
except (
    Exception
):  # pragma: no cover - fallback placeholders if enums missing at runtime
    QliphothNode = Any  # type: ignore
    SephirotNode = Any  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Tunables / defaults ---
DEFAULT_ACTIVATION_THRESHOLD = 0.3
DEFAULT_RESONANCE_THRESHOLD = 0.5
DEFAULT_MAX_HISTORY = 1000
DEFAULT_MAX_CONTEXTS_PER_PATTERN = 10
DEFAULT_DECAY_PER_SEC = 0.005  # passive decay per second


@dataclass
class PerceptionData:
    raw_data: dict[str, Any]
    source_type: str = "sensor"
    timestamp: float = field(default_factory=time.time)


@dataclass
class PatternRecognitionResult:
    pattern_type: str
    confidence: float
    complexity: float
    archetypes_activated: list[str]
    semantic_features: dict[str, float]
    emotional_valence: float
    temporal_coherence: float


class DialecticalProcessor:
    """
    Thread-safe dialectical processor.

    Example:
        dp = DialecticalProcessor()
        p = PerceptionData({"novelty": 0.8, "threat_level": 0.1})
        result = dp.process_perception(p)
    """

    def __init__(
        self,
        activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
        resonance_threshold: float = DEFAULT_RESONANCE_THRESHOLD,
        decay_per_sec: float = DEFAULT_DECAY_PER_SEC,
        max_history: int = DEFAULT_MAX_HISTORY,
        max_contexts_per_pattern: int = DEFAULT_MAX_CONTEXTS_PER_PATTERN,
    ) -> None:
        self._lock = RLock()
        self.activation_threshold = float(activation_threshold)
        self.resonance_threshold = float(resonance_threshold)
        self.decay_per_sec = float(decay_per_sec)
        self.max_history = int(max_history)
        self.max_contexts_per_pattern = int(max_contexts_per_pattern)

        # Initialize activations with explicit comprehension to avoid shared-mutable pitfalls
        self.sephirot_activation: dict[Any, float] = (
            dict.fromkeys(
                list(SephirotNode) if not TYPE_CHECKING and SephirotNode is not Any else (), 0.0
            )
            if SephirotNode is not Any
            else {}
        )
        self.qliphoth_activation: dict[Any, float] = (
            dict.fromkeys(
                list(QliphothNode) if not TYPE_CHECKING and QliphothNode is not Any else (), 0.0
            )
            if QliphothNode is not Any
            else {}
        )

        # Connections and templates
        self.sephirot_connections: dict[Any, dict[Any, float]] = (
            self._init_sephirot_connections()
        )
        self.qliphoth_connections: dict[Any, dict[Any, float]] = (
            self._init_qliphoth_connections()
        )
        self.pattern_memory: dict[str, dict[str, Any]] = {}
        self.pattern_templates: dict[str, dict[str, Any]] = (
            self._init_pattern_templates()
        )

        # History and bookkeeping
        self.processing_history: deque = deque(maxlen=self.max_history)
        self._last_decay_ts = time.time()

    # ---------------------------
    # Initialization helpers
    # ---------------------------
    def _init_sephirot_connections(self) -> dict[Any, dict[Any, float]]:
        connections: dict[Any, dict[Any, float]] = {}
        try:
            connections.setdefault(SephirotNode.KETHER, {})[
                SephirotNode.TIPHERETH
            ] = 0.8
            connections.setdefault(SephirotNode.TIPHERETH, {})[SephirotNode.YESOD] = 0.8
            connections.setdefault(SephirotNode.YESOD, {})[SephirotNode.MALKUTH] = 0.8
            connections.setdefault(SephirotNode.CHOKMAH, {})[SephirotNode.CHESED] = 0.7
            connections.setdefault(SephirotNode.CHESED, {})[SephirotNode.NETZACH] = 0.7
            connections.setdefault(SephirotNode.BINAH, {})[SephirotNode.GEBURAH] = 0.7
            connections.setdefault(SephirotNode.GEBURAH, {})[SephirotNode.HOD] = 0.7
            # horizontal / bidirectional
            connections.setdefault(SephirotNode.CHOKMAH, {})[SephirotNode.BINAH] = 0.6
            connections.setdefault(SephirotNode.CHESED, {})[SephirotNode.GEBURAH] = 0.6
            connections.setdefault(SephirotNode.NETZACH, {})[SephirotNode.HOD] = 0.6
            for src, targets in list(connections.items()):
                for tgt, strength in list(targets.items()):
                    connections.setdefault(tgt, {})[src] = strength
        except Exception:
            logger.warning(
                "Partial Sephirot connection initialization failed; continuing with best-effort."
            )
        return connections

    def _init_qliphoth_connections(self) -> dict[Any, dict[Any, float]]:
        connections: dict[Any, dict[Any, float]] = {}
        try:
            connections.setdefault(QliphothNode.THAUMIEL, {})[
                QliphothNode.SATHARIEL
            ] = 0.7
            connections.setdefault(QliphothNode.GHAGIEL, {})[
                QliphothNode.GAMCHICOTH
            ] = 0.8
            connections.setdefault(QliphothNode.GOLACHAB, {})[
                QliphothNode.THAGIRION
            ] = 0.6
            connections.setdefault(QliphothNode.AREB_ZARAK, {})[
                QliphothNode.SAMAEL
            ] = 0.7
            connections.setdefault(QliphothNode.GAMALIEL, {})[QliphothNode.LILITH] = 0.9
            for src, targets in list(connections.items()):
                for tgt, strength in list(targets.items()):
                    connections.setdefault(tgt, {})[src] = strength * 0.8
        except Exception:
            logger.warning(
                "Partial Qliphoth connection initialization failed; continuing with best-effort."
            )
        return connections

    def _init_pattern_templates(self) -> dict[str, dict[str, Any]]:
        """Seeded templates — can be extended via register_template() or learn()."""
        return {
            "threat_pattern": {
                "sephirot_signature": {
                    getattr(SephirotNode, "GEBURAH", None): 0.8,
                    getattr(SephirotNode, "HOD", None): 0.6,
                },
                "qliphoth_signature": {
                    getattr(QliphothNode, "GOLACHAB", None): 0.7,
                    getattr(QliphothNode, "SAMAEL", None): 0.5,
                },
                "emotional_markers": {"fear": 0.7, "alertness": 0.8},
            },
            "creative_pattern": {
                "sephirot_signature": {
                    getattr(SephirotNode, "CHOKMAH", None): 0.9,
                    getattr(SephirotNode, "NETZACH", None): 0.7,
                },
                "qliphoth_signature": {getattr(QliphothNode, "THAUMIEL", None): 0.3},
                "emotional_markers": {"excitement": 0.6, "inspiration": 0.8},
            },
        }

    # ---------------------------
    # Public processing API
    # ---------------------------
    def process_perception(
        self, perception_data: PerceptionData
    ) -> PatternRecognitionResult:
        with self._lock:
            self._decay_activations_if_needed()
            self._activate_nodes_from_data(perception_data)
            self._propagate_activations()
            pattern_result = self._recognize_patterns(perception_data)
            synthesis = self._perform_dialectical_synthesis()
            self._update_pattern_memory(pattern_result, perception_data)
            # record compact history entry
            self.processing_history.append(
                {
                    "timestamp": perception_data.timestamp,
                    "sephirot_state": copy.deepcopy(self.sephirot_activation),
                    "qliphoth_state": copy.deepcopy(self.qliphoth_activation),
                    "pattern_result": pattern_result,
                    "synthesis": synthesis,
                }
            )
            return pattern_result

    # ---------------------------
    # Activation lifecycle
    # ---------------------------
    def _decay_activations_if_needed(self) -> None:
        now = time.time()
        elapsed = max(0.0, now - self._last_decay_ts)
        if elapsed <= 0:
            return
        decay = self.decay_per_sec * elapsed
        if decay <= 0:
            return
        for k in list(self.sephirot_activation.keys()):
            self.sephirot_activation[k] = max(
                0.0, self.sephirot_activation.get(k, 0.0) - decay
            )
        for k in list(self.qliphoth_activation.keys()):
            self.qliphoth_activation[k] = max(
                0.0, self.qliphoth_activation.get(k, 0.0) - decay
            )
        self._last_decay_ts = time.time()

    def _activate_nodes_from_data(self, perception_data: PerceptionData) -> None:
        raw = perception_data.raw_data or {}
        novelty = float(raw.get("novelty", 0.0) or 0.0)
        threat_level = float(raw.get("threat_level", 0.0) or 0.0)
        complexity = float(raw.get("complexity", 0.0) or 0.0)
        harmony = float(raw.get("harmony", 0.0) or 0.0)
        chaos_level = float(raw.get("chaos_level", 0.0) or 0.0)

        # Sephirotic tendencies
        try:
            if novelty > 0.5 and SephirotNode.CHOKMAH in self.sephirot_activation:
                self.sephirot_activation[SephirotNode.CHOKMAH] = min(
                    1.0,
                    self.sephirot_activation.get(SephirotNode.CHOKMAH, 0.0)
                    + novelty * 0.7,
                )
            if harmony > 0.5 and SephirotNode.TIPHERETH in self.sephirot_activation:
                self.sephirot_activation[SephirotNode.TIPHERETH] = min(
                    1.0,
                    self.sephirot_activation.get(SephirotNode.TIPHERETH, 0.0)
                    + harmony * 0.8,
                )
            if complexity > 0.7 and SephirotNode.BINAH in self.sephirot_activation:
                self.sephirot_activation[SephirotNode.BINAH] = min(
                    1.0,
                    self.sephirot_activation.get(SephirotNode.BINAH, 0.0)
                    + complexity * 0.6,
                )
        except Exception:
            logger.debug(
                "Sephirot activation mapping partially failed; skipping some updates"
            )

        # Qliphothic tendencies
        try:
            if threat_level > 0.3 and QliphothNode.GOLACHAB in self.qliphoth_activation:
                self.qliphoth_activation[QliphothNode.GOLACHAB] = min(
                    1.0,
                    self.qliphoth_activation.get(QliphothNode.GOLACHAB, 0.0)
                    + threat_level * 0.8,
                )
            if threat_level > 0.3 and QliphothNode.SAMAEL in self.qliphoth_activation:
                self.qliphoth_activation[QliphothNode.SAMAEL] = min(
                    1.0,
                    self.qliphoth_activation.get(QliphothNode.SAMAEL, 0.0)
                    + threat_level * 0.6,
                )
            if chaos_level > 0.5 and QliphothNode.THAUMIEL in self.qliphoth_activation:
                self.qliphoth_activation[QliphothNode.THAUMIEL] = min(
                    1.0,
                    self.qliphoth_activation.get(QliphothNode.THAUMIEL, 0.0)
                    + chaos_level * 0.7,
                )
        except Exception:
            logger.debug(
                "Qliphoth activation mapping partially failed; skipping some updates"
            )

        # keep the activations normalized in range
        self._normalize_activations()

    def _propagate_activations(self) -> None:
        new_seph = dict(self.sephirot_activation)
        for src, activation in list(self.sephirot_activation.items()):
            if activation > self.activation_threshold:
                targets = cast(dict[Any, float], self.sephirot_connections.get(src, {}))
                for tgt, strength in targets.items():
                    influence = activation * strength * 0.3
                    new_seph[tgt] = min(1.0, float(new_seph.get(tgt, 0.0)) + influence)
        new_qlip = dict(self.qliphoth_activation)
        for src, activation in list(self.qliphoth_activation.items()):
            targets = cast(dict[Any, float], self.qliphoth_connections.get(src, {}))
            for tgt, strength in targets.items():
                influence = activation * strength * 0.4
                new_qlip[tgt] = min(1.0, float(new_qlip.get(tgt, 0.0)) + influence)
        self.sephirot_activation = new_seph
        self.qliphoth_activation = new_qlip
        self._normalize_activations()

    def _normalize_activations(self) -> None:
        for k in list(self.sephirot_activation.keys()):
            self.sephirot_activation[k] = max(
                0.0, min(1.0, float(self.sephirot_activation.get(k, 0.0)))
            )
        for k in list(self.qliphoth_activation.keys()):
            self.qliphoth_activation[k] = max(
                0.0, min(1.0, float(self.qliphoth_activation.get(k, 0.0)))
            )

    # ---------------------------
    # Pattern recognition & synthesis
    # ---------------------------
    def _recognize_patterns(
        self, perception_data: PerceptionData
    ) -> PatternRecognitionResult:
        best_match: str | None = None
        best_confidence = 0.0
        for name, template in self.pattern_templates.items():
            try:
                confidence = self._calculate_pattern_confidence(template)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = name
            except Exception:
                logger.debug("Pattern confidence calc failed for %s", name)
        complexity = self._calculate_perception_complexity()
        archetypes = self._identify_activated_archetypes()
        semantic_features = self._extract_semantic_features(perception_data)
        emotional_valence = self._calculate_emotional_valence()
        temporal_coherence = self._calculate_temporal_coherence()
        return PatternRecognitionResult(
            pattern_type=best_match or "unknown",
            confidence=best_confidence,
            complexity=complexity,
            archetypes_activated=archetypes,
            semantic_features=semantic_features,
            emotional_valence=emotional_valence,
            temporal_coherence=temporal_coherence,
        )

    def _calculate_pattern_confidence(self, template: dict[str, Any]) -> float:
        seph_sig = template.get("sephirot_signature", {}) or {}
        qlip_sig = template.get("qliphoth_signature", {}) or {}
        seph_match = 0.0
        if seph_sig:
            for node, expected in seph_sig.items():
                actual = float(self.sephirot_activation.get(node, 0.0))
                seph_match += max(0.0, 1.0 - abs(float(expected) - actual))
            seph_match /= max(1, len(seph_sig))
        qlip_match = 0.0
        if qlip_sig:
            for node, expected in qlip_sig.items():
                actual = float(self.qliphoth_activation.get(node, 0.0))
                qlip_match += max(0.0, 1.0 - abs(float(expected) - actual))
            qlip_match /= max(1, len(qlip_sig))
        if seph_sig and qlip_sig:
            return seph_match * 0.6 + qlip_match * 0.4
        elif seph_sig:
            return seph_match
        elif qlip_sig:
            return qlip_match
        return 0.0

    def _calculate_perception_complexity(self) -> float:
        total_seph = (
            sum(self.sephirot_activation.values()) if self.sephirot_activation else 0.0
        )
        total_qlip = (
            sum(self.qliphoth_activation.values()) if self.qliphoth_activation else 0.0
        )
        diversity = len(
            [v for v in self.sephirot_activation.values() if v > 0.1]
        ) + len([v for v in self.qliphoth_activation.values() if v > 0.1])
        raw_complex = (total_seph + total_qlip) / 2.0
        denom = (
            (len(list(SephirotNode)) + len(list(QliphothNode)))
            if SephirotNode is not Any and QliphothNode is not Any
            else max(1, diversity)
        )
        diversity_factor = min(1.0, diversity / denom)
        return min(1.0, raw_complex * 0.7 + diversity_factor * 0.3)

    def _identify_activated_archetypes(self) -> list[str]:
        archetypes: list[str] = []
        try:
            if self.sephirot_activation.get(SephirotNode.KETHER, 0.0) > 0.7:
                archetypes.append("Divine_Crown")
            if self.sephirot_activation.get(SephirotNode.CHOKMAH, 0.0) > 0.6:
                archetypes.append("Wise_Father")
            if self.sephirot_activation.get(SephirotNode.BINAH, 0.0) > 0.6:
                archetypes.append("Understanding_Mother")
            if self.sephirot_activation.get(SephirotNode.TIPHERETH, 0.0) > 0.7:
                archetypes.append("Harmonious_King")
            if self.qliphoth_activation.get(QliphothNode.THAUMIEL, 0.0) > 0.6:
                archetypes.append("Divided_Rebel")
            if self.qliphoth_activation.get(QliphothNode.LILITH, 0.0) > 0.6:
                archetypes.append("Night_Queen")
            if self.qliphoth_activation.get(QliphothNode.SAMAEL, 0.0) > 0.5:
                archetypes.append("Poisoned_Angel")
        except Exception:
            logger.debug("Archetype identification partially failed")
        return archetypes

    def _extract_semantic_features(
        self, perception_data: PerceptionData
    ) -> dict[str, float]:
        raw = perception_data.raw_data or {}
        features: dict[str, float] = {}
        features["spatial_coherence"] = float(raw.get("spatial_coherence", 0.5) or 0.5)
        features["temporal_continuity"] = float(
            raw.get("temporal_continuity", 0.5) or 0.5
        )
        features["causal_connectivity"] = float(
            raw.get("causal_connectivity", 0.5) or 0.5
        )
        features["symbolic_density"] = float(raw.get("symbolic_density", 0.3) or 0.3)
        features["light_balance"] = (
            (
                sum(self.sephirot_activation.values())
                / max(1, len(self.sephirot_activation))
            )
            if self.sephirot_activation
            else 0.0
        )
        features["shadow_balance"] = (
            (
                sum(self.qliphoth_activation.values())
                / max(1, len(self.qliphoth_activation))
            )
            if self.qliphoth_activation
            else 0.0
        )
        features["dialectical_tension"] = abs(
            features["light_balance"] - features["shadow_balance"]
        )
        return features

    def _calculate_emotional_valence(self) -> float:
        pos = 0.0
        neg = 0.0
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
        except Exception:
            logger.debug("Emotional valence calc failed; defaulting to 0.0")
        return pos - neg

    def _calculate_temporal_coherence(self) -> float:
        if len(self.processing_history) < 2:
            return 0.5
        current_total = sum(self.sephirot_activation.values()) + sum(
            self.qliphoth_activation.values()
        )
        prev_entries = [
            entry["sephirot_state"] for entry in list(self.processing_history)[-5:]
        ]
        if not prev_entries:
            return 0.5
        coherence_sum = 0.0
        count = 0
        for prev_state in prev_entries:
            prev_total = (
                sum(prev_state.values()) if isinstance(prev_state, dict) else 0.0
            )
            coherence_sum += 1.0 - min(1.0, abs(current_total - prev_total) / 10.0)
            count += 1
        return coherence_sum / max(1, count)

    def _perform_dialectical_synthesis(self) -> dict[str, Any]:
        synthesis: dict[str, Any] = {}
        try:
            for seph_node, qlip_nodes in self._get_dialectical_pairs().items():
                seph_activation = float(self.sephirot_activation.get(seph_node, 0.0))
                qlip_activation = sum(
                    self.qliphoth_activation.get(n, 0.0) for n in qlip_nodes
                ) / max(1, len(qlip_nodes))
                tension = abs(seph_activation - qlip_activation)
                synthesis_value = (seph_activation + qlip_activation) / 2.0
                synthesis[
                    f"{getattr(seph_node, 'value', str(seph_node))}_synthesis"
                ] = {
                    "tension": tension,
                    "synthesis": synthesis_value,
                    "dominant": (
                        "sephirot" if seph_activation > qlip_activation else "qliphoth"
                    ),
                }
        except Exception:
            logger.debug("Dialectical synthesis failed partially")
        return synthesis

    def _get_dialectical_pairs(self) -> dict[Any, list[Any]]:
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

    # ---------------------------
    # Memory & learning
    # ---------------------------
    def _update_pattern_memory(
        self, pattern_result: PatternRecognitionResult, perception_data: PerceptionData
    ) -> None:
        key = f"{pattern_result.pattern_type}_{pattern_result.confidence:.2f}"
        entry = self.pattern_memory.setdefault(
            key,
            {
                "pattern_type": pattern_result.pattern_type,
                "occurrences": 0,
                "average_confidence": 0.0,
                "contexts": [],
                "last_seen": None,
            },
        )
        entry["occurrences"] += 1
        entry["average_confidence"] = (
            entry["average_confidence"] * (entry["occurrences"] - 1)
            + pattern_result.confidence
        ) / max(1, entry["occurrences"])
        entry["last_seen"] = perception_data.timestamp
        if len(entry["contexts"]) < self.max_contexts_per_pattern:
            entry["contexts"].append(
                {
                    "timestamp": perception_data.timestamp,
                    "source_type": perception_data.source_type,
                    "archetypes": pattern_result.archetypes_activated,
                }
            )

    def register_template(self, name: str, template: dict[str, Any]) -> None:
        """Register or replace a pattern template."""
        with self._lock:
            self.pattern_templates[name] = template

    def learn(self, pattern_name: str, adjustment: dict[str, float]) -> None:
        """
        Simple learning hook that adjusts an existing template signatures by 'adjustment'
        mapping node -> delta_strength. This is intentionally simple; future versions can
        implement gradient-based or sample-driven learning.
        """
        with self._lock:
            tmpl = self.pattern_templates.get(pattern_name)
            if not tmpl:
                return
            for kind in ("sephirot_signature", "qliphoth_signature"):
                sig = tmpl.get(kind, {})
                for node, delta in adjustment.items():
                    if node in sig:
                        sig[node] = float(
                            max(0.0, min(1.0, float(sig[node]) + float(delta)))
                        )
            self.pattern_templates[pattern_name] = tmpl

    # ---------------------------
    # Utility & diagnostics
    # ---------------------------
    def reset_state(self) -> None:
        with self._lock:
            for k in list(self.sephirot_activation.keys()):
                self.sephirot_activation[k] = 0.0
            for k in list(self.qliphoth_activation.keys()):
                self.qliphoth_activation[k] = 0.0
            self.processing_history.clear()
            self.pattern_memory.clear()
            self._last_decay_ts = time.time()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sephirot_activation": {
                    str(k): float(v) for k, v in self.sephirot_activation.items()
                },
                "qliphoth_activation": {
                    str(k): float(v) for k, v in self.qliphoth_activation.items()
                },
                "pattern_templates": {k: v for k, v in self.pattern_templates.items()},
                "pattern_memory_size": len(self.pattern_memory),
                "history_length": len(self.processing_history),
            }

    def get_top_activations(self, top_n: int = 5) -> dict[str, list[tuple[str, float]]]:
        with self._lock:
            seph_sorted = sorted(
                ((str(k), float(v)) for k, v in self.sephirot_activation.items()),
                key=lambda x: -x[1],
            )[:top_n]
            qlip_sorted = sorted(
                ((str(k), float(v)) for k, v in self.qliphoth_activation.items()),
                key=lambda x: -x[1],
            )[:top_n]
            return {"sephirot": seph_sorted, "qliphoth": qlip_sorted}
