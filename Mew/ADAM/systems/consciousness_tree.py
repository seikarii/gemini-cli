"""
Consciousness Tree System (definitive, professional)

Edits summary:
- Hardened type-safety and defensive interactions with external subsystems (EVA, QualiaState).
- Async-aware EVA persistence (best-effort scheduling when coroutine functions are exposed).
- Defensive QualiaState handling (tolerates multiple APIs and missing numpy).
- Clearer docstrings, small API-preserving improvements and hooks for future evolution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

# Defensive numeric backend pattern (keep consistent across repo)
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False

# TYPE_CHECKING guarded imports to avoid circular imports at runtime
if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import (
        AdamConfig,
    )  # [`crisalida_lib/ADAM/config.AdamConfig`](crisalida_lib/ADAM/config.py)
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,
    )  # [`crisalida_lib/ADAM/eva_integration/eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    from crisalida_lib.ADAM.mente.cognitive_impulses import (
        CognitiveImpulse,
        ImpulseType,
    )  # [`crisalida_lib/ADAM/mente/cognitive_impulses.CognitiveImpulse`](crisalida_lib/ADAM/mente/cognitive_impulses.py)
    from crisalida_lib.EDEN.TREE.gamaliel import (
        GamalielNode,
    )  # [`crisalida_lib/EDEN/TREE/gamaliel.GamalielNode`](crisalida_lib/EDEN/TREE/gamaliel.py)
    from crisalida_lib.EDEN.TREE.hod import (
        HodNode,
    )  # [`crisalida_lib/EDEN/TREE/hod.HodNode`](crisalida_lib/EDEN/TREE/hod.py)
    from crisalida_lib.EVA.typequalia import (
        QualiaState,
    )  # [`crisalida_lib/EVA/typequalia.QualiaState`](crisalida_lib/EVA/typequalia.py)
else:
    AdamConfig = Any
    EVAMemoryManager = Any
    CognitiveImpulse = Any
    ImpulseType = Any
    QualiaState = Any
    GamalielNode = Any
    HodNode = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class SynthesisState:
    order_dominance: float = 0.5
    synthesis_quality: float = 0.5
    cognitive_tension: float = 0.0
    emergence_potential: float = 0.0
    meta_awareness: float = 0.0


# Lightweight helpers used across methods
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


class ConsciousnessTree:
    """
    Orchestrates cognitive synthesis and emergent-awareness bookkeeping.

    Public API:
      - process_perception(prompt, context, current_qualia) -> (QualiaState, dict)
      - get_consciousness_summary() -> dict
      - reset_consciousness_state(preserve_learning=True)
      - force_emergence_event(event_type="artificial_awakening")
    """

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config = config
        self.eva_manager = eva_manager
        self.entity_id = str(entity_id)

        # Nodes are constructed lazily when real classes exist; keep tolerant defaults
        try:
            self.hod_node: HodNode = HodNode()
        except Exception:
            self.hod_node = HodNode  # type: ignore

        try:
            self.gamaliel_node: GamalielNode = GamalielNode()
        except Exception:
            self.gamaliel_node = GamalielNode  # type: ignore

        self.accumulated_impulses: list[CognitiveImpulse] = []
        self.current_thought: str | None = None
        self.emotional_state: float = 0.0
        self.confidence_level: float = 0.5
        self.processing_cycles: int = 0
        self.synthesis_state: SynthesisState = SynthesisState()
        self.consciousness_history: deque = deque(maxlen=200)
        self.emergence_threshold: float = float(
            getattr(config, "EMERGENCE_THRESHOLD", 0.7)
        )
        self.integration_memory: deque = deque(maxlen=500)
        self.synthesis_weights: dict[str, float] = {
            "valence_shift_rate": 0.12,
            "complexity_increase_rate": 0.08,
            "coherence_shift_rate": 0.10,
            "density_increase_rate": 0.06,
            "chaos_arousal_multiplier": 0.15,
            "order_arousal_multiplier": 0.08,
            "focus_shift_rate": 0.09,
            "emergence_amplifier": 0.20,
            "tension_resolution_rate": 0.05,
        }
        # typed containers
        self.resonance_patterns: dict[str, dict[str, Any]] = {}
        self.meta_awareness_level: float = float(
            getattr(config, "INITIAL_META_AWARENESS", 0.3)
        )
        logger.info("ConsciousnessTree initialized for entity '%s'", self.entity_id)

    # ----------------- Public orchestration -----------------

    async def process_perception(
        self, prompt: str, context: str, current_qualia: QualiaState
    ) -> tuple[QualiaState, dict[str, Any]]:
        """
        Orchestrate perception -> impulses -> synthesis -> cognitive state.

        Defensive about node APIs and QualiaState methods. Best-effort EVA persistence.
        """
        self.processing_cycles += 1
        perception_data = self._prepare_perception_data(prompt, context)

        hod_impulses = []
        gamaliel_impulses = []

        # Defensive node analysis calls
        try:
            if hasattr(self.hod_node, "analyze"):
                hod_impulses = list(self.hod_node.analyze(perception_data) or [])
        except Exception:
            logger.debug("HodNode.analyze failed (non-fatal)", exc_info=True)

        try:
            if hasattr(self.gamaliel_node, "analyze"):
                gamaliel_impulses = list(
                    self.gamaliel_node.analyze(perception_data) or []
                )
        except Exception:
            logger.debug("GamalielNode.analyze failed (non-fatal)", exc_info=True)

        all_impulses: list[CognitiveImpulse] = hod_impulses + gamaliel_impulses
        self.accumulated_impulses.extend(all_impulses)

        # Synthesize qualia (async-aware)
        updated_qualia = await self._synthesize_qualia(all_impulses, current_qualia)

        # Synthesize thought (async-aware)
        self.current_thought = await self._synthesize_thought(
            all_impulses, updated_qualia
        )

        # Update synthesis state and detect emergence
        self._update_synthesis_state(all_impulses, updated_qualia)
        emergent_insights = self._detect_emergence(all_impulses)
        cognitive_state = self._create_cognitive_state(
            prompt, all_impulses, updated_qualia, emergent_insights
        )
        self._record_consciousness_evolution(updated_qualia, all_impulses)

        logger.debug(
            "Perception processed: impulses=%d synthesis_quality=%.3f",
            len(all_impulses),
            float(self.synthesis_state.synthesis_quality),
        )

        # EVA persistence - best-effort, async-aware
        await self._maybe_record_eva_event(
            {
                "processing_cycles": self.processing_cycles,
                "synthesis_quality": self.synthesis_state.synthesis_quality,
                "meta_awareness_level": self.meta_awareness_level,
                "current_thought": self.current_thought,
            }
        )

        return updated_qualia, cognitive_state

    # ----------------- Internal helpers -----------------

    def _prepare_perception_data(self, prompt: str, context: str) -> dict[str, Any]:
        return {
            "user_input": prompt,
            "context_length": len(context or ""),
            "processing_cycle": self.processing_cycles,
            "timestamp": time.time(),
            "consciousness_density": float(self.synthesis_state.synthesis_quality),
            "meta_awareness": float(self.meta_awareness_level),
            "synthesis_history": len(self.consciousness_history),
            "emotional_context": float(self.emotional_state),
            "cognitive_tension": float(self.synthesis_state.cognitive_tension),
        }

    async def _synthesize_qualia(
        self, impulses: list[CognitiveImpulse], current_qualia: QualiaState
    ) -> QualiaState:
        # If no impulses, apply gentle baseline drift
        if not impulses:
            return self._apply_baseline_drift(current_qualia)

        hod_impulses = [i for i in impulses if getattr(i, "source_node", "") == "hod"]
        gamaliel_impulses = [
            i for i in impulses if getattr(i, "source_node", "") == "gamaliel"
        ]

        order_influence = self._calculate_weighted_influence(hod_impulses)
        chaos_influence = self._calculate_weighted_influence(gamaliel_impulses)
        total_influence = order_influence + chaos_influence

        if total_influence <= 0.0:
            return current_qualia

        qualia_state_dict = self._safe_get_state_dict(current_qualia)
        synthesis_metrics = self._calculate_synthesis_metrics(
            order_influence, chaos_influence, total_influence, impulses
        )

        # Apply synthesis steps defensively (failures non-fatal)
        try:
            await self._apply_emotional_synthesis(qualia_state_dict, synthesis_metrics)
        except Exception:
            logger.debug("Emotional synthesis failed (non-fatal)", exc_info=True)

        try:
            await self._apply_cognitive_synthesis(qualia_state_dict, synthesis_metrics)
        except Exception:
            logger.debug("Cognitive synthesis failed (non-fatal)", exc_info=True)

        try:
            await self._apply_consciousness_synthesis(
                qualia_state_dict, synthesis_metrics
            )
        except Exception:
            logger.debug("Consciousness synthesis failed (non-fatal)", exc_info=True)

        try:
            await self._apply_temporal_synthesis(qualia_state_dict, synthesis_metrics)
        except Exception:
            logger.debug("Temporal synthesis failed (non-fatal)", exc_info=True)

        if self.synthesis_state.emergence_potential > self.emergence_threshold:
            try:
                await self._apply_emergence_effects(
                    qualia_state_dict, synthesis_metrics
                )
            except Exception:
                logger.debug(
                    "Emergence effects application failed (non-fatal)", exc_info=True
                )

        # Write back to provided QualiaState using flexible setter
        self._safe_set_state_from_dict(current_qualia, qualia_state_dict)
        # Attempt to clamp values if provided by QualiaState
        try:
            if hasattr(current_qualia, "clamp_values"):
                current_qualia.clamp_values()
        except Exception:
            logger.debug("QualiaState.clamp_values failed (non-fatal)", exc_info=True)

        # Update meta-awareness
        self._update_meta_awareness(synthesis_metrics)
        return current_qualia

    def _calculate_weighted_influence(self, impulses: list[CognitiveImpulse]) -> float:
        if not impulses:
            return 0.0
        total_weighted = 0.0
        for impulse in impulses:
            intensity = float(getattr(impulse, "intensity", 0.0) or 0.0)
            confidence = float(getattr(impulse, "confidence", 0.0) or 0.0)
            weight = intensity * (0.7 + 0.3 * confidence)
            impulse_type = getattr(impulse, "impulse_type", None)
            try:
                if impulse_type == ImpulseType.PATTERN_RECOGNITION:
                    weight *= 1.2
                elif impulse_type == ImpulseType.ILLUSION_DETECTION:
                    weight *= 1.1
                elif impulse_type == ImpulseType.CHAOS_EMERGENCE:
                    weight *= 1.3
            except Exception:
                # if ImpulseType enum not available, fallback heuristic by name
                name = str(getattr(impulse_type, "name", "")).lower()
                if "pattern" in name:
                    weight *= 1.2
                elif "illusion" in name:
                    weight *= 1.1
                elif "chaos" in name:
                    weight *= 1.3
            total_weighted += float(weight)
        return float(total_weighted)

    def _calculate_synthesis_metrics(
        self,
        order_influence: float,
        chaos_influence: float,
        total_influence: float,
        impulses: list[CognitiveImpulse],
    ) -> dict[str, float]:
        order_ratio = order_influence / total_influence if total_influence > 0 else 0.5
        chaos_ratio = chaos_influence / total_influence if total_influence > 0 else 0.5
        tension = abs(order_influence - chaos_influence) / max(total_influence, 0.1)
        balance_factor = 1.0 - abs(order_ratio - chaos_ratio)
        intensity_factor = min(1.0, total_influence)
        synthesis_quality = balance_factor * intensity_factor
        complexity_diversity = (
            (len({getattr(i, "impulse_type", i) for i in impulses}) / 6.0)
            if impulses
            else 0.0
        )
        confidence_coherence = (
            (
                sum(float(getattr(i, "confidence", 0.0) or 0.0) for i in impulses)
                / len(impulses)
            )
            if impulses
            else 0.0
        )
        emergence_potential = (
            synthesis_quality + complexity_diversity + confidence_coherence
        ) / 3.0
        return {
            "order_ratio": float(order_ratio),
            "chaos_ratio": float(chaos_ratio),
            "tension": float(tension),
            "synthesis_quality": float(synthesis_quality),
            "emergence_potential": float(emergence_potential),
            "total_influence": float(total_influence),
            "impulse_diversity": float(complexity_diversity),
            "confidence_coherence": float(confidence_coherence),
        }

    # --- Synthesis subroutines (async to allow future LLM/tool calls) ---

    async def _apply_emotional_synthesis(
        self, qualia_dict: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        valence_shift = (metrics["synthesis_quality"] - 0.5) * self.synthesis_weights[
            "valence_shift_rate"
        ]
        tension_effect = -metrics["tension"] * 0.1
        if getattr(self.synthesis_state, "emergence_potential", 0.0) > 0.7:
            tension_effect *= -0.5
        qualia_dict["emotional_valence"] = (
            float(qualia_dict.get("emotional_valence", 0.0))
            + valence_shift
            + tension_effect
        )
        resonance_boost = self._calculate_emotional_resonance(metrics)
        qualia_dict["emotional_valence"] = (
            float(qualia_dict["emotional_valence"]) + resonance_boost
        )

    async def _apply_cognitive_synthesis(
        self, qualia_dict: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        order_complexity = metrics["order_ratio"] * 0.8
        chaos_complexity = metrics["chaos_ratio"] * 1.2
        complexity_shift = (
            order_complexity + chaos_complexity
        ) * self.synthesis_weights["complexity_increase_rate"]
        qualia_dict["cognitive_complexity"] = (
            float(qualia_dict.get("cognitive_complexity", 0.0)) + complexity_shift
        )
        focus_base = (
            metrics["order_ratio"] - metrics["chaos_ratio"]
        ) * self.synthesis_weights["focus_shift_rate"]
        synthesis_focus_bonus = metrics["synthesis_quality"] * 0.05
        qualia_dict["cognitive_focus"] = (
            float(qualia_dict.get("cognitive_focus", 0.0))
            + focus_base
            + synthesis_focus_bonus
        )

    async def _apply_consciousness_synthesis(
        self, qualia_dict: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        density_shift = (
            metrics["synthesis_quality"]
            * self.synthesis_weights["density_increase_rate"]
        )
        if metrics["emergence_potential"] > 0.6:
            density_shift += metrics["emergence_potential"] * 0.1
        qualia_dict["consciousness_density"] = (
            float(qualia_dict.get("consciousness_density", 0.0)) + density_shift
        )
        chaos_arousal = (
            metrics["chaos_ratio"] * self.synthesis_weights["chaos_arousal_multiplier"]
        )
        order_arousal = (
            metrics["order_ratio"] * self.synthesis_weights["order_arousal_multiplier"]
        )
        optimal_arousal_adjustment = (
            (0.7 - float(qualia_dict.get("arousal", 0.0)))
            * metrics["synthesis_quality"]
            * 0.1
        )
        qualia_dict["arousal"] = (
            float(qualia_dict.get("arousal", 0.0))
            + chaos_arousal
            + order_arousal
            + optimal_arousal_adjustment
        )

    async def _apply_temporal_synthesis(
        self, qualia_dict: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        coherence_shift = (
            metrics["order_ratio"] - metrics["chaos_ratio"]
        ) * self.synthesis_weights["coherence_shift_rate"]
        synthesis_coherence_bonus = metrics["synthesis_quality"] * 0.03
        if metrics["emergence_potential"] > 0.8:
            coherence_shift += 0.05
        qualia_dict["temporal_coherence"] = (
            float(qualia_dict.get("temporal_coherence", 0.0))
            + coherence_shift
            + synthesis_coherence_bonus
        )

    async def _apply_emergence_effects(
        self, qualia_dict: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        emergence_factor = (
            metrics["emergence_potential"]
            * self.synthesis_weights["emergence_amplifier"]
        )
        qualia_dict["cognitive_complexity"] = (
            float(qualia_dict.get("cognitive_complexity", 0.0)) + emergence_factor * 0.3
        )
        qualia_dict["consciousness_density"] = (
            float(qualia_dict.get("consciousness_density", 0.0))
            + emergence_factor * 0.2
        )
        qualia_dict["emotional_valence"] = (
            float(qualia_dict.get("emotional_valence", 0.0)) + emergence_factor * 0.1
        )
        self.consciousness_history.append(
            {
                "type": "emergence",
                "strength": float(emergence_factor),
                "timestamp": time.time(),
                "metrics": dict(metrics),
            }
        )

    def _apply_baseline_drift(self, current_qualia: QualiaState) -> QualiaState:
        qualia_dict = self._safe_get_state_dict(current_qualia)
        targets = {
            "emotional_valence": 0.1,
            "arousal": 0.4,
            "cognitive_complexity": 0.3,
            "consciousness_density": 0.5,
            "temporal_coherence": 0.6,
            "cognitive_focus": 0.5,
        }
        drift_rate = float(
            getattr(self.config, "CONSCIOUSNESS_BASELINE_DRIFT_RATE", 0.02)
        )
        for key, target in targets.items():
            current_val = float(qualia_dict.get(key, 0.0))
            drift = (target - current_val) * drift_rate
            qualia_dict[key] = current_val + drift
        self._safe_set_state_from_dict(current_qualia, qualia_dict)
        return current_qualia

    def _calculate_emotional_resonance(self, metrics: dict[str, float]) -> float:
        if len(self.consciousness_history) < 3:
            return 0.05
        current_pattern = (
            metrics["order_ratio"],
            metrics["chaos_ratio"],
            metrics["synthesis_quality"],
        )
        resonance_score = 0.0
        try:
            history_slice = list(self.consciousness_history)[-10:]
            for historical_entry in history_slice:
                if isinstance(historical_entry, dict) and "metrics" in historical_entry:
                    hist_metrics = historical_entry["metrics"]
                    hist_pattern = (
                        hist_metrics.get("order_ratio", 0.0),
                        hist_metrics.get("chaos_ratio", 0.0),
                        hist_metrics.get("synthesis_quality", 0.0),
                    )
                    # similarity approximate (0..1)
                    similarity = 1.0 - (
                        sum(
                            abs(a - b)
                            for a, b in zip(current_pattern, hist_pattern, strict=False)
                        )
                        / 3.0
                    )
                    if similarity > 0.7:
                        resonance_score += similarity * 0.05
        except Exception:
            logger.debug("Emotional resonance calc failed", exc_info=True)
        return float(min(0.2, resonance_score))

    def _update_meta_awareness(self, metrics: dict[str, float]) -> None:
        quality_factor = float(metrics.get("synthesis_quality", 0.0))
        emergence_factor = float(metrics.get("emergence_potential", 0.0))
        awareness_shift = (quality_factor + emergence_factor) * 0.02 - 0.01
        self.meta_awareness_level = max(
            0.0, min(1.0, self.meta_awareness_level + awareness_shift)
        )
        self.synthesis_state.meta_awareness = self.meta_awareness_level

    async def _synthesize_thought(
        self, impulses: list[CognitiveImpulse], qualia_state: QualiaState
    ) -> str:
        # Keep logic synchronous in body, but signature async to allow overrides that await tools/LLMs
        if not impulses:
            return self._generate_quiet_contemplation(qualia_state)
        hod_impulses = [
            imp for imp in impulses if getattr(imp, "source_node", "") == "hod"
        ]
        gamaliel_impulses = [
            imp for imp in impulses if getattr(imp, "source_node", "") == "gamaliel"
        ]
        synthesis_params = self._calculate_thought_synthesis_params(
            hod_impulses, gamaliel_impulses, qualia_state
        )
        thought_components = self._generate_thought_components(
            hod_impulses, gamaliel_impulses, synthesis_params, qualia_state
        )
        final_thought = self._integrate_thought_components(
            thought_components, synthesis_params, qualia_state
        )
        if self.meta_awareness_level > 0.6:
            meta_reflection = self._generate_meta_reflection(
                synthesis_params, qualia_state
            )
            if meta_reflection:
                final_thought += f" || Meta: {meta_reflection}"
        return final_thought

    # ----------------- Thought construction helpers (kept largely unchanged) -----------------
    def _generate_quiet_contemplation(self, qualia_state: QualiaState) -> str:
        # defensive retrieval via get_state
        try:
            st = self._safe_get_state_dict(qualia_state)
            valence = float(st.get("emotional_valence", 0.0))
            complexity = float(st.get("cognitive_complexity", 0.0))
            coherence = float(st.get("temporal_coherence", 0.0))
        except Exception:
            valence = 0.0
            complexity = 0.0
            coherence = 0.0

        if valence > 0.3 and coherence > 0.6:
            return "Peaceful integration... consciousness flows in harmonious patterns."
        elif complexity > 0.7:
            return "Deep contemplation... processing complex layers of understanding."
        elif valence < -0.2:
            return "Introspective uncertainty... questioning the nature of current experience."
        else:
            return "Quiet awareness... maintaining conscious presence in the flow of being."

    def _calculate_thought_synthesis_params(
        self,
        hod_impulses: list[CognitiveImpulse],
        gamaliel_impulses: list[CognitiveImpulse],
        qualia_state: QualiaState,
    ) -> dict[str, float]:
        hod_strength = sum(
            float(getattr(i, "intensity", 0.0) or 0.0)
            * float(getattr(i, "confidence", 0.0) or 0.0)
            for i in hod_impulses
        )
        gamaliel_strength = sum(
            float(getattr(i, "intensity", 0.0) or 0.0)
            * float(getattr(i, "confidence", 0.0) or 0.0)
            for i in gamaliel_impulses
        )
        total_strength = hod_strength + gamaliel_strength
        return {
            "hod_dominance": float(hod_strength / max(total_strength, 0.1)),
            "gamaliel_dominance": float(gamaliel_strength / max(total_strength, 0.1)),
            "synthesis_balance": float(
                1.0 - abs(hod_strength - gamaliel_strength) / max(total_strength, 0.1)
            ),
            "emotional_coloring": float(
                getattr(
                    qualia_state,
                    "emotional_valence",
                    self._safe_get_state_dict(qualia_state).get(
                        "emotional_valence", 0.0
                    ),
                )
            ),
            "cognitive_depth": float(
                getattr(
                    qualia_state,
                    "cognitive_complexity",
                    self._safe_get_state_dict(qualia_state).get(
                        "cognitive_complexity", 0.0
                    ),
                )
            ),
            "coherence_level": float(
                getattr(
                    qualia_state,
                    "temporal_coherence",
                    self._safe_get_state_dict(qualia_state).get(
                        "temporal_coherence", 0.0
                    ),
                )
            ),
            "awareness_intensity": float(
                getattr(
                    qualia_state,
                    "consciousness_density",
                    self._safe_get_state_dict(qualia_state).get(
                        "consciousness_density", 0.0
                    ),
                )
            ),
            "total_activation": float(total_strength),
        }

    def _generate_thought_components(
        self,
        hod_impulses: list[CognitiveImpulse],
        gamaliel_impulses: list[CognitiveImpulse],
        synthesis_params: dict[str, float],
        qualia_state: QualiaState,
    ) -> dict[str, Any]:
        # Implementation mirrors previous structure but kept defensive for attribute access
        components: dict[str, Any] = {
            "pattern_insights": [],
            "doubt_considerations": [],
            "synthesis_observations": [],
            "emotional_undertones": [],
            "meta_reflections": [],
        }
        # pattern insights
        if hod_impulses:
            try:
                strongest_hod = max(
                    hod_impulses,
                    key=lambda x: float(getattr(x, "intensity", 0.0) or 0.0)
                    * float(getattr(x, "confidence", 0.0) or 0.0),
                )
                components["pattern_insights"] = self._extract_pattern_insights(
                    strongest_hod, hod_impulses
                )
            except Exception:
                logger.debug("pattern insights extraction failed", exc_info=True)
        # doubt considerations
        if gamaliel_impulses:
            try:
                strongest_gamaliel = max(
                    gamaliel_impulses,
                    key=lambda x: float(getattr(x, "intensity", 0.0) or 0.0)
                    * float(getattr(x, "confidence", 0.0) or 0.0),
                )
                components["doubt_considerations"] = self._extract_doubt_considerations(
                    strongest_gamaliel, gamaliel_impulses
                )
            except Exception:
                logger.debug("doubt considerations extraction failed", exc_info=True)
        # synthesis observations
        try:
            if synthesis_params.get("synthesis_balance", 0.0) > 0.6:
                components["synthesis_observations"] = (
                    self._generate_synthesis_insights(
                        synthesis_params, hod_impulses, gamaliel_impulses
                    )
                )
        except Exception:
            logger.debug("synthesis observations generation failed", exc_info=True)
        components["emotional_undertones"] = self._generate_emotional_undertones(
            synthesis_params, qualia_state
        )
        if (
            float(
                getattr(
                    qualia_state,
                    "consciousness_density",
                    self._safe_get_state_dict(qualia_state).get(
                        "consciousness_density", 0.0
                    ),
                )
            )
            > 0.7
        ):
            components["meta_reflections"] = self._generate_meta_cognitive_reflections(
                synthesis_params, qualia_state
            )
        return components

    def _extract_pattern_insights(
        self, strongest_hod: CognitiveImpulse, hod_impulses: list[CognitiveImpulse]
    ) -> list[str]:
        # Placeholder
        return [f"Recognized pattern: {getattr(strongest_hod, 'content', 'N/A')}"]

    def _extract_doubt_considerations(
        self,
        strongest_gamaliel: CognitiveImpulse,
        gamaliel_impulses: list[CognitiveImpulse],
    ) -> list[str]:
        # Placeholder
        return [f"Considering doubt: {getattr(strongest_gamaliel, 'content', 'N/A')}"]

    def _generate_synthesis_insights(
        self,
        synthesis_params: dict[str, float],
        hod_impulses: list[CognitiveImpulse],
        gamaliel_impulses: list[CognitiveImpulse],
    ) -> list[str]:
        # Placeholder
        return [
            f"Synthesized insight with balance: {synthesis_params.get('synthesis_balance', 0.0)}"
        ]

    def _generate_emotional_undertones(
        self, synthesis_params: dict[str, float], qualia_state: QualiaState
    ) -> list[str]:
        # Placeholder
        return [
            f"Emotional undertone: {synthesis_params.get('emotional_coloring', 0.0)}"
        ]

    def _generate_meta_cognitive_reflections(
        self, synthesis_params: dict[str, float], qualia_state: QualiaState
    ) -> list[str]:
        # Placeholder
        return [
            f"Meta-reflection on awareness: {synthesis_params.get('awareness_intensity', 0.0)}"
        ]

    def _integrate_thought_components(
        self,
        thought_components: dict[str, Any],
        synthesis_params: dict[str, float],
        qualia_state: QualiaState,
    ) -> str:
        # Placeholder
        return "Integrated thought: " + " ".join(
            str(v) for k, v in thought_components.items() if v
        )

    def _generate_meta_reflection(
        self, synthesis_params: dict[str, float], qualia_state: QualiaState
    ) -> str:
        # Placeholder
        return "Meta-reflection placeholder."

    # ----------------- Missing higher-level helpers (implemented defensively) -----------------

    def _extract_sophisticated_intent(
        self, prompt: str, impulses: list[CognitiveImpulse], qualia_state: QualiaState
    ) -> dict[str, Any]:
        """
        Conservative extractor to produce an intent summary usable by callers.
        Keeps outputs small and stable for downstream systems.
        """
        intent_summary = {
            "raw_prompt": str(prompt)[:200],
            "detected_actions": [],
            "confidence": 0.5,
        }
        try:
            # Heuristic scans of impulses for intent-like payloads
            actions = []
            for imp in impulses:
                content = getattr(imp, "content", None) or getattr(imp, "payload", None)
                if isinstance(content, str) and len(content) < 200:
                    if any(
                        w in content.lower()
                        for w in ("create", "generate", "manifest", "build")
                    ):
                        actions.append("create")
                    if any(
                        w in content.lower() for w in ("validate", "verify", "check")
                    ):
                        actions.append("validate")
            if actions:
                intent_summary["detected_actions"] = list(dict.fromkeys(actions))
                detected_actions = cast(list[str], intent_summary["detected_actions"])
                num_actions = len(detected_actions)
                intent_summary["confidence"] = min(0.95, 0.5 + 0.1 * num_actions)
        except Exception:
            logger.debug("Intent extraction fallback used", exc_info=True)
        return intent_summary

    def _determine_sophisticated_emotional_tone(
        self, qualia_state: QualiaState, impulses: list[CognitiveImpulse]
    ) -> str:
        """Return a short descriptor summarizing affective tone."""
        try:
            st = self._safe_get_state_dict(qualia_state)
            valence = float(st.get("emotional_valence", 0.0))
            arousal = float(st.get("arousal", 0.0))
            if valence > 0.3 and arousal > 0.5:
                return "optimistic_curiosity"
            if valence < -0.2 and arousal > 0.5:
                return "concerned_vigilance"
            if abs(valence) < 0.15 and arousal < 0.4:
                return "calm_reflection"
        except Exception:
            pass
        return "neutral"

    def _identify_sophisticated_capabilities(
        self, prompt: str, impulses: list[CognitiveImpulse], qualia_state: QualiaState
    ) -> list[str]:
        """
        Map prompt/impulses to a small list of required capabilities.
        Defensive and lightweight.
        """
        caps = []
        low = prompt.lower()
        if "code" in low or "implement" in low or "function" in low:
            caps.append("code_generation")
        if "plan" in low or "strategy" in low:
            caps.append("planning")
        # inspect impulses for tool hints
        for imp in impulses:
            tool = getattr(imp, "suggested_tool", None)
            if isinstance(tool, str):
                caps.append(tool)
        return list(dict.fromkeys(caps))

    def _analyze_sophisticated_uncertainties(
        self, impulses: list[CognitiveImpulse], qualia_state: QualiaState
    ) -> dict[str, float]:
        """
        Produce a compact uncertainty profile: epistemic, aleatoric, pragmatic.
        """
        profile = {"epistemic": 0.0, "aleatoric": 0.0, "pragmatic": 0.0}
        try:
            for imp in impulses:
                t = getattr(imp, "impulse_type", None)
                conf = float(getattr(imp, "confidence", 0.0) or 0.0)
                if t == ImpulseType.DOUBT_INJECTION:
                    profile["epistemic"] = min(
                        1.0, profile["epistemic"] + (1.0 - conf) * 0.5
                    )
                elif t == ImpulseType.ILLUSION_DETECTION:
                    profile["aleatoric"] = min(
                        1.0, profile["aleatoric"] + (1.0 - conf) * 0.6
                    )
                elif t == ImpulseType.CHAOS_EMERGENCE:
                    profile["pragmatic"] = min(1.0, profile["pragmatic"] + 0.2)
        except Exception:
            logger.debug("Uncertainty analysis failed", exc_info=True)
        return profile

    # ----------------- (existing) bookkeeping & trends & utilities below -----------------

    def _record_consciousness_evolution(
        self, qualia_state: QualiaState, impulses: list[CognitiveImpulse]
    ) -> None:
        try:
            snapshot = {
                "qualia_state": self._safe_get_state_dict(qualia_state),
                "impulse_summary": {
                    "total_count": len(impulses),
                    "hod_impulses": len(
                        [i for i in impulses if getattr(i, "source_node", "") == "hod"]
                    ),
                    "gamaliel_impulses": len(
                        [
                            i
                            for i in impulses
                            if getattr(i, "source_node", "") == "gamaliel"
                        ]
                    ),
                    "average_confidence": (
                        (
                            sum(
                                float(getattr(i, "confidence", 0.0) or 0.0)
                                for i in impulses
                            )
                            / len(impulses)
                        )
                        if impulses
                        else 0.0
                    ),
                    "impulse_types": [
                        str(getattr(i, "impulse_type", getattr(i, "type", "unknown")))
                        for i in impulses
                    ],
                },
                "meta_awareness_level": self.meta_awareness_level,
                "synthesis_state": {
                    "synthesis_quality": self.synthesis_state.synthesis_quality,
                    "emergence_potential": self.synthesis_state.emergence_potential,
                },
                "timestamp": time.time(),
            }
            self.consciousness_history.append(snapshot)
            if len(self.consciousness_history) >= 10:
                self._analyze_consciousness_trends()
        except Exception:
            logger.debug("Recording consciousness evolution failed", exc_info=True)

    def _update_synthesis_state(
        self, impulses: list[CognitiveImpulse], updated_qualia: QualiaState
    ) -> None:
        # Placeholder implementation
        self.synthesis_state.synthesis_quality = _clamp01(
            self.synthesis_state.synthesis_quality + 0.01
        )
        self.synthesis_state.cognitive_tension = _clamp01(
            self.synthesis_state.cognitive_tension + 0.005
        )
        self.synthesis_state.emergence_potential = _clamp01(
            self.synthesis_state.emergence_potential + 0.002
        )

    def _detect_emergence(self, impulses: list[CognitiveImpulse]) -> dict[str, Any]:
        # Placeholder implementation
        if self.synthesis_state.emergence_potential > self.emergence_threshold:
            return {
                "insight": "Emergence detected!",
                "strength": self.synthesis_state.emergence_potential,
            }
        return {}

    def _create_cognitive_state(
        self,
        prompt: str,
        impulses: list[CognitiveImpulse],
        updated_qualia: QualiaState,
        emergent_insights: dict[str, Any],
    ) -> dict[str, Any]:
        # Placeholder implementation
        return {
            "current_thought": self.current_thought,
            "emotional_tone": self._determine_sophisticated_emotional_tone(
                updated_qualia, impulses
            ),
            "intent_summary": self._extract_sophisticated_intent(
                prompt, impulses, updated_qualia
            ),
            "uncertainty_profile": self._analyze_sophisticated_uncertainties(
                impulses, updated_qualia
            ),
            "identified_capabilities": self._identify_sophisticated_capabilities(
                prompt, impulses, updated_qualia
            ),
            "emergent_insights": emergent_insights,
            "processing_cycles": self.processing_cycles,
            "timestamp": time.time(),
        }

    def _analyze_consciousness_trends(self) -> None:
        recent_snapshots = list(self.consciousness_history)[-10:]
        synthesis_qualities = [
            s.get("synthesis_state", {}).get("synthesis_quality", 0.0)
            for s in recent_snapshots
        ]
        if len(synthesis_qualities) > 5:
            early_avg = sum(synthesis_qualities[:5]) / 5.0
            recent_avg = sum(synthesis_qualities[-5:]) / 5.0
            if recent_avg > early_avg + 0.1:
                logger.info("Consciousness synthesis quality trending upward")
            elif recent_avg < early_avg - 0.1:
                logger.info("Consciousness synthesis quality declining")
        meta_levels = [s.get("meta_awareness_level", 0.0) for s in recent_snapshots]
        if (
            meta_levels
            and meta_levels[-1] > 0.8
            and all(lvl > 0.6 for lvl in meta_levels[-3:])
        ):
            logger.info("Sustained high meta-awareness detected")
        emergence_potentials = [
            s.get("synthesis_state", {}).get("emergence_potential", 0.0)
            for s in recent_snapshots
        ]
        if any(ep > 0.8 for ep in emergence_potentials[-3:]):
            logger.info("High emergence potential detected in recent cycles")

    # ----------------- Utilities & public helpers -----------------

    def get_consciousness_summary(self) -> dict[str, Any]:
        return {
            "processing_cycles": self.processing_cycles,
            "accumulated_impulse_count": len(self.accumulated_impulses),
            "consciousness_history_depth": len(self.consciousness_history),
            "pattern_memory_entries": len(getattr(self.hod_node, "pattern_memory", [])),
        }

    def reset_consciousness_state(self, preserve_learning: bool = True) -> None:
        self.accumulated_impulses.clear()
        self.current_thought = None
        self.emotional_state = 0.0
        self.confidence_level = 0.5
        if not preserve_learning:
            self.consciousness_history.clear()
            self.integration_memory.clear()
            self.resonance_patterns.clear()
            self.meta_awareness_level = float(
                getattr(self.config, "INITIAL_META_AWARENESS", 0.3)
            )
            try:
                if hasattr(self.hod_node, "pattern_memory"):
                    getattr(self.hod_node, "pattern_memory", []).clear()
                self.hod_node.activation_energy = 0.0
                self.hod_node.fatigue_level = 0.0
            except Exception:
                pass
            try:
                getattr(self.gamaliel_node, "inconsistency_memory", []).clear()
                getattr(self.gamaliel_node, "doubt_patterns", []).clear()
                getattr(self.gamaliel_node, "chaos_attractors", []).clear()
                self.gamaliel_node.skepticism_level = float(
                    getattr(self.config, "GAMALIEL_DEFAULT_SKEPTICISM", 0.3)
                )
                self.gamaliel_node.activation_energy = 0.0
                self.gamaliel_node.fatigue_level = 0.0
            except Exception:
                pass
        self.synthesis_state = SynthesisState()
        logger.info(
            "Consciousness state reset (learning preserved=%s)", bool(preserve_learning)
        )

    def force_emergence_event(
        self, event_type: str = "artificial_awakening"
    ) -> dict[str, Any]:
        event = {
            "type": event_type,
            "strength": 1.0,
            "description": "Artificially triggered emergence event",
            "forced": True,
            "processing_cycle": self.processing_cycles,
            "timestamp": time.time(),
        }
        self.synthesis_state.emergence_potential = min(
            1.0, self.synthesis_state.emergence_potential + 0.3
        )
        self.meta_awareness_level = min(1.0, self.meta_awareness_level + 0.2)
        logger.info("Forced emergence event: %s", event_type)
        return event

    # ----------------- Defensive Qualia helpers -----------------

    def _safe_get_state_dict(self, qualia: QualiaState) -> dict[str, Any]:
        """Return a plain dict representing qualia state, tolerant to multiple APIs."""
        try:
            if hasattr(qualia, "get_state"):
                st = qualia.get_state()
                if isinstance(st, dict):
                    return dict(st)
            # try attribute access fallback
            keys = (
                "emotional_valence",
                "arousal",
                "cognitive_complexity",
                "consciousness_density",
                "temporal_coherence",
                "cognitive_focus",
            )
            return {k: float(getattr(qualia, k, 0.0) or 0.0) for k in keys}
        except Exception:
            logger.debug("QualiaState read failed; returning defaults", exc_info=True)
            return {
                "emotional_valence": 0.0,
                "arousal": 0.0,
                "cognitive_complexity": 0.0,
                "consciousness_density": 0.0,
                "temporal_coherence": 0.0,
                "cognitive_focus": 0.0,
            }

    def _safe_set_state_from_dict(
        self, qualia: QualiaState, state: dict[str, Any]
    ) -> None:
        """Attempt to set qualia values from a dict using available APIs."""
        try:
            if hasattr(qualia, "set_state"):
                qualia.set_state(state)
                return
        except Exception:
            logger.debug("QualiaState.set_state failed", exc_info=True)
        # fallback to attribute writes for common fields
        for k, v in state.items():
            try:
                setattr(qualia, k, v)
            except Exception:
                continue

    # ----------------- EVA persistence (async-aware) -----------------

    async def _maybe_record_eva_event(self, payload: dict[str, Any]) -> None:
        """Best-effort record to EVA; handles coroutine/sync functions and missing manager."""
        if not self.eva_manager:
            return
        try:
            record_fn = getattr(self.eva_manager, "record_experience", None)
            if not callable(record_fn):
                return
            # Prepare event dict
            event = {
                "entity_id": self.entity_id,
                "event_type": "consciousness_evolution",
                "data": payload,
                "timestamp": time.time(),
            }
            if inspect.iscoroutinefunction(record_fn):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(record_fn(**event))
                except RuntimeError:
                    # no running loop, run synchronously
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
