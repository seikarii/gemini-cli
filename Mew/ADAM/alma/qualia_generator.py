"""
Professionalized Qualia Generator - definitive implementation.

Improvements:
- Defensive numpy import and graceful fallbacks.
- Robust QualiaState integration (supports legacy shapes).
- Async/sync-aware EVA recording (best-effort scheduling).
- Hardened numeric guards, logging, and adaptive weight learning.
- Clearer typings and serialization helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any

# defensive numpy import (repo pattern)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
from crisalida_lib.ADAM.systems.dialectical_processor import ProcesadorDialectico
from crisalida_lib.EVA.typequalia import QualiaState

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PatternRecognitionResult:
    """
    Resultado de reconocimiento de patrones avanzado.
    Adaptador alrededor del motor dialéctico para exponer una forma estable.
    """

    def __init__(
        self,
        perception_data: dict[str, Any],
        dialectical_processor: ProcesadorDialectico | None = None,
    ) -> None:
        self.processor = dialectical_processor or ProcesadorDialectico()
        try:
            result = self.processor.process_perception(perception_data)
        except Exception:
            logger.exception("Dialectical processor failed; using safe defaults")
            result = type("R", (), {})()  # simple fallback object
            result.pattern_type = "unknown"
            result.confidence = 0.0
            result.data = {}
            result.archetypes_activated = []
            result.temporal_coherence = 0.5
            result.complexity = 0.5
            result.semantic_features = {}
            result.emotional_valence = 0.0
            result.timestamp = time.time()

        self.pattern_type: str = getattr(result, "pattern_type", "unknown")
        self.confidence: float = float(getattr(result, "confidence", 0.0))
        self.data: dict[str, Any] = getattr(result, "data", {}) or {}
        self.archetypes_activated: list[str] = list(
            getattr(result, "archetypes_activated", []) or []
        )
        self.temporal_coherence: float = float(
            getattr(result, "temporal_coherence", 0.5)
        )
        self.complexity: float = float(getattr(result, "complexity", 0.5))
        self.semantic_features: dict[str, Any] = (
            getattr(result, "semantic_features", {}) or {}
        )
        self.emotional_valence: float = float(getattr(result, "emotional_valence", 0.0))
        self.timestamp: float = float(getattr(result, "timestamp", time.time()))


# --- Synthesizers ---------------------------------------------------------
class EmotionalQualiaSynthesizer:
    """Síntesis emocional basada en patrones y arquetipos (robusta)."""

    def synthesize(
        self, pattern_result: PatternRecognitionResult, internal_state: dict[str, Any]
    ) -> dict[str, float]:
        emotional_qualia: dict[str, float] = {
            "arousal": 0.0,
            "dominance": 0.5,
            "emotional_valence": 0.0,
        }
        pt = pattern_result.pattern_type
        archs = pattern_result.archetypes_activated
        valence = pattern_result.emotional_valence

        if pt == "threat_pattern":
            emotional_qualia.update(
                {
                    "emotional_valence": valence if valence else -0.7,
                    "arousal": 0.9,
                    "dominance": 0.2,
                    "chaos": 0.8,
                    "order": 0.2,
                }
            )
        elif pt == "creative_pattern":
            emotional_qualia.update(
                {
                    "emotional_valence": valence if valence else 0.6,
                    "arousal": 0.7,
                    "dominance": 0.8,
                    "chaos": 0.7,
                    "order": 0.6,
                }
            )
        elif pt == "wisdom_pattern":
            emotional_qualia.update(
                {
                    "emotional_valence": valence if valence else 0.4,
                    "arousal": 0.5,
                    "dominance": 0.7,
                    "order": 0.8,
                    "coherence": 0.9,
                }
            )
        else:
            emotional_qualia.update(
                {
                    "emotional_valence": valence or 0.0,
                    "arousal": 0.4,
                    "dominance": 0.5,
                    "chaos": 0.5,
                    "order": 0.5,
                }
            )

        # Modulation by archetypes
        for archetype in archs:
            if "Shadow" in archetype or "Night" in archetype:
                emotional_qualia["emotional_valence"] = (
                    emotional_qualia.get("emotional_valence", 0.0) - 0.2
                )
                emotional_qualia["chaos"] = emotional_qualia.get("chaos", 0.5) + 0.1
            elif (
                "Divine" in archetype
                or "Harmonious" in archetype
                or "Crown" in archetype
            ):
                emotional_qualia["emotional_valence"] = (
                    emotional_qualia.get("emotional_valence", 0.0) + 0.2
                )
                emotional_qualia["coherence"] = (
                    emotional_qualia.get("coherence", 0.5) + 0.1
                )

        # Internal-state modulation (stress, fatigue...)
        stress = float(internal_state.get("stress_level", 0.3))
        if stress > 0.7:
            emotional_qualia["arousal"] = min(
                1.0, emotional_qualia.get("arousal", 0.0) + (stress - 0.7) * 0.3
            )
            emotional_qualia["emotional_valence"] = (
                emotional_qualia.get("emotional_valence", 0.0) - (stress - 0.7) * 0.2
            )

        # Ensure values are sane
        for k, v in list(emotional_qualia.items()):
            if isinstance(v, (int, float)):
                emotional_qualia[k] = max(-1.0, min(1.0, float(v)))
        return emotional_qualia


class CognitiveQualiaSynthesizer:
    """Síntesis cognitiva basada en confianza, complejidad y carga de arquetipos."""

    def synthesize(
        self, pattern_result: PatternRecognitionResult, internal_state: dict[str, Any]
    ) -> dict[str, float]:
        cognitive_qualia: dict[str, float] = {
            "attention_focus": 0.5,
            "working_memory_load": 0.3,
            "cognitive_clarity": 0.5,
        }
        cognitive_qualia["cognitive_clarity"] = float(
            min(1.0, pattern_result.confidence + 0.2)
        )
        complexity_factor = float(pattern_result.complexity)
        cognitive_qualia["attention_focus"] = float(
            max(0.1, 1.0 - complexity_factor * 0.7)
        )
        archetype_load = float(len(pattern_result.archetypes_activated) * 0.1)
        cognitive_qualia["working_memory_load"] = float(
            min(0.9, complexity_factor * 0.5 + archetype_load)
        )
        cognitive_qualia["order"] = float(pattern_result.temporal_coherence)
        cognitive_qualia["coherence"] = float(
            (pattern_result.temporal_coherence + pattern_result.confidence) / 2.0
        )

        energy = float(internal_state.get("energy_balance", 0.7))
        if energy < 0.4:
            cognitive_qualia["attention_focus"] *= energy / 0.4
        # clamp
        for k, v in list(cognitive_qualia.items()):
            cognitive_qualia[k] = float(max(0.0, min(1.0, v)))
        return cognitive_qualia


class TranscendentQualiaSynthesizer:
    """Síntesis de aspectos trascendentes (orden, unidad, conciencia expandida)."""

    def synthesize(
        self, pattern_result: PatternRecognitionResult, internal_state: dict[str, Any]
    ) -> dict[str, float]:
        transcendent_qualia: dict[str, float] = {
            "transcendent_awareness": 0.0,
            "unity_experience": 0.0,
            "order": 0.5,
            "coherence": 0.5,
        }
        divine_archetypes = [
            arch
            for arch in pattern_result.archetypes_activated
            if any(x in arch for x in ("Divine", "Crown", "Unity"))
        ]
        if divine_archetypes:
            transcendent_qualia["transcendent_awareness"] = float(
                min(1.0, len(divine_archetypes) * 0.4)
            )
        if pattern_result.confidence > 0.8 and pattern_result.temporal_coherence > 0.7:
            transcendent_qualia["unity_experience"] = float(
                max(
                    0.0,
                    (pattern_result.confidence + pattern_result.temporal_coherence)
                    / 2.0
                    - 0.3,
                )
            )
        if transcendent_qualia["transcendent_awareness"] > 0.5:
            transcendent_qualia["order"] = 0.8
            transcendent_qualia["coherence"] = 0.9
        # clamp
        for k in list(transcendent_qualia.keys()):
            transcendent_qualia[k] = float(max(0.0, min(1.0, transcendent_qualia[k])))
        return transcendent_qualia


# --- Generator -----------------------------------------------------------
class QualiaGenerator:
    """
    Motor definitivo de síntesis de qualia.

    Public API:
      - generate_qualia(perception_data, internal_state, environmental_factors, qualia_field_influence=None)
      - adapt_synthesis_weights()
      - to_serializable()/load_serializable()
    """

    def __init__(
        self,
        dialectical_processor: ProcesadorDialectico | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
        config: AdamConfig | None = None,
    ) -> None:
        self.dialectical_processor = dialectical_processor or ProcesadorDialectico()
        self.emotional_synthesizer = EmotionalQualiaSynthesizer()
        self.cognitive_synthesizer = CognitiveQualiaSynthesizer()
        self.transcendent_synthesizer = TranscendentQualiaSynthesizer()
        self.synthesis_weights: dict[str, float] = {
            "emotional": 0.3,
            "cognitive": 0.4,
            "transcendent": 0.2,
            "environmental": 0.1,
        }
        self.synthesis_history: deque = deque(maxlen=500)
        self.qualia_templates = self._initialize_qualia_templates()
        self.current_mood: str = "neutral"
        self.energy_level: float = 0.7
        self.coherence_target: float = 0.6
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id
        self.config: AdamConfig = config or AdamConfig()

    def _initialize_qualia_templates(self) -> dict[str, dict[str, float]]:
        return {
            "meditative": {
                "order": 0.8,
                "chaos": 0.2,
                "coherence": 0.9,
                "emotional_valence": 0.3,
                "emotional_arousal": 0.2,
                "cognitive_clarity": 0.8,
                "transcendent_awareness": 0.7,
            },
            "creative": {
                "order": 0.4,
                "chaos": 0.8,
                "coherence": 0.6,
                "emotional_valence": 0.5,
                "emotional_arousal": 0.7,
                "cognitive_clarity": 0.6,
                "transcendent_awareness": 0.4,
            },
            "analytical": {
                "order": 0.9,
                "chaos": 0.1,
                "coherence": 0.8,
                "emotional_valence": 0.0,
                "emotional_arousal": 0.3,
                "cognitive_clarity": 0.9,
                "transcendent_awareness": 0.2,
            },
            "crisis": {
                "order": 0.2,
                "chaos": 0.9,
                "coherence": 0.3,
                "emotional_valence": -0.6,
                "emotional_arousal": 0.9,
                "cognitive_clarity": 0.4,
                "transcendent_awareness": 0.1,
            },
            "flow": {
                "order": 0.7,
                "chaos": 0.6,
                "coherence": 0.85,
                "emotional_valence": 0.6,
                "emotional_arousal": 0.6,
                "cognitive_clarity": 0.7,
                "transcendent_awareness": 0.5,
            },
        }

    def generate_qualia(
        self,
        perception_data: dict[str, Any],
        internal_state: dict[str, Any],
        environmental_factors: dict[str, float],
        qualia_field_influence: dict[str, float] | None = None,
    ) -> QualiaState:
        # 1. Recognize patterns
        pattern_result = PatternRecognitionResult(
            perception_data, self.dialectical_processor
        )

        # 2-4. Synthesize components
        emotional = self.emotional_synthesizer.synthesize(
            pattern_result, internal_state
        )
        cognitive = self.cognitive_synthesizer.synthesize(
            pattern_result, internal_state
        )
        transcendent = self.transcendent_synthesizer.synthesize(
            pattern_result, internal_state
        )

        # 5. Environmental influence
        environmental = self._synthesize_environmental_influence(
            environmental_factors, qualia_field_influence
        )

        # 6. Integrate
        qualia = self._integrate_qualia_components(
            emotional, cognitive, transcendent, environmental
        )

        # 7. Coherence modulation
        qualia = self._apply_coherence_modulation(qualia, internal_state)

        # 8. Record locally and to EVA (best-effort)
        self._record_synthesis(qualia, pattern_result, internal_state)
        self._record_to_eva(
            perception_data,
            internal_state,
            environmental_factors,
            qualia_field_influence,
            qualia,
        )

        return qualia

    def _synthesize_environmental_influence(
        self,
        environmental_factors: dict[str, float],
        qualia_field_influence: dict[str, float] | None = None,
    ) -> dict[str, float]:
        env_qualia: dict[str, float] = {
            "order": 0.5,
            "chaos": 0.5,
            "coherence": 0.5,
            "emotional_valence": 0.0,
            "cognitive_clarity": 0.5,
        }
        harmony_level = float(environmental_factors.get("harmony", 0.5))
        noise_level = float(environmental_factors.get("noise", 0.3))
        env_qualia["order"] = float(
            max(0.0, min(1.0, env_qualia["order"] + (harmony_level - 0.5) * 0.4))
        )
        env_qualia["chaos"] = float(
            max(0.0, min(1.0, env_qualia["chaos"] + (noise_level - 0.3) * 0.6))
        )
        env_qualia["coherence"] = float(
            max(
                0.0,
                min(1.0, env_qualia["coherence"] + (harmony_level - noise_level) * 0.3),
            )
        )
        env_qualia["cognitive_clarity"] = float(
            max(0.0, min(1.0, env_qualia["cognitive_clarity"] - noise_level * 0.4))
        )
        if qualia_field_influence:
            for k, v in qualia_field_influence.items():
                if k in env_qualia:
                    env_qualia[k] = float(
                        max(0.0, min(1.0, env_qualia[k] + float(v) * 0.1))
                    )
        return env_qualia

    def _integrate_qualia_components(
        self,
        emotional: dict[str, float],
        cognitive: dict[str, float],
        transcendent: dict[str, float],
        environmental: dict[str, float],
    ) -> QualiaState:
        qualia = QualiaState()
        # Weighted integration
        for attr in (
            "order",
            "chaos",
            "coherence",
            "emotional_valence",
            "cognitive_clarity",
        ):
            base_vals = {
                "emotional": emotional.get(attr, 0.5),
                "cognitive": cognitive.get(attr, 0.5),
                "transcendent": transcendent.get(attr, 0.5),
                "environmental": environmental.get(attr, 0.5),
            }
            value = sum(
                base_vals[k] * self.synthesis_weights.get(k, 0.0) for k in base_vals
            )
            try:
                if hasattr(qualia, attr):
                    setattr(qualia, attr, float(value))
                else:
                    # For legacy QualiaState shapes, attempt to set via dict interface
                    try:
                        qualia.__dict__[attr] = float(value)
                    except Exception:
                        pass
            except Exception:
                logger.debug("Failed to set qualia attr %s", attr, exc_info=True)

        # Specific attributes
        try:
            qualia.emotional_arousal = float(emotional.get("arousal", 0.0))
            qualia.attention_focus = float(cognitive.get("attention_focus", 0.5))
            qualia.working_memory_load = float(
                cognitive.get("working_memory_load", 0.3)
            )
            qualia.transcendent_awareness = float(
                transcendent.get("transcendent_awareness", 0.0)
            )
            qualia.unity_experience = float(transcendent.get("unity_experience", 0.0))
        except Exception:
            logger.debug(
                "Partial qualia attribute assignment failed; continuing", exc_info=True
            )
        return qualia

    def _apply_coherence_modulation(
        self, qualia: QualiaState, internal_state: dict[str, Any]
    ) -> QualiaState:
        stress_level = float(internal_state.get("stress_level", 0.3))
        energy_balance = float(internal_state.get("energy_balance", 0.6))
        consciousness_coherence = float(
            internal_state.get("consciousness_coherence", 0.5)
        )

        try:
            if stress_level > 0.6 and hasattr(qualia, "coherence"):
                qualia.coherence = float(
                    max(0.0, qualia.coherence * (1.0 - (stress_level - 0.6) * 0.5))
                )
            if stress_level > 0.6 and hasattr(qualia, "cognitive_clarity"):
                qualia.cognitive_clarity = float(
                    max(
                        0.0,
                        qualia.cognitive_clarity * (1.0 - (stress_level - 0.6) * 0.3),
                    )
                )
            if stress_level > 0.6 and hasattr(qualia, "chaos"):
                qualia.chaos = float(
                    min(1.0, getattr(qualia, "chaos", 0.0) + (stress_level - 0.6) * 0.4)
                )
            if energy_balance < 0.4:
                if hasattr(qualia, "emotional_arousal"):
                    qualia.emotional_arousal = float(
                        getattr(qualia, "emotional_arousal", 0.0) * energy_balance / 0.4
                    )
                if hasattr(qualia, "attention_focus"):
                    qualia.attention_focus = float(
                        getattr(qualia, "attention_focus", 0.5) * energy_balance / 0.4
                    )
                if hasattr(qualia, "working_memory_load"):
                    qualia.working_memory_load = float(
                        getattr(qualia, "working_memory_load", 0.3)
                        + (0.4 - energy_balance) * 0.5
                    )
            # temporal coherence mirror
            if hasattr(qualia, "coherence"):
                qualia.temporal_coherence = float(consciousness_coherence)
        except Exception:
            logger.exception("Error while applying coherence modulation")

        return qualia

    def _record_synthesis(
        self,
        qualia: QualiaState,
        perception_result: PatternRecognitionResult,
        internal_state: dict[str, Any],
    ) -> None:
        try:
            harmony = 0.0
            if hasattr(qualia, "calculate_harmony"):
                try:
                    harmony = float(qualia.calculate_harmony())
                except Exception:
                    harmony = 0.0
            elif hasattr(qualia, "get_state"):
                try:
                    st = qualia.get_state()
                    harmony = float(
                        st.get("coherence", 0.0) if isinstance(st, dict) else 0.0
                    )
                except Exception:
                    harmony = 0.0

            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "qualia_state": (
                    qualia.get_state()
                    if hasattr(qualia, "get_state")
                    else getattr(qualia, "__dict__", {})
                ),
                "pattern_type": perception_result.pattern_type,
                "pattern_confidence": perception_result.confidence,
                "internal_coherence": internal_state.get(
                    "consciousness_coherence", 0.5
                ),
                "synthesis_quality": float(harmony),
            }
            self.synthesis_history.append(record)
        except Exception:
            logger.exception("Failed to record synthesis locally")

    def _record_to_eva(
        self,
        perception_data: dict[str, Any],
        internal_state: dict[str, Any],
        environmental_factors: dict[str, float],
        qualia_field_influence: dict[str, float] | None,
        qualia: QualiaState,
    ) -> None:
        if not self.eva_manager:
            return
        try:
            rec = getattr(self.eva_manager, "record_experience", None)
            if rec is None:
                logger.debug("EVAMemoryManager.record_experience not found")
                return
            experience_id = f"qualia_generation:{self.entity_id}:{int(time.time())}:{abs(hash(str(perception_data))) & 0xFFFF}"
            payload = {
                "entity_id": self.entity_id,
                "perception_data": perception_data,
                "internal_state": internal_state,
                "environmental_factors": environmental_factors,
                "qualia_field_influence": qualia_field_influence,
                "generated_qualia": (
                    qualia.get_state()
                    if hasattr(qualia, "get_state")
                    else getattr(qualia, "__dict__", {})
                ),
                "timestamp": time.time(),
            }
            result = rec(
                entity_id=self.entity_id,
                event_type="qualia_generation",
                data=payload,
                experience_id=experience_id,
            )
            # If coroutine returned, schedule best-effort
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(result)
                    else:
                        loop.run_until_complete(result)
                except Exception:
                    logger.debug("Failed to schedule async EVA record", exc_info=True)
        except Exception:
            logger.exception("Failed to record qualia to EVA")

    def adapt_synthesis_weights(self) -> None:
        """Adapts synthesis weights based on historical synthesis quality."""
        try:
            if len(self.synthesis_history) < 10:
                return
            recent = list(self.synthesis_history)[-50:]
            emotional_q = []
            cognitive_q = []
            for rec in recent:
                q = float(rec.get("synthesis_quality", 0.0))
                pt = rec.get("pattern_type", "")
                if pt in ("creative_pattern", "chaos_pattern", "threat_pattern"):
                    emotional_q.append(q)
                elif pt in ("wisdom_pattern", "analytical", "analytical_pattern"):
                    cognitive_q.append(q)

            # compute means with safe fallback
            def safe_mean(xs: list[float]) -> float:
                if not xs:
                    return 0.0
                if np is not None:
                    return float(np.mean(xs))
                return float(sum(xs) / len(xs))

            e_mean = safe_mean(emotional_q)
            c_mean = safe_mean(cognitive_q)
            if e_mean > 0.7:
                self.synthesis_weights["emotional"] = min(
                    0.6, self.synthesis_weights["emotional"] + 0.03
                )
            if c_mean > 0.7:
                self.synthesis_weights["cognitive"] = min(
                    0.6, self.synthesis_weights["cognitive"] + 0.03
                )
            # normalize
            total = sum(self.synthesis_weights.values()) or 1.0
            for k in list(self.synthesis_weights.keys()):
                self.synthesis_weights[k] = float(self.synthesis_weights[k] / total)
        except Exception:
            logger.exception("adapt_synthesis_weights failed")

    # --- Serialization helpers -------------------------------------------
    def to_serializable(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "synthesis_weights": dict(self.synthesis_weights),
            "history_len": len(self.synthesis_history),
            "current_mood": self.current_mood,
            "energy_level": float(self.energy_level),
            "coherence_target": float(self.coherence_target),
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        try:
            self.entity_id = data.get("entity_id", self.entity_id)
            sw = data.get("synthesis_weights")
            if isinstance(sw, dict):
                self.synthesis_weights.update({k: float(v) for k, v in sw.items()})
            self.current_mood = data.get("current_mood", self.current_mood)
            self.energy_level = float(data.get("energy_level", self.energy_level))
            self.coherence_target = float(
                data.get("coherence_target", self.coherence_target)
            )
        except Exception:
            logger.exception("load_serializable failed")
