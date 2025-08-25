"""
Defines the `GamalielNode`, a cognitive node representing the "Obscene Ones".

This module contains the implementation of the Gamaliel Qliphoth, which
focuses on generating cognitive impulses related to chaos, doubt, and the
detection of illusions within the system's perceptions.
"""

import logging
import math
import random
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
    CosmicNode = Any  # runtime alias
    DivineSignature = Any  # runtime alias

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
else:
    LivingSymbolRuntime = Any
    DivineLanguageEvolved = Any

logger = logging.getLogger(__name__)


class GamalielNode(CosmicNode):
    """
    Enhanced cognitive node representing chaos, doubt, and illusion detection.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        activation_threshold: float = 0.1,
        mass: float = 3.5,
        influence_radius: float = 3.0,
    ) -> None:
        # prepare divine signature and attempt to forward it to base init
        gamaliel_divine_signature = None
        try:
            # prefer runtime local import from divine_sigils to avoid attr-defined issues
            from crisalida_lib.EVA.divine_sigils import DivineSignature as _DS

            gamaliel_divine_signature = _DS(glyph="ג")
        except Exception:
            gamaliel_divine_signature = None

        try:
            super().__init__(
                entity_id="gamaliel",
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="qliphoth",
                node_name="gamaliel",
                divine_signature=gamaliel_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__("gamaliel", activation_threshold)
            except Exception:
                object.__init__(self)

        self.inconsistency_memory: deque[Any] = deque(maxlen=50)
        self.doubt_patterns: deque[Any] = deque(maxlen=30)
        self.chaos_attractors: dict[int, Any] = {}  # Points that tend to generate chaos
        self.skepticism_level = 0.3  # Base skepticism

        # Keep signature attribute for backward compatibility
        self.divine_signature = gamaliel_divine_signature

        # EVA: memoria viviente y runtime de simulación (defensive init)
        try:
            self.eva_runtime: Optional[LivingSymbolRuntime] = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None
        try:
            self.divine_compiler: Optional[DivineLanguageEvolved] = (
                DivineLanguageEvolved(None)
            )
        except Exception:
            self.divine_compiler = None
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []
        self._current_phase: str = "default"

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        """Generate impulses focused on doubt, inconsistencies, and chaos detection.

        If no `perception` is provided, perceive local qualia and normalize the perception
        (including `resonance`) before running the analysis pipeline.
        """
        perception = perception if perception is not None else self.perceive_local_qualia()
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", 1.0))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))

        impulses = []
        base_intensity = self._calculate_base_intensity(perception)
        # Enhanced Doubt Injection
        doubt_impulse = self._inject_sophisticated_doubt(perception, base_intensity)
        if doubt_impulse:
            impulses.append(doubt_impulse)
        # Advanced Illusion Detection
        illusion_impulse = self._detect_sophisticated_illusions(
            perception, base_intensity
        )
        if illusion_impulse:
            impulses.append(illusion_impulse)
        # Chaos Pattern Recognition
        chaos_impulse = self._detect_chaos_patterns(perception, base_intensity)
        if chaos_impulse:
            impulses.append(chaos_impulse)
        # Existential Uncertainty Analysis
        uncertainty_impulse = self._analyze_existential_uncertainty(
            perception, base_intensity
        )
        if uncertainty_impulse:
            impulses.append(uncertainty_impulse)
        # Update node state
        total_intensity = sum(i.intensity for i in impulses)
        self._update_activation_energy(len(impulses), total_intensity)
        self._evolve_skepticism(impulses)
        return impulses

    def _calculate_base_intensity(self, data: dict[str, Any]) -> float:
        """Base intensity calculation for impulses."""
        qualia = data.get("qualia_state", {})
        chaos = qualia.get("chaos", 0.5)
        entropy = qualia.get("entropy", 0.5)
        return min(1.0, 0.3 + chaos * 0.4 + entropy * 0.3)

    def _inject_sophisticated_doubt(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Generate sophisticated doubt about validity, completeness, or assumptions."""
        doubt_factors = []
        doubt_strength = 0.0
        uncertainty_sources = []
        user_input = str(data.get("user_input", ""))
        # Linguistic ambiguity analysis
        ambiguous_words = ["maybe", "probably", "might", "could", "possibly", "perhaps"]
        ambiguity_count = sum(
            1 for word in ambiguous_words if word in user_input.lower()
        )
        if ambiguity_count > 0:
            doubt_factors.append(f"linguistic_ambiguity_{ambiguity_count}")
            doubt_strength += ambiguity_count * 0.15
            uncertainty_sources.append("language")
        # Information completeness assessment
        if len(user_input.split()) < 3:
            doubt_factors.append("insufficient_information")
            doubt_strength += 0.25
            uncertainty_sources.append("completeness")
        # Contradiction detection with context
        contradictions = self._find_sophisticated_contradictions(user_input)
        if contradictions:
            doubt_factors.extend(contradictions)
            doubt_strength += len(contradictions) * 0.2
            uncertainty_sources.append("internal_consistency")
        # Temporal inconsistency
        current_time = time.time()
        timestamp = data.get("timestamp", current_time)
        time_gap = abs(current_time - timestamp)
        if time_gap > 300:  # 5 minutes
            doubt_factors.append("temporal_displacement")
            doubt_strength += min(0.3, time_gap / 3600)  # Scale by hours
            uncertainty_sources.append("temporal")
        # Probabilistic doubt injection based on skepticism level
        doubt_threshold = 0.5 - self.skepticism_level
        if random.random() < doubt_threshold:
            doubt_factors.append("existential_skepticism")
            doubt_strength += self.skepticism_level * 0.2
            uncertainty_sources.append("existential")
        # Pattern-based doubt from history
        doubt_pattern = self._analyze_doubt_patterns()
        if doubt_pattern:
            doubt_factors.append(f"historical_doubt_pattern_{doubt_pattern}")
            doubt_strength += 0.18
            uncertainty_sources.append("historical")
        if doubt_strength > self.activation_threshold:
            content = (
                f"Epistemic uncertainty detected: {', '.join(doubt_factors)}. "
                f"Reality matrix shows potential instabilities. "
                f"Confidence in perception: {1.0 - doubt_strength:.2f}"
            )
            # Store doubt pattern for learning
            self.doubt_patterns.append(
                {
                    "factors": doubt_factors,
                    "strength": doubt_strength,
                    "sources": uncertainty_sources,
                    "timestamp": time.time(),
                }
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.DOUBT_INJECTION,
                content=content,
                intensity=min(1.0, base_intensity + doubt_strength),
                confidence=doubt_strength,
                source_node=self.node_name,
                metadata={
                    "doubt_factors": doubt_factors,
                    "uncertainty_sources": uncertainty_sources,
                    "skepticism_level": self.skepticism_level,
                },
            )
        return None

    def _detect_sophisticated_illusions(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Detect sophisticated illusions, deceptions, or reality distortions."""
        illusion_indicators = []
        illusion_strength = 0.0
        detection_methods = []
        # Memory-based repetition detection
        data_signature = str(sorted(data.items()))
        self.inconsistency_memory.append(
            {
                "signature": data_signature,
                "timestamp": time.time(),
                "hash": hash(data_signature),
            }
        )
        # Check for exact repetition patterns
        if len(self.inconsistency_memory) >= 3:
            recent_signatures = [
                item["signature"] for item in list(self.inconsistency_memory)[-10:]
            ]
            repetition_count = (
                recent_signatures.count(data_signature) - 1
            )  # Exclude current
            if repetition_count > 0:
                illusion_indicators.append(f"repetition_loop_{repetition_count}")
                illusion_strength += min(0.4, repetition_count * 0.15)
                detection_methods.append("memory_comparison")
        # Semantic consistency analysis
        semantic_inconsistencies = []
        if "user_input" in data:
            semantic_inconsistencies = self._detect_semantic_inconsistencies(
                data["user_input"]
            )
            if semantic_inconsistencies:
                illusion_indicators.extend(semantic_inconsistencies)
                illusion_strength += len(semantic_inconsistencies) * 0.12
                detection_methods.append("semantic_analysis")
        # Reality coherence assessment
        coherence_score = self._assess_reality_coherence(data.get("user_input", ""))
        if coherence_score < 0.3:
            illusion_indicators.append(f"low_reality_coherence_{coherence_score:.2f}")
            illusion_strength += (0.3 - coherence_score) * 0.5
            detection_methods.append("coherence_analysis")
        # Chaos attractor detection
        attractor_strength = self._detect_chaos_attractors(data)
        if attractor_strength > 0.2:
            illusion_indicators.append(f"chaos_attractor_{attractor_strength:.2f}")
            illusion_strength += attractor_strength * 0.3
            detection_methods.append("chaos_theory")
        if illusion_strength > self.activation_threshold:
            content = (
                f"Reality distortion detected: {', '.join(illusion_indicators)}. "
                f"Perception may be compromised by {', '.join(detection_methods)}. "
                f"Illusion probability: {illusion_strength:.2f}"
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.ILLUSION_DETECTION,
                content=content,
                intensity=min(1.0, base_intensity + illusion_strength),
                confidence=illusion_strength,
                source_node=self.node_name,
                metadata={
                    "illusion_indicators": illusion_indicators,
                    "strength": illusion_strength,
                    "detection_methods": detection_methods,
                    "reality_coherence": coherence_score,
                },
            )
        return None

    def _detect_chaos_patterns(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Detect patterns of chaos and emergent complexity."""
        chaos_indicators = []
        chaos_strength = 0.0
        # Information complexity analysis
        data_str = str(data)
        complexity_score = len(set(data_str)) / len(data_str) if data_str else 0
        if complexity_score > 0.7:  # High character diversity indicates complexity
            chaos_indicators.append(f"high_complexity_{complexity_score:.2f}")
            chaos_strength += complexity_score * 0.2
        # Entropy calculation for chaos detection
        char_counts: dict[str, int] = {}
        for char in data_str:
            char_counts[char] = char_counts.get(char, 0) + 1
        if char_counts:
            entropy: float = 0.0
            total_chars = len(data_str)
            for count in char_counts.values():
                p = count / total_chars
                entropy -= p * math.log2(p) if p > 0 else 0
            normalized_entropy = entropy / 8.0  # Normalize
            if normalized_entropy > 0.8:  # High entropy indicates chaos
                chaos_indicators.append(f"high_entropy_{normalized_entropy:.2f}")
                chaos_strength += normalized_entropy * 0.25
        if chaos_strength > self.activation_threshold:
            content = (
                f"Chaos patterns detected: {', '.join(chaos_indicators)}. "
                f"System exhibits emergent complexity and unpredictable behaviors."
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=content,
                intensity=min(1.0, base_intensity + chaos_strength),
                confidence=chaos_strength,
                source_node=self.node_name,
                metadata={
                    "chaos_indicators": chaos_indicators,
                    "strength": chaos_strength,
                    "complexity_score": complexity_score,
                    "entropy": (
                        normalized_entropy if "normalized_entropy" in locals() else None
                    ),
                },
            )
        return None

    def _analyze_existential_uncertainty(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Analyze deep existential uncertainties and philosophical doubts."""
        uncertainty_strength = 0.0
        existential_factors = []
        user_input = str(data.get("user_input", "")).lower()
        existential_keywords = [
            "meaning",
            "purpose",
            "existence",
            "reality",
            "consciousness",
            "being",
            "identity",
            "self",
            "truth",
            "certainty",
        ]
        existential_count = sum(
            1 for keyword in existential_keywords if keyword in user_input
        )
        if existential_count > 0:
            existential_factors.append(f"existential_themes_{existential_count}")
            uncertainty_strength += existential_count * 0.1
        # Meta-cognitive uncertainty about own processing
        if self.activation_energy > 0.7:  # High activation might indicate confusion
            existential_factors.append("meta_cognitive_uncertainty")
            uncertainty_strength += 0.15
        if uncertainty_strength > self.activation_threshold:
            content = (
                f"Existential uncertainty emerges: {', '.join(existential_factors)}. "
                f"Questioning the foundation of knowledge and experience."
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.EXISTENTIAL_UNCERTAINTY,
                content=content,
                intensity=min(1.0, base_intensity + uncertainty_strength),
                confidence=uncertainty_strength,
                source_node=self.node_name,
                metadata={
                    "existential_factors": existential_factors,
                    "strength": uncertainty_strength,
                    "theme_count": existential_count,
                },
            )
        return None

    def _find_sophisticated_contradictions(self, user_input: str) -> list[str]:
        """Find sophisticated contradictions in the user input."""
        contradictions = []
        contradiction_pairs = [
            (["yes", "true", "correct"], ["no", "false", "wrong"]),
            (["always"], ["never"]),
            (["everything"], ["nothing"]),
            (["increase", "grow"], ["decrease", "shrink"]),
            (["possible"], ["impossible"]),
        ]
        for positive, negative in contradiction_pairs:
            has_positive = any(word in user_input for word in positive)
            has_negative = any(word in user_input for word in negative)
            if has_positive and has_negative:
                contradictions.append(
                    f"logical_contradiction_{positive[0]}_vs_{negative[0]}"
                )
        return contradictions

    def _detect_semantic_inconsistencies(self, text: str) -> list[str]:
        """Detect semantic inconsistencies in text."""
        inconsistencies = []
        text_lower = text.lower()
        if "i don't know" in text_lower and any(
            word in text_lower for word in ["definitely", "certainly", "sure"]
        ):
            inconsistencies.append("knowledge_certainty_contradiction")
        return inconsistencies

    def _assess_reality_coherence(self, user_input: str) -> float:
        """Assess how coherent the user input is with expected reality."""
        coherence_score = 1.0
        if "yesterday tomorrow" in user_input or "future past" in user_input:
            coherence_score -= 0.3
        impossible_phrases = ["square circle", "colorless green", "silent sound"]
        for phrase in impossible_phrases:
            if phrase in user_input:
                coherence_score -= 0.2
        return max(0.0, coherence_score)

    def _detect_chaos_attractors(self, data: dict[str, Any]) -> float:
        """Detect strange attractors that might indicate chaotic dynamics."""
        attractor_strength = 0.0
        data_hash = hash(str(data))
        if data_hash in self.chaos_attractors:
            self.chaos_attractors[data_hash] += 1
            attractor_strength = min(0.5, self.chaos_attractors[data_hash] * 0.1)
        else:
            self.chaos_attractors[data_hash] = 1
        return attractor_strength

    def _analyze_doubt_patterns(self) -> str | None:
        """Analyze historical doubt patterns to predict future doubts."""
        if len(self.doubt_patterns) < 3:
            return None
        recent_patterns = list(self.doubt_patterns)[-5:]
        common_factors: dict[str, int] = {}
        for pattern in recent_patterns:
            for factor in pattern["factors"]:
                common_factors[factor] = common_factors.get(factor, 0) + 1
        if common_factors:
            most_common = max(common_factors, key=lambda k: common_factors.get(k, 0))
            if common_factors[most_common] >= 2:
                return most_common
        return None

    def _evolve_skepticism(self, impulses: list[CognitiveImpulse]):
        """Evolve skepticism level based on recent impulses."""
        doubt_impulses = [
            i for i in impulses if i.impulse_type == ImpulseType.DOUBT_INJECTION
        ]
        illusion_impulses = [
            i for i in impulses if i.impulse_type == ImpulseType.ILLUSION_DETECTION
        ]
        total_doubt_intensity = sum(
            i.intensity for i in doubt_impulses + illusion_impulses
        )
        if total_doubt_intensity > 0.5:
            self.skepticism_level = min(0.9, self.skepticism_level + 0.05)
        elif total_doubt_intensity < 0.1:
            self.skepticism_level = max(0.1, self.skepticism_level - 0.02)

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de caos/duda/ilusión en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_GAMALIEL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        bytecode = __import__(
            "crisalida_lib.EDEN.utils", fromlist=["*"]
        ).compile_intention_safe(_dc, intention)
        experience_id = (
            experience_data.get("timestamp") or f"exp_{hash(str(experience_data))}"
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = bytecode
        self.eva_memory_store[experience_id] = bytecode
        return experience_id

    def eva_recall_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        bytecode = self.eva_phases.get(phase, {}).get(cue) or self.eva_memory_store.get(
            cue
        )
        if not bytecode:
            return {"error": "No bytecode found for experience"}
        runtime = getattr(self, "eva_runtime", None)
        simulation_state: dict[str, Any]
        if runtime is None:
            simulation_state = {}
        else:
            exec_res = runtime.execute_bytecode(bytecode)
            simulation_state = (
                __import__(
                    "crisalida_lib.EDEN.utils", fromlist=["*"]
                ).run_maybe_awaitable(exec_res)
                or {}
            )
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                logger.warning(f"EVA Gamaliel environment hook failed: {e}")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de caos/duda/ilusión.
        """
        intention = {
            "intention_type": "ARCHIVE_GAMALIEL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        compile_fn = (
            getattr(_dc, "compile_intention", None) if _dc is not None else None
        )
        if callable(compile_fn):
            try:
                bytecode = compile_fn(intention)
            except Exception:
                bytecode = []
        else:
            bytecode = []
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self._current_phase = phase

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
