"""
HodNode - Nodo Cognitivo de Esplendor, Lógica y Reconocimiento de Patrones (Hod, Splendor).

Procesa percepciones para generar impulsos de orden, estructuras lógicas y reconocimiento avanzado de patrones.
Incluye memoria de patrones, frameworks lógicos y análisis meta-cognitivo para diagnóstico y simulación avanzada.
"""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

# For static typing import the real types; at runtime use Any fallbacks
if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
    CosmicNode = Any
    DivineSignature = Any

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QualiaState,
        RealityBytecode,
    )
else:
    EVAExperience = Any
    LivingSymbolRuntime = Any
    QualiaState = Any
    RealityBytecode = Any

logger = logging.getLogger(__name__)


class HodNode(CosmicNode):
    """
    Hod (Esplendor/Lógica) - Nodo Cognitivo de Orden y Reconocimiento de Patrones.
    Procesa percepciones para generar impulsos de orden, lógica y patrones.
    """

    def __init__(
        self,
        node_name: str = "hod",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
        activation_threshold: float = 0.1,
    ) -> None:
        # Pre-create divine signature
        hod_divine_signature = None
        try:
            if DivineSignature is not None:
                hod_divine_signature = DivineSignature(glyph="Η")
        except Exception:
            hod_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                divine_signature=hod_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = hod_divine_signature
        self.activation_threshold = activation_threshold
        self.pattern_memory: dict[str, int] = {}
        self.logical_frameworks: deque[Any] = deque(maxlen=20)
        self.pattern_confidence_history: deque[float] = deque(maxlen=50)

    def analyze(
        self, perception_data: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        """
        Genera impulsos cognitivos enfocados en orden, patrones y estructura lógica.
        If no perception_data is provided, use local qualia via perceive_local_qualia().
        """
        perception_data = (
            perception_data
            if perception_data is not None
            else self.perceive_local_qualia()
        )
        if isinstance(perception_data, dict) and "resonance" in perception_data:
            resonance = perception_data["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception_data, dict):
            perception_data.setdefault("coherence", resonance.get("coherence", 1.0))
            perception_data.setdefault("intensity", resonance.get("intensity", 1.0))

        impulses = []
        base_intensity = self._calculate_base_intensity(perception_data)
        # Reconocimiento avanzado de patrones
        pattern_impulse = self._analyze_patterns(perception_data, base_intensity)
        if pattern_impulse:
            impulses.append(pattern_impulse)
        # Análisis avanzado de estructura lógica
        structure_impulse = self._analyze_logical_structure(
            perception_data, base_intensity
        )
        if structure_impulse:
            impulses.append(structure_impulse)
        # Evaluación meta-cognitiva de orden
        meta_impulse = self._assess_cognitive_order(perception_data, base_intensity)
        if meta_impulse:
            impulses.append(meta_impulse)
        # Actualizar energía de activación
        total_intensity = sum(i.intensity for i in impulses)
        self._update_activation_energy(len(impulses), total_intensity)
        return impulses

    def _calculate_base_intensity(self, data: dict[str, Any]) -> float:
        """Calcula la intensidad base según la coherencia y complejidad."""
        coherence = data.get("coherence", 0.5)
        complexity = data.get("cognitive_complexity", 0.5)
        return 0.2 + coherence * 0.5 + (1.0 - complexity) * 0.3

    def _analyze_patterns(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Análisis avanzado de patrones con memoria y seguimiento de confianza."""
        pattern_strength = 0.0
        detected_patterns = []
        confidence_factors = []
        # Detección de patrones numéricos
        numbers = [v for v in data.values() if isinstance(v, (int, float))]
        if len(numbers) >= 2:
            if numbers == sorted(numbers):
                detected_patterns.append("ascending_sequence")
                pattern_strength += 0.3
                confidence_factors.append(0.8)
            elif numbers == sorted(numbers, reverse=True):
                detected_patterns.append("descending_sequence")
                pattern_strength += 0.3
                confidence_factors.append(0.8)
            if len(numbers) > 3:
                variance = sum(
                    (x - sum(numbers) / len(numbers)) ** 2 for x in numbers
                ) / len(numbers)
                if variance < 0.1:
                    detected_patterns.append("statistical_consistency")
                    pattern_strength += 0.2
                    confidence_factors.append(0.6)
        # Patrones estructurales
        string_values = [str(v) for v in data.values()]
        if len(set(string_values)) < len(string_values):
            repetition_ratio = 1 - (len(set(string_values)) / len(string_values))
            detected_patterns.append(f"repeated_elements_{repetition_ratio:.2f}")
            pattern_strength += repetition_ratio * 0.3
            confidence_factors.append(repetition_ratio)
        # Consistencia de tipos
        value_types = [type(v).__name__ for v in data.values()]
        if len(set(value_types)) == 1:
            detected_patterns.append("perfect_type_consistency")
            pattern_strength += 0.15
            confidence_factors.append(0.9)
        elif len(set(value_types)) <= len(value_types) * 0.5:
            detected_patterns.append("partial_type_consistency")
            pattern_strength += 0.1
            confidence_factors.append(0.5)
        # Refuerzo histórico de patrones
        pattern_signature = str(sorted(detected_patterns))
        if pattern_signature in self.pattern_memory:
            self.pattern_memory[pattern_signature] += 1
            pattern_strength += min(0.2, self.pattern_memory[pattern_signature] * 0.02)
            confidence_factors.append(0.7)
        else:
            self.pattern_memory[pattern_signature] = 1
        # Calcular confianza final
        overall_confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.0
        )
        self.pattern_confidence_history.append(overall_confidence)
        if pattern_strength > self.activation_threshold:
            confidence_trend = "stable"
            if len(self.pattern_confidence_history) > 5:
                recent_conf = list(self.pattern_confidence_history)[-5:]
                if recent_conf[-1] > recent_conf[0]:
                    confidence_trend = "increasing"
                elif recent_conf[-1] < recent_conf[0]:
                    confidence_trend = "decreasing"
            content = (
                f"Pattern recognition active: {', '.join(detected_patterns)}. "
                f"Confidence: {overall_confidence:.2f} ({confidence_trend}). "
                f"Structural coherence confirmed with {pattern_strength:.2f} certainty."
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.PATTERN_RECOGNITION,
                content=content,
                intensity=min(1.0, base_intensity + pattern_strength),
                confidence=overall_confidence,
                source_node=self.node_name,
                metadata={
                    "patterns": detected_patterns,
                    "strength": pattern_strength,
                    "confidence_factors": confidence_factors,
                    "confidence_trend": confidence_trend,
                    "historical_reinforcement": self.pattern_memory.get(
                        pattern_signature, 0
                    ),
                },
            )
        return None

    def _analyze_logical_structure(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Análisis avanzado de relaciones lógicas y estructura."""
        logical_strength = 0.0
        structural_elements = []
        complexity_metrics = {}

        # Análisis de profundidad jerárquica
        def calculate_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max(
                    [calculate_depth(v, current_depth + 1) for v in obj.values()]
                    + [current_depth]
                )
            elif isinstance(obj, list):
                return max(
                    [calculate_depth(item, current_depth + 1) for item in obj]
                    + [current_depth]
                )
            else:
                return current_depth

        max_depth = max([calculate_depth(v) for v in data.values()] + [0])
        if max_depth > 0:
            structural_elements.append(f"hierarchical_depth_{max_depth}")
            logical_strength += min(0.3, max_depth * 0.1)
            complexity_metrics["hierarchical_depth"] = max_depth
        # Coherencia semántica de clave-valor
        if "user_input" in data:
            input_length = len(str(data["user_input"]))
            if input_length > 10:
                structural_elements.append("meaningful_user_input")
                logical_strength += min(0.3, input_length / 100.0)
                complexity_metrics["input_complexity"] = input_length
        # Análisis de referencias cruzadas
        string_data = [str(v) for v in data.values()]
        cross_references = 0
        for i, val1 in enumerate(string_data):
            for _j, val2 in enumerate(string_data[i + 1 :], i + 1):
                if len(val1) > 3 and val1.lower() in val2.lower():
                    cross_references += 1
        if cross_references > 0:
            structural_elements.append(f"cross_references_{cross_references}")
            logical_strength += min(0.2, cross_references * 0.05)
            complexity_metrics["cross_references"] = cross_references
        # Evaluación de frameworks lógicos
        framework_score = self._assess_logical_framework(data)
        if framework_score > 0.1:
            structural_elements.append(f"logical_framework_{framework_score:.2f}")
            logical_strength += framework_score
            complexity_metrics["framework_score"] = framework_score
        if logical_strength > self.activation_threshold:
            content = (
                f"Logical architecture detected: {', '.join(structural_elements)}. "
                f"Information demonstrates organized structure with "
                f"complexity score {sum(complexity_metrics.values()):.2f}."
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=content,
                intensity=min(1.0, base_intensity + logical_strength),
                confidence=min(1.0, logical_strength * 1.2),
                source_node=self.node_name,
                metadata={
                    "structures": structural_elements,
                    "strength": logical_strength,
                    "complexity_metrics": complexity_metrics,
                    "framework_score": framework_score,
                },
            )
        return None

    def _assess_cognitive_order(
        self, data: dict[str, Any], base_intensity: float
    ) -> CognitiveImpulse | None:
        """Evalúa el orden cognitivo global y meta-patrones."""
        order_strength = 0.0
        meta_patterns = []
        # Cálculo de entropía informacional
        if data:
            string_rep = str(data)
            char_freq: dict[str, int] = {}
            for char in string_rep:
                char_freq[char] = char_freq.get(char, 0) + 1
            entropy: float = 0.0
            total_chars = len(string_rep)
            for freq in char_freq.values():
                p = freq / total_chars
                entropy -= p * math.log2(p) if p > 0 else 0
            normalized_entropy = (
                entropy / 8.0
            )  # Normalizar suponiendo entropía máxima ~8
            order_from_entropy = max(0, 1.0 - normalized_entropy)
            if order_from_entropy > 0.6:
                meta_patterns.append("high_information_order")
                order_strength += order_from_entropy * 0.3
        # Coherencia de energía de activación
        if self.activation_energy > 0.5:
            meta_patterns.append("sustained_activation")
            order_strength += 0.2
        if order_strength > self.activation_threshold:
            content = (
                f"Meta-cognitive order assessment: {', '.join(meta_patterns)}. "
                f"Consciousness demonstrates organized processing patterns."
            )
            return CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=content,
                intensity=min(1.0, base_intensity + order_strength),
                confidence=order_strength,
                source_node=self.node_name,
                metadata={
                    "meta_patterns": meta_patterns,
                    "order_strength": order_strength,
                    "entropy_analysis": (
                        normalized_entropy if "normalized_entropy" in locals() else None
                    ),
                },
            )
        return None

    def _assess_logical_framework(self, data: dict[str, Any]) -> float:
        """Evalúa la presencia de frameworks lógicos en los datos."""
        framework_score = 0.0
        text_content = " ".join(str(v) for v in data.values()).lower()
        causal_indicators = [
            "because",
            "therefore",
            "thus",
            "consequently",
            "as a result",
        ]
        for indicator in causal_indicators:
            if indicator in text_content:
                framework_score += 0.1
        conditional_indicators = ["if", "then", "unless", "provided that", "assuming"]
        for indicator in conditional_indicators:
            if indicator in text_content:
                framework_score += 0.08
        if framework_score > 0:
            self.logical_frameworks.append(
                {
                    "score": framework_score,
                    "timestamp": time.time(),
                    "data_signature": hash(str(data)),
                }
            )
        return framework_score


class EVAHodNode(HodNode):
    """
    Nodo Hod extendido para integración con EVA.
    Permite compilar impulsos de orden, lógica y patrones como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_hod_splendor", phase: str = "default"):
        # Allow HodNode to initialize with overridable node_name
        super().__init__(node_name=node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_splendor_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de orden/lógica/patrones y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_HOD_SPLENDOR_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "pattern_memory": dict(self.pattern_memory),
            "logical_frameworks": list(self.logical_frameworks),
            "pattern_confidence_history": list(self.pattern_confidence_history),
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive: eva_runtime or its divine_compiler may be None or missing
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if compile_fn:
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                logger.exception("[EVA] HodNode compile_intention failed: %s", e)
        experience_id = f"hod_splendor_{hash(str(perception))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_splendor_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de orden/lógica/patrones almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Hod experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        exec_fn = getattr(self.eva_runtime, "execute_instruction", None)
        for instr in reality_bytecode.instructions or []:
            if not exec_fn:
                continue
            try:
                symbol_manifest = exec_fn(instr, quantum_field)
            except Exception as e:
                logger.exception("[EVA] HodNode execute_instruction failed: %s", e)
                continue
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.exception("[EVA] HodNode environment hook failed: %s", e)
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
        }

    def add_splendor_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de orden/lógica/patrones.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_HOD_SPLENDOR_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "pattern_memory": dict(self.pattern_memory),
            "logical_frameworks": list(self.logical_frameworks),
            "pattern_confidence_history": list(self.pattern_confidence_history),
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive: eva_runtime or its divine_compiler may be None or missing
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if compile_fn:
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                logger.exception("[EVA] HodNode compile_intention failed: %s", e)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self._current_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de Hod."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_splendor_experience": self.eva_ingest_splendor_experience,
            "eva_recall_splendor_experience": self.eva_recall_splendor_experience,
            "add_splendor_experience_phase": self.add_splendor_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
