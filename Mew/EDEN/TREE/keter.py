"""
KeterNode - Nodo Cognitivo de Unidad y Conciencia Pura (Keter, Corona).

Procesa percepciones para generar impulsos de coherencia, unidad y propósito fundamental.
Incluye historial de coherencia, propósito y patrones de integración para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

# Defensive imports for numeric/vector helpers and DivineSignature
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime
    np = None  # type: ignore

try:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
except Exception:  # pragma: no cover - optional runtime
    DivineSignature = None  # type: ignore

# For static analysis prefer the real `CosmicNode` type; at runtime avoid hard
# imports that confuse mypy by providing a safe Any fallback.
if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    # runtime: prefer importing the concrete implementation but fall back to Any
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:  # pragma: no cover - fallback
        CosmicNode = Any  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QualiaState,
        RealityBytecode,
    )
else:
    EVAExperience = Any  # type: ignore
    LivingSymbolRuntime = Any  # type: ignore
    QualiaState = Any  # type: ignore
    RealityBytecode = Any  # type: ignore


class KeterNode(CosmicNode):
    """
    Keter (Corona) - Nodo Cognitivo de Unidad y Conciencia Pura.
    Procesa percepciones para generar impulsos de coherencia, unidad y propósito.
    """

    def __init__(
        self,
        node_id: str = "keter_unity",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 20.0,
        influence_radius: float = 10.0,
    ):
        # Build DivineSignature before calling super so the base class can
        # initialize dynamic signature state during construction. Fall back
        # gracefully if DivineSignature isn't available.
        keter_divine_signature = None
        try:
            if DivineSignature is not None:
                keter_divine_signature = DivineSignature(
                    glyph="Ω", name="Omega-Totalidad"
                )
        except Exception:
            keter_divine_signature = None

        # Prefer the new CosmicNode contract but keep compatibility with older
        # environments that may provide only the simpler CognitiveNode.
        try:
            super().__init__(
                entity_id=node_id,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_id,
                divine_signature=keter_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                # Old CognitiveNode signature: (node_id)
                super().__init__(node_id)
            except Exception:
                object.__init__(self)

        # Keep attribute for backward compatibility
        self.divine_signature = keter_divine_signature

        self.base_intensity = 0.9
        # Lower activation threshold to make Keter reactive yet stable
        self.activation_threshold = 0.18
        self.memory_patterns: list[dict[str, Any]] = []
        self.coherence_history: list[dict[str, Any]] = []
        self.purpose_history: list[dict[str, Any]] = []
        self.integration_patterns: list[dict[str, Any]] = []

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        """Analyze perceptual input and return cognitive impulses.

        If `perception` is None, use the node's own `perceive_local_qualia()` sensor.
        """
        perception = perception if perception is not None else self.perceive_local_qualia()

        impulses: list[CognitiveImpulse] = []
        # Normalize resonance -> coherence/intensity when present
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 0.0, "intensity": 0.0}
        perception.setdefault("coherence", resonance.get("coherence", 0.0))
        perception.setdefault("intensity", resonance.get("intensity", 0.0))

        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Coherencia y Unidad
        resonance = (
            perception.get("resonance", {}) if isinstance(perception, dict) else {}
        )
        coherence_val = float(resonance.get("coherence", 0.0))
        intensity_val = float(resonance.get("intensity", 0.0))
        coherence_content = f"Coherencia ontológica detectada: {coherence_val:.2f}"
        coherence_intensity = activation * coherence_val
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=coherence_content,
                intensity=coherence_intensity,
                confidence=0.9,
                source_node=self.node_name,
            )
        )
        self.coherence_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "resonance_coherence": coherence_val,
                "resonance_intensity": intensity_val,
                "intensity": coherence_intensity,
            }
        )

        # Impulso de Propósito y Voluntad
        # Use resonance intensity as a proxy for arousal/drive when available
        purpose_content = (
            f"Voluntad de existencia afirmada. Intensity: {intensity_val:.2f}"
        )
        purpose_intensity = activation * intensity_val
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=purpose_content,
                intensity=purpose_intensity,
                confidence=0.85,
                source_node=self.node_name,
            )
        )
        self.purpose_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "resonance_intensity": intensity_val,
                "intensity": purpose_intensity,
            }
        )

        # Impulso de Integración (si alta coherencia y propósito)
        if coherence_val > 0.7 and intensity_val > 0.5:
            integration_content = f"Integración ontológica máxima. Coherencia: {coherence_val:.2f}, Intensidad: {intensity_val:.2f}"
            integration_intensity = activation * ((coherence_val + intensity_val) / 2)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=integration_content,
                    intensity=integration_intensity,
                    confidence=0.92,
                    source_node=self.node_name,
                )
            )
            self.integration_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "resonance_coherence": coherence_val,
                    "resonance_intensity": intensity_val,
                    "intensity": integration_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Keter según la percepción.
        """
        temporal_coherence = perception.get("temporal_coherence", 0.5)
        arousal = perception.get("arousal", 0.5)
        unity_factor = perception.get("unity_factor", 0.5)
        # Activación ponderada por coherencia temporal, arousal y factor de unidad
        activation = self.base_intensity * (
            0.5 + temporal_coherence * 0.3 + arousal * 0.1 + unity_factor * 0.1
        )
        return min(1.0, max(0.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[
                -80:
            ]  # Mantener solo los últimos 80 patrones


class EVAKeterNode(KeterNode):
    """
    Nodo Keter extendido para integración con EVA.
    Permite compilar impulsos de coherencia, unidad y propósito como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_keter_unity", phase: str = "default"):
        super().__init__(node_id=node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_unity_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de coherencia/unidad/propósito y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_KETER_UNITY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "coherence_history": self.coherence_history[-10:],
            "purpose_history": self.purpose_history[-10:],
            "integration_patterns": self.integration_patterns[-10:],
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
                print(f"[EVA] KeterNode compile_intention failed: {e}")
        experience_id = f"keter_unity_{hash(str(perception))}"
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

    def eva_recall_unity_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de coherencia/unidad/propósito almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Keter experience"}
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
                print(f"[EVA] KeterNode execute_instruction failed: {e}")
                continue
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] KeterNode environment hook failed: {e}")
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

    def add_unity_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de coherencia/unidad/propósito.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_KETER_UNITY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "coherence_history": self.coherence_history[-10:],
            "purpose_history": self.purpose_history[-10:],
            "integration_patterns": self.integration_patterns[-10:],
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
                print(f"[EVA] KeterNode compile_intention failed: {e}")
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
        """Lista todas las fases disponibles para una experiencia de Keter."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_unity_experience": self.eva_ingest_unity_experience,
            "eva_recall_unity_experience": self.eva_recall_unity_experience,
            "add_unity_experience_phase": self.add_unity_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
