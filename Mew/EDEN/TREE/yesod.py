"""
YesodNode - Nodo Cognitivo de Fundamento, Estabilidad Psíquica y Cohesión (Yesod, Fundamento).

Procesa percepciones para generar impulsos de estabilidad psíquica, cohesión de conciencia y anclaje a la realidad.
Incluye historial de cohesión, estabilidad y patrones de anclaje para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import Any

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

# Defensive numeric + divine signature imports (optional at runtime)
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime
    np = None  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
    DivineSignature = Any

from typing import TYPE_CHECKING, cast

from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    QualiaState,
    RealityBytecode,
)

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback
        CosmicNode = Any  # type: ignore


class YesodNode(CosmicNode):
    """
    Yesod (Fundamento) - Nodo Cognitivo de Conexión y Estabilidad Psíquica.
    Procesa percepciones para generar impulsos de cohesión y anclaje a la realidad.
    """

    def __init__(
        self,
        node_name: str = "yesod_foundation",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        # Build DivineSignature before calling super so base can initialize
        yesod_divine_signature = None
        try:
            if DivineSignature is not None:
                yesod_divine_signature = DivineSignature(glyph="Ψ")
        except Exception:
            yesod_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                divine_signature=yesod_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = yesod_divine_signature

        self.activation_threshold: float = 0.13
        self.memory_patterns: list[dict[str, Any]] = []
        self.cohesion_history: list[dict[str, Any]] = []
        self.stability_history: list[dict[str, Any]] = []
        self.anchoring_patterns: list[dict[str, Any]] = []

    def analyze(
        self, perception: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        perception = (
            perception if perception is not None else self.perceive_local_qualia()
        )
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", 1.0))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))
        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Cohesión de Conciencia
        cohesion_content = f"Cohesión de conciencia. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
        cohesion_intensity = activation * perception.get("sensory_clarity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=cohesion_content,
                intensity=cohesion_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )
        self.cohesion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "sensory_clarity": perception.get("sensory_clarity", 0.0),
                "intensity": cohesion_intensity,
            }
        )

        # Impulso de Estabilidad Psíquica
        stability_content = f"Anclaje psíquico. Foco cognitivo: {perception.get('cognitive_focus', 0.0):.2f}"
        stability_intensity = activation * perception.get("cognitive_focus", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=stability_content,
                intensity=stability_intensity,
                confidence=0.68,
                source_node=self.node_name,
            )
        )
        self.stability_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_focus": perception.get("cognitive_focus", 0.0),
                "intensity": stability_intensity,
            }
        )

        # Impulso de Anclaje a la Realidad (si alta claridad y foco)
        if (
            perception.get("sensory_clarity", 0.0) > 0.7
            and perception.get("cognitive_focus", 0.0) > 0.7
        ):
            anchoring_content = f"Anclaje a la realidad consolidado. Claridad: {perception.get('sensory_clarity', 0.0):.2f}, Foco: {perception.get('cognitive_focus', 0.0):.2f}"
            anchoring_intensity = (
                activation
                * (
                    perception.get("sensory_clarity", 0.0)
                    + perception.get("cognitive_focus", 0.0)
                )
                / 2
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=anchoring_content,
                    intensity=anchoring_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )
            self.anchoring_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "sensory_clarity": perception.get("sensory_clarity", 0.0),
                    "cognitive_focus": perception.get("cognitive_focus", 0.0),
                    "intensity": anchoring_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Yesod según la percepción.
        """
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        cognitive_focus = perception.get("cognitive_focus", 0.5)
        # Activación ponderada por claridad sensorial y foco cognitivo
        activation = sensory_clarity * 0.6 + cognitive_focus * 0.4
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[-80:]


class EVAYesodNode(YesodNode):
    """
    Nodo Yesod extendido para integración con EVA.
    Permite compilar impulsos cognitivos y patrones de cohesión como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_yesod_foundation", phase: str = "default"):
        super().__init__(node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_cohesion_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de cohesión/anclaje y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_YESOD_COHESION_EXPERIENCE",
            "perception": perception,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if _eva is None:
            bytecode = []
        else:
            _dc = getattr(_eva, "divine_compiler", None)
            if _dc is not None and hasattr(_dc, "compile_intention"):
                try:
                    bytecode = _dc.compile_intention(intention)
                except Exception:
                    bytecode = []
            else:
                bytecode = []
        experience_id = f"yesod_cohesion_{hash(str(perception))}"
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

    def eva_recall_cohesion_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de cohesión/anclaje almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Yesod experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        instrs = getattr(reality_bytecode, "instructions", [])
        if instrs is None:
            instrs = []
        _exec = getattr(self.eva_runtime, "execute_instruction", None)
        for instr in instrs:
            if _exec is None:
                continue
            try:
                symbol_manifest = _exec(instr, quantum_field)
            except Exception:
                symbol_manifest = None
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] YesodNode environment hook failed: {e}")
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

    def add_cohesion_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de cohesión/anclaje.
        """
        intention = {
            "intention_type": "ARCHIVE_YESOD_COHESION_EXPERIENCE",
            "perception": perception,
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
                print(f"[EVA] YesodNode compile_intention failed: {e}")
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
        """Lista todas las fases disponibles para una experiencia de Yesod."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_cohesion_experience": self.eva_ingest_cohesion_experience,
            "eva_recall_cohesion_experience": self.eva_recall_cohesion_experience,
            "add_cohesion_experience_phase": self.add_cohesion_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
