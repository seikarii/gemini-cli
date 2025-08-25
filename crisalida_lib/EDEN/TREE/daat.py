"""
DaatNode - Nodo Cognitivo de Conocimiento y Revelación (Da'at).

Procesa percepciones para generar impulsos de verdad, revelación y disolución de ilusiones.
Actúa como puente entre los nodos Sephirot, facilitando la integración y el despertar cognitivo.
Integración avanzada con EVA: memoria viviente, faseo, hooks de entorno y simulación activa.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional
import inspect

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

# Prefer the static `CosmicNode` type for type-checking; at runtime try import and
# fall back to Any to avoid import cycles and keep behavior unchanged.
if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
    CosmicNode = Any
    DivineSignature = Any

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
else:
    # runtime fallbacks to avoid import-time cycles
    LivingSymbolRuntime = Any  # type: ignore
    DivineLanguageEvolved = Any  # type: ignore


class DaatNode(CosmicNode):
    """
    Daat (Conocimiento) - Nodo Cognitivo de Revelación y Disolución de Ilusiones.
    Procesa percepciones para generar impulsos de verdad y trascendencia.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "daat_knowledge",
        mass: float = 4.5,
        influence_radius: float = 3.5,
    ):
        # pre-create DivineSignature and attempt to forward it to the base
        daat_divine_signature = None
        daat_divine_signature = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore

            daat_divine_signature = DivineSignature(glyph="Δ")
        except Exception:
            daat_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="sephirot",
                node_name=node_name,
                divine_signature=daat_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.activation_threshold: float = 0.22
        self.memory_patterns: list[dict[str, Any]] = []
        self.revelation_history: list[dict[str, Any]] = []
        self.illusion_history: list[dict[str, Any]] = []

        # Keep signature attribute for backward compatibility
        self.divine_signature = daat_divine_signature

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
        perception = perception if perception is not None else self.perceive_local_qualia()
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

        # Impulso de Revelación de la Verdad
        revelation_content = f"Verdad revelada. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
        revelation_intensity = activation * perception.get("sensory_clarity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.PATTERN_RECOGNITION,
                content=revelation_content,
                intensity=revelation_intensity,
                confidence=0.9,
                source_node=self.node_name,
            )
        )
        self.revelation_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "clarity": perception.get("sensory_clarity", 0.0),
                "intensity": revelation_intensity,
            }
        )

        # Impulso de Disolución de Ilusiones
        illusion_dissolution_content = f"Ilusión disolviéndose. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        illusion_intensity = activation * (
            1 - perception.get("cognitive_complexity", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.ILLUSION_DETECTION,
                content=illusion_dissolution_content,
                intensity=illusion_intensity,
                confidence=0.85,
                source_node=self.node_name,
            )
        )
        self.illusion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": illusion_intensity,
            }
        )

        # Impulso de Trascendencia (si awareness alto)
        transcendence_intensity = 0.0
        if perception.get("transcendent_awareness", 0.0) > 0.6:
            transcendence_content = f"Trascendencia activada. Nivel: {perception.get('transcendent_awareness', 0.0):.2f}"
            transcendence_intensity = activation * perception.get(
                "transcendent_awareness", 0.0
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.FUTURE_PROJECTION,
                    content=transcendence_content,
                    intensity=transcendence_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )

        # EVA: Archivar experiencia como bytecode de revelación/conocimiento
        qualia_state = {
            "revelation": revelation_intensity,
            "illusion": illusion_intensity,
            "transcendence": transcendence_intensity,
            "clarity": perception.get("sensory_clarity", 0.0),
            "complexity": perception.get("cognitive_complexity", 0.0),
            "awareness": perception.get("transcendent_awareness", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Daat environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Daat según la percepción.
        """
        clarity = perception.get("sensory_clarity", 0.5)
        complexity = perception.get("cognitive_complexity", 0.5)
        awareness = perception.get("transcendent_awareness", 0.0)
        activation = (clarity * 0.5 + awareness * 0.3) * (1.0 - complexity * 0.2)
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[-80:]

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de revelación/conocimiento en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_DAAT_EXPERIENCE",
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
        experience_id = (
            experience_data.get("timestamp") or f"exp_{hash(str(experience_data))}"
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = bytecode
        self.eva_memory_store[experience_id] = bytecode
        return experience_id

    def eva_recall_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        """
        Ejecuta el RealityBytecode de una experiencia almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        bytecode = self.eva_phases.get(phase, {}).get(cue) or self.eva_memory_store.get(
            cue
        )
        if not bytecode:
            return {"error": "No bytecode found for experience"}
        _rt = getattr(self, "eva_runtime", None)
        if _rt is None:
            return {"error": "No EVA runtime available"}
        try:
            simulation_state = _rt.execute_bytecode(bytecode)
        except Exception:
            return {"error": "EVA execution failed"}
        # If runtime returned an awaitable (async API), don't attempt to run it here — return a safe dict
        if inspect.isawaitable(simulation_state):
            return {"status": "awaitable", "awaitable": True}
        # Normalize to a dict result for callers
        if not isinstance(simulation_state, dict):
            return {"result": simulation_state}
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de revelación/conocimiento.
        """
        intention = {
            "intention_type": "ARCHIVE_DAAT_EXPERIENCE",
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
