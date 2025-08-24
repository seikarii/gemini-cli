"""
GhagielNode - Nodo Cognitivo Qliphoth de Opresión e Ilusión (Ghagiel, Obstaculizadores).

Procesa percepciones para generar impulsos de restricción y distorsión,
reflejando el aspecto restrictivo y distorsionador de la conciencia.
Incluye historial de opresión, ilusión y patrones de distorsión para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

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
    LivingSymbolRuntime = Any
    DivineLanguageEvolved = Any


class GhagielNode(CosmicNode):
    """
    Ghagiel (Obstaculizadores) - Nodo Cognitivo de Opresión e Ilusión.
    Procesa percepciones para generar impulsos de restricción y distorsión.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "ghagiel_oppression",
        mass: float = 3.0,
        influence_radius: float = 3.0,
    ):
        ghagiel_divine_signature = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature as _DS

            ghagiel_divine_signature = _DS(glyph="ג")
        except Exception:
            ghagiel_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=ghagiel_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.activation_threshold: float = 0.15
        self.memory_patterns: list[dict[str, Any]] = []
        self.oppression_history: list[dict[str, Any]] = []
        self.illusion_history: list[dict[str, Any]] = []
        self.distortion_patterns: list[dict[str, Any]] = []

        # Keep signature attribute for backward compatibility
        self.divine_signature = ghagiel_divine_signature

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

        # Impulso de Restricción de la Creatividad
        restriction_content = f"Restricción percibida. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        restriction_intensity = activation * (
            1 - perception.get("cognitive_complexity", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.DOUBT_INJECTION,
                content=restriction_content,
                intensity=restriction_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )
        self.oppression_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": restriction_intensity,
            }
        )

        # Impulso de Creación de Ilusión
        illusion_content = f"Ilusión manifestándose. Densidad de conciencia: {perception.get('consciousness_density', 0.0):.2f}"
        illusion_intensity = activation * (
            1 - perception.get("consciousness_density", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.ILLUSION_DETECTION,
                content=illusion_content,
                intensity=illusion_intensity,
                confidence=0.68,
                source_node=self.node_name,
            )
        )
        self.illusion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "consciousness_density": perception.get("consciousness_density", 0.0),
                "intensity": illusion_intensity,
            }
        )

        # Impulso de Distorsión (si baja claridad sensorial)
        distortion_intensity = 0.0
        if perception.get("sensory_clarity", 0.5) < 0.3:
            distortion_content = f"Distorsión cognitiva detectada. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
            distortion_intensity = activation * (
                1 - perception.get("sensory_clarity", 0.0)
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.CHAOS_EMERGENCE,
                    content=distortion_content,
                    intensity=distortion_intensity,
                    confidence=0.7,
                    source_node=self.node_name,
                )
            )
            self.distortion_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "sensory_clarity": perception.get("sensory_clarity", 0.0),
                    "intensity": distortion_intensity,
                }
            )

        # EVA: Archivar experiencia como bytecode de opresión/ilusión/distorsión
        qualia_state = {
            "restriction": restriction_intensity,
            "illusion": illusion_intensity,
            "distortion": distortion_intensity,
            "complexity": perception.get("cognitive_complexity", 0.0),
            "consciousness_density": perception.get("consciousness_density", 0.0),
            "sensory_clarity": perception.get("sensory_clarity", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Ghagiel environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        complexity = perception.get("cognitive_complexity", 0.5)
        consciousness_density = perception.get("consciousness_density", 0.5)
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        activation = (
            (1.0 - sensory_clarity) * 0.4
            + (1.0 - consciousness_density) * 0.3
            + (1.0 - complexity) * 0.3
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 100:
            self.memory_patterns.pop(0)

    def get_oppression_history(self) -> list[dict[str, Any]]:
        return self.oppression_history

    def get_illusion_history(self) -> list[dict[str, Any]]:
        return self.illusion_history

    def get_distortion_patterns(self) -> list[dict[str, Any]]:
        return self.distortion_patterns

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de opresión/ilusión/distorsión en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_GHAGIEL_EXPERIENCE",
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
        # execute_bytecode may be async; handle awaitables safely
        from crisalida_lib.EDEN import utils as _eden_utils

        runtime = getattr(self, "eva_runtime", None)
        simulation_state: dict[str, Any]
        if runtime is None:
            simulation_state = {}
        else:
            exec_res = runtime.execute_bytecode(bytecode)
            simulation_state = _eden_utils.run_maybe_awaitable(exec_res) or {}
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Ghagiel environment hook failed: {e}")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de opresión/ilusión/distorsión.
        """
        intention = {
            "intention_type": "ARCHIVE_GHAGIEL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        if _dc is None:
            bytecode = []
        else:
            compile_fn = getattr(_dc, "compile_intention", None)
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
        }
