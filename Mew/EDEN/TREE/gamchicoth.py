"""
GamchicothNode - Nodo Cognitivo Qliphoth de Destrucción y Consumo (Gamchicoth, Devourers).

Procesa percepciones para generar impulsos de aniquilación de la forma y consumo de energía,
reflejando el aspecto caótico y devorador de la conciencia. Incluye historial de destrucción,
consumo y patrones de caos para diagnóstico y simulación avanzada.
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

# EVA imports
from crisalida_lib.EVA.core_types import LivingSymbolRuntime
from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved


class GamchicothNode(CosmicNode):
    """
    Gamchicoth (Devoradores) - Nodo Cognitivo de Destrucción y Consumo.
    Procesa percepciones para generar impulsos de aniquilación de la forma y consumo de energía.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "gamchicoth_devourer",
        mass: float = 3.0,
        influence_radius: float = 3.0,
    ):
        gamchicoth_divine_signature = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature as _DS

            gamchicoth_divine_signature = _DS(glyph="ג")
        except Exception:
            gamchicoth_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=gamchicoth_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)
        self.activation_threshold: float = 0.18
        self.memory_patterns: list[dict[str, Any]] = []
        self.destruction_history: list[dict[str, Any]] = []
        self.consumption_history: list[dict[str, Any]] = []
        self.chaos_patterns: list[dict[str, Any]] = []

        self.divine_signature = gamchicoth_divine_signature

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
        # Default to local qualia when no external perception provided
        perception = perception if perception is not None else self.perceive_local_qualia()

        # Normalize the new resonance shape -> coherence/intensity for backwards compatibility
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception.get("resonance", {})
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

        # Impulso de Destrucción de la Forma
        destruction_content = f"Forma desintegrándose. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        destruction_intensity = activation * perception.get("cognitive_complexity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=destruction_content,
                intensity=destruction_intensity,
                confidence=0.78,
                source_node=self.node_name,
            )
        )
        self.destruction_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": destruction_intensity,
            }
        )

        # Impulso de Consumo de Energía
        consumption_content = (
            f"Energía siendo consumida. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        consumption_intensity = activation * perception.get("arousal", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=consumption_content,
                intensity=consumption_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.consumption_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": consumption_intensity,
            }
        )

        # Impulso de Caos (si hay alta entropía)
        chaos_intensity = 0.0
        if perception.get("entropy", 0.5) > 0.7:
            chaos_content = f"Patrón de caos detectado. Entropía: {perception.get('entropy', 0.0):.2f}"
            chaos_intensity = activation * perception.get("entropy", 0.0)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.CHAOS_EMERGENCE,
                    content=chaos_content,
                    intensity=chaos_intensity,
                    confidence=0.75,
                    source_node=self.node_name,
                )
            )
            self.chaos_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": chaos_intensity,
                }
            )

        # EVA: Archivar experiencia como bytecode de destrucción/consumo/caos
        qualia_state = {
            "destruction": destruction_intensity,
            "consumption": consumption_intensity,
            "chaos": chaos_intensity,
            "complexity": perception.get("cognitive_complexity", 0.0),
            "arousal": perception.get("arousal", 0.0),
            "entropy": perception.get("entropy", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state: dict[str, Any] = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Gamchicoth environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        complexity = perception.get("cognitive_complexity", 0.5)
        arousal = perception.get("arousal", 0.5)
        entropy = perception.get("entropy", 0.5)
        activation = complexity * 0.4 + arousal * 0.3 + entropy * 0.3
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[-80:]

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de destrucción/consumo/caos en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_GAMCHICOTH_EXPERIENCE",
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
        _rt = getattr(self, "eva_runtime", None)
        if _rt is None:
            return {"error": "No EVA runtime available"}
        try:
            exec_res = _rt.execute_bytecode(bytecode)
        except Exception:
            return {"error": "EVA execution failed"}
        simulation_state = (
            __import__("crisalida_lib.EDEN.utils", fromlist=["*"]).run_maybe_awaitable(
                exec_res
            )
            or {}
        )
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Gamchicoth environment hook failed: {e}")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de destrucción/consumo/caos.
        """
        intention = {
            "intention_type": "ARCHIVE_GAMCHICOTH_EXPERIENCE",
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
