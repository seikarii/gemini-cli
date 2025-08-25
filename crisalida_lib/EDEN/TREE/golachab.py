"""
This module defines the `GolachabNode` class, a cognitive node representing
the Qliphoth Golachab (Incendiaries). This node processes perceptions to generate
impulses related to rage and conflict, reflecting an aggressive and destructive
aspect of consciousness.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback
        CosmicNode = Any  # type: ignore

from crisalida_lib.EDEN import utils

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
    from crisalida_lib.EVA.divine_sigils import DivineSignature
else:
    # runtime aliases to avoid importing heavy EVA modules at import time
    LivingSymbolRuntime = Any  # type: ignore
    DivineLanguageEvolved = Any  # type: ignore
    DivineSignature = Any  # type: ignore


class GolachabNode(CosmicNode):
    """
    Golachab (Incendiarios) - Nodo Cognitivo de Ira y Conflicto.
    Procesa percepciones para generar impulsos de agresión y propagación del conflicto.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "golachab_rage",
        mass: float = 3.5,
        influence_radius: float = 3.0,
    ):
        # create DivineSignature before calling base init
        golachab_divine_signature = None
        try:
            # prefer runtime local import when EVA present
            from crisalida_lib.EVA.divine_sigils import DivineSignature as _DS

            golachab_divine_signature = _DS(glyph="ג")
        except Exception:
            golachab_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=golachab_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)
        self.activation_threshold: float = 0.17
        self.memory_patterns: list[dict[str, Any]] = []
        self.aggression_history: list[dict[str, Any]] = []
        self.conflict_history: list[dict[str, Any]] = []
        self.chaos_patterns: list[dict[str, Any]] = []

        # Keep signature for backward compatibility
        self.divine_signature = golachab_divine_signature

        # EVA: memoria viviente y runtime de simulación (defensive)
        try:
            self.eva_runtime: Optional[LivingSymbolRuntime] = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None
        try:
            # DivineLanguageEvolved may require runtime args; create defensively
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
        # Normalize perception/resonance
        perception = perception if perception is not None else self.perceive_local_qualia()
        resonance = (
            perception.get("resonance")
            if isinstance(perception.get("resonance"), dict)
            else None
        )
        coherence = (
            resonance.get("coherence")
            if resonance is not None
            else perception.get("coherence", 0.5)
        )
        intensity = (
            resonance.get("intensity")
            if resonance is not None
            else perception.get("intensity", 0.5)
        )

        perception.setdefault("coherence", coherence)
        perception.setdefault("intensity", intensity)

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault(
                "coherence",
                (
                    resonance.get("coherence", 1.0)
                    if resonance is not None
                    else coherence
                ),
            )
            perception.setdefault(
                "intensity",
                (
                    resonance.get("intensity", 1.0)
                    if resonance is not None
                    else intensity
                ),
            )

        # Expose normalized fields for older code
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = None
        coherence = (
            resonance.get("coherence") if resonance is not None else perception.get("coherence", 0.5)
        )
        intensity = (
            resonance.get("intensity") if resonance is not None else perception.get("intensity", 0.5)
        )
        perception.setdefault("coherence", coherence)
        perception.setdefault("intensity", intensity)

        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Agresión
        aggression_content = f"Agresión en aumento. Valencia emocional: {perception.get('emotional_valence', 0.0):.2f}"
        aggression_intensity = activation * (
            1 - (perception.get("emotional_valence", 0.0) + 1) / 2
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=aggression_content,
                intensity=aggression_intensity,
                confidence=0.72,
                source_node=self.node_name,
            )
        )
        self.aggression_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "emotional_valence": perception.get("emotional_valence", 0.0),
                "intensity": aggression_intensity,
            }
        )

        # Impulso de Propagación del Conflicto
        conflict_content = (
            f"Conflicto propagándose. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        conflict_intensity = activation * perception.get("arousal", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=conflict_content,
                intensity=conflict_intensity,
                confidence=0.65,
                source_node=self.node_name,
            )
        )
        self.conflict_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": conflict_intensity,
            }
        )

        # Impulso de Caos (si baja coherencia)
        chaos_intensity = 0.0
        if perception.get("coherence", 0.5) < 0.3:
            chaos_content = f"Patrón de caos detectado. Coherencia: {perception.get('coherence', 0.0):.2f}"
            chaos_intensity = activation * (1 - perception.get("coherence", 0.0))
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.CHAOS_EMERGENCE,
                    content=chaos_content,
                    intensity=chaos_intensity,
                    confidence=0.7,
                    source_node=self.node_name,
                )
            )
            self.chaos_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "coherence": perception.get("coherence", 0.0),
                    "intensity": chaos_intensity,
                }
            )

        # EVA: Archivar experiencia como bytecode de agresión/conflicto/caos
        qualia_state = {
            "aggression": aggression_intensity,
            "conflict": conflict_intensity,
            "chaos": chaos_intensity,
            "valence": perception.get("emotional_valence", 0.0),
            "arousal": perception.get("arousal", 0.0),
            "coherence": perception.get("coherence", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Golachab environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Golachab según la percepción.
        """
        emotional_valence = perception.get("emotional_valence", 0.0)
        arousal = perception.get("arousal", 0.5)
        coherence = perception.get("coherence", 0.5)
        # Activación ponderada por arousal y baja coherencia, penalizada por valencia positiva
        activation = (arousal * 0.5 + (1.0 - coherence) * 0.3) * (
            1.0 - emotional_valence * 0.2
        )
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
        Compila una experiencia de agresión/conflicto/caos en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_GOLACHAB_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        bytecode: Any = utils.compile_intention_safe(_dc, intention)
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
        runtime = getattr(self, "eva_runtime", None)
        simulation_state: dict[str, Any]
        if runtime is None:
            simulation_state = {}
        else:
            exec_res = runtime.execute_bytecode(bytecode)
            simulation_state = utils.run_maybe_awaitable(exec_res) or {}
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Golachab environment hook failed: {e}")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de agresión/conflicto/caos.
        """
        intention = {
            "intention_type": "ARCHIVE_GOLACHAB_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        bytecode = utils.compile_intention_safe(_dc, intention)
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
