"""
Defines the `GeburahNode`, a cognitive node representing justice and severity.

This module contains the implementation of the Geburah Sephirot, which
processes perceptions to generate impulses related to discipline, restraint,
and the containment of chaos.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

try:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
except Exception:
    DivineSignature = None  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback
        CosmicNode = Any  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
else:
    # runtime fallbacks to avoid import-time EVA dependency
    LivingSymbolRuntime = Any  # type: ignore
    DivineLanguageEvolved = Any  # type: ignore


class GeburahNode(CosmicNode):
    """
    Geburah (Justicia/Severidad) - Nodo Cognitivo de Disciplina y Restricción.
    Procesa percepciones para generar impulsos de contención del caos y límites.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "geburah_discipline",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 12.0,
        influence_radius: float = 5.0,
    ):
        # pre-create DivineSignature so base can receive it during init
        geburah_divine_signature = None
        try:
            if DivineSignature is not None:
                geburah_divine_signature = DivineSignature(
                    glyph="Δ", name="Delta-Severidad"
                )
        except Exception:
            geburah_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                divine_signature=geburah_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = geburah_divine_signature
        self.activation_threshold: float = 0.2
        self.memory_patterns: list[dict[str, Any]] = []
        self.limit_history: list[dict[str, Any]] = []
        self.chaos_containment_history: list[dict[str, Any]] = []
        self.discipline_patterns: list[dict[str, Any]] = []

        # EVA: memoria viviente y runtime de simulación (instantiate defensively)
        try:
            self.eva_runtime: Optional[LivingSymbolRuntime] = (
                LivingSymbolRuntime() if callable(LivingSymbolRuntime) else None
            )
        except Exception:
            self.eva_runtime = None

        try:
            self.divine_compiler: Optional[DivineLanguageEvolved] = (
                DivineLanguageEvolved(None) if callable(DivineLanguageEvolved) else None
            )
        except Exception:
            self.divine_compiler = None
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []
        self._current_phase: str = "default"

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        perception = perception if perception is not None else self.perceive_local_qualia()
        # Normalize new perception format: prefer resonance dict (coherence/intensity)
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
        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Imposición de Límites
        limit_content = (
            f"Necesidad de restricción. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        limit_intensity = activation * (1 - perception.get("arousal", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=limit_content,
                intensity=limit_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )
        self.limit_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": limit_intensity,
            }
        )
        # Impulso de Contención del Caos
        chaos_containment_content = f"Conteniendo el caos. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        chaos_intensity = activation * perception.get("cognitive_complexity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=chaos_containment_content,
                intensity=chaos_intensity,
                confidence=0.68,
                source_node=self.node_name,
            )
        )
        self.chaos_containment_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": chaos_intensity,
            }
        )

        # Impulso de Disciplina (si baja entropía y alta orden)
        discipline_intensity = 0.0
        if perception.get("entropy", 0.5) < 0.3 and perception.get("order", 0.5) > 0.6:
            discipline_content = f"Disciplina reforzada. Orden: {perception.get('order', 0.0):.2f}, Entropía: {perception.get('entropy', 0.0):.2f}"
            discipline_intensity = activation * perception.get("order", 0.0)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=discipline_content,
                    intensity=discipline_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )
            self.discipline_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "order": perception.get("order", 0.0),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": discipline_intensity,
                }
            )

        # EVA: Archivar experiencia como bytecode de disciplina/contención
        qualia_state = {
            "limit": limit_intensity,
            "chaos_containment": chaos_intensity,
            "discipline": discipline_intensity,
            "arousal": perception.get("arousal", 0.0),
            "complexity": perception.get("cognitive_complexity", 0.0),
            "order": perception.get("order", 0.0),
            "entropy": perception.get("entropy", 0.0),
        }
        # Additional behavior: test nearby crystals for weakness and emit dissonant impulses
        prox = (
            perception.get("proximal_crystals", [])
            if isinstance(perception, dict)
            else []
        )
        weak_crystals = [c for c in prox if float(c.get("influence", 0.0)) < 0.15]
        if weak_crystals:
            # emit a dissonant testing impulse
            test_imp = CognitiveImpulse(
                impulse_type=ImpulseType.PERCEPTION_INTUITION,
                content={"dissonant_test": True, "weak_crystals": len(weak_crystals)},
                intensity=min(1.0, activation * 0.6),
                confidence=0.6,
                source_node=self.node_name,
            )
            impulses.append(test_imp)

        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Geburah environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Geburah según la percepción.
        """
        arousal = perception.get("arousal", 0.5)
        complexity = perception.get("cognitive_complexity", 0.5)
        order = perception.get("order", 0.5)
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por orden y complejidad, penalizada por arousal y entropía
        activation = (
            (order * 0.4 + complexity * 0.3)
            * (1.0 - arousal * 0.2)
            * (1.0 - entropy * 0.2)
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
        Compila una experiencia de disciplina/contención en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_GEBURAH_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        # use central utils helper
        from crisalida_lib.EDEN import utils as _eden_utils

        bytecode = _eden_utils.compile_intention_safe(_dc, intention)
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
                print(f"EVA Geburah environment hook failed: {e}")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de disciplina/contención.
        """
        intention = {
            "intention_type": "ARCHIVE_GEBURAH_EXPERIENCE",
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
