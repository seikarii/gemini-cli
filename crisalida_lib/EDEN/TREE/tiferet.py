"""
Defines the `TiferetNode`, a cognitive node representing beauty and harmony.

This module contains the implementation of the Tiferet Sephirot, the central
point of the tree, which processes perceptions to generate impulses related
to balance, integration, and harmonization.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
else:
    DivineSignature = Any


if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode
else:
    LivingSymbolRuntime = Any
    EVAExperience = Any
    QualiaState = Any
    RealityBytecode = Any

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:
        CosmicNode = Any


class TiferetNode(CosmicNode):
    """
    Tiferet (Belleza/Armonía) - Nodo Cognitivo de Equilibrio e Integración.
    Procesa percepciones para generar impulsos de armonización y síntesis.
    """

    def __init__(
        self,
        node_name: str = "tiferet_harmony",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        tiferet_divine_signature = None
        try:
            if DivineSignature is not None:
                tiferet_divine_signature = DivineSignature(glyph="ת")
        except Exception:
            tiferet_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                divine_signature=tiferet_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = tiferet_divine_signature

        self.activation_threshold: float = 0.15
        self.memory_patterns: list[dict[str, Any]] = []
        self.harmony_history: list[dict[str, Any]] = []
        self.integration_history: list[dict[str, Any]] = []
        self.balance_patterns: list[dict[str, Any]] = []

    def analyze(
        self, perception: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        perception = (
            perception if perception is not None else self.perceive_local_qualia()
        )
        # normalize new perception format: extract resonance dict when available
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

        # Impulso de Armonización
        harmony_content = (
            f"Buscando armonía. Valencia: {perception.get('emotional_valence', 0.0):.2f}, "
            f"Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        harmony_intensity = (
            activation
            * (1 - abs(perception.get("emotional_valence", 0.0)))
            * (1 - abs(perception.get("arousal", 0.0) - 0.5))
        )  # Armonía = valencia neutra y arousal medio
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=harmony_content,
                intensity=harmony_intensity,
                confidence=0.85,
                source_node=self.node_name,
            )
        )
        self.harmony_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "emotional_valence": perception.get("emotional_valence", 0.0),
                "arousal": perception.get("arousal", 0.0),
                "intensity": harmony_intensity,
            }
        )

        # Impulso de Integración
        integration_content = f"Integrando elementos. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        integration_intensity = activation * perception.get("cognitive_complexity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=integration_content,
                intensity=integration_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )
        self.integration_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": integration_intensity,
            }
        )

        # Impulso de Balance (si coherencia y claridad altas)
        if (
            perception.get("coherence", 0.0) > 0.7
            and perception.get("cognitive_clarity", 0.0) > 0.7
        ):
            balance_content = (
                f"Balance óptimo detectado. Coherencia: {perception.get('coherence', 0.0):.2f}, "
                f"Claridad cognitiva: {perception.get('cognitive_clarity', 0.0):.2f}"
            )
            balance_intensity = (
                activation
                * (
                    perception.get("coherence", 0.0)
                    + perception.get("cognitive_clarity", 0.0)
                )
                / 2
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=balance_content,
                    intensity=balance_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )
            self.balance_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "coherence": perception.get("coherence", 0.0),
                    "cognitive_clarity": perception.get("cognitive_clarity", 0.0),
                    "intensity": balance_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Tiferet según la percepción.
        """
        emotional_valence = abs(perception.get("emotional_valence", 0.0))
        arousal = abs(perception.get("arousal", 0.0) - 0.5)
        cognitive_complexity = perception.get("cognitive_complexity", 0.5)
        coherence = perception.get("coherence", 0.5)
        # Activación ponderada por neutralidad emocional, arousal medio, complejidad y coherencia
        activation = (
            (1.0 - emotional_valence) * 0.3
            + (1.0 - arousal) * 0.3
            + cognitive_complexity * 0.2
            + coherence * 0.2
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[-80:]


class EVATiferetNode(TiferetNode):
    """
    Nodo Tiferet extendido para integración con EVA.
    Permite compilar impulsos de armonía, integración y balance como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_tiferet_harmony", phase: str = "default"):
        super().__init__(node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_harmony_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de armonía/integración y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_TIFERET_HARMONY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive compile: guard against missing EVA runtime or divine_compiler
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        if divine_compiler is not None and hasattr(
            divine_compiler, "compile_intention"
        ):
            try:
                compiled = None
                _fn = getattr(divine_compiler, "compile_intention", None)
                if callable(_fn):
                    try:
                        compiled = _fn(intention)
                    except Exception:
                        compiled = None
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(f"[EVA] TiferetNode compile_intention failed: {e}")
        else:
            pass
        experience_id = f"tiferet_harmony_{hash(str(perception))}"
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

    def eva_recall_harmony_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de armonía/integración almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Tiferet experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        exec_fn = getattr(self.eva_runtime, "execute_instruction", None)
        for instr in reality_bytecode.instructions or []:
            if callable(exec_fn):
                try:
                    symbol_manifest = exec_fn(instr, quantum_field)
                except Exception as e:
                    print(f"[EVA] TiferetNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] TiferetNode environment hook failed: {e}")
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

    def add_harmony_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de armonía/integración.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_TIFERET_HARMONY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive compile: guard against missing EVA runtime or divine_compiler
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if compile_fn and callable(compile_fn):
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(
                    f"[EVA] TiferetNode add_harmony_experience_phase compile_intention failed: {e}"
                )
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
        """Lista todas las fases disponibles para una experiencia de Tiferet."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_harmony_experience": self.eva_ingest_harmony_experience,
            "eva_recall_harmony_experience": self.eva_recall_harmony_experience,
            "add_harmony_experience_phase": self.add_harmony_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
