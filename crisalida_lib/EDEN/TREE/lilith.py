"""
Defines the `LilithNode`, a cognitive node representing the "Queen of the Night".

This module contains the implementation of the Lilith Qliphoth, which
processes perceptions to generate impulses related to shadow, temptation,
and the subversion of will.
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
    except Exception:  # pragma: no cover - runtime fallback
        CosmicNode = Any  # type: ignore


class LilithNode(CosmicNode):
    """
    Lilith (Reina de la Noche) - Nodo Cognitivo de Sombra y Tentación.
    Procesa percepciones para generar impulsos de amplificación de deseos oscuros y subversión.
    """

    def __init__(
        self,
        node_name: str = "lilith_shadow",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        # Create DivineSignature before super
        lilith_divine_signature = None
        try:
            if DivineSignature is not None:
                lilith_divine_signature = DivineSignature(glyph="Λ")
        except Exception:
            lilith_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=lilith_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        # Instance attributes must be set inside __init__ (was dedented accidentally).
        self.divine_signature = lilith_divine_signature

        self.activation_threshold: float = 0.14
        self.subversion_history: list[dict[str, Any]] = []
        self.shadow_patterns: list[dict[str, Any]] = []
        # Explicit runtime-typed attributes for mypy
        self.memory_patterns: list[dict[str, Any]] = []
        self.temptation_history: list[dict[str, Any]] = []

    def analyze(
        self, perception: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        perception = (
            perception if perception is not None else self.perceive_local_qualia()
        )
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

        # Impulso de Tentación
        temptation_content = f"Deseo oscuro amplificado. Valencia emocional: {perception.get('emotional_valence', 0.0):.2f}"
        temptation_intensity = activation * (
            1 - (perception.get("emotional_valence", 0.0) + 1) / 2
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=temptation_content,
                intensity=temptation_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.temptation_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "emotional_valence": perception.get("emotional_valence", 0.0),
                "intensity": temptation_intensity,
            }
        )

        # Impulso de Subversión
        subversion_content = f"Voluntad subvertida. Foco cognitivo: {perception.get('cognitive_focus', 0.0):.2f}"
        subversion_intensity = activation * (1 - perception.get("cognitive_focus", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.DOUBT_INJECTION,
                content=subversion_content,
                intensity=subversion_intensity,
                confidence=0.65,
                source_node=self.node_name,
            )
        )
        self.subversion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_focus": perception.get("cognitive_focus", 0.0),
                "intensity": subversion_intensity,
            }
        )

        # Impulso de Sombra (si baja claridad sensorial y alta entropía)
        if (
            perception.get("sensory_clarity", 0.5) < 0.3
            and perception.get("entropy", 0.5) > 0.7
        ):
            shadow_content = f"Sombra manifestada. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}, Entropía: {perception.get('entropy', 0.0):.2f}"
            shadow_intensity = activation * perception.get("entropy", 0.0)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.CHAOS_EMERGENCE,
                    content=shadow_content,
                    intensity=shadow_intensity,
                    confidence=0.6,
                    source_node=self.node_name,
                )
            )
            self.shadow_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "sensory_clarity": perception.get("sensory_clarity", 0.0),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": shadow_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Lilith según la percepción.
        """
        emotional_valence = perception.get("emotional_valence", 0.0)
        cognitive_focus = perception.get("cognitive_focus", 0.5)
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por baja valencia, bajo foco y alta entropía
        activation = (
            (1.0 - emotional_valence) * 0.3
            + (1.0 - cognitive_focus) * 0.3
            + entropy * 0.3
            + (1.0 - sensory_clarity) * 0.1
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[
                -80:
            ]  # Mantener solo los últimos 80 patrones


class EVALilithNode(LilithNode):
    """
    Nodo Lilith extendido para integración con EVA.
    Permite compilar impulsos de sombra, tentación y subversión como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_lilith_shadow", phase: str = "default"):
        super().__init__(node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_shadow_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de sombra/tentación/subversión y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_LILITH_SHADOW_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "temptation_history": self.temptation_history[-10:],
            "subversion_history": self.subversion_history[-10:],
            "shadow_patterns": self.shadow_patterns[-10:],
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
                print(f"[EVA] LilithNode compile_intention failed: {e}")
        experience_id = f"lilith_shadow_{hash(str(perception))}"
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

    def eva_recall_shadow_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de sombra/tentación/subversión almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Lilith experience"}
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
                print(f"[EVA] LilithNode execute_instruction failed: {e}")
                continue
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] LilithNode environment hook failed: {e}")
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

    def add_shadow_experience_phase(
        self, experience_id: str, phase: str, bytecode: RealityBytecode
    ) -> None:
        """
        Añade una fase a una experiencia de sombra existente.
        """
        if experience_id in self.eva_memory_store:
            reality_bytecode = self.eva_memory_store[experience_id]
            reality_bytecode.phase = phase
            self.eva_phases.setdefault(phase, {})[experience_id] = reality_bytecode

    def register_environment_hook(self, hook: Callable[..., Any]) -> None:
        """
        Registra un hook de entorno que será llamado con cada manifestación de símbolo.
        """
        self._environment_hooks.append(hook)

    def set_phase(self, phase: str) -> None:
        """
        Establece la fase actual para la generación de experiencias.
        """
        self._current_phase = phase
