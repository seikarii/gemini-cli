"""
NetzachNode - Nodo Cognitivo de Victoria, Resiliencia y Expresión Emocional (Netzach, Eternidad).

Procesa percepciones para generar impulsos de persistencia, resiliencia y expresión emocional.
Incluye historial de resiliencia, expresión y patrones de persistencia para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
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

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback
        CosmicNode = Any  # type: ignore


class NetzachNode(CosmicNode):
    """
    Netzach (Victoria/Eternidad) - Nodo Cognitivo de Resistencia y Emoción.
    Procesa percepciones para generar impulsos de persistencia y expresión emocional.
    """

    def __init__(
        self,
        node_name: str = "netzach_resilience",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        # Pre-create divine signature
        netzach_divine_signature = None
        try:
            if DivineSignature is not None:
                netzach_divine_signature = DivineSignature(glyph="Ν")
        except Exception:
            netzach_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                divine_signature=netzach_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = netzach_divine_signature

        self.activation_threshold: float = 0.13
        self.memory_patterns: list[dict[str, Any]] = []
        self.resilience_history: list[dict[str, Any]] = []
        self.expression_history: list[dict[str, Any]] = []
        self.persistence_patterns: list[dict[str, Any]] = []

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

        # Impulso de Resiliencia y Persistencia
        resilience_content = (
            f"Fuerza para persistir. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        resilience_intensity = activation * perception.get("arousal", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=resilience_content,
                intensity=resilience_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.resilience_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": resilience_intensity,
            }
        )

        # Impulso de Expresión Emocional
        emotional_content = f"Necesidad de expresión. Valencia: {perception.get('emotional_valence', 0.0):.2f}"
        emotional_intensity = activation * abs(perception.get("emotional_valence", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=emotional_content,
                intensity=emotional_intensity,
                confidence=0.65,
                source_node=self.node_name,
            )
        )
        self.expression_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "emotional_valence": perception.get("emotional_valence", 0.0),
                "intensity": emotional_intensity,
            }
        )

        # Impulso de Persistencia (si baja entropía y alta arousal)
        if (
            perception.get("entropy", 0.5) < 0.3
            and perception.get("arousal", 0.0) > 0.6
        ):
            persistence_content = f"Patrón de persistencia activado. Entropía: {perception.get('entropy', 0.0):.2f}, Arousal: {perception.get('arousal', 0.0):.2f}"
            persistence_intensity = (
                activation
                * (1 - perception.get("entropy", 0.0))
                * perception.get("arousal", 0.0)
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=persistence_content,
                    intensity=persistence_intensity,
                    confidence=0.72,
                    source_node=self.node_name,
                )
            )
            self.persistence_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "entropy": perception.get("entropy", 0.0),
                    "arousal": perception.get("arousal", 0.0),
                    "intensity": persistence_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Netzach según la percepción.
        """
        arousal = perception.get("arousal", 0.5)
        emotional_valence = abs(perception.get("emotional_valence", 0.0))
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por arousal y valencia emocional, penalizada por entropía
        activation = (arousal * 0.5 + emotional_valence * 0.4) * (1.0 - entropy * 0.2)
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


class EVANetzachNode(NetzachNode):
    """
    Nodo Netzach extendido para integración con EVA.
    Permite compilar impulsos de resiliencia, persistencia y expresión emocional como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(
        self, node_name: str = "eva_netzach_resilience", phase: str = "default"
    ):
        super().__init__(node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_resilience_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de resiliencia/persistencia/expresión emocional y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_NETZACH_RESILIENCE_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "resilience_history": self.resilience_history[-10:],
            "expression_history": self.expression_history[-10:],
            "persistence_patterns": self.persistence_patterns[-10:],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive: the EVA runtime or its divine_compiler may be absent in some
        # environments (CI/smoke-import). Guard the call to avoid runtime errors
        # and to keep static checkers happy about possible None values.
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
                print(f"[EVA] NetzachNode compile_intention failed: {e}")
        else:
            # Keep empty bytecode if compilation isn't available.
            pass
        experience_id = f"netzach_resilience_{hash(str(perception))}"
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

    def eva_recall_resilience_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de resiliencia/persistencia/expresión emocional almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Netzach experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        exec_fn = getattr(self.eva_runtime, "execute_instruction", None)
        # Iterate defensively over instructions; they may be None or not iterable.
        for instr in reality_bytecode.instructions or []:
            if callable(exec_fn):
                try:
                    symbol_manifest = exec_fn(instr, quantum_field)
                except Exception as e:
                    print(f"[EVA] NetzachNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                # If no executor is available, skip manifesting instructions.
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] NetzachNode environment hook failed: {e}")
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

    def add_resilience_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de resiliencia/persistencia/expresión emocional.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_NETZACH_RESILIENCE_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "resilience_history": self.resilience_history[-10:],
            "expression_history": self.expression_history[-10:],
            "persistence_patterns": self.persistence_patterns[-10:],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Guarded compile for add_resilience_experience_phase
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if compile_fn and callable(compile_fn):
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(f"[EVA] NetzachNode compile_intention failed: {e}")
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
        """Lista todas las fases disponibles para una experiencia de Netzach."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_resilience_experience": self.eva_ingest_resilience_experience,
            "eva_recall_resilience_experience": self.eva_recall_resilience_experience,
            "add_resilience_experience_phase": self.add_resilience_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
