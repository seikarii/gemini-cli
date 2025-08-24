"""
TogaririmNode - Nodo Cognitivo Qliphoth de Contienda y Crueldad (Togaririm, Disputadores).

Procesa percepciones para generar impulsos de conflicto, discordia y crueldad,
reflejando el aspecto de la conciencia que promueve la desarmonía y la agresión.
Incluye historial de discordia, crueldad y patrones de conflicto para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
else:
    DivineSignature = Any

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
    except Exception:
        CosmicNode = Any


class TogaririmNode(CosmicNode):
    """
    Togaririm (Disputadores) - Nodo Cognitivo de Contienda y Crueldad.
    Procesa percepciones para generar impulsos de conflicto y desarmonía.
    """

    def __init__(
        self,
        node_name: str = "togaririm_strife",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        togaririm_divine_signature = None
        try:
            if DivineSignature is not None:
                togaririm_divine_signature = DivineSignature(glyph="ט")
        except Exception:
            togaririm_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=togaririm_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = togaririm_divine_signature

        self.activation_threshold: float = 0.14
        self.memory_patterns: list[dict[str, Any]] = []
        self.discord_history: list[dict[str, Any]] = []
        self.cruelty_history: list[dict[str, Any]] = []
        self.conflict_patterns: list[dict[str, Any]] = []

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

        # Impulso de Discordia
        discord_content = f"Discordia sembrándose. Valencia emocional: {perception.get('emotional_valence', 0.0):.2f}"
        discord_intensity = activation * (
            1 - (perception.get("emotional_valence", 0.0) + 1) / 2
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=discord_content,
                intensity=discord_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.discord_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "emotional_valence": perception.get("emotional_valence", 0.0),
                "intensity": discord_intensity,
            }
        )

        # Impulso de Crueldad
        cruelty_content = (
            f"Crueldad manifestándose. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        cruelty_intensity = activation * perception.get("arousal", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=cruelty_content,
                intensity=cruelty_intensity,
                confidence=0.6,
                source_node=self.node_name,
            )
        )
        self.cruelty_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": cruelty_intensity,
            }
        )

        # Impulso de Conflicto (si baja coherencia y alta arousal)
        if (
            perception.get("coherence", 0.5) < 0.3
            and perception.get("arousal", 0.0) > 0.7
        ):
            conflict_content = f"Conflicto activo. Coherencia: {perception.get('coherence', 0.0):.2f}, Arousal: {perception.get('arousal', 0.0):.2f}"
            conflict_intensity = (
                activation
                * (1 - perception.get("coherence", 0.0))
                * perception.get("arousal", 0.0)
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.DOUBT_INJECTION,
                    content=conflict_content,
                    intensity=conflict_intensity,
                    confidence=0.65,
                    source_node=self.node_name,
                )
            )
            self.conflict_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "coherence": perception.get("coherence", 0.0),
                    "arousal": perception.get("arousal", 0.0),
                    "intensity": conflict_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Togaririm según la percepción.
        """
        emotional_valence = perception.get("emotional_valence", 0.0)
        arousal = perception.get("arousal", 0.5)
        coherence = perception.get("coherence", 0.5)
        # Activación ponderada por baja valencia, alto arousal y baja coherencia
        activation = (
            (1.0 - ((emotional_valence + 1) / 2)) * 0.4
            + arousal * 0.4
            + (1.0 - coherence) * 0.2
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

    def get_discord_history(self) -> list[dict[str, Any]]:
        """
        Devuelve el historial de discordia registrado por el nodo.
        """
        return self.discord_history

    def get_cruelty_history(self) -> list[dict[str, Any]]:
        """
        Devuelve el historial de crueldad registrado por el nodo.
        """
        return self.cruelty_history

    def get_conflict_patterns(self) -> list[dict[str, Any]]:
        """
        Devuelve los patrones de conflicto registrados por el nodo.
        """
        return self.conflict_patterns

    def clear_histories(self) -> None:
        """
        Limpia los historiales de discordia y crueldad, y los patrones de conflicto.
        """
        self.discord_history.clear()
        self.cruelty_history.clear()
        self.conflict_patterns.clear()

    def simulate_internal_conversation(self) -> str:
        """
        Simula una conversación interna del nodo Togaririm, revelando sus patrones de discordia y crueldad.
        """
        if not self.discord_history and not self.cruelty_history:
            return "Sin historial de discordia o crueldad para mostrar."
        conversation = ["--- Conversación Interna Togaririm ---"]
        for discord_record in self.discord_history[-5:]:
            conversation.append(
                f"Discordia (Valencia: {discord_record['emotional_valence']:.2f}, Intensidad: {discord_record['intensity']:.2f})"
            )
        for cruelty_record in self.cruelty_history[-5:]:
            conversation.append(
                f"Crueldad (Arousal: {cruelty_record['arousal']:.2f}, Intensidad: {cruelty_record['intensity']:.2f})"
            )
        for conflict_record in self.conflict_patterns[-5:]:
            conversation.append(
                f"Conflicto (Coherencia: {conflict_record['coherence']:.2f}, Arousal: {conflict_record['arousal']:.2f}, Intensidad: {conflict_record['intensity']:.2f})"
            )
        return "\n".join(conversation)


class EVATogaririmNode(TogaririmNode):
    """
    Nodo Togaririm extendido para integración con EVA.
    Permite compilar impulsos de conflicto y crueldad como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_togaririm_strife", phase: str = "default"):
        super().__init__(node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_conflict_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de conflicto/crueldad y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_TOGARIRIM_CONFLICT_EXPERIENCE",
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
                print(f"[EVA] TogaririmNode compile_intention failed: {e}")
        else:
            pass
        experience_id = f"togaririm_conflict_{hash(str(perception))}"
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

    def eva_recall_conflict_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de conflicto/crueldad almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Togaririm experience"}
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
                    print(f"[EVA] TogaririmNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] TogaririmNode environment hook failed: {e}")
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

    def add_conflict_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de conflicto/crueldad.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_TOGARIRIM_CONFLICT_EXPERIENCE",
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
                    f"[EVA] TogaririmNode add_conflict_experience_phase compile_intention failed: {e}"
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
        """Lista todas las fases disponibles para una experiencia de conflicto/crueldad."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_conflict_experience": self.eva_ingest_conflict_experience,
            "eva_recall_conflict_experience": self.eva_recall_conflict_experience,
            "add_conflict_experience_phase": self.add_conflict_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
