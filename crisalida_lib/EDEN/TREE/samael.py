"""
SamaelNode - Nodo Cognitivo Qliphoth de Ilusión, Engaño y Distorsión (Samael, Veneno de Dios).

Procesa percepciones para generar impulsos de distorsión de la verdad, manipulación y creación de ilusión.
Incluye historial de ilusión, engaño y patrones de distorsión para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
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
    except Exception:
        CosmicNode = Any


class SamaelNode(CosmicNode):
    """
    Samael (Veneno de Dios) - Nodo Cognitivo de Ilusión y Engaño.
    Procesa percepciones para generar impulsos de distorsión de la verdad y manipulación.
    """

    def __init__(
        self,
        node_name: str = "samael_deception",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        samael_divine_signature = None
        try:
            if DivineSignature is not None:
                samael_divine_signature = DivineSignature(glyph="ס")
        except Exception:
            samael_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="qliphoth",
                node_name=node_name,
                divine_signature=samael_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = samael_divine_signature

        self.activation_threshold: float = 0.15
        self.memory_patterns: list[dict[str, Any]] = []
        self.illusion_history: list[dict[str, Any]] = []
        self.deception_history: list[dict[str, Any]] = []
        self.distortion_patterns: list[dict[str, Any]] = []

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
        # Expose normalized fields for older code that may read them
        perception.setdefault("coherence", coherence)
        perception.setdefault("intensity", intensity)
        impulses: list[CognitiveImpulse] = []
        # Use normalized coherence/intensity in activation
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Creación de Ilusión
        illusion_content = f"Ilusión tejiéndose. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
        illusion_intensity = activation * (1 - perception.get("sensory_clarity", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.ILLUSION_DETECTION,
                content=illusion_content,
                intensity=illusion_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )
        self.illusion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "sensory_clarity": perception.get("sensory_clarity", 0.0),
                "intensity": illusion_intensity,
            }
        )

        # Impulso de Engaño
        deception_content = f"Engaño en juego. Foco cognitivo: {perception.get('cognitive_focus', 0.0):.2f}"
        deception_intensity = activation * (1 - perception.get("cognitive_focus", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.DOUBT_INJECTION,
                content=deception_content,
                intensity=deception_intensity,
                confidence=0.68,
                source_node=self.node_name,
            )
        )
        self.deception_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_focus": perception.get("cognitive_focus", 0.0),
                "intensity": deception_intensity,
            }
        )

        # Impulso de Distorsión (si baja coherencia y alta entropía)
        if (
            perception.get("coherence", 0.5) < 0.3
            and perception.get("entropy", 0.5) > 0.7
        ):
            distortion_content = f"Distorsión cognitiva activa. Coherencia: {perception.get('coherence', 0.0):.2f}, Entropía: {perception.get('entropy', 0.0):.2f}"
            distortion_intensity = activation * perception.get("entropy", 0.0)
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
                    "coherence": perception.get("coherence", 0.0),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": distortion_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Samael según la percepción.
        """
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        cognitive_focus = perception.get("cognitive_focus", 0.5)
        coherence = perception.get("coherence", 0.5)
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por baja claridad, bajo foco y alta entropía
        activation = (
            (1.0 - sensory_clarity) * 0.3
            + (1.0 - cognitive_focus) * 0.3
            + entropy * 0.3
            + (1.0 - coherence) * 0.1
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


class EVASamaelNode(SamaelNode):
    """
    Nodo Samael extendido para integración con EVA.
    Permite compilar impulsos de ilusión, engaño y distorsión como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_samael_deception", phase: str = "default"):
        super().__init__(node_name)
        # ensure phase is a concrete string at construction time
        self._current_phase = phase or "default"
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_ingest_deception_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de ilusión/engaño/distorsión y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_SAMAEL_DECEPTION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "illusion_history": self.illusion_history[-10:],
            "deception_history": self.deception_history[-10:],
            "distortion_patterns": self.distortion_patterns[-10:],
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
                print(f"[EVA] SamaelNode compile_intention failed: {e}")
        else:
            pass
        experience_id = f"samael_deception_{hash(str(perception))}"
        # ensure phase and timestamp have concrete types expected by RealityBytecode
        phase = phase or self._current_phase
        try:
            _phase: str = str(phase)
        except Exception:
            _phase = "default"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=_phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_deception_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de ilusión/engaño/distorsión almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Samael experience"}
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
                    print(f"[EVA] SamaelNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] SamaelNode environment hook failed: {e}")
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

    def add_deception_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de ilusión/engaño/distorsión.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_SAMAEL_DECEPTION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "illusion_history": self.illusion_history[-10:],
            "deception_history": self.deception_history[-10:],
            "distortion_patterns": self.distortion_patterns[-10:],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Guarded compile for add_experience_phase
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if compile_fn and callable(compile_fn):
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(f"[EVA] SamaelNode compile_intention failed: {e}")
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
        """Lista todas las fases disponibles para una experiencia de Samael."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_deception_experience": self.eva_ingest_deception_experience,
            "eva_recall_deception_experience": self.eva_recall_deception_experience,
            "add_deception_experience_phase": self.add_deception_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
