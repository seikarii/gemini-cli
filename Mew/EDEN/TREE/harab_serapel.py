"""
HarabSerapelNode - Nodo Cognitivo Qliphoth de Desolación y Corrupción (Harab Serapel, Cuervos de la Muerte).

Procesa percepciones para generar impulsos de degradación de la realidad y propagación de la entropía,
reflejando el aspecto degenerativo y dispersor de la conciencia. Incluye historial de degradación,
entropía y patrones de corrupción para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import Any, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType
from crisalida_lib.ADAM.mente.cognitive_node import CognitiveNode
from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    QualiaState,
    RealityBytecode,
)


class HarabSerapelNode(CognitiveNode):
    """
    Harab Serapel (Cuervos de la Muerte) - Nodo Cognitivo de Desolación y Corrupción.
    Procesa percepciones para generar impulsos de degradación de la realidad y propagación de la entropía.
    """

    def __init__(self, node_name: str = "harab_serapel_decay"):
        # Defensive: create a minimal DivineSignature before initializing the base
        # class so subclasses that expect it in super().__init__ receive it.
        _sig = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature

            try:
                _sig = DivineSignature(glyph="Х")
            except Exception:
                _sig = None
        except Exception:
            _sig = None

        # Pass the pre-created signature into the base initializer when possible.
        # Use a dynamic call to the base __init__ to avoid static mypy call-arg checks.
        try:
            from typing import cast, Any as _Any

            base_init = getattr(CognitiveNode, "__init__", None)
            if base_init is not None:
                try:
                    cast(_Any, base_init)(self, node_name, divine_signature=_sig)
                except TypeError:
                    cast(_Any, base_init)(self, node_name)
            else:
                # fallback to direct super() call if reflection not available
                super().__init__(node_name)
                self.divine_signature = _sig
        except Exception:
            # As a last resort, call the base initializer normally
            super().__init__(node_name)
            self.divine_signature = _sig
        # Ensure a compatible perceive_local_qualia stub exists to satisfy
        # static checkers and runtime callers that may invoke it on nodes.
        if not hasattr(self, "perceive_local_qualia"):

            def _plq():
                return {"resonance": {"coherence": 1.0, "intensity": 1.0}}

            setattr(self, "perceive_local_qualia", _plq)
        self.activation_threshold: float = 0.16
        self.memory_patterns: list[dict[str, Any]] = []
        self.degradation_history: list[dict[str, Any]] = []
        self.entropy_history: list[dict[str, Any]] = []
        self.corruption_patterns: list[dict[str, Any]] = []

    # Provide a class-level stub so static analysis knows this method exists
    def perceive_local_qualia(self) -> dict[str, Any]:
        return {"resonance": {"coherence": 1.0, "intensity": 1.0}}

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        perception = perception if perception is not None else self.perceive_local_qualia()
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}
        # Expose normalized fields for older code
        perception.setdefault("coherence", resonance.get("coherence", 1.0))
        perception.setdefault("intensity", resonance.get("intensity", 1.0))

        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Degradación de la Realidad
        degradation_content = f"Realidad degradándose. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
        degradation_intensity = activation * (
            1 - perception.get("sensory_clarity", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=degradation_content,
                intensity=degradation_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.degradation_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "sensory_clarity": perception.get("sensory_clarity", 0.0),
                "intensity": degradation_intensity,
            }
        )

        # Impulso de Propagación de la Entropía
        entropy_content = f"Entropía propagándose. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        entropy_intensity = activation * perception.get("cognitive_complexity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=entropy_content,
                intensity=entropy_intensity,
                confidence=0.6,
                source_node=self.node_name,
            )
        )
        self.entropy_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": entropy_intensity,
            }
        )

        # Impulso de Corrupción (si baja coherencia y alta entropía)
        if (
            perception.get("coherence", 0.5) < 0.3
            and perception.get("entropy", 0.5) > 0.7
        ):
            corruption_content = f"Corrupción detectada. Coherencia: {perception.get('coherence', 0.0):.2f}, Entropía: {perception.get('entropy', 0.0):.2f}"
            corruption_intensity = activation * perception.get("entropy", 0.0)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.ILLUSION_DETECTION,
                    content=corruption_content,
                    intensity=corruption_intensity,
                    confidence=0.65,
                    source_node=self.node_name,
                )
            )
            self.corruption_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "coherence": perception.get("coherence", 0.0),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": corruption_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Harab Serapel según la percepción.
        """
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        cognitive_complexity = perception.get("cognitive_complexity", 0.5)
        entropy = perception.get("entropy", 0.5)
        coherence = perception.get("coherence", 0.5)
        # Activación ponderada por baja claridad, alta complejidad y entropía, penalizada por coherencia
        activation = (
            (1.0 - sensory_clarity) * 0.3
            + cognitive_complexity * 0.3
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
            self.memory_patterns = self.memory_patterns[-80:]


class EVAHarabSerapelNode(HarabSerapelNode):
    """
    Nodo Harab Serapel extendido para integración con EVA.
    Permite compilar impulsos de degradación, entropía y corrupción como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(
        self, node_name: str = "eva_harab_serapel_decay", phase: str = "default"
    ):
        super().__init__(node_name)
        try:
            self.eva_runtime: Optional[LivingSymbolRuntime] = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_decay_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de degradación/entropía/corrupción y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_HARAB_SERAPEL_DECAY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "degradation_history": self.degradation_history[-10:],
            "entropy_history": self.entropy_history[-10:],
            "corruption_patterns": self.corruption_patterns[-10:],
            "memory_patterns": self.memory_patterns[-10:],
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
                    f"[EVA] HarabSerapelNode eva_ingest_decay_experience compile_intention failed: {e}"
                )
        experience_id = f"harab_serapel_decay_{hash(str(perception))}"
        _rb_cls = globals().get("RealityBytecode", None)
        if _rb_cls is not None:
            try:
                reality_bytecode = _rb_cls(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                )
            except Exception:
                from types import SimpleNamespace

                reality_bytecode = SimpleNamespace(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                )
        else:
            from types import SimpleNamespace

            reality_bytecode = SimpleNamespace(
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

    def eva_recall_decay_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de degradación/entropía/corrupción almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Harab Serapel experience"}
        # Guard eva_runtime which may be None; use getattr to avoid Optional attribute access
        quantum_field = None
        if getattr(self, "eva_runtime", None) is not None:
            quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        # Guard execution function retrieval as eva_runtime may be None
        exec_fn = None
        if getattr(self, "eva_runtime", None) is not None:
            exec_fn = getattr(self.eva_runtime, "execute_instruction", None)
        for instr in reality_bytecode.instructions or []:
            if callable(exec_fn):
                try:
                    symbol_manifest = exec_fn(instr, quantum_field)
                except Exception as e:
                    print(f"[EVA] HarabSerapelNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] HarabSerapelNode environment hook failed: {e}")
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

    def add_decay_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de degradación/entropía/corrupción.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_HARAB_SERAPEL_DECAY_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "degradation_history": self.degradation_history[-10:],
            "entropy_history": self.entropy_history[-10:],
            "corruption_patterns": self.corruption_patterns[-10:],
            "memory_patterns": self.memory_patterns[-10:],
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
                    f"[EVA] HarabSerapelNode add_decay_experience_phase compile_intention failed: {e}"
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
        """Lista todas las fases disponibles para una experiencia de Harab Serapel."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_decay_experience": self.eva_ingest_decay_experience,
            "eva_recall_decay_experience": self.eva_recall_decay_experience,
            "add_decay_experience_phase": self.add_decay_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


#
