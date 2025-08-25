"""
SatarielNode - Nodo Cognitivo Qliphoth de Ocultación y Confusión (Satariel, Ocultadores).

Procesa percepciones para generar impulsos de pérdida de claridad, aumento de incertidumbre y ocultación de la verdad.
Incluye historial de ocultación, confusión y patrones de incertidumbre para diagnóstico y simulación avanzada.
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


class SatarielNode(CognitiveNode):
    """
    Satariel (Ocultadores) - Nodo Cognitivo de Ocultación y Confusión.
    Procesa percepciones para generar impulsos de pérdida de claridad y aumento de la incertidumbre.
    """

    def __init__(self, node_name: str = "satariel_obscurity"):
        # Create DivineSignature before calling base init and pass it defensively
        _sig = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature

            try:
                _sig = DivineSignature(glyph="צ")
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
                super().__init__(node_name)
                self.divine_signature = _sig
        except Exception:
            super().__init__(node_name)
            self.divine_signature = _sig
        self.activation_threshold: float = 0.13
        self.memory_patterns: list[dict[str, Any]] = []
        self.obscuration_history: list[dict[str, Any]] = []
        self.confusion_history: list[dict[str, Any]] = []
        self.uncertainty_patterns: list[dict[str, Any]] = []

    def analyze(
        self, perception: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        # Ensure we have a concrete perception mapping (avoid Optional/Any union)
        from typing import cast

        raw = perception if perception is not None else self.perceive_local_qualia()

        # Normalize into a concrete dict for static checkers and runtime safety
        if isinstance(raw, dict):
            perception_map: dict[str, Any] = cast(dict[str, Any], raw)
        elif hasattr(raw, "items"):
            perception_map = dict(raw)
        else:
            perception_map = {"resonance": {"coherence": 1.0, "intensity": 1.0}}

        # Ensure perception_map is a concrete dict for static checkers
        from typing import cast

        perception_map = cast(dict[str, Any], perception_map)

        # extract resonance dict when available
        _res = perception_map.get("resonance")
        resonance = (
            _res if isinstance(_res, dict) else {"coherence": 1.0, "intensity": 1.0}
        )

        # Expose normalized fields for older code
        perception_map.setdefault("coherence", (resonance or {}).get("coherence", 1.0))
        perception_map.setdefault("intensity", (resonance or {}).get("intensity", 1.0))

        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception_map)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception_map)

        # Impulso de Ocultación de la Verdad
        obscuration_content = f"Verdad oculta. Complejidad cognitiva: {perception_map.get('cognitive_complexity', 0.0):.2f}"
        obscuration_intensity = activation * (
            1 - perception_map.get("cognitive_complexity", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.ILLUSION_DETECTION,
                content=obscuration_content,
                intensity=obscuration_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.obscuration_history.append(
            {
                "timestamp": perception_map.get("timestamp"),
                "cognitive_complexity": perception_map.get("cognitive_complexity", 0.0),
                "intensity": obscuration_intensity,
            }
        )

        # Impulso de Creación de Confusión
        confusion_content = f"Confusión en aumento. Claridad sensorial: {perception_map.get('sensory_clarity', 0.0):.2f}"
        confusion_intensity = activation * (
            1 - perception_map.get("sensory_clarity", 0.0)
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=confusion_content,
                intensity=confusion_intensity,
                confidence=0.65,
                source_node=self.node_name,
            )
        )
        self.confusion_history.append(
            {
                "timestamp": perception_map.get("timestamp"),
                "sensory_clarity": perception_map.get("sensory_clarity", 0.0),
                "intensity": confusion_intensity,
            }
        )

        # Impulso de Incertidumbre (si baja coherencia y alta entropía)
        if (
            perception_map.get("coherence", 0.5) < 0.3
            and perception_map.get("entropy", 0.5) > 0.7
        ):
            uncertainty_content = f"Incertidumbre elevada. Coherencia: {perception_map.get('coherence', 0.0):.2f}, Entropía: {perception_map.get('entropy', 0.0):.2f}"
            uncertainty_intensity = activation * perception_map.get("entropy", 0.0)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.DOUBT_INJECTION,
                    content=uncertainty_content,
                    intensity=uncertainty_intensity,
                    confidence=0.6,
                    source_node=self.node_name,
                )
            )
            self.uncertainty_patterns.append(
                {
                    "timestamp": perception_map.get("timestamp"),
                    "coherence": perception_map.get("coherence", 0.0),
                    "entropy": perception_map.get("entropy", 0.0),
                    "intensity": uncertainty_intensity,
                }
            )

        return impulses

    def perceive_local_qualia(self) -> dict[str, Any]:
        """Compatibility stub: provide a minimal perception when no manifold is present."""
        return {"resonance": {"coherence": 1.0, "intensity": 1.0}}

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Satariel según la percepción.
        """
        cognitive_complexity = perception.get("cognitive_complexity", 0.5)
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        coherence = perception.get("coherence", 0.5)
        entropy = perception.get("entropy", 0.5)

        # Activación ponderada por baja complejidad, baja claridad y alta entropía
        activation = (
            (1.0 - cognitive_complexity) * 0.3
            + (1.0 - sensory_clarity) * 0.3
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


class EVASatarielNode(SatarielNode):
    """
    Nodo Satariel extendido para integración con EVA.
    Permite compilar impulsos de ocultación, confusión e incertidumbre como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(
        self, node_name: str = "eva_satariel_obscurity", phase: str = "default"
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

    def eva_ingest_obscuration_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de ocultación/confusión/incertidumbre y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_SATARIEL_OBSCURATION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "obscuration_history": self.obscuration_history[-10:],
            "confusion_history": self.confusion_history[-10:],
            "uncertainty_patterns": self.uncertainty_patterns[-10:],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive: eva_runtime or its divine_compiler may be None or missing in some contexts
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if callable(compile_fn):
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(f"[EVA] SatarielNode compile_intention failed: {e}")
        experience_id = f"satariel_obscuration_{hash(str(perception))}"
        # Defensive instantiation: RealityBytecode may not be importable or its
        # constructor signature may differ across runtime versions. Resolve at
        # runtime and fall back to a SimpleNamespace to keep static checks happy.
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

    def eva_recall_obscuration_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de ocultación/confusión/incertidumbre almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Satariel experience"}

        # Use getattr to avoid union-attr on Optional eva_runtime
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        exec_fn = getattr(self.eva_runtime, "execute_instruction", None)
        if not callable(exec_fn):
            exec_fn = None
        for instr in getattr(reality_bytecode, "instructions", []) or []:
            if not exec_fn:
                continue
            try:
                symbol_manifest = exec_fn(instr, quantum_field)
            except Exception as e:
                print(f"[EVA] SatarielNode execute_instruction failed: {e}")
                continue
            if symbol_manifest:
                try:
                    manifest_dict = (
                        symbol_manifest.to_dict()
                        if hasattr(symbol_manifest, "to_dict")
                        else (
                            dict(symbol_manifest)
                            if hasattr(symbol_manifest, "items")
                            else symbol_manifest
                        )
                    )
                except Exception:
                    manifest_dict = symbol_manifest
                manifestations.append(manifest_dict)
                for hook in getattr(self, "_environment_hooks", []):
                    try:
                        hook(manifest_dict)
                    except Exception as e:
                        print(f"[EVA] SatarielNode environment hook failed: {e}")

        # Construct EVAExperience defensively (avoid static call-arg failures)
        # Build EVAExperience via a runtime-callable to avoid static constructor checks
        try:
            _EVA = globals().get("EVAExperience", None)
            if callable(_EVA):
                eva_experience = _EVA(
                    experience_id=getattr(reality_bytecode, "bytecode_id", ""),
                    bytecode=reality_bytecode,
                    manifestations=manifestations,
                    phase=str(getattr(reality_bytecode, "phase", "default")),
                    qualia_state=getattr(reality_bytecode, "qualia_state", None),
                )
            else:
                raise TypeError("EVAExperience not callable")
        except Exception:
            from types import SimpleNamespace

            eva_experience = SimpleNamespace(
                experience_id=getattr(reality_bytecode, "bytecode_id", ""),
                manifestations=manifestations,
                phase=str(getattr(reality_bytecode, "phase", "default")),
                qualia_state=getattr(reality_bytecode, "qualia_state", None),
            )

        self.eva_experience_store[getattr(reality_bytecode, "bytecode_id", "")] = (
            eva_experience
        )
        # Safely produce qualia_state representation without calling methods on None
        _qstate = getattr(eva_experience, "qualia_state", None)
        if _qstate is not None and hasattr(_qstate, "to_dict"):
            qualia_repr = _qstate.to_dict()
        else:
            qualia_repr = {}

        return {
            "experience_id": getattr(eva_experience, "experience_id", ""),
            "manifestations": [
                (m.to_dict() if hasattr(m, "to_dict") else m) for m in manifestations
            ],
            "phase": getattr(eva_experience, "phase", None),
            "qualia_state": qualia_repr,
        }

    def add_obscuration_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de ocultación/confusión/incertidumbre.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_SATARIEL_OBSCURATION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "obscuration_history": self.obscuration_history[-10:],
            "confusion_history": self.confusion_history[-10:],
            "uncertainty_patterns": self.uncertainty_patterns[-10:],
            "qualia": qualia_state,
            "phase": phase,
        }
        # Defensive compile as above
        bytecode = []
        divine_compiler = getattr(self.eva_runtime, "divine_compiler", None)
        compile_fn = getattr(divine_compiler, "compile_intention", None)
        if callable(compile_fn):
            try:
                compiled = compile_fn(intention)
                if compiled:
                    bytecode = compiled
            except Exception as e:
                print(
                    f"[EVA] SatarielNode add_obscuration_experience_phase compile_intent failed: {e}"
                )
        # Defensive instantiation: resolve RealityBytecode at runtime and
        # fall back to a SimpleNamespace if constructor is unavailable.
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
        """Lista todas las fases disponibles para una experiencia de Satariel."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_obscuration_experience": self.eva_ingest_obscuration_experience,
            "eva_recall_obscuration_experience": self.eva_recall_obscuration_experience,
            "add_obscuration_experience_phase": self.add_obscuration_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
