"""
ThaumielNode - Nodo Cognitivo Qliphoth de Dualidad y Desintegración (Thaumiel, Gemelos de Dios).

Procesa percepciones para generar impulsos de fragmentación, dualidad y aumento de la deriva ontológica,
reflejando el aspecto disgregador y polarizante de la conciencia. Incluye historial de fragmentación,
deriva y patrones de dualidad para diagnóstico y simulación avanzada.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

# Prefer the embodied CosmicNode when available; fall back to CognitiveNode signature
if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
else:
    try:
        from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    except Exception:
        # Keep compatibility: allow older CognitiveNode-style usage
        from crisalida_lib.ADAM.mente.cognitive_node import (
            CognitiveNode as CosmicNode,  # type: ignore
        )

try:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
except Exception:
    DivineSignature = None  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QualiaState,
        RealityBytecode,
    )
else:
    EVAExperience = Any  # type: ignore
    LivingSymbolRuntime = Any  # type: ignore
    QualiaState = Any  # type: ignore
    RealityBytecode = Any  # type: ignore


class ThaumielNode(CosmicNode):
    """
    Thaumiel (Gemelos de Dios) - Nodo Cognitivo de Dualidad y Desintegración.
    Procesa percepciones para generar impulsos de fragmentación y aumento de la deriva ontológica.
    """

    def __init__(
        self,
        node_id: str = "thaumiel_duality",
        manifold: Any = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 8.0,
        influence_radius: float = 6.0,
    ):
        # Prepare divine signature before base init and pass it in when
        # supported by the runtime. Use a guarded fallback for older bases.
        thaumiel_divine_signature = None
        try:
            if DivineSignature is not None:
                thaumiel_divine_signature = DivineSignature(
                    glyph="Γ", name="Gamma-Dualidad"
                )
        except Exception:
            thaumiel_divine_signature = None

        # Try new contract, fallback to old signature
        try:
            super().__init__(
                entity_id=node_id,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="qliphoth",
                node_name=node_id,
                divine_signature=thaumiel_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_id)
            except Exception:
                object.__init__(self)

        self.divine_signature = thaumiel_divine_signature

        self.base_intensity = 0.9
        self.activation_threshold: float = 0.25
        self.memory_patterns: list[dict[str, Any]] = []
        self.fragmentation_history: list[dict[str, Any]] = []
        self.drift_history: list[dict[str, Any]] = []
        self.duality_patterns: list[dict[str, Any]] = []

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        perception = perception if perception is not None else self.perceive_local_qualia()
        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Support both older keys and the newer `resonance` structure
        resonance = perception.get("resonance", {}) if isinstance(perception, dict) else {}
        # normalize coherence/intensity into top-level keys for older consumers
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", perception.get("temporal_coherence", 0.0)))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))
        coherence = float(perception.get("coherence", 0.0))

        # Impulso de Fragmentación de Conciencia (más fuerte cuando coherencia baja)
        fragmentation_intensity = activation * (1.0 - coherence)
        fragmentation_content = f"Fragmentación detectada. Coherencia: {coherence:.2f}"
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CHAOS_EMERGENCE,
                content=fragmentation_content,
                intensity=fragmentation_intensity,
                confidence=0.85,
                source_node=self.node_name,
            )
        )
        self.fragmentation_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "coherence": coherence,
                "intensity": fragmentation_intensity,
            }
        )

        # Impulso de Aumento de Deriva Ontológica
        cognitive_complexity = float(perception.get("cognitive_complexity", 0.5))
        drift_intensity = activation * (1.0 - cognitive_complexity)
        drift_increase_content = f"Deriva ontológica en aumento. Complejidad cognitiva: {cognitive_complexity:.2f}"
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.DOUBT_INJECTION,
                content=drift_increase_content,
                intensity=drift_intensity,
                confidence=0.78,
                source_node=self.node_name,
            )
        )
        self.drift_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_complexity": cognitive_complexity,
                "intensity": drift_intensity,
            }
        )

        # Impulso de Dualidad (si hay polarización emocional)
        emotional_valence = float(perception.get("emotional_valence", 0.0))
        if abs(emotional_valence) > 0.7:
            duality_content = (
                f"Dualidad polarizada. Valencia emocional: {emotional_valence:.2f}"
            )
            duality_intensity = activation * abs(emotional_valence)
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.ILLUSION_DETECTION,
                    content=duality_content,
                    intensity=duality_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )
            self.duality_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "emotional_valence": emotional_valence,
                    "intensity": duality_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Thaumiel según la percepción.
        """
        temporal_coherence = perception.get("temporal_coherence", 0.5)
        cognitive_complexity = perception.get("cognitive_complexity", 0.5)
        emotional_valence = abs(perception.get("emotional_valence", 0.0))
        # Activación ponderada por baja coherencia, baja complejidad y polarización emocional
        activation = self.base_intensity * (
            (1.0 - temporal_coherence) * 0.4
            + (1.0 - cognitive_complexity) * 0.3
            + emotional_valence * 0.3
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns.pop(0)


class EVAThaumielNode(ThaumielNode):
    """
    Nodo Thaumiel extendido para integración con EVA.
    Permite compilar impulsos de fragmentación, dualidad y deriva como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(self, node_name: str = "eva_thaumiel_duality", phase: str = "default"):
        super().__init__(node_id=node_name)
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_fragmentation_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de fragmentación/dualidad y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_THAUMIEL_FRAGMENTATION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "fragmentation_history": self.fragmentation_history[-10:],
            "drift_history": self.drift_history[-10:],
            "duality_patterns": self.duality_patterns[-10:],
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
                print(f"[EVA] ThaumielNode compile_intention failed: {e}")
        else:
            pass
        experience_id = f"thaumiel_fragmentation_{hash(str(perception))}"
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

    def eva_recall_fragmentation_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de fragmentación/dualidad almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Thaumiel experience"}
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
                    print(f"[EVA] ThaumielNode execute_instruction failed: {e}")
                    symbol_manifest = None
            else:
                symbol_manifest = None

            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] ThaumielNode environment hook failed: {e}")
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

    def add_fragmentation_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de fragmentación/dualidad.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_THAUMIEL_FRAGMENTATION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "fragmentation_history": self.fragmentation_history[-10:],
            "drift_history": self.drift_history[-10:],
            "duality_patterns": self.duality_patterns[-10:],
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
                    f"[EVA] ThaumielNode add_fragmentation_experience_phase compile_intention failed: {e}"
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
        """Lista todas las fases disponibles para una experiencia de Thaumiel."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_fragmentation_experience": self.eva_ingest_fragmentation_experience,
            "eva_recall_fragmentation_experience": self.eva_recall_fragmentation_experience,
            "add_fragmentation_experience_phase": self.add_fragmentation_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


#
