"""
BinahNode - Nodo Cognitivo de Comprensión y Estructura (Binah, Entendimiento).

Procesa percepciones para generar impulsos de claridad, orden y reducción de deriva ontológica.
Preparado para simulación, diagnóstico, visualización y extensibilidad.
Integración avanzada con EVA: memoria viviente, faseo, hooks de entorno y simulación activa.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, Optional

from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType

if TYPE_CHECKING:
    from crisalida_lib.EDEN.TREE.cosmic_node import CosmicNode  # type: ignore
    from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore
else:
    CosmicNode = Any
    DivineSignature = Any

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
else:
    LivingSymbolRuntime = Any
    DivineLanguageEvolved = Any


class BinahNode(CosmicNode):
    """
    Binah (Entendimiento) - Nodo Cognitivo de Comprensión y Estructura.
    Procesa percepciones para generar impulsos de claridad, orden y reducción de la deriva.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "binah_understanding",
        mass: float = 6.0,
        influence_radius: float = 4.0,
    ):
        # Create DivineSignature before super().__init__
        try:
            import numpy as np  # Ensure numpy is imported for base_vector

            from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore

            # Use a minimal signature compatible with the lightweight divine_sigils
            binah_divine_signature = DivineSignature(
                glyph="Β", name="Binah-Understanding"
            )
        except Exception:
            binah_divine_signature = (
                None  # Fallback if DivineSignature or numpy not available
            )

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="sephirot",
                node_name=node_name,
                mass=mass,
                influence_radius=influence_radius,
                divine_signature=binah_divine_signature,  # Pass the created DivineSignature
            )
        except TypeError:
            # Fallback for older CognitiveNode constructor if CosmicNode fails
            super().__init__(node_name)
        self.divine_signature = (
            binah_divine_signature  # Keep for consistency if needed elsewhere
        )
        self.activation_threshold: float = (
            0.25  # Sensibilidad mínima para activar impulsos
        )
        self.memory_patterns: list[dict[str, Any]] = []
        self.clarity_history: list[dict[str, Any]] = []
        self.drift_history: list[dict[str, Any]] = []
        self.structure_history: list[dict[str, Any]] = []

        # EVA: memoria viviente y runtime de simulacion (defensive init)
        try:
            self.eva_runtime: Optional[LivingSymbolRuntime] = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None
        try:
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
        perception = perception if perception is not None else self.perceive_local_qualia()
        """
        Analiza la percepción y genera impulsos cognitivos de claridad y orden.
        Además, archiva la experiencia en la memoria EVA y permite simulación activa.
        """
        impulses: list[CognitiveImpulse] = []
        # allow perception to be a raw patch or the new perception format
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", 1.0))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))

        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Claridad Cognitiva
        clarity_content = f"Claridad en la percepción. Foco: {perception.get('cognitive_focus', 0.0):.2f}"
        clarity_intensity = activation * perception.get("cognitive_focus", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=clarity_content,
                intensity=clarity_intensity,
                confidence=0.85,
                source_node=self.node_name,
            )
        )
        self.clarity_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "focus": perception.get("cognitive_focus", 0.0),
                "intensity": clarity_intensity,
            }
        )

        # Impulso de Reducción de Deriva (Orden)
        drift_reduction_content = f"Estabilizando deriva. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        drift_intensity = activation * (1 - perception.get("cognitive_complexity", 0.0))
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=drift_reduction_content,
                intensity=drift_intensity,
                confidence=0.80,
                source_node=self.node_name,
            )
        )
        self.drift_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "complexity": perception.get("cognitive_complexity", 0.0),
                "intensity": drift_intensity,
            }
        )

        # Impulso de Estructuración Adicional (si hay baja entropía)
        if perception.get("entropy", 0.5) < 0.3:
            structure_content = f"Estructuración reforzada. Entropía baja: {perception.get('entropy', 0.0):.2f}"
            structure_intensity = activation * (1 - perception.get("entropy", 0.0))
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=structure_content,
                    intensity=structure_intensity,
                    confidence=0.75,
                    source_node=self.node_name,
                )
            )
            self.structure_history.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "entropy": perception.get("entropy", 0.0),
                    "intensity": structure_intensity,
                }
            )

        # EVA: Archivar experiencia como bytecode de comprensión
        qualia_state = {
            "clarity": clarity_intensity,
            "drift": drift_intensity,
            "structure": (
                structure_intensity if perception.get("entropy", 0.5) < 0.3 else 0.0
            ),
            "focus": perception.get("cognitive_focus", 0.0),
            "complexity": perception.get("cognitive_complexity", 0.0),
            "entropy": perception.get("entropy", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Binah environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Binah según la percepción.
        """
        focus = perception.get("cognitive_focus", 0.5)
        clarity = perception.get("sensory_clarity", 0.5)
        complexity = perception.get("cognitive_complexity", 0.5)
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por claridad y foco, penalizada por entropía y complejidad
        activation = (
            (focus * 0.4 + clarity * 0.4)
            * (1.0 - entropy * 0.3)
            * (1.0 - complexity * 0.2)
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        Integra con sistemas de memoria y clustering para aprendizaje adaptativo.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 100:
            self.memory_patterns = self.memory_patterns[-100:]
        # Se podría integrar clustering, análisis de patrones y feedback adaptativo aquí.

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de comprensión en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_BINAH_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if _eva is None:
            bytecode = []
        else:
            _dc = getattr(_eva, "divine_compiler", None)
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
        _eva = getattr(self, "eva_runtime", None)
        if _eva is None:
            return {"error": "No EVA runtime available"}
        exec_fn = getattr(_eva, "execute_bytecode", None)
        if not callable(exec_fn):
            return {"error": "EVA runtime has no execute_bytecode"}
        try:
            res = exec_fn(bytecode)
        except Exception as e:
            return {"error": f"execution failed: {e}"}
        try:
            import inspect

            if inspect.isawaitable(res):
                return {"status": "awaitable", "result": None}
        except Exception:
            pass
        if isinstance(res, dict):
            return res
        return {"result": res}

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de comprensión.
        """
        intention = {
            "intention_type": "ARCHIVE_BINAH_EXPERIENCE",
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
