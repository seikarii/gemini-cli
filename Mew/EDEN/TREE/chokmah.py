"""
ChokmahNode - Nodo Cognitivo de la Sephirot Chokmah (Sabiduría).
Procesa percepciones para generar impulsos de intuición, creatividad, expansión y visión.
Integra patrones de memoria, modulación de intensidad y generación de ideas emergentes.
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


class ChokmahNode(CosmicNode):
    """
    Nodo Cognitivo Chokmah (Sabiduría).
    Procesa percepciones para generar impulsos de nuevas ideas, expansión y visión.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_id: str = "chokmah_wisdom",
        mass: float = 4.0,
        influence_radius: float = 3.0,
    ):
        chokmah_divine_signature = None
        chokmah_divine_signature = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore

            # runtime: divine_sigils may expose a lighter runtime signature
            chokmah_divine_signature = DivineSignature(glyph="Κ")
        except Exception:
            chokmah_divine_signature = None

        try:
            super().__init__(
                entity_id=node_id,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="sephirot",
                node_name=node_id,
                divine_signature=chokmah_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_id)
            except Exception:
                object.__init__(self)

        self.divine_signature = chokmah_divine_signature
        self.base_intensity = 0.85
        self.activation_threshold = 0.3
        self.memory_patterns: list[dict[str, Any]] = []

        # EVA: memoria viviente y runtime de simulación (defensive init)
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
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", 1.0))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))
        """
        Analiza una percepción y genera impulsos cognitivos de creatividad y expansión.
        Además, archiva la experiencia en la memoria EVA y permite simulación activa.
        """
        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Creatividad e Innovación
        creativity_content = f"Nueva idea emergente. Complejidad: {perception.get('cognitive_complexity', 0.0):.2f}"
        creativity_intensity = activation * perception.get("cognitive_complexity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.PATTERN_RECOGNITION,
                content=creativity_content,
                intensity=creativity_intensity,
                confidence=0.8,
                source_node=self.node_name,
            )
        )

        # Impulso de Expansión y Visión
        expansion_content = f"Potencial de expansión percibido. Valencia: {perception.get('emotional_valence', 0.0):.2f}"
        expansion_intensity = activation * (
            (perception.get("emotional_valence", 0.0) + 1) / 2
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=expansion_content,
                intensity=expansion_intensity,
                confidence=0.75,
                source_node=self.node_name,
            )
        )

        # Impulso de Visión y Proyección
        vision_intensity = 0.0
        if perception.get("transcendent_awareness", 0.0) > 0.5:
            vision_content = f"Visión trascendente detectada. Nivel: {perception.get('transcendent_awareness', 0.0):.2f}"
            vision_intensity = activation * perception.get(
                "transcendent_awareness", 0.0
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.FUTURE_PROJECTION,
                    content=vision_content,
                    intensity=vision_intensity,
                    confidence=0.7,
                    source_node=self.node_name,
                )
            )

        # EVA: Archivar experiencia como bytecode de sabiduría/creatividad
        qualia_state = {
            "creativity": creativity_intensity,
            "expansion": expansion_intensity,
            "vision": vision_intensity,
            "complexity": perception.get("cognitive_complexity", 0.0),
            "valence": perception.get("emotional_valence", 0.0),
            "awareness": perception.get("transcendent_awareness", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Chokmah environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Chokmah basado en la percepción.
        """
        complexity = perception.get("cognitive_complexity", 0.0)
        valence = perception.get("emotional_valence", 0.0)
        awareness = perception.get("transcendent_awareness", 0.0)
        activation = self.base_intensity * (
            0.5 + complexity * 0.3 + valence * 0.1 + awareness * 0.1
        )
        return min(1.0, max(0.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria internos con la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 50:
            self.memory_patterns = self.memory_patterns[-50:]

    # --- EVA Memory System Methods ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de sabiduría/creatividad en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_CHOKMAH_EXPERIENCE",
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
        # Normalize coroutine/awaitable results to a dict without awaiting
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
        Añade una fase alternativa (timeline) para una experiencia de sabiduría/creatividad.
        """
        intention = {
            "intention_type": "ARCHIVE_CHOKMAH_EXPERIENCE",
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
