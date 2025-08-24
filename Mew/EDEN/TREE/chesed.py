"""
ChesedNode - Nodo Cognitivo de Misericordia, Abundancia y Compasión (Chesed, Gracia).

Procesa percepciones para generar impulsos de armonía social, crecimiento positivo y resonancia emocional.
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
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QualiaState,
        RealityBytecode,
    )
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved
else:
    EVAExperience = Any
    LivingSymbolRuntime = Any
    QualiaState = Any
    RealityBytecode = Any
    DivineLanguageEvolved = Any


class ChesedNode(CosmicNode):
    """
    Chesed (Misericordia/Gracia) - Nodo Cognitivo de Abundancia y Compasión.
    Procesa percepciones para generar impulsos de crecimiento positivo y armonía.
    Extendido para integración con EVA: memoria viviente, simulación, faseo y hooks de entorno.
    """

    def __init__(
        self,
        node_name: str = "chesed_compassion",
        mass: float = 5.0,
        influence_radius: float = 3.5,
    ):
        # Build a DivineSignature before base init and pass it in when possible
        chesed_divine_signature = None
        chesed_divine_signature = None
        try:
            from crisalida_lib.EVA.divine_sigils import DivineSignature  # type: ignore

            chesed_divine_signature = DivineSignature(glyph="Χ")
        except Exception:
            chesed_divine_signature = None

        try:
            super().__init__(
                entity_id=node_name,
                manifold=cast(Any, None),
                initial_position=(0.0, 0.0, 0.0),
                node_type="sephirot",
                node_name=node_name,
                divine_signature=chesed_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            try:
                super().__init__(node_name)
            except Exception:
                object.__init__(self)

        self.divine_signature = chesed_divine_signature
        self.activation_threshold: float = (
            0.2  # Sensibilidad mínima para activar impulsos
        )
        self.memory_patterns: list[dict[str, Any]] = []
        self.harmony_history: list[dict[str, Any]] = []
        self.growth_history: list[dict[str, Any]] = []
        self.compassion_history: list[dict[str, Any]] = []

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
        self._environment_hooks: list = []
        self._current_phase: str = "default"

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        perception = perception if perception is not None else self.perceive_local_qualia()
        """Analiza la percepción y genera impulsos cognitivos de armonía, crecimiento y resonancia emocional.
        Además, archiva la experiencia en la memoria EVA y permite simulación activa.
        """
        impulses: list[CognitiveImpulse] = []
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

        # Impulso de Armonía Social
        harmony_content = f"Armonía detectada. Valencia emocional: {perception.get('emotional_valence', 0.0):.2f}"
        harmony_intensity = activation * (
            (perception.get("emotional_valence", 0.0) + 1) / 2
        )
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.EMOTIONAL_RESONANCE,
                content=harmony_content,
                intensity=harmony_intensity,
                confidence=0.82,
                source_node=self.node_name,
            )
        )
        self.harmony_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "valence": perception.get("emotional_valence", 0.0),
                "intensity": harmony_intensity,
            }
        )

        # Impulso de Crecimiento Positivo
        growth_content = (
            f"Potencial de crecimiento. Arousal: {perception.get('arousal', 0.0):.2f}"
        )
        growth_intensity = activation * perception.get("arousal", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=growth_content,
                intensity=growth_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.growth_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "arousal": perception.get("arousal", 0.0),
                "intensity": growth_intensity,
            }
        )

        # Impulso de Compasión (si hay baja entropía y alta valencia)
        structure_intensity = 0.0
        if (
            perception.get("entropy", 0.5) < 0.3
            and perception.get("emotional_valence", 0.0) > 0.3
        ):
            compassion_content = f"Compasión activada. Entropía baja y valencia alta: {perception.get('entropy', 0.0):.2f}, {perception.get('emotional_valence', 0.0):.2f}"
            compassion_intensity = (
                activation
                * (1 - perception.get("entropy", 0.0))
                * perception.get("emotional_valence", 0.0)
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.SOCIAL_BONDING,
                    content=compassion_content,
                    intensity=compassion_intensity,
                    confidence=0.78,
                    source_node=self.node_name,
                )
            )
            self.compassion_history.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "entropy": perception.get("entropy", 0.0),
                    "valence": perception.get("emotional_valence", 0.0),
                    "intensity": compassion_intensity,
                }
            )
            structure_intensity = compassion_intensity

        # EVA: Archivar experiencia como bytecode de compasión/armonía
        qualia_state = {
            "harmony": harmony_intensity,
            "growth": growth_intensity,
            "compassion": structure_intensity,
            "valence": perception.get("emotional_valence", 0.0),
            "arousal": perception.get("arousal", 0.0),
            "entropy": perception.get("entropy", 0.0),
        }
        experience_id = self.eva_ingest_experience(perception, qualia_state)

        # EVA: Simulación activa (opcional, para diagnóstico o visualización)
        # Guard runtime-based recall: eva_recall_experience may return awaitables or require runtime
        simulation_state = self.eva_recall_experience(experience_id)
        for hook in self._environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                print(f"EVA Chesed environment hook failed: {e}")

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Chesed según la percepción.
        """
        valence = (perception.get("emotional_valence", 0.0) + 1) / 2
        arousal = perception.get("arousal", 0.5)
        entropy = perception.get("entropy", 0.5)
        # Activación ponderada por valencia y arousal, penalizada por entropía
        activation = (valence * 0.5 + arousal * 0.5) * (1.0 - entropy * 0.3)
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
        Compila una experiencia de compasión/armonía en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_CHESED_EXPERIENCE",
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
        Añade una fase alternativa (timeline) para una experiencia de compasión/armonía.
        """
        intention = {
            "intention_type": "ARCHIVE_CHESED_EXPERIENCE",
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
        }


class EVAChesedNode(ChesedNode):
    """
    Nodo Chesed extendido para integración avanzada con EVA.
    Permite compilar impulsos de compasión, abundancia y armonía como experiencias vivientes,
    soporta faseo, hooks de entorno y recall activo en el QuantumField.
    """

    def __init__(
        self, node_name: str = "eva_chesed_compassion", phase: str | None = "default"
    ):
        super().__init__(node_name)
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
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases = {}
        self._environment_hooks = []
        # ensure phase is concrete str for downstream consumers
        self._current_phase = phase or "default"

    def eva_ingest_compassion_experience(
        self,
        perception: dict[str, Any],
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de compasión/armonía y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_CHESED_COMPASSION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "harmony_history": self.harmony_history[-10:],
            "growth_history": self.growth_history[-10:],
            "compassion_history": self.compassion_history[-10:],
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
        experience_id = f"chesed_compassion_{hash(str(perception))}"
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

    def eva_recall_compassion_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de compasión/armonía almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for Chesed experience"}
        quantum_field = None
        _rt = getattr(self, "eva_runtime", None)
        if _rt is not None and hasattr(_rt, "quantum_field"):
            try:
                quantum_field = getattr(_rt, "quantum_field")
            except Exception:
                quantum_field = None
        manifestations = []
        instrs = getattr(reality_bytecode, "instructions", []) or []
        exec_fn = getattr(_rt, "execute_instruction", None)
        for instr in instrs:
            if not callable(exec_fn):
                continue
            try:
                res = exec_fn(instr, quantum_field)
            except Exception:
                res = None
            # avoid awaiting; detect awaitable and mark as pending
            try:
                import inspect

                if inspect.isawaitable(res):
                    symbol_manifest = None
                else:
                    symbol_manifest = res
            except Exception:
                symbol_manifest = res
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] ChesedNode environment hook failed: {e}")
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

    def add_compassion_experience_phase(
        self,
        experience_id: str,
        phase: str,
        perception: dict[str, Any],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de compasión/armonía.
        """
        impulses = self.analyze(perception)
        intention = {
            "intention_type": "ARCHIVE_CHESED_COMPASSION_EXPERIENCE",
            "perception": perception,
            "impulses": [impulse.to_dict() for impulse in impulses],
            "harmony_history": self.harmony_history[-10:],
            "growth_history": self.growth_history[-10:],
            "compassion_history": self.compassion_history[-10:],
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
        """Lista todas las fases disponibles para una experiencia de Chesed."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_compassion_experience": self.eva_ingest_compassion_experience,
            "eva_recall_compassion_experience": self.eva_recall_compassion_experience,
            "add_compassion_experience_phase": self.add_compassion_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "add_environment_hook": self.add_environment_hook,
        }
