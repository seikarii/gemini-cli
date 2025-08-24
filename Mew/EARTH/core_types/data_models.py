"""
Data models for consciousness and symbolic processing.
Provides advanced data structures for pattern recognition and perception data.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import EVAExperience, QualiaState, RealityBytecode
else:
    EVAExperience = Any
    QualiaState = Any
    RealityBytecode = Any


@dataclass
class PerceptionData:
    """
    Advanced perception data structure for dialectical processing.
    Contains raw sensory data and metadata about the perception.
    """

    timestamp: float = field(default_factory=time.time)
    source_type: str = "external"  # "external", "internal", "memory"
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.raw_data:
            self.raw_data = {}


@dataclass
class PatternRecognitionResult:
    """
    Advanced pattern recognition result from dialectical processing.
    Contains symbolic and archetypal information for conscious integration.
    """

    pattern_type: str = "unknown"
    confidence: float = 0.5
    complexity: float = 0.5
    archetypes_activated: list[str] = field(default_factory=list)
    semantic_features: dict[str, float] = field(default_factory=dict)
    emotional_valence: float = 0.0
    temporal_coherence: float = 0.5
    data: dict[str, Any] | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if not self.semantic_features:
            self.semantic_features = {}
        if not self.archetypes_activated:
            self.archetypes_activated = []


@dataclass
class EVAPerceptionData(PerceptionData):
    """
    Percepción avanzada extendida para integración con EVA.
    Incluye estado de qualia, fase, hooks de entorno y memoria viviente.
    """

    qualia_state: QualiaState | None = None
    eva_phase: str = "default"
    eva_runtime: Any | None = None
    eva_memory_store: dict = field(default_factory=dict)
    eva_experience_store: dict = field(default_factory=dict)
    eva_phases: dict = field(default_factory=dict)
    _environment_hooks: list = field(default_factory=list)

    def ingest_experience(self, phase: str | None = None) -> str:
        """
        Compila la percepción en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_PERCEPTION_EXPERIENCE",
            "experience": self.raw_data,
            "qualia": self.qualia_state,
            "phase": phase,
        }
        if self.eva_runtime and hasattr(self.eva_runtime, "divine_compiler"):
            try:
                _eva = getattr(self, "eva_runtime", None)
                if _eva is None:
                    bytecode = []
                else:
                    _dc = getattr(_eva, "divine_compiler", None)
                    compile_fn = (
                        getattr(_dc, "compile_intention", None)
                        if _dc is not None
                        else None
                    )
                    if callable(compile_fn):
                        try:
                            bytecode = compile_fn(intention)
                        except Exception:
                            bytecode = []
                    else:
                        bytecode = []
            except Exception:
                bytecode = []
            experience_id = f"eva_perception_{hash(str(self.raw_data))}"
            reality_bytecode = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=self.qualia_state or QualiaState(),
                phase=phase,
                timestamp=self.timestamp,
            )
            self.eva_memory_store[experience_id] = reality_bytecode
            if phase not in self.eva_phases:
                self.eva_phases[phase] = {}
            self.eva_phases[phase][experience_id] = reality_bytecode
            self.eva_experience_store[experience_id] = reality_bytecode
            for hook in self._environment_hooks:
                try:
                    hook(reality_bytecode)
                except Exception as e:
                    print(f"[EVA-PERCEPTION] Environment hook failed: {e}")
            return experience_id
        return ""

    def recall_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de percepción almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA perception experience"}
        manifestations = []
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        _eva = getattr(self, "eva_runtime", None)
        if quantum_field and _eva and hasattr(_eva, "execute_instruction"):
            for instr in reality_bytecode.instructions:
                symbol_manifest = _eva.execute_instruction(instr, quantum_field)
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            print(f"[EVA-PERCEPTION] Manifestation hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state or QualiaState(),
            timestamp=reality_bytecode.timestamp,
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
            "timestamp": eva_experience.timestamp,
        }

    def add_environment_hook(self, hook: Callable[..., Any]):
        self._environment_hooks.append(hook)

    def set_memory_phase(self, phase: str):
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-PERCEPTION] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def get_eva_api(self) -> dict:
        return {
            "ingest_experience": self.ingest_experience,
            "recall_experience": self.recall_experience,
            "add_environment_hook": self.add_environment_hook,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
        }


@dataclass
class EVAPatternRecognitionResult(PatternRecognitionResult):
    """
    Resultado avanzado de reconocimiento de patrones extendido para EVA.
    Incluye estado de qualia, fase, hooks de entorno y memoria viviente.
    """

    qualia_state: QualiaState | None = None
    eva_phase: str = "default"
    eva_runtime: Any | None = None
    eva_memory_store: dict = field(default_factory=dict)
    eva_experience_store: dict = field(default_factory=dict)
    eva_phases: dict = field(default_factory=dict)
    _environment_hooks: list = field(default_factory=list)

    def ingest_experience(self, phase: str | None = None) -> str:
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_PATTERN_RECOGNITION_EXPERIENCE",
            "experience": self.data,
            "qualia": self.qualia_state,
            "phase": phase,
        }
        if self.eva_runtime and hasattr(self.eva_runtime, "divine_compiler"):
            try:
                _eva = getattr(self, "eva_runtime", None)
                if _eva is None:
                    bytecode = []
                else:
                    _dc = getattr(_eva, "divine_compiler", None)
                    compile_fn = (
                        getattr(_dc, "compile_intention", None)
                        if _dc is not None
                        else None
                    )
                    if callable(compile_fn):
                        try:
                            bytecode = compile_fn(intention)
                        except Exception:
                            bytecode = []
                    else:
                        bytecode = []
            except Exception:
                bytecode = []
            experience_id = f"eva_pattern_{hash(str(self.data))}"
            reality_bytecode = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=self.qualia_state or QualiaState(),
                phase=phase,
                timestamp=time.time(),
            )
            self.eva_memory_store[experience_id] = reality_bytecode
            if phase not in self.eva_phases:
                self.eva_phases[phase] = {}
            self.eva_phases[phase][experience_id] = reality_bytecode
            self.eva_experience_store[experience_id] = reality_bytecode
            for hook in self._environment_hooks:
                try:
                    hook(reality_bytecode)
                except Exception as e:
                    print(f"[EVA-PATTERN] Environment hook failed: {e}")
            return experience_id
        return ""

    def recall_experience(self, cue: str, phase: str | None = None) -> dict:
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA pattern experience"}
        manifestations = []
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        _eva = getattr(self, "eva_runtime", None)
        if quantum_field and _eva and hasattr(_eva, "execute_instruction"):
            for instr in reality_bytecode.instructions:
                symbol_manifest = _eva.execute_instruction(instr, quantum_field)
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            print(f"[EVA-PATTERN] Manifestation hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state or QualiaState(),
            timestamp=reality_bytecode.timestamp,
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
            "timestamp": eva_experience.timestamp,
        }
