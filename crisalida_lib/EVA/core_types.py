from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

EVA_CORE_TYPES_DOC = """
EVA CORE TYPES - Consolidated Type Definitions
==============================================

This file contains the master list of all core data types, enums, and dataclasses
for the EVA (Evolved Virtual Architecture) module. Centralizing these types
ensures consistency and simplifies dependency management across the new architecture.
"""

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime fallback when numpy unavailable
    np = None  # type: ignore

# Annotation-only alias: when type-checking import the real ndarray type,
# at runtime fall back to Any so importing this module doesn't require numpy.
if TYPE_CHECKING:  # pragma: no cover - static typing only
    from numpy import ndarray as NDArray  # type: ignore
else:
    NDArray = Any  # type: ignore

# The following import may be placed after optional runtime checks to avoid
# circular import problems; keep as-is for runtime stability and silence the
# import-top warning from the linter for this specific line.
from .typequalia import QualiaState  # noqa: E402

# Lightweight runtime protocol aliases exported for other packages to import.
# These are conservative Any aliases to reduce import-time typing issues across packages.
# NOTE: concrete classes for QuantumField and LivingSymbolRuntime are defined
# later in this module; avoid placeholder aliases that produce redefinition
# errors with static checkers.

# Attempt to import from upper-level packages; use placeholders on failure.
# QualiaState will remain in types.py for now, or be handled by a placeholder if not found.
# from crisalida_lib.EDEN.qualia_manifold import QualiaState

# --- SEMANTIC & ONTOLOGICAL ENUMS ---


class SemanticType(Enum):
    """Semantic types recognized by the VM and compiler."""

    ENTITY = "LivingEntity"
    SOUL = "SoulKernel"
    BODY = "CuerpoBiologico"
    CHAKRA_SYSTEM = "SistemaDeChakras"
    CHAKRA = "Chakra"
    GENOME = "GenomaComportamiento"
    QUALIA = "QualiaState"
    MIND = "ConsciousMind"
    UNIFIED_FIELD = "UnifiedField"
    EVOLUTION_ENGINE = "SelfModifyingEngine"
    QUALIA_CHAIN = "QualiaChain"
    NOOSPHERE = "Noosphere"
    CRYSTALLIZED_QUALIA = "CrystallizedQualia"
    SIGNED_TRANSACTION = "SignedTransaction"
    PURPOSE = "Proposito"
    HORMONAL_SYSTEM = "SistemaHormonal"
    NUMBER = "Number"
    STRING = "String"
    LIST = "List"
    DICT = "Dict"
    ABSTRACT = "Abstract"
    EXPERIENCE = "Experience"
    FIELD_PATTERN = "FieldPattern"
    KNOT = "ToroidalKnot"
    BYTECODE = "QuantumBytecode"
    ERROR = "Error"


class OntologicalOpcode(Enum):
    """Opcodes for the Ontological Virtual Machine (OVM)."""

    Q_CREATE = "Q_CREATE"
    Q_GET_ENTITY = "Q_GET_ENTITY"
    Q_REMOVE_ENTITY = "Q_REMOVE_ENTITY"
    Q_TRANSFORM = "Q_TRANSFORM"
    Q_NAVIGATE_PATH = "Q_NAVIGATE_PATH"
    Q_MODIFY_ATTRIBUTE = "Q_MODIFY_ATTRIBUTE"
    Q_QUERY_REALITY = "Q_QUERY_REALITY"
    Q_ENTANGLE = "Q_ENTANGLE"
    Q_CALL_METHOD = "Q_CALL_METHOD"
    Q_EVOLVE_CODE = "Q_EVOLVE_CODE"
    Q_JUMP = "Q_JUMP"
    Q_JUMP_IF = "Q_JUMP_IF"
    Q_RETURN = "Q_RETURN"
    Q_PUSH = "Q_PUSH"
    Q_POP = "Q_POP"
    Q_DUP = "Q_DUP"
    Q_NOT_IMPLEMENTED = "Q_NOT_IMPLEMENTED"
    Q_CRYSTALLIZE = "Q_CRYSTALLIZE"
    Q_SIGN_TRANSACTION = "Q_SIGN_TRANSACTION"
    Q_SUBMIT_TRANSACTION = "Q_SUBMIT_TRANSACTION"
    Q_GET_BALANCE = "Q_GET_BALANCE"
    Q_MODIFY_BODY = "Q_MODIFY_BODY"
    Q_RELEASE_HORMONES = "Q_RELEASE_HORMONES"
    Q_SET_PURPOSE = "Q_SET_PURPOSE"
    Q_START_QUALIA_STREAM = "Q_START_QUALIA_STREAM"
    Q_CREATE_PLAYLIST = "Q_CREATE_PLAYLIST"
    Q_MANIFEST_MATTER = "Q_MANIFEST_MATTER"
    Q_FIELD_APPLY = "Q_FIELD_APPLY"
    Q_FIELD_MODULATE = "Q_FIELD_MODULATE"
    Q_FIELD_COLLAPSE = "Q_FIELD_COLLAPSE"
    Q_RESONATE = "Q_RESONATE"
    Q_RESONANCE_HARMONIC = "Q_RESONANCE_HARMONIC"
    Q_RESONANCE_DISSONANT = "Q_RESONANCE_DISSONANT"
    Q_RESONANCE_MODULATE = "Q_RESONANCE_MODULATE"
    Q_RESONANCE_ENTANGLE = "Q_RESONANCE_ENTANGLE"
    Q_RESONANCE_EMERGENT = "Q_RESONANCE_EMERGENT"
    Q_RESONANCE_TRANSCENDENT = "Q_RESONANCE_TRANSCENDENT"
    Q_META_EVOLVE = "Q_META_EVOLVE"
    Q_SYNC = "Q_SYNC"
    Q_BRANCH = "Q_BRANCH"
    Q_OBSERVE = "Q_OBSERVE"
    Q_EVA_INGEST = "Q_EVA_INGEST"
    Q_EVA_RECALL = "Q_EVA_RECALL"
    Q_EVA_PHASE = "Q_EVA_PHASE"
    Q_EVA_MANIFEST = "Q_EVA_MANIFEST"
    Q_EVA_ENV_HOOK = "Q_EVA_ENV_HOOK"


class QuantumState(Enum):
    SUPERPOSITION = "superposición"
    COLLAPSED = "colapsado"
    ENTANGLED = "entrelazado"
    RESONANT = "resonante"
    COHERENT = "coherente"


class OntologicalDomain(Enum):
    """Fundamental ontological domains of reality."""

    CONSCIOUSNESS = "consciencia"
    INFORMATION = "información"
    ENERGY = "energía"
    MATTER = "materia"
    TIME = "tiempo"
    SPACE = "espacio"
    EMERGENCE = "emergencia"
    VOID = "vacío"
    QUANTUM = "cuántica"


class DivineCategory(Enum):
    """Primordial divine categories for sigils."""

    CREATOR = "creador"
    PRESERVER = "preservador"
    TRANSFORMER = "transformador"
    CONNECTOR = "conector"
    OBSERVER = "observador"
    DESTROYER = "destructor"
    INFINITE = "infinito"
    FOCAL = "focal"


class ResonanceType(Enum):
    """Fundamental types of resonance between symbols."""

    HARMONIC = "armónica"
    DISSONANT = "disonante"
    MODULATORY = "modulatoria"
    ENTANGLEMENT = "entrelazamiento"
    EMERGENT = "emergente"
    TRANSCENDENT = "trascendente"


class GrammarRule(Enum):
    """Fundamental grammatical rules of the divine language."""

    CREATION = "creación"
    PRESERVATION = "preservación"
    TRANSFORMATION = "transformación"
    CONNECTION = "conexión"
    OBSERVATION = "observación"
    DESTRUCTION = "destrucción"
    INFINITY = "infinito"
    SYNTHESIS = "síntesis"
    RESONANCE = "resonancia"
    TRANSCENDENCE = "trascendente"


class MovementPattern(Enum):
    STATIC = "estático"
    PULSE = "pulso"
    ORBIT = "órbita"
    SPIRAL = "espiral"
    WAVE = "onda"
    CHAOS = "caos"
    DANCE = "danza"
    TRANSCENDENT = "trascendente"


class ResonanceMode(Enum):
    HARMONIC = "armónica"
    DISONANT = "disonante"
    MODULATORY = "modulatoria"
    ENTANGLED = "entrelazada"
    EMERGENT = "emergente"
    TRANSCENDENT = "trascendente"


# --- CORE DATA STRUCTURES ---


@dataclass
class TypedValue:
    """Value with semantic type information."""

    value: Any
    semantic_type: SemanticType
    type_info: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    error: str | None = None
    phase: str = "default"
    manifestation: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumAmplitude:
    """Complex quantum amplitude for state superposition."""

    real: float
    imag: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.real**2 + self.imag**2)

    @property
    def phase(self) -> float:
        return math.atan2(self.imag, self.real)

    def collapse(self) -> float:
        return self.magnitude**2


@dataclass
class DivineSignature:
    """Unique quantum signature of each divine symbol."""

    glyph: str
    name: str = ""
    category: DivineCategory = DivineCategory.CREATOR
    domains: set[OntologicalDomain] = field(default_factory=set)
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    amplitude: QuantumAmplitude = field(
        default_factory=lambda: QuantumAmplitude(1.0, 0.0)
    )
    frequency: float = 0.0
    resonance_frequency: float = 0.0
    resonance_modes: list[str] = field(default_factory=list)
    consciousness_density: float = 0.0
    causal_potency: float = 0.0
    emergence_tendency: float = 0.0


@dataclass
class OntologicalInstruction:
    """Extended ontological instruction for the OVM."""

    # Accept either OntologicalOpcode enum or a string opcode for compatibility
    opcode: Any
    operands: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumInstruction(OntologicalInstruction):
    """Instruction with quantum properties for the quantum compiler."""

    quantum_coherence: float = 1.0
    resonance_frequency: float = 0.0
    instruction_id: str | None = None
    entanglement_group: str | None = None


@dataclass
class QuantumOperand:
    type: str
    value: Any


@dataclass
class QuantumBytecode:
    """Represents compiled quantum code from a symbolic matrix."""

    bytecode_id: str = field(default_factory=lambda: str(uuid4()))
    source_matrix: list[list[str]] = field(default_factory=list)
    instructions: list[QuantumInstruction] = field(default_factory=list)
    total_coherence: float = 1.0
    entanglement_networks: dict[str, set[str]] = field(default_factory=dict)
    resonance_map: dict[str, float] = field(default_factory=dict)
    compilation_phases: list[str] = field(default_factory=list)
    optimization_level: int = 0
    optimization_passes: list[str] = field(default_factory=list)
    phase: str = "default"
    simulation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealityBytecode:
    """
    Ontological bytecode for EVA's living memory.
    Representa una experiencia compilada, lista para ser ejecutada en el QuantumField.
    """

    bytecode_id: str = field(default_factory=lambda: str(uuid4()))
    instructions: list[OntologicalInstruction] = field(default_factory=list)
    qualia_state: QualiaState = field(default_factory=QualiaState)
    metadata: dict[str, Any] = field(default_factory=dict)
    phase: str = "default"
    timestamp: float = field(default_factory=time.time)


@dataclass
class LivingSymbolManifestation:
    """
    Active manifestation of a living symbol in the QuantumField.
    Usado por EVA para renderizar y simular símbolos en tiempo real.
    """

    manifestation_id: str = field(default_factory=lambda: str(uuid4()))
    signature: DivineSignature = field(
        default_factory=lambda: DivineSignature(glyph="")
    )
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    consciousness_level: float = 0.0
    phase: str = "default"
    state: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class EVAExperience:
    """
    A living experience recorded by EVA.
    Permite gestión avanzada de fases, hooks y simulación.
    """

    experience_id: str = field(default_factory=lambda: str(uuid4()))
    bytecode: RealityBytecode = field(default_factory=RealityBytecode)
    manifestations: list[LivingSymbolManifestation] = field(default_factory=list)
    phase: str = "default"
    # QualiaState is part of an experience and many call sites pass it.
    qualia_state: QualiaState = field(default_factory=QualiaState)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ASTNode:
    """Abstract Syntax Tree node for the Divine Parser."""

    node_type: str
    value: Any = None
    children: list[ASTNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResonancePattern:
    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    primary_sigil: str = ""
    secondary_sigil: str = ""
    resonance_type: ResonanceType = ResonanceType.HARMONIC
    strength: float = 1.0
    frequency: float = 440.0
    phase_shift: float = 0.0
    effects: list[str] = field(default_factory=list)
    emergence_probability: float = 0.0
    distance_threshold: float = 2.0
    alignment_requirement: float = 0.7
    minimum_amplitude: float = 0.5


@dataclass
class EmergentProperty:
    property_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    required_sigils: set[str] = field(default_factory=set)
    spatial_pattern: str = ""
    temporal_sequence: list[str] = field(default_factory=list)
    emergence_threshold: float = 0.8
    stability_factor: float = 0.7
    decay_rate: float = 0.1
    effects: dict[str, float] = field(default_factory=dict)
    duration: float = 1.0


@dataclass
class ExecutionResult:
    """Result of a hardware execution cycle."""

    success: bool
    execution_time: float
    # QualiaState will be imported from types.py or handled by placeholder
    # final_qualia_states: dict[str, QualiaState] = field(default_factory=dict)
    created_entities: list[str] = field(default_factory=list)
    emergent_effects: list[Any] = field(default_factory=list)


@dataclass
class ExecutionContext:
    """Execution context for the OVM."""

    unified_field: Any  # Placeholder for UnifiedField
    current_entity: Any | None = None  # Placeholder for LivingEntity
    local_variables: dict[str, TypedValue] = field(default_factory=dict)


@dataclass
class SemanticTypeSystem:
    """Semantic type system for the OVM."""

    type_mapping: dict[Any, SemanticType] = field(default_factory=dict)
    abstract_types: dict[Any, str] = field(default_factory=dict)


@dataclass
class Node:
    id: int
    kind: str
    embed: NDArray
    data: dict
    valence: float
    arousal: float
    salience: float = 0.1
    last_access: float = field(default_factory=time.time)
    freq: int = 0
    origin: str = "WAKE"
    edges_out: dict[int, float] = field(default_factory=dict)
    edges_in: dict[int, float] = field(default_factory=dict)


@dataclass
class MentalLaby:
    d: int
    embedder: Any
    predictor: Any
    max_nodes: int
    storage_pressure: float
    nodes: dict[int, Node] = field(default_factory=dict)
    next_id: int = 0
    K: int = 8
    temp_store: float = 0.7
    tau: float = 60.0
    replay: Any = field(default_factory=list)
    policy: Any = field(default_factory=list)
    mode: str = "WAKE"
    _last_observed: Any = None


@dataclass
class EntityMemory:
    entity_id: str
    mind: MentalLaby
    affect_bias: float = 0.0
    valence_bias: float = 0.0
    storage_budget: int = 100_000


@dataclass
class QualiaCrystal:
    crystal_id: str
    entity_id: str
    content: dict[str, Any]
    timestamp: float
    resonance_frequency: float
    likes: int = 0
    shares: int = 0
    comments: list[dict] = field(default_factory=list)


@dataclass
class NoosphereBus:
    feed: list[QualiaCrystal] = field(default_factory=list)


@dataclass
class UniverseMindOrchestrator:
    d: int
    embedder: Any
    predictor_factory: Any
    entities: dict[str, EntityMemory] = field(default_factory=dict)
    noosphere: NoosphereBus = field(default_factory=NoosphereBus)


@dataclass
class HashingEmbedder:
    d: int
    ngram: tuple[int, int]
    use_words: bool
    seed: int
    key: Any


@dataclass
class ARPredictor:
    d: int
    lr: float
    l2: float
    init_scale: float
    A: Any


# --- LIVING SYMBOL SYSTEM ---

logger = logging.getLogger(__name__)


@dataclass
class KineticState:
    position: NDArray
    velocity: NDArray
    acceleration: NDArray
    phase: float = 0.0
    amplitude: float = 1.0
    frequency: float = 1.0
    damping: float = 0.99
    quantum_coherence: float = 1.0
    entanglement_partners: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        self.acceleration = np.array(self.acceleration, dtype=float)

    def update(self, dt: float):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.velocity *= self.damping
        self.acceleration *= self.damping
        self.phase = (self.phase + 2 * math.pi * self.frequency * dt) % (2 * math.pi)
        self.quantum_coherence = max(0.1, self.quantum_coherence * 0.9999)


@dataclass
class LivingSymbol:
    symbol_id: str
    signature: DivineSignature
    kinetic_state: KineticState
    movement_pattern: MovementPattern
    consciousness_level: float = 0.0
    awareness_radius: float = 5.0
    connections: dict[str, float] = field(default_factory=dict)
    resonance_partners: dict[str, ResonanceMode] = field(default_factory=dict)
    emergent_properties: dict[str, Any] = field(default_factory=dict)
    memory: list[dict[str, Any]] = field(default_factory=list)
    birth_time: float = field(default_factory=time.time)
    age: float = 0.0
    last_update: float = field(default_factory=time.time)

    def update(self, dt: float, field: QuantumField):
        self.age += dt
        self._apply_movement_pattern(dt)
        self.kinetic_state.update(dt)
        self._interact_with_field(field)
        self._update_consciousness(dt)
        self._process_connections(dt)
        self._record_state()

    def _apply_movement_pattern(self, dt: float):
        pos = self.kinetic_state.position
        acc = self.kinetic_state.acceleration
        phase = self.kinetic_state.phase
        amp = self.kinetic_state.amplitude
        if self.movement_pattern == MovementPattern.STATIC:
            acc += np.random.normal(0, 0.01, 3) * 0.1
        elif self.movement_pattern == MovementPattern.PULSE:
            pulse_force = amp * math.sin(phase) * 0.5
            direction = pos / (np.linalg.norm(pos) + 1e-6)
            acc += direction * pulse_force
        elif self.movement_pattern == MovementPattern.ORBIT:
            radius = np.linalg.norm(pos)
            if radius > 0:
                acc += -pos / radius * amp * 0.1
                tangent = np.array([-pos[1], pos[0], 0])
                tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
                acc += tangent * amp * 0.05
        elif self.movement_pattern == MovementPattern.SPIRAL:
            radial_force = -pos / (np.linalg.norm(pos) + 1e-6) * amp * 0.1
            tangential_force = np.array([-pos[1], pos[0], 0])
            tangential_force = tangential_force / (
                np.linalg.norm(tangential_force) + 1e-6
            )
            acc += radial_force + tangential_force * amp * 0.05
            self.kinetic_state.amplitude *= 0.999
        elif self.movement_pattern == MovementPattern.WAVE:
            wave_force = amp * math.sin(phase + pos[0]) * 0.1
            acc[1] += wave_force
        elif self.movement_pattern == MovementPattern.CHAOS:
            sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
            x, y, z = pos
            acc[0] = sigma * (y - x) * 0.01
            acc[1] = (x * (rho - z) - y) * 0.01
            acc[2] = (x * y - beta * z) * 0.01
        elif self.movement_pattern == MovementPattern.DANCE:
            acc += (
                np.array(
                    [
                        amp * math.sin(phase) * math.cos(phase * 0.7),
                        amp * math.cos(phase) * math.sin(phase * 1.3),
                        amp * math.sin(phase * 1.1) * math.cos(phase * 0.9),
                    ]
                )
                * 0.1
            )
        elif self.movement_pattern == MovementPattern.TRANSCENDENT:
            transcend_force = (
                amp
                * math.sin(phase)
                * np.array(
                    [
                        math.cos(phase * 1.618),
                        math.sin(phase * 1.618),
                        math.cos(phase * 0.618),
                    ]
                )
                * 0.2
            )
            acc += transcend_force
            self.kinetic_state.amplitude *= 1.001

    def _interact_with_field(self, field: QuantumField):
        x, y, z = self.kinetic_state.position.astype(int)
        field_amplitude = field.get_amplitude_at(x, y, z)
        resonance_factor = abs(
            math.sin(
                getattr(self.signature, "phase", 0.0)
                - getattr(field_amplitude, "phase", 0.0)
            )
        )
        field_force = (
            np.array(
                [
                    field_amplitude.real * resonance_factor,
                    getattr(field_amplitude, "imag", field_amplitude.real)
                    * resonance_factor,
                    getattr(field_amplitude, "magnitude", 1.0) * resonance_factor,
                ]
            )
            * 0.1
        )
        self.kinetic_state.acceleration += field_force
        self.kinetic_state.quantum_coherence = min(
            1.0, self.kinetic_state.quantum_coherence * (1.0 + resonance_factor * 0.1)
        )

    def _update_consciousness(self, dt: float):
        age_factor = min(self.age / 100.0, 1.0)
        connection_factor = min(len(self.connections) / 10.0, 1.0)
        resonance_factor = min(len(self.resonance_partners) / 5.0, 1.0)
        target_consciousness = (
            getattr(self.signature, "consciousness_density", 0.5) * age_factor
            + connection_factor * 0.3
            + resonance_factor * 0.3
        )
        self.consciousness_level += (
            (target_consciousness - self.consciousness_level) * dt * 0.1
        )
        self.consciousness_level = max(0.0, min(1.0, self.consciousness_level))

    def _process_connections(self, dt: float):
        weak_connections = [
            sid for sid, strength in self.connections.items() if strength < 0.1
        ]
        for sid in weak_connections:
            del self.connections[sid]
        for sid in self.connections:
            self.connections[sid] = min(1.0, self.connections[sid] + dt * 0.01)

    def _record_state(self):
        current_time = time.time()
        if current_time - self.last_update > 1.0:
            memory_entry = {
                "timestamp": current_time,
                "position": self.kinetic_state.position.copy(),
                "consciousness_level": self.consciousness_level,
                "movement_pattern": self.movement_pattern.value,
                "connections": len(self.connections),
                "resonance_partners": list(self.resonance_partners.keys()),
            }
            self.memory.append(memory_entry)
            if len(self.memory) > 1000:
                self.memory = self.memory[-1000:]
            self.last_update = current_time

    def connect_to(self, other: LivingSymbol, strength: float = 0.5):
        self.connections[other.symbol_id] = strength
        other.connections[self.symbol_id] = strength

    def resonate_with(self, other: LivingSymbol, mode: ResonanceMode):
        self.resonance_partners[other.symbol_id] = mode
        other.resonance_partners[self.symbol_id] = mode
        if mode == ResonanceMode.ENTANGLED:
            self.kinetic_state.entanglement_partners.add(other.symbol_id)
            other.kinetic_state.entanglement_partners.add(self.symbol_id)
            avg_phase = (self.kinetic_state.phase + other.kinetic_state.phase) / 2
            self.kinetic_state.phase = avg_phase
            other.kinetic_state.phase = avg_phase

    def get_influence_radius(self) -> float:
        base_radius = self.awareness_radius
        consciousness_factor = 1.0 + self.consciousness_level
        amplitude_factor = 1.0 + self.kinetic_state.amplitude
        return base_radius * consciousness_factor * amplitude_factor

    def get_symbolic_meaning(self) -> dict[str, Any]:
        return {
            "glyph": self.signature.glyph,
            "name": self.signature.name,
            "category": self.signature.category.value,
            "consciousness_level": self.consciousness_level,
            "movement_pattern": self.movement_pattern.value,
            "position": self.kinetic_state.position.tolist(),
            "age": self.age,
            "connections": len(self.connections),
            "resonance_partners": len(self.resonance_partners),
            "emergent_properties": list(self.emergent_properties.keys()),
        }


@dataclass
class QuantumField:
    def __init__(self, dimensions: tuple[int, int, int] = (20, 20, 20)):
        self.dimensions = dimensions
        self.field_matrix = np.random.normal(0, 0.1, dimensions + (2,)).astype(
            np.complex64
        )
        self.vacuum_energy = 1.0
        self.field_strength = 1.0
        self.coherence_length = 1.0
        self.correlation_length = 2.0
        self.history: list[dict[str, Any]] = []
        self.evolution_time = 0.0

    def get_amplitude_at(self, x: int, y: int, z: int) -> QuantumAmplitude:
        if (
            0 <= x < self.dimensions[0]
            and 0 <= y < self.dimensions[1]
            and 0 <= z < self.dimensions[2]
        ):
            complex_val = self.field_matrix[x, y, z]
            return QuantumAmplitude(complex_val.real, complex_val.imag)
        return QuantumAmplitude(0, 0)

    def set_amplitude_at(self, x: int, y: int, z: int, amplitude: QuantumAmplitude):
        if (
            0 <= x < self.dimensions[0]
            and 0 <= y < self.dimensions[1]
            and 0 <= z < self.dimensions[2]
        ):
            self.field_matrix[x, y, z] = complex(amplitude.real, amplitude.imag)

    def apply_symbol(self, symbol: LivingSymbol):
        pos = symbol.kinetic_state.position.astype(int)
        x, y, z = pos
        influence_radius = symbol.get_influence_radius()
        for dx in range(-int(influence_radius), int(influence_radius) + 1):
            for dy in range(-int(influence_radius), int(influence_radius) + 1):
                for dz in range(-int(influence_radius), int(influence_radius) + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (
                        0 <= nx < self.dimensions[0]
                        and 0 <= ny < self.dimensions[1]
                        and 0 <= nz < self.dimensions[2]
                    ):
                        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if distance <= influence_radius and distance > 0:
                            influence = symbol.signature.amplitude.magnitude / (
                                1 + distance
                            )
                            phase_shift = getattr(
                                symbol.signature, "phase", 0.0
                            ) + distance * getattr(symbol.signature, "frequency", 1.0)
                            current = self.get_amplitude_at(nx, ny, nz)
                            new_real = current.real + influence * math.cos(phase_shift)
                            new_imag = current.imag + influence * math.sin(phase_shift)
                            self.set_amplitude_at(
                                nx, ny, nz, QuantumAmplitude(new_real, new_imag)
                            )

    def evolve(self, dt: float):
        self.evolution_time += dt
        laplacian = self._calculate_laplacian()
        potential = self._calculate_potential()
        hamiltonian = -0.5 * laplacian + potential
        self.field_matrix += -1j * hamiltonian * dt
        total_energy = np.sum(np.abs(self.field_matrix) ** 2)
        if total_energy > 0:
            self.field_matrix /= math.sqrt(total_energy)
        if (
            len(self.history) == 0
            or self.evolution_time - self.history[-1]["time"] > 1.0
        ):
            self.history.append(
                {
                    "time": self.evolution_time,
                    "total_energy": total_energy,
                    "coherence": self._calculate_field_coherence(),
                }
            )
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

    def _calculate_laplacian(self) -> NDArray:
        laplacian = np.zeros_like(self.field_matrix)
        for i in range(1, self.dimensions[0] - 1):
            for j in range(1, self.dimensions[1] - 1):
                for k in range(1, self.dimensions[2] - 1):
                    d2_dx2 = (
                        self.field_matrix[i + 1, j, k]
                        - 2 * self.field_matrix[i, j, k]
                        + self.field_matrix[i - 1, j, k]
                    )
                    d2_dy2 = (
                        self.field_matrix[i, j + 1, k] + self.field_matrix[i, j - 1, k]
                    )
                    d2_dz2 = (
                        self.field_matrix[i, j, k + 1] + self.field_matrix[i, j, k - 1]
                    )
                    laplacian[i, j, k] = d2_dx2 + d2_dy2 + d2_dz2
        return laplacian

    def _calculate_potential(self) -> NDArray:
        potential = np.zeros_like(self.field_matrix)
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    center = np.array(self.dimensions) / 2
                    r = np.linalg.norm(np.array([i, j, k]) - center)
                    potential[i, j, k] = 0.5 * r**2
        return potential

    def _calculate_field_coherence(self) -> float:
        phases = np.angle(self.field_matrix)
        phase_vectors = np.exp(1j * phases)
        coherence_vector = np.mean(phase_vectors)
        coherence = abs(coherence_vector)
        return coherence

    def collapse_wavefunction(self) -> dict[str, Any]:
        probability_density = np.abs(self.field_matrix) ** 2
        threshold = np.mean(probability_density) + np.std(probability_density)
        high_prob_regions = np.where(probability_density > threshold)
        collapsed_reality = {}
        for i in range(len(high_prob_regions[0])):
            x, y, z = (
                high_prob_regions[0][i],
                high_prob_regions[1][i],
                high_prob_regions[2][i],
            )
            amplitude = self.get_amplitude_at(x, y, z)
            collapsed_reality[f"region_{x}_{y}_{z}"] = {
                "position": (x, y, z),
                "probability": getattr(amplitude, "magnitude", 1.0),
                "phase": getattr(amplitude, "phase", 0.0),
                "energy": probability_density[x, y, z],
            }
        return collapsed_reality


# Fix: dynamic import for divine_sigil_registry to avoid circular import
def get_divine_sigil_registry():
    try:
        from crisalida_lib.EVA.divine_sigils import eva_divine_sigil_registry

        return eva_divine_sigil_registry
    except ImportError:
        # Return None or a placeholder if import fails
        return None


# Placeholder classes for missing dependencies
class EVABenchmarkSuite:
    def __init__(self, *args, **kwargs):
        pass

    def run_comprehensive_benchmark(self, *args, **kwargs):
        return type(
            "obj",
            (object,),
            {
                "fps_avg": 0,
                "memory_usage_mb": 0,
                "symbol_count": 0,
                "visual_quality_score": 0,
                "simulation_time_ms": 0,
            },
        )()

    def validate_optimization_performance(self, *args, **kwargs):
        return True, []

    def generate_performance_report(self, *args, **kwargs):
        return {}


class PerformanceThresholds:
    def __init__(self, *args, **kwargs):
        pass


class EVAComputeShaderManager:
    def __init__(self, *args, **kwargs):
        pass

    def simulate_parallel_update(self, *args, **kwargs):
        pass

    def get_compute_performance_metrics(self, *args, **kwargs):
        return {
            "memory_usage_mb": 0,
            "symbols_in_compute": 0,
            "utilization": 0,
            "avg_consciousness": 0,
        }

    # Conservative attributes used by callers; provide simple defaults so static
    # checkers know these attributes exist even when the real implementation
    # is provided at runtime.
    symbol_count: int = 0
    positions: list = []

    def add_symbol_to_compute(self, *args, **kwargs):
        pass


class EVAOptimizationManager:
    def __init__(self, *args, **kwargs):
        self.lod_distance_thresholds = {}
        self.compute_components = {}

    def start_frame_timing(self, *args, **kwargs):
        pass

    def end_frame_timing(self, *args, **kwargs):
        pass

    def optimize_for_target_fps(self, *args, **kwargs):
        pass

    def update_lod_system(self, *args, **kwargs):
        pass

    def update_instance_data(self, *args, **kwargs) -> Any:
        pass

    def get_performance_metrics(self, *args, **kwargs):
        return {
            "fps": 0,
            "avg_frame_time_ms": 0,
            "instance_count": 0,
            "visible_instances": 0,
            "draw_calls_saved": 0,
            "lod_distribution": {},
            "memory_usage_mb": 0,
            "gpu_memory_usage_mb": 0,
        }

    def register_living_symbol(self, *args, **kwargs):
        pass

    def get_instancing_data(self, *args, **kwargs):
        return {
            "positions": [],
            "scales": [],
            "rotations": [],
            "colors": [],
            "lod_levels": [],
            "instance_count": 0,
            "visible_count": 0,
        }


@dataclass
class LivingSymbolRuntime:
    # Optional runtime helpers commonly expected by consumers/tests. Declared here
    # so static checkers (mypy) recognize these attributes even if concrete
    # implementations set them later.
    divine_compiler: Any | None = None
    quantum_field: Any | None = None

    def __init__(self, field_dimensions: tuple[int, int, int] = (20, 20, 20)):
        self.field = QuantumField(field_dimensions)
        self.living_symbols: dict[str, LivingSymbol] = {}
        self.grammar = get_divine_sigil_registry()
        self.dt = 0.016
        self.running = False
        self.time = 0.0
        self.frame_count = 0
        self.total_symbols_created = 0
        self.emergent_events: list[dict[str, Any]] = []
        self._manifestation_hooks: list[Callable[..., Any]] = []

        self.eva_optimization_manager = EVAOptimizationManager(max_instances=10000)
        self.eva_compute_manager = EVAComputeShaderManager(max_symbols=10000)
        self.eva_benchmark_suite = EVABenchmarkSuite(PerformanceThresholds())
        self._enable_eva_optimizations = True
        self._camera_position = np.array([0.0, 0.0, 0.0])
        self._last_performance_check = 0.0

    def execute_instruction(self, instruction: Any, quantum_field: Any = None) -> Any:
        """Best-effort synchronous wrapper used by consumers that expect a
        non-async `execute_instruction`. If an async handler exists we'll
        attempt to run it in an isolated event loop; otherwise return None.

        This function is intentionally defensive: it should not raise in
        normal import-time or static-checking scenarios.
        """
        try:
            # Prefer explicit quantum_field passed in, otherwise fall back to
            # runtime quantum_field if available.
            # prefer explicit quantum_field when provided; otherwise fall back to runtime quantum_field
            _ = (
                quantum_field
                if quantum_field is not None
                else getattr(self, "quantum_field", None)
            )
            # If there is a synchronous implementation, call it.
            if hasattr(
                self, "_execute_instruction"
            ) and not asyncio.iscoroutinefunction(self._execute_instruction):
                return self._execute_instruction(instruction)

            # If there is an async implementation, run it safely.
            if hasattr(self, "_execute_instruction"):
                coro = self._execute_instruction(instruction)
                try:
                    # Try to use existing loop if not running; otherwise create a new one.
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't run a new loop; bail out gracefully.
                        return None
                except Exception:
                    loop = None
                # Run the coroutine in a fresh event loop.
                return asyncio.run(coro)

        except Exception:
            return None
        return None

    def add_manifestation_hook(self, hook: Callable[..., Any]) -> None:
        """Add a hook to be called when symbols are manifested or updated."""
        self._manifestation_hooks.append(hook)

    def _notify_symbol_manifested(self, symbol: LivingSymbol):
        """Notify all hooks about a symbol manifestation."""
        symbol_state = {
            "symbol_id": symbol.symbol_id,
            "signature": (
                symbol.signature.glyph
                if hasattr(symbol.signature, "glyph")
                else str(symbol.signature)
            ),
            "position": (
                symbol.kinetic_state.position.tolist()
                if symbol.kinetic_state is not None
                and hasattr(symbol.kinetic_state, "position")
                and hasattr(symbol.kinetic_state.position, "tolist")
                else [0, 0, 0]
            ),
            "consciousness_level": getattr(symbol, "consciousness_level", 0.5),
            "movement_pattern": (
                symbol.movement_pattern.value
                if hasattr(symbol.movement_pattern, "value")
                else str(symbol.movement_pattern)
            ),
            "amplitude": symbol.kinetic_state.amplitude,
            "phase": symbol.kinetic_state.phase,
            "frequency": symbol.kinetic_state.frequency,
        }

        for hook in self._manifestation_hooks:
            try:
                hook(symbol_state)
            except Exception as e:
                logger.warning(f"Manifestation hook failed: {e}")

    async def execute_bytecode(self, bytecode: QuantumBytecode) -> bool:
        try:
            self.living_symbols.clear()

            # Handle different bytecode formats
            instructions = bytecode.instructions
            if hasattr(instructions, "instructions"):
                # If it's an OntologicalBytecode with nested instructions
                instructions = instructions.instructions
            elif not isinstance(instructions, (list, tuple)):
                # If it's a single instruction, wrap it in a list
                instructions = [instructions] if instructions else []

            for instruction in instructions:
                await self._execute_instruction(instruction)
            return True
        except Exception as e:
            print(f"Error ejecutando bytecode: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return False

    async def _execute_instruction(self, instruction: QuantumInstruction):
        opcode = instruction.opcode
        operands = instruction.operands
        if opcode == OntologicalOpcode.Q_CREATE:
            await self._handle_create(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_RESONATE:
            await self._handle_resonate(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_TRANSFORM:
            await self._handle_transform(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_ENTANGLE:
            await self._handle_entangle(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_FIELD_APPLY:
            await self._handle_field_apply(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_SYNC:
            await self._handle_sync(operands, instruction.metadata)
        elif opcode == OntologicalOpcode.Q_META_EVOLVE:
            await self._handle_meta_evolve(operands, instruction.metadata)

    async def _handle_create(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        if len(operands) >= 2:
            glyph = operands[0].value
            position = operands[1].value
            # Use dynamic divine_sigil_registry import
            registry = get_divine_sigil_registry()
            signature = (
                registry.get_sigil(glyph) if registry else DivineSignature(glyph=glyph)
            )
            kinetic_state = KineticState(
                position=np.array(position, dtype=float),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                phase=getattr(signature, "phase", 0.0),
                amplitude=getattr(
                    signature, "amplitude", QuantumAmplitude(1.0, 0.0)
                ).magnitude,
                frequency=getattr(signature, "frequency", 1.0),
            )
            movement_pattern = self._determine_movement_pattern(signature)
            living_symbol = LivingSymbol(
                symbol_id=f"symbol_{self.total_symbols_created}",
                signature=signature,
                kinetic_state=kinetic_state,
                movement_pattern=movement_pattern,
            )
            self.living_symbols[living_symbol.symbol_id] = living_symbol
            self.total_symbols_created += 1
            self._notify_symbol_manifested(living_symbol)

    def _determine_movement_pattern(
        self,
        signature: DivineSignature,
    ) -> MovementPattern:
        pattern_map = {
            DivineCategory.CREATOR: MovementPattern.PULSE,
            DivineCategory.PRESERVER: MovementPattern.STATIC,
            DivineCategory.TRANSFORMER: MovementPattern.SPIRAL,
            DivineCategory.CONNECTOR: MovementPattern.ORBIT,
            DivineCategory.OBSERVER: MovementPattern.WAVE,
            DivineCategory.DESTROYER: MovementPattern.CHAOS,
            DivineCategory.INFINITE: MovementPattern.TRANSCENDENT,
        }
        return pattern_map.get(signature.category, MovementPattern.STATIC)

    async def _handle_resonate(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        if len(operands) >= 2:
            glyph1 = operands[0].value
            glyph2 = operands[1].value
            symbol1 = self._find_symbol_by_glyph(glyph1)
            symbol2 = self._find_symbol_by_glyph(glyph2)
            if symbol1 and symbol2:
                symbol1.connect_to(symbol2)
                resonance_mode = self._determine_resonance_mode(metadata)
                symbol1.resonate_with(symbol2, resonance_mode)

    def _find_symbol_by_glyph(self, glyph: str) -> LivingSymbol | None:
        for symbol in self.living_symbols.values():
            if symbol.signature.glyph == glyph:
                return symbol
        return None

    def _determine_resonance_mode(self, metadata: dict[str, Any]) -> ResonanceMode:
        pattern_type = metadata.get("pattern_type", "armónica")
        mode_map = {
            "armónica": ResonanceMode.HARMONIC,
            "disonante": ResonanceMode.DISONANT,
            "modulatoria": ResonanceMode.MODULATORY,
            "entrelazamiento": ResonanceMode.ENTANGLED,
            "emergente": ResonanceMode.EMERGENT,
            "trascendente": ResonanceMode.TRANSCENDENT,
        }
        return mode_map.get(pattern_type, ResonanceMode.HARMONIC)

    async def _handle_transform(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        pass

    async def _handle_entangle(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        pass

    async def _handle_field_apply(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        pass

    async def _handle_sync(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        pass

    async def _handle_meta_evolve(
        self,
        operands: list[QuantumOperand],
        metadata: dict[str, Any],
    ):
        pass

    async def run_simulation(self, duration: float = 10.0):
        self.running = True
        start_time = time.time()
        print(
            f"🌟 Iniciando simulación de símbolos vivos ({len(self.living_symbols)} símbolos)"
        )

        # EVA: Start performance monitoring
        if self._enable_eva_optimizations:
            print("🚀 EVA optimizations enabled - monitoring performance")

        try:
            while self.running and (time.time() - start_time) < duration:
                frame_start = time.time()

                # EVA: Start frame timing for performance monitoring
                if self._enable_eva_optimizations:
                    self.eva_optimization_manager.start_frame_timing()

                # Update quantum field
                self.field.evolve(self.dt)

                # EVA: Update LOD system for optimal rendering
                if self._enable_eva_optimizations:
                    # Simulate camera movement for dynamic LOD
                    self._camera_position = np.array(
                        [
                            np.sin(self.time * 0.1) * 30,
                            np.cos(self.time * 0.1) * 30,
                            10.0 + np.sin(self.time * 0.05) * 5,
                        ]
                    )
                    self.eva_optimization_manager.update_lod_system(
                        self._camera_position
                    )

                # Update symbols - use parallel compute if available
                if self._enable_eva_optimizations and len(self.living_symbols) > 100:
                    # Use GPU compute for large symbol counts
                    self.eva_compute_manager.simulate_parallel_update(
                        symbol_interactions=True
                    )

                    # Update symbol positions from GPU compute results
                    for i, (symbol_id, symbol) in enumerate(
                        self.living_symbols.items()
                    ):
                        if i < self.eva_compute_manager.symbol_count:
                            gpu_pos = self.eva_compute_manager.positions[i]
                            if hasattr(symbol, "kinetic_state") and hasattr(
                                symbol.kinetic_state, "position"
                            ):
                                symbol.kinetic_state.position = gpu_pos

                            # Update optimization manager instance data
                            self.eva_optimization_manager.update_instance_data(
                                symbol_id, position=gpu_pos
                            )
                else:
                    # Traditional CPU update for smaller symbol counts
                    for symbol in self.living_symbols.values():
                        symbol.update(self.dt, self.field)
                        self.field.apply_symbol(symbol)

                await self._detect_emergent_events()
                self.time += self.dt
                self.frame_count += 1

                # EVA: End frame timing and adaptive optimization
                if self._enable_eva_optimizations:
                    self.eva_optimization_manager.end_frame_timing()

                    # Adaptive FPS optimization every 60 frames
                    if self.frame_count % 60 == 0:
                        self.eva_optimization_manager.optimize_for_target_fps(60.0)

                if self.frame_count % 60 == 0:
                    await self._print_simulation_status()

                frame_time = time.time() - frame_start
                sleep_time = max(0, self.dt - frame_time)
                await asyncio.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n🛑 Simulación detenida por usuario")
        finally:
            self.running = False
            elapsed = time.time() - start_time
            print(
                f"✅ Simulación completada - {self.frame_count} frames en {elapsed:.2f}s"
            )

            # EVA: Generate performance report
            if self._enable_eva_optimizations:
                await self._print_eva_performance_summary()

    async def _detect_emergent_events(self):
        registry = self.grammar or get_divine_sigil_registry()
        active_sigils = {
            symbol.signature.glyph for symbol in self.living_symbols.values()
        }
        if registry and hasattr(registry, "detect_emergent_properties"):
            emergent_properties = registry.detect_emergent_properties(active_sigils)
            for prop in emergent_properties:
                if not any(
                    event["property"] == prop.name for event in self.emergent_events
                ):
                    event = {
                        "time": self.time,
                        "property": prop.name,
                        "probability": prop.emergence_threshold,
                        "effects": prop.effects,
                    }
                    self.emergent_events.append(event)
                    print(f"🌟 Evento emergente detectado: {prop.name}")

    async def _print_simulation_status(self):
        total_consciousness = sum(
            symbol.consciousness_level for symbol in self.living_symbols.values()
        )
        avg_consciousness = (
            total_consciousness / len(self.living_symbols) if self.living_symbols else 0
        )
        total_connections = sum(
            len(symbol.connections) for symbol in self.living_symbols.values()
        )
        field_coherence = self.field._calculate_field_coherence()
        print(f"⏱️  Frame {self.frame_count} - Tiempo: {self.time:.2f}s")
        print(
            f"   Símbolos: {len(self.living_symbols)} | Consciencia avg: {avg_consciousness:.3f}"
        )
        print(
            f"   Conexiones: {total_connections} | Coherencia campo: {field_coherence:.3f}"
        )
        print(f"   Eventos emergentes: {len(self.emergent_events)}")

    def get_simulation_state(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "frame_count": self.frame_count,
            "living_symbols": {
                symbol_id: symbol.get_symbolic_meaning()
                for symbol_id, symbol in self.living_symbols.items()
            },
            "field_coherence": self.field._calculate_field_coherence(),
            "emergent_events": self.emergent_events[-10:],
            "total_symbols_created": self.total_symbols_created,
        }

    def collapse_reality(self) -> dict[str, Any]:
        return self.field.collapse_wavefunction()

    # --- EVA: Extensión avanzada para integración con la Memoria Viviente, hooks 3D/4D, faseo y optimización orientada a datos ---

    def add_eva_hook(self, eva_hook: EVAHook):
        """
        Registra un hook EVA para integración avanzada con entornos 3D/4D.
        El hook debe implementar on_symbol_manifested, on_phase_changed, on_experience_recalled.
        """
        self._manifestation_hooks.append(eva_hook.on_symbol_manifested)

    def notify_eva_hooks_phase(self, phase: str):
        """
        Notifica a los hooks EVA el cambio de fase/timeline.
        """
        for hook in self._manifestation_hooks:
            if hasattr(hook, "on_phase_changed"):
                try:
                    hook.on_phase_changed(phase)
                except Exception as e:
                    logger.warning(f"EVA phase hook failed: {e}")

    def notify_eva_hooks_experience(self, experience: dict):
        """
        Notifica a los hooks EVA la manifestación de una experiencia completa.
        """
        for hook in self._manifestation_hooks:
            if hasattr(hook, "on_experience_recalled"):
                try:
                    hook.on_experience_recalled(experience)
                except Exception as e:
                    logger.warning(f"EVA experience hook failed: {e}")

    def manifest_symbol(self, symbol: LivingSymbol, phase: str = "default"):
        """
        Manifiesta un símbolo vivo en el QuantumField y notifica a los hooks EVA.
        """
        manifestation = LivingSymbolManifestation(
            manifestation_id=symbol.symbol_id,
            signature=symbol.signature,
            position=tuple(symbol.kinetic_state.position.tolist()),
            consciousness_level=symbol.consciousness_level,
            phase=phase,
            state=symbol.get_symbolic_meaning(),
            timestamp=time.time(),
        )
        self._notify_symbol_manifested(symbol)
        for hook in self._manifestation_hooks:
            try:
                hook(manifestation.to_dict())
            except Exception as e:
                logger.warning(f"EVA manifestation hook failed: {e}")

    # --- EVA: Faseo y timeline ---
    def set_phase(self, phase: str):
        """
        Cambia la fase/timeline activa y notifica a los hooks EVA.
        """
        self.notify_eva_hooks_phase(phase)

    def get_phase(self) -> str:
        """
        Devuelve la fase/timeline actual (si se gestiona internamente).
        """
        # Si se gestiona internamente, devolver el valor; si no, devolver "default"
        return getattr(self, "_current_phase", "default")

    # --- EVA: Optimización orientada a datos (ECS pattern) ---
    # Nota: Para una implementación completa ECS, separar componentes y sistemas.
    # Aquí se muestra la estructura básica para integración futura.
    def get_ecs_components(self) -> dict:
        """
        Devuelve los componentes ECS de todos los símbolos vivos.
        """
        return {
            symbol_id: {
                "PositionComponent": symbol.kinetic_state.position.tolist(),
                "EmotionalStateComponent": symbol.emergent_properties.get(
                    "emotional_state", {}
                ),
                "CognitiveStateComponent": symbol.emergent_properties.get(
                    "cognitive_state", {}
                ),
                "RealityConstructorComponent": symbol.get_symbolic_meaning(),
            }
            for symbol_id, symbol in self.living_symbols.items()
        }

    # --- EVA: Benchmark y diagnóstico de rendimiento ---
    def get_performance_metrics(self) -> dict:
        """
        Devuelve métricas clave de rendimiento para benchmarking EVA.
        """
        return {
            "frame_count": self.frame_count,
            "living_symbols": len(self.living_symbols),
            "field_coherence": self.field._calculate_field_coherence(),
            "total_symbols_created": self.total_symbols_created,
            "avg_consciousness": (
                sum(
                    symbol.consciousness_level
                    for symbol in self.living_symbols.values()
                )
                / len(self.living_symbols)
                if self.living_symbols
                else 0
            ),
            "memory_usage": self._estimate_memory_usage(),
        }

    def _estimate_memory_usage(self) -> float:
        """
        Estima el uso de memoria (RAM) por los símbolos vivos y el campo cuántico.
        """
        symbol_mem = sum(
            symbol.kinetic_state.position.nbytes
            + symbol.kinetic_state.velocity.nbytes
            + symbol.kinetic_state.acceleration.nbytes
            for symbol in self.living_symbols.values()
        )
        field_mem = (
            self.field.field_matrix.nbytes if hasattr(self.field, "field_matrix") else 0
        )
        return (symbol_mem + field_mem) / (1024 * 1024)  # MB

    async def _print_eva_performance_summary(self):
        """Print comprehensive EVA performance summary"""
        if not self._enable_eva_optimizations:
            return

        print("\n" + "=" * 60)
        print("🚀 EVA PERFORMANCE SUMMARY")
        print("=" * 60)

        # Get optimization metrics
        opt_metrics = self.eva_optimization_manager.get_performance_metrics()
        compute_metrics = self.eva_compute_manager.get_compute_performance_metrics()

        print("📊 Rendering Performance:")
        print(f"   FPS: {opt_metrics['fps']:.1f}")
        print(f"   Frame Time: {opt_metrics['avg_frame_time_ms']:.2f}ms")
        print(
            f"   Instances: {opt_metrics['instance_count']} total, {opt_metrics['visible_instances']} visible"
        )
        print(f"   Draw Calls Saved: {opt_metrics['draw_calls_saved']}")

        print("\n🎮 LOD Distribution:")
        lod_dist = opt_metrics["lod_distribution"]
        for lod_level, count in lod_dist.items():
            print(f"   {lod_level.upper()}: {count} symbols")

        print("\n💾 Memory Usage:")
        print(f"   System RAM: {opt_metrics['memory_usage_mb']:.1f}MB")
        print(f"   GPU Memory: {compute_metrics['gpu_memory_usage_mb']:.1f}MB")
        print(f"   Compute Memory: {compute_metrics['memory_usage_mb']:.1f}MB")

        print("\n⚡ Compute Performance:")
        print(f"   Symbols in GPU: {compute_metrics['symbols_in_compute']}")
        print(f"   GPU Utilization: {compute_metrics['utilization'] * 100:.1f}%")
        print(f"   Avg Consciousness: {compute_metrics['avg_consciousness']:.3f}")

        # Run quick benchmark
        if len(self.living_symbols) > 10:
            print("\n🧪 Running Performance Benchmark...")
            try:
                self.eva_benchmark_suite.run_comprehensive_benchmark(
                    self,
                    self.eva_optimization_manager,
                    self.eva_compute_manager,
                    test_duration=2.0,
                )

                passed, failures = (
                    self.eva_benchmark_suite.validate_optimization_performance()
                )
                print(f"   Benchmark Status: {'✅ PASS' if passed else '❌ FAIL'}")

                if failures:
                    print("   Issues:")
                    for failure in failures[:3]:  # Show first 3 failures
                        print(f"     - {failure}")

            except Exception as e:
                print(f"   Benchmark failed: {e}")

        print("=" * 60)

    def get_eva_rendering_data(self) -> dict[str, Any]:
        """Get rendering data optimized for GPU instancing"""
        if not self._enable_eva_optimizations:
            return {}

        instancing_data = self.eva_optimization_manager.get_instancing_data()

        def safe_list(arr):
            return arr.tolist() if hasattr(arr, "tolist") else arr

        return {
            "gpu_instance_data": {
                "positions": safe_list(instancing_data["positions"]),
                "scales": safe_list(instancing_data["scales"]),
                "rotations": safe_list(instancing_data["rotations"]),
                "colors": safe_list(instancing_data["colors"]),
                "lod_levels": safe_list(instancing_data["lod_levels"]),
            },
            "instance_count": instancing_data["instance_count"],
            "visible_count": instancing_data["visible_count"],
            "camera_position": safe_list(self._camera_position),
            "lod_thresholds": {
                str(level): threshold
                for level, threshold in self.eva_optimization_manager.lod_distance_thresholds.items()
            },
        }

    def set_eva_camera_position(self, position: tuple[float, float, float]):
        """Update camera position for LOD calculations"""
        self._camera_position = np.array(position, dtype=np.float32)

    def toggle_eva_optimizations(self, enabled: bool = True):
        """Enable or disable EVA optimizations"""
        self._enable_eva_optimizations = enabled
        if enabled:
            print("🚀 EVA optimizations enabled")
        else:
            print("⚠️ EVA optimizations disabled")

    def run_eva_benchmark(self, test_duration: float = 5.0) -> dict:
        """Run comprehensive EVA performance benchmark"""
        if not self._enable_eva_optimizations:
            return {"error": "EVA optimizations not enabled"}

        try:
            result = self.eva_benchmark_suite.run_comprehensive_benchmark(
                self,
                self.eva_optimization_manager,
                self.eva_compute_manager,
                test_duration=test_duration,
            )

            passed, failures = (
                self.eva_benchmark_suite.validate_optimization_performance(result)
            )

            return {
                "benchmark_result": {
                    "fps_avg": result.fps_avg,
                    "memory_usage_mb": result.memory_usage_mb,
                    "symbol_count": result.symbol_count,
                    "visual_quality": result.visual_quality_score,
                    "simulation_time_ms": result.simulation_time_ms,
                },
                "validation": {"passed": passed, "failures": failures},
                "performance_report": self.eva_benchmark_suite.generate_performance_report(),
            }
        except Exception as e:
            return {"error": f"Benchmark failed: {e}"}


# --- PLACEHOLDER & FORWARD-DECLARED TYPES ---
# These are placeholders for complex types that might have their own files
# or are defined in other high-level packages (e.g., ADAM, EARTH).


class TopologicalGraph:
    pass


class CompilationResultV7:
    pass


class CompilationPhase:
    pass


# --- CONSTANTS ---


class DIVINE_CONSTANTS:
    RESONANCE_THRESHOLD = 0.7


# --- HOOKS & INTERFACES ---


class EVAHook(ABC):
    """
    Abstract Base Class for EVA hooks.
    Allows external systems (like a 3D renderer) to react to events in the EVA runtime.
    """

    @abstractmethod
    def on_symbol_manifested(self, manifestation: LivingSymbolManifestation):
        """Called when a symbol is created or its state changes significantly."""
        pass

    @abstractmethod
    def on_phase_changed(self, new_phase: str):
        """Called when the active simulation phase/timeline changes."""
        pass

    @abstractmethod
    def on_experience_recalled(self, experience: EVAExperience):
        """Called when a complete experience is recalled from memory."""
        pass


@dataclass
class GrammarConstraint:
    """
    Represents a constraint in the divine grammar, defining forbidden combinations
    or required conditions for resonance.
    """

    constraint_id: str = field(default_factory=lambda: f"constraint_{uuid4()}")
    name: str = "Unnamed Constraint"
    description: str = ""
    restricted_sigils: set[str] = field(default_factory=set)
    forbidden_pairs: set[tuple[str, str]] = field(default_factory=set)
    context_domains: set[OntologicalDomain] = field(default_factory=set)
    violation_effects: list[str] = field(default_factory=list)
    penalty_factor: float = 0.5
    temporal_window: float | None = None
