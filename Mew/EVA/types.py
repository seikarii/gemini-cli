"""
EVA CORE TYPES - Consolidated Type Definitions
==============================================

This file contains the master list of all core data types, enums, and dataclasses
for the EVA (Evolved Virtual Architecture) module. Centralizing these types
ensures consistency and simplifies dependency management across the new architecture.
"""

from dataclasses import dataclass, field
from typing import Any

# Import all core types from the new core_types.py
from crisalida_lib.EVA.core_types import (
    DIVINE_CONSTANTS,
    ARPredictor,
    ASTNode,
    CompilationPhase,
    CompilationResultV7,
    DivineCategory,
    DivineSignature,
    EmergentProperty,
    EntityMemory,
    EVABenchmarkSuite,
    EVAComputeShaderManager,
    EVAExperience,
    EVAHook,
    EVAOptimizationManager,
    ExecutionContext,
    ExecutionResult,
    GrammarConstraint,
    GrammarRule,
    HashingEmbedder,
    KineticState,
    LivingSymbol,
    LivingSymbolManifestation,
    LivingSymbolRuntime,
    MentalLaby,
    MovementPattern,
    Node,
    NoosphereBus,
    OntologicalDomain,
    OntologicalInstruction,
    OntologicalOpcode,
    PerformanceThresholds,
    QualiaCrystal,
    QuantumAmplitude,
    QuantumBytecode,
    QuantumField,
    QuantumInstruction,
    QuantumOperand,
    QuantumState,
    RealityBytecode,
    ResonanceMode,
    ResonancePattern,
    ResonanceType,
    SemanticType,
    SemanticTypeSystem,
    TopologicalGraph,
    TypedValue,
    UniverseMindOrchestrator,
    get_divine_sigil_registry,
)

# EVA-specific types
from .typequalia import QualiaState

# Re-export all types for backward compatibility and simplified imports
__all__ = [
    "SemanticType",
    "OntologicalOpcode",
    "QuantumState",
    "OntologicalDomain",
    "DivineCategory",
    "ResonanceType",
    "GrammarRule",
    "MovementPattern",
    "ResonanceMode",
    "TypedValue",
    "QuantumAmplitude",
    "DivineSignature",
    "OntologicalInstruction",
    "QuantumInstruction",
    "QuantumOperand",
    "QuantumBytecode",
    "RealityBytecode",
    "LivingSymbolManifestation",
    "EVAExperience",
    "ASTNode",
    "ResonancePattern",
    "EmergentProperty",
    "ExecutionResult",
    "ExecutionContext",
    "SemanticTypeSystem",
    "Node",
    "MentalLaby",
    "EntityMemory",
    "QualiaCrystal",
    "NoosphereBus",
    "UniverseMindOrchestrator",
    "HashingEmbedder",
    "ARPredictor",
    "KineticState",
    "LivingSymbol",
    "QuantumField",
    "LivingSymbolRuntime",
    "EVABenchmarkSuite",
    "PerformanceThresholds",
    "EVAComputeShaderManager",
    "EVAOptimizationManager",
    "get_divine_sigil_registry",
    "EVAHook",
    "TopologicalGraph",
    "CompilationResultV7",
    "CompilationPhase",
    "DIVINE_CONSTANTS",
    "GrammarConstraint",
    "QualiaState",
]


# You can add any EVA-specific types here that are not general core types
# For example, if EVA had a unique "MemoryFragment" type that wasn't a core type
@dataclass
class MemoryFragment:
    fragment_id: str
    content: Any
    timestamp: float
    source_experience_id: str
    relevance_score: float = 0.0
    associated_qualia: dict[str, float] = field(default_factory=dict)


# Example of a type that might be specific to EVA's internal memory management
@dataclass
class MemoryIndexEntry:
    experience_id: str
    fragment_ids: list[str]
    keywords: list[str]
    time_range: tuple[float, float]
    associated_entities: list[str]


@dataclass
class QualiaSignature:
    fingerprint: str
    vector: list[float]
    metadata: dict[str, Any]


# This file now primarily serves as an aggregation point for core_types.py
# and for any truly EVA-specific types that don't belong in the general core.
# It helps maintain a clean separation while providing a single import point for EVA.


# If you need to add new types, consider if they are general core types (core_types.py)
# or specific to EVA's memory system (this file).
