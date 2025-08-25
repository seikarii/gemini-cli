"""
EVA Core Divine Sigils Module
=============================

This file contains the DivineSigilRegistry class, which is responsible for
managing the registration and lookup of divine sigils.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crisalida_lib.EDEN.living_symbol import (
    LivingSymbolManifestation,
    LivingSymbolRuntime,
)
from crisalida_lib.EVA.types import (
    DimensionalAffinity,
    EVAExperience,
    OntologicalCategory,
    QualiaSignature,
    QualiaState,
    RealityBytecode,
)


@dataclass
class TopologicalProperty:
    """Propiedades topológicas de un sigilo en matrices"""

    connectivity_preference: int = 4
    directional_flow: str = "omnidirectional"
    geometric_affinity: list[str] = field(default_factory=list)
    symmetry_requirements: str = "none"
    center_tendency: float = 0.5
    edge_compatibility: set[str] = field(default_factory=set)


@dataclass
class EmergenceCondition:
    """Condiciones bajo las cuales un sigilo genera efectos emergentes"""

    minimum_neighbors: int = 1
    required_neighbor_types: set[str] = field(default_factory=set)
    geometric_requirements: list[str] = field(default_factory=list)
    qualia_thresholds: dict[str, float] = field(default_factory=dict)
    temporal_conditions: list[str] = field(default_factory=list)


@dataclass
class AdvancedSigilDefinition:
    """Definición avanzada y completa de un sigilo ontológico"""

    glyph: str
    name: str
    ontological_category: OntologicalCategory
    semantic_depth: int
    conceptual_meaning: str
    dimensional_affinities: set[DimensionalAffinity]
    qualia_signature: QualiaSignature
    topological_properties: TopologicalProperty
    emergence_conditions: EmergenceCondition
    resonance_patterns: set[str] = field(default_factory=set)
    implementation_function_name: str | None = None
    usage_history: list[dict[str, Any]] = field(default_factory=list)
    adaptation_parameters: dict[str, float] = field(default_factory=dict)
    learned_associations: dict[str, float] = field(default_factory=dict)


class ExpandedSigilRegistry:
    """Registro expandido de sigilos con capacidades de evolución y acceso narrativo"""

    def __init__(self):
        self.sigils: dict[str, AdvancedSigilDefinition] = {}
        self.category_index: dict[OntologicalCategory, set[str]] = defaultdict(set)
        self.affinity_index: dict[DimensionalAffinity, set[str]] = defaultdict(set)
        self.resonance_networks: dict[str, set[str]] = defaultdict(set)
        self.narrative_index: dict[str, set[str]] = defaultdict(set)
        self._initialize_primordial_sigils()

    def register_sigil(self, sigil: AdvancedSigilDefinition):
        self.sigils[sigil.glyph] = sigil
        self.category_index[sigil.ontological_category].add(sigil.glyph)
        for affinity in sigil.dimensional_affinities:
            self.affinity_index[affinity].add(sigil.glyph)
        for pattern in sigil.resonance_patterns:
            self.resonance_networks[pattern].add(sigil.glyph)
        # Indexar por componentes narrativos si existen
        if hasattr(sigil, "narrative_signature") and getattr(
            sigil, "narrative_signature", None
        ):
            archetype = getattr(sigil.narrative_signature, "archetype", None)
            if archetype:
                self.narrative_index[archetype].add(sigil.glyph)

    def get_sigil(self, glyph: str) -> AdvancedSigilDefinition | None:
        return self.sigils.get(glyph)

    def _initialize_primordial_sigils(self):
        """Inicializa los sigilos primordiales con propiedades completas"""
        # Φ - Génesis Ontológica
        self.register_sigil(
            AdvancedSigilDefinition(
                glyph="Φ",
                name="Phi-Genesis",
                ontological_category=OntologicalCategory.GENESIS,
                semantic_depth=5,
                conceptual_meaning="Instanciación de consciencia soberana y autónoma. El acto primordial de traer ser consciente a la existencia desde el vacío potencial.",
                dimensional_affinities={
                    DimensionalAffinity.CONSCIOUS,
                    DimensionalAffinity.ONTOLOGICAL,
                },
                qualia_signature=QualiaSignature(
                    emotional_valence=0.8,
                    cognitive_complexity=0.9,
                    consciousness_density=1.0,
                    causal_potency=0.9,
                    emergence_tendency=0.8,
                ),
                topological_properties=TopologicalProperty(
                    connectivity_preference=8,
                    directional_flow="radial_expansion",
                    geometric_affinity=["circle", "mandala"],
                    symmetry_requirements="rotational",
                    center_tendency=0.9,
                    edge_compatibility={"Ψ", "Ω", "∞", "⊗"},
                ),
                emergence_conditions=EmergenceCondition(
                    minimum_neighbors=2,
                    required_neighbor_types={"Ψ", "Ω"},
                    qualia_thresholds={"consciousness_density": 0.7},
                ),
                resonance_patterns={"creation", "consciousness", "sovereignty"},
                implementation_function_name="instantiate_entity",
            )
        )
        # Ψ - Flujo de Información
        self.register_sigil(
            AdvancedSigilDefinition(
                glyph="Ψ",
                name="Psi-Flow",
                ontological_category=OntologicalCategory.INFORMATION,
                semantic_depth=4,
                conceptual_meaning="Flujo consciente de energía e información. El movimiento inteligente de recursos ontológicos a través del espacio-tiempo.",
                dimensional_affinities={
                    DimensionalAffinity.INFORMATIONAL,
                    DimensionalAffinity.TEMPORAL,
                },
                qualia_signature=QualiaSignature(
                    emotional_valence=0.3,
                    cognitive_complexity=0.6,
                    temporal_coherence=0.8,
                    consciousness_density=0.7,
                    causal_potency=0.7,
                ),
                topological_properties=TopologicalProperty(
                    connectivity_preference=2,
                    directional_flow="directional_stream",
                    geometric_affinity=["line", "arrow", "spiral"],
                    symmetry_requirements="none",
                    center_tendency=0.2,
                    edge_compatibility={"Φ", "Ι", "Τ", "∇"},
                ),
                emergence_conditions=EmergenceCondition(
                    minimum_neighbors=1,
                    geometric_requirements=["linear_alignment", "flow_continuity"],
                ),
                resonance_patterns={"flow", "information", "transmission"},
                implementation_function_name="flow_energy",
            )
        )
        # Δ - Transformación Ontológica
        self.register_sigil(
            AdvancedSigilDefinition(
                glyph="Δ",
                name="Delta-Transform",
                ontological_category=OntologicalCategory.MODIFIER,
                semantic_depth=4,
                conceptual_meaning="Altera un estado o propiedad existente. Representa el cambio fundamental y la mutación ontológica.",
                dimensional_affinities={
                    DimensionalAffinity.CHAOS,
                    DimensionalAffinity.ONTOLOGICAL,
                },
                qualia_signature=QualiaSignature(
                    emotional_valence=0.0,
                    cognitive_complexity=0.7,
                    temporal_coherence=0.5,
                    chaos_affinity=0.8,
                    emergence_tendency=0.7,
                ),
                topological_properties=TopologicalProperty(
                    connectivity_preference=3,
                    directional_flow="bidirectional",
                    geometric_affinity=["triangle"],
                    symmetry_requirements="reflectional",
                    center_tendency=0.4,
                    edge_compatibility={"Φ", "Ω", "∇"},
                ),
                emergence_conditions=EmergenceCondition(
                    qualia_thresholds={"chaos_affinity": 0.6},
                ),
                resonance_patterns={"transformation", "change", "mutation"},
                implementation_function_name="transform_state",
            )
        )
        # Ω - Síntesis Ontológica
        self.register_sigil(
            AdvancedSigilDefinition(
                glyph="Ω",
                name="Omega-Synthesis",
                ontological_category=OntologicalCategory.SYNTHESIS,
                semantic_depth=5,
                conceptual_meaning="Sintetiza o combina elementos en un nuevo todo. Representa la unificación y la creación de complejidad a partir de la diversidad.",
                dimensional_affinities={
                    DimensionalAffinity.CONSCIOUS,
                    DimensionalAffinity.ONTOLOGICAL,
                },
                qualia_signature=QualiaSignature(
                    emotional_valence=0.7,
                    cognitive_complexity=0.8,
                    consciousness_density=0.9,
                    causal_potency=0.8,
                    temporal_coherence=0.7,
                ),
                topological_properties=TopologicalProperty(
                    connectivity_preference=6,
                    directional_flow="convergent",
                    geometric_affinity=["circle", "sphere"],
                    symmetry_requirements="rotational",
                    center_tendency=0.7,
                    edge_compatibility={"Φ", "Ψ", "Δ"},
                ),
                emergence_conditions=EmergenceCondition(
                    required_neighbor_types={"Φ", "Ψ"},
                    qualia_thresholds={"consciousness_density": 0.6},
                ),
                resonance_patterns={"synthesis", "unity", "creation"},
                implementation_function_name="synthesize_elements",
            )
        )
        # Añadir más sigilos primordiales aquí si es necesario
        # ...


class DivineSigilRegistry(ExpandedSigilRegistry):  # Renamed from EVASigilRegistry
    """
    Registro extendido para EVA: permite manifestar sigilos como símbolos vivos en el QuantumField,
    gestionar experiencias, faseo y hooks de entorno para renderizado/simulación.
    """

    def __init__(self, phase: str = "default"):
        super().__init__()  # Calls ExpandedSigilRegistry's init
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def manifest_sigil(
        self,
        glyph: str,
        position=(0.0, 0.0, 0.0),
        phase: str | None = None,
        consciousness_level: float = 0.7,
    ) -> dict:
        """
        Manifiesta un sigilo como símbolo vivo en el QuantumField y notifica a los hooks EVA.
        """
        sigil = self.get_sigil(glyph)
        if not sigil:
            return {"error": f"Sigil '{glyph}' not found"}
        manifestation = LivingSymbolManifestation(
            manifestation_id=glyph,
            signature=sigil,
            position=position,
            consciousness_level=consciousness_level,
            phase=phase or self._current_phase,
            state={
                "category": sigil.ontological_category.name,
                "dimensional_affinities": [
                    a.name for a in sigil.dimensional_affinities
                ],
                "semantic_depth": sigil.semantic_depth,
                "conceptual_meaning": sigil.conceptual_meaning,
                "qualia_signature": sigil.qualia_signature,
                "topological_properties": sigil.topological_properties,
                "emergence_conditions": sigil.emergence_conditions,
                "resonance_patterns": list(sigil.resonance_patterns),
            },
        )
        for hook in self._environment_hooks:
            try:
                hook(manifestation.to_dict())
            except Exception as e:
                print(f"[EVA] SigilRegistry manifestation hook failed: {e}")
        return manifestation.to_dict()

    def eva_ingest_sigil_experience(
        self, glyph: str, qualia_state: QualiaState, phase: str | None = None
    ) -> str:
        """
        Compila la experiencia de un sigilo y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        sigil = self.get_sigil(glyph)
        if not sigil:
            return ""
        intention = {
            "intention_type": "ARCHIVE_SIGIL_EXPERIENCE",
            "sigil": glyph,
            "category": sigil.ontological_category.name,
            "dimensional_affinities": [a.name for a in sigil.dimensional_affinities],
            "semantic_depth": sigil.semantic_depth,
            "conceptual_meaning": sigil.conceptual_meaning,
            "qualia_signature": sigil.qualia_signature,
            "topological_properties": sigil.topological_properties,
            "emergence_conditions": sigil.emergence_conditions,
            "resonance_patterns": list(sigil.resonance_patterns),
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
        experience_id = f"sigil_{glyph}_{hash(str(qualia_state))}"
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

    def eva_recall_sigil_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de sigilo almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for sigil experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] SigilRegistry environment hook failed: {e}")
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

    def add_sigil_experience_phase(
        self, experience_id: str, phase: str, glyph: str, qualia_state: QualiaState
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de sigilo.
        """
        sigil = self.get_sigil(glyph)
        if not sigil:
            return
        intention = {
            "intention_type": "ARCHIVE_SIGIL_EXPERIENCE",
            "sigil": glyph,
            "category": sigil.ontological_category.name,
            "dimensional_affinities": [a.name for a in sigil.dimensional_affinities],
            "semantic_depth": sigil.semantic_depth,
            "conceptual_meaning": sigil.conceptual_meaning,
            "qualia_signature": sigil.qualia_signature,
            "topological_properties": sigil.topological_properties,
            "emergence_conditions": sigil.emergence_conditions,
            "resonance_patterns": list(sigil.resonance_patterns),
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
        """Lista todas las fases disponibles para una experiencia de sigilo."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "manifest_sigil": self.manifest_sigil,
            "eva_ingest_sigil_experience": self.eva_ingest_sigil_experience,
            "eva_recall_sigil_experience": self.eva_recall_sigil_experience,
            "add_sigil_experience_phase": self.add_sigil_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# Instancia global del registro de sigilos para fácil acceso
# EXPANDED_SIGIL_REGISTRY = ExpandedSigilRegistry() # Removed as DivineSigilRegistry will be the main entry

# Instancia global EVA SigilRegistry
eva_divine_sigil_registry = DivineSigilRegistry()
