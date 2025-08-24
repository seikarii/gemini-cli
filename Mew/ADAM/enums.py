"""
ADAM Core Enums Module - Consolidated (professionalized)
========================================================

- Use str-backed Enum variants for safe string comparisons across the codebase.
- Provide small helper APIs on enums (values(), from_value(), is_valid()) to make
  runtime usage robust and reduce repetitive boilerplate in callers.
- Preserve semantic helper methods (is_critical, is_recovering, is_optimal, etc.)
  and tighten type hints.
- Keep backward-compatible string values (some use Spanish labels intentionally).
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Any


class _BaseStrEnum(str, Enum):
    """Mixin enum to expose common helpful utilities."""

    def __str__(self) -> str:  # pragma: no cover - tiny utility
        return self.value

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def names(cls) -> list[str]:
        return [e.name for e in cls]

    @classmethod
    def from_value(cls, value: Any) -> _BaseStrEnum | None:
        """Resolve enum by value or name (permissive). Returns None if not found."""
        if value is None:
            return None
        for e in cls:
            if value == e or str(value) == e.value or str(value) == e.name:
                return e
        return None

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        return cls.from_value(value) is not None


@unique
class OmegaState(_BaseStrEnum):
    DORMANT = "dormant"
    AWAKENING = "awakening"
    ACTIVE = "active"
    ENHANCED = "enhanced"
    TRANSCENDENT = "transcendent"
    QUANTUM_SYNC = "quantum_sync"
    OMNIPRESENT = "omnipresent"


@unique
class ProtectionLevel(_BaseStrEnum):
    BASIC = "basic"
    ADVANCED = "advanced"
    QUANTUM = "quantum"
    DIVINE = "divine"
    ABSOLUTE = "absolute"


@unique
class ThreatType(_BaseStrEnum):
    CORRUPTION = "corruption"
    MANIPULATION = "manipulation"
    ENERGY_DRAIN = "energy_drain"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    CONSCIOUSNESS_ATTACK = "consciousness_attack"
    GENOME_HACK = "genome_hack"
    REALITY_BREACH = "reality_breach"


@unique
class CrisisType(_BaseStrEnum):
    HEALTH_CRITICAL = "health_critical"
    PURPOSE_FAILURE = "purpose_failure"
    IDENTITY_CORRUPTION = "identity_corruption"
    EXISTENTIAL_THREAT = "existential_threat"
    INTERNAL_CONTRADICTION = "internal_contradiction"

    # lightweight severity mapping used by crisis handlers
    def severity_score(self) -> float:
        mapping = {
            CrisisType.HEALTH_CRITICAL: 0.9,
            CrisisType.EXISTENTIAL_THREAT: 1.0,
            CrisisType.IDENTITY_CORRUPTION: 0.8,
            CrisisType.PURPOSE_FAILURE: 0.6,
            CrisisType.INTERNAL_CONTRADICTION: 0.5,
        }
        return float(mapping.get(self, 0.5))


@unique
class EvolutionType(_BaseStrEnum):
    OVERCLOCK_CHAKRAS = "overclock_chakras"
    UNLOCK_DORMANT_ABILITY = "unlock_dormant_ability"
    METAMORPHOSIS_PHYSICAL = "metamorphosis_physical"
    GENOME_EMERGENCY_REVERT = "genome_emergency_revert"
    NONE = "none"


@unique
class ArchetypeType(_BaseStrEnum):
    INNOCENT = "innocent"
    ORPHAN = "orphan"
    WARRIOR = "warrior"
    CAREGIVER = "caregiver"
    SEEKER = "seeker"
    LOVER = "lover"
    DESTROYER = "destroyer"
    CREATOR = "creator"
    RULER = "ruler"
    MAGICIAN = "magician"
    SAGE = "sage"
    JESTER = "jester"
    HERO = "hero"
    TRICKSTER = "trickster"
    OUTCAST = "outcast"
    REBEL = "rebel"
    MYSTIC = "mystic"
    GUARDIAN = "guardian"
    EXPLORER = "explorer"
    TRANSFORMER = "transformer"
    GUIDE = "guide"
    DESTINY = "destiny"


@unique
class ChakraType(_BaseStrEnum):
    RAIZ = "raiz"
    SACRO = "sacro"
    PLEXO_SOLAR = "plexo_solar"
    CORAZON = "corazon"
    GARGANTA = "garganta"
    TERCER_OJO = "tercer_ojo"
    CORONA = "corona"
    SOUL_STAR = "estrella_del_alma"
    EARTH_STAR = "estrella_de_la_tierra"


@unique
class ActionCategory(_BaseStrEnum):
    PHYSICAL = "physical"
    MENTAL = "mental"
    SOCIAL = "social"
    SPIRITUAL = "spiritual"
    CREATIVE = "creative"
    SURVIVAL = "survival"
    COMMUNICATION = "communication"
    MANIPULATION = "manipulation"
    EMOTIONAL = "emotional"
    ADAPTIVE = "adaptive"
    DIAGNOSTIC = "diagnostic"
    META = "meta"


@unique
class ImpulseType(_BaseStrEnum):
    PATTERN_RECOGNITION = "pattern_recognition"
    LOGICAL_STRUCTURE = "logical_structure"
    CAUSAL_INFERENCE = "causal_inference"
    DOUBT_INJECTION = "doubt_injection"
    ILLUSION_DETECTION = "illusion_detection"
    CHAOS_PROPAGATION = "chaos_propagation"
    MEMORY_FORMATION = "memory_formation"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    CHAOS_EMERGENCE = "chaos_emergence"
    INTENTIONAL_ACTIVATION = "intentional_activation"
    SELF_REFLECTION = "self_reflection"
    GOAL_PRIORITIZATION = "goal_prioritization"
    ERROR_CORRECTION = "error_correction"
    NOVELTY_DETECTION = "novelty_detection"
    SYNCHRONICITY_EVENT = "synchronicity_event"
    STRUCTURE_ENHANCEMENT = "structure_enhancement"
    EXISTENTIAL_UNCERTAINTY = "existential_uncertainty"
    FUTURE_PROJECTION = "future_projection"
    SOCIAL_BONDING = "social_bonding"
    PERCEPTION_INTUITION = "perception_intuition"
    MEMORY_RECALL = "memory_recall"
    MULTIVERSE_BRANCHING = "multiverse_branching"
    BENCHMARKING = "benchmarking"


@unique
class FisiologicalState(_BaseStrEnum):
    HEALTHY = "healthy"
    STRESSED = "stressed"
    EXHAUSTED = "exhausted"
    RECOVERING = "recovering"
    OPTIMAL = "optimal"
    CRITICAL = "critical"
    HOMEOSTASIS = "homeostasis"
    INFLAMED = "inflamed"
    SLEEPING = "sleeping"
    SHOCK = "shock"
    ADAPTIVE = "adaptive"

    @classmethod
    def is_critical(cls, state: FisiologicalState | str) -> bool:
        resolved = cls.from_value(state)
        return resolved in {cls.CRITICAL, cls.SHOCK, cls.EXHAUSTED}

    @classmethod
    def is_recovering(cls, state: FisiologicalState | str) -> bool:
        resolved = cls.from_value(state)
        return resolved in {cls.RECOVERING, cls.ADAPTIVE, cls.SLEEPING}

    @classmethod
    def is_optimal(cls, state: FisiologicalState | str) -> bool:
        resolved = cls.from_value(state)
        return resolved in {cls.OPTIMAL, cls.HEALTHY, cls.HOMEOSTASIS}


@unique
class NeurotransmitterType(_BaseStrEnum):
    DOPAMINE = "dopamina"
    SEROTONIN = "serotonina"
    CORTISOL = "cortisol"
    ADRENALINE = "adrenalina"
    OXYTOCIN = "oxitocina"
    GABA = "gaba"
    ACETYLCHOLINE = "acetilcolina"
    NOREPINEPHRINE = "norepinefrina"
    ENDORPHIN = "endorfinas"
    MELATONIN = "melatonina"
    HISTAMINE = "histamina"
    GLUTAMATE = "glutamato"
    SUBSTANCE_P = "sustancia_p"

    @classmethod
    def is_stress_related(cls, nt: NeurotransmitterType | str) -> bool:
        resolved = cls.from_value(nt)
        return resolved in {
            cls.CORTISOL,
            cls.ADRENALINE,
            cls.NOREPINEPHRINE,
            cls.HISTAMINE,
        }

    @classmethod
    def is_relaxing(cls, nt: NeurotransmitterType | str) -> bool:
        resolved = cls.from_value(nt)
        return resolved in {
            cls.GABA,
            cls.SEROTONIN,
            cls.OXYTOCIN,
            cls.MELATONIN,
            cls.ENDORPHIN,
        }

    @classmethod
    def is_activating(cls, nt: NeurotransmitterType | str) -> bool:
        resolved = cls.from_value(nt)
        return resolved in {
            cls.DOPAMINE,
            cls.GLUTAMATE,
            cls.ACETYLCHOLINE,
            cls.SUBSTANCE_P,
        }


@unique
class PhysiologicalEventType(_BaseStrEnum):
    EXERCISE = "exercise"
    REST = "rest"
    STRESS = "stress"
    EAT = "eat"
    INJURY = "injury"
    HYDRATE = "hydrate"
    SLEEP = "sleep"


@unique
class ItemType(_BaseStrEnum):
    WEAPON = "Arma"
    ARMOR = "Armadura"
    ACCESSORY = "Accesorio"
    CONSUMABLE = "Consumible"
    TOOL = "Herramienta"
    ARTIFACT = "Artefacto"


@unique
class ItemRarity(_BaseStrEnum):
    COMMON = "Común"
    UNCOMMON = "Poco Común"
    RARE = "Raro"
    EPIC = "Épico"
    LEGENDARY = "Legendario"
    MYTHIC = "Mítico"


@unique
class TipoPaso(_BaseStrEnum):
    ANALISIS = "analisis"
    CREACION = "creacion"
    MODIFICACION = "modificacion"
    VALIDACION = "validacion"
    EXPLORACION = "exploracion"
    OPTIMIZACION = "optimizacion"
    META_MODIFICACION = "meta_modificacion"
    ACCION = "accion"


__all__ = [
    "OmegaState",
    "ProtectionLevel",
    "ThreatType",
    "CrisisType",
    "EvolutionType",
    "ArchetypeType",
    "ChakraType",
    "ActionCategory",
    "ImpulseType",
    "FisiologicalState",
    "NeurotransmitterType",
    "PhysiologicalEventType",
    "ItemType",
    "ItemRarity",
    "TipoPaso",
    "_BaseStrEnum",
]
