"""
Chakra System (definitive)
--------------------------

Professionalized, hardened version of the Chakra management subsystem.

Key changes:
- Defensive imports and TYPE_CHECKING-aware typing.
- Robust EVA recording (sync/async aware) and serialization helpers.
- Bounded histories, explicit clamps, and clearer invariants.
- Public API: Chakra, SistemaDeChakras with snapshot/load, meditation, balancing,
  diagnostics and EVA persistence hooks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# defensive numpy import (repo pattern)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.enums import ChakraType

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
else:
    EVAMemoryManager = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ChakraProperties:
    """Properties describing each chakra (static metadata)."""

    name: str
    color: str
    element: str
    frequency: float
    base_capacity: float
    regeneration_rate: float
    depletion_sensitivity: float
    physical_associations: list[str]
    psychological_aspects: list[str]


# -- Default configuration map (kept compatible with prior layout) --
CHAKRA_CONFIG: dict[ChakraType, ChakraProperties] = {
    ChakraType.RAIZ: ChakraProperties(
        name="Muladhara",
        color="Red",
        element="Earth",
        frequency=194.18,
        base_capacity=1.0,
        regeneration_rate=0.02,
        depletion_sensitivity=0.8,
        physical_associations=["survival", "grounding", "stability"],
        psychological_aspects=["security", "basic_needs", "foundation"],
    ),
    ChakraType.SACRO: ChakraProperties(
        name="Svadhisthana",
        color="Orange",
        element="Water",
        frequency=210.42,
        base_capacity=0.9,
        regeneration_rate=0.025,
        depletion_sensitivity=0.7,
        physical_associations=["creativity", "sexuality", "emotion"],
        psychological_aspects=["pleasure", "creativity", "relationships"],
    ),
    ChakraType.PLEXO_SOLAR: ChakraProperties(
        name="Manipura",
        color="Yellow",
        element="Fire",
        frequency=126.22,
        base_capacity=0.85,
        regeneration_rate=0.03,
        depletion_sensitivity=0.6,
        physical_associations=["personal_power", "metabolism", "transformation"],
        psychological_aspects=["confidence", "will", "personal_power"],
    ),
    ChakraType.CORAZON: ChakraProperties(
        name="Anahata",
        color="Green",
        element="Air",
        frequency=341.3,
        base_capacity=0.95,
        regeneration_rate=0.028,
        depletion_sensitivity=0.5,
        physical_associations=["love", "compassion", "connection"],
        psychological_aspects=["love", "compassion", "healing", "forgiveness"],
    ),
    ChakraType.GARGANTA: ChakraProperties(
        name="Vishuddha",
        color="Blue",
        element="Space",
        frequency=141.27,
        base_capacity=0.8,
        regeneration_rate=0.022,
        depletion_sensitivity=0.4,
        physical_associations=["communication", "expression", "truth"],
        psychological_aspects=["communication", "truth", "expression"],
    ),
    ChakraType.TERCER_OJO: ChakraProperties(
        name="Ajna",
        color="Indigo",
        element="Light",
        frequency=221.23,
        base_capacity=0.75,
        regeneration_rate=0.018,
        depletion_sensitivity=0.3,
        physical_associations=["intuition", "wisdom", "psychic_abilities"],
        psychological_aspects=["intuition", "wisdom", "spiritual_insight"],
    ),
    ChakraType.CORONA: ChakraProperties(
        name="Sahasrara",
        color="Violet",
        element="Thought",
        frequency=172.06,
        base_capacity=0.7,
        regeneration_rate=0.015,
        depletion_sensitivity=0.2,
        physical_associations=["consciousness", "spirituality", "enlightenment"],
        psychological_aspects=["spiritual_connection", "cosmic_consciousness"],
    ),
    ChakraType.EARTH_STAR: ChakraProperties(
        name="Earth Star",
        color="Brown",
        element="Earth",
        frequency=68.05,
        base_capacity=0.6,
        regeneration_rate=0.018,
        depletion_sensitivity=0.25,
        physical_associations=["grounding", "connection_to_earth"],
        psychological_aspects=["stability", "ancestral_wisdom"],
    ),
    ChakraType.SOUL_STAR: ChakraProperties(
        name="Soul Star",
        color="Magenta",
        element="Cosmic",
        frequency=1000.0,
        base_capacity=0.5,
        regeneration_rate=0.01,
        depletion_sensitivity=0.1,
        physical_associations=["transcendence", "cosmic_connection"],
        psychological_aspects=["divine_will", "universal_love"],
    ),
}


class Chakra:
    """Dynamic chakra representation with safe energy operations and state introspection."""

    def __init__(self, chakra_type: ChakraType, initial_energy: float = 0.6) -> None:
        self.type: ChakraType = chakra_type
        self.properties: ChakraProperties = CHAKRA_CONFIG[chakra_type]
        self.max_capacity: float = max(0.01, float(self.properties.base_capacity))
        self.current_energy: float = max(
            0.0, min(float(initial_energy), self.max_capacity)
        )
        self.energy_flow_rate: float = 0.0
        self.blockage_level: float = 0.0
        self.alignment_with_system: float = 0.5
        self.resonance_frequency: float = float(self.properties.frequency)
        self.energy_history: list[float] = [self.current_energy]
        self.activation_events: list[dict[str, Any]] = []
        self.last_update_time: float = time.time()
        self.is_overcharged: bool = False
        self.is_depleted: bool = False
        self.harmonic_resonance: bool = False
        logger.debug(
            "Initialized chakra %s energy=%s", self.properties.name, self.current_energy
        )

    # -- Core dynamics -------------------------------------------------
    def update_energy(
        self, delta_time: float, external_influences: dict[str, float] | None = None
    ) -> None:
        external_influences = external_influences or {}
        delta_time = float(max(1e-6, delta_time))
        base_regen = float(self.properties.regeneration_rate) * delta_time
        blockage_modifier = 1.0 - (self.blockage_level * 0.8)
        effective_regen = base_regen * blockage_modifier

        external_modifier = 0.0
        for inf_type, strength in external_influences.items():
            try:
                s = float(strength)
            except Exception:
                s = 0.0
            if inf_type in self.properties.physical_associations:
                external_modifier += s * 0.1
            if inf_type in self.properties.psychological_aspects:
                external_modifier += s * 0.15

        energy_change = effective_regen + external_modifier
        prev = self.current_energy
        self.current_energy = max(
            0.0, min(self.max_capacity, self.current_energy + energy_change)
        )
        self.energy_flow_rate = (self.current_energy - prev) / delta_time
        self._update_special_states()
        self.energy_history.append(self.current_energy)
        if len(self.energy_history) > 200:
            self.energy_history = self.energy_history[-200:]
        self.last_update_time = time.time()

    def _update_special_states(self) -> None:
        self.is_overcharged = self.current_energy > (self.max_capacity * 0.9)
        self.is_depleted = self.current_energy < (self.max_capacity * 0.2)
        opt_min = 0.6 * self.max_capacity
        opt_max = 0.8 * self.max_capacity
        self.harmonic_resonance = opt_min <= self.current_energy <= opt_max

    def consume_energy(self, amount: float, purpose: str = "general") -> bool:
        amount = float(max(0.0, amount))
        if self.current_energy >= amount:
            self.current_energy -= amount
            event = {
                "timestamp": time.time(),
                "energy_consumed": amount,
                "purpose": purpose,
                "remaining_energy": self.current_energy,
            }
            self.activation_events.append(event)
            if len(self.activation_events) > 100:
                self.activation_events = self.activation_events[-100:]
            self._update_special_states()
            return True
        return False

    def add_blockage(self, amount: float, source: str = "unknown") -> None:
        amount = float(max(0.0, amount))
        self.blockage_level = min(1.0, self.blockage_level + amount)
        logger.info(
            "Added blockage: %s -> %s (source=%s)", self.properties.name, amount, source
        )

    def clear_blockage(self, amount: float) -> float:
        amount = float(max(0.0, amount))
        cleared = min(self.blockage_level, amount)
        self.blockage_level = max(0.0, self.blockage_level - cleared)
        return cleared

    # -- Utilities ----------------------------------------------------
    def get_resonance_with(self, other: Chakra) -> float:
        freq_diff = abs(self.resonance_frequency - other.resonance_frequency)
        freq_res = 1.0 / (1.0 + freq_diff / 100.0)
        energy_diff = abs(self.current_energy - other.current_energy)
        energy_res = 1.0 - min(
            1.0, energy_diff / max(1e-6, max(self.max_capacity, other.max_capacity))
        )
        return float(max(0.0, min(1.0, freq_res * 0.6 + energy_res * 0.4)))

    def get_influence_on_abilities(self) -> dict[str, float]:
        energy_factor = (
            (self.current_energy / self.max_capacity) if self.max_capacity > 0 else 0.0
        )
        blockage_factor = 1.0 - self.blockage_level
        overall = max(0.0, min(1.0, energy_factor * blockage_factor))
        influences: dict[str, float] = {}
        t = self.type
        if t == ChakraType.RAIZ:
            influences.update(
                {
                    "physical_strength": overall * 0.8,
                    "stability": overall * 0.9,
                    "grounding": overall * 1.0,
                }
            )
        elif t == ChakraType.SACRO:
            influences.update(
                {
                    "creativity": overall * 0.9,
                    "emotional_balance": overall * 0.7,
                    "telekinesis_power": overall * 0.6,
                }
            )
        elif t == ChakraType.PLEXO_SOLAR:
            influences.update(
                {
                    "personal_power": overall * 1.0,
                    "energy_manipulation": overall * 0.8,
                    "confidence": overall * 0.7,
                }
            )
        elif t == ChakraType.CORAZON:
            influences.update(
                {
                    "empathy": overall * 0.9,
                    "healing_abilities": overall * 0.8,
                    "connection_to_others": overall * 0.85,
                }
            )
        elif t == ChakraType.GARGANTA:
            influences.update(
                {
                    "communication": overall * 0.9,
                    "truth_perception": overall * 0.7,
                    "sound_manipulation": overall * 0.6,
                }
            )
        elif t == ChakraType.TERCER_OJO:
            influences.update(
                {
                    "intuition": overall * 0.95,
                    "psychic_abilities": overall * 0.8,
                    "blink_accuracy": overall * 0.9,
                }
            )
        elif t == ChakraType.CORONA:
            influences.update(
                {
                    "spiritual_connection": overall * 1.0,
                    "consciousness_expansion": overall * 0.9,
                    "reality_perception": overall * 0.8,
                }
            )
        return influences

    def get_state_summary(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.properties.name,
            "energy": float(self.current_energy),
            "capacity": float(self.max_capacity),
            "energy_percentage": (
                float((self.current_energy / self.max_capacity) * 100.0)
                if self.max_capacity > 0
                else 0.0
            ),
            "blockage_level": float(self.blockage_level),
            "flow_rate": float(self.energy_flow_rate),
            "is_overcharged": bool(self.is_overcharged),
            "is_depleted": bool(self.is_depleted),
            "harmonic_resonance": bool(self.harmonic_resonance),
            "recent_activations": len(self.activation_events),
            "frequency": float(self.resonance_frequency),
        }


class SistemaDeChakras:
    """Full chakra system. Designed to be resilient and EV A-friendly."""

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config = config or AdamConfig()
        self.eva_manager = eva_manager
        self.entity_id = entity_id

        self.chakras: dict[ChakraType, Chakra] = {}
        for chakra_type in ChakraType:
            initial = self._get_initial_energy_for_chakra(chakra_type)
            self.chakras[chakra_type] = Chakra(chakra_type, initial)

        self.overall_alignment: float = 0.5
        self.energy_circulation_rate: float = 0.02
        self.system_coherence: float = 0.6
        self.harmonic_resonance_active: bool = False
        self.alignment_history: list[float] = [self.overall_alignment]
        self.last_system_update: float = time.time()
        self.spiritual_development_level: float = 0.3
        logger.info("Chakra system initialized for entity '%s'", self.entity_id)

    def _get_initial_energy_for_chakra(self, chakra_type: ChakraType) -> float:
        defaults: dict[ChakraType, float] = {
            ChakraType.RAIZ: 0.7,
            ChakraType.SACRO: 0.6,
            ChakraType.PLEXO_SOLAR: 0.65,
            ChakraType.CORAZON: 0.6,
            ChakraType.GARGANTA: 0.5,
            ChakraType.TERCER_OJO: 0.45,
            ChakraType.CORONA: 0.4,
            ChakraType.EARTH_STAR: 0.55,
            ChakraType.SOUL_STAR: 0.4,
        }
        return float(defaults.get(chakra_type, 0.5))

    # -- Main loop integration ---------------------------------------
    def update(
        self, delta_time: float, external_influences: dict[str, float] | None = None
    ) -> None:
        external_influences = external_influences or {}
        delta_time = float(max(1e-6, delta_time))
        for chakra in self.chakras.values():
            chakra.update_energy(delta_time, external_influences)

        self._update_energy_circulation(delta_time)
        self._update_system_alignment()
        self._check_harmonic_resonance()

        self.alignment_history.append(self.overall_alignment)
        if len(self.alignment_history) > 500:
            self.alignment_history = self.alignment_history[-500:]
        self.last_system_update = time.time()

        # Best-effort EVA record (sync/async aware)
        if self.eva_manager:
            try:
                rec = getattr(self.eva_manager, "record_experience", None)
                if rec:
                    result = rec(
                        entity_id=self.entity_id,
                        event_type="chakra_system_update",
                        data={
                            "overall_alignment": self.overall_alignment,
                            "system_coherence": self.system_coherence,
                            "harmonic_resonance_active": self.harmonic_resonance_active,
                            "snapshot": self.get_energy_distribution(),
                        },
                        qualia_state=None,
                    )
                    if hasattr(result, "__await__"):
                        try:
                            asyncio.create_task(result)
                        except Exception:
                            # fallback: attempt to run on loop (best-effort)
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # can't run synchronously when loop active, just log
                                    logger.debug(
                                        "EVA record returned coroutine but loop already running"
                                    )
                                else:
                                    loop.run_until_complete(result)
                            except Exception:
                                logger.debug(
                                    "Failed to schedule EVA async record", exc_info=True
                                )
            except Exception:
                logger.debug("Failed to record chakra update to EVA", exc_info=True)

    def _update_energy_circulation(self, delta_time: float) -> None:
        circulation_amount = float(self.energy_circulation_rate) * delta_time
        connections = [
            (ChakraType.RAIZ, ChakraType.SACRO, 0.8),
            (ChakraType.SACRO, ChakraType.PLEXO_SOLAR, 0.7),
            (ChakraType.PLEXO_SOLAR, ChakraType.CORAZON, 0.9),
            (ChakraType.CORAZON, ChakraType.GARGANTA, 0.7),
            (ChakraType.GARGANTA, ChakraType.TERCER_OJO, 0.6),
            (ChakraType.TERCER_OJO, ChakraType.CORONA, 0.8),
            (ChakraType.RAIZ, ChakraType.CORONA, 0.3),
            (ChakraType.SACRO, ChakraType.GARGANTA, 0.2),
        ]
        for a, b, strength in connections:
            ca = self.chakras.get(a)
            cb = self.chakras.get(b)
            if ca is None or cb is None:
                continue
            energy_diff = ca.current_energy - cb.current_energy
            flow_amount = energy_diff * circulation_amount * float(strength) * 0.1
            if flow_amount > 0:
                max_flow = min(
                    ca.current_energy * 0.05,
                    (cb.max_capacity - cb.current_energy) * 0.1,
                    abs(flow_amount),
                )
                flow_eff = (1.0 - ca.blockage_level) * (1.0 - cb.blockage_level)
                actual_flow = max_flow * flow_eff
                ca.current_energy = max(0.0, ca.current_energy - actual_flow)
                cb.current_energy = min(
                    cb.max_capacity, cb.current_energy + actual_flow
                )

    def _update_system_alignment(self) -> None:
        energies = [
            chakra.current_energy / chakra.max_capacity
            for chakra in self.chakras.values()
            if chakra.max_capacity > 0
        ]
        if not energies:
            self.overall_alignment = 0.0
            self.system_coherence = 0.0
            return
        mean = sum(energies) / len(energies)
        variance = sum((e - mean) ** 2 for e in energies) / len(energies)
        avg_block = sum(
            chakra.blockage_level for chakra in self.chakras.values()
        ) / len(self.chakras)
        variance_factor = float(1.0 / (1.0 + variance * 10.0))
        blockage_factor = float(1.0 - avg_block)
        self.overall_alignment = max(
            0.0, min(1.0, variance_factor * 0.6 + blockage_factor * 0.4)
        )
        self.system_coherence = float(self.overall_alignment * mean)

    def _check_harmonic_resonance(self) -> None:
        resonant = sum(
            1 for chakra in self.chakras.values() if chakra.harmonic_resonance
        )
        self.harmonic_resonance_active = resonant >= 4
        if self.harmonic_resonance_active:
            for chakra in self.chakras.values():
                if not chakra.harmonic_resonance:
                    chakra.current_energy = min(
                        chakra.max_capacity, chakra.current_energy + 0.01
                    )

    # -- Public interventions ---------------------------------------
    def align(self, success_rate: float) -> None:
        improvement = (float(success_rate) - 0.5) * 0.05
        key = [ChakraType.RAIZ, ChakraType.PLEXO_SOLAR, ChakraType.CORAZON]
        for t in key:
            c = self.chakras.get(t)
            if not c:
                continue
            if improvement > 0:
                c.current_energy = min(c.max_capacity, c.current_energy + improvement)
                c.blockage_level = max(0.0, c.blockage_level - improvement * 0.5)
            else:
                c.add_blockage(abs(improvement) * 0.3, "performance_stress")
        self._update_system_alignment()

    def get_chakra_influence(self, chakra_type: ChakraType, ability: str) -> float:
        c = self.chakras.get(chakra_type)
        if not c:
            return 0.0
        return float(c.get_influence_on_abilities().get(ability, 0.0))

    def get_combined_influence(self, chakras: list[ChakraType], ability: str) -> float:
        vals: list[float] = []
        for t in chakras:
            vals.append(self.get_chakra_influence(t, ability))
        if not vals:
            return 0.0
        avg = sum(vals) / len(vals)
        synergy = float(self.overall_alignment * 0.2)
        return float(min(1.0, avg + synergy))

    def get_overall_alignment(self) -> float:
        return float(self.overall_alignment)

    def get_energy_distribution(self) -> dict[str, float]:
        return {t.value: float(c.current_energy) for t, c in self.chakras.items()}

    def get_system_status(self) -> dict[str, Any]:
        chakra_states = {
            t.value: c.get_state_summary() for t, c in self.chakras.items()
        }
        return {
            "chakras": chakra_states,
            "overall_alignment": float(self.overall_alignment),
            "system_coherence": float(self.system_coherence),
            "energy_circulation_rate": float(self.energy_circulation_rate),
            "harmonic_resonance_active": bool(self.harmonic_resonance_active),
            "total_system_energy": float(
                sum(c.current_energy for c in self.chakras.values())
            ),
            "average_blockage": float(
                sum(c.blockage_level for c in self.chakras.values())
                / max(1, len(self.chakras))
            ),
        }

    def perform_chakra_balancing(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "energy_redistributed": 0.0,
            "blockages_cleared": 0.0,
            "chakras_balanced": [],
        }
        energy_levels = [
            (t, c.current_energy / c.max_capacity) for t, c in self.chakras.items()
        ]
        energy_levels.sort(key=lambda x: x[1])
        low = energy_levels[:3]
        high = energy_levels[-3:]
        for (ht, hlevel), (lt, llevel) in zip(high, low, strict=False):
            if hlevel > llevel + 0.2:
                high_c = self.chakras[ht]
                low_c = self.chakras[lt]
                transfer = min(
                    (hlevel - llevel) * 0.1,
                    high_c.current_energy * 0.05,
                    low_c.max_capacity - low_c.current_energy,
                )
                if transfer > 0.01:
                    high_c.current_energy = max(0.0, high_c.current_energy - transfer)
                    low_c.current_energy = min(
                        low_c.max_capacity, low_c.current_energy + transfer
                    )
                    result["energy_redistributed"] += transfer
                    result["chakras_balanced"].extend([ht.value, lt.value])
        for c in self.chakras.values():
            if c.current_energy > c.max_capacity * 0.6 and c.blockage_level > 0:
                cleared = c.clear_blockage(0.05)
                result["blockages_cleared"] += cleared
        self._update_system_alignment()
        return result

    def perform_chakra_meditation(
        self, focus_chakra: ChakraType | None = None, duration: float = 60.0
    ) -> dict[str, Any]:
        res: dict[str, Any] = {
            "energy_gained": 0.0,
            "alignment_improvement": 0.0,
            "spiritual_experience_gained": False,
        }
        effectiveness = 0.5 + (float(self.spiritual_development_level) * 0.5)
        dur_factor = min(1.0, float(duration) / 300.0)
        if focus_chakra:
            c = self.chakras.get(focus_chakra)
            if c:
                gain = 0.1 * effectiveness * dur_factor
                cleared = c.clear_blockage(0.05 * effectiveness * dur_factor)
                c.current_energy = min(c.max_capacity, c.current_energy + gain)
                res["energy_gained"] = gain
                res["blockages_cleared"] = cleared
                if c.current_energy > c.max_capacity * 0.9:
                    res["spiritual_experience_gained"] = True
        else:
            total_gain = 0.0
            total_cleared = 0.0
            for c in self.chakras.values():
                gain = 0.05 * effectiveness * dur_factor
                cleared = c.clear_blockage(0.02 * effectiveness * dur_factor)
                c.current_energy = min(c.max_capacity, c.current_energy + gain)
                total_gain += gain
                total_cleared += cleared
            res["energy_gained"] = total_gain
            res["blockages_cleared"] = total_cleared
        prev = self.get_overall_alignment()
        self._update_system_alignment()
        res["alignment_improvement"] = self.get_overall_alignment() - prev

        # EVA recording best-effort (supports async/sync)
        if self.eva_manager:
            try:
                rec = getattr(self.eva_manager, "record_experience", None)
                if rec:
                    result = rec(
                        entity_id=self.entity_id,
                        event_type="chakra_meditation",
                        data={
                            "focus_chakra": (
                                focus_chakra.value if focus_chakra else "all"
                            ),
                            "duration": duration,
                            "results": res,
                        },
                        qualia_state=None,
                    )
                    if hasattr(result, "__await__"):
                        try:
                            asyncio.create_task(result)
                        except Exception:
                            logger.debug(
                                "Could not schedule async EVA meditation record",
                                exc_info=True,
                            )
            except Exception:
                logger.debug("Failed to record meditation to EVA", exc_info=True)
        return res

    def diagnose_spiritual_blockages(self) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "overall_health": "good",
            "critical_issues": [],
            "recommendations": [],
            "chakra_analysis": {},
        }
        for t, c in self.chakras.items():
            analysis = {
                "energy_level": (
                    c.current_energy / c.max_capacity if c.max_capacity > 0 else 0.0
                ),
                "blockage_severity": c.blockage_level,
                "flow_health": abs(c.energy_flow_rate),
                "status": "healthy",
            }
            if c.blockage_level > 0.7:
                analysis["status"] = "severely_blocked"
                diag["critical_issues"].append(f"{t.value} chakra severely blocked")
                diag["recommendations"].append(f"Focus meditation on {t.value} chakra")
            elif c.current_energy < c.max_capacity * 0.3:
                analysis["status"] = "depleted"
                diag["recommendations"].append(
                    f"Energy restoration needed for {t.value}"
                )
            elif c.is_overcharged:
                analysis["status"] = "overcharged"
                diag["recommendations"].append(f"Balance {t.value} chakra energy")
            diag["chakra_analysis"][t.value] = analysis
        if len(diag["critical_issues"]) > 2:
            diag["overall_health"] = "critical"
        elif len(diag["recommendations"]) > 3:
            diag["overall_health"] = "needs_attention"
        return diag

    # -- Persistence / Serialization ---------------------------------
    def to_serializable(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "timestamp": time.time(),
            "overall_alignment": float(self.overall_alignment),
            "chakras": {
                t.value: c.get_state_summary() for t, c in self.chakras.items()
            },
            "spiritual_development_level": float(self.spiritual_development_level),
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        try:
            self.entity_id = data.get("entity_id", self.entity_id)
            stored = data.get("chakras", {})
            for key, state in stored.items():
                try:
                    t = ChakraType(key)
                except Exception:
                    logger.debug("Unknown chakra key on load: %s", key)
                    continue
                c = self.chakras.get(t)
                if not c:
                    continue
                # Load safe numeric fields
                c.current_energy = float(state.get("energy", c.current_energy))
                c.blockage_level = float(state.get("blockage_level", c.blockage_level))
                c.energy_history = (
                    list(state.get("energy_history", c.energy_history))
                    if isinstance(state.get("energy_history", None), list)
                    else c.energy_history
                )
            self._update_system_alignment()
        except Exception:
            logger.exception("Failed to load chakra system state")
