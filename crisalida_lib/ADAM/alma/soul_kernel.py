"""
Unified Spiritual Kernel
======================

This module defines the SoulKernel, which integrates chakras, mudras, archetypes,
and the logic for spiritual evolution.

This file has been refactored and completed with full orchestration logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

# Defensive imports for EVA/Qualia types (support legacy layout)
try:
    from crisalida_lib.EVA.types import QualiaState  # preferred modern location
except Exception:  # pragma: no cover - fallback for older layouts
    try:
        from crisalida_lib.EVA.typequalia import QualiaState  # legacy
    except Exception:  # pragma: no cover - runtime fallback
        QualiaState = Any  # type: ignore

# Core subsystems (expected to exist in the refactored ADAM/alma package)
from crisalida_lib.ADAM.alma.archetype_system import SistemaDeArquetipos
from crisalida_lib.ADAM.alma.chakra_system import ChakraType, SistemaDeChakras
from crisalida_lib.ADAM.alma.mudra_system import SistemaDeMudras
from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.enums import ArchetypeType

# EVA memory manager (best-effort integration)
try:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,
    )
except Exception:  # pragma: no cover - fallback typing
    EVAMemoryManager = Any  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SoulKernel:
    """
    The unified spiritual kernel, orchestrating the soul's subsystems.

    Responsibilities:
      - Keep chakra, archetype and mudra subsystems in sync.
      - Compute qualia influences and intentionality biases.
      - Expose a compact serializable snapshot and best-effort EVA persistence.
      - Provide safe guards for absent subsystems and async EVA backends.
    """

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config: AdamConfig = config or AdamConfig()  # type: ignore
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id

        # instantiate subsystems defensively (best-effort)
        try:
            self.chakras: SistemaDeChakras = SistemaDeChakras(
                config=self.config,
                eva_manager=self.eva_manager,
                entity_id=self.entity_id,
            )
        except Exception:
            logger.exception("Failed to initialize SistemaDeChakras; using stub")
            self.chakras = SistemaDeChakras()  # type: ignore

        try:
            self.mudras: SistemaDeMudras = SistemaDeMudras(
                config=self.config,
                eva_manager=self.eva_manager,
                entity_id=self.entity_id,
            )
        except Exception:
            logger.exception("Failed to initialize SistemaDeMudras; using stub")
            self.mudras = SistemaDeMudras()  # type: ignore

        try:
            self.arquetipos: SistemaDeArquetipos = SistemaDeArquetipos(
                config=self.config,
                eva_manager=self.eva_manager,
                entity_id=self.entity_id,
            )
        except Exception:
            logger.exception("Failed to initialize SistemaDeArquetipos; using stub")
            self.arquetipos = SistemaDeArquetipos()  # type: ignore

        self.chakra_archetype_affinities = (
            self._initialize_chakra_archetype_affinities()
        )

        # core state
        self.spiritual_development_level: float = 0.3
        self.karma_balance: float = 0.0
        self.enlightenment_progress: float = 0.0
        self.soul_integrity: float = 1.0
        self.spiritual_experiences: list[dict[str, Any]] = []
        self.last_update_ts: float = time.time()
        self._hooks: list[Any] = []

        logger.info("SoulKernel initialized for entity '%s'.", self.entity_id)

    def _initialize_chakra_archetype_affinities(
        self,
    ) -> dict[ChakraType, list[tuple[ArchetypeType, float]]]:
        """Define affinities between chakras and archetypes (tunable mapping)."""
        return {
            ChakraType.RAIZ: [
                (ArchetypeType.ORPHAN, 0.8),
                (ArchetypeType.CAREGIVER, 0.6),
            ],
            ChakraType.SACRO: [
                (ArchetypeType.CREATOR, 0.9),
                (ArchetypeType.LOVER, 0.8),
            ],
            ChakraType.PLEXO_SOLAR: [
                (ArchetypeType.RULER, 0.9),
                (ArchetypeType.HERO, 0.7),
            ],
            ChakraType.CORAZON: [
                (ArchetypeType.LOVER, 0.9),
                (ArchetypeType.CAREGIVER, 0.8),
            ],
            ChakraType.GARGANTA: [
                (ArchetypeType.SAGE, 0.8),
                (ArchetypeType.JESTER, 0.7),
            ],
            ChakraType.TERCER_OJO: [
                (ArchetypeType.SAGE, 0.9),
                (ArchetypeType.MAGICIAN, 0.8),
            ],
            ChakraType.CORONA: [
                (ArchetypeType.MAGICIAN, 0.9),
                (ArchetypeType.INNOCENT, 0.6),
            ],
        }

    def add_hook(self, hook: Any) -> None:
        """Add a monitoring hook that receives the kernel state dict after each update."""
        self._hooks.append(hook)

    def update(
        self,
        delta_time: float,
        consciousness_coherence: float,
        external_influences: dict[str, float] | None = None,
        dominant_strategy: str = "balanced",
    ) -> dict[str, Any]:
        """Advance internal dynamics by delta_time and return a snapshot of kernel state."""
        external_influences = external_influences or {}

        # 1) Propagate energy dynamics
        try:
            self.chakras.update(delta_time, external_influences)
        except Exception:
            logger.exception("Chakra update failed")

        # 2) Collect chakra-derived influences for archetype evolution
        try:
            chakra_influences = {
                f"{ct.value}_influence": self.chakras.get_chakra_influence(
                    ct, "spiritual_connection"
                )
                for ct in ChakraType
                if self.chakras.get_chakra_influence(ct, "spiritual_connection") > 0.1
            }
        except Exception:
            chakra_influences = {}

        # 3) Evolve archetypes
        try:
            self.arquetipos.evolve(dominant_strategy, chakra_influences)
        except Exception:
            logger.exception("Archetype evolution failed")

        # 4) Integrate chakra <-> archetype coupling
        try:
            self._update_chakra_archetype_integration()
        except Exception:
            logger.exception("Chakra-archetype integration failed")

        # 5) Update development/enlightenment progress
        try:
            self._update_spiritual_development(consciousness_coherence, delta_time)
        except Exception:
            logger.exception("Spiritual development update failed")

        # 6) Let mudra system decay/update continuous effects
        try:
            self.mudras.update_active_mudras(delta_time, physics_engine=None)
        except Exception:
            # older mudra_system may expose update_active_mudras(delta_time)
            try:
                self.mudras.update_active_mudras(delta_time)
            except Exception:
                logger.exception("Mudra system update failed")

        # 7) Recompute integrity
        try:
            self._update_soul_integrity()
        except Exception:
            logger.exception("Soul integrity update failed")

        self.last_update_ts = time.time()
        state = self._generate_kernel_state()

        # 8) Hooks and EVA recording (best-effort, async-aware)
        for hook in list(self._hooks):
            try:
                hook(state)
            except Exception:
                logger.exception("SoulKernel hook failed")

        if self.eva_manager:
            try:
                rec = getattr(self.eva_manager, "record_experience", None)
                if rec:
                    experience_id = f"soul_update:{self.entity_id}:{int(time.time())}"
                    coro_or_res = rec(
                        entity_id=self.entity_id,
                        event_type="soul_state_update",
                        data=state,
                    )
                    if hasattr(coro_or_res, "__await__"):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(coro_or_res)
                            else:
                                loop.run_until_complete(coro_or_res)
                        except Exception:
                            logger.debug(
                                "Failed to schedule async EVA record", exc_info=True
                            )
            except Exception:
                logger.exception("Failed to record soul state to EVA")

        return state

    def _derive_dominant_strategy(self) -> str:
        """Derive behavioral strategy from archetypal modifiers (useful for planning)."""
        try:
            modifiers = self.arquetipos.get_behavioral_modifiers()
        except Exception:
            modifiers = {}
        if modifiers.get("aggression", 0.0) > 0.6:
            return "aggressive"
        if modifiers.get("creativity", 0.0) > 0.6:
            return "creative"
        if modifiers.get("compassion", 0.0) > 0.6:
            return "nurturing"
        if modifiers.get("caution", 0.0) > 0.6:
            return "cautious"
        return "balanced"

    def _update_chakra_archetype_integration(self) -> None:
        """Softly push archetype shifts driven by dominant chakras."""
        try:
            dominant_chakras = self.chakras.get_dominant_chakras(5)
        except Exception:
            dominant_chakras = []
        for chakra_type, influence_factor in dominant_chakras:
            affinities = self.chakra_archetype_affinities.get(chakra_type, [])
            for archetype_type, affinity in affinities:
                archetype_influence = influence_factor * affinity * 0.01
                try:
                    if hasattr(self.arquetipos, "shift_archetype"):
                        self.arquetipos.shift_archetype(
                            archetype_type,
                            archetype_influence,
                            f"chakra_{chakra_type.value}_influence",
                        )
                except Exception:
                    logger.debug(
                        "shift_archetype failed for %s", archetype_type, exc_info=True
                    )

    def get_qualia_influences(self) -> dict[str, float]:
        """Map internal spiritual state into qualia influence primitives."""
        influences = {
            "emotional_valence": 0.0,
            "arousal_level": 0.0,
            "cognitive_focus": 0.0,
            "complexity": 0.0,
            "spiritual_resonance": 0.0,
        }
        try:
            for chakra_type, chakra in getattr(self.chakras, "chakras", {}).items():
                influence_factor = getattr(
                    chakra, "get_influence_factor", lambda: 0.0
                )()
                if chakra_type == ChakraType.CORAZON:
                    influences["emotional_valence"] += influence_factor * 0.3
                elif chakra_type == ChakraType.PLEXO_SOLAR:
                    influences["arousal_level"] += influence_factor * 0.3
                elif chakra_type == ChakraType.TERCER_OJO:
                    influences["cognitive_focus"] += influence_factor * 0.3
                elif chakra_type == ChakraType.CORONA:
                    influences["spiritual_resonance"] += influence_factor * 0.4
        except Exception:
            logger.debug(
                "Failed to compute chakra-derived qualia influences", exc_info=True
            )

        try:
            behavioral_weights = self.arquetipos.get_behavioral_modifiers()
            if behavioral_weights.get("creativity", 0.0):
                influences["complexity"] += behavioral_weights["creativity"] * 0.2
            if behavioral_weights.get("aggression", 0.0):
                influences["arousal_level"] += behavioral_weights["aggression"] * 0.2
        except Exception:
            logger.debug(
                "Failed to factor archetypal behavioral weights", exc_info=True
            )

        # clamp results to sensible ranges
        for k in list(influences.keys()):
            influences[k] = float(max(-1.0, min(1.0, influences[k])))
        return influences

    def get_intentionality_biases(self) -> dict[str, float]:
        """Produce lightweight behavioral biases for higher-level planners."""
        biases = {
            "exploration_tendency": 0.0,
            "cooperation_tendency": 0.0,
            "dominance_tendency": 0.0,
            "creative_tendency": 0.0,
            "protective_tendency": 0.0,
            "transcendence_tendency": 0.0,
        }
        try:
            behavioral_weights = self.arquetipos.get_behavioral_modifiers()
            archetype_bias_mapping = {
                ArchetypeType.HERO: {
                    "exploration_tendency": 0.6,
                    "protective_tendency": 0.7,
                },
                ArchetypeType.CAREGIVER: {
                    "cooperation_tendency": 0.8,
                    "protective_tendency": 0.9,
                },
                ArchetypeType.RULER: {
                    "dominance_tendency": 0.9,
                    "cooperation_tendency": -0.3,
                },
                ArchetypeType.CREATOR: {
                    "creative_tendency": 0.9,
                    "exploration_tendency": 0.5,
                },
                ArchetypeType.SAGE: {
                    "transcendence_tendency": 0.7,
                    "cooperation_tendency": 0.4,
                },
                ArchetypeType.MAGICIAN: {
                    "transcendence_tendency": 0.9,
                    "creative_tendency": 0.6,
                },
            }
            for archetype_name, weight in behavioral_weights.items():
                try:
                    archetype_type = ArchetypeType(archetype_name)
                    mapping = archetype_bias_mapping.get(archetype_type, {})
                    for bias_type, bias_strength in mapping.items():
                        biases[bias_type] += weight * bias_strength
                except Exception:
                    continue
        except Exception:
            logger.debug("Failed to compute intentionality biases", exc_info=True)
        # clamp to 0..1
        for k in list(biases.keys()):
            biases[k] = float(max(0.0, min(1.0, biases[k])))
        return biases

    def apply_experience(self, qualia_state: QualiaState) -> None:
        """Apply a QualiaState as an experience that nudges the spiritual subsystems."""
        try:
            valence = float(getattr(qualia_state, "emotional_valence", 0.0))
            arousal = float(getattr(qualia_state, "arousal", 0.0))
            clarity = float(getattr(qualia_state, "cognitive_clarity", 0.5))
        except Exception:
            valence, arousal, clarity = 0.0, 0.0, 0.5

        try:
            if valence > 0.7:
                self._safe_increment_chakra(ChakraType.CORAZON, 0.01)
                self._safe_increment_chakra(ChakraType.CORONA, 0.01)
                self.arquetipos.evolve(
                    "nurturing", {"spiritual_connection": valence * clarity}
                )
            if arousal > 0.8:
                self._safe_increment_chakra(ChakraType.PLEXO_SOLAR, 0.02)
                self.arquetipos.evolve(
                    "aggressive", {"personal_power": arousal * clarity}
                )
            if valence < -0.5:
                self.arquetipos.evolve(
                    "transform", {"rebelde": abs(valence) * (1.0 - clarity)}
                )
        except Exception:
            logger.exception("apply_experience failed")

        self._record_spiritual_experience(
            f"Qualia applied: valence={valence:.2f}, arousal={arousal:.2f}, clarity={clarity:.2f}"
        )

    def _safe_increment_chakra(self, chakra_type: ChakraType, amount: float) -> None:
        try:
            chakra = self.chakras.chakras.get(chakra_type)
            if chakra:
                chakra.current_energy = min(
                    getattr(chakra, "max_capacity", 1.0), chakra.current_energy + amount
                )
        except Exception:
            logger.debug("Failed to increment chakra %s", chakra_type, exc_info=True)

    def _update_spiritual_development(
        self, consciousness_coherence: float, delta_time: float
    ) -> None:
        try:
            if consciousness_coherence > 0.7:
                development_gain = (consciousness_coherence - 0.7) * 0.001 * delta_time
                self.spiritual_development_level = min(
                    1.0, self.spiritual_development_level + development_gain
                )
            chakra_alignment = getattr(
                self.chakras, "get_overall_alignment", lambda: 0.0
            )()
            harmonic_bonus = (
                0.002
                if getattr(self.chakras, "harmonic_resonance_active", False)
                else 0.0
            )
            enlightenment_gain = (
                consciousness_coherence * 0.0005
                + chakra_alignment * 0.0003
                + harmonic_bonus
            ) * delta_time
            self.enlightenment_progress = min(
                1.0, self.enlightenment_progress + enlightenment_gain
            )
        except Exception:
            logger.exception("Failed to update development/enlightenment")

    def _update_soul_integrity(self) -> None:
        try:
            base_integrity = 1.0
            karma_penalty = max(0.0, -self.karma_balance * 0.2)
            mudra_strain = 0.0
            total = getattr(self.mudras, "total_mudra_activations", 0)
            if total > 0:
                success = getattr(self.mudras, "successful_mudra_activations", 0)
                success_rate = float(success) / max(1.0, float(total))
                if success_rate < 0.5 and self.spiritual_development_level < 0.5:
                    mudra_strain = 0.1
            alignment_bonus = (
                getattr(self.chakras, "get_overall_alignment", lambda: 0.0)() * 0.05
            )
            self.soul_integrity = max(
                0.1,
                min(
                    1.0, base_integrity - karma_penalty - mudra_strain + alignment_bonus
                ),
            )
        except Exception:
            logger.exception("Failed to recompute soul integrity")

    def _generate_kernel_state(self) -> dict[str, Any]:
        """Return a compact serializable snapshot of the kernel."""
        try:
            chakra_status = getattr(self.chakras, "get_system_status", lambda: {})()
        except Exception:
            chakra_status = {}
        try:
            archetype_influences = getattr(
                self.arquetipos, "get_archetype_influences", lambda: {}
            )()
            behavior_modifiers = getattr(
                self.arquetipos, "get_behavioral_modifiers", lambda: {}
            )()
            dominant_archetype = getattr(self.arquetipos, "dominant_archetype", None)
        except Exception:
            archetype_influences, behavior_modifiers, dominant_archetype = {}, {}, None

        return {
            "chakra_system": chakra_status,
            "archetypal_influences": archetype_influences,
            "behavioral_modifiers": behavior_modifiers,
            "dominant_archetype": dominant_archetype,
            "spiritual_development": self.spiritual_development_level,
            "enlightenment_progress": self.enlightenment_progress,
            "karma_balance": self.karma_balance,
            "soul_integrity": self.soul_integrity,
            "mudra_capabilities": (
                self.mudras.get_mudra_capabilities(self.chakras)
                if hasattr(self.mudras, "get_mudra_capabilities")
                else {}
            ),
            "harmonic_resonance_active": getattr(
                self.chakras, "harmonic_resonance_active", False
            ),
            "overall_spiritual_power": self._calculate_spiritual_power(),
            "last_update_ts": self.last_update_ts,
            "spiritual_experience_count": len(self.spiritual_experiences),
        }

    def _calculate_spiritual_power(self) -> float:
        try:
            chakra_power = getattr(self.chakras, "system_coherence", 0.5)
            spiritual_power = self.spiritual_development_level
            integrity_modifier = self.soul_integrity
            resonance_bonus = (
                0.2
                if getattr(self.chakras, "harmonic_resonance_active", False)
                else 0.0
            )
            total_power = (
                chakra_power * 0.4 + spiritual_power * 0.4 + resonance_bonus
            ) * integrity_modifier
            return min(1.0, float(total_power))
        except Exception:
            logger.exception("Failed to calculate spiritual power")
            return 0.0

    def activate_mudra(
        self, mudra_name: str, physics_engine: Any | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Public wrapper to activate a mudra; applies karma/experience bookkeeping."""
        if self.soul_integrity < 0.3:
            return {
                "success": False,
                "reason": "low_soul_integrity",
                "soul_integrity": self.soul_integrity,
            }
        try:
            result = self.mudras.activate_mudra(
                mudra_name, self.chakras, physics_engine, **kwargs
            )
        except Exception:
            logger.exception("Mudra activation failed")
            return {
                "success": False,
                "reason": "activation_error",
                "soul_integrity": self.soul_integrity,
            }

        try:
            if result.get("success"):
                self._update_karma_from_mudra_use(mudra_name, kwargs)
                if float(result.get("energy_consumed", 0.0)) > 0.2:
                    self._record_spiritual_experience(
                        f"Successful mudra activation: {mudra_name}"
                    )
        except Exception:
            logger.debug("Post-activation bookkeeping failed", exc_info=True)

        result.update(
            {
                "soul_integrity": self.soul_integrity,
                "spiritual_power": self._calculate_spiritual_power(),
                "karma_impact": self._calculate_karma_impact(mudra_name, kwargs),
            }
        )
        return result

    def _update_karma_from_mudra_use(
        self, mudra_name: str, kwargs: dict[str, Any]
    ) -> None:
        karma_change = 0.0
        try:
            if mudra_name == "telekinesis":
                target_mass = float(kwargs.get("target_mass", 1.0))
                if target_mass > 50:
                    karma_change = -0.001
            elif mudra_name == "blink":
                if getattr(self.mudras, "total_mudra_activations", 0) % 10 == 0:
                    karma_change = -0.0005
            spiritual_modifier = self.spiritual_development_level - 0.5
            karma_change *= 1.0 - spiritual_modifier
            self.karma_balance = max(-1.0, min(1.0, self.karma_balance + karma_change))
        except Exception:
            logger.debug("Karma update failed for mudra %s", mudra_name, exc_info=True)

    def _calculate_karma_impact(self, mudra_name: str, kwargs: dict[str, Any]) -> str:
        try:
            if mudra_name == "telekinesis":
                target_mass = float(kwargs.get("target_mass", 1.0))
                return "potentially_negative" if target_mass > 100 else "neutral"
            if mudra_name == "blink":
                distance = float(kwargs.get("distance", 10.0))
                return "minor_negative" if distance > 50 else "neutral"
        except Exception:
            logger.debug("Failed to calculate karma impact", exc_info=True)
        return "neutral"

    def _record_spiritual_experience(self, experience: str) -> None:
        self.spiritual_experiences.append(
            {
                "timestamp": time.time(),
                "experience": experience,
                "spiritual_level": self.spiritual_development_level,
                "chakra_alignment": getattr(
                    self.chakras, "get_overall_alignment", lambda: 0.0
                )(),
            }
        )
        if len(self.spiritual_experiences) > 50:
            self.spiritual_experiences = self.spiritual_experiences[-50:]

    # --- High-level helpers: serialization & persistence -----------------
    def to_serializable(self) -> dict[str, Any]:
        """Return a compact serializable snapshot excluding heavy objects (EVA excluded)."""
        return {
            "entity_id": self.entity_id,
            "last_update_ts": self.last_update_ts,
            "spiritual_development_level": float(self.spiritual_development_level),
            "enlightenment_progress": float(self.enlightenment_progress),
            "karma_balance": float(self.karma_balance),
            "soul_integrity": float(self.soul_integrity),
            "spiritual_experiences": list(self.spiritual_experiences),
            "chakras": getattr(self.chakras, "to_serializable", lambda: {})(),
            "arquetipos": getattr(self.arquetipos, "to_serializable", lambda: {})(),
            "mudras": getattr(self.mudras, "to_serializable", lambda: {})(),
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Load a previously produced snapshot (best-effort)."""
        try:
            self.entity_id = data.get("entity_id", self.entity_id)
            self.last_update_ts = data.get("last_update_ts", self.last_update_ts)
            self.spiritual_development_level = float(
                data.get(
                    "spiritual_development_level", self.spiritual_development_level
                )
            )
            self.enlightenment_progress = float(
                data.get("enlightenment_progress", self.enlightenment_progress)
            )
            self.karma_balance = float(data.get("karma_balance", self.karma_balance))
            self.soul_integrity = float(data.get("soul_integrity", self.soul_integrity))
            # Delegate to subsystems where available
            chak = data.get("chakras")
            if chak and hasattr(self.chakras, "load_serializable"):
                try:
                    self.chakras.load_serializable(chak)
                except Exception:
                    logger.debug("Failed to load chakra snapshot", exc_info=True)
            arqu = data.get("arquetipos")
            if arqu and hasattr(self.arquetipos, "load_serializable"):
                try:
                    self.arquetipos.load_serializable(arqu)
                except Exception:
                    logger.debug("Failed to load archetype snapshot", exc_info=True)
            mud = data.get("mudras")
            if mud and hasattr(self.mudras, "load_serializable"):
                try:
                    self.mudras.load_serializable(mud)
                except Exception:
                    logger.debug("Failed to load mudra snapshot", exc_info=True)
        except Exception:
            logger.exception("Failed to load SoulKernel serializable data")
