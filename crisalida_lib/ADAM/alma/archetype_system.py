"""
Archetype System - Professional / definitive implementation.

Goals:
- Hardened typing (typing module forms) and defensive imports.
- Restore and extend legacy evolution heuristics.
- Add utilities for normalization, blending, serialization, and EVA recording.
- Preserve public API: SistemaDeArquetipos and its methods.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any

# defensive numpy import (repo pattern)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.enums import ArchetypeType
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SistemaDeArquetipos:
    """Sistema avanzado de arquetipos que controla la 'personalidad' del agente.

    Diseño:
    - Mantiene un mapa de fuerzas por arquetipo (0.1..1.0).
    - Expone métodos para evolucionar, normalizar y mezclar arquetipos.
    - Registra eventos relevantes en EVA cuando esté disponible.
    """

    MIN_STRENGTH = 0.1
    MAX_STRENGTH = 1.0

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config = config or AdamConfig()
        self.eva_manager = eva_manager
        self.entity_id = entity_id

        # thread-safety for concurrent updates
        self._lock = RLock()

        # initialise archetype strengths (legacy-informed defaults)
        self.arquetipos: dict[ArchetypeType, float] = dict.fromkeys(ArchetypeType, 0.1)
        # tuned legacy defaults
        initial = {
            ArchetypeType.HERO: 0.7,
            ArchetypeType.SAGE: 0.6,
            ArchetypeType.EXPLORER: 0.5,
            ArchetypeType.CAREGIVER: 0.4,
            ArchetypeType.REBEL: 0.3,
            ArchetypeType.LOVER: 0.4,
            ArchetypeType.CREATOR: 0.5,
            ArchetypeType.INNOCENT: 0.3,
            ArchetypeType.RULER: 0.4,
            ArchetypeType.MAGICIAN: 0.2,
            ArchetypeType.SEEKER: 0.3,
            ArchetypeType.TRANSFORMER: 0.2,
            ArchetypeType.GUIDE: 0.3,
            ArchetypeType.DESTINY: 0.2,
        }
        self.arquetipos.update({k: float(v) for k, v in initial.items()})

        # synergies / antagonisms used by evolution heuristics
        self.archetype_synergies: dict[tuple[ArchetypeType, ArchetypeType], float] = {
            (ArchetypeType.HERO, ArchetypeType.RULER): 0.3,
            (ArchetypeType.SAGE, ArchetypeType.MAGICIAN): 0.4,
            (ArchetypeType.EXPLORER, ArchetypeType.REBEL): 0.2,
            (ArchetypeType.CAREGIVER, ArchetypeType.LOVER): 0.3,
            (ArchetypeType.CREATOR, ArchetypeType.MAGICIAN): 0.5,
            (ArchetypeType.SEEKER, ArchetypeType.SAGE): 0.2,
            (ArchetypeType.TRANSFORMER, ArchetypeType.REBEL): 0.3,
            (ArchetypeType.GUIDE, ArchetypeType.DESTINY): 0.4,
        }
        self.archetype_antagonisms: dict[tuple[ArchetypeType, ArchetypeType], float] = {
            (ArchetypeType.HERO, ArchetypeType.REBEL): 0.15,
            (ArchetypeType.RULER, ArchetypeType.REBEL): 0.2,
            (ArchetypeType.SAGE, ArchetypeType.INNOCENT): 0.1,
            (ArchetypeType.LOVER, ArchetypeType.TRANSFORMER): 0.1,
        }

        self.evolution_history: list[dict[str, Any]] = []
        self.last_narrative_event: dict[str, Any] = {}
        self.dominant_archetype: ArchetypeType = self._calculate_dominant_archetype()

        # clamp on init
        self._clamp_all()
        logger.info(
            "SistemaDeArquetipos initialized for '%s'. Dominant=%s",
            self.entity_id,
            self.dominant_archetype,
        )

    # --------------------
    # Core utilities
    # --------------------
    def _clamp_all(self) -> None:
        with self._lock:
            for k in list(self.arquetipos.keys()):
                self.arquetipos[k] = max(
                    self.MIN_STRENGTH, min(self.MAX_STRENGTH, float(self.arquetipos[k]))
                )

    def _calculate_dominant_archetype(self) -> ArchetypeType:
        with self._lock:
            if not self.arquetipos:
                return ArchetypeType.INNOCENT
            # prefer deterministic tie-break (sorted by enum name)
            items = sorted(
                self.arquetipos.items(), key=lambda x: (x[1], x[0].name), reverse=True
            )
            return items[0][0]

    def set_archetype_strength(self, archetype: ArchetypeType, value: float) -> None:
        with self._lock:
            self.arquetipos[archetype] = float(value)
            self._clamp_all()

    def get_archetype_strength(self, archetype: ArchetypeType) -> float:
        return float(self.arquetipos.get(archetype, self.MIN_STRENGTH))

    # --------------------
    # Evolution engine
    # --------------------
    def evolve(
        self,
        dominant_strategy: str,
        chakra_influences: dict[str, float] | None = None,
        narrative_event: dict[str, Any] | None = None,
        evolution_amount: float = 0.02,
    ) -> None:
        """Evolve archetypes according to strategy, chakra signals and narrative events.

        Keeps backward-compatible heuristics from legacy implementations but:
        - normalizes strengths,
        - applies synergies/antagonisms,
        - records evolution events to EVA when available.
        """
        with self._lock:
            chakra_influences = chakra_influences or {}
            previous_dominant = self.dominant_archetype

            s = dominant_strategy.lower() if dominant_strategy else ""
            # strategy-driven updates (legacy-informed)
            if "creative" in s:
                self.arquetipos[ArchetypeType.CREATOR] += evolution_amount
                self.arquetipos[ArchetypeType.EXPLORER] += evolution_amount * 0.7
            elif "cautious" in s or "caution" in s:
                self.arquetipos[ArchetypeType.SAGE] += evolution_amount
                self.arquetipos[ArchetypeType.CAREGIVER] += evolution_amount * 0.5
            elif "aggressive" in s:
                self.arquetipos[ArchetypeType.HERO] += evolution_amount
                self.arquetipos[ArchetypeType.REBEL] += evolution_amount * 0.6
            elif "nurtur" in s:
                self.arquetipos[ArchetypeType.CAREGIVER] += evolution_amount
                self.arquetipos[ArchetypeType.LOVER] += evolution_amount * 0.4
            elif "transform" in s or "evolve" in s:
                self.arquetipos[ArchetypeType.TRANSFORMER] += evolution_amount
                self.arquetipos[ArchetypeType.REBEL] += evolution_amount * 0.3
            elif "guide" in s or "mentor" in s:
                self.arquetipos[ArchetypeType.GUIDE] += evolution_amount
                self.arquetipos[ArchetypeType.SAGE] += evolution_amount * 0.3

            # chakra influences
            for chakra, strength in chakra_influences.items():
                try:
                    if chakra == "spiritual_connection" and strength > 0.7:
                        self.arquetipos[ArchetypeType.MAGICIAN] += (
                            evolution_amount * 0.5
                        )
                    elif chakra == "personal_power" and strength > 0.8:
                        self.arquetipos[ArchetypeType.RULER] += evolution_amount * 0.3
                    elif chakra == "creativity" and strength > 0.7:
                        self.arquetipos[ArchetypeType.CREATOR] += evolution_amount * 0.4
                    elif chakra == "innocence" and strength > 0.6:
                        self.arquetipos[ArchetypeType.INNOCENT] += (
                            evolution_amount * 0.3
                        )
                    elif chakra == "destiny_alignment" and strength > 0.7:
                        self.arquetipos[ArchetypeType.DESTINY] += evolution_amount * 0.4
                except Exception:
                    logger.debug(
                        "Ignored chakra influence %s due to error",
                        chakra,
                        exc_info=True,
                    )

            # narrative event handling (flexible)
            if narrative_event:
                archetype_str = narrative_event.get("archetype")
                impact = float(narrative_event.get("evolutionary_impact", 0.0))
                try:
                    if archetype_str:
                        archetype = (
                            ArchetypeType(archetype_str)
                            if isinstance(archetype_str, str)
                            else archetype_str
                        )
                        if archetype in self.arquetipos:
                            self.arquetipos[archetype] += impact * 0.05
                except Exception:
                    logger.warning(
                        "Invalid archetype in narrative_event: %s", archetype_str
                    )

                dramatic_function = narrative_event.get("dramatic_function")
                if dramatic_function == "transformación":
                    self.arquetipos[ArchetypeType.TRANSFORMER] += impact * 0.03
                elif dramatic_function == "guía":
                    self.arquetipos[ArchetypeType.GUIDE] += impact * 0.03
                elif dramatic_function == "destino":
                    self.arquetipos[ArchetypeType.DESTINY] += impact * 0.03

                self.last_narrative_event = dict(narrative_event)

            # apply synergies/antagonisms and decay
            self._apply_archetype_synergies()
            self._apply_archetype_antagonisms()
            self._apply_natural_decay(evolution_amount * 0.3)
            self._clamp_all()
            # optional normalization to keep distributions meaningful
            self._normalize_total_strengths(target_total=3.0)

            new_dominant = self._calculate_dominant_archetype()
            self.dominant_archetype = new_dominant

            if new_dominant != previous_dominant:
                record = {
                    "timestamp": time.time(),
                    "previous_dominant": previous_dominant.value,
                    "new_dominant": new_dominant.value,
                    "trigger_strategy": dominant_strategy,
                    "narrative_event": narrative_event,
                }
                self.evolution_history.append(record)
                # keep history bounded
                if len(self.evolution_history) > 30:
                    self.evolution_history = self.evolution_history[-30:]
                if self.eva_manager:
                    try:
                        self.eva_manager.record_experience(
                            entity_id=self.entity_id,
                            event_type="archetype_evolution",
                            data=record,
                        )
                    except Exception:
                        logger.debug(
                            "Failed to record archetype evolution in EVA", exc_info=True
                        )

    # --------------------
    # Helpers: synergies, antagonisms, decay
    # --------------------
    def _apply_archetype_synergies(self) -> None:
        with self._lock:
            for (a1, a2), strength in self.archetype_synergies.items():
                if a1 in self.arquetipos and a2 in self.arquetipos:
                    base = min(self.arquetipos[a1], self.arquetipos[a2])
                    boost = base * strength * 0.01
                    self.arquetipos[a1] = min(
                        self.MAX_STRENGTH, self.arquetipos[a1] + boost
                    )
                    self.arquetipos[a2] = min(
                        self.MAX_STRENGTH, self.arquetipos[a2] + boost
                    )

    def _apply_archetype_antagonisms(self) -> None:
        with self._lock:
            for (a1, a2), strength in self.archetype_antagonisms.items():
                if a1 in self.arquetipos and a2 in self.arquetipos:
                    base = min(self.arquetipos[a1], self.arquetipos[a2])
                    reduction = base * strength * 0.01
                    self.arquetipos[a1] = max(
                        self.MIN_STRENGTH, self.arquetipos[a1] - reduction
                    )
                    self.arquetipos[a2] = max(
                        self.MIN_STRENGTH, self.arquetipos[a2] - reduction
                    )

    def _apply_natural_decay(self, decay_amount: float) -> None:
        with self._lock:
            for archetype in list(self.arquetipos.keys()):
                actual_decay = (
                    decay_amount * 0.2
                    if archetype == self.dominant_archetype
                    else decay_amount
                )
                self.arquetipos[archetype] = max(
                    self.MIN_STRENGTH, self.arquetipos[archetype] - actual_decay
                )

    def _normalize_total_strengths(self, target_total: float = 3.0) -> None:
        """Scale archetype strengths so their sum is approximately target_total while preserving ratios.
        Guardado: target_total default chosen to keep overall expressivity without runaway.
        """
        with self._lock:
            total = sum(self.arquetipos.values()) or 1.0
            if total == 0:
                return
            factor = float(target_total) / float(total)
            for k in self.arquetipos:
                self.arquetipos[k] = max(
                    self.MIN_STRENGTH,
                    min(self.MAX_STRENGTH, self.arquetipos[k] * factor),
                )

    # --------------------
    # Public getters/serializers
    # --------------------
    def get_archetype_influences(self) -> dict[str, dict[str, float]]:
        influences: dict[str, dict[str, float]] = {}
        with self._lock:
            for archetype, strength in self.arquetipos.items():
                if archetype == ArchetypeType.HERO:
                    influences[archetype.value] = {
                        "courage": strength * 0.9,
                        "leadership": strength * 0.7,
                        "risk_taking": strength * 0.6,
                        "protective_instinct": strength * 0.8,
                        "narrative_weight": strength * 0.7,
                    }
                elif archetype == ArchetypeType.SAGE:
                    influences[archetype.value] = {
                        "wisdom": strength * 0.95,
                        "patience": strength * 0.8,
                        "analytical_thinking": strength * 0.85,
                        "teaching_ability": strength * 0.7,
                        "narrative_weight": strength * 0.6,
                    }
                elif archetype == ArchetypeType.EXPLORER:
                    influences[archetype.value] = {
                        "curiosity": strength * 0.9,
                        "adaptability": strength * 0.8,
                        "innovation": strength * 0.75,
                        "independence": strength * 0.7,
                        "narrative_weight": strength * 0.5,
                    }
                elif archetype == ArchetypeType.MAGICIAN:
                    influences[archetype.value] = {
                        "spiritual_power": strength * 0.95,
                        "reality_manipulation": strength * 0.8,
                        "intuitive_abilities": strength * 0.85,
                        "transformation_catalyst": strength * 0.9,
                        "narrative_weight": strength * 0.8,
                    }
                elif archetype == ArchetypeType.TRANSFORMER:
                    influences[archetype.value] = {
                        "change_agent": strength * 0.9,
                        "adaptability": strength * 0.7,
                        "narrative_weight": strength * 0.7,
                    }
                elif archetype == ArchetypeType.GUIDE:
                    influences[archetype.value] = {
                        "guidance": strength * 0.95,
                        "wisdom": strength * 0.7,
                        "narrative_weight": strength * 0.8,
                    }
                elif archetype == ArchetypeType.DESTINY:
                    influences[archetype.value] = {
                        "purpose_alignment": strength * 0.95,
                        "narrative_weight": strength * 0.9,
                    }
        return influences

    def get_behavioral_modifiers(self) -> dict[str, float]:
        modifiers: dict[str, float] = {
            "aggression": 0.0,
            "compassion": 0.0,
            "creativity": 0.0,
            "caution": 0.0,
            "sociability": 0.0,
            "spiritual_openness": 0.0,
            "narrative_weight": 0.0,
            "purpose_alignment": 0.0,
        }

        contributions: dict[ArchetypeType, dict[str, float]] = {
            ArchetypeType.HERO: {
                "aggression": 0.3,
                "compassion": 0.1,
                "narrative_weight": 0.7,
            },
            ArchetypeType.SAGE: {
                "caution": 0.4,
                "spiritual_openness": 0.3,
                "narrative_weight": 0.6,
            },
            ArchetypeType.EXPLORER: {
                "creativity": 0.4,
                "caution": -0.2,
                "narrative_weight": 0.5,
            },
            ArchetypeType.CAREGIVER: {"compassion": 0.5, "sociability": 0.3},
            ArchetypeType.REBEL: {"aggression": 0.2, "creativity": 0.2},
            ArchetypeType.LOVER: {"compassion": 0.3, "sociability": 0.4},
            ArchetypeType.CREATOR: {"creativity": 0.6, "spiritual_openness": 0.2},
            ArchetypeType.MAGICIAN: {
                "spiritual_openness": 0.8,
                "creativity": 0.3,
                "narrative_weight": 0.8,
            },
            ArchetypeType.TRANSFORMER: {"creativity": 0.3, "narrative_weight": 0.7},
            ArchetypeType.GUIDE: {"spiritual_openness": 0.4, "narrative_weight": 0.8},
            ArchetypeType.DESTINY: {"purpose_alignment": 0.9, "narrative_weight": 0.9},
        }

        with self._lock:
            for archetype, strength in self.arquetipos.items():
                contrib = contributions.get(archetype)
                if contrib:
                    for modifier, c in contrib.items():
                        modifiers[modifier] += strength * c

            # Clamp modifiers to sensible range [-1, 1]
            for k in list(modifiers.keys()):
                modifiers[k] = max(-1.0, min(1.0, modifiers[k]))

        return modifiers

    def get_narrative_context(self) -> dict[str, Any]:
        with self._lock:
            return {
                "dominant_archetype": self.dominant_archetype.value,
                "archetype_strengths": {k.value: v for k, v in self.arquetipos.items()},
                "last_narrative_event": dict(self.last_narrative_event),
                "evolution_history": list(self.evolution_history),
                "behavioral_modifiers": self.get_behavioral_modifiers(),
            }

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "arquetipos": {k.value: v for k, v in self.arquetipos.items()},
                "dominant_archetype": self.dominant_archetype.value,
                "evolution_history_count": len(self.evolution_history),
            }

    # --------------------
    # Advanced utilities (anticipated evolutions)
    # --------------------
    def blend_with(self, other: SistemaDeArquetipos, weight: float = 0.5) -> None:
        """Blend another system into this one (useful for merging learned patterns)."""
        with self._lock:
            weight = max(0.0, min(1.0, float(weight)))
            for k in self.arquetipos:
                self.arquetipos[k] = max(
                    self.MIN_STRENGTH,
                    min(
                        self.MAX_STRENGTH,
                        (
                            self.arquetipos[k] * (1.0 - weight)
                            + other.arquetipos.get(k, self.MIN_STRENGTH) * weight
                        ),
                    ),
                )
            self._clamp_all()
            self._normalize_total_strengths()

    def simulate_evolution_tick(
        self, external_signals: dict[str, Any] | None = None
    ) -> None:
        """Convenience method to run one tick of autonomous evolution using config heuristics."""
        external_signals = external_signals or {}
        # infer a dominant_strategy from signals if present
        strategy = external_signals.get("dominant_strategy", "background")
        chakra = external_signals.get("chakra_influences", {})
        narrative = external_signals.get("narrative_event", None)
        self.evolve(
            strategy,
            chakra_influences=chakra,
            narrative_event=narrative,
            evolution_amount=float(
                self.config.get("archetype_base_evolution", 0.02)
                if hasattr(self.config, "get")
                else 0.02
            ),
        )

    def to_serializable(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        with self._lock:
            return {
                "entity_id": self.entity_id,
                "arquetipos": {k.value: float(v) for k, v in self.arquetipos.items()},
                "dominant_archetype": self.dominant_archetype.value,
                "last_narrative_event": dict(self.last_narrative_event),
                "evolution_history": list(self.evolution_history),
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Load state previously generated by to_serializable."""
        with self._lock:
            try:
                for k, v in (data.get("arquetipos") or {}).items():
                    try:
                        at = ArchetypeType(k)
                        self.arquetipos[at] = float(v)
                    except Exception:
                        logger.debug(
                            "Ignoring unknown archetype key during load: %s", k
                        )
                dom = data.get("dominant_archetype")
                if dom:
                    try:
                        self.dominant_archetype = ArchetypeType(dom)
                    except Exception:
                        logger.debug("Invalid dominant archetype on load: %s", dom)
                self.last_narrative_event = dict(data.get("last_narrative_event") or {})
                self.evolution_history = list(data.get("evolution_history") or [])
                self._clamp_all()
            except Exception:
                logger.exception("Failed to load archetype state")
