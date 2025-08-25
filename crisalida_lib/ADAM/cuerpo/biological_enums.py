"""
This file contains biological dataclasses that were previously mixed with enums.

This definitive version hardens numeric guards, adds serialization, light simulation tick,
batch event application, and best-effort EVA recording integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.enums import FisiologicalState, PhysiologicalEventType

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _clamp01(v: float) -> float:
    try:
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.0


@dataclass
class DetailedPhysiologicalState:
    """
    Estado fisiológico detallado que incluye niveles específicos de:
    energía, fatiga, estrés, hambre, dolor, hidratación, temperatura e inmunidad.

    - Valores normalizados 0..1.
    - Thread-safe modifications via internal lock.
    - Supports serialization and light-time tick simulation.
    """

    energy_level: float = 0.7
    fatigue_level: float = 0.3
    stress_level: float = 0.2
    hunger_level: float = 0.4
    pain_level: float = 0.1
    hydration_level: float = 0.8
    temperature_regulation: float = 0.9
    immune_response: float = 0.7
    sleep_debt: float = 0.2

    # internal lock for thread-safety and small local history
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _history_limit: int = field(default=200, init=False, repr=False)

    def __post_init__(self) -> None:
        with self._lock:
            self.energy_level = _clamp01(self.energy_level)
            self.fatigue_level = _clamp01(self.fatigue_level)
            self.stress_level = _clamp01(self.stress_level)
            self.hunger_level = _clamp01(self.hunger_level)
            self.pain_level = _clamp01(self.pain_level)
            self.hydration_level = _clamp01(self.hydration_level)
            self.temperature_regulation = _clamp01(self.temperature_regulation)
            self.immune_response = _clamp01(self.immune_response)
            self.sleep_debt = _clamp01(self.sleep_debt)

    # --- Health evaluation ----------------------------------------------
    def calculate_overall_health_score(self) -> float:
        """Calcula un puntaje general de salud basado en indicadores (0..1)."""
        with self._lock:
            try:
                positive_factors = (
                    self.energy_level
                    + self.hydration_level
                    + self.temperature_regulation
                    + self.immune_response
                    + (1.0 - self.sleep_debt)
                ) / 5.0

                negative_factors = (
                    self.fatigue_level
                    + self.stress_level
                    + self.hunger_level
                    + self.pain_level
                ) / 4.0

                score = positive_factors - negative_factors * 0.7
                return float(max(0.0, min(1.0, score)))
            except Exception:
                logger.exception("calculate_overall_health_score failed")
                return 0.0

    def get_primary_physiological_state(self) -> FisiologicalState:
        """Determina el estado fisiológico primario basado en los niveles detallados."""
        with self._lock:
            overall_health = self.calculate_overall_health_score()

            if self.pain_level > 0.8 or self.energy_level < 0.1:
                return FisiologicalState.CRITICAL
            if self.stress_level > 0.9 or self.fatigue_level > 0.9:
                return (
                    FisiologicalState.SHOCK
                    if self.stress_level > 0.95
                    else FisiologicalState.EXHAUSTED
                )
            if self.stress_level > 0.6 or self.fatigue_level > 0.7:
                return FisiologicalState.STRESSED
            if self.sleep_debt > 0.6 or self.energy_level < 0.4:
                return FisiologicalState.RECOVERING
            if overall_health > 0.65 and all(
                [
                    self.energy_level >= 0.8,
                    self.stress_level <= 0.2,
                    self.fatigue_level <= 0.2,
                    self.pain_level <= 0.15,
                ]
            ):
                return FisiologicalState.OPTIMAL
            if overall_health > 0.6:
                return FisiologicalState.HEALTHY

            return FisiologicalState.HOMEOSTASIS

    def get_feedback_modifiers(self) -> dict[str, float]:
        """
        Genera modificadores de retroalimentación para evaluadores emocionales.

        Returns keys: 'valence', 'arousal', 'dopamine' in bounded ranges.
        """
        with self._lock:
            energy_mood_impact = (self.energy_level - 0.5) * 0.3
            stress_mood_impact = -self.stress_level * 0.4
            pain_mood_impact = -self.pain_level * 0.5
            hunger_mood_impact = -max(0.0, self.hunger_level - 0.6) * 0.2

            valence_modifier = (
                energy_mood_impact
                + stress_mood_impact
                + pain_mood_impact
                + hunger_mood_impact
            )

            arousal_modifier = (
                self.stress_level * 0.3
                + self.pain_level * 0.4
                + (self.energy_level - 0.5) * 0.2
            )

            dopamine_modifier = (
                (self.energy_level - 0.5) * 0.3
                - self.fatigue_level * 0.2
                - self.pain_level * 0.3
            )

            return {
                "valence": float(max(-0.5, min(0.5, valence_modifier))),
                "arousal": float(max(-0.3, min(0.5, arousal_modifier))),
                "dopamine": float(max(-0.4, min(0.3, dopamine_modifier))),
            }

    # --- Event application / batch -------------------------------------
    def apply_physiological_event(
        self, event_type: PhysiologicalEventType, intensity: float = 1.0
    ) -> None:
        """Aplica un evento fisiológico que modifica los niveles. Intensity 0..1."""
        with self._lock:
            intensity = float(max(0.0, min(1.0, intensity)))

            try:
                if event_type == PhysiologicalEventType.EXERCISE:
                    self.energy_level = _clamp01(self.energy_level - 0.1 * intensity)
                    self.fatigue_level = _clamp01(self.fatigue_level + 0.15 * intensity)
                    self.stress_level = _clamp01(
                        max(0.0, self.stress_level - 0.05 * intensity)
                    )
                elif event_type == PhysiologicalEventType.REST:
                    self.energy_level = _clamp01(self.energy_level + 0.2 * intensity)
                    self.fatigue_level = _clamp01(
                        max(0.0, self.fatigue_level - 0.3 * intensity)
                    )
                    self.stress_level = _clamp01(
                        max(0.0, self.stress_level - 0.1 * intensity)
                    )
                elif event_type == PhysiologicalEventType.STRESS:
                    self.stress_level = _clamp01(self.stress_level + 0.2 * intensity)
                    self.energy_level = _clamp01(
                        max(0.0, self.energy_level - 0.1 * intensity)
                    )
                    self.fatigue_level = _clamp01(self.fatigue_level + 0.1 * intensity)
                elif event_type == PhysiologicalEventType.EAT:
                    self.hunger_level = _clamp01(
                        max(0.0, self.hunger_level - 0.4 * intensity)
                    )
                    self.energy_level = _clamp01(self.energy_level + 0.1 * intensity)
                elif event_type == PhysiologicalEventType.INJURY:
                    self.pain_level = _clamp01(self.pain_level + 0.3 * intensity)
                    self.stress_level = _clamp01(self.stress_level + 0.2 * intensity)
                    self.energy_level = _clamp01(
                        max(0.0, self.energy_level - 0.15 * intensity)
                    )
                elif event_type == PhysiologicalEventType.HYDRATE:
                    self.hydration_level = _clamp01(
                        self.hydration_level + 0.3 * intensity
                    )
                    self.energy_level = _clamp01(self.energy_level + 0.05 * intensity)
                elif event_type == PhysiologicalEventType.SLEEP:
                    self.sleep_debt = _clamp01(
                        max(0.0, self.sleep_debt - 0.4 * intensity)
                    )
                    self.fatigue_level = _clamp01(
                        max(0.0, self.fatigue_level - 0.5 * intensity)
                    )
                    self.energy_level = _clamp01(self.energy_level + 0.3 * intensity)
                else:
                    # Unknown event: no-op but log
                    logger.debug(
                        "apply_physiological_event: unknown event %s", event_type
                    )
            except Exception:
                logger.exception("apply_physiological_event failed for %s", event_type)

            self._record_history_entry(
                {
                    "event": getattr(event_type, "name", str(event_type)),
                    "intensity": intensity,
                    "timestamp": time.time(),
                }
            )

    def apply_event_batch(self, events: list[dict[str, Any]]) -> None:
        """Apply a list of {'event': PhysiologicalEventType, 'intensity': float} sequentially."""
        for ev in events:
            try:
                self.apply_physiological_event(
                    ev.get("event"), float(ev.get("intensity", 1.0))
                )
            except Exception:
                logger.debug("apply_event_batch: failed on ev=%s", ev, exc_info=True)

    # --- Light tick simulation ----------------------------------------
    def tick(self, dt_seconds: float = 1.0) -> None:
        """
        Lightweight metabolic tick. dt_seconds is elapsed seconds (can be coarse).
        This updates slow-changing variables: energy drain, fatigue accumulation,
        hydration loss and passive immune modulation.
        """
        with self._lock:
            try:
                # Scaled small rates so tick can be called per-second or per-minute.
                decay = min(1.0, float(dt_seconds) / 60.0)  # normalize to minutes
                # Energy drains with fatigue and sleep debt
                energy_drain = (
                    0.001 * decay
                    + 0.002 * self.fatigue_level * decay
                    + 0.0015 * self.sleep_debt * decay
                )
                self.energy_level = _clamp01(self.energy_level - energy_drain)
                # Fatigue increases slowly if energy low or sleep debt high
                self.fatigue_level = _clamp01(
                    self.fatigue_level
                    + max(
                        0.0,
                        0.001 * (1.0 - self.energy_level) + 0.0008 * self.sleep_debt,
                    )
                    * decay
                )
                # Hydration loss
                self.hydration_level = _clamp01(self.hydration_level - 0.0005 * decay)
                # Immune response slowly recovers if not stressed
                immune_delta = (
                    0.0008 * (1.0 - self.stress_level)
                    - 0.0005 * self.inflammatory_proxy()
                ) * decay
                self.immune_response = _clamp01(self.immune_response + immune_delta)
                # Small natural homeostatic correction for temperature regulation
                if self.temperature_regulation < 0.5:
                    self.temperature_regulation = _clamp01(
                        self.temperature_regulation + 0.0007 * decay
                    )
            except Exception:
                logger.exception("tick failed", exc_info=True)

    def inflammatory_proxy(self) -> float:
        """
        Lightweight proxy for inflammatory state derived from pain/stress; 0..1.
        Used internally for immune modulation.
        """
        with self._lock:
            return float(
                max(0.0, min(1.0, (self.pain_level * 0.6 + self.stress_level * 0.4)))
            )

    # --- History / recording -----------------------------------------
    def _record_history_entry(self, entry: dict[str, Any]) -> None:
        self._history.append(entry)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

    def to_serializable(self) -> dict[str, Any]:
        """Return a compact serializable snapshot of the physiological state."""
        with self._lock:
            payload = asdict(self)
            # remove non-serializable internal members
            payload.pop("_lock", None)
            payload.pop("_history_limit", None)
            return payload

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Best-effort restore from a serializable snapshot produced by to_serializable()."""
        if not isinstance(data, dict):
            return
        with self._lock:
            try:
                for k, v in data.items():
                    if hasattr(self, k) and k not in (
                        "_lock",
                        "_history",
                        "_history_limit",
                    ):
                        try:
                            setattr(
                                self,
                                k,
                                (
                                    _clamp01(float(v))
                                    if isinstance(v, (int, float))
                                    else v
                                ),
                            )
                        except Exception:
                            setattr(self, k, v)
            except Exception:
                logger.exception("load_serializable failed", exc_info=True)

    # --- EVA helpers (best-effort, sync/async-aware) ------------------
    def record_physiological_event(
        self,
        entity_id: str,
        eva_manager: EVAMemoryManager | None,
        event_type: Any,
        intensity: float = 1.0,
    ) -> str | None:
        """
        Best-effort persistence of a physiological event to EVA.
        Returns an experience id when available.
        """
        if not eva_manager:
            logger.debug("record_physiological_event: no eva_manager provided")
            return None
        try:
            recorder = getattr(eva_manager, "record_experience", None)
            if not callable(recorder):
                logger.debug("record_physiological_event: recorder not callable")
                return None
            payload = {
                "entity_id": entity_id,
                "event": getattr(event_type, "name", str(event_type)),
                "intensity": float(intensity),
                "phys_state": self.to_serializable(),
                "timestamp": time.time(),
            }
            res = recorder(
                entity_id=entity_id, event_type="physiological_event", data=payload
            )
            # schedule coroutine if necessary
            if hasattr(res, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(res)
                    else:
                        loop.run_until_complete(res)
                except Exception:
                    logger.debug("Could not schedule async EVA record", exc_info=True)
            exp_id = f"phys_event:{entity_id}:{int(time.time())}"
            return exp_id
        except Exception:
            logger.exception("record_physiological_event failed", exc_info=True)
            return None
