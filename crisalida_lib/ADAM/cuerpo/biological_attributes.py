"""
Biological Attributes
=====================

This module defines PhysicalAttributes and ResourceManagement classes,
representing the physical and resource metrics of the entity's biological body.

Refactored to be decoupled from EVA, using EVAMemoryManager for persistence.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager

logger = logging.getLogger(__name__)


def _clamp01(v: float) -> float:
    try:
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.0


@dataclass
class PhysicalAttributes:
    """
    PhysicalAttributes - Metrics and simple physiology model.

    - Defensive: values are clamped to sensible ranges.
    - Tick/update methods allow continuous simulation.
    - Best-effort EVA recording supports sync and async backends.
    """

    # Vital signs (human-like defaults)
    heart_rate: float = field(default=70.0)  # BPM
    blood_pressure_systolic: float = field(default=120.0)  # mmHg
    blood_pressure_diastolic: float = field(default=80.0)  # mmHg
    body_temperature: float = field(default=37.0)  # °C
    respiratory_rate: float = field(default=16.0)  # breaths / min

    # Energetic resources (0..1 normalized)
    glucose_level: float = field(default=0.7)
    atp_reserves: float = field(default=0.8)
    oxygen_saturation: float = field(default=0.98)
    hydration_level: float = field(default=0.75)

    # Physical / systemic state (0..1 normalized)
    muscle_tension: float = field(default=0.3)
    neural_conductivity: float = field(default=0.85)
    inflammatory_response: float = field(default=0.1)
    immune_system_strength: float = field(default=0.8)

    # Circadian and recovery
    circadian_phase: float = field(default=0.5)  # 0..1 (0=midnight,0.5=noon)
    sleep_debt: float = field(default=0.2)  # 0..1
    recovery_rate: float = field(default=0.6)  # 0..1 baseline

    # internal lock for thread-safety
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        # normalize ranges
        with self._lock:
            self.oxygen_saturation = _clamp01(self.oxygen_saturation)
            self.glucose_level = _clamp01(self.glucose_level)
            self.atp_reserves = _clamp01(self.atp_reserves)
            self.hydration_level = _clamp01(self.hydration_level)
            self.muscle_tension = _clamp01(self.muscle_tension)
            self.neural_conductivity = _clamp01(self.neural_conductivity)
            self.inflammatory_response = _clamp01(self.inflammatory_response)
            self.immune_system_strength = _clamp01(self.immune_system_strength)
            self.circadian_phase = _clamp01(self.circadian_phase)
            self.sleep_debt = _clamp01(self.sleep_debt)
            self.recovery_rate = _clamp01(self.recovery_rate)

    # --- Accessors / snapshots ---------------------------------------
    def get_vital_signs(self) -> dict[str, float]:
        with self._lock:
            return {
                "heart_rate": float(self.heart_rate),
                "blood_pressure_systolic": float(self.blood_pressure_systolic),
                "blood_pressure_diastolic": float(self.blood_pressure_diastolic),
                "body_temperature": float(self.body_temperature),
                "respiratory_rate": float(self.respiratory_rate),
            }

    def get_energy_status(self) -> dict[str, float]:
        with self._lock:
            return {
                "glucose_level": float(self.glucose_level),
                "atp_reserves": float(self.atp_reserves),
                "oxygen_saturation": float(self.oxygen_saturation),
                "hydration_level": float(self.hydration_level),
            }

    def get_physical_state(self) -> dict[str, float]:
        with self._lock:
            return {
                "muscle_tension": float(self.muscle_tension),
                "neural_conductivity": float(self.neural_conductivity),
                "inflammatory_response": float(self.inflammatory_response),
                "immune_system_strength": float(self.immune_system_strength),
                "circadian_phase": float(self.circadian_phase),
                "sleep_debt": float(self.sleep_debt),
                "recovery_rate": float(self.recovery_rate),
            }

    def get_overall_health_index(self) -> float:
        with self._lock:
            try:
                vital = (
                    (70.0 / max(1.0, abs(self.heart_rate)))  # normalized inverse for HR
                    + (37.0 / max(32.0, abs(self.body_temperature)))  # temp closeness
                    + self.oxygen_saturation
                ) / 3.0
                energy = (
                    self.glucose_level + self.atp_reserves + self.hydration_level
                ) / 3.0
                immune = self.immune_system_strength
                recovery = 1.0 - self.sleep_debt
                score = min(
                    1.0, (vital * 0.3 + energy * 0.3 + immune * 0.2 + recovery * 0.2)
                )
                return float(score)
            except Exception:
                logger.exception("get_overall_health_index failed")
                return 0.0

    # --- Simulation / dynamics --------------------------------------
    def tick(self, dt: float = 1.0, activity_load: float = 0.0) -> None:
        """
        Advance a lightweight physiology model by dt (seconds or ticks).
        activity_load: 0..1 multiplier of exertion (higher => faster depletion).
        """
        with self._lock:
            try:
                # energy depletion proportional to activity and muscle_tension
                depletion = (
                    0.01 + activity_load * 0.05 + self.muscle_tension * 0.02
                ) * dt
                self.glucose_level = _clamp01(self.glucose_level - depletion * 0.7)
                self.atp_reserves = _clamp01(self.atp_reserves - depletion)
                # hydration slowly decreases
                self.hydration_level = _clamp01(self.hydration_level - 0.002 * dt)
                # oxygen depends on respiratory rate and activity
                o2_delta = -0.005 * activity_load * dt
                self.oxygen_saturation = _clamp01(self.oxygen_saturation + o2_delta)
                # inflammatory response increases with exertion
                self.inflammatory_response = _clamp01(
                    self.inflammatory_response + 0.001 * activity_load * dt
                )
                # sleep debt increases if activity persists without recovery
                self.sleep_debt = _clamp01(
                    self.sleep_debt + 0.0005 * activity_load * dt
                )
                # small passive recovery via baseline recovery_rate
                self._passive_recovery(dt)
            except Exception:
                logger.exception("PhysicalAttributes.tick failed", exc_info=True)

    def _passive_recovery(self, dt: float) -> None:
        # recovery improves ATP, glucose and reduces inflammatory response slightly
        try:
            recover_factor = self.recovery_rate * 0.01 * dt
            self.atp_reserves = _clamp01(self.atp_reserves + recover_factor)
            self.glucose_level = _clamp01(self.glucose_level + recover_factor * 0.5)
            self.inflammatory_response = _clamp01(
                self.inflammatory_response - recover_factor * 0.2
            )
            # if well-rested, slightly improve neural conductivity
            if self.sleep_debt < 0.3:
                self.neural_conductivity = _clamp01(
                    self.neural_conductivity + recover_factor * 0.1
                )
        except Exception:
            logger.exception("_passive_recovery failed", exc_info=True)

    def apply_sleep(self, hours: float) -> None:
        """Apply sleep to reduce sleep debt, recover energy and improve coherence."""
        with self._lock:
            try:
                reduction = min(self.sleep_debt, hours * 0.05)
                self.sleep_debt = _clamp01(self.sleep_debt - reduction)
                # stronger recovery for ATP and neural conductivity
                self.atp_reserves = _clamp01(self.atp_reserves + hours * 0.02)
                self.neural_conductivity = _clamp01(
                    self.neural_conductivity + hours * 0.005
                )
                self.recovery_rate = _clamp01(self.recovery_rate + hours * 0.002)
            except Exception:
                logger.exception("apply_sleep failed", exc_info=True)

    def exertion_event(self, intensity: float, duration: float) -> None:
        """Apply a burst of exertion that temporarily raises heart rate and consumes resources."""
        with self._lock:
            try:
                intensity = max(0.0, min(1.0, float(intensity)))
                # instantaneous effects
                self.heart_rate += 20.0 * intensity
                self.respiratory_rate += 5.0 * intensity
                self.muscle_tension = _clamp01(self.muscle_tension + 0.2 * intensity)
                # resource consumption scaled by duration
                delta = duration * (0.01 + intensity * 0.05)
                self.glucose_level = _clamp01(self.glucose_level - delta * 0.8)
                self.atp_reserves = _clamp01(self.atp_reserves - delta)
            except Exception:
                logger.exception("exertion_event failed", exc_info=True)

    # --- EVA recording (best-effort, sync/async-aware) --------------
    def record_physical_event(
        self,
        entity_id: str,
        eva_manager: EVAMemoryManager | None,
        event_type: str,
        data: Any,
    ) -> str | None:
        """
        Record a physical event using the EVAMemoryManager.

        Returns experience id when available or None.
        """
        if not eva_manager:
            logger.debug("No EVA manager provided to record_physical_event")
            return None
        try:
            rec = getattr(eva_manager, "record_experience", None)
            if not callable(rec):
                logger.debug("EVAMemoryManager.record_experience is not callable")
                return None
            payload = {
                "entity_id": entity_id,
                "event_type": event_type,
                "data": data,
                "vitals": self.get_vital_signs(),
                "energy": self.get_energy_status(),
                "timestamp": time.time(),
            }
            result = rec(entity_id=entity_id, event_type=event_type, data=payload)
            # schedule if coroutine
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(result)
                    else:
                        loop.run_until_complete(result)
                except Exception:
                    logger.debug(
                        "Async scheduling failed for record_physical_event",
                        exc_info=True,
                    )
            # best-effort experience id
            exp_id = f"physical:{entity_id}:{event_type}:{int(time.time())}"
            return exp_id
        except Exception:
            logger.exception("record_physical_event failed")
            return None

    # --- Serialization helpers --------------------------------------
    def to_serializable(self) -> dict[str, Any]:
        with self._lock:
            return {
                "vitals": self.get_vital_signs(),
                "energy": self.get_energy_status(),
                "state": self.get_physical_state(),
                "sleep_debt": float(self.sleep_debt),
                "circadian_phase": float(self.circadian_phase),
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        if not data:
            return
        with self._lock:
            try:
                vitals = data.get("vitals", {})
                energy = data.get("energy", {})
                state = data.get("state", {})
                for k, v in vitals.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
                for k, v in energy.items():
                    if hasattr(self, k):
                        setattr(self, k, _clamp01(v))
                for k, v in state.items():
                    if hasattr(self, k):
                        setattr(self, k, _clamp01(v))
                self.sleep_debt = _clamp01(data.get("sleep_debt", self.sleep_debt))
                self.circadian_phase = _clamp01(
                    data.get("circadian_phase", self.circadian_phase)
                )
            except Exception:
                logger.exception("load_serializable failed", exc_info=True)


@dataclass
class ResourceManagement:
    """
    ResourceManagement - Advanced management of named biological energy pools.

    - Pools, consumption & recovery rates are configurable.
    - Provides helpers for allocation, emergency restoration, and serialization.
    """

    energy_pools: dict[str, float] = field(
        default_factory=lambda: {
            "cellular": 0.8,
            "neural": 0.7,
            "muscular": 0.75,
            "cognitive": 0.6,
            "metabolic": 0.8,
        }
    )

    consumption_rates: dict[str, float] = field(
        default_factory=lambda: {
            "rest": 0.02,
            "thinking": 0.05,
            "stress": 0.08,
            "physical": 0.10,
            "emergency": 0.15,
        }
    )

    recovery_rates: dict[str, float] = field(
        default_factory=lambda: {
            "cellular": 0.03,
            "neural": 0.025,
            "muscular": 0.035,
            "cognitive": 0.02,
            "metabolic": 0.04,
        }
    )

    critical_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "energy_depletion": 0.2,
            "stress_overload": 0.8,
            "recovery_needed": 0.3,
            "optimal_performance": 0.75,
        }
    )

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        with self._lock:
            for k in list(self.energy_pools.keys()):
                self.energy_pools[k] = _clamp01(self.energy_pools[k])

    def consume_energy(self, activity: str, dt: float = 1.0) -> float:
        """
        Consume energy from pools based on activity. Returns total consumed.
        """
        with self._lock:
            try:
                rate = (
                    self.consumption_rates.get(
                        activity, self.consumption_rates.get("thinking", 0.05)
                    )
                    * dt
                )
                total_consumed = 0.0
                # heuristic: consume proportionally from all pools weighted by pool importance
                weights = {p: max(0.01, v) for p, v in self.energy_pools.items()}
                total_weight = sum(weights.values()) or 1.0
                for pool, val in list(self.energy_pools.items()):
                    share = (weights[pool] / total_weight) * rate
                    consumed = min(val, share)
                    self.energy_pools[pool] = _clamp01(val - consumed)
                    total_consumed += consumed
                return float(total_consumed)
            except Exception:
                logger.exception("consume_energy failed")
                return 0.0

    def recover_energy(self, dt: float = 1.0) -> None:
        with self._lock:
            try:
                for pool, rec_rate in self.recovery_rates.items():
                    self.energy_pools[pool] = _clamp01(
                        self.energy_pools.get(pool, 0.0) + rec_rate * dt
                    )
            except Exception:
                logger.exception("recover_energy failed")

    def is_critical(self) -> bool:
        with self._lock:
            return any(
                v < self.critical_thresholds["energy_depletion"]
                for v in self.energy_pools.values()
            )

    def get_resource_status(self) -> dict[str, float]:
        with self._lock:
            return copy.deepcopy(self.energy_pools)

    def get_performance_index(self) -> float:
        with self._lock:
            try:
                avg_energy = sum(self.energy_pools.values()) / max(
                    1, len(self.energy_pools)
                )
                optimal = max(
                    1e-6,
                    float(self.critical_thresholds.get("optimal_performance", 0.75)),
                )
                return float(min(1.0, avg_energy / optimal))
            except Exception:
                logger.exception("get_performance_index failed")
                return 0.0

    def allocate_energy(self, target_pool: str, amount: float) -> float:
        """Try to move a small amount from other pools into target_pool. Returns actually allocated."""
        with self._lock:
            try:
                amount = float(max(0.0, amount))
                if target_pool not in self.energy_pools:
                    return 0.0
                deficit = max(0.0, amount - (1.0 - self.energy_pools[target_pool]))
                if deficit <= 0.0:
                    # there's room; just increase
                    self.energy_pools[target_pool] = _clamp01(
                        self.energy_pools[target_pool] + amount
                    )
                    return amount
                # steal proportionally from other pools
                donors = [
                    p
                    for p in self.energy_pools.keys()
                    if p != target_pool and self.energy_pools[p] > 0.05
                ]
                if not donors:
                    return 0.0
                per_donor = deficit / len(donors)
                actual_moved = 0.0
                for d in donors:
                    move = min(per_donor, self.energy_pools[d] - 0.01)
                    self.energy_pools[d] = _clamp01(self.energy_pools[d] - move)
                    actual_moved += move
                self.energy_pools[target_pool] = _clamp01(
                    self.energy_pools[target_pool] + amount
                )
                return float(min(amount, amount + actual_moved))
            except Exception:
                logger.exception("allocate_energy failed")
                return 0.0

    def emergency_restore(self, factor: float = 0.2) -> None:
        """Emergency restore: small boost to all pools (bounded)."""
        with self._lock:
            try:
                factor = max(0.0, min(1.0, factor))
                for p in list(self.energy_pools.keys()):
                    self.energy_pools[p] = _clamp01(
                        self.energy_pools[p] + factor * (1.0 - self.energy_pools[p])
                    )
            except Exception:
                logger.exception("emergency_restore failed")

    # --- EVA helpers (best-effort) ---------------------------------
    def record_resource_event(
        self,
        entity_id: str,
        eva_manager: EVAMemoryManager | None,
        event_type: str,
        data: Any,
    ) -> str | None:
        if not eva_manager:
            logger.debug("No EVA manager provided to record_resource_event")
            return None
        try:
            rec = getattr(eva_manager, "record_experience", None)
            if not callable(rec):
                logger.debug("EVAMemoryManager.record_experience is not callable")
                return None
            payload = {
                "entity_id": entity_id,
                "event_type": event_type,
                "data": data,
                "resource_status": self.get_resource_status(),
                "timestamp": time.time(),
            }
            result = rec(entity_id=entity_id, event_type=event_type, data=payload)
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(result)
                    else:
                        loop.run_until_complete(result)
                except Exception:
                    logger.debug(
                        "Async scheduling failed for record_resource_event",
                        exc_info=True,
                    )
            exp_id = f"resource:{entity_id}:{event_type}:{int(time.time())}"
            return exp_id
        except Exception:
            logger.exception("record_resource_event failed")
            return None

    # --- Serialization helpers --------------------------------------
    def to_serializable(self) -> dict[str, Any]:
        with self._lock:
            return {
                "energy_pools": copy.deepcopy(self.energy_pools),
                "consumption_rates": copy.deepcopy(self.consumption_rates),
                "recovery_rates": copy.deepcopy(self.recovery_rates),
                "critical_thresholds": copy.deepcopy(self.critical_thresholds),
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        if not data:
            return
        with self._lock:
            try:
                ep = data.get("energy_pools", {})
                for k, v in ep.items():
                    self.energy_pools[k] = _clamp01(float(v))
                cr = data.get("consumption_rates", {})
                self.consumption_rates.update({k: float(v) for k, v in cr.items()})
                rr = data.get("recovery_rates", {})
                self.recovery_rates.update({k: float(v) for k, v in rr.items()})
                ct = data.get("critical_thresholds", {})
                self.critical_thresholds.update({k: float(v) for k, v in ct.items()})
            except Exception:
                logger.exception("load_serializable failed", exc_info=True)
