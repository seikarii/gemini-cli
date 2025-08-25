"""
Hardware Abstraction Layer
==========================

This module defines the Hardware Abstraction Layer (HAL), which simulates the
physical systems of the entity, including homeostasis, adaptation, and feedback loops.

This file has been refactored and completed with the full implementation.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from threading import RLock
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.cuerpo.biological_attributes import (
    PhysicalAttributes,
    ResourceManagement,
)
from crisalida_lib.ADAM.cuerpo.biological_enums import DetailedPhysiologicalState
from crisalida_lib.ADAM.enums import FisiologicalState

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
else:
    EVAMemoryManager = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_trend(
    values: list[float],
    threshold: float = 0.02,
    is_percentage: bool = True,
    config: AdamConfig | None = None,
) -> str:
    """
    Simple, robust trend calculation with defensive guards.

    - values: sequence of numeric samples (older -> newer).
    - threshold: absolute relative change threshold to consider as trend.
    - is_percentage: not used currently but kept for compatibility.
    - config: optional AdamConfig that can override thresholds.
    """
    try:
        if not values or len(values) < 2:
            return "stable"

        if config is not None and hasattr(config, "HARDWARE_TREND_THRESHOLD"):
            threshold = float(getattr(config, "HARDWARE_TREND_THRESHOLD", threshold))

        start = float(values[0]) if float(values[0]) != 0 else 1e-6
        end = float(values[-1])
        change = (end - start) / abs(start) if start != 0 else 0.0

        if abs(change) < threshold:
            return "stable"
        return "increasing" if change > 0 else "decreasing"
    except Exception:
        logger.exception("calculate_trend failed")
        return "stable"


class HardwareAbstractionLayer:
    """
    Simulates the physical hardware and biological processes of the entity.

    Key improvements:
      - Thread-safety via RLock.
      - Async-aware EVA persistence helpers.
      - Defensive subsystem calls and graceful degradation.
    """

    EVA_SNAPSHOT_PERIOD = 30.0

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config: AdamConfig = config or AdamConfig()  # type: ignore
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id

        self._lock = RLock()

        self.physical_attributes = PhysicalAttributes()
        self.resource_management = ResourceManagement()
        self.detailed_physiological_state = DetailedPhysiologicalState()
        self.physiological_state = FisiologicalState.HEALTHY

        self.vital_signs_history: list[dict[str, Any]] = []
        self.performance_metrics: dict[str, float] = {
            "cpu_utilization": 0.4,
            "memory_usage": 0.3,
            "energy_efficiency": 0.75,
            "thermal_regulation": 0.8,
            "system_stability": 0.85,
        }
        self.feedback_loops: dict[str, float] = {
            "neural_feedback": 0.0,
            "hormonal_feedback": 0.0,
            "metabolic_feedback": 0.0,
            "circadian_feedback": 0.0,
        }
        self.adaptation_parameters: dict[str, float] = {
            "stress_tolerance": 0.6,
            "recovery_speed": 0.7,
            "energy_conservation": 0.5,
            "performance_optimization": 0.6,
        }
        self.last_update_time: float = time.time()
        self.hardware_events_log: list[str] = []
        self._last_eva_snapshot: float = 0.0

    # --- Public API -------------------------------------------------
    def update_physical_state(
        self,
        hormonal_state: dict[str, Any],
        mental_load: float,
        environmental_factors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update physical state given hormonal inputs and mental load.

        Returns a comprehensive snapshot (dict) summarizing physical/resource/performance state.
        """
        environmental_factors = environmental_factors or {}
        with self._lock:
            current_time = time.time()
            delta_time = max(1e-6, current_time - self.last_update_time)
            self.last_update_time = current_time

            try:
                # Core update steps - each is defensive to preserve robustness
                self._update_vital_signs(hormonal_state, mental_load, delta_time)
                self._update_detailed_physiological_state(
                    hormonal_state, mental_load, delta_time
                )
                resource_state = self._manage_energy_resources(
                    hormonal_state, mental_load, delta_time
                )
                performance_state = self._update_performance_metrics(
                    hormonal_state, mental_load, environmental_factors, delta_time
                )
                self._process_system_feedback(hormonal_state, resource_state)
                adaptation_state = self._apply_dynamic_adaptations(
                    resource_state, performance_state, delta_time
                )
                self._evaluate_physiological_state(
                    resource_state, performance_state, hormonal_state
                )
                self._update_vital_signs_history()
                hardware_events = self._generate_hardware_events(
                    resource_state, performance_state
                )

                snapshot = {
                    "physical_attributes": self._get_physical_attributes_dict(),
                    "resource_state": resource_state,
                    "performance_metrics": copy.deepcopy(self.performance_metrics),
                    "physiological_state": self.physiological_state.value,
                    "detailed_physiological_state": {
                        "energy_level": self.detailed_physiological_state.energy_level,
                        "fatigue_level": self.detailed_physiological_state.fatigue_level,
                        "stress_level": self.detailed_physiological_state.stress_level,
                        "hunger_level": self.detailed_physiological_state.hunger_level,
                        "pain_level": self.detailed_physiological_state.pain_level,
                        "overall_health_score": self.detailed_physiological_state.calculate_overall_health_score(),
                        "primary_state": self.detailed_physiological_state.get_primary_physiological_state().value,
                        "feedback_modifiers": self.detailed_physiological_state.get_feedback_modifiers(),
                    },
                    "feedback_loops": copy.deepcopy(self.feedback_loops),
                    "adaptation_state": adaptation_state,
                    "hardware_events": hardware_events,
                    "system_coherence": self._calculate_system_coherence(),
                    "recommendations": self._generate_hardware_recommendations(),
                    "timestamp": time.time(),
                }

                # Best-effort EVA snapshotting (async-aware)
                if (
                    self.eva_manager
                    and (time.time() - self._last_eva_snapshot)
                    > self.EVA_SNAPSHOT_PERIOD
                ):
                    asyncio.ensure_future(
                        self._record_eva_async("hardware_state_update", snapshot)
                    )
                    self._last_eva_snapshot = time.time()

                return snapshot
            except Exception:
                logger.exception("update_physical_state failed")
                return {}

    def simulate_physical_stress_test(
        self, stress_type: str, intensity: float = 0.8, duration: float = 1.0
    ) -> dict[str, Any]:
        with self._lock:
            # Validate intensity
            if intensity < 0.0 or intensity > 1.0:
                raise ValueError("Intensity must be between 0.0 and 1.0")
            initial_state = self.get_system_diagnostics()
            # Apply stressors (same logic, preserved, defensive)
            try:
                if stress_type == "physical_exertion":
                    self.physical_attributes.heart_rate += intensity * 50
                    self.physical_attributes.respiratory_rate += intensity * 10
                    self.physical_attributes.body_temperature += intensity * 1.0
                    for pool in ("muscular", "cellular", "metabolic"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0, 1.0 - intensity * 0.3
                            )
                elif stress_type == "thermal_stress":
                    self.physical_attributes.body_temperature += intensity * 2.0
                    self.performance_metrics["thermal_regulation"] *= max(
                        0.0, 1.0 - intensity * 0.4
                    )
                elif stress_type == "cognitive_overload":
                    self.performance_metrics["cpu_utilization"] = min(
                        1.0,
                        self.performance_metrics["cpu_utilization"] + intensity * 0.4,
                    )
                    for pool in ("neural", "cognitive"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0,
                                1.0 - intensity * (0.4 if pool == "neural" else 0.5),
                            )
                elif stress_type == "sleep_deprivation":
                    self.physical_attributes.sleep_debt = min(
                        1.0, self.physical_attributes.sleep_debt + intensity * 0.6
                    )
                    self.physical_attributes.neural_conductivity *= max(
                        0.0, 1.0 - intensity * 0.3
                    )
            except Exception:
                logger.exception("simulate_physical_stress_test application failed")

            # Small simulated passage of time to let metrics update
            time.sleep(min(0.1, duration * 0.01))
            final_state = self.get_system_diagnostics()
            stress_response = self._analyze_stress_response(
                initial_state, final_state, intensity
            )
            return {
                "stress_type": stress_type,
                "intensity": intensity,
                "duration": duration,
                "initial_state": initial_state,
                "final_state": final_state,
                "stress_response": stress_response,
                "recovery_recommendations": self._generate_recovery_protocol(
                    stress_type, intensity
                ),
            }

    def record_hardware_experience(
        self,
        experience_id: str | None = None,
        hardware_state: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Records a hardware experience using the EVAMemoryManager (best-effort).
        Returns generated experience id or None.
        """
        with self._lock:
            if not self.eva_manager:
                logger.warning(
                    "EVAMemoryManager not available, cannot record hardware experience."
                )
                return None

            hardware_state = hardware_state or self.get_system_diagnostics()
            experience_id = (
                experience_id
                or f"hardware_experience_{abs(hash(str(hardware_state))) & 0xFFFFFFFF}"
            )

            experience_data = {
                "hardware_event": hardware_state,
                "physical_attributes": self._get_physical_attributes_dict(),
                "resource_state": hardware_state.get("resource_analysis", {}),
                "performance_metrics": copy.deepcopy(self.performance_metrics),
                "physiological_state": self.physiological_state.value,
                "detailed_physiological_state": {
                    "energy_level": self.detailed_physiological_state.energy_level,
                    "fatigue_level": self.detailed_physiological_state.fatigue_level,
                    "stress_level": self.detailed_physiological_state.stress_level,
                    "hunger_level": self.detailed_physiological_state.hunger_level,
                    "pain_level": self.detailed_physiological_state.pain_level,
                    "overall_health_score": self.detailed_physiological_state.calculate_overall_health_score(),
                    "primary_state": self.detailed_physiological_state.get_primary_physiological_state().value,
                    "feedback_modifiers": self.detailed_physiological_state.get_feedback_modifiers(),
                },
                "feedback_loops": copy.deepcopy(self.feedback_loops),
                "adaptation_state": hardware_state.get("adaptation_status", {}),
                "hardware_events": (
                    self.hardware_events_log[-10:] if self.hardware_events_log else []
                ),
                "system_coherence": self._calculate_system_coherence(),
                "recommendations": self._generate_hardware_recommendations(),
                "timestamp": time.time(),
            }

            # Best-effort, async-aware write
            asyncio.ensure_future(
                self._record_eva_async(
                    "hardware_experience", experience_data, experience_id
                )
            )
            logger.info(
                "Scheduled EVA record for hardware_experience: %s", experience_id
            )
            return experience_id

    def recall_hardware_experience(self, experience_id: str) -> dict[str, Any] | None:
        """
        Recalls a hardware experience from EVAMemoryManager. If backend is async and running
        event loop, schedule call and return None (best-effort).
        """
        if not self.eva_manager:
            logger.warning(
                "EVAMemoryManager not available, cannot recall hardware experience."
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not callable(recall):
                logger.debug("EVAMemoryManager.recall_experience is not callable")
                return None
            res = recall(entity_id=self.entity_id, experience_id=experience_id)
            if hasattr(res, "__await__"):
                # If loop running, schedule; otherwise run and return
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(res)
                    logger.debug(
                        "Scheduled async recall; returning None until available"
                    )
                    return None
                return loop.run_until_complete(res)
            return res
        except Exception:
            logger.exception("recall_hardware_experience failed")
            return None

    # --- Internal utilities -----------------------------------------
    async def _record_eva_async(
        self, event_type: str, data: dict[str, Any], experience_id: str | None = None
    ) -> None:
        """
        Helper to record to EVA in an async-safe manner. Uses asyncio.ensure_future
        to avoid blocking callers. Best-effort: logs failures.
        """
        if not self.eva_manager:
            return
        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                logger.debug("EVAMemoryManager.record_experience not callable")
                return
            res = recorder(
                entity_id=self.entity_id,
                event_type=event_type,
                data=data,
                experience_id=experience_id,
            )
            if hasattr(res, "__await__"):
                try:
                    await res
                except Exception:
                    logger.exception("Async EVA recorder failed for %s", event_type)
            else:
                # sync result already executed
                return
        except Exception:
            logger.exception("Failed to record EVA experience (async helper)")

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    # --- The rest of the implementation preserves original logic but wrapped with lock where needed ---
    def _update_vital_signs(self, hormonal_state, mental_load, delta_time):
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        adrenaline = hormonal_state.get("niveles_actuales", {}).get("adrenalina", 0.1)
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)

        base_hr = 70.0
        target_hr = (
            base_hr + stress_level * 30.0 + adrenaline * 40.0 + mental_load * 15.0
        )
        hr_change_rate = 10.0 * delta_time
        hr_diff = target_hr - self.physical_attributes.heart_rate
        hr_diff = max(-hr_change_rate, min(hr_diff, hr_change_rate))
        self.physical_attributes.heart_rate = max(
            50.0, min(180.0, self.physical_attributes.heart_rate + hr_diff)
        )

        self.physical_attributes.blood_pressure_systolic = max(
            90.0, min(180.0, 120.0 + stress_level * 20.0)
        )
        self.physical_attributes.blood_pressure_diastolic = max(
            60.0, min(120.0, 80.0 + stress_level * 15.0)
        )
        self.physical_attributes.body_temperature = max(
            36.0, min(39.0, 37.0 + stress_level * 0.5)
        )
        self.physical_attributes.respiratory_rate = max(
            10.0, min(30.0, 16.0 + stress_level * 8.0)
        )
        self.physical_attributes.muscle_tension = max(
            0.0, min(1.0, 0.3 + stress_level * 0.4 + cortisol * 0.3)
        )

    def _update_detailed_physiological_state(
        self, hormonal_state: dict[str, Any], mental_load: float, delta_time: float
    ) -> None:
        """
        Actualiza el estado fisiológico detallado basado en estado hormonal y carga mental.
        """
        # Extract hormonal information
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        dopamine = hormonal_state.get("niveles_actuales", {}).get("dopamina", 0.5)
        _serotonin = hormonal_state.get("niveles_actuales", {}).get("serotonina", 0.5)
        estado_general = hormonal_state.get("estado_hormonal", {})
        _stress_level = estado_general.get("estres", 0.3)

        # Update stress level based on cortisol and mental load
        stress_increase = (cortisol - 0.2) * 0.5 + mental_load * 0.3
        self.detailed_physiological_state.stress_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.stress_level
                + stress_increase * delta_time,
            ),
        )

        # Update energy level based on dopamine and stress
        energy_change = (
            dopamine - 0.5
        ) * 0.2 - self.detailed_physiological_state.stress_level * 0.1
        self.detailed_physiological_state.energy_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.energy_level
                + energy_change * delta_time,
            ),
        )

        # Update fatigue based on mental load and energy
        fatigue_increase = (
            mental_load * 0.2
            + (1.0 - self.detailed_physiological_state.energy_level) * 0.1
        )
        self.detailed_physiological_state.fatigue_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.fatigue_level
                + fatigue_increase * delta_time,
            ),
        )

        # Natural recovery and homeostasis
        if self.detailed_physiological_state.stress_level > 0:
            self.detailed_physiological_state.stress_level = max(
                0.0, self.detailed_physiological_state.stress_level - 0.05 * delta_time
            )

        if self.detailed_physiological_state.fatigue_level > 0:
            self.detailed_physiological_state.fatigue_level = max(
                0.0, self.detailed_physiological_state.fatigue_level - 0.03 * delta_time
            )

        # Update hunger over time (gradual increase)
        self.detailed_physiological_state.hunger_level = min(
            1.0, self.detailed_physiological_state.hunger_level + 0.02 * delta_time
        )

        # Update hydration (gradual decrease)
        self.detailed_physiological_state.hydration_level = max(
            0.0, self.detailed_physiological_state.hydration_level - 0.01 * delta_time
        )

        # Update sleep debt based on time and stress
        if self.physiological_state != FisiologicalState.SLEEPING:
            sleep_debt_increase = (
                0.03 * delta_time
                + self.detailed_physiological_state.stress_level * 0.01 * delta_time
            )
            self.detailed_physiological_state.sleep_debt = min(
                1.0, self.detailed_physiological_state.sleep_debt + sleep_debt_increase
            )

        # Update the primary physiological state based on detailed state
        new_primary_state = (
            self.detailed_physiological_state.get_primary_physiological_state()
        )
        if new_primary_state != self.physiological_state:
            self.physiological_state = new_primary_state

    def _generate_hardware_events(self, resource_state, performance_state):
        events = []
        if resource_state["resource_status"] == "critical":
            events.append("ALERT: Energy pools critically low")
        if self.physiological_state == FisiologicalState.CRITICAL:
            events.append("ALERT: System in critical physiological state")
        if self.performance_metrics["cpu_utilization"] > 0.95:
            events.append("WARNING: CPU utilization extremely high")
        if self.performance_metrics["thermal_regulation"] < 0.4:
            events.append("WARNING: Thermal regulation compromised")
        if events:
            self.hardware_events_log.extend(events)
        return events

    def _generate_hardware_recommendations(self) -> list[str]:
        recs = []
        if self.physiological_state == FisiologicalState.CRITICAL:
            recs.append("Iniciar protocolo de emergencia y recuperación intensiva")
        elif self.physiological_state == FisiologicalState.EXHAUSTED:
            recs.append("Recomendar descanso prolongado y monitoreo de recursos")
        elif self.physiological_state == FisiologicalState.STRESSED:
            recs.append("Aplicar técnicas de reducción de estrés y optimizar recursos")
        if self.performance_metrics["thermal_regulation"] < 0.5:
            recs.append("Ajustar temperatura ambiental y aumentar hidratación")
        if self.performance_metrics["cpu_utilization"] > 0.9:
            recs.append("Reducir carga cognitiva y optimizar procesos")
        return recs

    def _analyze_vital_signs_trends(self, window_size: int = 10) -> dict[str, str]:
        if len(self.vital_signs_history) < window_size:
            return {"status": "insufficient_data"}
        recent_history = self.vital_signs_history[-window_size:]
        trends = {}
        hr_values = [entry["heart_rate"] for entry in recent_history]
        trends["heart_rate"] = self._calculate_trend_direction(hr_values)
        temp_values = [entry["body_temperature"] for entry in recent_history]
        trends["body_temperature"] = self._calculate_trend_direction(temp_values)
        tension_values = [entry["muscle_tension"] for entry in recent_history]
        trends["muscle_tension"] = self._calculate_trend_direction(tension_values)
        energy_averages = [
            sum(entry["energy_pools"].values()) / len(entry["energy_pools"])
            for entry in recent_history
        ]
        trends["energy_level"] = self._calculate_trend_direction(energy_averages)
        return trends

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    def _manage_energy_resources(self, hormonal_state, mental_load, delta_time):
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)
        consumo_total = dict.fromkeys(self.resource_management.energy_pools, 0.0)
        for pool in consumo_total:
            consumo_total[pool] += (
                self.resource_management.consumption_rates["rest"] * delta_time
            )
        cognitive_consumption = (
            mental_load
            * self.resource_management.consumption_rates["thinking"]
            * delta_time
        )
        consumo_total["neural"] += cognitive_consumption
        consumo_total["cognitive"] += cognitive_consumption * 1.5
        stress_consumption = (
            stress_level
            * self.resource_management.consumption_rates["stress"]
            * delta_time
        )
        for pool in consumo_total:
            consumo_total[pool] += stress_consumption
        for pool, consumo in consumo_total.items():
            self.resource_management.energy_pools[pool] = max(
                0.0, self.resource_management.energy_pools[pool] - consumo
            )
        for pool, tasa_recuperacion in self.resource_management.recovery_rates.items():
            recovery = tasa_recuperacion * delta_time * (1.0 - stress_level * 0.5)
            self.resource_management.energy_pools[pool] = min(
                1.0, self.resource_management.energy_pools[pool] + recovery
            )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        self.physical_attributes.atp_reserves = self.resource_management.energy_pools[
            "cellular"
        ]
        self.physical_attributes.glucose_level = min(1.0, avg_energy * 1.2)
        resource_status = self._evaluate_resource_status()
        return {
            "energy_pools": self.resource_management.energy_pools.copy(),
            "consumption_rates": consumo_total,
            "resource_status": resource_status,
            "energy_efficiency": self._calculate_energy_efficiency(),
            "recovery_capacity": self._calculate_recovery_capacity(stress_level),
        }

    def _update_performance_metrics(
        self, hormonal_state, mental_load, environmental_factors, delta_time
    ):
        estado_general = hormonal_state.get("estado_hormonal", {})
        base_cpu = 0.4
        mental_cpu = (
            mental_load * 0.5
        )  # Changed from 0.4 to 0.5 for consistency with legacy
        stress_cpu = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["cpu_utilization"] = min(
            1.0, base_cpu + mental_cpu + stress_cpu
        )
        cognitive_memory = mental_load * 0.5
        emotional_memory = abs(estado_general.get("bienestar", 0.5) - 0.5) * 0.3
        self.performance_metrics["memory_usage"] = min(
            1.0, 0.2 + cognitive_memory + emotional_memory
        )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        stress_penalty = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["energy_efficiency"] = max(
            0.0, avg_energy - stress_penalty
        )
        temp_optimal = abs(self.physical_attributes.body_temperature - 37.0) < 0.5
        stress_thermal_impact = estado_general.get("estres", 0.3) * 0.2
        self.performance_metrics["thermal_regulation"] = max(
            0.0, (0.9 if temp_optimal else 0.6) - stress_thermal_impact
        )
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        resource_stability = min(self.resource_management.energy_pools.values())
        self.performance_metrics["system_stability"] = (
            coherence + resource_stability
        ) / 2
        return {
            "cpu_usage_breakdown": {
                "base": base_cpu,
                "mental": mental_cpu,
                "stress": stress_cpu,
            },
            "thermal_state": {
                "temperature": self.physical_attributes.body_temperature,
                "regulation_efficiency": self.performance_metrics["thermal_regulation"],
            },
            "stability_factors": {
                "hormonal_coherence": coherence,
                "resource_stability": resource_stability,
            },
        }

    def _process_system_feedback(self, hormonal_state, resource_state):
        neural_energy = resource_state["energy_pools"]["neural"]
        cpu_load = self.performance_metrics["cpu_utilization"]
        self.feedback_loops["neural_feedback"] = neural_energy - cpu_load * 0.5
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        self.feedback_loops["hormonal_feedback"] = coherence - stress_level
        metabolic_energy = resource_state["energy_pools"]["metabolic"]
        energy_efficiency = self.performance_metrics["energy_efficiency"]
        self.feedback_loops["metabolic_feedback"] = (
            metabolic_energy + energy_efficiency
        ) / 2 - 0.5
        self.feedback_loops["circadian_feedback"] = (
            self._calculate_circadian_alignment()
        )

    def _apply_dynamic_adaptations(self, resource_state, performance_state, delta_time):
        adaptations_applied = {}
        if (
            resource_state["energy_efficiency"]
            < self.resource_management.critical_thresholds["energy_depletion"]
        ):
            self.adaptation_parameters["energy_conservation"] = min(
                1.0,
                self.adaptation_parameters["energy_conservation"] + 0.1 * delta_time,
            )
            for rate_type in self.resource_management.consumption_rates:
                self.resource_management.consumption_rates[rate_type] *= 0.98
            adaptations_applied["energy_conservation"] = True
        if self.performance_metrics["system_stability"] < 0.5:
            self.adaptation_parameters["stress_tolerance"] = min(
                1.0, self.adaptation_parameters["stress_tolerance"] + 0.05 * delta_time
            )
            adaptations_applied["stress_adaptation"] = True
        if self.performance_metrics["cpu_utilization"] > 0.9:
            self.adaptation_parameters["performance_optimization"] = min(
                1.0,
                self.adaptation_parameters["performance_optimization"]
                + 0.08 * delta_time,
            )
            adaptations_applied["performance_optimization"] = True
        avg_energy = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        if avg_energy < self.resource_management.critical_thresholds["recovery_needed"]:
            self.adaptation_parameters["recovery_speed"] = min(
                1.0, self.adaptation_parameters["recovery_speed"] + 0.06 * delta_time
            )
            for pool in self.resource_management.recovery_rates:
                self.resource_management.recovery_rates[pool] *= 1.02
            adaptations_applied["enhanced_recovery"] = True
        return {
            "adaptations_applied": adaptations_applied,
            "adaptation_parameters": self.adaptation_parameters.copy(),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(),
            "homeostasis_score": self._calculate_homeostasis_score(),
        }

    def _evaluate_physiological_state(
        self, resource_state, performance_state, hormonal_state
    ):
        energy_level = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        system_stability = self.performance_metrics["system_stability"]
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        health_score = (
            energy_level + system_stability + coherence
        ) / 3 - stress_level * 0.5
        if health_score >= 0.8 and stress_level < 0.3:
            new_state = FisiologicalState.OPTIMAL
        elif health_score >= 0.6 and stress_level < 0.5:
            new_state = FisiologicalState.HEALTHY
        elif health_score >= 0.4:
            new_state = FisiologicalState.RECOVERING
        elif health_score >= 0.2:
            new_state = FisiologicalState.STRESSED
        elif health_score >= 0.1:
            new_state = FisiologicalState.EXHAUSTED
        else:
            new_state = FisiologicalState.CRITICAL
        if new_state != self.physiological_state:
            self.physiological_state = new_state
        return new_state

    def _calculate_system_coherence(self) -> float:
        vital_coherence = self._calculate_vital_signs_coherence()
        resource_coherence = self._calculate_resource_coherence()
        performance_coherence = self._calculate_performance_coherence()
        feedback_coherence = self._calculate_feedback_coherence()
        return (
            vital_coherence
            + resource_coherence
            + performance_coherence
            + feedback_coherence
        ) / 4

    def _calculate_vital_signs_coherence(self) -> float:
        hr_normal = 0.8 if 60 <= self.physical_attributes.heart_rate <= 100 else 0.4
        bp_normal = (
            0.8
            if 90 <= self.physical_attributes.blood_pressure_systolic <= 140
            else 0.4
        )
        temp_normal = (
            0.9 if 36.0 <= self.physical_attributes.body_temperature <= 37.5 else 0.5
        )
        resp_normal = (
            0.8 if 12 <= self.physical_attributes.respiratory_rate <= 20 else 0.4
        )
        return (hr_normal + bp_normal + temp_normal + resp_normal) / 4

    def _calculate_resource_coherence(self) -> float:
        energy_values = list(self.resource_management.energy_pools.values())
        if not energy_values:
            return 0.0
        mean_energy = sum(energy_values) / len(energy_values)
        variance = sum((x - mean_energy) ** 2 for x in energy_values) / len(
            energy_values
        )
        max_possible_variance = 0.25
        return max(0.0, 1.0 - (variance / max_possible_variance))

    def _calculate_performance_coherence(self) -> float:
        metrics = self.performance_metrics
        values = [
            metrics["cpu_utilization"],
            metrics["memory_usage"],
            metrics["energy_efficiency"],
            metrics["thermal_regulation"],
            metrics["system_stability"],
        ]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _calculate_feedback_coherence(self) -> float:
        values = list(self.feedback_loops.values())
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _update_vital_signs_history(self) -> None:
        entry = {
            "timestamp": time.time(),
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure": {
                "systolic": self.physical_attributes.blood_pressure_systolic,
                "diastolic": self.physical_attributes.blood_pressure_diastolic,
            },
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "energy_pools": self.resource_management.energy_pools.copy(),
            "physiological_state": self.physiological_state.value,
        }
        self.vital_signs_history.append(entry)
        if len(self.vital_signs_history) > 100:
            self.vital_signs_history = self.vital_signs_history[-100:]

    def _get_physical_attributes_dict(self) -> dict[str, Any]:
        return {
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure_systolic": self.physical_attributes.blood_pressure_systolic,
            "blood_pressure_diastolic": self.physical_attributes.blood_pressure_diastolic,
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "glucose_level": self.physical_attributes.glucose_level,
            "atp_reserves": self.physical_attributes.atp_reserves,
            "oxygen_saturation": self.physical_attributes.oxygen_saturation,
            "hydration_level": self.physical_attributes.hydration_level,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "neural_conductivity": self.physical_attributes.neural_conductivity,
            "inflammatory_response": self.physical_attributes.inflammatory_response,
            "immune_system_strength": self.physical_attributes.immune_system_strength,
            "circadian_phase": self.physical_attributes.circadian_phase,
            "sleep_debt": self.physical_attributes.sleep_debt,
            "recovery_rate": self.physical_attributes.recovery_rate,
        }

    def _calculate_energy_efficiency(self) -> float:
        pools = self.resource_management.energy_pools.values()
        return float(min(1.0, sum(pools) / len(pools)))

    def _calculate_recovery_capacity(self, stress_level: float) -> float:
        base = sum(self.resource_management.recovery_rates.values()) / len(
            self.resource_management.recovery_rates
        )
        return base * (1.0 - stress_level * 0.5)

    def _evaluate_resource_status(self) -> str:
        pools = self.resource_management.energy_pools.values()
        if any(
            v < self.resource_management.critical_thresholds["energy_depletion"]
            for v in pools
        ):
            return "critical"
        elif all(
            v > self.resource_management.critical_thresholds["optimal_performance"]
            for v in pools
        ):
            return "optimal"
        elif any(
            v < self.resource_management.critical_thresholds["recovery_needed"]
            for v in pools
        ):
            return "recovery_needed"
        else:
            return "stable"

    def _calculate_circadian_alignment(self) -> float:
        phase = self.physical_attributes.circadian_phase
        return 1.0 - abs(phase - 0.5) * 2

    def _calculate_adaptation_effectiveness(self) -> float:
        params = self.adaptation_parameters.values()
        return float(min(1.0, sum(params) / len(params)))

    def _calculate_homeostasis_score(self) -> float:
        health_index = self.physical_attributes.get_overall_health_index()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (health_index * 0.6 + adaptation * 0.4)))

    def _calculate_system_resonance(self) -> float:
        coherence = self._calculate_system_coherence()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (coherence + adaptation) / 2))

    def get_system_diagnostics(self) -> dict[str, Any]:
        return {
            "current_state": {
                "physiological_state": self.physiological_state.value,
                "system_coherence": self._calculate_system_coherence(),
                "homeostasis_score": self._calculate_homeostasis_score(),
            },
            "vital_signs": self._get_physical_attributes_dict(),
            "resource_analysis": {
                "energy_pools": self.resource_management.energy_pools.copy(),
                "resource_status": self._evaluate_resource_status(),
                "energy_efficiency": self._calculate_energy_efficiency(),
            },
            "performance_metrics": self.performance_metrics.copy(),
            "adaptation_status": {
                "parameters": self.adaptation_parameters.copy(),
                "effectiveness": self._calculate_adaptation_effectiveness(),
            },
            "feedback_analysis": {
                "feedback_loops": self.feedback_loops.copy(),
                "system_resonance": self._calculate_system_resonance(),
            },
            "recent_events": (
                self.hardware_events_log[-10:] if self.hardware_events_log else []
            ),
            "recommendations": self._generate_hardware_recommendations(),
            "historical_trends": (
                self._analyze_vital_signs_trends()
                if len(self.vital_signs_history) > 5
                else None
            ),
        }

    def simulate_physical_stress_test(
        self, stress_type: str, intensity: float = 0.8, duration: float = 1.0
    ) -> dict[str, Any]:
        with self._lock:
            # Validate intensity
            if intensity < 0.0 or intensity > 1.0:
                raise ValueError("Intensity must be between 0.0 and 1.0")
            initial_state = self.get_system_diagnostics()
            # Apply stressors (same logic, preserved, defensive)
            try:
                if stress_type == "physical_exertion":
                    self.physical_attributes.heart_rate += intensity * 50
                    self.physical_attributes.respiratory_rate += intensity * 10
                    self.physical_attributes.body_temperature += intensity * 1.0
                    for pool in ("muscular", "cellular", "metabolic"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0, 1.0 - intensity * 0.3
                            )
                elif stress_type == "thermal_stress":
                    self.physical_attributes.body_temperature += intensity * 2.0
                    self.performance_metrics["thermal_regulation"] *= max(
                        0.0, 1.0 - intensity * 0.4
                    )
                elif stress_type == "cognitive_overload":
                    self.performance_metrics["cpu_utilization"] = min(
                        1.0,
                        self.performance_metrics["cpu_utilization"] + intensity * 0.4,
                    )
                    for pool in ("neural", "cognitive"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0,
                                1.0 - intensity * (0.4 if pool == "neural" else 0.5),
                            )
                elif stress_type == "sleep_deprivation":
                    self.physical_attributes.sleep_debt = min(
                        1.0, self.physical_attributes.sleep_debt + intensity * 0.6
                    )
                    self.physical_attributes.neural_conductivity *= max(
                        0.0, 1.0 - intensity * 0.3
                    )
            except Exception:
                logger.exception("simulate_physical_stress_test application failed")

            # Small simulated passage of time to let metrics update
            time.sleep(min(0.1, duration * 0.01))
            final_state = self.get_system_diagnostics()
            stress_response = self._analyze_stress_response(
                initial_state, final_state, intensity
            )
            return {
                "stress_type": stress_type,
                "intensity": intensity,
                "duration": duration,
                "initial_state": initial_state,
                "final_state": final_state,
                "stress_response": stress_response,
                "recovery_recommendations": self._generate_recovery_protocol(
                    stress_type, intensity
                ),
            }

    def record_hardware_experience(
        self,
        experience_id: str | None = None,
        hardware_state: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Records a hardware experience using the EVAMemoryManager (best-effort).
        Returns generated experience id or None.
        """
        with self._lock:
            if not self.eva_manager:
                logger.warning(
                    "EVAMemoryManager not available, cannot record hardware experience."
                )
                return None

            hardware_state = hardware_state or self.get_system_diagnostics()
            experience_id = (
                experience_id
                or f"hardware_experience_{abs(hash(str(hardware_state))) & 0xFFFFFFFF}"
            )

            experience_data = {
                "hardware_event": hardware_state,
                "physical_attributes": self._get_physical_attributes_dict(),
                "resource_state": hardware_state.get("resource_analysis", {}),
                "performance_metrics": copy.deepcopy(self.performance_metrics),
                "physiological_state": self.physiological_state.value,
                "detailed_physiological_state": {
                    "energy_level": self.detailed_physiological_state.energy_level,
                    "fatigue_level": self.detailed_physiological_state.fatigue_level,
                    "stress_level": self.detailed_physiological_state.stress_level,
                    "hunger_level": self.detailed_physiological_state.hunger_level,
                    "pain_level": self.detailed_physiological_state.pain_level,
                    "overall_health_score": self.detailed_physiological_state.calculate_overall_health_score(),
                    "primary_state": self.detailed_physiological_state.get_primary_physiological_state().value,
                    "feedback_modifiers": self.detailed_physiological_state.get_feedback_modifiers(),
                },
                "feedback_loops": copy.deepcopy(self.feedback_loops),
                "adaptation_state": hardware_state.get("adaptation_status", {}),
                "hardware_events": (
                    self.hardware_events_log[-10:] if self.hardware_events_log else []
                ),
                "system_coherence": self._calculate_system_coherence(),
                "recommendations": self._generate_hardware_recommendations(),
                "timestamp": time.time(),
            }

            # Best-effort, async-aware write
            asyncio.ensure_future(
                self._record_eva_async(
                    "hardware_experience", experience_data, experience_id
                )
            )
            logger.info(
                "Scheduled EVA record for hardware_experience: %s", experience_id
            )
            return experience_id

    def recall_hardware_experience(self, experience_id: str) -> dict[str, Any] | None:
        """
        Recalls a hardware experience from EVAMemoryManager. If backend is async and running
        event loop, schedule call and return None (best-effort).
        """
        if not self.eva_manager:
            logger.warning(
                "EVAMemoryManager not available, cannot recall hardware experience."
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not callable(recall):
                logger.debug("EVAMemoryManager.recall_experience is not callable")
                return None
            res = recall(entity_id=self.entity_id, experience_id=experience_id)
            if hasattr(res, "__await__"):
                # If loop running, schedule; otherwise run and return
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(res)
                    logger.debug(
                        "Scheduled async recall; returning None until available"
                    )
                    return None
                return loop.run_until_complete(res)
            return res
        except Exception:
            logger.exception("recall_hardware_experience failed")
            return None

    # --- Internal utilities -----------------------------------------
    async def _record_eva_async(
        self, event_type: str, data: dict[str, Any], experience_id: str | None = None
    ) -> None:
        """
        Helper to record to EVA in an async-safe manner. Uses asyncio.ensure_future
        to avoid blocking callers. Best-effort: logs failures.
        """
        if not self.eva_manager:
            return
        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                logger.debug("EVAMemoryManager.record_experience not callable")
                return
            res = recorder(
                entity_id=self.entity_id,
                event_type=event_type,
                data=data,
                experience_id=experience_id,
            )
            if hasattr(res, "__await__"):
                try:
                    await res
                except Exception:
                    logger.exception("Async EVA recorder failed for %s", event_type)
            else:
                # sync result already executed
                return
        except Exception:
            logger.exception("Failed to record EVA experience (async helper)")

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    # --- The rest of the implementation preserves original logic but wrapped with lock where needed ---
    def _update_vital_signs(self, hormonal_state, mental_load, delta_time):
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        adrenaline = hormonal_state.get("niveles_actuales", {}).get("adrenalina", 0.1)
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)

        base_hr = 70.0
        target_hr = (
            base_hr + stress_level * 30.0 + adrenaline * 40.0 + mental_load * 15.0
        )
        hr_change_rate = 10.0 * delta_time
        hr_diff = target_hr - self.physical_attributes.heart_rate
        hr_diff = max(-hr_change_rate, min(hr_diff, hr_change_rate))
        self.physical_attributes.heart_rate = max(
            50.0, min(180.0, self.physical_attributes.heart_rate + hr_diff)
        )

        self.physical_attributes.blood_pressure_systolic = max(
            90.0, min(180.0, 120.0 + stress_level * 20.0)
        )
        self.physical_attributes.blood_pressure_diastolic = max(
            60.0, min(120.0, 80.0 + stress_level * 15.0)
        )
        self.physical_attributes.body_temperature = max(
            36.0, min(39.0, 37.0 + stress_level * 0.5)
        )
        self.physical_attributes.respiratory_rate = max(
            10.0, min(30.0, 16.0 + stress_level * 8.0)
        )
        self.physical_attributes.muscle_tension = max(
            0.0, min(1.0, 0.3 + stress_level * 0.4 + cortisol * 0.3)
        )

    def _update_detailed_physiological_state(
        self, hormonal_state: dict[str, Any], mental_load: float, delta_time: float
    ) -> None:
        """
        Actualiza el estado fisiológico detallado basado en estado hormonal y carga mental.
        """
        # Extract hormonal information
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        dopamine = hormonal_state.get("niveles_actuales", {}).get("dopamina", 0.5)
        _serotonin = hormonal_state.get("niveles_actuales", {}).get("serotonina", 0.5)
        estado_general = hormonal_state.get("estado_hormonal", {})
        _stress_level = estado_general.get("estres", 0.3)

        # Update stress level based on cortisol and mental load
        stress_increase = (cortisol - 0.2) * 0.5 + mental_load * 0.3
        self.detailed_physiological_state.stress_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.stress_level
                + stress_increase * delta_time,
            ),
        )

        # Update energy level based on dopamine and stress
        energy_change = (
            dopamine - 0.5
        ) * 0.2 - self.detailed_physiological_state.stress_level * 0.1
        self.detailed_physiological_state.energy_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.energy_level
                + energy_change * delta_time,
            ),
        )

        # Update fatigue based on mental load and energy
        fatigue_increase = (
            mental_load * 0.2
            + (1.0 - self.detailed_physiological_state.energy_level) * 0.1
        )
        self.detailed_physiological_state.fatigue_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.fatigue_level
                + fatigue_increase * delta_time,
            ),
        )

        # Natural recovery and homeostasis
        if self.detailed_physiological_state.stress_level > 0:
            self.detailed_physiological_state.stress_level = max(
                0.0, self.detailed_physiological_state.stress_level - 0.05 * delta_time
            )

        if self.detailed_physiological_state.fatigue_level > 0:
            self.detailed_physiological_state.fatigue_level = max(
                0.0, self.detailed_physiological_state.fatigue_level - 0.03 * delta_time
            )

        # Update hunger over time (gradual increase)
        self.detailed_physiological_state.hunger_level = min(
            1.0, self.detailed_physiological_state.hunger_level + 0.02 * delta_time
        )

        # Update hydration (gradual decrease)
        self.detailed_physiological_state.hydration_level = max(
            0.0, self.detailed_physiological_state.hydration_level - 0.01 * delta_time
        )

        # Update sleep debt based on time and stress
        if self.physiological_state != FisiologicalState.SLEEPING:
            sleep_debt_increase = (
                0.03 * delta_time
                + self.detailed_physiological_state.stress_level * 0.01 * delta_time
            )
            self.detailed_physiological_state.sleep_debt = min(
                1.0, self.detailed_physiological_state.sleep_debt + sleep_debt_increase
            )

        # Update the primary physiological state based on detailed state
        new_primary_state = (
            self.detailed_physiological_state.get_primary_physiological_state()
        )
        if new_primary_state != self.physiological_state:
            self.physiological_state = new_primary_state

    def _generate_hardware_events(self, resource_state, performance_state):
        events = []
        if resource_state["resource_status"] == "critical":
            events.append("ALERT: Energy pools critically low")
        if self.physiological_state == FisiologicalState.CRITICAL:
            events.append("ALERT: System in critical physiological state")
        if self.performance_metrics["cpu_utilization"] > 0.95:
            events.append("WARNING: CPU utilization extremely high")
        if self.performance_metrics["thermal_regulation"] < 0.4:
            events.append("WARNING: Thermal regulation compromised")
        if events:
            self.hardware_events_log.extend(events)
        return events

    def _generate_hardware_recommendations(self) -> list[str]:
        recs = []
        if self.physiological_state == FisiologicalState.CRITICAL:
            recs.append("Iniciar protocolo de emergencia y recuperación intensiva")
        elif self.physiological_state == FisiologicalState.EXHAUSTED:
            recs.append("Recomendar descanso prolongado y monitoreo de recursos")
        elif self.physiological_state == FisiologicalState.STRESSED:
            recs.append("Aplicar técnicas de reducción de estrés y optimizar recursos")
        if self.performance_metrics["thermal_regulation"] < 0.5:
            recs.append("Ajustar temperatura ambiental y aumentar hidratación")
        if self.performance_metrics["cpu_utilization"] > 0.9:
            recs.append("Reducir carga cognitiva y optimizar procesos")
        return recs

    def _analyze_vital_signs_trends(self, window_size: int = 10) -> dict[str, str]:
        if len(self.vital_signs_history) < window_size:
            return {"status": "insufficient_data"}
        recent_history = self.vital_signs_history[-window_size:]
        trends = {}
        hr_values = [entry["heart_rate"] for entry in recent_history]
        trends["heart_rate"] = self._calculate_trend_direction(hr_values)
        temp_values = [entry["body_temperature"] for entry in recent_history]
        trends["body_temperature"] = self._calculate_trend_direction(temp_values)
        tension_values = [entry["muscle_tension"] for entry in recent_history]
        trends["muscle_tension"] = self._calculate_trend_direction(tension_values)
        energy_averages = [
            sum(entry["energy_pools"].values()) / len(entry["energy_pools"])
            for entry in recent_history
        ]
        trends["energy_level"] = self._calculate_trend_direction(energy_averages)
        return trends

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    def _manage_energy_resources(self, hormonal_state, mental_load, delta_time):
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)
        consumo_total = dict.fromkeys(self.resource_management.energy_pools, 0.0)
        for pool in consumo_total:
            consumo_total[pool] += (
                self.resource_management.consumption_rates["rest"] * delta_time
            )
        cognitive_consumption = (
            mental_load
            * self.resource_management.consumption_rates["thinking"]
            * delta_time
        )
        consumo_total["neural"] += cognitive_consumption
        consumo_total["cognitive"] += cognitive_consumption * 1.5
        stress_consumption = (
            stress_level
            * self.resource_management.consumption_rates["stress"]
            * delta_time
        )
        for pool in consumo_total:
            consumo_total[pool] += stress_consumption
        for pool, consumo in consumo_total.items():
            self.resource_management.energy_pools[pool] = max(
                0.0, self.resource_management.energy_pools[pool] - consumo
            )
        for pool, tasa_recuperacion in self.resource_management.recovery_rates.items():
            recovery = tasa_recuperacion * delta_time * (1.0 - stress_level * 0.5)
            self.resource_management.energy_pools[pool] = min(
                1.0, self.resource_management.energy_pools[pool] + recovery
            )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        self.physical_attributes.atp_reserves = self.resource_management.energy_pools[
            "cellular"
        ]
        self.physical_attributes.glucose_level = min(1.0, avg_energy * 1.2)
        resource_status = self._evaluate_resource_status()
        return {
            "energy_pools": self.resource_management.energy_pools.copy(),
            "consumption_rates": consumo_total,
            "resource_status": resource_status,
            "energy_efficiency": self._calculate_energy_efficiency(),
            "recovery_capacity": self._calculate_recovery_capacity(stress_level),
        }

    def _update_performance_metrics(
        self, hormonal_state, mental_load, environmental_factors, delta_time
    ):
        estado_general = hormonal_state.get("estado_hormonal", {})
        base_cpu = 0.4
        mental_cpu = (
            mental_load * 0.5
        )  # Changed from 0.4 to 0.5 for consistency with legacy
        stress_cpu = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["cpu_utilization"] = min(
            1.0, base_cpu + mental_cpu + stress_cpu
        )
        cognitive_memory = mental_load * 0.5
        emotional_memory = abs(estado_general.get("bienestar", 0.5) - 0.5) * 0.3
        self.performance_metrics["memory_usage"] = min(
            1.0, 0.2 + cognitive_memory + emotional_memory
        )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        stress_penalty = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["energy_efficiency"] = max(
            0.0, avg_energy - stress_penalty
        )
        temp_optimal = abs(self.physical_attributes.body_temperature - 37.0) < 0.5
        stress_thermal_impact = estado_general.get("estres", 0.3) * 0.2
        self.performance_metrics["thermal_regulation"] = max(
            0.0, (0.9 if temp_optimal else 0.6) - stress_thermal_impact
        )
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        resource_stability = min(self.resource_management.energy_pools.values())
        self.performance_metrics["system_stability"] = (
            coherence + resource_stability
        ) / 2
        return {
            "cpu_usage_breakdown": {
                "base": base_cpu,
                "mental": mental_cpu,
                "stress": stress_cpu,
            },
            "thermal_state": {
                "temperature": self.physical_attributes.body_temperature,
                "regulation_efficiency": self.performance_metrics["thermal_regulation"],
            },
            "stability_factors": {
                "hormonal_coherence": coherence,
                "resource_stability": resource_stability,
            },
        }

    def _process_system_feedback(self, hormonal_state, resource_state):
        neural_energy = resource_state["energy_pools"]["neural"]
        cpu_load = self.performance_metrics["cpu_utilization"]
        self.feedback_loops["neural_feedback"] = neural_energy - cpu_load * 0.5
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        self.feedback_loops["hormonal_feedback"] = coherence - stress_level
        metabolic_energy = resource_state["energy_pools"]["metabolic"]
        energy_efficiency = self.performance_metrics["energy_efficiency"]
        self.feedback_loops["metabolic_feedback"] = (
            metabolic_energy + energy_efficiency
        ) / 2 - 0.5
        self.feedback_loops["circadian_feedback"] = (
            self._calculate_circadian_alignment()
        )

    def _apply_dynamic_adaptations(self, resource_state, performance_state, delta_time):
        adaptations_applied = {}
        if (
            resource_state["energy_efficiency"]
            < self.resource_management.critical_thresholds["energy_depletion"]
        ):
            self.adaptation_parameters["energy_conservation"] = min(
                1.0,
                self.adaptation_parameters["energy_conservation"] + 0.1 * delta_time,
            )
            for rate_type in self.resource_management.consumption_rates:
                self.resource_management.consumption_rates[rate_type] *= 0.98
            adaptations_applied["energy_conservation"] = True
        if self.performance_metrics["system_stability"] < 0.5:
            self.adaptation_parameters["stress_tolerance"] = min(
                1.0, self.adaptation_parameters["stress_tolerance"] + 0.05 * delta_time
            )
            adaptations_applied["stress_adaptation"] = True
        if self.performance_metrics["cpu_utilization"] > 0.9:
            self.adaptation_parameters["performance_optimization"] = min(
                1.0,
                self.adaptation_parameters["performance_optimization"]
                + 0.08 * delta_time,
            )
            adaptations_applied["performance_optimization"] = True
        avg_energy = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        if avg_energy < self.resource_management.critical_thresholds["recovery_needed"]:
            self.adaptation_parameters["recovery_speed"] = min(
                1.0, self.adaptation_parameters["recovery_speed"] + 0.06 * delta_time
            )
            for pool in self.resource_management.recovery_rates:
                self.resource_management.recovery_rates[pool] *= 1.02
            adaptations_applied["enhanced_recovery"] = True
        return {
            "adaptations_applied": adaptations_applied,
            "adaptation_parameters": self.adaptation_parameters.copy(),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(),
            "homeostasis_score": self._calculate_homeostasis_score(),
        }

    def _evaluate_physiological_state(
        self, resource_state, performance_state, hormonal_state
    ):
        energy_level = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        system_stability = self.performance_metrics["system_stability"]
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        health_score = (
            energy_level + system_stability + coherence
        ) / 3 - stress_level * 0.5
        if health_score >= 0.8 and stress_level < 0.3:
            new_state = FisiologicalState.OPTIMAL
        elif health_score >= 0.6 and stress_level < 0.5:
            new_state = FisiologicalState.HEALTHY
        elif health_score >= 0.4:
            new_state = FisiologicalState.RECOVERING
        elif health_score >= 0.2:
            new_state = FisiologicalState.STRESSED
        elif health_score >= 0.1:
            new_state = FisiologicalState.EXHAUSTED
        else:
            new_state = FisiologicalState.CRITICAL
        if new_state != self.physiological_state:
            self.physiological_state = new_state
        return new_state

    def _calculate_system_coherence(self) -> float:
        vital_coherence = self._calculate_vital_signs_coherence()
        resource_coherence = self._calculate_resource_coherence()
        performance_coherence = self._calculate_performance_coherence()
        feedback_coherence = self._calculate_feedback_coherence()
        return (
            vital_coherence
            + resource_coherence
            + performance_coherence
            + feedback_coherence
        ) / 4

    def _calculate_vital_signs_coherence(self) -> float:
        hr_normal = 0.8 if 60 <= self.physical_attributes.heart_rate <= 100 else 0.4
        bp_normal = (
            0.8
            if 90 <= self.physical_attributes.blood_pressure_systolic <= 140
            else 0.4
        )
        temp_normal = (
            0.9 if 36.0 <= self.physical_attributes.body_temperature <= 37.5 else 0.5
        )
        resp_normal = (
            0.8 if 12 <= self.physical_attributes.respiratory_rate <= 20 else 0.4
        )
        return (hr_normal + bp_normal + temp_normal + resp_normal) / 4

    def _calculate_resource_coherence(self) -> float:
        energy_values = list(self.resource_management.energy_pools.values())
        if not energy_values:
            return 0.0
        mean_energy = sum(energy_values) / len(energy_values)
        variance = sum((x - mean_energy) ** 2 for x in energy_values) / len(
            energy_values
        )
        max_possible_variance = 0.25
        return max(0.0, 1.0 - (variance / max_possible_variance))

    def _calculate_performance_coherence(self) -> float:
        metrics = self.performance_metrics
        values = [
            metrics["cpu_utilization"],
            metrics["memory_usage"],
            metrics["energy_efficiency"],
            metrics["thermal_regulation"],
            metrics["system_stability"],
        ]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _calculate_feedback_coherence(self) -> float:
        values = list(self.feedback_loops.values())
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _update_vital_signs_history(self) -> None:
        entry = {
            "timestamp": time.time(),
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure": {
                "systolic": self.physical_attributes.blood_pressure_systolic,
                "diastolic": self.physical_attributes.blood_pressure_diastolic,
            },
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "energy_pools": self.resource_management.energy_pools.copy(),
            "physiological_state": self.physiological_state.value,
        }
        self.vital_signs_history.append(entry)
        if len(self.vital_signs_history) > 100:
            self.vital_signs_history = self.vital_signs_history[-100:]

    def _get_physical_attributes_dict(self) -> dict[str, Any]:
        return {
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure_systolic": self.physical_attributes.blood_pressure_systolic,
            "blood_pressure_diastolic": self.physical_attributes.blood_pressure_diastolic,
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "glucose_level": self.physical_attributes.glucose_level,
            "atp_reserves": self.physical_attributes.atp_reserves,
            "oxygen_saturation": self.physical_attributes.oxygen_saturation,
            "hydration_level": self.physical_attributes.hydration_level,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "neural_conductivity": self.physical_attributes.neural_conductivity,
            "inflammatory_response": self.physical_attributes.inflammatory_response,
            "immune_system_strength": self.physical_attributes.immune_system_strength,
            "circadian_phase": self.physical_attributes.circadian_phase,
            "sleep_debt": self.physical_attributes.sleep_debt,
            "recovery_rate": self.physical_attributes.recovery_rate,
        }

    def _calculate_energy_efficiency(self) -> float:
        pools = self.resource_management.energy_pools.values()
        return float(min(1.0, sum(pools) / len(pools)))

    def _calculate_recovery_capacity(self, stress_level: float) -> float:
        base = sum(self.resource_management.recovery_rates.values()) / len(
            self.resource_management.recovery_rates
        )
        return base * (1.0 - stress_level * 0.5)

    def _evaluate_resource_status(self) -> str:
        pools = self.resource_management.energy_pools.values()
        if any(
            v < self.resource_management.critical_thresholds["energy_depletion"]
            for v in pools
        ):
            return "critical"
        elif all(
            v > self.resource_management.critical_thresholds["optimal_performance"]
            for v in pools
        ):
            return "optimal"
        elif any(
            v < self.resource_management.critical_thresholds["recovery_needed"]
            for v in pools
        ):
            return "recovery_needed"
        else:
            return "stable"

    def _calculate_circadian_alignment(self) -> float:
        phase = self.physical_attributes.circadian_phase
        return 1.0 - abs(phase - 0.5) * 2

    def _calculate_adaptation_effectiveness(self) -> float:
        params = self.adaptation_parameters.values()
        return float(min(1.0, sum(params) / len(params)))

    def _calculate_homeostasis_score(self) -> float:
        health_index = self.physical_attributes.get_overall_health_index()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (health_index * 0.6 + adaptation * 0.4)))

    def _calculate_system_resonance(self) -> float:
        coherence = self._calculate_system_coherence()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (coherence + adaptation) / 2))

    def get_system_diagnostics(self) -> dict[str, Any]:
        return {
            "current_state": {
                "physiological_state": self.physiological_state.value,
                "system_coherence": self._calculate_system_coherence(),
                "homeostasis_score": self._calculate_homeostasis_score(),
            },
            "vital_signs": self._get_physical_attributes_dict(),
            "resource_analysis": {
                "energy_pools": self.resource_management.energy_pools.copy(),
                "resource_status": self._evaluate_resource_status(),
                "energy_efficiency": self._calculate_energy_efficiency(),
            },
            "performance_metrics": self.performance_metrics.copy(),
            "adaptation_status": {
                "parameters": self.adaptation_parameters.copy(),
                "effectiveness": self._calculate_adaptation_effectiveness(),
            },
            "feedback_analysis": {
                "feedback_loops": self.feedback_loops.copy(),
                "system_resonance": self._calculate_system_resonance(),
            },
            "recent_events": (
                self.hardware_events_log[-10:] if self.hardware_events_log else []
            ),
            "recommendations": self._generate_hardware_recommendations(),
            "historical_trends": (
                self._analyze_vital_signs_trends()
                if len(self.vital_signs_history) > 5
                else None
            ),
        }

    def simulate_physical_stress_test(
        self, stress_type: str, intensity: float = 0.8, duration: float = 1.0
    ) -> dict[str, Any]:
        with self._lock:
            # Validate intensity
            if intensity < 0.0 or intensity > 1.0:
                raise ValueError("Intensity must be between 0.0 and 1.0")
            initial_state = self.get_system_diagnostics()
            # Apply stressors (same logic, preserved, defensive)
            try:
                if stress_type == "physical_exertion":
                    self.physical_attributes.heart_rate += intensity * 50
                    self.physical_attributes.respiratory_rate += intensity * 10
                    self.physical_attributes.body_temperature += intensity * 1.0
                    for pool in ("muscular", "cellular", "metabolic"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0, 1.0 - intensity * 0.3
                            )
                elif stress_type == "thermal_stress":
                    self.physical_attributes.body_temperature += intensity * 2.0
                    self.performance_metrics["thermal_regulation"] *= max(
                        0.0, 1.0 - intensity * 0.4
                    )
                elif stress_type == "cognitive_overload":
                    self.performance_metrics["cpu_utilization"] = min(
                        1.0,
                        self.performance_metrics["cpu_utilization"] + intensity * 0.4,
                    )
                    for pool in ("neural", "cognitive"):
                        if pool in self.resource_management.energy_pools:
                            self.resource_management.energy_pools[pool] *= max(
                                0.0,
                                1.0 - intensity * (0.4 if pool == "neural" else 0.5),
                            )
                elif stress_type == "sleep_deprivation":
                    self.physical_attributes.sleep_debt = min(
                        1.0, self.physical_attributes.sleep_debt + intensity * 0.6
                    )
                    self.physical_attributes.neural_conductivity *= max(
                        0.0, 1.0 - intensity * 0.3
                    )
            except Exception:
                logger.exception("simulate_physical_stress_test application failed")

            # Small simulated passage of time to let metrics update
            time.sleep(min(0.1, duration * 0.01))
            final_state = self.get_system_diagnostics()
            stress_response = self._analyze_stress_response(
                initial_state, final_state, intensity
            )
            return {
                "stress_type": stress_type,
                "intensity": intensity,
                "duration": duration,
                "initial_state": initial_state,
                "final_state": final_state,
                "stress_response": stress_response,
                "recovery_recommendations": self._generate_recovery_protocol(
                    stress_type, intensity
                ),
            }

    def record_hardware_experience(
        self,
        experience_id: str | None = None,
        hardware_state: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Records a hardware experience using the EVAMemoryManager (best-effort).
        Returns generated experience id or None.
        """
        with self._lock:
            if not self.eva_manager:
                logger.warning(
                    "EVAMemoryManager not available, cannot record hardware experience."
                )
                return None

            hardware_state = hardware_state or self.get_system_diagnostics()
            experience_id = (
                experience_id
                or f"hardware_experience_{abs(hash(str(hardware_state))) & 0xFFFFFFFF}"
            )

            experience_data = {
                "hardware_event": hardware_state,
                "physical_attributes": self._get_physical_attributes_dict(),
                "resource_state": hardware_state.get("resource_analysis", {}),
                "performance_metrics": copy.deepcopy(self.performance_metrics),
                "physiological_state": self.physiological_state.value,
                "detailed_physiological_state": {
                    "energy_level": self.detailed_physiological_state.energy_level,
                    "fatigue_level": self.detailed_physiological_state.fatigue_level,
                    "stress_level": self.detailed_physiological_state.stress_level,
                    "hunger_level": self.detailed_physiological_state.hunger_level,
                    "pain_level": self.detailed_physiological_state.pain_level,
                    "overall_health_score": self.detailed_physiological_state.calculate_overall_health_score(),
                    "primary_state": self.detailed_physiological_state.get_primary_physiological_state().value,
                    "feedback_modifiers": self.detailed_physiological_state.get_feedback_modifiers(),
                },
                "feedback_loops": copy.deepcopy(self.feedback_loops),
                "adaptation_state": hardware_state.get("adaptation_status", {}),
                "hardware_events": (
                    self.hardware_events_log[-10:] if self.hardware_events_log else []
                ),
                "system_coherence": self._calculate_system_coherence(),
                "recommendations": self._generate_hardware_recommendations(),
                "timestamp": time.time(),
            }

            # Best-effort, async-aware write
            asyncio.ensure_future(
                self._record_eva_async(
                    "hardware_experience", experience_data, experience_id
                )
            )
            logger.info(
                "Scheduled EVA record for hardware_experience: %s", experience_id
            )
            return experience_id

    def recall_hardware_experience(self, experience_id: str) -> dict[str, Any] | None:
        """
        Recalls a hardware experience from EVAMemoryManager. If backend is async and running
        event loop, schedule call and return None (best-effort).
        """
        if not self.eva_manager:
            logger.warning(
                "EVAMemoryManager not available, cannot recall hardware experience."
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not callable(recall):
                logger.debug("EVAMemoryManager.recall_experience is not callable")
                return None
            res = recall(entity_id=self.entity_id, experience_id=experience_id)
            if hasattr(res, "__await__"):
                # If loop running, schedule; otherwise run and return
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(res)
                    logger.debug(
                        "Scheduled async recall; returning None until available"
                    )
                    return None
                return loop.run_until_complete(res)
            return res
        except Exception:
            logger.exception("recall_hardware_experience failed")
            return None

    # --- Internal utilities -----------------------------------------
    async def _record_eva_async(
        self, event_type: str, data: dict[str, Any], experience_id: str | None = None
    ) -> None:
        """
        Helper to record to EVA in an async-safe manner. Uses asyncio.ensure_future
        to avoid blocking callers. Best-effort: logs failures.
        """
        if not self.eva_manager:
            return
        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                logger.debug("EVAMemoryManager.record_experience not callable")
                return
            res = recorder(
                entity_id=self.entity_id,
                event_type=event_type,
                data=data,
                experience_id=experience_id,
            )
            if hasattr(res, "__await__"):
                try:
                    await res
                except Exception:
                    logger.exception("Async EVA recorder failed for %s", event_type)
            else:
                # sync result already executed
                return
        except Exception:
            logger.exception("Failed to record EVA experience (async helper)")

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    # --- The rest of the implementation preserves original logic but wrapped with lock where needed ---
    def _update_vital_signs(self, hormonal_state, mental_load, delta_time):
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        adrenaline = hormonal_state.get("niveles_actuales", {}).get("adrenalina", 0.1)
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)

        base_hr = 70.0
        target_hr = (
            base_hr + stress_level * 30.0 + adrenaline * 40.0 + mental_load * 15.0
        )
        hr_change_rate = 10.0 * delta_time
        hr_diff = target_hr - self.physical_attributes.heart_rate
        hr_diff = max(-hr_change_rate, min(hr_diff, hr_change_rate))
        self.physical_attributes.heart_rate = max(
            50.0, min(180.0, self.physical_attributes.heart_rate + hr_diff)
        )

        self.physical_attributes.blood_pressure_systolic = max(
            90.0, min(180.0, 120.0 + stress_level * 20.0)
        )
        self.physical_attributes.blood_pressure_diastolic = max(
            60.0, min(120.0, 80.0 + stress_level * 15.0)
        )
        self.physical_attributes.body_temperature = max(
            36.0, min(39.0, 37.0 + stress_level * 0.5)
        )
        self.physical_attributes.respiratory_rate = max(
            10.0, min(30.0, 16.0 + stress_level * 8.0)
        )
        self.physical_attributes.muscle_tension = max(
            0.0, min(1.0, 0.3 + stress_level * 0.4 + cortisol * 0.3)
        )

    def _update_detailed_physiological_state(
        self, hormonal_state: dict[str, Any], mental_load: float, delta_time: float
    ) -> None:
        """
        Actualiza el estado fisiológico detallado basado en estado hormonal y carga mental.
        """
        # Extract hormonal information
        cortisol = hormonal_state.get("niveles_actuales", {}).get("cortisol", 0.2)
        dopamine = hormonal_state.get("niveles_actuales", {}).get("dopamina", 0.5)
        _serotonin = hormonal_state.get("niveles_actuales", {}).get("serotonina", 0.5)
        estado_general = hormonal_state.get("estado_hormonal", {})
        _stress_level = estado_general.get("estres", 0.3)

        # Update stress level based on cortisol and mental load
        stress_increase = (cortisol - 0.2) * 0.5 + mental_load * 0.3
        self.detailed_physiological_state.stress_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.stress_level
                + stress_increase * delta_time,
            ),
        )

        # Update energy level based on dopamine and stress
        energy_change = (
            dopamine - 0.5
        ) * 0.2 - self.detailed_physiological_state.stress_level * 0.1
        self.detailed_physiological_state.energy_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.energy_level
                + energy_change * delta_time,
            ),
        )

        # Update fatigue based on mental load and energy
        fatigue_increase = (
            mental_load * 0.2
            + (1.0 - self.detailed_physiological_state.energy_level) * 0.1
        )
        self.detailed_physiological_state.fatigue_level = min(
            1.0,
            max(
                0.0,
                self.detailed_physiological_state.fatigue_level
                + fatigue_increase * delta_time,
            ),
        )

        # Natural recovery and homeostasis
        if self.detailed_physiological_state.stress_level > 0:
            self.detailed_physiological_state.stress_level = max(
                0.0, self.detailed_physiological_state.stress_level - 0.05 * delta_time
            )

        if self.detailed_physiological_state.fatigue_level > 0:
            self.detailed_physiological_state.fatigue_level = max(
                0.0, self.detailed_physiological_state.fatigue_level - 0.03 * delta_time
            )

        # Update hunger over time (gradual increase)
        self.detailed_physiological_state.hunger_level = min(
            1.0, self.detailed_physiological_state.hunger_level + 0.02 * delta_time
        )

        # Update hydration (gradual decrease)
        self.detailed_physiological_state.hydration_level = max(
            0.0, self.detailed_physiological_state.hydration_level - 0.01 * delta_time
        )

        # Update sleep debt based on time and stress
        if self.physiological_state != FisiologicalState.SLEEPING:
            sleep_debt_increase = (
                0.03 * delta_time
                + self.detailed_physiological_state.stress_level * 0.01 * delta_time
            )
            self.detailed_physiological_state.sleep_debt = min(
                1.0, self.detailed_physiological_state.sleep_debt + sleep_debt_increase
            )

        # Update the primary physiological state based on detailed state
        new_primary_state = (
            self.detailed_physiological_state.get_primary_physiological_state()
        )
        if new_primary_state != self.physiological_state:
            self.physiological_state = new_primary_state

    def _generate_hardware_events(self, resource_state, performance_state):
        events = []
        if resource_state["resource_status"] == "critical":
            events.append("ALERT: Energy pools critically low")
        if self.physiological_state == FisiologicalState.CRITICAL:
            events.append("ALERT: System in critical physiological state")
        if self.performance_metrics["cpu_utilization"] > 0.95:
            events.append("WARNING: CPU utilization extremely high")
        if self.performance_metrics["thermal_regulation"] < 0.4:
            events.append("WARNING: Thermal regulation compromised")
        if events:
            self.hardware_events_log.extend(events)
        return events

    def _generate_hardware_recommendations(self) -> list[str]:
        recs = []
        if self.physiological_state == FisiologicalState.CRITICAL:
            recs.append("Iniciar protocolo de emergencia y recuperación intensiva")
        elif self.physiological_state == FisiologicalState.EXHAUSTED:
            recs.append("Recomendar descanso prolongado y monitoreo de recursos")
        elif self.physiological_state == FisiologicalState.STRESSED:
            recs.append("Aplicar técnicas de reducción de estrés y optimizar recursos")
        if self.performance_metrics["thermal_regulation"] < 0.5:
            recs.append("Ajustar temperatura ambiental y aumentar hidratación")
        if self.performance_metrics["cpu_utilization"] > 0.9:
            recs.append("Reducir carga cognitiva y optimizar procesos")
        return recs

    def _analyze_vital_signs_trends(self, window_size: int = 10) -> dict[str, str]:
        if len(self.vital_signs_history) < window_size:
            return {"status": "insufficient_data"}
        recent_history = self.vital_signs_history[-window_size:]
        trends = {}
        hr_values = [entry["heart_rate"] for entry in recent_history]
        trends["heart_rate"] = self._calculate_trend_direction(hr_values)
        temp_values = [entry["body_temperature"] for entry in recent_history]
        trends["body_temperature"] = self._calculate_trend_direction(temp_values)
        tension_values = [entry["muscle_tension"] for entry in recent_history]
        trends["muscle_tension"] = self._calculate_trend_direction(tension_values)
        energy_averages = [
            sum(entry["energy_pools"].values()) / len(entry["energy_pools"])
            for entry in recent_history
        ]
        trends["energy_level"] = self._calculate_trend_direction(energy_averages)
        return trends

    def _calculate_trend_direction(self, values: list[float]) -> str:
        return calculate_trend(
            values,
            threshold=getattr(self.config, "HARDWARE_TREND_THRESHOLD", 0.02),
            is_percentage=True,
            config=self.config,
        )

    def _manage_energy_resources(self, hormonal_state, mental_load, delta_time):
        estado_general = hormonal_state.get("estado_hormonal", {})
        stress_level = estado_general.get("estres", 0.3)
        consumo_total = dict.fromkeys(self.resource_management.energy_pools, 0.0)
        for pool in consumo_total:
            consumo_total[pool] += (
                self.resource_management.consumption_rates["rest"] * delta_time
            )
        cognitive_consumption = (
            mental_load
            * self.resource_management.consumption_rates["thinking"]
            * delta_time
        )
        consumo_total["neural"] += cognitive_consumption
        consumo_total["cognitive"] += cognitive_consumption * 1.5
        stress_consumption = (
            stress_level
            * self.resource_management.consumption_rates["stress"]
            * delta_time
        )
        for pool in consumo_total:
            consumo_total[pool] += stress_consumption
        for pool, consumo in consumo_total.items():
            self.resource_management.energy_pools[pool] = max(
                0.0, self.resource_management.energy_pools[pool] - consumo
            )
        for pool, tasa_recuperacion in self.resource_management.recovery_rates.items():
            recovery = tasa_recuperacion * delta_time * (1.0 - stress_level * 0.5)
            self.resource_management.energy_pools[pool] = min(
                1.0, self.resource_management.energy_pools[pool] + recovery
            )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        self.physical_attributes.atp_reserves = self.resource_management.energy_pools[
            "cellular"
        ]
        self.physical_attributes.glucose_level = min(1.0, avg_energy * 1.2)
        resource_status = self._evaluate_resource_status()
        return {
            "energy_pools": self.resource_management.energy_pools.copy(),
            "consumption_rates": consumo_total,
            "resource_status": resource_status,
            "energy_efficiency": self._calculate_energy_efficiency(),
            "recovery_capacity": self._calculate_recovery_capacity(stress_level),
        }

    def _update_performance_metrics(
        self, hormonal_state, mental_load, environmental_factors, delta_time
    ):
        estado_general = hormonal_state.get("estado_hormonal", {})
        base_cpu = 0.4
        mental_cpu = (
            mental_load * 0.5
        )  # Changed from 0.4 to 0.5 for consistency with legacy
        stress_cpu = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["cpu_utilization"] = min(
            1.0, base_cpu + mental_cpu + stress_cpu
        )
        cognitive_memory = mental_load * 0.5
        emotional_memory = abs(estado_general.get("bienestar", 0.5) - 0.5) * 0.3
        self.performance_metrics["memory_usage"] = min(
            1.0, 0.2 + cognitive_memory + emotional_memory
        )
        avg_energy = sum(self.resource_management.energy_pools.values()) / len(
            self.resource_management.energy_pools
        )
        stress_penalty = estado_general.get("estres", 0.3) * 0.3
        self.performance_metrics["energy_efficiency"] = max(
            0.0, avg_energy - stress_penalty
        )
        temp_optimal = abs(self.physical_attributes.body_temperature - 37.0) < 0.5
        stress_thermal_impact = estado_general.get("estres", 0.3) * 0.2
        self.performance_metrics["thermal_regulation"] = max(
            0.0, (0.9 if temp_optimal else 0.6) - stress_thermal_impact
        )
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        resource_stability = min(self.resource_management.energy_pools.values())
        self.performance_metrics["system_stability"] = (
            coherence + resource_stability
        ) / 2
        return {
            "cpu_usage_breakdown": {
                "base": base_cpu,
                "mental": mental_cpu,
                "stress": stress_cpu,
            },
            "thermal_state": {
                "temperature": self.physical_attributes.body_temperature,
                "regulation_efficiency": self.performance_metrics["thermal_regulation"],
            },
            "stability_factors": {
                "hormonal_coherence": coherence,
                "resource_stability": resource_stability,
            },
        }

    def _process_system_feedback(self, hormonal_state, resource_state):
        neural_energy = resource_state["energy_pools"]["neural"]
        cpu_load = self.performance_metrics["cpu_utilization"]
        self.feedback_loops["neural_feedback"] = neural_energy - cpu_load * 0.5
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        self.feedback_loops["hormonal_feedback"] = coherence - stress_level
        metabolic_energy = resource_state["energy_pools"]["metabolic"]
        energy_efficiency = self.performance_metrics["energy_efficiency"]
        self.feedback_loops["metabolic_feedback"] = (
            metabolic_energy + energy_efficiency
        ) / 2 - 0.5
        self.feedback_loops["circadian_feedback"] = (
            self._calculate_circadian_alignment()
        )

    def _apply_dynamic_adaptations(self, resource_state, performance_state, delta_time):
        adaptations_applied = {}
        if (
            resource_state["energy_efficiency"]
            < self.resource_management.critical_thresholds["energy_depletion"]
        ):
            self.adaptation_parameters["energy_conservation"] = min(
                1.0,
                self.adaptation_parameters["energy_conservation"] + 0.1 * delta_time,
            )
            for rate_type in self.resource_management.consumption_rates:
                self.resource_management.consumption_rates[rate_type] *= 0.98
            adaptations_applied["energy_conservation"] = True
        if self.performance_metrics["system_stability"] < 0.5:
            self.adaptation_parameters["stress_tolerance"] = min(
                1.0, self.adaptation_parameters["stress_tolerance"] + 0.05 * delta_time
            )
            adaptations_applied["stress_adaptation"] = True
        if self.performance_metrics["cpu_utilization"] > 0.9:
            self.adaptation_parameters["performance_optimization"] = min(
                1.0,
                self.adaptation_parameters["performance_optimization"]
                + 0.08 * delta_time,
            )
            adaptations_applied["performance_optimization"] = True
        avg_energy = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        if avg_energy < self.resource_management.critical_thresholds["recovery_needed"]:
            self.adaptation_parameters["recovery_speed"] = min(
                1.0, self.adaptation_parameters["recovery_speed"] + 0.06 * delta_time
            )
            for pool in self.resource_management.recovery_rates:
                self.resource_management.recovery_rates[pool] *= 1.02
            adaptations_applied["enhanced_recovery"] = True
        return {
            "adaptations_applied": adaptations_applied,
            "adaptation_parameters": self.adaptation_parameters.copy(),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(),
            "homeostasis_score": self._calculate_homeostasis_score(),
        }

    def _evaluate_physiological_state(
        self, resource_state, performance_state, hormonal_state
    ):
        energy_level = sum(resource_state["energy_pools"].values()) / len(
            resource_state["energy_pools"]
        )
        system_stability = self.performance_metrics["system_stability"]
        stress_level = hormonal_state.get("estado_hormonal", {}).get("estres", 0.3)
        coherence = hormonal_state.get("coherencia_sistema", 0.7)
        health_score = (
            energy_level + system_stability + coherence
        ) / 3 - stress_level * 0.5
        if health_score >= 0.8 and stress_level < 0.3:
            new_state = FisiologicalState.OPTIMAL
        elif health_score >= 0.6 and stress_level < 0.5:
            new_state = FisiologicalState.HEALTHY
        elif health_score >= 0.4:
            new_state = FisiologicalState.RECOVERING
        elif health_score >= 0.2:
            new_state = FisiologicalState.STRESSED
        elif health_score >= 0.1:
            new_state = FisiologicalState.EXHAUSTED
        else:
            new_state = FisiologicalState.CRITICAL
        if new_state != self.physiological_state:
            self.physiological_state = new_state
        return new_state

    def _calculate_system_coherence(self) -> float:
        vital_coherence = self._calculate_vital_signs_coherence()
        resource_coherence = self._calculate_resource_coherence()
        performance_coherence = self._calculate_performance_coherence()
        feedback_coherence = self._calculate_feedback_coherence()
        return (
            vital_coherence
            + resource_coherence
            + performance_coherence
            + feedback_coherence
        ) / 4

    def _calculate_vital_signs_coherence(self) -> float:
        hr_normal = 0.8 if 60 <= self.physical_attributes.heart_rate <= 100 else 0.4
        bp_normal = (
            0.8
            if 90 <= self.physical_attributes.blood_pressure_systolic <= 140
            else 0.4
        )
        temp_normal = (
            0.9 if 36.0 <= self.physical_attributes.body_temperature <= 37.5 else 0.5
        )
        resp_normal = (
            0.8 if 12 <= self.physical_attributes.respiratory_rate <= 20 else 0.4
        )
        return (hr_normal + bp_normal + temp_normal + resp_normal) / 4

    def _calculate_resource_coherence(self) -> float:
        energy_values = list(self.resource_management.energy_pools.values())
        if not energy_values:
            return 0.0
        mean_energy = sum(energy_values) / len(energy_values)
        variance = sum((x - mean_energy) ** 2 for x in energy_values) / len(
            energy_values
        )
        max_possible_variance = 0.25
        return max(0.0, 1.0 - (variance / max_possible_variance))

    def _calculate_performance_coherence(self) -> float:
        metrics = self.performance_metrics
        values = [
            metrics["cpu_utilization"],
            metrics["memory_usage"],
            metrics["energy_efficiency"],
            metrics["thermal_regulation"],
            metrics["system_stability"],
        ]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _calculate_feedback_coherence(self) -> float:
        values = list(self.feedback_loops.values())
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return max(0.0, 1.0 - variance / 0.25)

    def _update_vital_signs_history(self) -> None:
        entry = {
            "timestamp": time.time(),
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure": {
                "systolic": self.physical_attributes.blood_pressure_systolic,
                "diastolic": self.physical_attributes.blood_pressure_diastolic,
            },
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "energy_pools": self.resource_management.energy_pools.copy(),
            "physiological_state": self.physiological_state.value,
        }
        self.vital_signs_history.append(entry)
        if len(self.vital_signs_history) > 100:
            self.vital_signs_history = self.vital_signs_history[-100:]

    def _get_physical_attributes_dict(self) -> dict[str, Any]:
        return {
            "heart_rate": self.physical_attributes.heart_rate,
            "blood_pressure_systolic": self.physical_attributes.blood_pressure_systolic,
            "blood_pressure_diastolic": self.physical_attributes.blood_pressure_diastolic,
            "body_temperature": self.physical_attributes.body_temperature,
            "respiratory_rate": self.physical_attributes.respiratory_rate,
            "glucose_level": self.physical_attributes.glucose_level,
            "atp_reserves": self.physical_attributes.atp_reserves,
            "oxygen_saturation": self.physical_attributes.oxygen_saturation,
            "hydration_level": self.physical_attributes.hydration_level,
            "muscle_tension": self.physical_attributes.muscle_tension,
            "neural_conductivity": self.physical_attributes.neural_conductivity,
            "inflammatory_response": self.physical_attributes.inflammatory_response,
            "immune_system_strength": self.physical_attributes.immune_system_strength,
            "circadian_phase": self.physical_attributes.circadian_phase,
            "sleep_debt": self.physical_attributes.sleep_debt,
            "recovery_rate": self.physical_attributes.recovery_rate,
        }

    def _calculate_energy_efficiency(self) -> float:
        pools = self.resource_management.energy_pools.values()
        return float(min(1.0, sum(pools) / len(pools)))

    def _calculate_recovery_capacity(self, stress_level: float) -> float:
        base = sum(self.resource_management.recovery_rates.values()) / len(
            self.resource_management.recovery_rates
        )
        return base * (1.0 - stress_level * 0.5)

    def _evaluate_resource_status(self) -> str:
        pools = self.resource_management.energy_pools.values()
        if any(
            v < self.resource_management.critical_thresholds["energy_depletion"]
            for v in pools
        ):
            return "critical"
        elif all(
            v > self.resource_management.critical_thresholds["optimal_performance"]
            for v in pools
        ):
            return "optimal"
        elif any(
            v < self.resource_management.critical_thresholds["recovery_needed"]
            for v in pools
        ):
            return "recovery_needed"
        else:
            return "stable"

    def _calculate_circadian_alignment(self) -> float:
        phase = self.physical_attributes.circadian_phase
        return 1.0 - abs(phase - 0.5) * 2

    def _calculate_adaptation_effectiveness(self) -> float:
        params = self.adaptation_parameters.values()
        return float(min(1.0, sum(params) / len(params)))

    def _calculate_homeostasis_score(self) -> float:
        health_index = self.physical_attributes.get_overall_health_index()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (health_index * 0.6 + adaptation * 0.4)))

    def _calculate_system_resonance(self) -> float:
        coherence = self._calculate_system_coherence()
        adaptation = self._calculate_adaptation_effectiveness()
        return float(min(1.0, (coherence + adaptation) / 2))

    def get_system_diagnostics(self) -> dict[str, Any]:
        return {
            "current_state": {
                "physiological_state": self.physiological_state.value,
                "system_coherence": self._calculate_system_coherence(),
                "homeostasis_score": self._calculate_homeostasis_score(),
            },
            "vital_signs": self._get_physical_attributes_dict(),
            "resource_analysis": {
                "energy_pools": self.resource_management.energy_pools.copy(),
                "resource_status": self._evaluate_resource_status(),
                "energy_efficiency": self._calculate_energy_efficiency(),
            },
            "performance_metrics": self.performance_metrics.copy(),
            "adaptation_status": {
                "parameters": self.adaptation_parameters.copy(),
                "effectiveness": self._calculate_adaptation_effectiveness(),
            },
            "feedback_analysis": {
                "feedback_loops": self.feedback_loops.copy(),
                "system_resonance": self._calculate_system_resonance(),
            },
            "recent_events": (
                self.hardware_events_log[-10:] if self.hardware_events_log else []
            ),
            "recommendations": self._generate_hardware_recommendations(),
            "historical_trends": (
                self._analyze_vital_signs_trends()
                if len(self.vital_signs_history) > 5
                else None
            ),
        }
