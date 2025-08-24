"""
Biological Body Core
====================

This module defines the BiologicalBody class, the integrated core of the entity's
physical existence. It orchestrates the interaction between the HardwareAbstractionLayer
and the HormonalSystem.

This file has been completed by filling in the placeholder logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.cuerpo.hardware import HardwareAbstractionLayer as HardwareLayer
from crisalida_lib.ADAM.cuerpo.hormonal_system import SistemaHormonal as HormonalSystem
from crisalida_lib.ADAM.enums import FisiologicalState
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager

logger = logging.getLogger(__name__)


class BiologicalBody:
    """
    The core of the biological self, orchestrating hardware and hormonal systems.

    - Defensive calls to subsystems (supports multiple method names used across versions).
    - Periodic EVA snapshotting (best-effort, async-aware).
    - Bounded history and clear snapshot API.
    """

    EVA_SNAPSHOT_PERIOD = 30.0  # seconds between best-effort snapshots to EVA

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config = config
        self.eva_manager = eva_manager
        self.entity_id = entity_id

        # instantiate subsystems
        try:
            self.sistema_hormonal = HormonalSystem(config=self.config)
        except Exception:
            logger.exception("Failed to initialize SistemaHormonal; using minimal stub")
            self.sistema_hormonal = HormonalSystem()  # type: ignore

        try:
            self.hardware_layer = HardwareLayer(config=self.config)
        except Exception:
            logger.exception(
                "Failed to initialize HardwareAbstractionLayer; using minimal stub"
            )
            self.hardware_layer = HardwareLayer()  # type: ignore

        self.integration_coherence = 0.7
        self.last_integration_update = time.time()
        self.integration_history: list[dict[str, Any]] = []
        self._last_eva_snapshot = 0.0

    def update(
        self, dt: float, external_inputs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """
        Advance biological subsystems by dt seconds and refresh integration state.

        external_inputs: optional dict with sensor cues, environment, or qualia influences.
        """
        external_inputs = external_inputs or {}

        # 1. Update hormonal system (support different API versions)
        try:
            if hasattr(self.sistema_hormonal, "tick"):
                self.sistema_hormonal.tick(dt, **external_inputs)
            elif hasattr(self.sistema_hormonal, "update_internal_states"):
                self.sistema_hormonal.update_internal_states(dt, **external_inputs)
            elif hasattr(self.sistema_hormonal, "update"):
                # some versions accept (event, intensity) but also support dt kw
                try:
                    self.sistema_hormonal.update(dt=dt, **external_inputs)  # type: ignore
                except TypeError:
                    # fallback: no dt param
                    self.sistema_hormonal.update(**external_inputs)  # type: ignore
            else:
                logger.debug(
                    "No known update method on SistemaHormonal; skipping hormonal tick"
                )
        except Exception:
            logger.exception("Hormonal system update failed")

        # 2. Update hardware layer (physical/body resources)
        try:
            if hasattr(self.hardware_layer, "tick"):
                self.hardware_layer.tick(dt, **external_inputs)
            elif hasattr(self.hardware_layer, "update_internal_states"):
                self.hardware_layer.update_internal_states(dt, **external_inputs)
            elif hasattr(self.hardware_layer, "update_physical_state"):
                # best-effort: call with current hormonal snapshot
                hormonal_snapshot = {
                    "niveles": getattr(self.sistema_hormonal, "niveles", {}),
                    "estado_hormonal": getattr(
                        self.sistema_hormonal, "_calculate_hormonal_state", lambda: {}
                    )(),
                }
                self.hardware_layer.update_physical_state(
                    hormonal_snapshot, **external_inputs
                )
            else:
                logger.debug(
                    "No known update method on HardwareAbstractionLayer; skipping hardware tick"
                )
        except Exception:
            logger.exception("Hardware layer update failed")

        # 3. Recompute integration coherence and record history
        try:
            hormonal_snapshot = getattr(
                self.sistema_hormonal, "_calculate_hormonal_state", lambda: {}
            )()
            hardware_status = {}
            try:
                hardware_status = self.hardware_layer._evaluate_resource_status()
            except Exception:
                hardware_status = getattr(
                    self.hardware_layer, "performance_metrics", {}
                ).copy()
            integration_state = self._calculate_integration_state(
                {
                    "estado_hormonal": hormonal_snapshot,
                    "coherencia_sistema": getattr(
                        self.sistema_hormonal,
                        "_calculate_system_coherence",
                        lambda: 0.5,
                    )(),
                },
                {
                    "system_coherence": hardware_status.get(
                        "system_coherence",
                        getattr(self.hardware_layer, "performance_metrics", {}).get(
                            "system_coherence", 0.5
                        ),
                    ),
                    "performance_metrics": getattr(
                        self.hardware_layer, "performance_metrics", {}
                    ),
                },
            )
            self._update_integration_history({}, {}, integration_state)
        except Exception:
            logger.exception("Integration recompute failed")

        # 4. Best-effort EVA snapshot (periodic)
        try:
            now = time.time()
            if (
                self.eva_manager
                and (now - self._last_eva_snapshot) > self.EVA_SNAPSHOT_PERIOD
            ):
                snapshot = self.get_comprehensive_biological_state()
                rec = getattr(self.eva_manager, "record_experience", None)
                if callable(rec):
                    experience_id = f"biological_snapshot:{self.entity_id}:{int(now)}"
                    res = rec(
                        entity_id=self.entity_id,
                        event_type="biological_snapshot",
                        data=snapshot,
                        experience_id=experience_id,
                    )
                    # if coroutine, schedule best-effort
                    if hasattr(res, "__await__"):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(res)
                            else:
                                loop.run_until_complete(res)
                        except Exception:
                            logger.debug(
                                "Failed to schedule EVA snapshot", exc_info=True
                            )
                    self._last_eva_snapshot = now
        except Exception:
            logger.exception("EVA snapshot failed (non-fatal)")

    def inject_qualia_influence(self, qualia_state: dict[str, Any]) -> dict[str, Any]:
        """
        Inject qualia influence into hormonal and hardware subsystems and return integration summary.
        """
        hormonal_qualia_response = self.sistema_hormonal.inject_qualia_event(
            qualia_state
        )
        hardware_response = self.hardware_layer.update_physical_state(
            {
                "niveles_actuales": getattr(self.sistema_hormonal, "niveles", {}),
                "estado_hormonal": getattr(
                    self.sistema_hormonal, "_calculate_hormonal_state", lambda: {}
                )(),
                "coherencia_sistema": getattr(
                    self.sistema_hormonal, "_calculate_system_coherence", lambda: 0.5
                )(),
            },
            qualia_state.get("cognitive_complexity", 0.5),
            {"qualia_influence": True},
        )
        total_mind_body_coherence = (
            float(hormonal_qualia_response.get("coherencia_mente_cuerpo", 0.5))
            + float(hardware_response.get("system_coherence", 0.5))
        ) / 2.0
        return {
            "hormonal_qualia_response": hormonal_qualia_response,
            "hardware_response": hardware_response,
            "total_mind_body_coherence": total_mind_body_coherence,
            "qualia_integration_success": total_mind_body_coherence > 0.6,
        }

    def process_event(
        self,
        event: str,
        intensity: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Processes a biological event, triggers hormonal and hardware responses,
        and records it to EVA if a manager is present.
        """
        context = context or {}
        hormonal_response = self.sistema_hormonal.update(event, intensity)
        hardware_response = self.hardware_layer.update_physical_state(
            hormonal_response,
            context.get("mental_load", 0.5),
            context.get("environmental_factors", {}),
        )
        integration_state = self._calculate_integration_state(
            hormonal_response, hardware_response
        )
        self._update_integration_history(
            hormonal_response, hardware_response, integration_state
        )

        # Prepare comprehensive return data
        return_data: dict[str, Any] = {
            "hormonal_response": hormonal_response,
            "hardware_response": hardware_response,
            "integration_state": integration_state,
            "overall_biological_coherence": self.integration_coherence,
            "system_recommendations": self._generate_integrated_recommendations(
                hormonal_response, hardware_response
            ),
        }

        # EVA integration (best-effort)
        if self.eva_manager:
            comprehensive_state = self.get_comprehensive_biological_state()
            experience_id = (
                context.get("experience_id")
                or f"biological_event_{abs(hash(str(comprehensive_state))) & 0xFFFFFFFF}"
            )
            try:
                rec = getattr(self.eva_manager, "record_experience", None)
                if callable(rec):
                    res = rec(
                        entity_id=self.entity_id,
                        event_type="biological_event",
                        data={
                            "summary": {
                                "event": event,
                                "intensity": intensity,
                                "context": context,
                            },
                            "state": comprehensive_state,
                            "timestamp": time.time(),
                        },
                        experience_id=experience_id,
                    )
                    if hasattr(res, "__await__"):
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(res)
                        else:
                            loop.run_until_complete(res)
            except Exception:
                logger.exception("Failed to record biological event to EVA (non-fatal)")

        return return_data

    def _calculate_integration_state(
        self, hormonal_response: dict[str, Any], hardware_response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compute a bounded integration state using available snapshots from subsystems.
        """
        hormonal_coherence = float(hormonal_response.get("coherencia_sistema", 0.5))
        hardware_coherence = float(hardware_response.get("system_coherence", 0.5))
        hormonal_stress = float(
            hormonal_response.get("estado_hormonal", {}).get("estres", 0.3)
        )
        hardware_stability = float(
            hardware_response.get("performance_metrics", {}).get(
                "system_stability", 0.7
            )
        )
        hardware_stress = 1.0 - hardware_stability
        stress_synchronization = max(0.0, 1.0 - abs(hormonal_stress - hardware_stress))
        hormonal_energy_state = float(
            hormonal_response.get("estado_hormonal", {}).get("balance_general", 0.5)
        )
        hardware_energy_state = float(
            hardware_response.get("resource_state", {}).get("energy_efficiency", 0.5)
        )
        energy_coherence = max(
            0.0, 1.0 - abs(hormonal_energy_state - hardware_energy_state)
        )
        # Weighted coherence aggregate
        self.integration_coherence = max(
            0.0,
            min(
                1.0,
                hormonal_coherence * 0.3
                + hardware_coherence * 0.3
                + stress_synchronization * 0.2
                + energy_coherence * 0.2,
            ),
        )
        return {
            "integration_coherence": self.integration_coherence,
            "hormonal_coherence": hormonal_coherence,
            "hardware_coherence": hardware_coherence,
            "stress_synchronization": stress_synchronization,
            "energy_coherence": energy_coherence,
            "integration_quality": self._classify_integration_quality(
                self.integration_coherence
            ),
        }

    def _classify_integration_quality(self, coherence: float) -> str:
        if coherence >= 0.8:
            return "excellent"
        elif coherence >= 0.65:
            return "good"
        elif coherence >= 0.5:
            return "adequate"
        elif coherence >= 0.35:
            return "poor"
        else:
            return "critical"

    def _update_integration_history(
        self,
        hormonal_response: dict[str, Any],
        hardware_response: dict[str, Any],
        integration_state: dict[str, Any],
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "integration_coherence": self.integration_coherence,
            "hormonal_state_summary": {
                "coherence": hormonal_response.get("coherencia_sistema", 0.5),
                "stress_level": hormonal_response.get("estado_hormonal", {}).get(
                    "estres", 0.3
                ),
                "balance": hormonal_response.get("estado_hormonal", {}).get(
                    "balance_general", 0.5
                ),
            },
            "hardware_state_summary": {
                "coherence": hardware_response.get("system_coherence", 0.5),
                "energy_efficiency": hardware_response.get("resource_state", {}).get(
                    "energy_efficiency", 0.5
                ),
                "physiological_state": hardware_response.get(
                    "physiological_state", "healthy"
                ),
            },
            "integration_quality": integration_state.get(
                "integration_quality", "unknown"
            ),
        }
        self.integration_history.append(entry)
        if len(self.integration_history) > 50:
            self.integration_history = self.integration_history[-50:]

    def _generate_integrated_recommendations(
        self, hormonal_response: dict[str, Any], hardware_response: dict[str, Any]
    ) -> list[str]:
        recommendations = []
        recommendations.extend(hormonal_response.get("recomendaciones", []))
        recommendations.extend(hardware_response.get("recommendations", []))
        if self.integration_coherence < 0.5:
            recommendations.append("Implementar prácticas de integración mente-cuerpo")
            recommendations.append(
                "Técnicas de biofeedback para mejorar sincronización"
            )
        if self.integration_coherence < 0.35:
            recommendations.append(
                "Considerar intervención terapéutica para desconexión mente-cuerpo"
            )
        return list(set(recommendations))

    def _analyze_integration_stability(self) -> dict[str, Any]:
        if len(self.integration_history) < 5:
            return {"status": "insufficient_data"}
        recent_coherence = [
            entry["integration_coherence"] for entry in self.integration_history[-10:]
        ]
        mean_coherence = sum(recent_coherence) / len(recent_coherence)
        variance = sum((c - mean_coherence) ** 2 for c in recent_coherence) / len(
            recent_coherence
        )
        if variance < 0.01:
            stability = "very_stable"
        elif variance < 0.05:
            stability = "stable"
        elif variance < 0.1:
            stability = "moderately_unstable"
        else:
            stability = "unstable"
        return {
            "stability_level": stability,
            "mean_coherence": mean_coherence,
            "variance": variance,
            "trend": self._calculate_trend_direction(recent_coherence),
        }

    def _calculate_trend_direction(self, values: list[float]) -> str:
        if len(values) < 2:
            return "stable"
        delta = values[-1] - values[0]
        if abs(delta) < 0.02:
            return "stable"
        elif delta > 0.02:
            return "improving"
        else:
            return "declining"

    def _calculate_synchronization_metrics(self) -> dict[str, float]:
        hormonal_state = self.sistema_hormonal._calculate_hormonal_state()
        hormonal_stress = hormonal_state.get("estres", 0.3)
        hardware_stress = 1.0 - self.hardware_layer.performance_metrics.get(
            "system_stability", 0.7
        )
        stress_sync = 1.0 - abs(hormonal_stress - hardware_stress)
        hormonal_energy = hormonal_state.get("balance_general", 0.5)
        hardware_energy = sum(
            self.hardware_layer.resource_management.energy_pools.values()
        ) / len(self.hardware_layer.resource_management.energy_pools)
        energy_sync = 1.0 - abs(hormonal_energy - hardware_energy)
        hormonal_arousal = hormonal_state.get("alerta", 0.3)
        hardware_arousal = self.hardware_layer.performance_metrics.get(
            "cpu_utilization", 0.4
        )
        arousal_sync = 1.0 - abs(hormonal_arousal - hardware_arousal)
        return {
            "stress_synchronization": stress_sync,
            "energy_synchronization": energy_sync,
            "arousal_synchronization": arousal_sync,
            "overall_synchronization": (stress_sync + energy_sync + arousal_sync) / 3,
        }

    def _calculate_overall_health_score(self) -> float:
        hormonal_health = self.sistema_hormonal._calculate_system_coherence()
        hardware_health = self.hardware_layer._calculate_system_coherence()
        integration_health = self.integration_coherence
        penalties = 0.0
        imbalances = self.sistema_hormonal._detect_imbalances()
        severe_imbalances = sum(
            1
            for imb in imbalances.values()
            if isinstance(imb, dict) and imb.get("severidad", 0) > 0.7
        )
        penalties += severe_imbalances * 0.1
        if self.hardware_layer.physiological_state in [
            FisiologicalState.CRITICAL,
            FisiologicalState.EXHAUSTED,
        ]:
            penalties += 0.2
        health_score = (
            (hormonal_health + hardware_health + integration_health) / 3
        ) - penalties
        return max(0.0, min(1.0, health_score))

    def _generate_critical_alerts(self) -> list[dict[str, Any]]:
        alerts = []
        imbalances = self.sistema_hormonal._detect_imbalances()
        for nt, imbalance in imbalances.items():
            if isinstance(imbalance, dict) and imbalance.get("severidad", 0) > 0.8:
                alerts.append(
                    {
                        "type": "critical_hormonal_imbalance",
                        "system": "hormonal",
                        "neurotransmitter": nt,
                        "severity": imbalance["severidad"],
                        "message": f"Desequilibrio crítico en {nt}: {imbalance['tipo']}",
                    }
                )
        if self.hardware_layer.physiological_state == FisiologicalState.CRITICAL:
            alerts.append(
                {
                    "type": "critical_physiological_state",
                    "system": "hardware",
                    "severity": 1.0,
                    "message": "Sistema fisiológico en estado crítico",
                }
            )
        if self.integration_coherence < 0.3:
            alerts.append(
                {
                    "type": "severe_mind_body_disconnection",
                    "system": "integration",
                    "severity": 1.0 - self.integration_coherence,
                    "message": "Desconexión severa mente-cuerpo detectada",
                }
            )
        for pool, level in self.hardware_layer.resource_management.energy_pools.items():
            if level < 0.15:
                alerts.append(
                    {
                        "type": "critical_energy_depletion",
                        "system": "hardware",
                        "resource_pool": pool,
                        "severity": (0.15 - level) / 0.15,
                        "message": f"Agotamiento crítico del pool energético {pool}",
                    }
                )
        return alerts

    def _identify_optimization_opportunities(self) -> list[dict[str, Any]]:
        opportunities = []
        hormonal_state = self.sistema_hormonal._calculate_hormonal_state()
        if hormonal_state.get("bienestar", 0.5) < 0.6:
            opportunities.append(
                {
                    "type": "hormonal_optimization",
                    "target": "bienestar",
                    "potential_improvement": 0.6 - hormonal_state["bienestar"],
                    "recommendation": "Optimizar niveles de serotonina y oxitocina",
                }
            )
        if hormonal_state.get("motivacion", 0.5) < 0.7:
            opportunities.append(
                {
                    "type": "hormonal_optimization",
                    "target": "motivacion",
                    "potential_improvement": 0.7 - hormonal_state["motivacion"],
                    "recommendation": "Incrementar actividad dopaminérgica",
                }
            )
        if self.hardware_layer.performance_metrics["energy_efficiency"] < 0.75:
            opportunities.append(
                {
                    "type": "hardware_optimization",
                    "target": "energy_efficiency",
                    "potential_improvement": 0.75
                    - self.hardware_layer.performance_metrics["energy_efficiency"],
                    "recommendation": "Optimizar patrones de consumo energético",
                }
            )
        if self.integration_coherence < 0.8:
            opportunities.append(
                {
                    "type": "integration_optimization",
                    "target": "mind_body_coherence",
                    "potential_improvement": 0.8 - self.integration_coherence,
                    "recommendation": "Implementar prácticas de biofeedback y mindfulness",
                }
            )
        return opportunities

    def _assess_recovery_status(self) -> dict[str, Any]:
        recovery_capacity = self.hardware_layer._calculate_recovery_capacity(
            self.sistema_hormonal._calculate_hormonal_state().get("estres", 0.3)
        )
        energy_depletion = 1.0 - (
            sum(self.hardware_layer.resource_management.energy_pools.values())
            / len(self.hardware_layer.resource_management.energy_pools)
        )
        stress_accumulation = self.sistema_hormonal._calculate_hormonal_state().get(
            "estres", 0.3
        )
        recovery_need = (energy_depletion + stress_accumulation) / 2
        if recovery_need < 0.3:
            recovery_status = "minimal_recovery_needed"
        elif recovery_need < 0.5:
            recovery_status = "moderate_recovery_needed"
        elif recovery_need < 0.7:
            recovery_status = "significant_recovery_needed"
        else:
            recovery_status = "urgent_recovery_needed"
        estimated_recovery_time = (
            recovery_need / recovery_capacity if recovery_capacity > 0 else float("inf")
        )
        return {
            "recovery_status": recovery_status,
            "recovery_need": recovery_need,
            "recovery_capacity": recovery_capacity,
            "estimated_recovery_time": min(24.0, estimated_recovery_time),
            "recovery_efficiency": self.hardware_layer.adaptation_parameters[
                "recovery_speed"
            ],
        }

    def _get_immediate_recommendations(self) -> list[str]:
        recommendations = []
        alerts = self._generate_critical_alerts()
        if alerts:
            recommendations.append(
                "Atender inmediatamente las alertas críticas del sistema"
            )
        if self.hardware_layer.physiological_state in [
            FisiologicalState.CRITICAL,
            FisiologicalState.EXHAUSTED,
        ]:
            recommendations.append(
                "Implementar descanso inmediato y reducir toda actividad"
            )
        avg_energy = sum(
            self.hardware_layer.resource_management.energy_pools.values()
        ) / len(self.hardware_layer.resource_management.energy_pools)
        if avg_energy < 0.3:
            recommendations.append(
                "Consumir recursos energéticos y reducir gasto inmediatamente"
            )
        stress_level = self.sistema_hormonal._calculate_hormonal_state().get(
            "estres", 0.3
        )
        if stress_level > 0.8:
            recommendations.append("Aplicar técnicas de reducción de estrés inmediatas")
        return recommendations

    def _get_short_term_recommendations(self) -> list[str]:
        recommendations = []
        imbalances = self.sistema_hormonal._detect_imbalances()
        if imbalances:
            recommendations.append("Implementar protocolo de rebalance hormonal")
        if self.integration_coherence < 0.6:
            recommendations.append("Establecer rutina diaria de prácticas mente-cuerpo")
        if self.hardware_layer.performance_metrics["energy_efficiency"] < 0.6:
            recommendations.append(
                "Revisar y optimizar patrones de actividad y descanso"
            )
        return recommendations

    def _get_long_term_recommendations(self) -> list[str]:
        recommendations = []
        health_score = self._calculate_overall_health_score()
        if health_score < 0.7:
            recommendations.append(
                "Desarrollar programa integral de fortalecimiento biológico"
            )
        adaptation_effectiveness = (
            self.hardware_layer._calculate_adaptation_effectiveness()
        )
        if adaptation_effectiveness < 0.7:
            recommendations.append("Entrenar capacidades de adaptación y resiliencia")
        sync_metrics = self._calculate_synchronization_metrics()
        if sync_metrics["overall_synchronization"] < 0.7:
            recommendations.append(
                "Desarrollar coherencia avanzada mente-cuerpo a largo plazo"
            )
        return recommendations

    def simulate_biological_intervention(
        self, intervention_type: str, parameters: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Simula una intervención biológica específica y sus efectos.
        """
        parameters = parameters or {}
        initial_state = self.get_comprehensive_biological_state()
        if intervention_type == "hormonal_rebalancing":
            self._simulate_hormonal_rebalancing(parameters)
        elif intervention_type == "energy_restoration":
            self._simulate_energy_restoration(parameters)
        elif intervention_type == "stress_reduction_protocol":
            self._simulate_stress_reduction(parameters)
        elif intervention_type == "circadian_realignment":
            self._simulate_circadian_realignment(parameters)
        elif intervention_type == "integration_enhancement":
            self._simulate_integration_enhancement(parameters)
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        final_state = self.get_comprehensive_biological_state()
        effectiveness = self._calculate_intervention_effectiveness(
            initial_state, final_state, intervention_type
        )
        return {
            "intervention_type": intervention_type,
            "parameters": parameters,
            "effectiveness": effectiveness,
            "side_effects": self._identify_intervention_side_effects(
                initial_state, final_state
            ),
            "sustainability": self._assess_intervention_sustainability(
                effectiveness, intervention_type
            ),
        }

    def _simulate_hormonal_rebalancing(self, parameters: dict[str, Any]) -> None:
        intensity = parameters.get("intensity", 0.5)
        for nt, current_level in self.sistema_hormonal.niveles.items():
            target_level = self.sistema_hormonal.niveles_base[nt]
            adjustment = (target_level - current_level) * intensity * 0.3
            self.sistema_hormonal.niveles[nt] = max(
                0.0, min(1.0, current_level + adjustment)
            )

    def _simulate_energy_restoration(self, parameters: dict[str, Any]) -> None:
        restoration_intensity = parameters.get("intensity", 0.7)
        focus_pools = parameters.get(
            "focus_pools",
            list(self.hardware_layer.resource_management.energy_pools.keys()),
        )
        for pool in focus_pools:
            if pool in self.hardware_layer.resource_management.energy_pools:
                current_level = self.hardware_layer.resource_management.energy_pools[
                    pool
                ]
                restoration = (1.0 - current_level) * restoration_intensity * 0.5
                self.hardware_layer.resource_management.energy_pools[pool] = min(
                    1.0, current_level + restoration
                )

    def _simulate_stress_reduction(self, parameters: dict[str, Any]) -> None:
        reduction_intensity = parameters.get("intensity", 0.6)
        self.sistema_hormonal.niveles["cortisol"] *= 1.0 - reduction_intensity * 0.4
        self.sistema_hormonal.niveles["adrenalina"] *= 1.0 - reduction_intensity * 0.5
        self.sistema_hormonal.niveles["gaba"] = min(
            1.0, self.sistema_hormonal.niveles["gaba"] + reduction_intensity * 0.3
        )
        self.sistema_hormonal.niveles["serotonina"] = min(
            1.0, self.sistema_hormonal.niveles["serotonina"] + reduction_intensity * 0.2
        )

    def _simulate_circadian_realignment(self, parameters: dict[str, Any]) -> None:
        target_phase = parameters.get("target_phase", 0.5)
        alignment_strength = parameters.get("strength", 0.6)
        current_phase = self.hardware_layer.physical_attributes.circadian_phase
        phase_adjustment = (target_phase - current_phase) * alignment_strength * 0.5
        self.hardware_layer.physical_attributes.circadian_phase = (
            current_phase + phase_adjustment
        ) % 1.0
        sleep_improvement = alignment_strength * 0.4
        self.hardware_layer.physical_attributes.sleep_debt = max(
            0.0, self.hardware_layer.physical_attributes.sleep_debt - sleep_improvement
        )

    def _simulate_integration_enhancement(self, parameters: dict[str, Any]) -> None:
        enhancement_intensity = parameters.get("intensity", 0.5)
        current_coherence = self.integration_coherence
        coherence_improvement = (0.85 - current_coherence) * enhancement_intensity * 0.6
        self.integration_coherence = min(1.0, current_coherence + coherence_improvement)
        for feedback_type in self.hardware_layer.feedback_loops:
            current_feedback = self.hardware_layer.feedback_loops[feedback_type]
            balance_adjustment = -current_feedback * enhancement_intensity * 0.3
            self.hardware_layer.feedback_loops[feedback_type] = (
                current_feedback + balance_adjustment
            )

    def _calculate_intervention_effectiveness(
        self, initial: dict[str, Any], final: dict[str, Any], intervention_type: str
    ) -> dict[str, float]:
        effectiveness = {}
        initial_health = initial["system_diagnostics"]["overall_health_score"]
        final_health = final["system_diagnostics"]["overall_health_score"]
        effectiveness["overall"] = final_health - initial_health
        if intervention_type == "hormonal_rebalancing":
            initial_coherence = initial["hormonal_system"]["system_coherence"]
            final_coherence = final["hormonal_system"]["system_coherence"]
            effectiveness["hormonal_coherence"] = final_coherence - initial_coherence
        elif intervention_type == "energy_restoration":
            initial_energy = sum(
                initial["hardware_system"]["resource_state"]["energy_pools"].values()
            ) / len(initial["hardware_system"]["resource_state"]["energy_pools"])
            final_energy = sum(
                final["hardware_system"]["resource_state"]["energy_pools"].values()
            ) / len(final["hardware_system"]["resource_state"]["energy_pools"])
            effectiveness["energy_restoration"] = final_energy - initial_energy
        elif intervention_type == "integration_enhancement":
            initial_integration = initial["integration_analysis"]["current_coherence"]
            final_integration = final["integration_analysis"]["current_coherence"]
            effectiveness["integration_improvement"] = (
                final_integration - initial_integration
            )
        return effectiveness

    def _identify_intervention_side_effects(
        self, initial: dict[str, Any], final: dict[str, Any]
    ) -> list[str]:
        side_effects = []
        initial_alerts = len(initial["system_diagnostics"]["critical_alerts"])
        final_alerts = len(final["system_diagnostics"]["critical_alerts"])
        if final_alerts > initial_alerts:
            side_effects.append("Incremento en alertas críticas del sistema")
        initial_imbalances = len(initial["hormonal_system"]["active_imbalances"])
        final_imbalances = len(final["hormonal_system"]["active_imbalances"])
        if final_imbalances > initial_imbalances:
            side_effects.append("Nuevos desequilibrios hormonales detectados")
        return side_effects

    def _assess_intervention_sustainability(
        self, effectiveness: dict[str, float], intervention_type: str
    ) -> dict[str, Any]:
        overall_effectiveness = effectiveness.get("overall", 0.0)
        base_sustainability = min(1.0, max(0.0, overall_effectiveness * 2))
        sustainability_modifiers = {
            "hormonal_rebalancing": 0.8,
            "energy_restoration": 0.6,
            "stress_reduction_protocol": 0.7,
            "circadian_realignment": 0.9,
            "integration_enhancement": 0.85,
        }
        sustainability_score = base_sustainability * sustainability_modifiers.get(
            intervention_type, 0.7
        )
        if sustainability_score >= 0.8:
            level = "highly_sustainable"
        elif sustainability_score >= 0.6:
            level = "moderately_sustainable"
        elif sustainability_score >= 0.4:
            level = "requires_maintenance"
        else:
            level = "unsustainable"
        return {
            "sustainability_score": sustainability_score,
            "sustainability_level": level,
            "maintenance_requirements": self._generate_maintenance_requirements(
                intervention_type, sustainability_score
            ),
        }

    def _generate_maintenance_requirements(
        self, intervention_type: str, sustainability_score: float
    ) -> list[str]:
        requirements = []
        if sustainability_score < 0.6:
            requirements.append("Monitoreo frecuente del sistema biológico")
            requirements.append("Reajustes periódicos de la intervención")
        if intervention_type == "energy_restoration":
            requirements.append("Mantener rutinas de gestión energética")
            requirements.append("Evitar patrones de agotamiento recurrentes")
        elif intervention_type == "hormonal_rebalancing":
            requirements.append("Monitorear niveles hormonales semanalmente")
            requirements.append("Ajustar factores de modulación externa")
        elif intervention_type == "circadian_realignment":
            requirements.append("Mantener consistencia en horarios de sueño")
            requirements.append("Minimizar disrupciones del ritmo circadiano")
        return requirements

    def recall_biological_experience(self, cue: str) -> dict[str, Any]:
        """
        Recalls a biological experience from EVA memory.
        """
        if self.eva_manager:
            return self.eva_manager.recall_experience(
                entity_id=self.entity_id,
                experience_id=cue,
            )
        return {"error": "EVA Memory Manager not available."}

    def get_eva_api(self) -> dict[str, Any]:
        """
        Returns a dictionary with the EVA API specific for biological body,
        exposing compatible EVA functionalities.
        """
        return {
            "record_biological_event": self.process_event,  # process_event now records to EVA
            "recall_biological_experience": self.recall_biological_experience,
            # Add other EVA-related methods if they become available and compatible
        }

    def get_comprehensive_biological_state(self) -> dict[str, Any]:
        """
        Obtiene el estado biológico completo e integrado.
        """
        return {
            "hormonal_system": {
                "current_levels": self.sistema_hormonal.niveles.copy(),
                "hormonal_state": self.sistema_hormonal._calculate_hormonal_state(),
                "system_coherence": self.sistema_hormonal._calculate_system_coherence(),
                "recent_trends": self.sistema_hormonal.get_neurotransmitter_trends(),
                "active_imbalances": self.sistema_hormonal._detect_imbalances(),
            },
            "hardware_system": {
                "physical_attributes": self.hardware_layer._get_physical_attributes_dict(),
                "resource_state": {
                    "energy_pools": self.hardware_layer.resource_management.energy_pools.copy(),
                    "resource_status": self.hardware_layer._evaluate_resource_status(),
                },
                "performance_metrics": self.hardware_layer.performance_metrics.copy(),
                "physiological_state": self.hardware_layer.physiological_state.value,
                "adaptation_parameters": self.hardware_layer.adaptation_parameters.copy(),
            },
            "integration_analysis": {
                "current_coherence": self.integration_coherence,
                "integration_quality": self._classify_integration_quality(
                    self.integration_coherence
                ),
                "historical_stability": self._analyze_integration_stability(),
                "synchronization_metrics": self._calculate_synchronization_metrics(),
            },
            "system_diagnostics": {
                "overall_health_score": self._calculate_overall_health_score(),
                "critical_alerts": self._generate_critical_alerts(),
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "recovery_status": self._assess_recovery_status(),
            },
            "recommendations": {
                "immediate": self._get_immediate_recommendations(),
                "short_term": self._get_short_term_recommendations(),
                "long_term": self._get_long_term_recommendations(),
            },
        }

    def _analyze_integration_stability(self) -> dict[str, Any]:
        if len(self.integration_history) < 5:
            return {"status": "insufficient_data"}
        recent_coherence = [
            entry["integration_coherence"] for entry in self.integration_history[-10:]
        ]
        mean_coherence = sum(recent_coherence) / len(recent_coherence)
        variance = sum((c - mean_coherence) ** 2 for c in recent_coherence) / len(
            recent_coherence
        )
        if variance < 0.01:
            stability = "very_stable"
        elif variance < 0.05:
            stability = "stable"
        elif variance < 0.1:
            stability = "moderately_unstable"
        else:
            stability = "unstable"
        return {
            "stability_level": stability,
            "mean_coherence": mean_coherence,
            "variance": variance,
            "trend": self._calculate_trend_direction(recent_coherence),
        }

    def _calculate_trend_direction(self, values: list[float]) -> str:
        if len(values) < 2:
            return "stable"
        delta = values[-1] - values[0]
        if abs(delta) < 0.02:
            return "stable"
        elif delta > 0.02:
            return "improving"
        else:
            return "declining"

    def _calculate_synchronization_metrics(self) -> dict[str, float]:
        hormonal_state = self.sistema_hormonal._calculate_hormonal_state()
        hormonal_stress = hormonal_state.get("estres", 0.3)
        hardware_stress = 1.0 - self.hardware_layer.performance_metrics.get(
            "system_stability", 0.7
        )
        stress_sync = 1.0 - abs(hormonal_stress - hardware_stress)
        hormonal_energy = hormonal_state.get("balance_general", 0.5)
        hardware_energy = sum(
            self.hardware_layer.resource_management.energy_pools.values()
        ) / len(self.hardware_layer.resource_management.energy_pools)
        energy_sync = 1.0 - abs(hormonal_energy - hardware_energy)
        hormonal_arousal = hormonal_state.get("alerta", 0.3)
        hardware_arousal = self.hardware_layer.performance_metrics.get(
            "cpu_utilization", 0.4
        )
        arousal_sync = 1.0 - abs(hormonal_arousal - hardware_arousal)
        return {
            "stress_synchronization": stress_sync,
            "energy_synchronization": energy_sync,
            "arousal_synchronization": arousal_sync,
            "overall_synchronization": (stress_sync + energy_sync + arousal_sync) / 3,
        }
