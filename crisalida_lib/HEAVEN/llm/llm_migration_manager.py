"""
LLMMigrationManager - Orquestador avanzado de transición y migración LLM
=======================================================================

Gestiona la transición gradual desde dependencia LLM hacia independencia cognitiva.
Permite avanzar por fases, ajustar patrones de uso y diagnosticar el progreso.
Incluye trazabilidad, diagnóstico extendido y logging robusto.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMMigrationManager:
    """
    Orquestador de migración entre modelos LLM y transición hacia independencia.
    """

    def __init__(self, llm_gateway: Any):
        self.llm_gateway = llm_gateway
        self.migration_phases = {
            "development": {
                "brain_usage": 0.8,
                "cerebellum_usage": 0.9,
                "description": "Fase inicial - aprendizaje intensivo",
            },
            "maturation": {
                "brain_usage": 0.4,
                "cerebellum_usage": 0.6,
                "description": "Fase intermedia - reducción gradual",
            },
            "independence": {
                "brain_usage": 0.1,
                "cerebellum_usage": 0.2,
                "description": "Fase avanzada - casi independiente",
            },
            "autonomous": {
                "brain_usage": 0.0,
                "cerebellum_usage": 0.0,
                "description": "Independencia completa",
            },
        }
        self.current_phase = "development"
        self.task_counter = 0
        self.phase_history: list[dict[str, Any]] = []

    async def evaluate_migration_readiness(self) -> dict[str, Any]:
        """
        Evalúa si es momento de avanzar a la siguiente fase.
        Utiliza métricas de IndependenceTracker y LLMMetricsCollector si disponibles.
        """
        # Placeholder: integrar métricas reales en producción
        internal_success_rate = getattr(self.llm_gateway, "internal_success_rate", 0.7)
        llm_dependency_rate = getattr(self.llm_gateway, "llm_dependency_rate", 0.3)
        complexity_handling = getattr(self.llm_gateway, "complexity_handling", 0.7)
        total_tasks = getattr(self.llm_gateway, "total_tasks", self.task_counter)
        current_phase_config = self.migration_phases[self.current_phase]

        ready_to_advance = (
            internal_success_rate > 0.75
            and complexity_handling > 0.6
            and total_tasks > 100
        )
        next_phase = self._get_next_phase()
        diagnostics = {
            "current_phase": self.current_phase,
            "next_phase": next_phase,
            "ready_to_advance": ready_to_advance,
            "metrics": {
                "internal_success_rate": internal_success_rate,
                "llm_dependency_rate": llm_dependency_rate,
                "complexity_handling": complexity_handling,
                "total_tasks": total_tasks,
            },
            "current_config": current_phase_config,
        }
        logger.info(f"LLMMigrationManager: Diagnostics {diagnostics}")
        return diagnostics

    async def advance_migration_phase(self) -> bool:
        """
        Avanza a la siguiente fase de migración si es apropiado.
        Ajusta patrones de uso y registra el cambio.
        """
        evaluation = await self.evaluate_migration_readiness()
        if evaluation["ready_to_advance"] and evaluation["next_phase"]:
            old_phase = self.current_phase
            self.current_phase = evaluation["next_phase"]
            await self._adjust_llm_usage_patterns()
            self.phase_history.append(
                {
                    "from": old_phase,
                    "to": self.current_phase,
                    "timestamp": getattr(self.llm_gateway, "last_update", None),
                    "diagnostics": evaluation,
                }
            )
            logger.info(
                f"🚀 Migration phase advanced: {old_phase} → {self.current_phase}"
            )
            return True
        logger.info("LLMMigrationManager: No phase advancement performed.")
        return False

    def _get_next_phase(self) -> str | None:
        """Obtiene la siguiente fase en el plan de migración."""
        phases = list(self.migration_phases.keys())
        try:
            current_index = phases.index(self.current_phase)
            return (
                phases[current_index + 1] if current_index < len(phases) - 1 else None
            )
        except ValueError:
            return None

    async def _adjust_llm_usage_patterns(self):
        """
        Ajusta patrones de uso de LLM según la fase actual.
        Deshabilita LLMs en fase autónoma.
        """
        current_config = self.migration_phases[self.current_phase]
        if self.current_phase == "autonomous":
            await self.llm_gateway.enable_offline_mode()
            logger.info("🎯 Autonomous phase reached - LLMs disabled permanently")
        else:
            # Ajustar probabilidades de uso (deben existir en LLMGatewayOrchestrator)
            self.llm_gateway.brain_usage_probability = current_config["brain_usage"]
            self.llm_gateway.cerebellum_usage_probability = current_config[
                "cerebellum_usage"
            ]
            logger.info(
                f"LLMMigrationManager: Usage probabilities set - Brain: {current_config['brain_usage']}, Cerebellum: {current_config['cerebellum_usage']}"
            )

    def record_task(self):
        """Registra una tarea procesada para el contador de experiencia."""
        self.task_counter += 1

    def get_phase_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Devuelve historial de cambios de fase recientes."""
        return self.phase_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Devuelve el estado actual del gestor de migración."""
        return {
            "current_phase": self.current_phase,
            "task_counter": self.task_counter,
            "phase_history": self.get_phase_history(),
            "current_config": self.migration_phases[self.current_phase],
        }
