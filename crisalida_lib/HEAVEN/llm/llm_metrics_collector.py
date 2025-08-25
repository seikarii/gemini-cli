"""
LLMMetricsCollector - Monitor avanzado de métricas y eficiencia LLM
===================================================================

Recopila, analiza y reporta métricas detalladas sobre el uso, rendimiento y progreso
de independencia de los LLMs en Crisalida. Incluye diagnóstico extendido, trazabilidad
histórica y recomendaciones para optimización.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class LLMMetricsCollector:
    """
    Monitor avanzado de métricas y eficiencia para LLMs.
    Registra interacciones, calcula tasas de éxito, latencia, uso de fallback y progreso de independencia.
    """

    def __init__(self):
        self.metrics = {
            "brain_calls": 0,
            "cerebellum_calls": 0,
            "successful_llm_tasks": 0,
            "failed_llm_tasks": 0,
            "fallback_activations": 0,
            "total_llm_response_time": 0.0,
            "independence_progress": 0.0,
        }
        self.detailed_logs: list[dict[str, Any]] = []
        self.daily_summaries: list[dict[str, Any]] = []
        self.last_update: datetime | None = None

    def record_llm_interaction(
        self,
        role: str,
        success: bool,
        response_time: float,
        task_type: str,
        fallback_used: bool = False,
        extra_info: dict[str, Any] | None = None,
    ):
        """
        Registra interacción con LLM, actualiza métricas y guarda log detallado.
        """
        timestamp = datetime.now()
        if role == "brain":
            self.metrics["brain_calls"] += 1
        elif role == "cerebellum":
            self.metrics["cerebellum_calls"] += 1
        if success:
            self.metrics["successful_llm_tasks"] += 1
        else:
            self.metrics["failed_llm_tasks"] += 1
        if fallback_used:
            self.metrics["fallback_activations"] += 1
        self.metrics["total_llm_response_time"] += response_time
        self.last_update = timestamp

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "role": role,
            "success": success,
            "response_time": response_time,
            "task_type": task_type,
            "fallback_used": fallback_used,
        }
        if extra_info:
            log_entry.update(extra_info)
        self.detailed_logs.append(log_entry)
        if len(self.detailed_logs) > 1000:
            self.detailed_logs.pop(0)

        logger.debug(f"LLMMetricsCollector: Recorded interaction {log_entry}")

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Genera resumen extendido de rendimiento y eficiencia.
        """
        total_calls = self.metrics["brain_calls"] + self.metrics["cerebellum_calls"]
        total_tasks = (
            self.metrics["successful_llm_tasks"] + self.metrics["failed_llm_tasks"]
        )
        if total_tasks == 0:
            return {"status": "no_data"}
        success_rate = self.metrics["successful_llm_tasks"] / total_tasks
        avg_response_time = (
            self.metrics["total_llm_response_time"] / total_calls
            if total_calls > 0
            else 0
        )
        fallback_rate = self.metrics["fallback_activations"] / total_tasks
        summary = {
            "total_llm_interactions": total_calls,
            "success_rate": round(success_rate, 3),
            "average_response_time": round(avg_response_time, 3),
            "fallback_activation_rate": round(fallback_rate, 3),
            "brain_usage": self.metrics["brain_calls"],
            "cerebellum_usage": self.metrics["cerebellum_calls"],
            "independence_progress": round(self._calculate_independence_progress(), 3),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
        logger.info(f"LLMMetricsCollector: Performance summary {summary}")
        return summary

    def _calculate_independence_progress(self) -> float:
        """
        Calcula progreso hacia la independencia cognitiva.
        Factores: menor uso de LLMs, mayor éxito con fallbacks, tareas complejas resueltas internamente.
        """
        recent_logs = self.detailed_logs[-100:]  # Últimos 100 eventos
        if len(recent_logs) < 10:
            return 0.0
        fallback_success_count = sum(
            1 for log in recent_logs if log["fallback_used"] and log["success"]
        )
        fallback_total = sum(1 for log in recent_logs if log["fallback_used"])
        fallback_success_rate = (
            fallback_success_count / fallback_total if fallback_total > 0 else 0.0
        )
        internal_task_rate = fallback_total / len(recent_logs)
        progress = (fallback_success_rate * 0.6) + (internal_task_rate * 0.4)
        self.metrics["independence_progress"] = progress
        return progress

    def get_detailed_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Devuelve los últimos logs detallados de interacción."""
        return self.detailed_logs[-limit:]

    def get_last_update(self) -> str | None:
        """Devuelve el timestamp del último evento registrado."""
        return self.last_update.isoformat() if self.last_update else None

    def reset(self):
        """Reinicia todas las métricas y logs."""
        self.metrics = {
            "brain_calls": 0,
            "cerebellum_calls": 0,
            "successful_llm_tasks": 0,
            "failed_llm_tasks": 0,
            "fallback_activations": 0,
            "total_llm_response_time": 0.0,
            "independence_progress": 0.0,
        }
        self.detailed_logs.clear()
        self.daily_summaries.clear()
        self.last_update = None
        logger.info("LLMMetricsCollector: Metrics and logs reset.")
