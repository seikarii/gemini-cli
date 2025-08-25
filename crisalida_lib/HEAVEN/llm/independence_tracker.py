"""
LLMIndependenceTracker - Monitor avanzado de autonomía y autosuficiencia LLM
============================================================================

Rastrea el grado de dependencia del sistema respecto a LLMs externos versus procesos internos.
Proporciona diagnóstico extendido, trazabilidad histórica y recomendaciones para la evolución
de la independencia cognitiva del agente.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class LLMIndependenceTracker:
    """Monitor avanzado de independencia y autosuficiencia respecto a LLMs externos."""

    def __init__(self, independence_threshold: float = 0.85, min_experience: int = 100):
        self.success_metrics = {
            "internal_translation_success": 0,
            "llm_assisted_success": 0,
            "fallback_usage": 0,
            "total_tasks": 0,
        }
        self.independence_threshold = independence_threshold
        self.min_experience = min_experience
        self.history: list[dict[str, Any]] = []
        self.last_assessment: dict[str, Any] | None = None
        self.last_update_time: float = time.time()

    def record_translation_attempt(
        self, method: str, success: bool, details: dict[str, Any] | None = None
    ):
        """
        Registra intento de traducción y su método, con trazabilidad extendida.
        Args:
            method: 'internal', 'llm_assisted', 'fallback'
            success: True si la tarea fue exitosa
            details: Información adicional (opcional)
        """
        self.success_metrics["total_tasks"] += 1
        if method == "internal" and success:
            self.success_metrics["internal_translation_success"] += 1
        elif method == "llm_assisted" and success:
            self.success_metrics["llm_assisted_success"] += 1
        elif method == "fallback":
            self.success_metrics["fallback_usage"] += 1

        record: dict[str, Any] = {
            "timestamp": float(time.time()),
            "method": method,
            "success": success,
            "details": details or {},
            "metrics_snapshot": self.success_metrics.copy(),
        }
        self.history.append(record)
        self.last_update_time = record["timestamp"]
        logger.debug(f"IndependenceTracker: Recorded attempt {record}")

    def assess_independence_readiness(self) -> dict[str, Any]:
        """
        Evalúa si el agente está listo para independencia cognitiva.
        Devuelve diagnóstico extendido y recomendación.
        """
        total = self.success_metrics["total_tasks"]
        if total < self.min_experience:
            assessment = {
                "ready": False,
                "reason": "insufficient_experience",
                "internal_success_rate": 0.0,
                "threshold": self.independence_threshold,
                "total_experience": total,
                "recommendation": "continue_learning",
            }
            self.last_assessment = assessment
            logger.info(
                "IndependenceTracker: Insufficient experience for independence."
            )
            return assessment

        internal_success_rate = (
            self.success_metrics["internal_translation_success"] / total
            if total > 0
            else 0.0
        )
        ready = internal_success_rate >= self.independence_threshold
        assessment = {
            "ready": ready,
            "internal_success_rate": internal_success_rate,
            "threshold": self.independence_threshold,
            "total_experience": total,
            "llm_assisted_success": self.success_metrics["llm_assisted_success"],
            "fallback_usage": self.success_metrics["fallback_usage"],
            "last_update_time": self.last_update_time,
            "recommendation": "enable_independence" if ready else "continue_learning",
        }
        self.last_assessment = assessment
        logger.info(f"IndependenceTracker: Assessment {assessment}")
        return assessment

    def get_metrics(self) -> dict[str, Any]:
        """Devuelve métricas actuales de independencia."""
        return self.success_metrics.copy()

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Devuelve historial de intentos recientes."""
        return self.history[-limit:]

    def get_last_assessment(self) -> dict[str, Any] | None:
        """Devuelve el último diagnóstico de independencia."""
        return self.last_assessment

    def reset(self):
        """Reinicia métricas e historial."""
        self.success_metrics = {
            "internal_translation_success": 0,
            "llm_assisted_success": 0,
            "fallback_usage": 0,
            "total_tasks": 0,
        }
        self.history.clear()
        self.last_assessment = None
        self.last_update_time = time.time()
        logger.info("IndependenceTracker: Metrics and history reset.")
