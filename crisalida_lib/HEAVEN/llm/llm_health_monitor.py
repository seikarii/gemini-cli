"""
LLMHealthMonitor - Monitor avanzado de salud y rendimiento para LLMs
====================================================================

Evalúa en tiempo real la disponibilidad, latencia y tasa de error de los conectores LLM.
Incluye diagnóstico extendido, trazabilidad histórica, alertas proactivas y recomendaciones.
Permite integración con sistemas de orquestación y fallback.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from .base_llm_connector import AbstractLLMConnector

logger = logging.getLogger(__name__)


class LLMRole(Enum):
    BRAIN = "brain"
    CEREBELLUM = "cerebellum"


class LLMHealthMonitor:
    """Monitor avanzado de salud y rendimiento para todos los conectores LLM."""

    def __init__(self, connectors: list[AbstractLLMConnector]):
        self.connectors = {conn.role.value: conn for conn in connectors}
        self.health_history: list[dict[str, Any]] = []
        self.latency_history: dict[str, list[float]] = {
            role: [] for role in self.connectors
        }
        self.error_history: dict[str, list[str]] = {
            role: [] for role in self.connectors
        }
        self.last_check: datetime | None = None
        self.alerts: list[dict[str, Any]] = []

    async def check_all(self) -> dict[str, Any]:
        """Verifica salud, latencia y errores de todos los conectores LLM."""
        results = {}
        any_available = False
        all_available = True
        for role, connector in self.connectors.items():
            start = datetime.now()
            try:
                health = await connector.health_check()
                latency = (
                    (datetime.now() - start).total_seconds()
                    if hasattr(connector, "last_response") and connector.last_response
                    else None
                )
                error = connector.get_last_error()
                results[role] = {
                    "available": health,
                    "model": connector.model_name,
                    "status": connector.health_status,
                    "latency": latency,
                    "last_error": error,
                }
                if latency is not None:
                    self.latency_history[role].append(latency)
                    # Mantener solo últimos 50 valores
                    if len(self.latency_history[role]) > 50:
                        self.latency_history[role].pop(0)
                if error:
                    self.error_history[role].append(error)
                    if len(self.error_history[role]) > 50:
                        self.error_history[role].pop(0)
                if health:
                    any_available = True
                else:
                    all_available = False
            except Exception as e:
                logger.error(f"Health check error for {role}: {e}")
                results[role] = {
                    "available": False,
                    "model": connector.model_name,
                    "status": "error",
                    "latency": None,
                    "last_error": str(e),
                }
                self.error_history[role].append(str(e))
                all_available = False
        self.last_check = datetime.now()
        health_record = {
            "timestamp": self.last_check,
            "results": results,
            "any_available": any_available,
            "all_available": all_available,
        }
        self.health_history.append(health_record)
        # Mantener solo últimos 200 checks
        if len(self.health_history) > 200:
            self.health_history.pop(0)
        # Alertas proactivas
        self._check_alerts(results)
        return {
            "individual_status": results,
            "any_available": any_available,
            "all_available": all_available,
            "timestamp": self.last_check.isoformat(),
            "alerts": self.alerts[-5:],
        }

    def _check_alerts(self, results: dict[str, Any]):
        """Genera alertas proactivas si hay degradación o errores frecuentes."""
        for role, status in results.items():
            if not status["available"]:
                alert = {
                    "timestamp": (
                        self.last_check.isoformat() if self.last_check else None
                    ),
                    "role": role,
                    "model": status["model"],
                    "type": "unavailable",
                    "details": status.get("last_error", ""),
                }
                self.alerts.append(alert)
            elif status["latency"] and status["latency"] > 5.0:
                alert = {
                    "timestamp": (
                        self.last_check.isoformat() if self.last_check else None
                    ),
                    "role": role,
                    "model": status["model"],
                    "type": "high_latency",
                    "latency": status["latency"],
                }
                self.alerts.append(alert)
        # Mantener solo últimos 20 alertas
        if len(self.alerts) > 20:
            self.alerts = self.alerts[-20:]

    def get_health_summary(self) -> dict[str, Any]:
        """Resumen extendido del estado de salud, latencia y errores."""
        if not self.health_history:
            return {"status": "no_data"}
        recent_checks = self.health_history[-20:]  # Últimos 20 checks
        availability_rate = {}
        avg_latency = {}
        error_counts = {}
        for role in self.connectors.keys():
            available_count = sum(
                1 for check in recent_checks if check["results"][role]["available"]
            )
            availability_rate[role] = available_count / len(recent_checks)
            latencies = [
                check["results"][role]["latency"]
                for check in recent_checks
                if check["results"][role]["latency"]
            ]
            avg_latency[role] = (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            )
            error_counts[role] = len(self.error_history[role])
        overall_health = (
            "healthy"
            if any(rate > 0.8 for rate in availability_rate.values())
            else "degraded"
        )
        recommendations = self._generate_recommendations(
            availability_rate, avg_latency, error_counts
        )
        return {
            "overall_health": overall_health,
            "availability_rates": availability_rate,
            "avg_latency": avg_latency,
            "error_counts": error_counts,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "total_checks": len(self.health_history),
            "alerts": self.alerts[-5:],
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self, availability_rate, avg_latency, error_counts
    ) -> list[str]:
        """Genera recomendaciones proactivas según diagnóstico."""
        recs = []
        for role in self.connectors.keys():
            if availability_rate[role] < 0.5:
                recs.append(f"Revisar disponibilidad del modelo {role} (tasa < 50%)")
            if avg_latency[role] and avg_latency[role] > 3.0:
                recs.append(f"Latencia alta en {role} (promedio > 3s)")
            if error_counts[role] > 10:
                recs.append(f"Errores frecuentes en {role} (últimos 10+ checks)")
        if not recs:
            recs.append("Todos los modelos LLM operan en estado óptimo.")
        return recs

    def get_latency_history(self, role: str) -> list[float]:
        """Devuelve historial de latencia para un rol dado."""
        return self.latency_history.get(role, []).copy()

    def get_error_history(self, role: str) -> list[str]:
        """Devuelve historial de errores para un rol dado."""
        return self.error_history.get(role, []).copy()

    def get_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Devuelve las alertas recientes."""
        return self.alerts[-limit:]

    def reset(self):
        """Reinicia historial de salud, latencia y alertas."""
        self.health_history.clear()
        for role in self.connectors:
            self.latency_history[role].clear()
            self.error_history[role].clear()
        self.alerts.clear()
        self.last_check = None
        logger.info("LLMHealthMonitor: Historial y alertas reiniciados.")
