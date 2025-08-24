"""
Mind Core Types - Shared Data Structures
=========================================

Contains shared data structures (Enums, Dataclasses) used by mind_core,
judgment, and other modules to prevent circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CategoriaError(Enum):
    VALIDACION = "VALIDACION"
    CREATIVIDAD = "CREATIVIDAD"
    DEPENDENCIAS = "DEPENDENCIAS"
    TIEMPO = "TIEMPO"
    RECURSOS = "RECURSOS"
    CALIDAD = "CALIDAD"
    SEGURIDAD = "SEGURIDAD"

    @classmethod
    def from_any(cls, value: CategoriaError | str) -> CategoriaError:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except Exception:
            # accept name-style strings too
            name = str(value).upper()
            if name in cls.__members__:
                return cls[name]
            # fallback to VALIDACION as conservative default
            return cls.VALIDACION


class TipoResultado(Enum):
    EXITO_TOTAL = "EXITO_TOTAL"
    EXITO_PARCIAL = "EXITO_PARCIAL"
    FALLO_MENOR = "FALLO_MENOR"
    FALLO_CRITICO = "FALLO_CRITICO"
    CANCELADO = "CANCELADO"

    @property
    def is_failure(self) -> bool:
        return self in (TipoResultado.FALLO_MENOR, TipoResultado.FALLO_CRITICO)

    @classmethod
    def from_any(cls, value: TipoResultado | str) -> TipoResultado:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except Exception:
            name = str(value).upper()
            if name in cls.__members__:
                return cls[name]
            return cls.EXITO_PARCIAL


@dataclass
class FeedbackEjecucion:
    """
    Typed container for execution feedback used across ADAM's mind modules.

    - plan_id: unique plan identifier.
    - resultado: normalized `TipoResultado`.
    - tiempo_ejecucion: seconds (float).
    - pasos_exitosos / pasos_totales: integers to compute success_rate.
    - errores: list of error records (dict) with optional 'code'/'msg'.
    - categorias_error: list of `CategoriaError` (normalized on construction).
    - metricas_calidad: numeric quality metrics (values 0.0..1.0).
    - contexto_ejecucion: arbitrary contextual payload.
    - timestamp: ISO timestamp of feedback (auto-filled if missing).
    - puntuacion_exito: computed convenience score (0.0..1.0).
    """

    plan_id: str
    resultado: TipoResultado = field(default=TipoResultado.EXITO_PARCIAL)
    tiempo_ejecucion: float = 0.0
    pasos_exitosos: int = 0
    pasos_totales: int = 0
    errores: list[dict[str, Any]] = field(default_factory=list)
    categorias_error: list[CategoriaError] = field(default_factory=list)
    metricas_calidad: dict[str, float] = field(default_factory=dict)
    contexto_ejecucion: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    puntuacion_exito: float = 0.0

    def __post_init__(self) -> None:
        # Normalize enums if raw values were provided
        try:
            self.resultado = TipoResultado.from_any(self.resultado)
        except Exception:
            self.resultado = TipoResultado.EXITO_PARCIAL

        normalized_cats: list[CategoriaError] = []
        for c in self.categorias_error or []:
            try:
                normalized_cats.append(CategoriaError.from_any(c))
            except Exception:
                # skip unknown entries
                continue
        self.categorias_error = normalized_cats

        # Guard against invalid numeric types
        try:
            self.tiempo_ejecucion = float(self.tiempo_ejecucion or 0.0)
        except Exception:
            self.tiempo_ejecucion = 0.0

        self.pasos_exitosos = int(self.pasos_exitosos or 0)
        self.pasos_totales = int(self.pasos_totales or 0)

        # Compute a fallback success score
        self.puntuacion_exito = float(
            self.metricas_calidad.get("overall_score")
            or self.metricas_calidad.get("success_rate")
            or self._compute_success_rate()
        )
        # Ensure within [0,1]
        self.puntuacion_exito = max(0.0, min(1.0, self.puntuacion_exito))

    def _compute_success_rate(self) -> float:
        if self.pasos_totales <= 0:
            return 0.0
        try:
            return float(self.pasos_exitosos) / float(self.pasos_totales)
        except Exception:
            return 0.0

    def success_rate(self) -> float:
        """Explicit helper returning success rate in 0.0..1.0."""
        sr = self.metricas_calidad.get("success_rate")
        if sr is not None:
            try:
                return float(sr)
            except Exception:
                pass
        return self._compute_success_rate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict (enums converted to names)."""
        return {
            "plan_id": self.plan_id,
            "resultado": (
                self.resultado.name
                if isinstance(self.resultado, TipoResultado)
                else str(self.resultado)
            ),
            "tiempo_ejecucion": float(self.tiempo_ejecucion),
            "pasos_exitosos": int(self.pasos_exitosos),
            "pasos_totales": int(self.pasos_totales),
            "errores": list(self.errores),
            "categorias_error": [c.name for c in (self.categorias_error or [])],
            "metricas_calidad": dict(self.metricas_calidad or {}),
            "contexto_ejecucion": dict(self.contexto_ejecucion or {}),
            "timestamp": self.timestamp,
            "puntuacion_exito": float(self.puntuacion_exito),
            "success_rate": self.success_rate(),
        }

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | FeedbackEjecucion) -> FeedbackEjecucion:
        """
        Robust constructor accepting either an existing FeedbackEjecucion or
        a raw dict coming from external subsystems (EVA, IntentionalityEngine, tests).
        Performs normalization and best-effort coercion.
        """
        if isinstance(raw, FeedbackEjecucion):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("from_raw expects a dict or FeedbackEjecucion")

        return cls(
            plan_id=str(raw.get("plan_id", raw.get("id", "unknown"))),
            resultado=TipoResultado.from_any(
                raw.get("resultado", raw.get("status", TipoResultado.EXITO_PARCIAL))
            ),
            tiempo_ejecucion=float(
                raw.get("tiempo_ejecucion", raw.get("duration", 0.0)) or 0.0
            ),
            pasos_exitosos=int(
                raw.get("pasos_exitosos", raw.get("steps_successful", 0)) or 0
            ),
            pasos_totales=int(raw.get("pasos_totales", raw.get("steps_total", 0)) or 0),
            errores=raw.get("errores", raw.get("errors", [])) or [],
            categorias_error=[
                CategoriaError.from_any(c)
                for c in (
                    raw.get("categorias_error", raw.get("error_categories", [])) or []
                )
            ],
            metricas_calidad={
                k: float(v)
                for k, v in (
                    raw.get("metricas_calidad", raw.get("quality_metrics", {})) or {}
                ).items()
            },
            contexto_ejecucion=raw.get("contexto_ejecucion", raw.get("context", {}))
            or {},
            timestamp=str(raw.get("timestamp", datetime.utcnow().isoformat() + "Z")),
        )


# Type aliases used across the mind modules
FeedbackList = list[FeedbackEjecucion]
ErrorSummary = dict[str, Any]
