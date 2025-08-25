"""
Judgment Module - The Learning Core (definitive)
================================================

Professionalized implementation:
- Thread-safe, deterministic numeric fallbacks (numpy optional).
- Async-aware EVA persistence hooks (best-effort).
- Clear public API: process_feedback, process_feedback_batch, obtener_parametros_mentes, generar_reporte_aprendizaje, reset_parametros.
- Tighter guards for parameter adjustments and logging.
- Small evolution hooks for scheduling resets and exporting metrics.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

    class np:  # type: ignore
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0

        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance**0.5

        @staticmethod
        def var(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            return sum((x - mean_val) ** 2 for x in values) / len(values)

        @staticmethod
        def median(values):
            if not values:
                return 0.0
            s = sorted(values)
            n = len(s)
            mid = n // 2
            if n % 2 == 1:
                return s[mid]
            return 0.5 * (s[mid - 1] + s[mid])

        @staticmethod
        def polyfit(x, y, degree):
            # Conservative linear polyfit fallback
            if degree == 1 and len(x) > 1 and len(x) == len(y):
                n = len(x)
                sx = sum(x)
                sy = sum(y)
                sxy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
                sx2 = sum(xi * xi for xi in x)
                denom = n * sx2 - sx * sx
                if abs(denom) < 1e-12:
                    return [0.0, float(np.mean(y) if y else 0.0)]
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
                return [slope, intercept]
            return [0.0, float(np.mean(y) if y else 0.0)]


from crisalida_lib.ADAM.cuerpo.genome import (
    GenomaComportamiento,
)  # [`crisalida_lib.ADAM.cuerpo.genome.GenomaComportamiento`](crisalida_lib/ADAM/cuerpo/genome.py)

from .mind_types import (
    CategoriaError,
    FeedbackEjecucion,
    TipoResultado,
)  # [`crisalida_lib.ADAM.mente.mind_types.FeedbackEjecucion`](crisalida_lib/ADAM/mente/mind_types.py)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Utilities ---
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _safe_mean(xs: list[float], fallback: float = 0.0) -> float:
    try:
        return float(np.mean(xs)) if xs else fallback
    except Exception:
        return fallback


# --- Dataclasses ---
@dataclass
class PerfilMente:
    audacia: float = 0.5
    cautela: float = 0.5
    creatividad: float = 0.5
    rigor: float = 0.5
    optimismo: float = 0.5
    prioridad_analisis: float = 0.5
    prioridad_accion: float = 0.5
    tolerancia_riesgo: float = 0.5
    confianza_base: float = 0.5
    ajustes_historia: deque = field(default_factory=lambda: deque(maxlen=200))
    rendimiento_historico: deque = field(default_factory=lambda: deque(maxlen=500))

    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def ajustar_parametro(self, parametro: str, ajuste: float, razon: str) -> None:
        with self._lock:
            if not hasattr(self, parametro):
                logger.warning("Parámetro desconocido: %s", parametro)
                return
            anterior = float(getattr(self, parametro))
            nuevo = _clamp01(anterior + float(ajuste))
            setattr(self, parametro, nuevo)
            self.ajustes_historia.append(
                {
                    "parametro": parametro,
                    "anterior": anterior,
                    "nuevo": nuevo,
                    "ajuste": ajuste,
                    "razon": razon,
                    "timestamp": datetime.now(),
                }
            )
            logger.debug(
                "Ajuste parametro %s: %s->%s (%s)", parametro, anterior, nuevo, razon
            )

    def registrar_rendimiento(self, puntuacion: float, contexto: str) -> None:
        with self._lock:
            self.rendimiento_historico.append(
                {
                    "puntuacion": float(puntuacion),
                    "contexto": contexto,
                    "timestamp": datetime.now(),
                }
            )

    def calcular_rendimiento_promedio(self, ventana_dias: int = 7) -> float:
        cutoff = datetime.now() - timedelta(days=ventana_dias)
        with self._lock:
            recientes = [
                r["puntuacion"]
                for r in self.rendimiento_historico
                if r["timestamp"] >= cutoff
            ]
        return _safe_mean(recientes, 0.5)


# --- Pattern analyzer (kept compact & deterministic) ---
class AnalizadorPatrones:
    def __init__(self):
        self.umbral_confianza = 0.6

    def analizar_patrones_fallo(
        self, feedbacks: list[FeedbackEjecucion]
    ) -> dict[CategoriaError, dict[str, Any]]:
        if not feedbacks:
            return {}
        by_cat: dict[CategoriaError, list[FeedbackEjecucion]] = defaultdict(list)
        for f in feedbacks:
            if f.resultado in (TipoResultado.FALLO_MENOR, TipoResultado.FALLO_CRITICO):
                for c in f.categorias_error:
                    by_cat[c].append(f)
        report: dict[CategoriaError, dict[str, Any]] = {}
        for cat, items in by_cat.items():
            if len(items) < 2:
                continue
            severidad = _safe_mean(
                [
                    1.0 if it.resultado == TipoResultado.FALLO_CRITICO else 0.5
                    for it in items
                ],
                0.5,
            )
            report[cat] = {
                "frecuencia": len(items),
                "severidad_promedio": float(severidad),
                "contextos_comunes": self._extraer_contextos_comunes(items),
                "recomendaciones": self._generar_recomendaciones(cat, items),
            }
        return report

    def analizar_patrones_exito(
        self, feedbacks: list[FeedbackEjecucion]
    ) -> dict[str, Any]:
        exitos = [f for f in feedbacks if getattr(f, "puntuacion_exito", 0.0) >= 0.7]
        if len(exitos) < 3:
            return {}
        return {
            "factores_exito": self._identificar_factores_exito(exitos),
            "contextos_exitosos": self._extraer_contextos_comunes(exitos),
        }

    def _extraer_contextos_comunes(
        self, feedbacks: list[FeedbackEjecucion]
    ) -> dict[str, Any]:
        balances = [
            f.contexto_ejecucion.get("balance")
            for f in feedbacks
            if f.contexto_ejecucion
        ]
        palabras = [
            f.contexto_ejecucion.get("objetivo", "")
            for f in feedbacks
            if f.contexto_ejecucion
        ]
        return {
            "balance_promedio": float(
                _safe_mean([b for b in balances if isinstance(b, (int, float))], 0.0)
            ),
            "palabras_frecuentes": self._contar_palabras_frecuentes(palabras),
        }

    def _contar_palabras_frecuentes(self, textos: list[str]) -> dict[str, int]:
        counter: dict[str, int] = defaultdict(int)
        for t in textos:
            for w in str(t).lower().split():
                if len(w) > 3:
                    counter[w] += 1
        return {k: v for k, v in counter.items() if v >= 2}

    def _identificar_factores_exito(
        self, exitos: list[FeedbackEjecucion]
    ) -> dict[str, float]:
        tiempos = [
            e.tiempo_ejecucion
            for e in exitos
            if getattr(e, "tiempo_ejecucion", None) is not None
        ]
        pasos = [
            e.pasos_totales
            for e in exitos
            if getattr(e, "pasos_totales", None) is not None
        ]
        return {
            "tiempo_optimo": float(np.median(tiempos)) if tiempos else 0.0,
            "pasos_optimo": float(np.median(pasos)) if pasos else 0.0,
        }

    def _generar_recomendaciones(
        self, categoria: CategoriaError, fallos: list[FeedbackEjecucion]
    ) -> list[str]:
        mapping = {
            CategoriaError.VALIDACION: [
                "Incrementar rigor en Omega",
                "Añadir pasos de validación intermedia",
            ],
            CategoriaError.CREATIVIDAD: [
                "Incrementar exploración en Alfa",
                "Reducir exceso de cautela en contextos seguros",
            ],
            CategoriaError.DEPENDENCIAS: [
                "Priorizar análisis de dependencias antes de ejecutar",
            ],
            CategoriaError.TIEMPO: [
                "Optimizar tiempo de ejecución y reducir pasos no críticos",
            ],
        }
        return mapping.get(categoria, ["Investigar causa raíz"])


# --- JudgmentModule (public API) ---
class JudgmentModule:
    def __init__(
        self, genome: GenomaComportamiento, config: AdamConfig | None = None
    ):
        self.genome = genome
        cfg = config or AdamConfig()
        # Configurable defaults
        self.tasa_aprendizaje_base = float(cfg.JUDGMENT_BASE_LEARNING_RATE)
        self.ventana_analisis_dias = int(cfg.JUDGMENT_ANALYSIS_WINDOW_DAYS)
        self.umbral_ajuste_minimo = float(
            cfg.JUDGMENT_MIN_ADJUSTMENT_THRESHOLD
        )
        self.max_ajuste_por_ciclo = float(
            cfg.JUDGMENT_MAX_ADJUSTMENT_PER_CYCLE
        )
        self.PERFORMANCE_EXECUTION_TIME_THRESHOLD = float(
            cfg.PERFORMANCE_EXECUTION_TIME_THRESHOLD
        )
        self.PERFORMANCE_SUCCESS_RATE_THRESHOLD = float(
            cfg.PERFORMANCE_SUCCESS_RATE_THRESHOLD
        )
        self.PERFORMANCE_HIGH_QUALITY_THRESHOLD = float(
            cfg.PERFORMANCE_HIGH_QUALITY_THRESHOLD
        )

        # Initialize profiles with genome defaults when available
        alfa_defaults = getattr(genome, "alfa_defaults", {}) if genome else {}
        omega_defaults = getattr(genome, "omega_defaults", {}) if genome else {}

        self.perfil_alfa = PerfilMente(
            **{
                **{
                    "audacia": 0.5,
                    "cautela": 0.5,
                    "creatividad": 0.5,
                    "rigor": 0.5,
                    "optimismo": 0.5,
                    "prioridad_analisis": 0.5,
                    "prioridad_accion": 0.5,
                    "tolerancia_riesgo": 0.5,
                    "confianza_base": 0.5,
                },
                **alfa_defaults,
            }
        )
        self.perfil_omega = PerfilMente(
            **{
                **{
                    "audacia": 0.5,
                    "cautela": 0.5,
                    "creatividad": 0.5,
                    "rigor": 0.5,
                    "optimismo": 0.5,
                    "prioridad_analisis": 0.5,
                    "prioridad_accion": 0.5,
                    "tolerancia_riesgo": 0.5,
                    "confianza_base": 0.5,
                },
                **omega_defaults,
            }
        )

        self.analizador_patrones = AnalizadorPatrones()
        self.historial_feedback: deque = deque(maxlen=2000)
        self.metricas_aprendizaje: dict[str, list[Any]] = defaultdict(list)
        self._lock = threading.RLock()
        logger.info("JudgmentModule inicializado (professionalized)")

    # Public: single-feedback entry point (async-aware)
    async def process_feedback(
        self, feedback: FeedbackEjecucion, eva_manager: Any | None = None
    ) -> dict[str, Any]:
        """
        Process a single feedback instance, adapt parameters and optionally persist summary to EVA.
        Returns a dict with applied adjustments and diagnostics.
        """
        with self._lock:
            self.historial_feedback.append(feedback)
        ajustes = self._realizar_ajustes_adaptativos(feedback)
        with self._lock:
            self._actualizar_metricas_aprendizaje(feedback, ajustes)
            # persist small summary to EVA if available (best-effort)
            if eva_manager is not None:
                await self._persist_feedback_summary(feedback, eva_manager)
        return {"ajustes": ajustes, "timestamp": datetime.now().isoformat()}

    async def _persist_feedback_summary(
        self, feedback: FeedbackEjecucion, eva_manager: Any
    ) -> None:
        try:
            record_fn = getattr(eva_manager, "record_experience", None)
            if record_fn is None:
                return
            payload = {
                "type": "feedback_summary",
                "plan_id": getattr(feedback, "plan_id", None),
                "resultado": getattr(feedback, "resultado", None),
                "puntuacion_exito": getattr(feedback, "puntuacion_exito", None),
                "timestamp": getattr(feedback, "timestamp", datetime.now()).isoformat(),
            }
            if inspect.iscoroutinefunction(record_fn):
                try:
                    asyncio.create_task(
                        record_fn(
                            entity_id=getattr(eva_manager, "entity_id", "unknown"),
                            event_type="judgment_feedback",
                            data=payload,
                        )
                    )
                except RuntimeError:
                    # no running loop: run sync
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        record_fn(
                            entity_id=getattr(eva_manager, "entity_id", "unknown"),
                            event_type="judgment_feedback",
                            data=payload,
                        )
                    )
                    loop.close()
            else:
                try:
                    record_fn(
                        entity_id=getattr(eva_manager, "entity_id", "unknown"),
                        event_type="judgment_feedback",
                        data=payload,
                    )
                except Exception:
                    logger.debug("Synchronous EVA record_experience failed (non-fatal)")
        except Exception:
            logger.exception("Failed to persist feedback summary (non-fatal)")

    async def process_feedback_batch(
        self, feedbacks: list[FeedbackEjecucion], eva_manager: Any | None = None
    ) -> dict[str, Any]:
        applied = []
        for f in feedbacks:
            res = await self.process_feedback(f, eva_manager=eva_manager)
            applied.append(res)
        return {"processed": len(applied), "results": applied}

    # Keep core adaptive logic (preserved and hardened)
    def _realizar_ajustes_adaptativos(
        self, feedback: FeedbackEjecucion
    ) -> dict[str, list[str]]:
        ajustes: dict[str, list[str]] = {"alfa": [], "omega": []}
        try:
            if feedback.resultado == TipoResultado.FALLO_CRITICO:
                ajustes.update(self._ajustar_por_fallo_critico(feedback))
            elif feedback.resultado == TipoResultado.FALLO_MENOR:
                ajustes.update(self._ajustar_por_fallo_menor(feedback))
            elif feedback.resultado == TipoResultado.EXITO_TOTAL:
                ajustes.update(self._ajustar_por_exito(feedback))
            for categoria in getattr(feedback, "categorias_error", []) or []:
                cat_adj = self._ajustar_por_categoria_error(categoria)
                ajustes["alfa"].extend(cat_adj.get("alfa", []))
                ajustes["omega"].extend(cat_adj.get("omega", []))
            # Periodic historical pattern adjustments
            if len(self.historial_feedback) % 10 == 0:
                hist_adj = self._ajustar_por_patrones_historicos()
                ajustes["alfa"].extend(hist_adj.get("alfa", []))
                ajustes["omega"].extend(hist_adj.get("omega", []))
        except Exception:
            logger.exception("_realizar_ajustes_adaptativos failed")
        return ajustes

    # The rest of the private methods largely mirror previous logic but with guards
    def _ajustar_por_fallo_critico(
        self, feedback: FeedbackEjecucion
    ) -> dict[str, list[str]]:
        ajuste = min(self.max_ajuste_por_ciclo, self.tasa_aprendizaje_base * 2.0)
        self.perfil_alfa.ajustar_parametro(
            "cautela", ajuste, f"Fallo critico {getattr(feedback,'plan_id',None)}"
        )
        self.perfil_alfa.ajustar_parametro(
            "tolerancia_riesgo", -ajuste, "Reducir riesgo tras fallo critico"
        )
        self.perfil_omega.ajustar_parametro(
            "rigor", ajuste * 0.5, "Incrementar rigor tras fallo critico"
        )
        self.perfil_omega.ajustar_parametro(
            "prioridad_analisis", ajuste * 0.3, "Priorizar analisis"
        )
        return {
            "alfa": ["cautela+", "tolerancia_riesgo-"],
            "omega": ["rigor+", "prioridad_analisis+"],
        }

    def _ajustar_por_fallo_menor(
        self, feedback: FeedbackEjecucion
    ) -> dict[str, list[str]]:
        ajuste = min(self.max_ajuste_por_ciclo, self.tasa_aprendizaje_base * 0.5)
        if getattr(feedback, "tasa_exito", 1.0) < 0.6:
            self.perfil_omega.ajustar_parametro("rigor", ajuste, "Baja tasa de exito")
            omega_adj = ["rigor+"]
        else:
            omega_adj = []
        alfa_adj = []
        if len(getattr(feedback, "errores", []) or []) > 2:
            self.perfil_alfa.ajustar_parametro(
                "optimismo", -ajuste, "Multiples errores"
            )
            alfa_adj.append("optimismo-")
        return {"alfa": alfa_adj, "omega": omega_adj}

    def _ajustar_por_exito(self, feedback: FeedbackEjecucion) -> dict[str, list[str]]:
        ajuste = min(self.max_ajuste_por_ciclo, self.tasa_aprendizaje_base * 0.3)
        applied = {"alfa": [], "omega": []}
        if (
            getattr(feedback, "tiempo_ejecucion", float("inf"))
            < self.PERFORMANCE_EXECUTION_TIME_THRESHOLD
            and getattr(feedback, "tasa_exito", 0.0)
            >= self.PERFORMANCE_SUCCESS_RATE_THRESHOLD
        ):
            self.perfil_alfa.ajustar_parametro(
                "audacia", ajuste, "Exito rapido y eficiente"
            )
            applied["alfa"].append("audacia+")
        calidad = _safe_mean(
            list(getattr(feedback, "metricas_calidad", {}).values()), 0.5
        )
        if calidad > self.PERFORMANCE_HIGH_QUALITY_THRESHOLD:
            self.perfil_omega.ajustar_parametro("rigor", ajuste * 0.5, "Alta calidad")
            applied["omega"].append("rigor+")
        return applied

    def _ajustar_por_categoria_error(
        self, categoria: CategoriaError
    ) -> dict[str, list[str]]:
        ajuste = min(self.max_ajuste_por_ciclo, self.tasa_aprendizaje_base)
        result = {"alfa": [], "omega": []}
        if categoria == CategoriaError.VALIDACION:
            self.perfil_omega.ajustar_parametro("rigor", ajuste, "Error de validacion")
            self.perfil_omega.ajustar_parametro(
                "cautela", ajuste * 0.5, "Error de validacion"
            )
            result["omega"].extend(["rigor+", "cautela+"])
        elif categoria == CategoriaError.CREATIVIDAD:
            self.perfil_alfa.ajustar_parametro(
                "creatividad", ajuste, "Error de creatividad"
            )
            result["alfa"].append("creatividad+")
            if self.perfil_omega.cautela > 0.8:
                self.perfil_omega.ajustar_parametro(
                    "cautela", -ajuste * 0.3, "Reducir exceso cautela"
                )
                result["omega"].append("cautela-")
        elif categoria == CategoriaError.DEPENDENCIAS:
            self.perfil_alfa.ajustar_parametro(
                "prioridad_analisis", ajuste, "Error dependencias"
            )
            self.perfil_omega.ajustar_parametro(
                "prioridad_analisis", ajuste * 0.5, "Error dependencias"
            )
            result["alfa"].append("prioridad_analisis+")
            result["omega"].append("prioridad_analisis+")
        elif categoria == CategoriaError.TIEMPO:
            self.perfil_alfa.ajustar_parametro(
                "prioridad_accion", ajuste * 0.5, "Error de tiempo"
            )
            self.perfil_omega.ajustar_parametro(
                "prioridad_analisis", -ajuste * 0.3, "Optimizar tiempo"
            )
            result["alfa"].append("prioridad_accion+")
            result["omega"].append("prioridad_analisis-")
        return result

    def _ajustar_por_patrones_historicos(self) -> dict[str, list[str]]:
        ajustes = {"alfa": [], "omega": []}
        recientes = list(self.historial_feedback)[-100:]
        patrones_fallo = self.analizador_patrones.analizar_patrones_fallo(recientes)
        for cat, anal in patrones_fallo.items():
            if anal["frecuencia"] >= 3:
                for rec in anal.get("recomendaciones", []):
                    applied = self._aplicar_recomendacion(
                        rec, anal["severidad_promedio"]
                    )
                    ajustes["alfa"].extend(applied.get("alfa", []))
                    ajustes["omega"].extend(applied.get("omega", []))
        patrones_exito = self.analizador_patrones.analizar_patrones_exito(recientes)
        if patrones_exito:
            ref = self._reforzar_factores_exito(patrones_exito)
            ajustes["alfa"].extend(ref.get("alfa", []))
            ajustes["omega"].extend(ref.get("omega", []))
        return ajustes

    def _aplicar_recomendacion(
        self, recomendacion: str, severidad: float
    ) -> dict[str, list[str]]:
        intensidad = min(
            self.max_ajuste_por_ciclo, self.tasa_aprendizaje_base * float(severidad)
        )
        rec = recomendacion.lower()
        result = {"alfa": [], "omega": []}
        if "incrementar rigor" in rec and "omega" in rec:
            self.perfil_omega.ajustar_parametro("rigor", intensidad, recomendacion)
            result["omega"].append("rigor+")
        elif "incrementar creatividad" in rec and "alfa" in rec:
            self.perfil_alfa.ajustar_parametro("creatividad", intensidad, recomendacion)
            result["alfa"].append("creatividad+")
        elif "reducir tolerancia al riesgo" in rec:
            self.perfil_alfa.ajustar_parametro(
                "tolerancia_riesgo", -intensidad, recomendacion
            )
            result["alfa"].append("tolerancia_riesgo-")
        elif "prioridad de análisis" in rec:
            self.perfil_alfa.ajustar_parametro(
                "prioridad_analisis", intensidad, recomendacion
            )
            self.perfil_omega.ajustar_parametro(
                "prioridad_analisis", intensidad * 0.5, recomendacion
            )
            result["alfa"].append("prioridad_analisis+")
            result["omega"].append("prioridad_analisis+")
        return result

    def _reforzar_factores_exito(
        self, patrones_exito: dict[str, Any]
    ) -> dict[str, list[str]]:
        ajustes = {"alfa": [], "omega": []}
        intensidad = self.tasa_aprendizaje_base * 0.5
        ctx = patrones_exito.get("contextos_exitosos", {})
        balance = ctx.get("balance_promedio", None)
        if balance is not None:
            if balance > 0.6:
                self.perfil_alfa.ajustar_parametro(
                    "audacia", intensidad, "Patron exito: balance alto"
                )
                ajustes["alfa"].append("audacia+")
            elif balance < 0.4:
                self.perfil_omega.ajustar_parametro(
                    "cautela", intensidad, "Patron exito: balance bajo"
                )
                ajustes["omega"].append("cautela+")
        return ajustes

    def _actualizar_metricas_aprendizaje(
        self, feedback: FeedbackEjecucion, ajustes: dict[str, list[str]]
    ) -> None:
        self.metricas_aprendizaje["puntuaciones_exito"].append(
            {
                "valor": getattr(feedback, "puntuacion_exito", 0.0),
                "timestamp": datetime.now(),
            }
        )
        self.metricas_aprendizaje["ajustes_por_ciclo"].append(
            {
                "total_ajustes": len(ajustes["alfa"]) + len(ajustes["omega"]),
                "timestamp": datetime.now(),
            }
        )
        # update perfil rendimiento
        self.perfil_alfa.registrar_rendimiento(
            getattr(feedback, "puntuacion_exito", 0.0),
            str(getattr(feedback, "contexto_ejecucion", {})),
        )
        self.perfil_omega.registrar_rendimiento(
            getattr(feedback, "puntuacion_exito", 0.0),
            str(getattr(feedback, "contexto_ejecucion", {})),
        )

    # Reporting & diagnostics
    def obtener_parametros_mentes(self) -> dict[str, dict[str, float]]:
        return {
            "alfa": {
                "audacia": self.perfil_alfa.audacia,
                "cautela": self.perfil_alfa.cautela,
                "creatividad": self.perfil_alfa.creatividad,
                "rigor": self.perfil_alfa.rigor,
                "optimismo": self.perfil_alfa.optimismo,
                "prioridad_analisis": self.perfil_alfa.prioridad_analisis,
                "prioridad_accion": self.perfil_alfa.prioridad_accion,
                "tolerancia_riesgo": self.perfil_alfa.tolerancia_riesgo,
                "confianza_base": self.perfil_alfa.confianza_base,
            },
            "omega": {
                "audacia": self.perfil_omega.audacia,
                "cautela": self.perfil_omega.cautela,
                "creatividad": self.perfil_omega.creatividad,
                "rigor": self.perfil_omega.rigor,
                "optimismo": self.perfil_omega.optimismo,
                "prioridad_analisis": self.perfil_omega.prioridad_analisis,
                "prioridad_accion": self.perfil_omega.prioridad_accion,
                "tolerancia_riesgo": self.perfil_omega.tolerancia_riesgo,
                "confianza_base": self.perfil_omega.confianza_base,
            },
        }

    def generar_reporte_aprendizaje(self, ventana_dias: int = 30) -> dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=ventana_dias)
        feedbacks = [
            f
            for f in self.historial_feedback
            if getattr(f, "timestamp", datetime.now()) >= cutoff
        ]
        if not feedbacks:
            return {
                "mensaje": "No hay datos suficientes en la ventana temporal especificada"
            }
        puntuaciones = [getattr(f, "puntuacion_exito", 0.0) for f in feedbacks]
        patrones_fallo = self.analizador_patrones.analizar_patrones_fallo(feedbacks)
        patrones_exito = self.analizador_patrones.analizar_patrones_exito(feedbacks)
        return {
            "periodo": f"Ultimos {ventana_dias} dias",
            "estadisticas": {
                "total_planes": len(feedbacks),
                "tasa_exito": sum(
                    1 for f in feedbacks if f.resultado == TipoResultado.EXITO_TOTAL
                )
                / len(feedbacks),
                "tasa_fallo_critico": sum(
                    1 for f in feedbacks if f.resultado == TipoResultado.FALLO_CRITICO
                )
                / len(feedbacks),
                "puntuacion_promedio": float(_safe_mean(puntuaciones, 0.0)),
                "tendencia": self._calcular_tendencia_puntuaciones(puntuaciones),
            },
            "evolucion_parametros": {
                "alfa": {
                    "ajustes": len(self.perfil_alfa.ajustes_historia),
                    "rendimiento": self.perfil_alfa.calcular_rendimiento_promedio(7),
                },
                "omega": {
                    "ajustes": len(self.perfil_omega.ajustes_historia),
                    "rendimiento": self.perfil_omega.calcular_rendimiento_promedio(7),
                },
            },
            "patrones_fallo": {
                k.name if hasattr(k, "name") else str(k): v["frecuencia"]
                for k, v in patrones_fallo.items()
            },
            "patrones_exito_detectados": bool(patrones_exito),
        }

    def _calcular_tendencia_puntuaciones(self, puntuaciones: list[float]) -> str:
        if len(puntuaciones) < 5:
            return "Insuficiente"
        mitad = len(puntuaciones) // 2
        primera = float(_safe_mean(puntuaciones[:mitad], 0.0))
        segunda = float(_safe_mean(puntuaciones[mitad:], 0.0))
        diff = segunda - primera
        if diff > 0.1:
            return "Mejorando significativamente"
        if diff > 0.05:
            return "Mejorando ligeramente"
        if diff < -0.1:
            return "Empeorando significativamente"
        if diff < -0.05:
            return "Empeorando ligeramente"
        return "Estable"

    def _calcular_metricas_convergencia(self) -> dict[str, float]:
        estabilidad_alfa = self._calcular_estabilidad_parametros(self.perfil_alfa)
        estabilidad_omega = self._calcular_estabilidad_parametros(self.perfil_omega)
        coherencia = self._calcular_coherencia_mentes()
        return {
            "estabilidad_alfa": estabilidad_alfa,
            "estabilidad_omega": estabilidad_omega,
            "coherencia_colaboracion": coherencia,
            "convergencia_general": (estabilidad_alfa + estabilidad_omega + coherencia)
            / 3.0,
        }

    def _calcular_estabilidad_parametros(self, perfil: PerfilMente) -> float:
        ajustes = list(perfil.ajustes_historia)
        if len(ajustes) < 5:
            return 0.5
        magnitudes = [abs(a["ajuste"]) for a in ajustes[-10:]]
        var = float(np.var(magnitudes)) if magnitudes else 0.0
        return (
            max(0.0, 1.0 - var / (self.max_ajuste_por_ciclo**2))
            if self.max_ajuste_por_ciclo > 0
            else 0.5
        )

    def _calcular_coherencia_mentes(self) -> float:
        if len(self.historial_feedback) < 10:
            return 0.5
        rec = list(self.historial_feedback)[-50:]
        puntuaciones = [getattr(f, "puntuacion_exito", 0.5) for f in rec]
        if not puntuaciones:
            return 0.5
        std = float(np.std(puntuaciones))
        coherencia = max(0.0, min(1.0, 1.0 - std))
        return coherencia

    def _generar_predicciones_rendimiento(self) -> dict[str, Any]:
        recientes = [
            getattr(f, "puntuacion_exito", 0.0)
            for f in list(self.historial_feedback)[-20:]
        ]
        if len(recientes) < 3:
            return {"mensaje": "Datos insuficientes para predicciones confiables"}
        x = list(range(len(recientes)))
        coef = float(np.polyfit(x, recientes, 1)[0])
        prox = max(0.0, min(1.0, recientes[-1] + coef))
        return {
            "tendencia": (
                "positiva"
                if coef > 0.01
                else ("negativa" if coef < -0.01 else "estable")
            ),
            "prediccion_proximo_plan": prox,
            "confianza_prediccion": min(0.8, len(recientes) / 20.0),
        }

    # Reset API (preserved logic, hardened)
    def reset_parametros(
        self, mente: str = "ambas", factor_reset: float = 0.5
    ) -> dict[str, str]:
        with self._lock:
            resultados = {}
            if mente in ("alfa", "ambas"):
                defaults_alfa = {
                    "audacia": 0.8,
                    "cautela": 0.3,
                    "creatividad": 0.9,
                    "rigor": 0.4,
                    "optimismo": 0.8,
                    "prioridad_analisis": 0.3,
                    "prioridad_accion": 0.8,
                    "tolerancia_riesgo": 0.7,
                    "confianza_base": 0.7,
                }
                for p, d in defaults_alfa.items():
                    valor_actual = getattr(self.perfil_alfa, p)
                    setattr(
                        self.perfil_alfa,
                        p,
                        valor_actual * factor_reset + d * (1.0 - factor_reset),
                    )
                resultados["alfa"] = "Reset aplicado"
            if mente in ("omega", "ambas"):
                defaults_omega = {
                    "audacia": 0.2,
                    "cautela": 0.9,
                    "creatividad": 0.3,
                    "rigor": 0.9,
                    "optimismo": 0.4,
                    "prioridad_analisis": 0.9,
                    "prioridad_accion": 0.2,
                    "tolerancia_riesgo": 0.2,
                    "confianza_base": 0.6,
                }
                for p, d in defaults_omega.items():
                    valor_actual = getattr(self.perfil_omega, p)
                    setattr(
                        self.perfil_omega,
                        p,
                        valor_actual * factor_reset + d * (1.0 - factor_reset),
                    )
                resultados["omega"] = "Reset aplicado"
            if factor_reset < 0.3:
                if mente in ("alfa", "ambas"):
                    self.perfil_alfa.ajustes_historia.clear()
                if mente in ("omega", "ambas"):
                    self.perfil_omega.ajustes_historia.clear()
                resultados["historial"] = "Limpiado"
            logger.info("reset_parametros applied for %s", mente)
            return resultados

    def obtener_estado_completo(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "parametros_alfa": self.obtener_parametros_mentes()["alfa"],
                "parametros_omega": self.obtener_parametros_mentes()["omega"],
                "total_feedbacks_procesados": len(self.historial_feedback),
                "ajustes_alfa": len(self.perfil_alfa.ajustes_historia),
                "ajustes_omega": len(self.perfil_omega.ajustes_historia),
            }
