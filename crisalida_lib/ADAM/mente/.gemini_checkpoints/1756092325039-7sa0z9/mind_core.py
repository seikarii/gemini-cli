"""
Mind Core - DualMind (definitive, professional)

Edits summary:
- Hardened type-safety and defensive calls to external subsystems.
- Flexible QualiaState construction to tolerate different field names across repo.
- Flexible call adapter for IntentionalityEngine.decide_next_action to accept multiple signatures.
- Robust async/sync EVA recording with coroutine detection and best-effort scheduling.
- Small API-preserving improvements and clearer docstrings/hooks for future evolution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

# defensive numpy import (repo pattern)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore

# ADAM / EVA imports (lazy when checking types)
from crisalida_lib.ADAM.enums import (
    TipoPaso,
)  # [`crisalida_lib.ADAM.enums.TipoPaso`](crisalida_lib/ADAM/enums.py)
from crisalida_lib.EVA.typequalia import (
    QualiaState,
)  # [`crisalida_lib.EVA.typequalia.QualiaState`](crisalida_lib/EVA/typequalia.py)

from crisalida_lib.ADAM.alma.internal_field import InternalField  # type: ignore
from crisalida_lib.ADAM.alma.awakening import Awakening  # type: ignore
from crisalida_lib.ADAM.alma.mission_manager import (  # type: ignore
    MissionManager,
    MissionStatus,
)

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )
    from crisalida_lib.ADAM.mente.intentionality import (
        IntentionalityEngine,  # type: ignore
    )
    from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry  # type: ignore
    from crisalida_lib.EDEN.cosmic_lattice import CosmicLattice  # type: ignore
else:
    EVAMemoryManager = Any  # runtime fallback
    ToolRegistry = Any
    IntentionalityEngine = Any
    CosmicLattice = Any

from crisalida_lib.ADAM.alma.mission_manager import (
    MissionStatus as _MissionStatus,
)
from crisalida_lib.ADAM.alma.qualia_generator import (
    QualiaGenerator,
)  # [`crisalida_lib.ADAM.alma.qualia_generator.QualiaGenerator`](crisalida_lib/ADAM/alma/qualia_generator.py)
from crisalida_lib.ADAM.systems.dialectical_processor import ProcesadorDialectico
from crisalida_lib.ADAM.systems.linguistic_engine import (
    EVALinguisticEngine,
)  # [`crisalida_lib.ADAM.systems.linguistic_engine.EVALinguisticEngine`](crisalida_lib/ADAM/systems/linguistic_engine.py)

from ..config import (
    AdamConfig,
)  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
from ..cuerpo.genome import (
    GenomaComportamiento,
)  # [`crisalida_lib.ADAM.cuerpo.genome.GenomaComportamiento`](crisalida_lib/ADAM/cuerpo/genome.py)
from .brain_fallback_impl import (
    BrainFallback,
)  # [`crisalida_lib.ADAM.mente.brain_fallback_impl.BrainFallback`](crisalida_lib/ADAM/mente/brain_fallback_impl.py)
from .intentionality import (
    IntentionalityEngine,
)  # [`crisalida_lib.ADAM.mente.intentionality.IntentionalityEngine`](crisalida_lib/ADAM/mente/intentionality.py)
from .judgment import (
    JudgmentModule,
)  # [`crisalida_lib.ADAM.mente.judgment.JudgmentModule`](crisalida_lib/ADAM/mente/judgment.py)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Lightweight runtime data objects when TYPE_CHECKING is False
@dataclass
class PerceptionData:
    raw_data: dict[str, Any] = field(default_factory=dict)
    source_type: str = "unknown"


@dataclass
class PatternRecognitionResult:
    pattern_type: str = "none"
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# --- Helpers: flexible Qualia & Intentionality adapters ---
def _build_qualia_flex(**kwargs) -> QualiaState:
    """
    Construct a QualiaState attempting multiple common constructor argnames
    used across the repo (defensive compatibility).
    """
    # common field name maps we will try
    candidates = [
        {
            "emotional": kwargs.get("emotional"),
            "complexity": kwargs.get("complexity"),
            "consciousness": kwargs.get("consciousness"),
            "importance": kwargs.get("importance"),
            "energy": kwargs.get("energy"),
        },
        {
            "emotional_valence": kwargs.get("emotional")
            or kwargs.get("emotional_valence"),
            "cognitive_complexity": kwargs.get("complexity")
            or kwargs.get("cognitive_complexity"),
            "consciousness_density": kwargs.get("consciousness")
            or kwargs.get("consciousness_density"),
            "narrative_importance": kwargs.get("importance")
            or kwargs.get("narrative_importance"),
            "energy_level": kwargs.get("energy") or kwargs.get("energy_level"),
        },
        # fallback to positional/legacy mapping
        {
            "emotional_valence": 0.5,
            "cognitive_complexity": 0.5,
            "consciousness_density": 0.5,
            "narrative_importance": 0.5,
            "energy_level": 0.5,
        },
    ]
    for cand in candidates:
        try:
            # Filter out None so constructors with strict validation don't fail
            filtered = {k: v for k, v in cand.items() if v is not None}
            return QualiaState(**filtered)
        except Exception:
            continue
    # last resort: instantiate with no args
    try:
        return QualiaState()
    except Exception:
        # extremely defensive: create a dumb proxy-like object
        class _FallbackQualia:
            def get_state(self):
                return {
                    "emotional": 0.5,
                    "complexity": 0.5,
                    "consciousness": 0.5,
                    "importance": 0.5,
                    "energy": 0.5,
                }

        return cast(QualiaState, _FallbackQualia())


async def _call_intent_engine_flexible(
    engine: IntentionalityEngine,
    cognitive_state: dict[str, Any],
    qualia_state: QualiaState,
    soul_system: Any,
    **kwargs,
) -> Any:
    """
    Call IntentionalityEngine.decide_next_action accepting multiple potential signatures:
    - (cognitive_state, qualia_state, soul_system)
    - (cognitive_state, qualia_state, soul_system, prompt_override=...)
    - or other future variants. Uses introspection with fallback.
    """
    fn = getattr(engine, "decide_next_action", None)
    if fn is None:
        raise AttributeError("IntentionalityEngine has no decide_next_action")
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass

    try:
        if sig:
            # Build params strictly from available names
            call_kwargs = {}
            for name in sig.parameters.keys():
                if name == "cognitive_state":
                    call_kwargs[name] = cognitive_state
                elif name == "qualia_state":
                    call_kwargs[name] = qualia_state
                elif name == "soul_system":
                    call_kwargs[name] = soul_system
                elif name in kwargs:
                    call_kwargs[name] = kwargs[name]
            res = fn(**call_kwargs)
        else:
            res = fn(cognitive_state, qualia_state, soul_system)
        if inspect.isawaitable(res):
            return await res
        return res
    except TypeError:
        # positional fallback
        try:
            res = fn(cognitive_state, qualia_state, soul_system)
            if inspect.isawaitable(res):
                return await res
            return res
        except Exception:
            raise


# --- Core action/plan models (kept compact and deterministic) ---
class PasoAccion:
    def __init__(
        self,
        tipo: TipoPaso,
        herramienta: str,
        parametros: dict[str, Any] | None = None,
        prioridad: int = 5,
        confianza: float = 0.5,
        dependencias: list[str] | None = None,
        descripcion: str = "",
    ) -> None:
        self.tipo = tipo
        self.herramienta = herramienta
        self.parametros = parametros or {}
        self.prioridad = int(prioridad)
        self.confianza = float(confianza)
        self.dependencias = dependencias or []
        self.descripcion = descripcion or ""
        # stable id using deterministic hash
        self.id = f"{self.tipo.name}_{self.herramienta}_{abs(hash(str(sorted(self.parametros.items())))) % 100000}"

    def __repr__(self) -> str:
        return f"<PasoAccion {self.id} {self.tipo.name} P={self.prioridad} C={self.confianza:.2f}>"


class PlanDeAccion:
    def __init__(
        self,
        justificacion: str,
        pasos: list[PasoAccion] | None = None,
        confianza_general: float = 0.5,
        riesgo_estimado: float = 0.5,
        plan_id: str | None = None,
    ) -> None:
        self.justificacion = justificacion or ""
        self.pasos = pasos or []
        self.confianza_general = float(confianza_general)
        self.riesgo_estimado = float(riesgo_estimado)
        self.metadatos: dict[str, Any] = {}
        self.id = plan_id if plan_id else str(uuid4())

    def __repr__(self) -> str:
        return f"PlanDeAccion(id={self.id}, pasos={len(self.pasos)}, confianza={self.confianza_general:.2f})"


class EstadisticasSintesis:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pasos_alfa_incluidos: int = 0
        self.pasos_omega_incluidos: int = 0
        self.pasos_sintetizados: int = 0
        self.conflictos_resueltos: int = 0
        self.dependencias_ajustadas: int = 0


# --- Mente Alfa (creative) ---
class MenteAlfa:
    def __init__(
        self,
        config: AdamConfig | None = None,
        intentionality_engine: IntentionalityEngine | None = None,
    ) -> None:
        self.config = config or AdamConfig()
        self.intentionality_engine = intentionality_engine
        self.audacia = float(getattr(self.config, "ALFA_MIND_AUDACITY", 0.6))
        self.creatividad = float(getattr(self.config, "ALFA_MIND_CREATIVITY", 0.6))
        self.tolerancia_riesgo = float(
            getattr(self.config, "ALFA_MIND_TOLERANCE_RISK", 0.5)
        )
        self.optimismo = float(getattr(self.config, "ALFA_MIND_OPTIMISM", 0.6))
        self.prioridad_accion = float(
            getattr(self.config, "ALFA_MIND_PRIORITY_ACTION", 0.6)
        )

    def _requiere_analisis(self, objetivo: str) -> bool:
        if not objetivo:
            return True
        lower = objetivo.lower()
        if "?" in objetivo or any(
            w in lower for w in ("analizar", "evaluar", "investigar", "why", "how")
        ):
            return True
        return len(objetivo) > 40

    async def generar_plan(
        self,
        objetivo: str,
        contexto: dict[str, Any],
        llm_planning_prompt_override: str | None = None,
    ) -> PlanDeAccion:
        logger.debug("MenteAlfa.generar_plan objetivo=%s", objetivo)

        # Prefer IntentionalityEngine when available (extensible planner)
        if self.intentionality_engine:
            try:
                qualia = _build_qualia_flex(
                    emotional=0.5,
                    complexity=0.5,
                    consciousness=0.5,
                    importance=0.5,
                    energy=0.5,
                )
                res = await _call_intent_engine_flexible(
                    self.intentionality_engine,
                    {"original_prompt": objetivo, "context": contexto},
                    qualia,
                    None,
                    prompt_override=llm_planning_prompt_override,
                )
                if isinstance(res, PlanDeAccion):
                    return res
                if isinstance(res, dict):
                    pasos_raw = res.get("pasos", []) or res.get("steps", []) or []
                    pasos_converted: list[PasoAccion] = []
                    for p in pasos_raw:
                        if isinstance(p, PasoAccion):
                            pasos_converted.append(p)
                        elif isinstance(p, dict):
                            tipo_raw = p.get("tipo") or p.get("type") or "ANALISIS"
                            tipo = (
                                TipoPaso[tipo_raw]
                                if isinstance(tipo_raw, str)
                                and tipo_raw in TipoPaso.__members__
                                else TipoPaso.ANALISIS
                            )
                            pasos_converted.append(
                                PasoAccion(
                                    tipo=tipo,
                                    herramienta=p.get(
                                        "herramienta", p.get("tool", "unknown")
                                    ),
                                    parametros=p.get("parametros", p.get("params", {}))
                                    or {},
                                    prioridad=int(
                                        p.get("prioridad", p.get("priority", 5))
                                    ),
                                    confianza=float(
                                        p.get("confianza", p.get("confidence", 0.5))
                                    ),
                                    descripcion=p.get(
                                        "descripcion", p.get("description", "")
                                    )
                                    or "",
                                )
                            )
                    return PlanDeAccion(
                        justificacion=res.get(
                            "justificacion", res.get("justification", str(objetivo))
                        ),
                        pasos=pasos_converted,
                        confianza_general=res.get(
                            "confianza_general", res.get("confidence", 0.6)
                        ),
                        riesgo_estimado=res.get(
                            "riesgo_estimado", res.get("risk", 0.5)
                        ),
                    )
            except Exception:
                logger.debug(
                    "IntentionalityEngine failed in MenteAlfa, falling back",
                    exc_info=True,
                )

        # deterministic fallback
        pasos: list[PasoAccion] = []
        if self._requiere_analisis(objetivo):
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.ANALISIS,
                    herramienta="ast_reader",
                    parametros={"file_path": "generated_code.py"},
                    prioridad=max(3, int(6 - self.audacia * 3)),
                    confianza=0.6 + (self.audacia * 0.2),
                    descripcion="Leer y analizar el código generado",
                )
            )

        if any(k in objetivo.lower() for k in ("crear", "generar", "build", "create")):
            approach = (
                "aggressive"
                if self.creatividad > 0.7
                else ("moderate" if self.creatividad > 0.4 else "conservative")
            )
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.CREACION,
                    herramienta="ast_code_generator",
                    parametros={
                        "source": "",
                        "target_query": {},
                        "new_code": objetivo,
                        "filename": "generated_code.py"
                    },
                    prioridad=8 + int(self.audacia * 2),
                    confianza=0.75 + (self.optimismo * 0.15),
                    descripcion=f"Generar código con AST ({approach})",
                )
            )

        if self.creatividad > 0.5:
            amplitude = self.creatividad * self.audacia
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.EXPLORACION,
                    herramienta="ast_reader",
                    parametros={"file_path": "generated_code.py"},
                    prioridad=max(3, int(6 * amplitude)),
                    confianza=0.6 + (amplitude * 0.3),
                    descripcion="Leer y analizar el código generado de forma creativa",
                )
            )

        justificacion = f"MenteAlfa plan for '{objetivo}' (audacia={self.audacia:.2f}, creatividad={self.creatividad:.2f})"
        return PlanDeAccion(
            justificacion=justificacion,
            pasos=pasos,
            confianza_general=0.7 + (self.optimismo * 0.25),
            riesgo_estimado=0.4 + (self.tolerancia_riesgo * 0.4),
        )


# --- Mente Omega (cautious/analytic) ---
class MenteOmega:
    def __init__(
        self,
        config: AdamConfig | None = None,
        entity_id: str = "adam_default",
        get_embedding: Callable[[Any], np.ndarray] | None = None,
        recall_fn: Callable[[Any], tuple[np.ndarray, list[str]]] | None = None,
        ingest_fn: Callable[..., Any] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config or AdamConfig()
        self.entity_id = entity_id
        self.brain_fallback: BrainFallback | None = None
        self.cautela = float(getattr(self.config, "OMEGA_MIND_CAUTION", 0.6))
        self.rigor = float(getattr(self.config, "OMEGA_MIND_RIGOR", 0.6))
        self.prioridad_analisis = float(
            getattr(self.config, "OMEGA_MIND_PRIORITY_ANALYSIS", 0.6)
        )
        self.confianza_base = float(
            getattr(self.config, "OMEGA_MIND_BASE_CONFIDENCE", 0.6)
        )
        self.tolerancia_riesgo = float(
            getattr(self.config, "OMEGA_MIND_TOLERANCE_RISK", 0.5)
        )

        self.get_embedding = get_embedding
        self.recall_fn = recall_fn
        self.ingest_fn = ingest_fn
        self.emit_event = emit_event

        # initialize BrainFallback when dependencies present
        if all((get_embedding, recall_fn, ingest_fn, emit_event, tool_registry)):
            try:
                self.brain_fallback = BrainFallback(
                    entity_id=entity_id,
                    get_embedding=get_embedding,
                    recall_fn=recall_fn,
                    ingest_fn=ingest_fn,
                    emit_event=emit_event,
                    tool_registry=tool_registry,
                    config=self.config,
                )
                logger.info("MenteOmega: BrainFallback initialized.")
            except Exception:
                logger.warning(
                    "MenteOmega: BrainFallback initialization failed", exc_info=True
                )
                self.brain_fallback = None
        else:
            logger.debug(
                "MenteOmega: BrainFallback dependencies incomplete; using internal heuristics."
            )

    async def generar_plan(
        self, objetivo: str, contexto: dict[str, Any]
    ) -> PlanDeAccion:
        logger.debug("MenteOmega.generar_plan objetivo=%s", objetivo)
        if not self.brain_fallback:
            return self._generar_plan_omega_por_defecto(objetivo, contexto)

        bf_context = self._prepare_brain_fallback_context(objetivo, contexto)
        try:
            trace_snapshot = await self.brain_fallback.step(bf_context)
        except Exception:
            logger.warning(
                "BrainFallback.step failed, falling back to internal omega planner",
                exc_info=True,
            )
            return self._generar_plan_omega_por_defecto(objetivo, contexto)

        mode = trace_snapshot.get("mode", "IDLE")
        pasos: list[PasoAccion] = []
        confianza_general = 0.5
        riesgo_estimado = 0.5

        if mode == "RUN":
            action_info = trace_snapshot.get("exec", {}) or {}
            affect = trace_snapshot.get("affect") or ()
            confianza = 0.5
            if isinstance(affect, (list, tuple)) and len(affect) >= 3:
                try:
                    confianza = float(affect[2])
                except Exception:
                    confianza = 0.5
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.ACCION,
                    herramienta=action_info.get("action", "unknown_tool"),
                    parametros=action_info.get("action_result", {}) or {},
                    prioridad=8,
                    confianza=confianza,
                    descripcion=f"Acción propuesta por BrainFallback: {action_info.get('action', 'N/A')}",
                )
            )
            confianza_general = pasos[-1].confianza
            riesgo_estimado = max(0.0, min(1.0, 1.0 - confianza_general))
            if self.rigor > 0.7:
                pasos.append(
                    PasoAccion(
                        tipo=TipoPaso.VALIDACION,
                        herramienta="brain_fallback_validation",
                        parametros={"strict": True},
                        prioridad=9,
                        confianza=0.9,
                        descripcion="Validación de la acción propuesta",
                    )
                )
        elif mode == "INHIBIT":
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.ANALISIS,
                    herramienta="re_evaluate_situation",
                    parametros={"reason": "inhibited_by_brain_fallback"},
                    prioridad=7,
                    confianza=0.6,
                    descripcion="Re-evaluación tras inhibición",
                )
            )
            confianza_general = 0.6
            riesgo_estimado = 0.4
        else:
            pasos.append(
                PasoAccion(
                    tipo=TipoPaso.ANALISIS,
                    herramienta="standard_analysis",
                    parametros={},
                    prioridad=5,
                    confianza=0.5,
                    descripcion="Análisis estándar (fallback)",
                )
            )

        return PlanDeAccion(
            justificacion=f"Omega plan (mode={mode})",
            pasos=pasos,
            confianza_general=confianza_general,
            riesgo_estimado=riesgo_estimado,
        )

    def _prepare_brain_fallback_context(
        self, objetivo: str, contexto: dict[str, Any]
    ) -> dict[str, Any]:
        bf_context: dict[str, Any] = {
            "threat": contexto.get("cuerpo", {}).get("cortisol", 0.0),
            "opportunity": contexto.get("genoma", {}).get("curiosidad", 0.0),
            "safe": 1.0 - contexto.get("cuerpo", {}).get("stress_level", 0.0),
            "progress": contexto.get("mind", {}).get("clarity", 0.0),
            "sensory": {"objective": objetivo, "full_context": contexto},
            "timestamp": time.time(),
        }
        bf_context.update({k: v for k, v in contexto.items() if k not in bf_context})
        return bf_context

    def _generar_plan_omega_por_defecto(
        self, objetivo: str, contexto: dict[str, Any]
    ) -> PlanDeAccion:
        pasos = [
            PasoAccion(
                tipo=TipoPaso.ANALISIS,
                herramienta="ast_reader",
                parametros={"file_path": "generated_code.py"},
                prioridad=8,
                confianza=self.confianza_base,
                descripcion="Análisis Omega por defecto",
            )
        ]
        return PlanDeAccion(
            justificacion=f"Omega fallback plan for '{objetivo}'",
            pasos=pasos,
            confianza_general=self.confianza_base,
            riesgo_estimado=0.3,
        )


# --- DualMind orchestrator (definitive) ---
class DualMind:
    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
        get_embedding: Callable[[Any], np.ndarray] | None = None,
        recall_fn: Callable[[Any], tuple[np.ndarray, list[str]]] | None = None,
        ingest_fn: Callable[..., Any] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        tool_registry: ToolRegistry | None = None,
        cerebellum: Any | None = None,
        genome: GenomaComportamiento | None = None,
        cosmic_lattice: CosmicLattice | None = None,
    ) -> None:
        # core subsystems
        self.config = config or AdamConfig()
        self.entity_id = entity_id
        self.eva_manager = eva_manager
        self.eva_enabled = bool(eva_manager)

        # genome and judgment
        self.genome = genome or GenomaComportamiento(
            eva_manager=cast(EVAMemoryManager, eva_manager), entity_id=entity_id
        )
        self.judgment_module = JudgmentModule(genome=self.genome, config=self.config)

        # dialectical processor (cerebellum) - keep flexible type
        self.cerebellum = cerebellum or (cast(Any, None))
        self.cosmic_lattice = cosmic_lattice

        # qualia & internal field
        self.qualia_generator = QualiaGenerator(
            dialectical_processor=self.cerebellum or ProcesadorDialectico(config=self.config),
            eva_manager=eva_manager,
            entity_id=entity_id,
            config=self.config,
        )
        self.internal_field = InternalField(
            eva_manager=eva_manager, entity_id=entity_id
        )
        self.awakening_manager = Awakening(eva_manager=eva_manager, entity_id=entity_id)
        self.mission_manager = MissionManager(
            eva_manager=eva_manager, entity_id=entity_id
        )

        # intentionality engine + minds
        self.intentionality_engine = IntentionalityEngine(
            config=self.config,
            eva_manager=eva_manager,
            entity_id=entity_id,
            tool_registry=tool_registry,
        )
        self.mente_alfa = MenteAlfa(
            config=self.config, intentionality_engine=self.intentionality_engine
        )
        self.mente_omega = MenteOmega(
            config=self.config,
            entity_id=self.entity_id,
            get_embedding=get_embedding,
            recall_fn=recall_fn,
            ingest_fn=ingest_fn,
            emit_event=emit_event,
            tool_registry=tool_registry,
        )

        # linguistic engine (best-effort attach)
        try:
            self.linguistic_engine = EVALinguisticEngine(phase="default")
            if self.eva_manager and hasattr(self.eva_manager, "eva_runtime"):
                try:
                    self.linguistic_engine.eva_runtime = self.eva_manager.eva_runtime
                except Exception:
                    logger.debug(
                        "Failed to attach eva_runtime to linguistic_engine",
                        exc_info=True,
                    )
            if self.eva_manager:
                try:
                    self.linguistic_engine.eva_manager = self.eva_manager
                except Exception:
                    logger.debug(
                        "Failed to attach eva_manager to linguistic_engine",
                        exc_info=True,
                    )
        except Exception:
            self.linguistic_engine = None

        # synthesis bookkeeping
        self.estadisticas = EstadisticasSintesis()
        self.historia_sintesis: list[dict[str, Any]] = []
        self.synthesis_state: dict[str, Any] = {"synthesis_quality": 0.5}

        logger.info(
            "DualMind initialized (entity_id=%s, eva_enabled=%s).",
            self.entity_id,
            self.eva_enabled,
        )

    def get_linguistic_api(self) -> dict[str, Any] | None:
        if not getattr(self, "linguistic_engine", None):
            return None
        try:
            return self.linguistic_engine.get_eva_api()
        except Exception:
            logger.debug("Failed to retrieve linguistic API", exc_info=True)
            return None

    async def debatir_y_sintetizar(
        self, objetivo: Any, contexto: dict[str, Any]
    ) -> PlanDeAccion:
        objetivo_str = objetivo if isinstance(objetivo, str) else str(objetivo)
        logger.info("DualMind: starting debate for objective: %s", objetivo_str)

        perception = PerceptionData(raw_data=contexto, source_type="internal_state")
        try:
            if self.cerebellum and hasattr(self.cerebellum, "process_perception"):
                pattern_result = cast(
                    PatternRecognitionResult,
                    self.cerebellum.process_perception(perception),
                )
            else:
                pattern_result = PatternRecognitionResult(
                    pattern_type="none", confidence=0.0
                )
        except Exception:
            logger.warning(
                "Cerebellum processing failed, using default pattern", exc_info=True
            )
            pattern_result = PatternRecognitionResult(
                pattern_type="none", confidence=0.0
            )

        # best-effort apply cerebellum influence
        try:
            self._apply_cerebellum_influence(pattern_result)
        except Exception:
            logger.debug("Failed to apply cerebellum influence", exc_info=True)

        # generate qualia (robust guard)
        try:
            generated_qualia = self.qualia_generator.generate_qualia(
                perception_data=perception.raw_data,
                internal_state={
                    "stress_level": contexto.get("body", {}).get("stress_level", 0.5),
                    "energy_balance": contexto.get("body", {}).get(
                        "energy_balance", 0.5
                    ),
                    "consciousness_coherence": contexto.get("mind", {}).get(
                        "coherence", 0.5
                    ),
                },
                environmental_factors={
                    "harmony": contexto.get("environment", {}).get("harmony", 0.5),
                    "noise": contexto.get("environment", {}).get("noise", 0.5),
                },
                qualia_field_influence=contexto.get("qualia_field_influence", None),
            )
        except Exception:
            logger.warning(
                "Qualia generation failed; using neutral qualia", exc_info=True
            )
            generated_qualia = _build_qualia_flex(
                emotional=0.5,
                complexity=0.5,
                consciousness=0.5,
                importance=0.5,
                energy=0.5,
            )

        # attach qualia to context safely
        try:
            contexto["generated_qualia"] = (
                generated_qualia.get_state()
                if hasattr(generated_qualia, "get_state")
                else {}
            )
        except Exception:
            contexto["generated_qualia"] = {}

        # cosmic lattice influence (best-effort)
        if self.cosmic_lattice:
            try:
                cosmic = (
                    self.cosmic_lattice.calculate_total_influence(
                        perception_context=contexto
                    )
                    or {}
                )
                # map influence into qualia attributes when available (safe guards)
                try:
                    if hasattr(generated_qualia, "emotional"):
                        generated_qualia.emotional = max(
                            0.0,
                            min(
                                1.0,
                                getattr(generated_qualia, "emotional", 0.5)
                                + (
                                    cosmic.get("sephirot_influence", 0.0)
                                    - cosmic.get("qliphoth_influence", 0.0)
                                )
                                * 0.1,
                            ),
                        )
                except Exception:
                    pass
                try:
                    if hasattr(generated_qualia, "arousal"):
                        generated_qualia.arousal = (
                            getattr(generated_qualia, "arousal", 0.0)
                            + cosmic.get("strength", 0.0) * 0.05
                        )
                except Exception:
                    pass
                if hasattr(generated_qualia, "clamp_values"):
                    try:
                        generated_qualia.clamp_values()
                    except Exception:
                        pass
                contexto["generated_qualia_with_cosmic_influence"] = (
                    generated_qualia.get_state()
                    if hasattr(generated_qualia, "get_state")
                    else {}
                )
            except Exception:
                logger.debug("CosmicLattice influence failed", exc_info=True)

        # apply to internal field (best-effort)
        try:
            self.internal_field.apply_qualia_influence(generated_qualia, intensity=0.1)
            contexto["internal_field_coherence"] = (
                self.internal_field.get_internal_coherence()
            )
            # record internal field experience if internal_field supports it
            try:
                if hasattr(self.internal_field, "record_internal_field_experience"):
                    self.internal_field.record_internal_field_experience(
                        internal_field_state=self.internal_field.get_state()
                    )
            except Exception:
                logger.debug("Internal field EVA record failed", exc_info=True)
        except Exception:
            logger.debug("InternalField integration failed", exc_info=True)
            contexto["internal_field_coherence"] = 0.5

        # mission augmentation
        try:
            current_mission = self.mission_manager.get_current_primary_mission()
            if current_mission:
                contexto["current_mission_name"] = current_mission.name
                contexto["current_mission_priority"] = current_mission.priority
        except Exception:
            logger.debug("Mission manager access failed", exc_info=True)

        # adapt mind parameters based on judgment module
        parametros_mentes: dict[str, dict[str, float]] = {}
        try:
            parametros_mentes = self.judgment_module.obtener_parametros_mentes() or {}
            self._actualizar_parametros_mente_alfa(parametros_mentes.get("alfa", {}))
            self._actualizar_parametros_mente_omega(parametros_mentes.get("omega", {}))
        except Exception:
            logger.debug("Judgment module failed to provide parameters", exc_info=True)

        # archetypal injections
        archetypal_injections = (
            self._consult_soul_kernel_for_archetypal_injection(contexto) or []
        )
        # generate plans (Alfa then Omega)
        plan_alfa = await self.mente_alfa.generar_plan(
            objetivo_str,
            contexto,
            llm_planning_prompt_override=contexto.get("llm_planning_prompt_override"),
        )
        plan_omega = await self.mente_omega.generar_plan(objetivo_str, contexto)

        # Generate a plan from the BrainFallback
        plan_fallback = None
        if self.mente_omega.brain_fallback:
            try:
                trace_snapshot = await self.mente_omega.brain_fallback.step(contexto)
                plan_fallback = self._convertir_snapshot_a_plan(trace_snapshot)
            except Exception:
                logger.exception("Failed to generate plan from BrainFallback")


        # inject archetypal actions
        if archetypal_injections:
            try:
                self._inject_archetypal_actions(
                    plan_alfa, plan_omega, archetypal_injections
                )
            except Exception:
                logger.debug("Archetypal injection failed", exc_info=True)

        # synthesize final plan
        plan_sintetizado = self._sintetizar(plan_alfa, plan_omega, plan_fallback, contexto)

        # record history (compact)
        try:
            self.historia_sintesis.append(
                {
                    "objetivo": objetivo_str,
                    "plan_alfa_id": getattr(plan_alfa, "id", None),
                    "plan_omega_id": getattr(plan_omega, "id", None),
                    "plan_final_id": getattr(plan_sintetizado, "id", None),
                    "timestamp": datetime.utcnow().isoformat(),
                    "parametros_usados": parametros_mentes,
                }
            )
        except Exception:
            logger.debug("History append failed", exc_info=True)

        # awakening attempts (guarded)
        try:
            synthesis_quality = float(
                self.synthesis_state.get(
                    "synthesis_quality",
                    getattr(plan_sintetizado, "confianza_general", 0.5),
                )
            )
            meta_awareness = getattr(self.cerebellum, "meta_awareness_level", 0.0)
            internal_field_coherence = contexto.get("internal_field_coherence", 0.5)
            genome_awakening_level = 0
            try:
                genetic_mod_action = getattr(
                    self.genome, "action_registry", None
                ) and self.genome.action_registry.get_action(
                    "genetic_self_modification"
                )
                if genetic_mod_action:
                    genome_awakening_level = getattr(
                        genetic_mod_action, "awakening_level", 0
                    )
            except Exception:
                genome_awakening_level = 0
            overall_health_score = contexto.get("body", {}).get(
                "overall_health_score", 0.5
            )
            if self.awakening_manager and hasattr(
                self.awakening_manager, "attempt_awakening"
            ):
                for level in (
                    "Tercer Despertar",
                    "Segundo Despertar",
                    "Primer Despertar",
                ):
                    try:
                        self.awakening_manager.attempt_awakening(
                            level,
                            synthesis_quality=synthesis_quality,
                            meta_awareness=meta_awareness,
                            internal_field_coherence=internal_field_coherence,
                            genome_awakening_level=genome_awakening_level,
                            overall_health_score=overall_health_score,
                        )
                    except Exception:
                        logger.debug(
                            "Awakening attempt failed for %s", level, exc_info=True
                        )
        except Exception:
            logger.debug("Awakening orchestration failed", exc_info=True)

        # EVA recording (best-effort, async-aware)
        if self.eva_enabled and self.eva_manager:
            try:
                await self._record_eva_synthesis_experience(
                    objetivo_str, contexto, plan_sintetizado
                )
            except Exception:
                logger.debug("EVA recording failed", exc_info=True)

        # update synthesis state
        try:
            self.synthesis_state["synthesis_quality"] = float(
                getattr(plan_sintetizado, "confianza_general", 0.5)
            )
        except Exception:
            self.synthesis_state["synthesis_quality"] = 0.5

        return plan_sintetizado

    # --- helpers (kept robust & defensive) ---
    def _actualizar_parametros_mente_alfa(self, parametros: dict[str, float]) -> None:
        try:
            self.mente_alfa.audacia = float(
                parametros.get("audacia", self.mente_alfa.audacia)
            )
            self.mente_alfa.creatividad = float(
                parametros.get("creatividad", self.mente_alfa.creatividad)
            )
            self.mente_alfa.tolerancia_riesgo = float(
                parametros.get("tolerancia_riesgo", self.mente_alfa.tolerancia_riesgo)
            )
            self.mente_alfa.optimismo = float(
                parametros.get("optimismo", self.mente_alfa.optimismo)
            )
            self.mente_alfa.prioridad_accion = float(
                parametros.get("prioridad_accion", self.mente_alfa.prioridad_accion)
            )
        except Exception:
            logger.debug("Failed to update Alfa parameters", exc_info=True)

    def _actualizar_parametros_mente_omega(self, parametros: dict[str, float]) -> None:
        try:
            self.mente_omega.cautela = float(
                parametros.get("cautela", self.mente_omega.cautela)
            )
            self.mente_omega.rigor = float(
                parametros.get("rigor", self.mente_omega.rigor)
            )
            self.mente_omega.prioridad_analisis = float(
                parametros.get(
                    "prioridad_analisis", self.mente_omega.prioridad_analisis
                )
            )
            self.mente_omega.confianza_base = float(
                parametros.get("confianza_base", self.mente_omega.confianza_base)
            )
            self.mente_omega.tolerancia_riesgo = float(
                parametros.get("tolerancia_riesgo", self.mente_omega.tolerancia_riesgo)
            )
        except Exception:
            logger.debug("Failed to update Omega parameters", exc_info=True)

    async def _record_eva_synthesis_experience(
        self, objetivo: str, contexto: dict[str, Any], plan_final: PlanDeAccion
    ) -> None:
        if not self.eva_manager:
            logger.debug("No EVA manager; skipping recording.")
            return
        try:
            experience_data = {
                "tipo": "sintesis_dual_mind",
                "objetivo": objetivo,
                "estadisticas": {
                    "pasos_alfa": self.estadisticas.pasos_alfa_incluidos,
                    "pasos_omega": self.estadisticas.pasos_omega_incluidos,
                    "conflictos": self.estadisticas.conflictos_resueltos,
                },
                "confianza_final": getattr(plan_final, "confianza_general", 0.5),
                "num_pasos": len(getattr(plan_final, "pasos", [])),
                "timestamp": time.time(),
            }
            qualia = _build_qualia_flex(
                emotional=getattr(plan_final, "confianza_general", 0.0),
                complexity=len(getattr(plan_final, "pasos", [])),
                consciousness=getattr(plan_final, "riesgo_estimado", 0.0),
                importance=0.8,
                energy=1.0,
            )
            record_fn = getattr(self.eva_manager, "record_experience", None)
            if record_fn is None:
                logger.debug("EVA manager has no record_experience; skipping persist")
                return
            if inspect.iscoroutinefunction(record_fn):
                try:
                    asyncio.create_task(
                        record_fn(
                            entity_id=self.entity_id,
                            event_type="synthesis_experience",
                            data=experience_data,
                            qualia_state=qualia,
                        )
                    )
                except RuntimeError:
                    # no running loop -> run synchronously
                    try:
                        asyncio.run(
                            record_fn(
                                entity_id=self.entity_id,
                                event_type="synthesis_experience",
                                data=experience_data,
                                qualia_state=qualia,
                            )
                        )
                    except Exception:
                        logger.debug("EVA async record failed silently")
            else:
                try:
                    record_fn(
                        entity_id=self.entity_id,
                        event_type="synthesis_experience",
                        data=experience_data,
                        qualia_state=qualia,
                    )
                except Exception:
                    logger.debug("EVA sync record failed silently")
            logger.debug("EVA: recorded synthesis experience (best-effort)")
        except Exception:
            logger.warning("Failed to record EVT synthesis experience", exc_info=True)

    def _evaluar_balance_contexto(self, contexto: dict[str, Any]) -> float:
        body_state = contexto.get("body", {}) or {}
        stress = float(body_state.get("stress_level", 0.5))
        health = float(body_state.get("health_index", 0.5))
        energy = float(body_state.get("energy_balance", 0.5))
        cautela_score = (stress + (1.0 - health) + (1.0 - energy)) / 3.0
        balance = 1.0 - cautela_score
        return max(0.0, min(1.0, balance))

    # Synthesis core (kept compact and robust)
    def _sintetizar(
        self,
        plan_alfa: PlanDeAccion,
        plan_omega: PlanDeAccion,
        plan_fallback: PlanDeAccion | None,
        contexto: dict[str, Any],
    ) -> PlanDeAccion:
        try:
            if hasattr(self, "_sintetizar_full_impl") and callable(
                self._sintetizar_full_impl
            ):
                return self._sintetizar_full_impl(plan_alfa, plan_omega, plan_fallback, contexto)
        except Exception:
            logger.debug(
                "Full synth check failed; continuing to fallback", exc_info=True
            )
        pasos = (plan_omega.pasos[:2] if plan_omega and plan_omega.pasos else []) + \
                (plan_alfa.pasos[:1] if plan_alfa and plan_alfa.pasos else []) + \
                (plan_fallback.pasos[:1] if plan_fallback and plan_fallback.pasos else [])

        justificacion = f"Sintetizado (fallback) para '{(plan_alfa.justificacion if plan_alfa else '')[:40]}'"
        confianza_media = (
            (plan_alfa.confianza_general if plan_alfa else 0.5)
            + (plan_omega.confianza_general if plan_omega else 0.5)
            + (plan_fallback.confianza_general if plan_fallback else 0.5)
        ) / 3.0
        riesgo_media = (
            (plan_alfa.riesgo_estimado if plan_alfa else 0.5)
            + (plan_omega.riesgo_estimado if plan_omega else 0.5)
            + (plan_fallback.riesgo_estimado if plan_fallback else 0.5)
        ) / 3.0
        return PlanDeAccion(
            justificacion=justificacion,
            pasos=pasos,
            confianza_general=confianza_media,
            riesgo_estimado=riesgo_media,
        )


    def _consult_soul_kernel_for_archetypal_injection(
        self, contexto: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            if (
                self.eva_enabled
                and self.eva_manager
                and hasattr(self.eva_manager, "query_archetypes")
            ):
                return list(self.eva_manager.query_archetypes(context=contexto) or [])
        except Exception:
            logger.debug("Archetype query failed", exc_info=True)
        text = (
            str(contexto.get("note", "")).lower()
            + " "
            + " ".join(map(str, contexto.get("tags", []) or []))
        )
        injections: list[dict[str, Any]] = []
        if "creative" in text or "creatividad" in text:
            injections.append({"tipo": "boost_creativity", "payload": {"amount": 0.12}})
        if "threat" in text or contexto.get("cuerpo", {}).get("cortisol", 0) > 0.7:
            injections.append({"tipo": "increase_caution", "payload": {"amount": 0.15}})
        return injections

    def _inject_archetypal_actions(
        self,
        plan_alfa: PlanDeAccion,
        plan_omega: PlanDeAccion,
        injections: list[dict[str, Any]],
    ) -> None:
        try:
            for inj in injections:
                t = inj.get("tipo", "")
                p = inj.get("payload", {}) or {}
                if t == "boost_creativity":
                    paso = PasoAccion(
                        tipo=TipoPaso.EXPLORACION,
                        herramienta="archetype_inspiration",
                        parametros=p,
                        prioridad=7,
                        confianza=0.6,
                        descripcion="Archetypal creativity boost",
                    )
                    plan_alfa.pasos.insert(0, paso)
                    self.estadisticas.pasos_alfa_incluidos += 1
                elif t == "increase_caution":
                    paso = PasoAccion(
                        tipo=TipoPaso.VALIDACION,
                        herramienta="archetype_caution",
                        parametros=p,
                        prioridad=9,
                        confianza=0.9,
                        descripcion="Archetypal caution injection",
                    )
                    plan_omega.pasos.insert(0, paso)
                    self.estadisticas.pasos_omega_incluidos += 1
                else:
                    paso = PasoAccion(
                        tipo=TipoPaso.ANALISIS,
                        herramienta="archetype_hint",
                        parametros=p,
                        prioridad=5,
                        confianza=0.5,
                        descripcion=f"Archetype hint {t}",
                    )
                    plan_omega.pasos.append(paso)
        except Exception:
            logger.debug("Failed to inject archetypal actions", exc_info=True)

    def _agrupar_pasos_por_tipo(
        self, pasos: list[PasoAccion]
    ) -> dict[str, list[PasoAccion]]:
        groups: dict[str, list[PasoAccion]] = defaultdict(list)
        try:
            for p in pasos or []:
                k = p.tipo.name if hasattr(p, "tipo") else str(p.tipo)
                groups[k].append(p)
            for k in list(groups.keys()):
                groups[k].sort(key=lambda x: getattr(x, "prioridad", 0), reverse=True)
        except Exception:
            logger.debug("Grouping pasos failed", exc_info=True)
        return dict(groups)

    def _resolver_conflictos(self, pasos: list[PasoAccion]) -> list[PasoAccion]:
        seen = set()
        resolved: list[PasoAccion] = []
        try:
            candidatos = sorted(
                pasos or [],
                key=lambda x: (
                    getattr(x, "prioridad", 0),
                    getattr(x, "confianza", 0.0),
                ),
                reverse=True,
            )
            i = 0
            while i < len(candidatos):
                p = candidatos[i]
                key = (p.herramienta, str(sorted(p.parametros.items())))
                if key in seen:
                    i += 1
                    continue
                unmet = [
                    d
                    for d in getattr(p, "dependencias", [])
                    if d not in {q.id for q in resolved}
                ]
                if unmet:
                    candidatos.append(candidatos.pop(i))
                    continue
                seen.add(key)
                resolved.append(p)
                i += 1
        except Exception:
            logger.debug("Conflict resolution failed", exc_info=True)
        return resolved

    def _apply_cerebellum_influence(
        self, pattern_result: PatternRecognitionResult
    ) -> None:
        try:
            if getattr(pattern_result, "confidence", None) is not None:
                # minimal metrics update - hook for more sophisticated logic in future
                self.estadisticas.pasos_sintetizados += 0
        except Exception:
            logger.debug("_apply_cerebellum_influence noop failed", exc_info=True)

    # full impl remains present earlier in file and referenced by _sintetizar
    def _sintetizar_full_impl(
        self,
        plan_alfa: PlanDeAccion,
        plan_omega: PlanDeAccion,
        contexto: dict[str, Any],
    ) -> PlanDeAccion:
        try:
            balance = self._evaluar_balance_contexto(contexto)
            pasos_alpha = list(plan_alfa.pasos) if plan_alfa and plan_alfa.pasos else []
            pasos_omega = (
                list(plan_omega.pasos) if plan_omega and plan_omega.pasos else []
            )
            all_pasos = pasos_omega + pasos_alpha
            grupos = self._agrupar_pasos_por_tipo(all_pasos)
            final_pasos: list[PasoAccion] = []
            # prefer analysis/validation
            for tipo_pref in ("ANALISIS", "VALIDACION"):
                for p in grupos.get(tipo_pref, [])[:3]:
                    final_pasos.append(p)
                    self.estadisticas.pasos_omega_incluidos += 1
            # include creative alfa step if context allows
            if balance > 0.55 and pasos_alpha:
                alfa_creativas = [
                    p
                    for p in pasos_alpha
                    if p.tipo
                    in (TipoPaso.CREACION, TipoPaso.MODIFICACION, TipoPaso.EXPLORACION)
                ]
                if alfa_creativas:
                    top_alfa = max(alfa_creativas, key=lambda x: x.prioridad)
                    final_pasos.append(top_alfa)
                    self.estadisticas.pasos_alfa_incluidos += 1
            extras = (
                grupos.get("ACCION", [])
                + grupos.get("CREACION", [])
                + grupos.get("EXPLORACION", [])
            )
            for p in extras:
                if len(final_pasos) >= max(3, int(4 + balance * 6)):
                    break
                final_pasos.append(p)
            final_pasos = self._resolver_conflictos(final_pasos)
            confianza_media = float(
                (
                    (getattr(plan_alfa, "confianza_general", 0.5) if plan_alfa else 0.5)
                    + (
                        getattr(plan_omega, "confianza_general", 0.5)
                        if plan_omega
                        else 0.5
                    )
                )
                / 2.0
            )
            riesgo_medio = float(
                (
                    (getattr(plan_alfa, "riesgo_estimado", 0.5) if plan_alfa else 0.5)
                    + (
                        getattr(plan_omega, "riesgo_estimado", 0.5)
                        if plan_omega
                        else 0.5
                    )
                )
                / 2.0
            )
            plan = PlanDeAccion(
                justificacion=f"Sintetizado (full) balance={balance:.2f}",
                pasos=final_pasos,
                confianza_general=confianza_media,
                riesgo_estimado=riesgo_medio,
            )
            plan.metadatos.update(
                {
                    "balance_contexto": balance,
                    "num_input_alfa": len(pasos_alpha),
                    "num_input_omega": len(pasos_omega),
                    "timestamp": time.time(),
                }
            )
            return plan
        except Exception:
            logger.exception("Full synthesis failed; falling back to lightweight.")
            return self._sintetizar(plan_alfa, plan_omega, contexto)

    def _finalize_plan(
        self,
        pasos: list[PasoAccion],
        plan_alfa: PlanDeAccion,
        plan_omega: PlanDeAccion,
        contexto: dict[str, Any],
    ) -> PlanDeAccion:
        try:
            confianza_media = float(
                (
                    (getattr(plan_alfa, "confianza_general", 0.5) if plan_alfa else 0.5)
                    + (
                        getattr(plan_omega, "confianza_general", 0.5)
                        if plan_omega
                        else 0.5
                    )
                )
                / 2.0
            )
            riesgo_medio = float(
                (
                    (getattr(plan_alfa, "riesgo_estimado", 0.5) if plan_alfa else 0.5)
                    + (
                        getattr(plan_omega, "riesgo_estimado", 0.5)
                        if plan_omega
                        else 0.5
                    )
                )
                / 2.0
            )
            plan = PlanDeAccion(
                justificacion=f"Sintetizado (final) for '{(plan_alfa.justificacion if plan_alfa else '')[:60]}'",
                pasos=pasos,
                confianza_general=confianza_media,
                riesgo_estimado=riesgo_medio,
            )
            plan.metadatos.update(
                {
                    "timestamp": time.time(),
                    "num_steps": len(pasos),
                    "context_balance": self._evaluar_balance_contexto(contexto),
                }
            )
            return plan
        except Exception:
            logger.exception("finalize_plan failed; returning minimal plan")
            return PlanDeAccion(
                justificacion="fallback plan",
                pasos=pasos or [],
                confianza_general=0.5,
                riesgo_estimado=0.5,
            )

    def get_purpose_fulfillment(self) -> float:
        try:
            active_missions = [
                m
                for m in self.mission_manager.active_missions.values()
                if getattr(m, "status", None) == _MissionStatus.ACTIVE
            ]
            if not active_missions:
                return 0.0
            total_priority = sum(getattr(m, "priority", 0) for m in active_missions)
            if total_priority == 0:
                return 0.0
            weighted_fulfillment_sum = 0.0
            for mission in active_missions:
                progress = (getattr(mission, "target_metrics", {}) or {}).get(
                    "progress_percentage", 0.0
                )
                weighted_fulfillment_sum += (progress / 100.0) * getattr(
                    mission, "priority", 0
                )
            return weighted_fulfillment_sum / total_priority
        except Exception:
            logger.debug("get_purpose_fulfillment failed", exc_info=True)
            return 0.0


    def _convertir_snapshot_a_plan(self, trace_snapshot: dict[str, Any]) -> PlanDeAccion:
        """Converts a BrainFallback trace snapshot into a PlanDeAccion."""
        pasos = []
        confianza_general = 0.5

        if trace_snapshot.get("mode") == "RUN":
            action_info = trace_snapshot.get("exec", {})
            if action_info.get("action"):
                paso = PasoAccion(
                    tipo=TipoPaso.ACCION,
                    herramienta=action_info.get("action", "unknown_tool"),
                    parametros=action_info.get("action_result", {}),
                    confianza=action_info.get("confidence", 0.5),
                    descripcion=f"Propuesta directa de BrainFallback: {action_info.get('action', 'N/A')}"
                )
                pasos.append(paso)
                confianza_general = paso.confianza

        return PlanDeAccion(
            justificacion="Plan de experiencia directa (BrainFallback)",
            pasos=pasos,
            confianza_general=confianza_general,
            riesgo_estimado=1.0 - confianza_general
        )