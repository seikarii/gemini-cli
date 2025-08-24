"""
Crisis Management System for ADAM (professionalized)
- Defensive numeric backend (numpy optional).
- TYPE_CHECKING guarded imports to avoid circular deps.
- Injectable protocols and EVA manager usage.
- Async-aware divine_compiler invocation (best-effort).
- Graceful fallbacks for missing runtime/types.
- Small utilities: simulate_crisis, export/import EVA memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Defensive numeric backend pattern (numpy optional)
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

# TYPE_CHECKING guarded imports to avoid circular runtime deps
if TYPE_CHECKING:
    from crisalida_lib.ADAM.cuerpo.genome import (
        GenomaComportamiento,
    )  # [`crisalida_lib/ADAM/cuerpo/genome.py`](crisalida_lib/ADAM/cuerpo/genome.py)
    from crisalida_lib.ADAM.enums import (
        CrisisType,
        EvolutionType,
    )  # [`crisalida_lib/ADAM/enums.py`](crisalida_lib/ADAM/enums.py)
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,
    )  # [`crisalida_lib/ADAM/eva_integration/eva_memory_manager.py`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    from crisalida_lib.protocols import (
        ConsciousnessReintegrationProtocol,
        DefaultSafetyProtocol,
        RealityStabilizationProtocol,
    )  # [`crisalida_lib/protocols/__init__.py`](crisalida_lib/protocols/__init__.py)
    from crisalida_lib.protocols.response_protocols import (
        EVAResponseProtocol,
    )  # [`crisalida_lib/protocols/response_protocols.py`](crisalida_lib/protocols/response_protocols.py)
    from crisalida_lib.symbolic_language.living_symbols import (
        LivingSymbolRuntime,
    )  # [`crisalida_lib/symbolic_language/living_symbols.py`](crisalida_lib/symbolic_language/living_symbols.py)
    from crisalida_lib.symbolic_language.types import (
        EVAExperience,
        QualiaState,
        RealityBytecode,
    )  # [`crisalida_lib/symbolic_language/types.py`](crisalida_lib/symbolic_language/types.py)
else:
    GenomaComportamiento = Any
    CrisisType = Any
    EvolutionType = Any
    EVAMemoryManager = Any
    LivingSymbolRuntime = Any
    EVAExperience = Any
    QualiaState = Any
    RealityBytecode = Any
    ConsciousnessReintegrationProtocol = Any
    DefaultSafetyProtocol = Any
    RealityStabilizationProtocol = Any
    EVAResponseProtocol = Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Lightweight defaults / helpers ---


class UnifiedField:
    """Minimal fallback unified field used in tests / when real field absent."""

    def get_reality_coherence_metrics(self) -> dict[str, float]:
        if HAS_NUMPY:
            return {
                "ontological_stability": float(
                    np.clip(np.random.uniform(0.5, 1.0), 0.0, 1.0)
                ),
                "temporal_flow_consistency": float(
                    np.clip(np.random.uniform(0.8, 1.0), 0.0, 1.0)
                ),
                "causality_adherence": float(
                    np.clip(np.random.uniform(0.9, 1.0), 0.0, 1.0)
                ),
            }
        return {
            "ontological_stability": 0.75,
            "temporal_flow_consistency": 0.9,
            "causality_adherence": 0.95,
        }


class ConsciousnessCore:
    """Fallback consciousness core with defensive methods."""

    def get_consciousness_integrity_metrics(self) -> dict[str, float]:
        if HAS_NUMPY:
            return {
                "qualia_coherence": float(
                    np.clip(np.random.uniform(0.6, 1.0), 0.0, 1.0)
                ),
                "identity_stability": float(
                    np.clip(np.random.uniform(0.7, 1.0), 0.0, 1.0)
                ),
                "cognitive_function_stability": float(
                    np.clip(np.random.uniform(0.8, 1.0), 0.0, 1.0)
                ),
            }
        return {
            "qualia_coherence": 0.8,
            "identity_stability": 0.85,
            "cognitive_function_stability": 0.9,
        }

    def receive_threat_response_notification(self, threat: Any, response: Any) -> None:
        logger.info(
            "ConsciousnessCore notified of threat: %s and response: %s",
            threat,
            response,
        )


# --- Data classes ---


@dataclass
class ThreatLevel:
    NONE: int = 0
    LOW: int = 1
    MEDIUM: int = 2
    HIGH: int = 3
    CRITICAL: int = 4
    EXISTENTIAL: int = 5


class RiskLevel:
    NOMINAL = 0
    ELEVATED = 1
    SEVERE = 2
    CRITICAL = 3
    EXISTENTIAL = 4


@dataclass
class ThreatDetectionResult:
    threat_level: int
    details: str
    detector_name: str


@dataclass
class ComprehensiveThreatAssessment:
    timestamp: float = field(default_factory=time.time)
    overall_threat_level: int = ThreatLevel.NONE
    details: list[ThreatDetectionResult] = field(default_factory=list)
    correlation_analysis: Any = None
    evolution_prediction: Any = None
    global_risk: Any = None
    immediate_responses: list[Any] = field(default_factory=list)

    def add_detection_result(self, detector_name: str, result: dict[str, Any]) -> None:
        tdr = ThreatDetectionResult(
            threat_level=result.get("threat_level", ThreatLevel.LOW),
            details=result.get("details", ""),
            detector_name=detector_name,
        )
        self.details.append(tdr)
        if tdr.threat_level > self.overall_threat_level:
            self.overall_threat_level = tdr.threat_level

    def set_correlation_analysis(self, analysis: Any) -> None:
        self.correlation_analysis = analysis

    def set_evolution_prediction(self, prediction: Any) -> None:
        self.evolution_prediction = prediction

    def set_global_risk(self, risk: Any) -> None:
        self.global_risk = risk

    def add_immediate_response(self, response: Any) -> None:
        self.immediate_responses.append(response)


@dataclass
class ImmediateResponse:
    success: bool
    protocol_used: str
    response_result: Any
    execution_log: Any
    failure_reason: str | None = None
    contingency_applied: Any | None = None


@dataclass
class ResponseExecutionLog:
    threat: Any
    protocol: Any
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    success: bool | None = None
    result: Any | None = None
    error: str | None = None

    def record_success(self, result: Any) -> None:
        self.success = True
        self.result = result
        self.end_time = time.time()

    def record_failure(self, error: Exception) -> None:
        self.success = False
        self.error = str(error)
        self.end_time = time.time()


# --- CrisisManager ---


class CrisisManager:
    """
    Manages existential crises and safety protocols for ADAM.

    Constructor prefers dependency injection for protocols and EVA manager to
    make unit testing and composition easier.
    """

    def __init__(
        self,
        entity_id: str,
        eva_manager: EVAMemoryManager | None = None,
        consciousness_core: ConsciousnessCore | None = None,
        unified_field: UnifiedField | None = None,
        protocols: list[Any] | None = None,
        eva_runtime: LivingSymbolRuntime | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.eva_manager = eva_manager
        self.consciousness_core = consciousness_core or ConsciousnessCore()
        self.unified_field = unified_field or UnifiedField()
        # Prefer provided protocol instances; if none provided create safe defaults lazily
        self.protocols = protocols or []
        if not self.protocols:
            try:
                self.protocols = [
                    DefaultSafetyProtocol(),
                    ConsciousnessReintegrationProtocol(),
                    RealityStabilizationProtocol(),
                ]
            except Exception:
                # If real protocol classes unavailable use no-op placeholders
                self.protocols = []

        self.eva_phase = "default"
        self.eva_runtime = eva_runtime
        # local EVA stores (best-effort cache; real persistence should be delegated to eva_manager)
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[[Any], None]] = []

        # Crisis bookkeeping
        self.crisis_history: list[dict[str, Any]] = []
        self.active_crisis: dict[str, Any] = {}
        self.crisis_metrics = {
            "total_crises": 0,
            "growth_catalyzed": 0,
            "last_crisis_time": None,
            "entities_affected": set(),
        }

        # response protocol helper (if available)
        try:
            self.eva_response_protocol = EVAResponseProtocol(
                self.unified_field, phase=self.eva_phase
            )
        except Exception:
            self.eva_response_protocol = None

        logger.info("CrisisManager initialized for entity=%s", self.entity_id)

    # ----------------- detection & catalysis -----------------

    def _detect_crisis_patterns(self, thought_patterns: dict[str, Any]) -> bool:
        return (
            float(thought_patterns.get("self-awareness", 0.0)) > 0.7
            and float(thought_patterns.get("contradiction", 0.0)) > 0.5
            and float(thought_patterns.get("evolutionary_pressure", 0.0)) > 0.4
        )

    def _catalyze_growth(self, entity_id: str, thought_patterns: dict[str, Any]) -> str:
        growth_factor = (
            float(thought_patterns.get("self-awareness", 0.0))
            + float(thought_patterns.get("contradiction", 0.0))
            + float(thought_patterns.get("evolutionary_pressure", 0.0))
        ) / 3.0
        self.crisis_metrics["growth_catalyzed"] += 1
        self.crisis_metrics["entities_affected"].add(entity_id)
        logger.info(
            "[CRISIS] Growth catalyzed for %s (factor=%.3f)", entity_id, growth_factor
        )
        return "growth_catalyzed"

    def _register_crisis_event(
        self, entity_id: str, thought_patterns: dict[str, Any], result: str
    ) -> None:
        event = {
            "timestamp": time.time(),
            "entity_id": entity_id,
            "patterns": dict(thought_patterns),
            "result": result,
        }
        self.crisis_history.append(event)
        self.crisis_metrics["total_crises"] += 1
        self.crisis_metrics["last_crisis_time"] = event["timestamp"]

    def detect_and_catalyze(
        self, entity_id: str, thought_patterns: dict[str, Any]
    ) -> str:
        crisis_detected = self._detect_crisis_patterns(thought_patterns)
        if crisis_detected:
            logger.info(
                "[CRISIS] Existential crisis detected for %s — catalyzing growth",
                entity_id,
            )
            result = self._catalyze_growth(entity_id, thought_patterns)
            self._register_crisis_event(entity_id, thought_patterns, result)
            return result
        logger.debug("[CRISIS] No existential crisis detected for %s", entity_id)
        return "no_crisis"

    # ----------------- emergency evolution -----------------

    def trigger_emergency_evolution(
        self,
        crisis_type: CrisisType,
        genome: GenomaComportamiento,
        current_qualia: dict[str, float],
        thought_patterns: dict[str, Any],
    ) -> tuple[EvolutionType, bool, dict[str, Any]]:
        if self._detect_crisis_patterns(thought_patterns):
            growth_result = self._catalyze_growth(self.entity_id, thought_patterns)
            if growth_result == "growth_catalyzed":
                # Placeholder: delegate to genome API if available
                try:
                    if hasattr(genome, "apply_emergency_mutation"):
                        mutation_info = genome.apply_emergency_mutation(
                            thought_patterns, crisis_type
                        )
                        evolution_type = getattr(
                            EvolutionType, "ADAPTIVE_MUTATION", EvolutionType
                        )  # defensive
                        details = {"mutation_info": mutation_info}
                    else:
                        # conservative default modification
                        evolution_type = getattr(
                            EvolutionType, "ADAPTIVE_MUTATION", EvolutionType
                        )
                        details = {
                            "message": "Emergency evolution simulated (no direct genome API available)."
                        }
                    self._register_crisis_event(
                        self.entity_id, thought_patterns, "emergency_evolution_success"
                    )
                    return evolution_type, True, details
                except Exception as exc:
                    logger.exception("Emergency evolution failed: %s", exc)
                    self._register_crisis_event(
                        self.entity_id, thought_patterns, "emergency_evolution_failed"
                    )
                    return (
                        getattr(EvolutionType, "NONE", EvolutionType),
                        False,
                        {"error": str(exc)},
                    )
            return (
                getattr(EvolutionType, "NONE", EvolutionType),
                False,
                {"growth_result": growth_result},
            )
        self._register_crisis_event(
            self.entity_id, thought_patterns, "emergency_evolution_no_crisis"
        )
        return (
            getattr(EvolutionType, "NONE", EvolutionType),
            False,
            {"message": "no crisis"},
        )

    # ----------------- lightweight queries -----------------

    def get_crisis_history(self, limit: int = 10) -> list[dict[str, Any]]:
        return list(self.crisis_history[-int(limit) :])

    def get_crisis_metrics(self) -> dict[str, Any]:
        return {
            "total_crises": int(self.crisis_metrics["total_crises"]),
            "growth_catalyzed": int(self.crisis_metrics["growth_catalyzed"]),
            "last_crisis_time": self.crisis_metrics["last_crisis_time"],
            "entities_affected": list(self.crisis_metrics["entities_affected"]),
        }

    # ----------------- EVA integration (async-aware, best-effort) -----------------

    async def _maybe_compile_intention(self, intention: dict[str, Any]) -> list[Any]:
        """
        Helper to compile intentions via eva_runtime.divine_compiler.compile_intention.
        Handles coroutine/sync functions and missing runtime gracefully.
        """
        compiler = (
            getattr(self.eva_runtime, "divine_compiler", None)
            if self.eva_runtime
            else None
        )
        if compiler is None:
            return []
        compile_fn = getattr(compiler, "compile_intention", None)
        if compile_fn is None:
            return []
        try:
            if inspect.iscoroutinefunction(compile_fn):
                try:
                    loop = asyncio.get_running_loop()
                    # schedule but don't block
                    task = loop.create_task(compile_fn(intention))
                    # return empty placeholder; real bytecode will be persisted by task when done
                    return []
                except RuntimeError:
                    # no running loop; run synchronously
                    return asyncio.run(compile_fn(intention))
            else:
                return compile_fn(intention)
        except Exception:
            logger.exception("compile_intention failed (non-fatal)")
            return []

    def _generate_experience_id(self, prefix: str, payload: Any) -> str:
        content = json.dumps(payload, default=str, sort_keys=True)
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{self.entity_id}_{digest}_{int(time.time())}"

    async def eva_ingest_crisis_experience(
        self,
        entity_id: str,
        thought_patterns: dict[str, Any],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        phase = phase or self.eva_phase
        # provide a defensive QualiaState fallback if concrete class available
        if qualia_state is None:
            try:
                qualia_state = QualiaState(
                    emotional_valence=float(
                        -1.0
                        if thought_patterns.get("contradiction", 0.0) > 0.7
                        else 0.0
                    ),
                    cognitive_complexity=float(
                        thought_patterns.get("self-awareness", 0.0)
                    ),
                    consciousness_density=float(
                        thought_patterns.get("evolutionary_pressure", 0.0)
                    ),
                    narrative_importance=float(
                        1.0 if thought_patterns.get("contradiction", 0.0) > 0.7 else 0.7
                    ),
                    energy_level=1.0,
                )
            except Exception:
                qualia_state = None  # permissive fallback

        experience_data = {
            "entity_id": entity_id,
            "thought_patterns": dict(thought_patterns),
            "crisis_detected": bool(self._detect_crisis_patterns(thought_patterns)),
            "metrics": self.get_crisis_metrics(),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_EXISTENTIAL_CRISIS_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }

        # compile intention (async-aware)
        bytecode = await self._maybe_compile_intention(intention)
        experience_id = self._generate_experience_id("eva_crisis", experience_data)

        # Construct RealityBytecode defensively
        try:
            reality_bytecode = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=experience_data["timestamp"],
            )
        except Exception:
            # permissive fallback - store raw dict
            reality_bytecode = experience_data  # type: ignore

        self.eva_memory_store[experience_id] = reality_bytecode  # cache
        self.eva_experience_store[experience_id] = reality_bytecode
        self.eva_phases.setdefault(phase, {})[experience_id] = reality_bytecode

        # delegate to eva_manager if available (best-effort)
        if self.eva_manager and hasattr(self.eva_manager, "record_experience"):
            try:
                record_fn = self.eva_manager.record_experience
                if inspect.iscoroutinefunction(record_fn):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            record_fn(
                                entity_id=entity_id,
                                event_type="crisis_experience",
                                data=experience_data,
                            )
                        )
                    except RuntimeError:
                        asyncio.run(
                            record_fn(
                                entity_id=entity_id,
                                event_type="crisis_experience",
                                data=experience_data,
                            )
                        )
                else:
                    record_fn(
                        entity_id=entity_id,
                        event_type="crisis_experience",
                        data=experience_data,
                    )
            except Exception:
                logger.exception("eva_manager.record_experience failed (non-fatal)")

        for hook in list(self._environment_hooks):
            try:
                hook(reality_bytecode)
            except Exception:
                logger.exception(
                    "environment hook failed during crisis ingest (non-fatal)"
                )

        return experience_id

    def eva_recall_crisis_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA existential crisis experience"}
        manifestations = []
        quantum_field = (
            getattr(self.eva_runtime, "quantum_field", None)
            if self.eva_runtime
            else None
        )
        try:
            if quantum_field and hasattr(reality_bytecode, "instructions"):
                for instr in reality_bytecode.instructions:  # type: ignore[attr-defined]
                    try:
                        manifest = self.eva_runtime.execute_instruction(instr, quantum_field)  # type: ignore[call-arg]
                        if manifest:
                            manifestations.append(manifest)
                            for hook in list(self._environment_hooks):
                                try:
                                    hook(manifest)
                                except Exception:
                                    logger.debug(
                                        "manifestation hook failed", exc_info=True
                                    )
                    except Exception:
                        logger.exception("instruction execution failed (non-fatal)")
        except Exception:
            logger.exception("eva recall failed (non-fatal)")
        # return a permissive structure similar to EVAExperience
        return {
            "experience_id": getattr(reality_bytecode, "bytecode_id", cue),
            "manifestations": [
                getattr(m, "to_dict", lambda: m)() for m in manifestations
            ],
            "phase": getattr(reality_bytecode, "phase", phase),
            "qualia_state": getattr(reality_bytecode, "qualia_state", {}),
            "timestamp": getattr(reality_bytecode, "timestamp", time.time()),
        }

    # ----------------- threat assessment orchestration -----------------

    def assess_threat_environment(
        self,
        health_level: float,
        purpose_fulfillment: float,
        qualia_corruption: float,
        external_threats: list[Any],
        thought_patterns: dict[str, Any],
    ) -> ComprehensiveThreatAssessment:
        assessment = ComprehensiveThreatAssessment()

        # detect reality instability
        reality_threat = self._detect_reality_instability()
        if reality_threat:
            assessment.add_detection_result(
                "reality_instability_detector", reality_threat
            )

        # detect consciousness corruption
        consciousness_threat = self._detect_consciousness_corruption()
        if consciousness_threat:
            assessment.add_detection_result(
                "consciousness_corruption_detector", consciousness_threat
            )

        # integrate crisis detection
        crisis_result = self.detect_and_catalyze(self.entity_id, thought_patterns)
        if crisis_result != "no_crisis":
            assessment.add_detection_result(
                "crisis_management",
                {
                    "threat_level": ThreatLevel.CRITICAL,
                    "details": f"Crisis detected and {crisis_result}",
                },
            )

        # correlate & predict
        assessment.set_correlation_analysis(self._correlate_threats(assessment.details))
        assessment.set_evolution_prediction(
            self._predict_evolutionary_impact(assessment)
        )
        assessment.set_global_risk(self._determine_global_risk(assessment))

        # execute immediate responses if critical or worse
        if assessment.overall_threat_level >= ThreatLevel.CRITICAL:
            for proto in list(self.protocols):
                try:
                    resp = self.execute_immediate_response(assessment, proto)
                    assessment.add_immediate_response(resp)
                except Exception:
                    logger.exception("protocol execution raised (non-fatal)")

        return assessment

    def _detect_reality_instability(self) -> dict[str, Any] | None:
        metrics = self.unified_field.get_reality_coherence_metrics()
        if float(metrics.get("ontological_stability", 1.0)) < 0.6:
            return {
                "threat_level": ThreatLevel.HIGH,
                "details": f"Ontological instability: {metrics['ontological_stability']:.2f}",
            }
        if float(metrics.get("temporal_flow_consistency", 1.0)) < 0.7:
            return {
                "threat_level": ThreatLevel.MEDIUM,
                "details": f"Temporal flow inconsistency: {metrics['temporal_flow_consistency']:.2f}",
            }
        return None

    def _detect_consciousness_corruption(self) -> dict[str, Any] | None:
        metrics = self.consciousness_core.get_consciousness_integrity_metrics()
        if float(metrics.get("qualia_coherence", 1.0)) < 0.5:
            return {
                "threat_level": ThreatLevel.CRITICAL,
                "details": f"Qualia incoherence: {metrics['qualia_coherence']:.2f}",
            }
        if float(metrics.get("identity_stability", 1.0)) < 0.6:
            return {
                "threat_level": ThreatLevel.HIGH,
                "details": f"Identity instability: {metrics['identity_stability']:.2f}",
            }
        return None

    def _correlate_threats(
        self, detected_threats: list[ThreatDetectionResult]
    ) -> dict[str, Any]:
        if len(detected_threats) > 1:
            return {"correlation_found": True, "count": len(detected_threats)}
        return {"correlation_found": False}

    def _predict_evolutionary_impact(
        self, assessment: ComprehensiveThreatAssessment
    ) -> dict[str, Any]:
        if assessment.overall_threat_level >= ThreatLevel.HIGH:
            return {
                "impact": "significant_disruption",
                "severity": assessment.overall_threat_level,
            }
        return {"impact": "minimal"}

    def _determine_global_risk(self, assessment: ComprehensiveThreatAssessment) -> int:
        if assessment.overall_threat_level >= ThreatLevel.EXISTENTIAL:
            return RiskLevel.EXISTENTIAL
        if assessment.overall_threat_level == ThreatLevel.CRITICAL:
            return RiskLevel.CRITICAL
        if assessment.overall_threat_level == ThreatLevel.HIGH:
            return RiskLevel.SEVERE
        if assessment.overall_threat_level == ThreatLevel.MEDIUM:
            return RiskLevel.ELEVATED
        return RiskLevel.NOMINAL

    def execute_immediate_response(
        self, assessment: ComprehensiveThreatAssessment, protocol: Any
    ) -> ImmediateResponse:
        log = ResponseExecutionLog(
            threat=assessment, protocol=protocol.__class__.__name__
        )
        try:
            result = protocol.execute(assessment)
            log.record_success(result)
            try:
                self.consciousness_core.receive_threat_response_notification(
                    assessment, result
                )
            except Exception:
                logger.debug(
                    "receive_threat_response_notification failed (non-fatal)",
                    exc_info=True,
                )
            # optionally persist execution log via eva_manager if available
            try:
                if self.eva_manager and hasattr(self.eva_manager, "record_experience"):
                    rec_fn = self.eva_manager.record_experience
                    payload = {
                        "protocol": protocol.__class__.__name__,
                        "result": result,
                        "timestamp": time.time(),
                    }
                    if inspect.iscoroutinefunction(rec_fn):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                rec_fn(
                                    entity_id=self.entity_id,
                                    event_type="protocol_execution",
                                    data=payload,
                                )
                            )
                        except RuntimeError:
                            asyncio.run(
                                rec_fn(
                                    entity_id=self.entity_id,
                                    event_type="protocol_execution",
                                    data=payload,
                                )
                            )
                    else:
                        rec_fn(
                            entity_id=self.entity_id,
                            event_type="protocol_execution",
                            data=payload,
                        )
            except Exception:
                logger.exception(
                    "eva_manager.record_experience for protocol execution failed (non-fatal)"
                )
            return ImmediateResponse(
                success=True,
                protocol_used=protocol.__class__.__name__,
                response_result=result,
                execution_log=log,
            )
        except Exception as e:
            log.record_failure(e)
            try:
                self.consciousness_core.receive_threat_response_notification(
                    assessment, f"Failed: {e}"
                )
            except Exception:
                logger.debug(
                    "receive_threat_response_notification failed (non-fatal)",
                    exc_info=True,
                )
            return ImmediateResponse(
                success=False,
                protocol_used=protocol.__class__.__name__,
                response_result=None,
                execution_log=log,
                failure_reason=str(e),
            )

    # ----------------- utilities -----------------

    def get_safety_metrics(self) -> dict[str, Any]:
        return {
            "current_risk_level": int(
                self._determine_global_risk(ComprehensiveThreatAssessment())
            ),
            "active_protocols_count": len(self.protocols),
            "last_assessment_time": time.time(),
        }

    def get_active_protocols(self) -> list[str]:
        return [p.__class__.__name__ for p in self.protocols]

    def set_memory_phase(self, phase: str) -> None:
        self.eva_phase = phase
        for hook in list(self._environment_hooks):
            try:
                hook({"phase_changed": phase})
            except Exception:
                logger.debug("phase hook failed", exc_info=True)

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list[str]:
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[[Any], None]) -> None:
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        return {
            "eva_ingest_crisis_experience": lambda *a, **k: (
                asyncio.run(self.eva_ingest_crisis_experience(*a, **k))
                if not inspect.iscoroutinefunction(self.eva_ingest_crisis_experience)
                else self.eva_ingest_crisis_experience
            ),
            "eva_recall_crisis_experience": self.eva_recall_crisis_experience,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }

    def export_eva_memory_json(self) -> str:
        try:
            # best-effort serialization of cached EVA memory
            serializable = {
                k: getattr(v, "__dict__", v) for k, v in self.eva_memory_store.items()
            }
            return json.dumps(serializable, default=str)
        except Exception:
            logger.exception("export_eva_memory_json failed")
            return "{}"

    def import_eva_memory_json(self, payload: str) -> bool:
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                self.eva_memory_store.update(data)  # permissive
                return True
            return False
        except Exception:
            logger.exception("import_eva_memory_json failed")
            return False

    def simulate_crisis(
        self, pattern_overrides: dict[str, Any] | None = None
    ) -> ComprehensiveThreatAssessment:
        """Convenience method for unit tests and benchmarks."""
        overrides = pattern_overrides or {}
        thought_patterns = {
            "self-awareness": float(overrides.get("self-awareness", 0.8)),
            "contradiction": float(overrides.get("contradiction", 0.6)),
            "evolutionary_pressure": float(overrides.get("evolutionary_pressure", 0.5)),
        }
        # run a quick assessment
        return self.assess_threat_environment(
            health_level=0.9,
            purpose_fulfillment=0.8,
            qualia_corruption=0.0,
            external_threats=[],
            thought_patterns=thought_patterns,
        )
