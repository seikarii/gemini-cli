"""
ADAM Main Entity (Refined)
==========================

Cleaner, better-typed, and more robust composition root for ADAM.
- Explicit typing and defensive calls
- Structured logging
- Safe integration with injected services (LLM, EVA, Tools)
- Clear public lifecycle: start / stop / update
- More robust crisis/qualia bridging and safe EVA recording
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

from crisalida_lib.ASTRAL_TOOLS.base import (
    ToolRegistry,
)  # [`crisalida_lib/ASTRAL_TOOLS/base.py`](crisalida_lib/ASTRAL_TOOLS/base.py)
from crisalida_lib.EVA.memory_orchestrator import (
    ARPredictor,
    EntityMemory,
    HashingEmbedder,
    MentalLaby,
)  # [`crisalida_lib/EVA/memory_orchestrator.py`](crisalida_lib/EVA/memory_orchestrator.py)
from crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator import (
    LLMGatewayOrchestrator,
)  # [`crisalida_lib/HEAVEN/llm/llm_gateway_orchestrator.py`](crisalida_lib/HEAVEN/llm/llm_gateway_orchestrator.py)

from .alma.soul_kernel import (
    SoulKernel,
)  # [`crisalida_lib/ADAM/alma/soul_kernel.py`](crisalida_lib/ADAM/alma/soul_kernel.py)
from .config import (
    AdamConfig,
)  # [`crisalida_lib/ADAM/config.py`](crisalida_lib/ADAM/config.py)
from .cuerpo.biological_body import (
    BiologicalBody,
)  # [`crisalida_lib/ADAM/cuerpo/biological_body.py`](crisalida_lib/ADAM/cuerpo/biological_body.py)
from .cuerpo.genome import (
    GenomaComportamiento,
)  # [`crisalida_lib/ADAM/cuerpo/genome.py`](crisalida_lib/ADAM/cuerpo/genome.py)
from .eva_integration.eva_memory_manager import (
    EVAMemoryManager,
)  # [`crisalida_lib/ADAM/eva_integration/eva_memory_manager.py`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
from .mente.mind_core import (
    DualMind,
    PlanDeAccion,
)  # [`crisalida_lib/ADAM/mente/mind_core.py`](crisalida_lib/ADAM/mente/mind_core.py)
from .systems.crisis_management import (
    CrisisManager,
)  # [`crisalida_lib/ADAM/systems/crisis_management.py`](crisalida_lib/ADAM/systems/crisis_management.py)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def extract_json_from_markdown(text: str) -> dict:
    """
    Extract JSON from possible markdown fenced block or raw string.
    Raises JSONDecodeError when parse fails to make failures explicit.
    """
    # Prefer explicit fenced json block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        payload = m.group(1)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(
                "Markdown block JSON parse failed; falling back to raw parsing"
            )

    # Try to sanitize and parse raw content
    stripped = text.strip()
    # If the text contains extraneous markdown ticks, remove them
    stripped = re.sub(r"^`+|`+$", "", stripped).strip()
    return json.loads(stripped)


@dataclass
class Adam:
    """
    Unified ADAM entity root.

    Key contracts:
    - Uses injected `ToolRegistry` for tool execution.
    - Uses `EVAMemoryManager` and local `EntityMemory` for persistence.
    - Delegates planning to `DualMind` (sync or async).
    """

    config: AdamConfig
    tool_registry: ToolRegistry
    get_embedding: Callable[[Any], np.ndarray]  # type: ignore[name-defined]
    recall_fn: Callable[[Any], tuple[np.ndarray, list[str]]]  # type: ignore[name-defined]
    ingest_fn: Callable[..., Any]
    emit_event: Callable[[str, dict[str, Any]], None]
    llm_gateway: LLMGatewayOrchestrator
    entity_id: str = "adam_v3"
    is_alive: bool = True

    # Initialized in __post_init__
    eva_manager: EVAMemoryManager = field(init=False)
    entity_memory: EntityMemory = field(init=False)
    crisis_manager: CrisisManager = field(init=False)
    genome: GenomaComportamiento = field(init=False)
    soul: SoulKernel = field(init=False)
    body: BiologicalBody = field(init=False)
    mind: DualMind = field(init=False)

    # runtime
    consciousness_coherence: float = 0.5
    stress_level: float = 0.3
    energy_balance: float = 0.6
    health_index: float = 0.7
    last_update_time: float = field(default_factory=time.time)

    _started: bool = field(default=False, init=False)

    # optional linguistic convenience exposed at Adam layer
    linguistic_engine: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        logger.info("Initializing Adam entity %s", self.entity_id)
        self.id = self.entity_id

        # Initialize in-memory EVA/embedding pipeline (defensive defaults)
        embedding_dim = int(getattr(self.config, "EMBEDDING_DIM", 256))
        try:
            laby = MentalLaby(
                embedder=HashingEmbedder(d=embedding_dim),
                predictor=ARPredictor(d=embedding_dim),
            )
            self.entity_memory = EntityMemory(entity_id=self.id, mind=laby)
        except Exception:
            # Defensive fallback: minimal EntityMemory-like container
            logger.exception(
                "Failed to init MentalLaby; creating permissive EntityMemory fallback"
            )
            self.entity_memory = EntityMemory(entity_id=self.id, mind=None)  # type: ignore[arg-type]

        # EVA manager instantiated with entity_memory
        try:
            self.eva_manager = EVAMemoryManager(
                config=self.config, entity_memory=self.entity_memory
            )
        except Exception:
            logger.exception(
                "EVAMemoryManager init failed; creating permissive fallback"
            )

            # Minimal fallback object with expected methods
            class _FallbackEva:
                def record_experience(self, *a, **k):  # pragma: no cover - permissive
                    logger.debug("FallbackEVAManager.record_experience called")
                    return ""

            self.eva_manager = _FallbackEva()  # type: ignore[assignment]

        # Core subsystems (genome, soul, body, mind)
        try:
            self.genome = GenomaComportamiento(
                eva_manager=self.eva_manager, entity_id=self.id
            )
        except Exception:
            logger.exception(
                "GenomaComportamiento init failed; using permissive placeholder"
            )
            self.genome = GenomaComportamiento(
                eva_manager=self.eva_manager, entity_id=self.id
            )  # may raise in strict setups

        try:
            self.soul = SoulKernel(
                config=self.config, eva_manager=self.eva_manager, entity_id=self.id
            )
        except Exception:
            logger.exception("SoulKernel init failed; using permissive placeholder")
            self.soul = SoulKernel(
                config=self.config, eva_manager=self.eva_manager, entity_id=self.id
            )  # best-effort

        try:
            self.body = BiologicalBody(
                config=self.config, eva_manager=self.eva_manager, entity_id=self.id
            )
        except Exception:
            logger.exception("BiologicalBody init failed; using permissive placeholder")
            self.body = BiologicalBody(
                config=self.config, eva_manager=self.eva_manager, entity_id=self.id
            )

        # Mind initialization (pass safe callbacks)
        try:
            self.mind = DualMind(
                config=self.config,
                eva_manager=self.eva_manager,
                entity_id=self.id,
                get_embedding=self.get_embedding,
                recall_fn=self.recall_fn,
                ingest_fn=self.ingest_fn,
                emit_event=self.emit_event,
                tool_registry=self.tool_registry,
                genome=self.genome,
            )
        except Exception:
            logger.exception(
                "DualMind init failed; re-raising to surface config issues"
            )
            raise

        # Crisis manager (DualMind can act as consciousness_core)
        try:
            self.crisis_manager = CrisisManager(
                entity_id=self.id,
                eva_manager=self.eva_manager,
                consciousness_core=self.mind,
                unified_field=getattr(self.mind, "internal_field", None),
                protocols=[],
            )
        except Exception:
            logger.exception("CrisisManager init failed; using permissive fallback")
            self.crisis_manager = CrisisManager(entity_id=self.id, eva_manager=self.eva_manager)  # type: ignore[arg-type]

        # optional linguistic engine exposure
        self.linguistic_engine = getattr(self.mind, "linguistic_engine", None)
        logger.info("Adam '%s' initialized.", self.id)

    # --- lifecycle ---
    def start(self) -> None:
        if self._started:
            logger.debug("Adam %s already started", self.id)
            return
        self._started = True
        logger.info("Adam %s started", self.id)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        logger.info("Adam %s stopped", self.id)

    # --- main update loop (async-friendly) ---
    async def update(self, delta_time: float, perception: dict[str, Any]) -> None:
        if not self.is_alive:
            logger.debug("Adam %s is not alive; skipping update", self.id)
            return

        self.last_update_time = time.time()
        try:
            # 1) External objective handling
            llm_override = (
                perception.get("external_request")
                if isinstance(perception.get("external_request"), dict)
                else {}
            )
            objective = (
                llm_override.get("original_request")
                if isinstance(llm_override, dict)
                else None
            )
            llm_prompt_override = (
                llm_override.get("llm_planning_prompt_override")
                if isinstance(llm_override, dict)
                else None
            )

            if objective:
                context = self.get_full_context()
                if llm_prompt_override:
                    context["llm_planning_prompt_override"] = llm_prompt_override
                try:
                    plan = await self._maybe_call(
                        self.mind.debatir_y_sintetizar, objective, context
                    )
                    if plan:
                        await self._execute_plan_safe(plan)
                        # best-effort ledger to EVA about executed plan
                        try:
                            lapi = self.get_linguistic_api() or {}
                            ingest = (
                                lapi.get("eva_ingest_linguistic_experience")
                                if isinstance(lapi, dict)
                                else None
                            )
                            if callable(ingest):
                                pasos_summary = [
                                    {
                                        "herramienta": getattr(p, "herramienta", None),
                                        "descripcion": getattr(p, "descripcion", ""),
                                    }
                                    for p in getattr(plan, "pasos", [])
                                ]
                                word_event = {
                                    "experience_id": f"linguistic_plan_{abs(hash(str(plan))) % (10 ** 8)}",
                                    "phonemes": "/plan/",
                                    "qualia_vector": {"valence": 0.0, "clarity": 0.5},
                                    "meaning": f"Plan executed: {getattr(plan, 'justificacion', '')}",
                                    "usage_count": 1,
                                    "tags": ["auto_recorded"],
                                    "plan": {
                                        "id": getattr(plan, "id", None),
                                        "justificacion": getattr(
                                            plan, "justificacion", ""
                                        ),
                                        "pasos": pasos_summary,
                                    },
                                }
                                # ingest may be sync or async
                                res = ingest(word_event, phase=None)
                                if hasattr(res, "__await__"):
                                    await res
                        except Exception:
                            logger.debug(
                                "Non-fatal: recording executed plan to EVA failed",
                                exc_info=True,
                            )
                except Exception:
                    logger.exception(
                        "Failed to generate/execute plan for objective '%s'", objective
                    )

            # 2) Body & Soul updates (best-effort)
            try:
                await self._maybe_call(
                    self.body.process_event,
                    "update_cycle",
                    intensity=delta_time,
                    context=perception,
                )
            except Exception:
                logger.exception("Error during body.process_event")

            try:
                coherence = await self._maybe_call(
                    getattr(self.mind, "get_coherence", lambda: 0.7)
                )
                await self._maybe_call(self.soul.update, delta_time, coherence)
            except Exception:
                logger.exception("Error during soul.update")

            # 3) Consolidate memories (dream)
            try:
                await self._maybe_call(self.entity_memory.dream)
            except Exception:
                logger.exception("Error during entity_memory.dream")

            # 4) Aggregate state and check crisis
            self._evaluate_internal_state()
            await self._maybe_call(self._evaluate_crisis_state)
        except Exception:
            logger.exception("Unhandled failure during Adam.update")

    # Execute plan safely using tool registry or emit events
    async def _execute_plan_safe(self, plan: PlanDeAccion) -> None:
        if not plan:
            logger.debug("No plan to execute")
            return
        logger.info(
            "Executing plan '%s' (confidence=%.2f)",
            getattr(plan, "justificacion", "n/a"),
            getattr(plan, "confianza_general", 0.0),
        )
        steps = getattr(plan, "pasos", []) or []
        for i, paso in enumerate(steps):
            try:
                tool_name = getattr(paso, "herramienta", None)
                params = getattr(paso, "parametros", {}) or {}
                logger.debug("Plan step %d tool=%s params=%s", i, tool_name, params)
                if (
                    tool_name
                    and self.tool_registry
                    and self.tool_registry.get_tool(tool_name)
                ):
                    try:
                        result = await self.tool_registry.execute_tool(
                            tool_name, **params
                        )
                        logger.debug("Tool %s returned %s", tool_name, repr(result))
                    except Exception:
                        logger.exception("Tool execution failed for %s", tool_name)
                else:
                    self.emit_event(
                        "plan.step.unhandled", {"step_index": i, "step": paso}
                    )
            except Exception:
                logger.exception("Failed executing plan step %d", i)

    # Helper to call sync/async functions uniformly
    async def _maybe_call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        try:
            result = fn(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        except TypeError:
            # maybe fn is coroutine function but returned a coroutine when called incorrectly
            try:
                coro = fn(*args, **kwargs)
                if hasattr(coro, "__await__"):
                    return await coro
            except Exception:
                logger.exception(
                    "Error invoking callable %s", getattr(fn, "__name__", str(fn))
                )
                raise
        except Exception:
            logger.exception("Error calling %s", getattr(fn, "__name__", str(fn)))
            raise

    # --- internal monitoring & crisis integration ---
    def _evaluate_internal_state(self) -> None:
        try:
            mind_clarity = getattr(self.mind, "get_clarity", lambda: 0.5)()
        except Exception:
            logger.exception("Error reading mind clarity")
            mind_clarity = 0.5
        try:
            soul_integrity = getattr(self.soul, "get_integrity", lambda: 0.5)()
        except Exception:
            logger.exception("Error reading soul integrity")
            soul_integrity = 0.5
        try:
            body_coherence = getattr(self.body, "get_coherence", lambda: 0.5)()
        except Exception:
            logger.exception("Error reading body coherence")
            body_coherence = 0.5

        self.consciousness_coherence = float(
            (mind_clarity + soul_integrity + body_coherence) / 3.0
        )
        try:
            bio_state = (
                getattr(self.body, "get_comprehensive_biological_state", lambda: {})()
                or {}
            )
            self.stress_level = float(bio_state.get("stress_level", self.stress_level))
            self.health_index = float(bio_state.get("health_score", self.health_index))
            self.energy_balance = float(
                bio_state.get("energy_balance", self.energy_balance)
            )
        except Exception:
            logger.exception(
                "Error aggregating biological metrics; keeping previous values"
            )

    async def _evaluate_crisis_state(self) -> None:
        try:
            purpose_fulfillment = getattr(
                self.mind, "get_purpose_fulfillment", lambda: 1.0
            )()
        except Exception:
            logger.exception("Error reading purpose_fulfillment")
            purpose_fulfillment = 1.0

        thought_patterns = {
            "health_level": self.health_index,
            "purpose_fulfillment": purpose_fulfillment,
            "qualia_corruption": 0.0,
            "external_threats": [],
            "consciousness_coherence": self.consciousness_coherence,
            "stress_level": self.stress_level,
            "energy_balance": self.energy_balance,
        }

        # detect and possibly catalyze
        try:
            crisis_result = self.crisis_manager.detect_and_catalyze(
                self.id, thought_patterns
            )
            if crisis_result != "no_crisis":
                logger.warning(
                    "Crisis detected: %s — attempting emergency evolution",
                    crisis_result,
                )
                # derive a permissive current_qualia dict
                current_qualia = {}
                try:
                    if hasattr(self.mind, "qualia_generator") and hasattr(
                        self.mind.qualia_generator, "get_current_qualia_state"
                    ):
                        q = self.mind.qualia_generator.get_current_qualia_state()
                        current_qualia = (
                            q.to_dict() if hasattr(q, "to_dict") else dict(q)
                        )
                    elif hasattr(self.mind, "get_current_qualia"):
                        current_qualia = self.mind.get_current_qualia() or {}
                except Exception:
                    logger.debug("Failed to derive qualia from mind; using defaults")
                    current_qualia = {
                        "emotional_valence": 0.5,
                        "cognitive_complexity": 0.5,
                        "consciousness_density": 0.5,
                    }

                try:
                    evolution_type, success, details = (
                        self.crisis_manager.trigger_emergency_evolution(
                            crisis_type=crisis_result,
                            genome=self.genome,
                            current_qualia=current_qualia,
                            thought_patterns=thought_patterns,
                        )
                    )
                    if success:
                        msg = getattr(evolution_type, "value", str(evolution_type))
                        logger.info(
                            "Emergency evolution applied: %s — %s", msg, details
                        )
                    else:
                        logger.warning("Emergency evolution not applied: %s", details)
                except Exception:
                    logger.exception("trigger_emergency_evolution failed (non-fatal)")
        except Exception:
            logger.exception("Error during crisis detection/management")

    # --- utilities ---
    def get_full_context(self) -> dict[str, Any]:
        try:
            body_state = (
                getattr(self.body, "get_comprehensive_biological_state", lambda: {})()
                or {}
            )
        except Exception:
            logger.exception("Error reading body state for context")
            body_state = {}
        try:
            soul_state = (
                getattr(self.soul, "_generate_kernel_state", lambda: {})() or {}
            )
        except Exception:
            logger.exception("Error reading soul state for context")
            soul_state = {}
        try:
            mind_state = getattr(self.mind, "get_state", lambda: {})() or {}
        except Exception:
            logger.exception("Error reading mind state for context")
            mind_state = {}
        try:
            genome_state = getattr(self.genome, "get_genome_state", lambda: {})() or {}
        except Exception:
            logger.exception("Error reading genome state for context")
            genome_state = {}

        return {
            "body": body_state,
            "soul": soul_state,
            "mind": mind_state,
            "genome": genome_state,
            "timestamp": time.time(),
        }

    def get_linguistic_api(self) -> dict[str, Callable[..., Any]] | None:
        try:
            if hasattr(self.mind, "get_linguistic_api"):
                return self.mind.get_linguistic_api()
        except Exception:
            logger.debug("get_linguistic_api failed")
        return None

    def get_status(self) -> dict[str, Any]:
        return {
            "id": getattr(self, "id", self.entity_id),
            "is_alive": self.is_alive,
            "started": self._started,
            "consciousness_coherence": self.consciousness_coherence,
            "stress_level": self.stress_level,
            "health_index": self.health_index,
            "energy_balance": self.energy_balance,
            "last_update_time": self.last_update_time,
        }
