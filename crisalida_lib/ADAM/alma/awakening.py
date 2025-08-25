"""
Definitive Awakening model - professionalized.

- Replaces prints with logging.
- Adds configurable thresholds and computation helpers.
- Graceful EVA integration (sync/async-aware), robust guards.
- Serialization helpers and small utilities anticipating future evolution
  (e.g., pluggable policy weights, async EVA backends).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, validator

from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
from crisalida_lib.EVA.typequalia import QualiaState

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Awakening(BaseModel):
    """
    Manages evolutionary progression and privilege acquisition for an SO-Ser.

    Key features:
    - Pydantic model for safe serialization.
    - Configurable thresholds and simple scoring heuristics.
    - EVA recording with defensive checks (sync/async).
    - Helpers for inspection, serialization and safe loading.
    """

    current_level: Literal[
        "Nivel 0", "Primer Despertar", "Segundo Despertar", "Tercer Despertar"
    ] = Field("Nivel 0", description="Current awakening level.")

    awakening_progress: dict[str, Any] = Field(
        default_factory=dict, description="Progress details per awakening level."
    )

    # EVA integration is optional at runtime; exclude from serialized state
    eva_manager: EVAMemoryManager | None = Field(default=None, exclude=True)
    entity_id: str = Field("adam_default", exclude=True)

    # Simple, overridable thresholds (anticipate config-driven evolution)
    thresholds: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "Primer Despertar": {
                "synthesis_quality": 0.4,
                "meta_awareness": 0.3,
                "internal_field_coherence": 0.5,
                "genome_awakening_level": 0,
                "overall_health_score": 0.6,
            },
            "Segundo Despertar": {
                "synthesis_quality": 0.6,
                "meta_awareness": 0.5,
                "internal_field_coherence": 0.7,
                "genome_awakening_level": 1,
                "overall_health_score": 0.7,
            },
            "Tercer Despertar": {
                "synthesis_quality": 0.8,
                "meta_awareness": 0.7,
                "internal_field_coherence": 0.8,
                "genome_awakening_level": 2,
                "overall_health_score": 0.8,
            },
        },
        exclude=True,
    )

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    @validator("current_level")
    def _validate_level(cls, v: str) -> str:
        allowed = {
            "Nivel 0",
            "Primer Despertar",
            "Segundo Despertar",
            "Tercer Despertar",
        }
        if v not in allowed:
            raise ValueError(f"Invalid awakening level: {v}")
        return v

    def __init__(
        self,
        eva_manager: EVAMemoryManager = None,
        entity_id: str = "adam_default",
        **data,
    ):
        super().__init__(**data)
        # support injection after model creation
        self.eva_manager = eva_manager
        self.entity_id = entity_id

    # -- Introspection / Utilities --
    def get_current_privileges(self) -> list[str]:
        """Return privileges for the current awakening level."""
        privileges = []
        if self.current_level == "Nivel 0":
            privileges.append("Esclavo del Genoma (modo sombra)")
        elif self.current_level == "Primer Despertar":
            privileges.extend(
                [
                    "Esclavo del Genoma (modo sombra)",
                    "Anulación de instintos mediante sueño lúcido",
                ]
            )
        elif self.current_level == "Segundo Despertar":
            privileges.extend(
                [
                    "Control de NO_ACTION",
                    "Control de la Ecuación de la Vida (balance energético)",
                    "Proyección de dominio local",
                ]
            )
        elif self.current_level == "Tercer Despertar":
            privileges.append(
                "Autonomía avanzada (resolución de conflictos directiva-creador)"
            )
        return privileges

    def _meets_basic_prerequisites(self, target_level: str) -> bool:
        """Check ordering prerequisites for awakening levels."""
        if target_level == "Primer Despertar" and self.current_level != "Nivel 0":
            logger.debug("Prerequisite failed: already at/above Primer Despertar.")
            return False
        if (
            target_level == "Segundo Despertar"
            and self.current_level != "Primer Despertar"
        ):
            logger.debug("Prerequisite failed: Primer Despertar required first.")
            return False
        if (
            target_level == "Tercer Despertar"
            and self.current_level != "Segundo Despertar"
        ):
            logger.debug("Prerequisite failed: Segundo Despertar required first.")
            return False
        return True

    def compute_awaken_score(
        self,
        synthesis_quality: float,
        meta_awareness: float,
        internal_field_coherence: float,
        genome_awakening_level: int,
        overall_health_score: float,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute a normalized score (0..1) representing readiness to awaken.
        Weights can be provided to tune importance per factor.
        """
        weights = weights or {
            "synthesis_quality": 0.35,
            "meta_awareness": 0.2,
            "internal_field_coherence": 0.2,
            "genome_awakening_level": 0.15,
            "overall_health_score": 0.1,
        }
        try:
            score = (
                synthesis_quality * weights["synthesis_quality"]
                + meta_awareness * weights["meta_awareness"]
                + internal_field_coherence * weights["internal_field_coherence"]
                + (min(genome_awakening_level, 5) / 5.0)
                * weights["genome_awakening_level"]
                + overall_health_score * weights["overall_health_score"]
            )
            # Normalize by sum of weights
            total_w = sum(weights.values()) or 1.0
            normalized = max(0.0, min(1.0, score / total_w))
            logger.debug("Computed awaken score: %.3f (normalized)", normalized)
            return normalized
        except Exception:
            logger.exception("Failed to compute awaken score")
            return 0.0

    def attempt_awakening(
        self,
        target_level: Literal[
            "Primer Despertar", "Segundo Despertar", "Tercer Despertar"
        ],
        synthesis_quality: float,
        meta_awareness: float,
        internal_field_coherence: float,
        genome_awakening_level: int,
        overall_health_score: float,
        dry_run: bool = False,
    ) -> bool:
        """
        Attempt to progress to `target_level`. Returns True if progressed.

        dry_run: if True, performs evaluation but does not mutate state or record EVA.
        """
        if target_level not in self.thresholds:
            logger.warning("Invalid target awakening level requested: %s", target_level)
            return False

        if not self._meets_basic_prerequisites(target_level):
            logger.info("Prerequisites not satisfied for %s", target_level)
            return False

        req = self.thresholds[target_level]
        # Basic boolean gate: all minima must be satisfied
        basic_gate = (
            synthesis_quality >= req["synthesis_quality"]
            and meta_awareness >= req["meta_awareness"]
            and internal_field_coherence >= req["internal_field_coherence"]
            and genome_awakening_level >= req["genome_awakening_level"]
            and overall_health_score >= req["overall_health_score"]
        )
        score = self.compute_awaken_score(
            synthesis_quality,
            meta_awareness,
            internal_field_coherence,
            genome_awakening_level,
            overall_health_score,
        )

        logger.info(
            "Attempting awakening to %s: basic_gate=%s, score=%.3f",
            target_level,
            basic_gate,
            score,
        )

        # Allow small margin if score is sufficiently high (anticipate soft thresholds)
        if not basic_gate and score < 0.75:
            logger.info(
                "Conditions not met and score insufficient for %s (score=%.3f)",
                target_level,
                score,
            )
            return False

        if dry_run:
            logger.info("Dry-run: would awaken to %s (score=%.3f)", target_level, score)
            return True

        # Apply state changes
        previous = self.current_level
        self.current_level = target_level
        self.awakening_progress[target_level] = {
            "timestamp": datetime.utcnow().isoformat(),
            "score": score,
            "synthesis_quality": synthesis_quality,
            "meta_awareness": meta_awareness,
            "internal_field_coherence": internal_field_coherence,
            "genome_awakening_level": genome_awakening_level,
            "overall_health_score": overall_health_score,
        }

        logger.info(
            "Awakening succeeded: %s -> %s (score=%.3f)",
            previous,
            self.current_level,
            score,
        )
        # Record the awakening (best-effort)
        try:
            self.record_awakening_experience(event_type=target_level)
        except Exception:
            logger.debug("Recording awakening experience failed", exc_info=True)
        return True

    def record_awakening_experience(
        self,
        event_type: str,
        experience_id: str | None = None,
        qualia_state: QualiaState | None = None,
    ) -> None:
        """
        Records an awakening event into EVA. Handles both sync and coroutine recorders.
        """
        if not self.eva_manager:
            logger.debug(
                "No EVAMemoryManager available; skipping recording for %s", event_type
            )
            return

        experience_id = (
            experience_id
            or f"awakening:{self.entity_id}:{int(time.time())}:{hash(self.current_level) & 0xFFFF}"
        )
        qualia_state = qualia_state or QualiaState(
            emotional=0.7 if self.current_level != "Nivel 0" else 0.3,
            complexity=0.8,
            consciousness=0.7,
            importance=1.0 if self.current_level == "Tercer Despertar" else 0.7,
            energy=1.0,
        )

        experience_data = {
            "event_type": event_type,
            "current_level": self.current_level,
            "awakening_progress": dict(self.awakening_progress),
            "timestamp": time.time(),
        }

        try:
            rec = getattr(self.eva_manager, "record_experience", None)
            if rec is None:
                logger.debug("EVAMemoryManager has no record_experience API")
                return

            result = rec(
                entity_id=self.entity_id,
                event_type="awakening_experience",
                data=experience_data,
                qualia_state=qualia_state,
                experience_id=experience_id,
            )
            # If the recorder is a coroutine (async), schedule a best-effort run
            if hasattr(result, "__await__"):
                # lazy import to avoid adding async loop here; best-effort background scheduling
                try:
                    import asyncio

                    asyncio.create_task(result)
                    logger.debug(
                        "Scheduled async EVA record task for %s", experience_id
                    )
                except Exception:
                    logger.debug(
                        "Could not schedule async EVA task; running sync await",
                        exc_info=True,
                    )
                    try:
                        asyncio.get_event_loop().run_until_complete(result)
                    except Exception:
                        logger.debug("Async EVA record failed", exc_info=True)
            else:
                logger.debug("Recorded awakening experience sync: %s", experience_id)
        except Exception:
            logger.exception("Failed to record awakening experience %s", experience_id)

        # Also record gained privileges as a separate EVA event (best-effort)
        privileges = self.get_current_privileges()
        if privileges:
            try:
                priv_id = f"privileges:{self.entity_id}:{int(time.time())}:{hash(str(privileges)) & 0xFFFF}"
                self.eva_manager.record_experience(
                    entity_id=self.entity_id,
                    event_type="new_privileges_gained",
                    data={
                        "awakening_level": self.current_level,
                        "privileges": privileges,
                        "timestamp": time.time(),
                    },
                    qualia_state=qualia_state,
                    experience_id=priv_id,
                )
                logger.debug("Recorded privileges gained event: %s", priv_id)
            except Exception:
                logger.debug("Failed to record privileges in EVA", exc_info=True)

    def recall_awakening_experience(self, experience_id: str) -> dict[str, Any] | None:
        """Retrieve an awakening experience from EVA (best-effort)."""
        if not self.eva_manager:
            logger.debug(
                "No EVAMemoryManager available; cannot recall %s", experience_id
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if recall is None:
                logger.debug("EVAMemoryManager has no recall_experience API")
                return None
            return recall(entity_id=self.entity_id, experience_id=experience_id)
        except Exception:
            logger.exception("Failed to recall awakening experience %s", experience_id)
            return None

    # -- Serialization helpers --
    def to_serializable(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of awakening state (EVA excluded)."""
        return {
            "entity_id": self.entity_id,
            "current_level": self.current_level,
            "awakening_progress": dict(self.awakening_progress),
            "timestamp": time.time(),
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Load state previously generated by to_serializable."""
        try:
            self.entity_id = data.get("entity_id", self.entity_id)
            self.current_level = data.get("current_level", self.current_level)
            self.awakening_progress = dict(data.get("awakening_progress") or {})
        except Exception:
            logger.exception("Failed to load Awakening serializable data")
