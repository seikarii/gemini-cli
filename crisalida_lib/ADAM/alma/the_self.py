"""
The Self System
===============

This module defines the ElYo class, representing the conscious 'Self' of the entity.
It integrates subjectivity, internal simulation, and identity management.

Refactored to be decoupled from EVA, using EVAMemoryManager for persistence.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import AdamConfig

# Prefer modern QualiaState location but gracefully fallback
try:
    from crisalida_lib.EVA.types import QualiaState  # type: ignore
except Exception:
    try:
        from crisalida_lib.EVA.typequalia import QualiaState  # type: ignore
    except Exception:
        QualiaState = Any  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )
else:
    EVAMemoryManager = Any  # runtime fallback

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MotorDeSubjetividad:
    """
    Núcleo generador de la experiencia subjetiva del "Yo".
    Gestiona narrativa interna, integración de experiencias y análisis de identidad.
    """

    def __init__(self) -> None:
        self.narrativa_interna: str = ""
        self.experiencias: list[dict[str, Any]] = []
        self.identidad_tags: list[str] = []
        self.last_update: str = datetime.utcnow().isoformat()
        self._history_limit: int = 200
        self._local_history: list[dict[str, Any]] = []

    def update_narrative(self, experience: str, tags: list[str] | None = None) -> None:
        now = datetime.utcnow().isoformat()
        self.narrativa_interna += f"\n- {experience}"
        rec = {"timestamp": now, "experience": experience, "tags": tags or []}
        self.experiencias.append(rec)
        self.last_update = now
        self._record_local(rec)

    def add_identity_tag(self, tag: str) -> None:
        if tag not in self.identidad_tags:
            self.identidad_tags.append(tag)

    def remove_identity_tag(self, tag: str) -> None:
        if tag in self.identidad_tags:
            self.identidad_tags.remove(tag)

    def get_narrative_summary(self) -> dict[str, Any]:
        return {
            "narrative_length": (
                len(self.narrativa_interna.split("\n")) if self.narrativa_interna else 0
            ),
            "last_update": self.last_update,
            "identity_tags": self.identidad_tags.copy(),
            "recent_experiences": copy.deepcopy(self.experiencias[-5:]),
        }

    def clear_narrative(self) -> None:
        self.narrativa_interna = ""
        self.experiencias.clear()
        self.last_update = datetime.utcnow().isoformat()
        self._local_history.clear()

    def _record_local(self, entry: dict[str, Any]) -> None:
        self._local_history.append(entry)
        if len(self._local_history) > self._history_limit:
            self._local_history = self._local_history[-self._history_limit :]


class SimuladorInterno:
    """
    SimuladorInterno - Simula escenarios para planificación, sueños y visualización interna.
    """

    def sonar(self, recent_memories: dict[str, Any]) -> str:
        if not recent_memories:
            return "Un sueño tranquilo y sin forma."
        dream_elements = list(recent_memories.values())
        dream_tags = [str(e)[:40] for e in dream_elements[:3]]
        return f"Un sueño vívido sobre: {', '.join(dream_tags)}"

    def plan_scenario(self, goal: str, context: dict[str, Any]) -> str:
        context_summary = ", ".join(f"{k}: {v}" for k, v in context.items())
        return f"Planificando para '{goal}' con contexto: {context_summary}"

    def visualize_identity(self, identity_tags: list[str]) -> str:
        if not identity_tags:
            return "Identidad difusa."
        return f"Identidad visualizada como: {', '.join(identity_tags)}"


class ElYo:
    """
    ElYo - Núcleo del "Yo" consciente.
    Integra subjetividad, simulación interna y gestión de identidad.

    Diseño:
      - Evitar acoplamientos de importación en tiempo de carga.
      - Soporte para EVAMemoryManager sin asumir su forma (best-effort).
      - Registro local e intento de persistencia async/sync (mejor esfuerzo).
    """

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config: AdamConfig = config or AdamConfig()  # type: ignore
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id

        self.id: str = str(uuid.uuid4())
        self.subjetividad: MotorDeSubjetividad = MotorDeSubjetividad()
        self.simulador: SimuladorInterno = SimuladorInterno()
        self.creation_time: str = datetime.utcnow().isoformat()

        self._lock = RLock()
        self._event_history: list[dict[str, Any]] = []
        self._history_limit: int = 500
        self._last_eva_record_ts: float = 0.0

        logger.info("ElYo initialized: id=%s entity=%s", self.id, self.entity_id)

    # --- Introspection & Serialization ---------------------------------
    def get_self_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "creation_time": self.creation_time,
                "narrative": self.subjetividad.narrativa_interna,
                "identity_tags": self.subjetividad.identidad_tags.copy(),
                "recent_experiences": copy.deepcopy(
                    self.subjetividad.experiencias[-3:]
                ),
                "local_history_len": len(self._event_history),
            }

    def to_serializable(self) -> dict[str, Any]:
        """Compact snapshot safe for persistence/transfer."""
        with self._lock:
            return {
                "id": self.id,
                "creation_time": self.creation_time,
                "narrativa_interna": self.subjetividad.narrativa_interna,
                "experiencias": copy.deepcopy(self.subjetividad.experiencias),
                "identity_tags": self.subjetividad.identidad_tags.copy(),
                "last_update": self.subjetividad.last_update,
                "metadata": {"entity_id": self.entity_id},
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Best-effort restore from a snapshot produced by to_serializable()."""
        if not data:
            return
        with self._lock:
            try:
                self.id = data.get("id", self.id)
                self.creation_time = data.get("creation_time", self.creation_time)
                self.subjetividad.narrativa_interna = data.get(
                    "narrativa_interna", self.subjetividad.narrativa_interna
                )
                self.subjetividad.experiencias = data.get(
                    "experiencias", list(self.subjetividad.experiencias)
                )
                self.subjetividad.identidad_tags = data.get(
                    "identity_tags", list(self.subjetividad.identidad_tags)
                )
                self.subjetividad.last_update = data.get(
                    "last_update", self.subjetividad.last_update
                )
            except Exception:
                logger.exception("Failed to load ElYo serializable data")

    # --- Narrative / Experience ------------------------------------------------
    def simulate_dream(self, recent_memories: dict[str, Any]) -> str:
        return self.simulador.sonar(recent_memories)

    def simulate_plan(self, goal: str, context: dict[str, Any]) -> str:
        return self.simulador.plan_scenario(goal, context)

    def visualize_identity(self) -> str:
        return self.simulador.visualize_identity(self.subjetividad.identidad_tags)

    def update_self_narrative(
        self, experience: str, tags: list[str] | None = None
    ) -> None:
        """Update internal narrative and attempt to persist the update to EVA (best-effort)."""
        with self._lock:
            self.subjetividad.update_narrative(experience, tags)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "experience": experience,
                "tags": tags or [],
            }
            self._event_history.append(entry)
            if len(self._event_history) > self._history_limit:
                self._event_history = self._event_history[-self._history_limit :]

        # Best-effort EVA recording (sync/async aware)
        if not self.eva_manager:
            return

        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                return
            payload = {
                "entity_id": self.entity_id,
                "self_id": self.id,
                "event_type": "self_narrative_update",
                "data": entry,
                "timestamp": time.time(),
            }
            res = recorder(
                entity_id=self.entity_id,
                event_type="self_narrative_update",
                data=payload,
            )
            # schedule coroutine if necessary
            if hasattr(res, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(res)
                    else:
                        loop.run_until_complete(res)
                except Exception:
                    logger.debug("Failed to schedule async EVA record", exc_info=True)
            self._last_eva_record_ts = time.time()
        except Exception:
            logger.exception("Failed to record self narrative to EVA")

    def ingest_self_experience(
        self,
        context: dict[str, Any],
        qualia_state: QualiaState | None = None,
        event_type: str = "self_experience",
    ) -> str | None:
        """Record a self experience into EVA via EVAMemoryManager (best-effort).

        Returns experience id or None if not recorded / manager absent.
        """
        if not self.eva_manager:
            return None

        # Construct a safe/default QualiaState if possible
        if qualia_state is None:
            try:
                qualia_state = QualiaState(
                    emotional_valence=0.7,
                    cognitive_complexity=0.8,
                    consciousness_density=0.7,
                    narrative_importance=0.8,
                    energy_level=1.0,
                )
            except Exception:
                # Fallback to a minimal dict-like payload if QualiaState constructor is unavailable
                qualia_state = {
                    "emotional_valence": 0.7,
                    "cognitive_complexity": 0.8,
                    "consciousness_density": 0.7,
                }

        experience_data = {
            "self_id": self.id,
            "creation_time": self.creation_time,
            "context": context,
            "narrative": self.subjetividad.narrativa_interna,
            "identity_tags": self.subjetividad.identidad_tags.copy(),
            "recent_experiences": copy.deepcopy(self.subjetividad.experiencias[-3:]),
        }

        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                return None
            res = recorder(
                entity_id=self.entity_id,
                event_type=event_type,
                data=experience_data,
                qualia_state=qualia_state,
            )
            if hasattr(res, "__await__"):
                # schedule it (best-effort) and return a generated id
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(res)
                    else:
                        loop.run_until_complete(res)
                except Exception:
                    logger.debug("Could not run async EVA record", exc_info=True)
            # best-effort: attempt to synthesize an experience id
            experience_id = f"self:{self.id}:{int(time.time())}"
            return experience_id
        except Exception:
            logger.exception("ingest_self_experience failed")
            return None

    def recall_self_experience(
        self, cue: str, event_type: str = "self_experience"
    ) -> dict[str, Any] | None:
        """Recall an experience from EVA (best-effort). Returns the recalled data or None."""
        if not self.eva_manager:
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not callable(recall):
                return None
            res = recall(entity_id=self.entity_id, cue=cue, event_type=event_type)
            if hasattr(res, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # cannot block the running loop reliably; schedule and return None
                        asyncio.create_task(res)
                        logger.debug(
                            "Scheduled async EVA recall; returning None until available"
                        )
                        return None
                    else:
                        return loop.run_until_complete(res)
                except Exception:
                    logger.exception("Async EVA recall failed")
                    return None
            return res
        except Exception:
            logger.exception("recall_self_experience failed")
            return None

    # Utility helpers
    def set_eva_manager(self, eva_manager: EVAMemoryManager | None) -> None:
        """Attach or replace EVAMemoryManager at runtime."""
        with self._lock:
            self.eva_manager = eva_manager

    def clear_local_history(self) -> None:
        with self._lock:
            self._event_history.clear()
            self.subjetividad._local_history.clear()
