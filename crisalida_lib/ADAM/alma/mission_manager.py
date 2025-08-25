import asyncio
import logging
import time
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field as PydanticField

from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
from crisalida_lib.EVA.typequalia import QualiaState

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MissionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class Mission(BaseModel):
    mission_id: str = PydanticField(
        default_factory=lambda: f"mission_{uuid4().hex}_{int(time.time())}",
        description="Unique identifier for the mission.",
    )
    name: str = PydanticField(..., description="Name of the mission.")
    description: str = PydanticField(
        ..., description="Detailed description of the mission."
    )
    priority: int = PydanticField(
        5, ge=1, le=10, description="Priority level (1-10, 10 being highest)."
    )
    status: MissionStatus = PydanticField(
        MissionStatus.ACTIVE, description="Current status of the mission."
    )
    assigned_by: str = PydanticField(
        "system", description="Entity that assigned the mission."
    )
    creation_timestamp: float = PydanticField(
        default_factory=time.time, description="Timestamp of mission creation."
    )
    last_update_timestamp: float = PydanticField(
        default_factory=time.time, description="Timestamp of last mission update."
    )
    target_metrics: dict[str, Any] = PydanticField(
        default_factory=dict, description="Metrics to track mission progress."
    )
    related_objectives: list[str] = PydanticField(
        default_factory=list, description="List of related objectives or goals."
    )
    metadata: dict[str, Any] = PydanticField(
        default_factory=dict, description="Additional metadata for the mission."
    )

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def update_status(
        self, new_status: MissionStatus, reason: str | None = None
    ) -> None:
        self.status = new_status
        self.last_update_timestamp = time.time()
        if reason:
            self.metadata["last_status_reason"] = reason
        logger.info("Mission '%s' status updated to %s.", self.name, new_status.value)


class MissionManager:
    """Thread-safe mission registry with EVA recording and simple history.

    Integrates with EVA via [`EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    and uses [`QualiaState`](crisalida_lib/EVA/typequalia.py) for event qualia.
    """

    MAX_HISTORY = 200

    def __init__(
        self,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self._lock = RLock()
        self.active_missions: dict[str, Mission] = {}
        self.completed_missions: dict[str, Mission] = {}
        self.failed_missions: dict[str, Mission] = {}
        self.eva_manager = eva_manager
        self.entity_id = entity_id
        self.mission_event_history: list[dict[str, Any]] = []
        logger.info("MissionManager initialized for entity '%s'.", self.entity_id)

    def add_mission(self, mission: Mission) -> bool:
        with self._lock:
            if (
                mission.mission_id in self.active_missions
                or mission.mission_id in self.completed_missions
                or mission.mission_id in self.failed_missions
            ):
                logger.warning(
                    "Mission '%s' (ID=%s) already exists.",
                    mission.name,
                    mission.mission_id,
                )
                return False
            self.active_missions[mission.mission_id] = mission
            self._record_mission_event(mission, "mission_added")
            logger.info("Mission '%s' added and set to ACTIVE.", mission.name)
            return True

    def get_mission(self, mission_id: str) -> Mission | None:
        with self._lock:
            return (
                self.active_missions.get(mission_id)
                or self.completed_missions.get(mission_id)
                or self.failed_missions.get(mission_id)
            )

    def get_all_missions(self) -> list[Mission]:
        with self._lock:
            return (
                list(self.active_missions.values())
                + list(self.completed_missions.values())
                + list(self.failed_missions.values())
            )

    def get_missions_by_status(self, status: MissionStatus) -> list[Mission]:
        with self._lock:
            if status == MissionStatus.ACTIVE:
                return list(self.active_missions.values())
            if status == MissionStatus.COMPLETED:
                return list(self.completed_missions.values())
            if status == MissionStatus.FAILED:
                return list(self.failed_missions.values())
            # For PAUSED/CANCELLED search across active set
            return [m for m in self.get_all_missions() if m.status == status]

    def get_current_primary_mission(self) -> Mission | None:
        with self._lock:
            active = [
                m
                for m in self.active_missions.values()
                if m.status == MissionStatus.ACTIVE
            ]
            if not active:
                return None
            # Prioritize by highest priority then by oldest creation timestamp
            active.sort(key=lambda m: (-m.priority, m.creation_timestamp))
            return active[0]

    def update_mission_status(
        self, mission_id: str, new_status: MissionStatus, reason: str | None = None
    ) -> bool:
        with self._lock:
            mission = self.active_missions.get(mission_id)
            # allow updating completed/failed missions only to record metadata, but not move them again
            if mission is None:
                # maybe it is in completed/failed already
                mission = self.completed_missions.get(
                    mission_id
                ) or self.failed_missions.get(mission_id)
                if mission is None:
                    logger.warning("Mission ID '%s' not found.", mission_id)
                    return False
                # allow metadata update but not status transition
                mission.update_status(new_status, reason)
                self._record_mission_event(
                    mission, f"status_updated_{new_status.value}"
                )
                return True

            old_status = mission.status
            mission.update_status(new_status, reason)

            # Move between buckets as appropriate
            if new_status == MissionStatus.COMPLETED:
                self.completed_missions[mission_id] = self.active_missions.pop(
                    mission_id
                )
            elif new_status == MissionStatus.FAILED:
                self.failed_missions[mission_id] = self.active_missions.pop(mission_id)
            elif new_status == MissionStatus.CANCELLED:
                # Remove silently from active missions (no failed entry)
                self.active_missions.pop(mission_id, None)

            self._record_mission_event(
                mission, f"status_updated_from_{old_status.value}_to_{new_status.value}"
            )
            return True

    def _record_mission_event(self, mission: Mission, event_type: str) -> None:
        """Best-effort EVA recording that is async/sync aware and keeps local history."""
        with self._lock:
            entry = {
                "entity_id": self.entity_id,
                "timestamp": time.time(),
                "mission_id": mission.mission_id,
                "event_type": event_type,
                "mission_snapshot": mission.model_dump(),
            }
            self.mission_event_history.append(entry)
            if len(self.mission_event_history) > self.MAX_HISTORY:
                self.mission_event_history = self.mission_event_history[
                    -self.MAX_HISTORY :
                ]

        if not self.eva_manager:
            logger.debug(
                "No EVAMemoryManager attached; skipping remote EVA record for event '%s'.",
                event_type,
            )
            return

        qualia = QualiaState(
            emotional=0.5,
            importance=float(mission.priority) / 10.0,
            consciousness=0.5,
            energy=0.5,
            complexity=0.5,
            temporal=0.5,
        )

        experience_id = f"mission:{mission.mission_id}:{event_type}:{int(time.time())}"

        try:
            rec = getattr(self.eva_manager, "record_experience", None)
            if rec is None:
                logger.debug(
                    "EVAMemoryManager.record_experience not available; skipping."
                )
                return
            result = rec(
                entity_id=self.entity_id,
                event_type=f"mission.{event_type}",
                data=entry["mission_snapshot"],
                qualia_state=qualia,
                experience_id=experience_id,
            )
            # If recorder returned a coroutine, schedule it (best-effort)
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(result)
                        logger.debug("Scheduled async EVA record for %s", experience_id)
                    else:
                        loop.run_until_complete(result)
                        logger.debug(
                            "Completed EVA async record sync for %s", experience_id
                        )
                except Exception:
                    logger.exception(
                        "Failed to schedule/run async EVA record for %s", experience_id
                    )
            else:
                logger.debug(
                    "Recorded mission event '%s' to EVA (sync).", experience_id
                )
        except Exception:
            logger.exception("Failed to record mission event to EVA: %s", event_type)

    def to_serializable(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entity_id": self.entity_id,
                "timestamp": time.time(),
                "active_missions": [
                    m.model_dump() for m in self.active_missions.values()
                ],
                "completed_missions": [
                    m.model_dump() for m in self.completed_missions.values()
                ],
                "failed_missions": [
                    m.model_dump() for m in self.failed_missions.values()
                ],
                "mission_event_history": list(self.mission_event_history),
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        with self._lock:
            try:
                self.entity_id = data.get("entity_id", self.entity_id)
                for mdata in data.get("active_missions", []):
                    m = Mission.model_validate(mdata)
                    self.active_missions[m.mission_id] = m
                for mdata in data.get("completed_missions", []):
                    m = Mission.model_validate(mdata)
                    self.completed_missions[m.mission_id] = m
                for mdata in data.get("failed_missions", []):
                    m = Mission.model_validate(mdata)
                    self.failed_missions[m.mission_id] = m
                self.mission_event_history = list(
                    data.get("mission_event_history", [])
                )[: self.MAX_HISTORY]
                self._trim_collections()
            except Exception:
                logger.exception("Failed to load serializable mission manager state")

    def _trim_collections(self) -> None:
        # Keep sizes bounded to avoid memory growth
        with self._lock:
            if len(self.mission_event_history) > self.MAX_HISTORY:
                self.mission_event_history = self.mission_event_history[
                    -self.MAX_HISTORY :
                ]
            # Optionally cap missions stored (not required but useful)
            # e.g., keep last N completed/failed
            MAX_ARCHIVE = 500
            if len(self.completed_missions) > MAX_ARCHIVE:
                keys = list(self.completed_missions.keys())[-MAX_ARCHIVE:]
                self.completed_missions = {k: self.completed_missions[k] for k in keys}
            if len(self.failed_missions) > MAX_ARCHIVE:
                keys = list(self.failed_missions.keys())[-MAX_ARCHIVE:]
                self.failed_missions = {k: self.failed_missions[k] for k in keys}
