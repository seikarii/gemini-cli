"""
EventBus - central asynchronous event dispatcher for the simulation.
Provides a lightweight EventBus and an EVA-aware EVAEventBus that integrates
with the EVAMemoryMixin when available.
"""

import asyncio
from typing import Any

from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin


class EventBus(EVAMemoryMixin):
    def __init__(self):
        # Initialize EVA Memory using centralized mixin if available
        init_fn = getattr(self, "_init_eva_memory", None)
        if callable(init_fn):
            try:
                init_fn()
            except Exception:
                # best-effort: proceed without EVA memory initialization
                pass

        self.subscribers: dict[str, list[dict[str, Any]]] = {}

    async def subscribe(self, event_type, handler, pass_delta_time: bool = False):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(
            {"handler": handler, "pass_delta_time": pass_delta_time}
        )

    async def unsubscribe(self, event_type, handler):
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                sub for sub in self.subscribers[event_type] if sub["handler"] != handler
            ]

    async def publish(self, event_type, data=None, delta_time=None):
        tasks = []
        for sub in self.subscribers.get(event_type, []):
            if sub.get("pass_delta_time"):
                tasks.append(sub["handler"](data, delta_time))
            else:
                tasks.append(sub["handler"](data))
        if tasks:
            await asyncio.gather(*tasks)


event_bus = EventBus()


class EVAEventBus(EventBus):
    """
    EVAEventBus - extended EventBus integrating with EVAMemoryMixin.
    Ingests published events into EVA memory as experiences.
    """

    def __init__(self, phase: str | None = "default"):
        super().__init__()
        # coerce Optional[str] to concrete str for mixin compatibility
        self.set_memory_phase(phase or "default")

    async def publish(
        self,
        event_type,
        data=None,
        delta_time=None,
        qualia_state: Any | None = None,
        phase: str | None = None,
    ):
        """
        Publishes an event and records it as an EVA experience.
        """
        await super().publish(event_type, data, delta_time)

        experience_data = {
            "experience_id": f"eva_event_{event_type}_{int(asyncio.get_event_loop().time())}",
            "event_type": event_type,
            "data": data,
            "delta_time": delta_time,
            "timestamp": asyncio.get_event_loop().time(),
        }

        return self.eva_ingest_experience(
            intention_type="ARCHIVE_EVENT_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

    def eva_recall_event_experience(self, cue: str, phase: str | None = None) -> dict:
        result = self.eva_recall_experience(cue, phase)
        if "error" in result and "No bytecode found" in result["error"]:
            result["error"] = "No bytecode found for EVA event experience"
        return result


eva_event_bus = EVAEventBus()
