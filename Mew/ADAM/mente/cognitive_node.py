from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crisalida_lib.ADAM.mente.cognitive_impulses import (
        CognitiveImpulse,  # type: ignore
    )

# Runtime fallback to avoid import cycles at module import time
try:
    from crisalida_lib.ADAM.mente.cognitive_impulses import (
        CognitiveImpulse,  # type: ignore
    )
except Exception:  # pragma: no cover - runtime fallback in light environments
    CognitiveImpulse = Any  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CognitiveNode(ABC):
    """
    Abstract base class for cognitive nodes.

    Enhancements over legacy version:
    - Thread-safe internal state (RLock).
    - Async-friendly analyze wrapper (`analyze_async`) that runs sync implementations
      in an executor to avoid blocking event loops.
    - Hooks for post-analysis processing and EVA ingestion (best-effort).
    - Serialization helpers `to_dict` / `from_dict`.
    - Defensive numeric guards and passive decay behaviour.
    """

    DEFAULT_DECAY = 0.9
    DEFAULT_ENERGY_EPS = 1e-8

    def __init__(
        self,
        node_name: str,
        activation_threshold: float = 0.5,
        config: dict[str, Any] | None = None,
    ) -> None:
        # Public attributes (avoid leading underscores for pydantic/serialization)
        self.node_name: str = node_name
        self.activation_threshold: float = float(activation_threshold)
        self.activation_energy: float = 0.0

        # Diagnostics / bookkeeping
        self.last_updated_ts: float = time.time()
        self.total_impulses_emitted: int = 0
        self.cumulative_intensity: float = 0.0

        # Extensible config bag for subclasses / runtime tuning
        self.config: dict[str, Any] = dict(config or {})

        # Hooks called with signature hook(node, impulses: List[CognitiveImpulse], context: dict)
        self._hooks: list[Any] = []

        # Optional ingest callback (entity_id: str, payload: dict) -> Optional[str]
        self.ingest_callback = None

        # Thread-safety
        self._lock = RLock()

    # Public API preserved: subclasses must implement analyze (sync)
    @abstractmethod
    def analyze(self, data: Any) -> list[CognitiveImpulse]:
        """
        Synchronous analysis entrypoint.

        Subclasses may implement synchronous logic returning a list of CognitiveImpulse
        objects. For async-capable subclasses, prefer overriding `analyze_async`.
        """
        raise NotImplementedError

    # Async wrapper that will run synchronous analyze in executor if needed
    async def analyze_async(self, data: Any) -> list[CognitiveImpulse]:
        """
        Async-friendly wrapper. If the subclass provides a native coroutine `analyze`,
        it will be awaited. Otherwise `analyze` executes in the default executor.
        """
        analyze_fn = self.analyze
        if asyncio.iscoroutinefunction(analyze_fn):
            return await analyze_fn(data)  # type: ignore
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: analyze_fn(data))

    async def emit_impulses(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[CognitiveImpulse]:
        """
        High-level helper that runs analysis, updates activation energy, calls hooks,
        and attempts to persist a compact record via `ingest_callback` if provided.

        Returns the list of impulses generated (may be empty).
        """
        context = dict(context or {})
        try:
            impulses = await self.analyze_async(data) or []
        except Exception:
            logger.exception("analyze failed for node %s", self.node_name)
            impulses = []

        # Defensive normalization
        impulses = [imp for imp in impulses if imp is not None]

        # Update internal state safely
        with self._lock:
            num = len(impulses)
            total_intensity = sum(
                getattr(imp, "intensity", 0.0) or 0.0 for imp in impulses
            )
            self._update_activation_energy(
                num_impulses=num, total_intensity=total_intensity
            )
            self.total_impulses_emitted += num
            self.cumulative_intensity += float(total_intensity)
            self.last_updated_ts = time.time()

        # Call hooks (best-effort)
        for hook in list(self._hooks):
            try:
                hook(self, impulses, context)
            except Exception:
                logger.exception("hook raised in CognitiveNode.emit_impulses")

        # Best-effort persist to EVA if ingest_callback provided (non-blocking if coroutine)
        if self.ingest_callback:
            payload = {
                "node": self.node_name,
                "timestamp": time.time(),
                "num_impulses": len(impulses),
            }
            try:
                res = self.ingest_callback(payload)
                if asyncio.iscoroutine(res):
                    try:
                        asyncio.create_task(res)  # schedule background persist
                    except RuntimeError:
                        # No running loop — run to completion synchronously
                        asyncio.run(res)
            except Exception:
                logger.debug(
                    "ingest_callback failed for node %s (non-fatal)", self.node_name
                )

        return impulses

    def _update_activation_energy(
        self, num_impulses: int, total_intensity: float
    ) -> None:
        """
        Maintain activation energy: average intensity when impulses present; passive decay otherwise.
        """
        try:
            with self._lock:
                if num_impulses > 0 and total_intensity > 0.0:
                    avg = float(total_intensity) / float(max(1, num_impulses))
                    # optionally weight by recent cumulative intensity
                    weight = float(self.config.get("activation_weight", 0.8))
                    self.activation_energy = max(
                        0.0,
                        min(1.0, self.activation_energy * (1 - weight) + avg * weight),
                    )
                else:
                    # passive decay
                    decay = float(self.config.get("decay_factor", self.DEFAULT_DECAY))
                    self.activation_energy = float(self.activation_energy) * float(
                        decay
                    )
                # small epsilon guard
                if abs(self.activation_energy) < self.DEFAULT_ENERGY_EPS:
                    self.activation_energy = 0.0
        except Exception:
            logger.exception("_update_activation_energy failed")

    def is_activated(self) -> bool:
        """
        Return whether activation energy meets configured threshold.
        """
        with self._lock:
            return float(self.activation_energy) >= float(self.activation_threshold)

    def reset_activation(self) -> None:
        """
        Reset activation energy and bookkeeping counters.
        """
        with self._lock:
            self.activation_energy = 0.0
            self.total_impulses_emitted = 0
            self.cumulative_intensity = 0.0
            self.last_updated_ts = time.time()

    # Hook management
    def add_hook(self, hook_callable) -> None:
        """Register a hook called after each emit_impulses(node, impulses, context)."""
        with self._lock:
            if hook_callable not in self._hooks:
                self._hooks.append(hook_callable)

    def remove_hook(self, hook_callable) -> None:
        with self._lock:
            if hook_callable in self._hooks:
                self._hooks.remove(hook_callable)

    # Serialization helpers
    def to_dict(self) -> dict[str, Any]:
        """
        Stable serialization of the node's runtime state (not code).
        Subclasses may extend the dict with domain-specific fields.
        """
        with self._lock:
            return {
                "node_name": self.node_name,
                "activation_threshold": float(self.activation_threshold),
                "activation_energy": float(self.activation_energy),
                "last_updated_ts": float(self.last_updated_ts),
                "total_impulses_emitted": int(self.total_impulses_emitted),
                "cumulative_intensity": float(self.cumulative_intensity),
                "config": dict(self.config or {}),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CognitiveNode:
        """
        Hydrate a node from a dict. This returns a minimal instance of the base
        class when a concrete subclass is not specified. Consumers should prefer
        subclass-specific from_dict implementations.
        """
        node = cls(
            node_name=str(data.get("node_name", "unknown")),
            activation_threshold=float(data.get("activation_threshold", 0.5)),
        )
        node.activation_energy = float(data.get("activation_energy", 0.0))
        node.last_updated_ts = float(data.get("last_updated_ts", time.time()))
        node.total_impulses_emitted = int(data.get("total_impulses_emitted", 0))
        node.cumulative_intensity = float(data.get("cumulative_intensity", 0.0))
        node.config = dict(data.get("config", {}) or {})
        return node

    def __repr__(self) -> str:
        return f"<CognitiveNode {self.node_name} energy={self.activation_energy:.3f} threshold={self.activation_threshold:.3f}>"
