"""ADAM integration contracts and lightweight adapters (definitive)

This module defines a minimal Protocol used by EDEN/other adapters to interact
with ADAM without importing heavy internals. It also provides a small,
well-documented in-memory adapter useful for tests and lightweight runtimes.

Design goals:
- Keep the public Protocol stable: ADAMIntegrationPoint (sync) is the canonical API.
- Provide an async-friendly variant AsyncADAMIntegrationPoint for codepaths that
  prefer coroutines.
- Offer a SimpleInMemoryADAMAdapter implementing both sync and async helpers,
  useful for unit tests and examples.
- Defensive, typed, and minimal dependencies so it can be imported in CI/test
  environments without pulling the full ADAM stack.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ADAMIntegrationError(RuntimeError):
    """Raised for integration-related failures."""


@runtime_checkable
class ADAMIntegrationPoint(Protocol):
    """Protocol describing the minimal methods EDEN expects from ADAM.

    Implementations living in ADAM should satisfy this protocol. Tests and
    adapters can provide lightweight mocks that match these signatures.

    Synchronous API:
      - register_living_symbol(signature) -> id
      - get_memory_snapshot(entity_id) -> snapshot|None

    The concrete runtime may implement additional helpers (async variants,
    richer snapshots). Consumers should use hasattr()/inspect to adapt.
    """

    def register_living_symbol(self, signature: dict[str, Any]) -> str:
        """Register a living symbol and return its id."""

    def get_memory_snapshot(self, entity_id: str) -> dict[str, Any] | None:
        """Return a memory snapshot for an entity or None if missing."""


@runtime_checkable
class AsyncADAMIntegrationPoint(Protocol):
    """Optional coroutine-friendly integration contract (async counterparts)."""

    async def register_living_symbol_async(self, signature: dict[str, Any]) -> str:
        """Async registration variant."""

    async def get_memory_snapshot_async(self, entity_id: str) -> dict[str, Any] | None:
        """Async snapshot retrieval variant."""


@dataclass
class MemorySnapshot:
    """Structured, minimal snapshot returned by get_memory_snapshot."""

    entity_id: str
    created_at: float = field(default_factory=time.time)
    signature: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "created_at": float(self.created_at),
            "signature": dict(self.signature),
            "annotations": dict(self.annotations),
        }


class SimpleInMemoryADAMAdapter:
    """
    Lightweight in-memory implementation of ADAMIntegrationPoint.

    Usage:
      adapter = SimpleInMemoryADAMAdapter()
      eid = adapter.register_living_symbol({"sigil": "Φ", "meta": {...}})
      snap = adapter.get_memory_snapshot(eid)

    This adapter also exposes async helpers that simply delegate to sync
    implementations and are safe to await in tests.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemorySnapshot] = {}
        logger.info("SimpleInMemoryADAMAdapter initialized")

    def register_living_symbol(self, signature: dict[str, Any]) -> str:
        """Register a living symbol; returns stable id string."""
        if not isinstance(signature, dict):
            raise ADAMIntegrationError("signature must be a dict")
        eid = f"adam_{uuid4().hex[:12]}"
        snapshot = MemorySnapshot(entity_id=eid, signature=dict(signature))
        self._store[eid] = snapshot
        logger.debug(
            "Registered living symbol: %s (keys=%s)", eid, list(signature.keys())
        )
        return eid

    def get_memory_snapshot(self, entity_id: str) -> dict[str, Any] | None:
        """Return a plain-dict snapshot or None."""
        snap = self._store.get(entity_id)
        if not snap:
            logger.debug("get_memory_snapshot: missing entity_id=%s", entity_id)
            return None
        return snap.to_dict()

    # --- Async convenience wrappers ---

    async def register_living_symbol_async(self, signature: dict[str, Any]) -> str:
        """Async wrapper for register_living_symbol (non-blocking)."""
        # allow cooperative scheduling in async contexts
        await asyncio.sleep(0)
        return self.register_living_symbol(signature)

    async def get_memory_snapshot_async(self, entity_id: str) -> dict[str, Any] | None:
        """Async wrapper for get_memory_snapshot (non-blocking)."""
        await asyncio.sleep(0)
        return self.get_memory_snapshot(entity_id)

    # --- Utilities useful for tests / debugging ---

    def clear(self) -> None:
        """Remove all stored snapshots (test helper)."""
        self._store.clear()
        logger.debug("SimpleInMemoryADAMAdapter store cleared")

    def snapshot_count(self) -> int:
        return len(self._store)


__all__ = [
    "ADAMIntegrationPoint",
    "AsyncADAMIntegrationPoint",
    "SimpleInMemoryADAMAdapter",
    "MemorySnapshot",
    "ADAMIntegrationError",
]
