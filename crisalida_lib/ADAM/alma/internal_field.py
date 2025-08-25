"""
InternalField - Definitive, hardened implementation.

Highlights:
- Defensive numpy import and graceful fallback to pure-Python implementations.
- Robust QualiaState attribute support (handles several legacy shapes).
- Async/sync-aware EVA recording and recall (best-effort scheduling).
- Clear serialization, clamping, history bounding, and diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

# defensive numpy import (repo pattern)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    np = None  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )
    from crisalida_lib.EVA.typequalia import QualiaState  # type: ignore
else:
    QualiaState = Any  # runtime fallback
    EVAMemoryManager = Any  # runtime fallback

from crisalida_lib.EVA.typequalia import (
    QualiaState as _QualiaState,
)  # keep import reference

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class InternalField:
    """Represents an entity's internal mental / energetic field.

    Design goals:
    - Run with or without numpy.
    - Accept several QualiaState shapes (legacy differences).
    - Provide serialization and EVA persistence hooks.
    """

    def __init__(
        self,
        dimensions: tuple[int, int, int] = (16, 16, 16),
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.dimensions: tuple[int, int, int] = tuple(int(d) for d in dimensions)
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id

        # state representation: numpy array when available, otherwise nested lists
        if np is not None:
            self.state = np.zeros(self.dimensions, dtype=float)
        else:
            self.state = [
                [
                    [0.0 for _ in range(self.dimensions[2])]
                    for _ in range(self.dimensions[1])
                ]
                for _ in range(self.dimensions[0])
            ]

        # bookkeeping
        self.history_limit: int = 200
        self.state_history: list[float] = []
        self.last_update_ts: float = time.time()

        logger.info(
            "InternalField initialized: entity=%s dims=%s numpy=%s",
            self.entity_id,
            self.dimensions,
            bool(np),
        )

    # --- Serialization / Introspection ---------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return serializable snapshot of the internal field."""
        try:
            if np is not None and hasattr(self.state, "tolist"):
                state_list = self.state.tolist()
                mean = float(float(self.state.mean())) if np is not None else 0.0
                std = float(float(self.state.std())) if np is not None else 0.0
            else:
                state_list = self.state  # already nested lists
                flat = [v for x in self.state for y in x for v in y]
                mean = float(sum(flat) / max(1, len(flat)))
                std = float(
                    math.sqrt(sum((v - mean) ** 2 for v in flat) / max(1, len(flat)))
                )
            return {
                "entity_id": self.entity_id,
                "dimensions": self.dimensions,
                "state": state_list,
                "stats": {
                    "mean": mean,
                    "std": std,
                    "coherence": self.get_internal_coherence(),
                },
                "last_update": datetime.utcnow().isoformat(),
            }
        except Exception:
            logger.exception("get_state failed")
            return {
                "entity_id": self.entity_id,
                "dimensions": self.dimensions,
                "state": None,
            }

    def to_serializable(self) -> dict[str, Any]:
        """Alias for external callers."""
        return self.get_state()

    def load_serializable(self, data: dict[str, Any]) -> None:
        """Load a snapshot produced by get_state / to_serializable (best-effort)."""
        try:
            dims = data.get("dimensions", self.dimensions)
            if dims and len(dims) == 3:
                self.dimensions = tuple(int(d) for d in dims)
            raw_state = data.get("state")
            if raw_state is None:
                # keep existing
                return
            if np is not None:
                arr = np.array(raw_state, dtype=float)
                if arr.shape == self.dimensions:
                    self.state = arr
                else:
                    # attempt reshape or reinit
                    self.state = np.zeros(self.dimensions, dtype=float)
                    flat = arr.flatten()
                    it = iter(flat)
                    for x in range(self.dimensions[0]):
                        for y in range(self.dimensions[1]):
                            for z in range(self.dimensions[2]):
                                try:
                                    self.state[x, y, z] = float(next(it))
                                except StopIteration:
                                    break
            else:
                # expect nested lists; attempt to coerce lengths
                self.state = raw_state
        except Exception:
            logger.exception("load_serializable failed")

    # --- Basic operations ----------------------------------------------
    def _in_bounds(self, coords: tuple[int, int, int]) -> bool:
        if len(coords) != 3:
            return False
        return all(0 <= int(coords[i]) < self.dimensions[i] for i in range(3))

    def update_state(self, coords: tuple[int, int, int], value: float) -> None:
        """Set a single voxel value (clamped 0..1)."""
        try:
            if not self._in_bounds(coords):
                logger.warning("update_state coords out of bounds: %s", coords)
                return
            v = float(value)
            v = max(0.0, min(1.0, v))
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
            if np is not None and hasattr(self.state, "__setitem__"):
                self.state[x, y, z] = v
            else:
                self.state[x][y][z] = v
            self.last_update_ts = time.time()
        except Exception:
            logger.exception("update_state failed for coords=%s", coords)

    def get_state_at(self, coords: tuple[int, int, int]) -> float | None:
        """Read a single voxel (or None if out of bounds)."""
        try:
            if not self._in_bounds(coords):
                return None
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
            if np is not None:
                return float(self.state[x, y, z])
            else:
                return float(self.state[x][y][z])
        except Exception:
            logger.exception("get_state_at failed for coords=%s", coords)
            return None

    # --- Higher-level generative / influence methods -------------------
    def apply_qualia_influence(
        self, qualia_state: QualiaState, intensity: float = 0.1
    ) -> None:
        """
        Apply influence coming from a QualiaState.

        Handles different legacy QualiaState shapes by probing attributes.
        Strategy: determine a center in the field from valence/arousal/focus and
        apply a gaussian-like falloff update (or a simple linear falloff without numpy).
        """
        try:
            # Normalize / probe likely attribute names with sensible defaults
            valence = (
                getattr(qualia_state, "emotional", None)
                or getattr(qualia_state, "emotional_valence", None)
                or getattr(qualia_state, "valence", None)
                or 0.0
            )
            arousal = (
                getattr(qualia_state, "arousal", None)
                or getattr(qualia_state, "activation", None)
                or getattr(qualia_state, "energy", None)
                or 0.5
            )
            focus = (
                getattr(qualia_state, "cognitive_focus", None)
                or getattr(qualia_state, "attention", None)
                or getattr(qualia_state, "complexity", None)
                or 0.5
            )
            # Map to coordinates inside field
            cx = int(self.dimensions[0] * (0.5 + float(valence) / 2.0))
            cy = int(self.dimensions[1] * float(arousal))
            cz = int(self.dimensions[2] * float(focus))
            cx = max(0, min(self.dimensions[0] - 1, cx))
            cy = max(0, min(self.dimensions[1] - 1, cy))
            cz = max(0, min(self.dimensions[2] - 1, cz))

            sigma = max(1.0, sum(self.dimensions) / 12.0)  # heuristic spread
            if np is not None:
                # efficient vectorized gaussian influence
                xs = np.arange(self.dimensions[0])[:, None, None]
                ys = np.arange(self.dimensions[1])[None, :, None]
                zs = np.arange(self.dimensions[2])[None, None, :]
                dx = xs - cx
                dy = ys - cy
                dz = zs - cz
                dist2 = dx * dx + dy * dy + dz * dz
                influence = intensity * np.exp(-dist2 / (2.0 * sigma * sigma))
                self.state = np.clip(self.state + influence, 0.0, 1.0)
            else:
                # pure-python fallback
                for x in range(self.dimensions[0]):
                    for y in range(self.dimensions[1]):
                        for z in range(self.dimensions[2]):
                            dx = x - cx
                            dy = y - cy
                            dz = z - cz
                            dist2 = dx * dx + dy * dy + dz * dz
                            inf = intensity * math.exp(-dist2 / (2.0 * sigma * sigma))
                            self.state[x][y][z] = max(
                                0.0, min(1.0, self.state[x][y][z] + inf)
                            )
            # bookkeeping
            coherence = self.get_internal_coherence()
            self.state_history.append(coherence)
            if len(self.state_history) > self.history_limit:
                self.state_history = self.state_history[-self.history_limit :]
            self.last_update_ts = time.time()
        except Exception:
            logger.exception("apply_qualia_influence failed")

    def generate_internal_pattern(self, pattern_type: str = "random") -> None:
        """Create a quick pattern: random, coherent, radial, or silence."""
        try:
            if pattern_type == "random":
                if np is not None:
                    self.state = np.random.random(self.dimensions).astype(float)
                else:
                    for x in range(self.dimensions[0]):
                        for y in range(self.dimensions[1]):
                            for z in range(self.dimensions[2]):
                                self.state[x][y][z] = random.random()
            elif pattern_type == "coherent":
                base = 0.7
                jitter = 0.05
                if np is not None:
                    self.state = np.clip(
                        base + np.random.random(self.dimensions) * jitter, 0.0, 1.0
                    )
                else:
                    for x in range(self.dimensions[0]):
                        for y in range(self.dimensions[1]):
                            for z in range(self.dimensions[2]):
                                self.state[x][y][z] = max(
                                    0.0, min(1.0, base + random.random() * jitter)
                                )
            elif pattern_type == "radial":
                cx = self.dimensions[0] // 2
                cy = self.dimensions[1] // 2
                cz = self.dimensions[2] // 2
                maxd = math.sqrt(sum((d / 2.0) ** 2 for d in self.dimensions))
                if np is not None:
                    xs = np.arange(self.dimensions[0])[:, None, None]
                    ys = np.arange(self.dimensions[1])[None, :, None]
                    zs = np.arange(self.dimensions[2])[None, None, :]
                    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2)
                    self.state = np.clip(1.0 - (dist / maxd), 0.0, 1.0)
                else:
                    for x in range(self.dimensions[0]):
                        for y in range(self.dimensions[1]):
                            for z in range(self.dimensions[2]):
                                d = math.sqrt(
                                    (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                                )
                                self.state[x][y][z] = max(
                                    0.0, min(1.0, 1.0 - (d / maxd))
                                )
            elif pattern_type == "silence":
                if np is not None:
                    self.state = np.zeros(self.dimensions, dtype=float)
                else:
                    for x in range(self.dimensions[0]):
                        for y in range(self.dimensions[1]):
                            for z in range(self.dimensions[2]):
                                self.state[x][y][z] = 0.0
            else:
                logger.debug("Unknown pattern_type=%s, generating random", pattern_type)
                self.generate_internal_pattern("random")
            self.last_update_ts = time.time()
            logger.info(
                "Generated internal pattern '%s' for %s", pattern_type, self.entity_id
            )
        except Exception:
            logger.exception("generate_internal_pattern failed")

    def get_internal_coherence(self) -> float:
        """Return 0..1 coherence metric: 1 - normalized stddev (clamped)."""
        try:
            if np is not None:
                std = float(self.state.std())
                # std of uniform 0..1 ~ 0.2887 ; normalize by expected max
                normalized = std / 0.6
            else:
                flat = [v for x in self.state for y in x for v in y]
                mean = sum(flat) / max(1, len(flat))
                var = sum((v - mean) ** 2 for v in flat) / max(1, len(flat))
                std = math.sqrt(var)
                normalized = std / 0.6
            coherence = max(0.0, min(1.0, 1.0 - normalized))
            return float(coherence)
        except Exception:
            logger.exception("get_internal_coherence failed")
            return 0.0

    # --- EVA persistence helpers -------------------------------------
    def record_internal_field_experience(
        self,
        experience_id: str | None = None,
        internal_field_state: dict[str, Any] | None = None,
        qualia_state: QualiaState | None = None,
    ) -> None:
        """Record current internal field snapshot in EVA (best-effort, sync/async aware)."""
        if not self.eva_manager:
            logger.debug("No EVA manager attached; skipping internal field recording")
            return
        try:
            snapshot = internal_field_state or self.to_serializable()
            experience_id = (
                experience_id
                or f"internal_field_{self.entity_id}_{int(time.time())}_{hash(str(snapshot)) & 0xFFFF}"
            )
            qualia_state = (
                qualia_state
                or _QualiaState(
                    emotional=0.5,
                    complexity=0.5,
                    consciousness=0.5,
                    importance=0.5,
                    energy=0.5,
                )
                if "QualiaState" in globals()
                else None
            )

            rec = getattr(self.eva_manager, "record_experience", None)
            if not rec:
                logger.debug("EVAMemoryManager.record_experience not available")
                return

            result = rec(
                entity_id=self.entity_id,
                event_type="internal_field_experience",
                data=snapshot,
                qualia_state=qualia_state,
                experience_id=experience_id,
            )
            # If coroutine, schedule best-effort
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(result)
                    else:
                        loop.run_until_complete(result)
                    logger.debug("Scheduled async EVA record task: %s", experience_id)
                except Exception:
                    logger.debug(
                        "Async scheduling failed for EVA record", exc_info=True
                    )
            else:
                logger.debug(
                    "Recorded internal field experience sync: %s", experience_id
                )
        except Exception:
            logger.exception("record_internal_field_experience failed")

    def recall_internal_field_experience(
        self, experience_id: str
    ) -> dict[str, Any] | None:
        """Recall an experience from EVA (best-effort)."""
        if not self.eva_manager:
            logger.debug(
                "No EVA manager attached; cannot recall internal field experience %s",
                experience_id,
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not recall:
                logger.debug("EVAMemoryManager.recall_experience not available")
                return None
            result = recall(entity_id=self.entity_id, experience_id=experience_id)
            # If coroutine, attempt to run (best-effort)
            if hasattr(result, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # cannot block; log and return None
                        logger.debug(
                            "EVA recall returned coroutine while loop running; returning None"
                        )
                        return None
                    else:
                        return loop.run_until_complete(result)
                except Exception:
                    logger.exception("Async recall failed")
                    return None
            return result
        except Exception:
            logger.exception("recall_internal_field_experience failed")
            return None
