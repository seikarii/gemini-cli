from __future__ import annotations

"""EVAMemoryMixin - minimal, import-safe EVAMemory mixin.

This module provides a conservative mixin that delegates to EVAMemoryHelper
while avoiding import-time cycles using TYPE_CHECKING and runtime fallbacks.
"""

import logging  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

# Runtime import of helper (may fail at import time in some environments)
try:
    from crisalida_lib.EVA.eva_memory_helper import EVAMemoryHelper
except Exception:
    EVAMemoryHelper = Any  # type: ignore


if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import RealityBytecode  # type: ignore
    from crisalida_lib.EVA.typequalia import QualiaState  # type: ignore
    from crisalida_lib.EVA.types import QualiaSignature  # type: ignore
else:
    try:
        from crisalida_lib.EVA.types import QualiaSignature
    except Exception:

        @dataclass
        class QualiaSignature:  # type: ignore
            fingerprint: str
            vector: list[float]
            metadata: dict[str, Any] | None = None

    try:
        from crisalida_lib.EVA.typequalia import QualiaState
    except Exception:

        @dataclass
        class QualiaState:  # type: ignore
            emotional_valence: float = 0.5
            cognitive_complexity: float = 0.5
            consciousness_density: float = 0.5
            narrative_importance: float = 0.5
            energy_level: float = 1.0

    try:
        from crisalida_lib.EVA.core_types import RealityBytecode
    except Exception:

        @dataclass
        class RealityBytecode:  # type: ignore
            bytecode_id: str
            instructions: list[Any]
            qualia_state: QualiaState | None = None
            phase: str | None = None
            timestamp: float | None = None


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EVAMemoryMixin:
    """Lightweight EVAMemory mixin with safe runtime fallbacks."""

    def __init__(self, *args, **kwargs) -> None:
        # Helper may be Any at runtime
        try:
            self._eva_helper = EVAMemoryHelper(self)
        except Exception:
            self._eva_helper = None  # type: ignore

        self.eva_phase: str = getattr(self, "eva_phase", "default")
        self.eva_memory_store: dict[str, Any] = getattr(self, "eva_memory_store", {})
        self.eva_experience_store: dict[str, Any] = getattr(
            self, "eva_experience_store", {}
        )
        self.eva_phases: dict[str, Any] = getattr(self, "eva_phases", {})
        self._environment_hooks: list[Callable[[Any], None]] = getattr(
            self, "_environment_hooks", []
        )
        self.eva_runtime = getattr(self, "eva_runtime", None)

    def _eva_ensure_containers(self) -> None:
        if getattr(self, "_eva_helper", None) is None:
            return
        try:
            self._eva_helper.init_containers()  # type: ignore[attr-defined]
        except Exception:
            return

    def register_environment_hook(self, hook: Callable[[Any], None]) -> None:
        if getattr(self, "_eva_helper", None) is not None:
            try:
                self._eva_helper.register_environment_hook(hook)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        self._environment_hooks.append(hook)

    def set_memory_phase(self, phase: str) -> None:
        if getattr(self, "_eva_helper", None) is not None:
            try:
                self._eva_helper.set_memory_phase(phase)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        self.eva_phase = phase

    def get_memory_phase(self) -> str:
        if getattr(self, "_eva_helper", None) is not None:
            try:
                return self._eva_helper.get_memory_phase()  # type: ignore[attr-defined]
            except Exception:
                pass
        return getattr(self, "eva_phase", "default")

    def get_eva_api(self) -> dict[str, Any]:
        """Return a minimal EVA API mapping for compatibility.

        Subclasses (like EVABrainFallback) may extend this mapping.
        """
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
        }

    def _compile_intention_to_bytecode(
        self, intention: dict[str, Any]
    ) -> dict[str, Any]:
        helper = getattr(self, "_eva_helper", None)
        if helper is None:
            return {"instructions": [], "qualia_signature": None}
        return getattr(
            helper,
            "_compile_intention_to_bytecode",
            lambda i: {"instructions": [], "qualia_signature": None},
        )(intention)

    def eva_ingest_experience(
        self,
        intention_type: str,
        experience_data: dict[str, Any],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        helper = getattr(self, "_eva_helper", None)
        if helper is None:
            return ""
        return getattr(helper, "eva_ingest_experience", lambda *a, **k: "")(
            intention_type=intention_type,
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

    def eva_recall_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        helper = getattr(self, "_eva_helper", None)
        if helper is None:
            return {}
        return getattr(helper, "eva_recall_experience", lambda *a, **k: {})(cue, phase)

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        intention_type: str,
        experience_data: dict[str, Any],
        qualia_state: QualiaState,
    ) -> bool:
        helper = getattr(self, "_eva_helper", None)
        if helper is None:
            return False
        return getattr(helper, "add_experience_phase", lambda *a, **k: False)(
            experience_id=experience_id,
            phase=phase,
            intention_type=intention_type,
            experience_data=experience_data,
            qualia_state=qualia_state,
        )
