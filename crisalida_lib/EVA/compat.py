"""Compatibility shims for EVA public symbols used at import time.

This file exposes minimal placeholders when the full implementations are not
importable at module import time. These are intentionally small and should be
replaced by the real classes during a real refactor.
"""

from typing import Any


class EVAMemoryMixin:  # pragma: no cover - shim
    """Tiny placeholder to satisfy module-level imports.

    Real implementation lives in `eva_memory_mixin.py` and may have a different
    internal structure. This shim only provides a safe surface for import-time
    checks and should not be relied upon at runtime.
    """

    def _init_eva_memory(self, *args: Any, **kwargs: Any) -> None:
        # typed annotation to satisfy linters and mypy for the placeholder
        self._eva_config: dict[str, Any] = {}

    def add_experience_phase(self, *args: Any, **kwargs: Any) -> bool:
        return False

    # The legacy `eva_delegate` surface is deprecated. Consumers should call
    # `eva_ingest_experience` / `eva_recall_experience` on the mixin/helper.
    @property
    def eva_delegate(self) -> Any:  # pragma: no cover - compatibility
        def _deprecated(*args, **kwargs):
            raise RuntimeError(
                "eva_delegate is deprecated; call eva_ingest_experience or eva_recall_experience on the EVAMemoryMixin instead"
            )

        return _deprecated

    @eva_delegate.setter
    def eva_delegate(self, value: Any) -> None:  # pragma: no cover - compatibility
        # Accept setting for import-time compatibility but warn at runtime via the getter.
        self._eva_delegate = value
