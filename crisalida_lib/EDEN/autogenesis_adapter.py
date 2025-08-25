"""AutoGenesisEngine adapter shim.

Provides a dynamic import for richer AutoGenesisEngine implementations (e.g., from
divine_language or ADAM) and a lightweight fallback that is safe to import at
runtime. This enables integration with ADAM without changing ADAM internals.
"""

from typing import Any


def get_autogenesis_engine_class() -> type[Any]:
    """Return the best-available AutoGenesisEngine class.

    Tries optional locations and falls back to a minimal local implementation.
    """
    try:
        # try known richer implementation locations
        from crisalida_lib.EARTH.auto_genesis_engine import (
            AutoGenesisEngine as _AGE,  # type: ignore
        )

        return _AGE
    except Exception:
        pass

    try:
        from crisalida_lib.EARTH.auto_genesis_engine import (
            AutoGenesisEngine as _AGE,  # type: ignore
        )

        return _AGE
    except Exception:
        pass

    # Fallback minimal implementation
    class AutoGenesisEngineLite:
        def __init__(self, *args, **kwargs):
            self.status = "lite-initialized"

        def generate(self, seed: Any | None = None) -> dict[str, Any]:
            # produce a trivial genesis artifact
            return {"seed": seed, "artifact": "auto-gen-lite"}

        def get_status(self) -> str:
            return self.status

    return AutoGenesisEngineLite


def create_autogenesis_engine(*args, **kwargs) -> Any:
    cls = get_autogenesis_engine_class()
    return cls(*args, **kwargs)
