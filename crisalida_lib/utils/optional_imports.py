"""Helpers for optional imports and safe fallbacks used across the codebase."""
from typing import Any, Tuple


def optional_import(module_name: str) -> Any:
    """Try to import module by name, return module or None on failure."""
    try:
        __import__(module_name)
        return __import__(module_name)
    except Exception:
        return None


def import_or_fallback(module_name: str, attr: str, fallback: Any) -> Any:
    """Return module.attr if present, otherwise fallback.

    Usage: X = import_or_fallback('numpy', 'array', list)
    """
    mod = optional_import(module_name)
    if mod is None:
        return fallback
    return getattr(mod, attr, fallback)
