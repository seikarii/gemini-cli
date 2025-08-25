"""Utilities to serialize EDEN objects for HTTP APIs.

Small, defensive helpers to convert LivingSymbol and simple EDEN structures to
JSON-friendly dicts. Keep the functions idempotent and resilient when optional
sub-attributes are missing.
"""
from typing import Any, Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover - degrade gracefully if numpy missing
    np = None  # type: ignore


def _to_floats(sequence: Any) -> List[float]:
    try:
        if sequence is None:
            return []
        # numpy arrays
        if hasattr(sequence, "tolist"):
            lst = sequence.tolist()
            return [float(x) for x in lst]
        # plain iterables
        return [float(x) for x in list(sequence)]
    except Exception:
        try:
            # fallback: coerce single numeric
            return [float(sequence)]
        except Exception:
            return []


def serialize_symbol(symbol: Any) -> Dict[str, Any]:
    """Return a JSON-friendly dict for a LivingSymbol-like object.

    Fields: id, sigil, position (list[float]), health, dynamic_signature_len (optional)
    """
    if symbol is None:
        return {}

    # id
    entity_id = getattr(symbol, "entity_id", None) or getattr(symbol, "id", None) or "unknown"

    # sigil: common alias set by QualiaEngine or by divine metadata
    sigil = getattr(symbol, "sigil", None)
    if not sigil:
        divine = getattr(symbol, "divine", None)
        if divine is not None:
            sigil = getattr(divine, "metadata", {}).get("sigil") if hasattr(getattr(divine, "metadata", {}), "get") else None

    # position
    pos = None
    kinetic = getattr(symbol, "kinetic_state", None)
    if kinetic is not None:
        pos = getattr(kinetic, "position", None)
    if pos is None:
        pos = getattr(symbol, "position", None)

    position = _to_floats(pos)

    # health
    health = getattr(symbol, "health", None)

    out = {
        "id": str(entity_id),
        "sigil": sigil,
        "position": position,
        "health": health,
    }

    # optional: dynamic signature length if available (avoid serializing full arrays)
    try:
        dyn = getattr(symbol, "get_dynamic_signature", None)
        if callable(dyn):
            sig = dyn()
            if hasattr(sig, "shape"):
                out["dynamic_signature_len"] = int(getattr(sig, "shape")[0])
            elif isinstance(sig, (list, tuple)):
                out["dynamic_signature_len"] = len(sig)
    except Exception:
        pass

    return out
