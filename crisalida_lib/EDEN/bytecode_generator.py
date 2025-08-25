"""EDEN-level adapter for bytecode generation.

Delegates to the richer generator implemented in `crisalida_lib.EVA.bytecode_generator` when available.
This keeps EDEN-side callers stable while centralizing generator logic.
"""

from typing import Any


def compile_intention_to_bytecode(intention: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile an intention into a list of instruction dicts.

    Prefer `crisalida_lib.EVA.bytecode_generator.generate_bytecode` when present.
    The function tries to extract a matrix-like payload from the intention and
    delegates; if that fails, it returns a small normalized instruction stream.
    """
    try:
        # lazy import to avoid import cycles at module-import time
        from crisalida_lib.EVA.bytecode_generator import generate_bytecode

        if isinstance(intention, dict):
            payload = (
                intention.get("experience")
                or intention.get("qualia")
                or intention.get("payload")
            )
            entity = intention.get("entity_id") or intention.get("phase")
        else:
            payload = intention
            entity = None

        # If payload looks like a matrix (list of lists), pass directly.
        if isinstance(payload, (list, tuple)):
            matrix = payload
        else:
            # wrap non-matrix payload as a single-cell matrix with a string token
            matrix = [[payload]] if payload is not None else [[]]

        instrs = generate_bytecode(matrix, entity=entity)
        # Ensure produced instructions follow the expected shape (opcode key)
        normalized = []
        for i in instrs or []:
            if isinstance(i, dict):
                # ensure both 'opcode' and legacy 'op' exist
                if "opcode" not in i and "op" in i:
                    i = {**i, "opcode": i.get("op")}
                if "op" not in i and "opcode" in i:
                    i = {**i, "op": i.get("opcode")}
                normalized.append(i)
        # if generator returned nothing, provide a conservative fallback preserving legacy key 'op'
        if not normalized:
            try:
                itype = (
                    intention.get("intention_type", "NOOP")
                    if isinstance(intention, dict)
                    else "NOOP"
                )
                payload = (
                    intention.get("experience") if isinstance(intention, dict) else {}
                )
                return [
                    {
                        "opcode": str(itype),
                        "op": str(itype),
                        "operands": [payload],
                        "metadata": {},
                    }
                ]
            except Exception:
                return []
        return normalized
    except Exception:
        # Conservative fallback: small normalized instruction
        try:
            itype = (
                intention.get("intention_type", "NOOP")
                if isinstance(intention, dict)
                else "NOOP"
            )
            payload = intention.get("experience") if isinstance(intention, dict) else {}
            return [{"opcode": str(itype), "operands": [payload], "metadata": {}}]
        except Exception:
            return []

        def safe_compile_intention(
            runtime: Any, intention: dict[str, Any]
        ) -> list[dict[str, Any]]:
            """Try to compile via runtime.divine_compiler.compile_intention with guards.

            If runtime or the compiler is not available, fallback to compile_intention_to_bytecode.
            """
            bytecode = []
            if runtime is not None:
                _dc = getattr(runtime, "divine_compiler", None)
                compile_fn = (
                    getattr(_dc, "compile_intention", None) if _dc is not None else None
                )
                if callable(compile_fn):
                    try:
                        bytecode = compile_fn(intention)
                    except Exception:
                        bytecode = []
            if not bytecode:
                try:
                    return compile_intention_to_bytecode(intention)
                except Exception:
                    return []
            return bytecode
