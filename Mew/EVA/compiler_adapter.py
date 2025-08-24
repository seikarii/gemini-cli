"""Simple compiler adapter used during fusion.

Delegates consistently to the in-repo EVA bytecode generator `generate_bytecode`.
Normalizes returned instructions to dicts with an 'opcode' key to satisfy
consumers and tests.
"""

import logging
from collections.abc import Callable
from typing import Any


def _normalize_instr(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        # ensure opcode/op aliases
        if "opcode" not in item and "op" in item:
            item = {**item, "opcode": item.get("op")}
        if "op" not in item and "opcode" in item:
            item = {**item, "op": item.get("opcode")}
        return item
    # try object shape
    try:
        opcode = getattr(item, "opcode", None) or getattr(item, "op", None)
        operands = getattr(item, "operands", None) or getattr(item, "args", None) or []
        meta = getattr(item, "metadata", None) or getattr(item, "meta", None) or {}
        return {
            "opcode": str(opcode),
            "op": str(opcode),
            "operands": list(operands) if operands is not None else [],
            "metadata": meta or {},
        }
    except Exception:
        return {"opcode": str(item), "op": str(item), "operands": [], "metadata": {}}


"""Compiler adapter: unify available compiler backends to a simple sync API.

This module will try to find a suitable compiler implementation in the
repository (EVA/quantum_compiler) or in `divine_language` and expose a
`compile_matrix` function that returns a list of instructions.
"""

# prefer local, lightweight generator when available
_local_generate: Callable[..., list[Any]] | None = None
try:
    from crisalida_lib.EVA.bytecode_generator import (
        generate_bytecode as _local_generate,
    )
except Exception:
    _local_generate = None

logger = logging.getLogger(__name__)


class _NullCompiler:
    def compile_matrix(self, matrix: Any) -> list[Any]:
        return []


def _find_eva_compiler():
    try:
        from crisalida_lib.EVA.quantum_compiler import QuantumCompilerV7

        return QuantumCompilerV7()
    except Exception:
        return None


def _find_divine_compiler():
    try:
        from divine_language.compiler.quantum_compiler_v7 import (
            QuantumCompilerV7 as QV7,
        )

        # QV7 in divine_language is async; return a simple wrapper
        return QV7
    except Exception:
        return None


class CompilerAdapter:
    def __init__(self):
        self._backend = _find_eva_compiler()
        self._divine_backend = _find_divine_compiler()
        if self._backend is None and self._divine_backend is None:
            self._backend = _NullCompiler()

    def compile_matrix(self, matrix: Any) -> list[Any]:
        # Prefer in-repo EVA compiler instance
        if self._backend is not None:
            try:
                res = self._backend.compile_matrix(matrix)
                if res:
                    return res
            except Exception as e:
                logger.debug("EVA compiler failed: %s", e)

        # Try divine_language async compiler if present (run sync by awaiting in loop)
        if self._divine_backend is not None:
            try:
                # instantiate
                qv = self._divine_backend(None)
                # compile_matrix is async in v7; try calling sync method if exists
                cm = getattr(qv, "compile_matrix", None)
                if cm is None:
                    return []
                # If coroutinefunction, run via asyncio loop
                import asyncio

                if asyncio.iscoroutinefunction(cm):
                    # Prefer asyncio.run when possible; if an event loop is already
                    # running, create a new loop to avoid DeprecationWarning or
                    # RuntimeError. This keeps behavior consistent both in sync
                    # test runs and async environments.
                    try:
                        return asyncio.run(cm(matrix))
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        try:
                            return loop.run_until_complete(cm(matrix))
                        finally:
                            try:
                                loop.close()
                            except Exception:
                                pass
                else:
                    return cm(matrix)
            except Exception as e:
                logger.debug("Divine compiler failed: %s", e)

                # As a last resort, produce a simple, local semantic compilation
                try:
                    # try a slightly richer local generator first
                    if _local_generate is not None:
                        try:
                            lb = _local_generate(matrix)
                            if lb:
                                return lb
                        except Exception:
                            pass
                    simple = _simple_compile_matrix(matrix)
                    if simple:
                        return simple
                except Exception:
                    pass

                return []

        # Ensure a list is always returned on all control paths
        # If no backends were present, try the local generator directly as a final fallback
        try:
            if _local_generate is not None:
                lb = _local_generate(matrix)
                if lb:
                    return lb
        except Exception:
            pass
        try:
            simple = _simple_compile_matrix(matrix)
            if simple:
                return simple
        except Exception:
            pass
        return []

    # end compile_matrix


_adapter = CompilerAdapter()


def _normalize_compiler_result(res: Any) -> list[Any]:
    """Normalize various compiler return types to a plain list of instruction dicts."""
    if res is None:
        return []

    def _item_to_dict(item: Any) -> Any:
        # if it's already a dict, keep
        if isinstance(item, dict):
            return item
        # objects with opcode/operands/metadata
        try:
            opcode = getattr(item, "opcode", None)
            operands = getattr(item, "operands", None)
            meta = getattr(item, "metadata", None) or getattr(item, "meta", None)
            if opcode is not None:
                # opcode may be Enum or object; coerce to string
                try:
                    opc = opcode.name
                except Exception:
                    try:
                        opc = str(opcode)
                    except Exception:
                        opc = repr(opcode)
                return {
                    "opcode": opc,
                    "operands": list(operands) if operands is not None else [],
                    "metadata": meta or {},
                }
        except Exception:
            pass
        # fallback: keep as-is
        return item

    if isinstance(res, list):
        return [_item_to_dict(i) for i in res]
    try:
        if hasattr(res, "bytecode"):
            bc = res.bytecode
            if hasattr(bc, "instructions"):
                return [_item_to_dict(i) for i in list(bc.instructions or [])]
            if isinstance(bc, list):
                return [_item_to_dict(i) for i in bc]
    except Exception:
        pass
    try:
        if hasattr(res, "instructions"):
            return [_item_to_dict(i) for i in list(res.instructions or [])]
    except Exception:
        pass
    try:
        return [_item_to_dict(i) for i in list(res)]
    except Exception:
        return []


def compile_matrix(matrix: Any) -> list[Any]:
    raw = _adapter.compile_matrix(matrix)
    return _normalize_compiler_result(raw)


def compile_intention(intention: dict[str, Any]) -> list[Any]:
    """Compatibility wrapper: compile an intention dict into instructions.

    This API is used by EDEN adapters and living symbol fallbacks.
    """
    try:
        payload = (
            intention.get("experience")
            or intention.get("qualia")
            or intention.get("payload")
        )
        if isinstance(payload, (list, tuple)):
            return compile_matrix(payload)
        single = [[payload]] if payload is not None else [[]]
        return compile_matrix(single)
    except Exception:
        return []


def _simple_compile_matrix(matrix: Any) -> list[Any]:
    """A conservative, local compiler fallback that generates Q_CREATE and
    Q_RESONATE instructions from a 2D matrix-like input.

    This mirrors the lightweight semantic compilation used in the living
    symbol V8 fallback and in `InternalVMShim`.
    """
    rows = 0
    cols = 0
    try:
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
    except Exception:
        # not matrix-like
        return []

    instructions: list[Any] = []
    coherence_val = 0.5
    try:
        coherence_val = float(getattr(matrix, "coherence", coherence_val))
    except Exception:
        pass

    # Q_CREATE
    for i in range(rows):
        for j in range(cols):
            try:
                sigil = matrix[i][j]
            except Exception:
                sigil = None
            # Special semantic sigils trigger different opcodes
            try:
                s = str(sigil) if sigil is not None else ""
            except Exception:
                s = ""

            evolution_sigils = {"Δ", "Χ", "∇", "⊗", "Ş"}
            snow_sigils = {"❄", "✻", "✺"}

            if s in evolution_sigils:
                instructions.append(
                    {
                        "opcode": "Q_TRANSFORM",
                        "operands": [sigil, f"cell_{i}_{j}"],
                        "metadata": {
                            "position": (i, j),
                            "matrix_coherence": coherence_val,
                        },
                    }
                )
            elif s in snow_sigils:
                instructions.append(
                    {
                        "opcode": "Q_CRYSTALLIZE",
                        "operands": [sigil, f"cell_{i}_{j}"],
                        "metadata": {
                            "position": (i, j),
                            "matrix_coherence": coherence_val,
                        },
                    }
                )
            else:
                instructions.append(
                    {
                        "opcode": "Q_CREATE",
                        "operands": [sigil, f"cell_{i}_{j}"],
                        "metadata": {
                            "position": (i, j),
                            "matrix_coherence": coherence_val,
                        },
                    }
                )

    # Q_RESONATE (right, down, diag)
    for i in range(rows):
        for j in range(cols):
            for di, dj in [(0, 1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if ni < rows and nj < cols:
                    instructions.append(
                        {
                            "opcode": "Q_RESONATE",
                            "operands": [f"cell_{i}_{j}", f"cell_{ni}_{nj}"],
                            "metadata": {"resonance_type": "harmonic", "strength": 0.8},
                        }
                    )

    return instructions
