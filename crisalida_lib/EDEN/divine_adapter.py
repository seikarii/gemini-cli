"""Adapter that exposes a LivingSymbol class from divine_language when available.

This module performs a safe, dynamic import and provides a lightweight
fallback implementation when the rich implementation is not present.
"""

from typing import Any, cast

"""Small adapter layer to expose a conservative LivingSymbol interface inside EDEN.

This module provides a minimal shim for living symbol behavior used by EDEN
so we can port functionality from divine_language without touching ADAM internals.
"""


class LivingSymbolAdapter:
    """Conservative runtime representation used by EDEN tests.

    The implementation is intentionally minimal: stores an id, a small state
    dict and exposes a `to_runtime` method expected by some EDEN hooks.
    """

    def __init__(
        self, symbol_id: str, initial_state: dict[str, Any] | None = None
    ) -> None:
        self.symbol_id = symbol_id
        self.state = dict(initial_state or {})

    def get_dynamic_signature(self) -> dict[str, Any]:
        return {"id": self.symbol_id, "state": dict(self.state)}

    def apply_force(self, force: tuple[float, float, float]) -> None:
        # simple accumulator for tests
        v = self.state.get("velocity", (0.0, 0.0, 0.0))
        self.state["velocity"] = (v[0] + force[0], v[1] + force[1], v[2] + force[2])

    def perceive_local_qualia(self, qualia: dict[str, Any]) -> None:
        # record last qualia for debugging/tests
        self.state["last_qualia"] = qualia


"""Adapter to expose LivingSymbolV8 when available, with safe fallback.

This module provides `get_living_symbol_class()` which returns the rich
`LivingSymbolV8` if `divine_language` is importable, otherwise returns a
lightweight local `LivingSymbolLite` class.
"""


try:
    from divine_language.livingsimbolv8 import LivingSymbolV8  # type: ignore

    RICH_AVAILABLE = True
except Exception:
    LivingSymbolV8 = None  # type: ignore
    RICH_AVAILABLE = False


class LivingSymbolLite:
    def __init__(
        self,
        entity_id: str,
        unified_field: Any = None,
        initial_intention_matrix: list[list[str]] | None = None,
    ):
        self.entity_id = entity_id
        self.unified_field = unified_field
        self.intention_matrix = initial_intention_matrix
        self.last_update = 0.0
        # lightweight attributes
        self.signature = [0.0] * 8
        self.position = (0.0, 0.0, 0.0)
        # internal VM shim (optional)
        self.internal_vm: Any | None = None
        try:
            from crisalida_lib.EDEN.intention import InternalVMShim

            self.internal_vm = InternalVMShim()
        except Exception:
            self.internal_vm = None

    def update(self, dt: float = 0.1, neighbors: list[Any] | None = None) -> None:
        self.last_update += float(dt)

    def apply_force(self, force: tuple[float, float, float]) -> None:
        """Apply a simple force by accumulating into position and velocity.

        This method mirrors the small contract used elsewhere in EDEN and
        provides parity with the `LivingSymbolAdapter` above.
        """
        v = getattr(self, "position", (0.0, 0.0, 0.0))
        try:
            self.position = (
                v[0] + float(force[0]),
                v[1] + float(force[1]),
                v[2] + float(force[2]),
            )
        except Exception:
            # best-effort: ignore malformed forces in tests
            pass

    def perceive_local_qualia(self, qualia: dict[str, Any]) -> None:
        """Record perceived qualia for tests and small integrations."""
        self.state = getattr(self, "state", {})
        try:
            self.state["perceived"] = dict(qualia or {})
        except Exception:
            self.state["perceived"] = {"raw": qualia}

    def get_dynamic_signature(self, alpha: float = 0.7):
        # deterministic lightweight signature for smoke tests
        base = sum(float(x) for x in self.signature)
        return [base * alpha + i * 0.01 for i in range(len(self.signature))]

    def resonance_with(self, other: "LivingSymbolLite", falloff: float = 1.0) -> float:
        sig_a = self.get_dynamic_signature()
        sig_b = other.get_dynamic_signature()
        # simple resonance: inverse of euclidean distance of first 3 components + falloff
        import math

        dist = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(sig_a[:3], sig_b[:3], strict=False))
        )
        if dist <= 0:
            return 1.0
        return max(0.0, 1.0 / (dist * float(falloff)))

    def compile_intention(self, intention: Any):
        # Normalize older dict-shaped intentions that contain an 'experience' matrix
        try:
            if isinstance(intention, dict) and "experience" in intention:
                from crisalida_lib.EDEN.intention import IntentionMatrix

                matrix = intention.get("experience")
                im = IntentionMatrix(matrix=matrix, coherence=float(intention.get("coherence", 0.5)))
                if self.internal_vm and hasattr(self.internal_vm, "compile_intention"):
                    try:
                        vm_res = self.internal_vm.compile_intention(im)
                        if vm_res:
                            return vm_res
                    except Exception:
                        pass
        except Exception:
            pass

        if self.internal_vm and hasattr(self.internal_vm, "compile_intention"):
            try:
                vm_res = self.internal_vm.compile_intention(intention)
                # If VM produced meaningful instructions, use them; otherwise fall through
                if vm_res:
                    return vm_res
            except Exception:
                pass
        # Fallback to compiler adapter (use compile_intention when available)
        try:
            from crisalida_lib.EVA.compiler_adapter import (
                compile_intention as _compile_int,
            )

            res = _compile_int(intention)
            # normalize common result shapes
            if hasattr(res, "bytecode"):
                bc = res.bytecode
                if hasattr(bc, "instructions"):
                    return list(bc.instructions or [])
            if hasattr(res, "instructions"):
                return list(res.instructions or [])
            if isinstance(res, list):
                return res
            return []
        except Exception:
            # try EDEN-level fallback
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                return compile_intention_to_bytecode(intention)
            except Exception:
                return []

    def execute_internal_logic(self, bytecode: Any):
        if self.internal_vm and hasattr(self.internal_vm, "execute_bytecode"):
            try:
                return self.internal_vm.execute_bytecode(bytecode)
            except Exception:
                return {"status": "error"}
        return {"status": "noop"}


def get_living_symbol_class() -> Any:
    """Return a callable that constructs a living symbol instance.

    If the rich `LivingSymbolV8` is available return a factory that
    supplies safe defaults so callers can use `LS(entity_id)`.
    Otherwise return the lightweight class.
    """
    if RICH_AVAILABLE and LivingSymbolV8 is not None:

        def factory(
            entity_id: str,
            unified_field: Any = None,
            initial_intention_matrix: Any | None = None,
        ) -> Any:
            # LivingSymbolV8 signature is (entity_id, unified_field, initial_position=..., divine_signature=..., initial_intention_matrix=..., config=...)
            try:
                # cast the potentially shallow list to Any to avoid mypy index-type complaints
                inst = LivingSymbolV8(
                    entity_id,
                    unified_field,
                    initial_intention_matrix=cast(Any, initial_intention_matrix),
                )
            except TypeError:
                # Fallback: try positional mapping for older signatures
                try:
                    inst = LivingSymbolV8(
                        entity_id, unified_field, cast(Any, initial_intention_matrix)
                    )
                except Exception:
                    # As last resort, construct minimally
                    return LivingSymbolLite(
                        entity_id, unified_field, initial_intention_matrix
                    )

            # Wrap to provide a stable subset of methods expected by the runtime
            class LivingSymbolWrapper:
                def __init__(self, inner: Any):
                    self._inner = inner

                def __getattr__(self, name: str):
                    return getattr(self._inner, name)

                def get_dynamic_signature(self, *args, **kwargs):
                    sig = None
                    if hasattr(self._inner, "get_dynamic_signature"):
                        try:
                            sig = self._inner.get_dynamic_signature(*args, **kwargs)
                        except Exception:
                            sig = None
                    if sig is None and hasattr(self._inner, "divine_signature"):
                        # derive a simple signature
                        ds = self._inner.divine_signature
                        try:
                            bv = getattr(ds, "base_vector", None)
                            if bv is not None:
                                # convert to list if numpy
                                if hasattr(bv, "tolist"):
                                    return list(bv.tolist())
                                return list(bv)
                        except Exception:
                            pass
                    # Normalize common numpy arrays to lists
                    if hasattr(sig, "tolist"):
                        try:
                            return list(sig.tolist())
                        except Exception:
                            pass
                    if isinstance(sig, (list, tuple)):
                        return list(sig)
                    return [0.0]

                def compile_intention(self, intention: Any):
                    # prefer inner implementation
                    if hasattr(self._inner, "compile_intention"):
                        try:
                            inner_res = self._inner.compile_intention(intention)
                            if inner_res:
                                return inner_res
                        except Exception:
                            pass

                    # try internal VM if present
                    if hasattr(self._inner, "internal_vm") and hasattr(
                        self._inner.internal_vm, "compile_intention"
                    ):
                        try:
                            vm_res = self._inner.internal_vm.compile_intention(
                                intention
                            )
                            if vm_res:
                                return vm_res
                        except Exception:
                            pass

                    # fallback to compiler adapter
                    try:
                        from crisalida_lib.EVA.compiler_adapter import (
                            compile_intention as _compile_int,
                        )

                        res = _compile_int(intention)
                        if hasattr(res, "bytecode"):
                            bc = res.bytecode
                            if hasattr(bc, "instructions"):
                                return list(bc.instructions or [])
                        if hasattr(res, "instructions"):
                            return list(res.instructions or [])
                        if isinstance(res, list):
                            return res
                        return []
                    except Exception:
                        # final fallback: EDEN-level generator
                        try:
                            from crisalida_lib.EDEN.bytecode_generator import (
                                compile_intention_to_bytecode,
                            )

                            return compile_intention_to_bytecode(intention)
                        except Exception:
                            return []

                def execute_internal_logic(self, bytecode: Any):
                    # Normalize wrappers that include .bytecode or .instructions
                    instrs = None
                    if hasattr(bytecode, "bytecode"):
                        bc = bytecode.bytecode
                        if hasattr(bc, "instructions"):
                            instrs = bc.instructions
                    if instrs is None and hasattr(bytecode, "instructions"):
                        instrs = bytecode.instructions
                    if instrs is None and isinstance(bytecode, list):
                        instrs = bytecode

                    if hasattr(self._inner, "execute_internal_logic"):
                        try:
                            return self._inner.execute_internal_logic(
                                instrs or bytecode
                            )
                        except Exception:
                            pass
                    if hasattr(self._inner, "internal_vm") and hasattr(
                        self._inner.internal_vm, "execute_bytecode"
                    ):
                        try:
                            return self._inner.internal_vm.execute_bytecode(
                                instrs or bytecode
                            )
                        except Exception:
                            pass
                    return {"status": "noop"}

            return LivingSymbolWrapper(inst)

        return factory
    return LivingSymbolLite
