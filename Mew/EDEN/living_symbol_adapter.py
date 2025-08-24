"""LivingSymbol adapter and wrapper for fusion plan.

Provides a factory that returns a LivingSymbol implementation: the rich
LivingSymbolV8 (if available) wrapped into a stable API, or a lightweight
LivingSymbolLite fallback.
"""

from collections.abc import Callable
from typing import Any


def get_living_symbol_factory() -> Callable[..., Any]:
    try:
        from divine_language.livingsimbolv8 import LivingSymbolV8  # type: ignore

        def factory(
            entity_id: str,
            unified_field: Any = None,
            initial_intention_matrix: Any = None,
        ):
            inst = LivingSymbolV8(
                entity_id,
                unified_field,
                initial_intention_matrix=initial_intention_matrix,
            )

            class Wrapper:
                def __init__(self, inner: Any):
                    self._inner = inner

                def __getattr__(self, name: str):
                    return getattr(self._inner, name)

                def get_dynamic_signature(self, *args, **kwargs):
                    if hasattr(self._inner, "get_dynamic_signature"):
                        return self._inner.get_dynamic_signature(*args, **kwargs)
                    if hasattr(self._inner, "divine_signature"):
                        ds = self._inner.divine_signature
                        return getattr(ds, "base_vector", [])
                    return [0.0]

                def compile_intention(self, intention: Any):
                    # Prefer inner implementation but fall back to shims if it returns empty
                    if hasattr(self._inner, "compile_intention"):
                        try:
                            compiled = self._inner.compile_intention(intention)
                            if compiled:
                                return compiled
                        except Exception:
                            pass
                    if hasattr(self._inner, "internal_vm") and hasattr(
                        self._inner.internal_vm, "compile_intention"
                    ):
                        try:
                            compiled = self._inner.internal_vm.compile_intention(intention)
                            if compiled:
                                return compiled
                        except Exception:
                            pass
                    from crisalida_lib.EDEN.intention import InternalVMShim

                    shim = InternalVMShim()
                    # If caller passed the older dict-shaped intention, convert it
                    try:
                        if isinstance(intention, dict) and "experience" in intention:
                            from crisalida_lib.EDEN.intention import IntentionMatrix

                            matrix = intention.get("experience")
                            im = IntentionMatrix(matrix=matrix, coherence=float(intention.get("coherence", 0.5)))
                            compiled = shim.compile_intention(im)
                            if compiled:
                                return compiled
                    except Exception:
                        pass

                    return shim.compile_intention(intention)

                def execute_internal_logic(self, bytecode: Any):
                    if hasattr(self._inner, "execute_internal_logic"):
                        try:
                            return self._inner.execute_internal_logic(bytecode)
                        except TypeError:
                            # Some LivingSymbol implementations expose a no-arg
                            # execute_internal_logic() that uses internal state.
                            try:
                                return self._inner.execute_internal_logic()
                            except Exception:
                                pass
                    if hasattr(self._inner, "internal_vm") and hasattr(
                        self._inner.internal_vm, "execute_bytecode"
                    ):
                        return self._inner.internal_vm.execute_bytecode(bytecode)
                    return {"status": "noop"}

            return Wrapper(inst)

        return factory
    except Exception:
        # Fallback lightweight implementation
        class LivingSymbolLite:
            def __init__(
                self,
                entity_id: str,
                unified_field: Any = None,
                initial_intention_matrix: Any = None,
            ):
                self.entity_id = entity_id
                self.unified_field = unified_field
                self.intention_matrix = initial_intention_matrix
                # Explicit runtime-typed state mapping to satisfy mypy var-annotated
                self.state: dict[str, Any] = {}
                self.signature = [0.0] * 8

            def update(self, dt: float = 0.1, neighbors: list[Any] | None = None):
                self.state["tick"] = self.state.get("tick", 0) + 1
                return self.state

            def get_dynamic_signature(self, alpha: float = 0.7):
                base = sum(float(x) for x in self.signature)
                return [base * alpha + i * 0.01 for i in range(len(self.signature))]

            def compile_intention(self, intention: Any):
                from crisalida_lib.EDEN.intention import InternalVMShim

                shim = InternalVMShim()
                # Accept older dict-shaped intention used in integration tests
                try:
                    if isinstance(intention, dict) and "experience" in intention:
                        from crisalida_lib.EDEN.intention import IntentionMatrix

                        matrix = intention.get("experience")
                        im = IntentionMatrix(matrix=matrix, coherence=float(intention.get("coherence", 0.5)))
                        compiled = shim.compile_intention(im)
                        return compiled
                except Exception:
                    pass

                return shim.compile_intention(intention)

            def execute_internal_logic(self, bytecode: Any):
                from crisalida_lib.EDEN.intention import InternalVMShim

                shim = InternalVMShim()
                return shim.execute_bytecode(bytecode)

        return LivingSymbolLite
