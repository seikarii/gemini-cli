from typing import Any


class InternalVM:
    """A minimal internal VM shim exposing compile and execute methods.

    The real system may delegate to EVA/backends; this shim offers a
    deterministic fallback behavior for tests and integration.
    """

    def compile_intention(self, intention: dict[str, Any]) -> list[Any]:
        # Very small 'compiler' that turns an intention dict into a list of
        # instructions (tuples) for smoke-testing the pipeline.
        return [("INTENT", intention.get("intention_type"), intention.get("phase"))]

    def execute_bytecode(
        self, instructions: list[Any], context: Any = None
    ) -> list[Any]:
        # Execute instructions by echoing them into a simple manifestation list
        manifestations = []
        for instr in instructions or []:
            manifestations.append({"instr": instr, "context": context})
        return manifestations


"""Internal VM shim used as a safe, tiny fallback for compilation and execution.
"""


class InternalVMShim:
    def compile_intention(self, intention: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            from crisalida_lib.EDEN.bytecode_generator import (
                compile_intention_to_bytecode,
            )

            return compile_intention_to_bytecode(
                intention if isinstance(intention, dict) else {}
            )
        except Exception:
            return []

    def execute_bytecode(
        self, bytecode: Any, quantum_field: Any = None
    ) -> list[dict[str, Any]]:
        # bytecode expected to be a list of instruction dicts; return simple manifestations
        out = []
        try:
            instrs = []
            if hasattr(bytecode, "instructions"):
                instrs = list(bytecode.instructions or [])
            elif isinstance(bytecode, list):
                instrs = bytecode
            else:
                instrs = []
            for instr in instrs:
                op = instr.get("op") if isinstance(instr, dict) else str(instr)
                payload = instr.get("payload") if isinstance(instr, dict) else {}
                out.append({"op": op, "payload": payload})
        except Exception:
            pass
        return out
