from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

"""
Intention shims for Crisalida
Provides a small, safe `IntentionMatrix` dataclass and a minimal
`ADAMIntegrationPoint` and `InternalVMShim` to be used as fallbacks or
integration points for richer implementations in `divine_language`.
"""


@dataclass
class IntentionMatrix:
    """Lightweight representation of a matrix of intention.

    Fields:
      matrix: 2D array of sigils (strings)
      description: human description
      coherence: heuristic score 0..1
      last_updated: epoch float
    """

    matrix: list[list[str]]
    description: str = ""
    coherence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "description": self.description,
            "coherence": float(self.coherence),
            "last_updated": float(self.last_updated),
        }

    def touch(self) -> None:
        self.last_updated = time.time()


class ADAMIntegrationPoint:
    """Minimal ADAM integration shim.

    Implementations may provide a richer module. This class exposes the
    minimal methods the rest of the runtime can call safely.
    """

    def __init__(self) -> None:
        self.adam_module: Any | None = None
        self.consciousness_level: float = 0.0
        self.awareness_threshold: float = 0.8

    def initialize_adam(self, adam_module: Any) -> None:
        self.adam_module = adam_module

    def is_conscious(self) -> bool:
        return float(self.consciousness_level) >= float(self.awareness_threshold)

    def reflect_on_self(self, context: dict[str, Any]) -> dict[str, Any]:
        """Perform a minimal reflection and return a dictionary of insights.

        If a real ADAM module is attached, delegate to it when possible.
        """
        if self.adam_module is not None and hasattr(
            self.adam_module, "reflect_on_self"
        ):
            try:
                return self.adam_module.reflect_on_self(context)
            except Exception:
                # degrade gracefully
                pass

        # Minimal safe reflection
        self.consciousness_level = min(1.0, float(self.consciousness_level) + 0.01)
        return {
            "timestamp": time.time(),
            "consciousness_level": float(self.consciousness_level),
            "self_awareness": min(1.0, float(self.consciousness_level) * 1.1),
            "context_summary": str(type(context)),
        }


class InternalVMShim:
    """Small interface for an internal VM used to execute intention matrices.

    Backends should implement `execute_bytecode` and `compile_intention`.
    The shim provides fallbacks that do not raise at import time.
    """

    def compile_intention(self, intention: Any) -> list[Any]:
        """Compile an IntentionMatrix-like object into a list of simple instructions.

        Output format: list of dicts: {"opcode": "MNFT", "operands": [sigil], "meta": {...}}
        """
        # Extract matrix
        matrix = None
        try:
            if hasattr(intention, "matrix"):
                matrix = intention.matrix
            elif hasattr(intention, "sigils"):
                matrix = intention.sigils
            elif isinstance(intention, list):
                matrix = intention
        except Exception:
            matrix = None

        if not matrix:
            return []

        # Prefer repository compiler adapter if available
        try:
            from crisalida_lib.EVA import compiler_adapter

            compiled = compiler_adapter.compile_matrix(matrix)
            # if adapter returns a list of instructions, use it
            if isinstance(compiled, list) and len(compiled) > 0:
                return compiled
        except Exception:
            # fall back to internal simple compiler
            pass

        # Fallback: produce richer, semantic bytecode similar to v7
        instructions: list[Any] = []
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0

        coherence_val = 0.5
        try:
            coherence_val = float(getattr(intention, "coherence", coherence_val))
        except Exception:
            pass

        # Q_CREATE instructions
        for i, row in enumerate(matrix):
            for j, sigil in enumerate(row):
                instruction = {
                    "opcode": "Q_CREATE",
                    "operands": [sigil, f"cell_{i}_{j}"],
                    "metadata": {
                        "position": (i, j),
                        "matrix_coherence": coherence_val,
                    },
                }
                instructions.append(instruction)

        # Q_RESONATE between adjacent sigils (right, down, diagonal)
        for i in range(rows):
            for j in range(cols):
                for di, dj in [(0, 1), (1, 0), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < rows and nj < cols:
                        instruction = {
                            "opcode": "Q_RESONATE",
                            "operands": [f"cell_{i}_{j}", f"cell_{ni}_{nj}"],
                            "metadata": {"resonance_type": "harmonic", "strength": 0.8},
                        }
                        instructions.append(instruction)

        return instructions

    def execute_bytecode(self, bytecode: list[Any]) -> dict[str, Any]:
        """Execute a list of simple instructions produced by compile_intention.

        For MNFT ops, create a minimal manifestation object.
        """
        manifestations: list[Any] = []
        executed = 0
        for idx, instr in enumerate(bytecode or []):
            try:
                # simple VM stack used for PUSH/POP_STACK simulation
                if idx == 0:
                    vm_stack = []

                if not isinstance(instr, dict):
                    continue

                opcode = instr.get("opcode")

                if opcode == "MNFT":
                    sig = instr.get("operands", [None])[0]
                    ent = {
                        "type": "created",
                        "entity_id": f"manifest_{idx}",
                        "sigil": sig,
                        "meta": instr.get("meta", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_CREATE":
                    sig = instr.get("operands", [None])[0]
                    cell = instr.get("operands", [None, None])[1]
                    ent = {
                        "type": "created",
                        "entity_id": cell or f"manifest_{idx}",
                        "sigil": sig,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_RESONATE":
                    a, b = instr.get("operands", [None, None])[:2]
                    link = {
                        "type": "resonance",
                        "link_id": f"link_{idx}",
                        "a": a,
                        "b": b,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(link)
                    executed += 1

                elif opcode == "Q_PUSH":
                    # push an entity or literal onto the VM 'stack' — produce a pushed manifestation
                    val = None
                    try:
                        val = instr.get("operands", [None])[0]
                    except Exception:
                        val = None
                    # resolve entity_ref if present and entity-like operand provided
                    if isinstance(val, dict) and val.get("type") == "entity_ref":
                        ent_id = val.get("id")
                        # best-effort: if the VM was provided a context entity, match by id
                        resolved = None
                        try:
                            # look for a provided 'entity' in metadata
                            ctx_entity = instr.get("metadata", {}).get("entity")
                            if (
                                ctx_entity
                                and isinstance(ctx_entity, dict)
                                and ctx_entity.get("id") == ent_id
                            ):
                                resolved = ctx_entity
                        except Exception:
                            resolved = None
                        ent = {
                            "type": "pushed",
                            "value": (
                                resolved if resolved is not None else {"id": ent_id}
                            ),
                            "meta": instr.get("metadata", {}),
                        }
                    else:
                        ent = {
                            "type": "pushed",
                            "value": val,
                            "meta": instr.get("metadata", {}),
                        }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_RELEASE_HORMONE":
                    hormone = None
                    strength = 1.0
                    try:
                        ops = instr.get("operands", [])
                        if len(ops) >= 1:
                            hormone = ops[0]
                        if len(ops) >= 2:
                            strength = float(ops[1])
                    except Exception:
                        hormone = None
                        strength = 1.0
                    ent = {
                        "type": "hormone_released",
                        "hormone": hormone,
                        "strength": strength,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_EVOLVE_CODE":
                    strategy = None
                    params = {}
                    try:
                        ops = instr.get("operands", [])
                        if len(ops) >= 1:
                            strategy = ops[0]
                        if len(ops) >= 2 and isinstance(ops[1], dict):
                            params = ops[1]
                    except Exception:
                        strategy = None
                        params = {}
                    ent = {
                        "type": "evolved",
                        "strategy": strategy,
                        "params": params,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_NAVIGATE_PATH":
                    path = instr.get("operands", [None])[0]
                    ent = {
                        "type": "navigation",
                        "path": path,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_QUERY_REALITY":
                    q = instr.get("operands", [None])[0]
                    ent = {
                        "type": "query",
                        "query": q,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_PUSH_STACK":
                    val = instr.get("operands", [None])[0]
                    vm_stack.append(val)
                    manifestations.append(
                        {
                            "type": "stack_pushed",
                            "value": val,
                            "meta": instr.get("metadata", {}),
                        }
                    )
                    executed += 1

                elif opcode == "Q_POP_STACK":
                    val = None
                    try:
                        val = vm_stack.pop() if vm_stack else None
                    except Exception:
                        val = None
                    manifestations.append(
                        {
                            "type": "stack_popped",
                            "value": val,
                            "meta": instr.get("metadata", {}),
                        }
                    )
                    executed += 1

                elif opcode == "Q_MODIFY_ATTRIBUTE":
                    ops = instr.get("operands", [])
                    # expect a list of dicts with attribute/operation/value
                    for op_item in ops:
                        try:
                            attr = op_item.get("attribute")
                            operation = op_item.get("operation")
                            value = op_item.get("value")
                            m = {
                                "type": "attribute_modified",
                                "attribute": attr,
                                "operation": operation,
                                "value": value,
                                "meta": instr.get("metadata", {}),
                            }
                            manifestations.append(m)
                            executed += 1
                        except Exception:
                            continue

                elif opcode == "Q_TRANSFORM":
                    sig, cell = instr.get("operands", [None, None])[:2]
                    ent = {
                        "type": "transform",
                        "entity_id": cell or f"manifest_{idx}",
                        "sigil": sig,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                elif opcode == "Q_CRYSTALLIZE":
                    sig, cell = instr.get("operands", [None, None])[:2]
                    ent = {
                        "type": "crystallize",
                        "entity_id": cell or f"manifest_{idx}",
                        "sigil": sig,
                        "meta": instr.get("metadata", {}),
                    }
                    manifestations.append(ent)
                    executed += 1

                else:
                    # Unknown opcode: create a generic manifestation so outputs
                    # from external compilers are not silently dropped.
                    try:
                        meta = instr.get("metadata", {}) or instr.get("meta", {})
                        ent = {
                            "type": "opcode",
                            "opcode": opcode,
                            "operands": instr.get("operands", []),
                            "meta": meta or {},
                        }
                        manifestations.append(ent)
                        executed += 1
                    except Exception:
                        continue

            except Exception:
                # swallow errors per shim philosophy
                continue

        return {
            "status": "executed",
            "manifestations": manifestations,
            "executed": executed,
            "instructions_executed": executed,
        }
