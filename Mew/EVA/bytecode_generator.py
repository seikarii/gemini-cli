"""Lightweight bytecode generator inspired by BytecodeGeneratorV7.

This module provides a conservative, dependency-free `generate_bytecode`
function suitable for use as a safe fallback during the fusion process.

It recognises a few simple markers in the matrix and emits a list of
instruction dicts with fields: opcode, operands, metadata.
"""

from typing import Any


def _map_operator(op: str) -> str:
    return {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}.get(op, "set")


def generate_bytecode(
    matrix: Any, semantic_analysis: Any = None, entity: Any | None = None
) -> list[Any]:
    """Generate a list of simple instruction dicts from a matrix.

    Heuristics:
    - presence of 'Δ' -> Q_EVOLVE_CODE
    - presence of snow sigils -> Q_CRYSTALLIZE (push entity if provided)
    - tokens starting with 'PATH:' -> Q_NAVIGATE_PATH
    - numeric sequences separated by operator tokens produce Q_PUSH/Q_MODIFY_ATTRIBUTE
    - tokens starting with 'Q:' or 'QUERY:' -> Q_QUERY_REALITY
    """
    try:
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
    except Exception:
        return []

    instrs: list[Any] = []

    # scan matrix for markers
    seen_evolve = False
    seen_crystal = False
    for i in range(rows):
        for j in range(cols):
            try:
                token = matrix[i][j]
            except Exception:
                token = None
            if token is None:
                continue
            s = str(token)
            # Evolve marker: can be explicit 'Δ' or token starting with 'EVOLVE:'
            if (("Δ" in s) or s.startswith("EVOLVE:")) and not seen_evolve:
                # EVOLVE may include a strategy and optional params after ':'
                if s.startswith("EVOLVE:") and ":" in s:
                    payload = s.split(":", 1)[1]
                    # parse payload like 'genetic,rate=0.1,depth=3'
                    parts = [p.strip() for p in payload.split(",") if p.strip()]
                    strategy = parts[0] if parts else "adaptive"
                    from typing import Any as _Any

                    params: dict[str, _Any] = {}
                    for p in parts[1:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            vv: _Any
                            try:
                                # coerce numeric
                                if "." in v:
                                    vv = float(v)
                                else:
                                    vv = int(v)
                            except Exception:
                                vv = v
                            params[k.strip()] = vv
                    instrs.append(
                        {
                            "opcode": "Q_EVOLVE_CODE",
                            "operands": [strategy, params],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                else:
                    instrs.append(
                        {
                            "opcode": "Q_EVOLVE_CODE",
                            "operands": [["system"], "adaptive"],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                seen_evolve = True
            if any(ch in s for ch in ("❄", "✻", "✺")) and not seen_crystal:
                if entity is not None:
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [entity],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                instrs.append(
                    {
                        "opcode": "Q_CRYSTALLIZE",
                        "operands": [s, f"cell_{i}_{j}"],
                        "metadata": {"pos": (i, j)},
                    }
                )
                seen_crystal = True
            if s.startswith("PATH:"):
                path = s[len("PATH:") :]
                # allow comma-separated segments
                segments = [seg.strip() for seg in path.split(",") if seg.strip()]
                instrs.append(
                    {
                        "opcode": "Q_NAVIGATE_PATH",
                        "operands": [segments],
                        "metadata": {"pos": (i, j)},
                    }
                )
            if s.startswith("QUERY:") or s.startswith("Q:"):
                q = s.split(":", 1)[1] if ":" in s else ""
                # parse key=value pairs separated by ',' into dict
                from typing import Any as _Any

                query_params: dict[str, _Any] = {}
                for part in q.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        query_params[k.strip()] = v.strip()
                if query_params:
                    instrs.append(
                        {
                            "opcode": "Q_QUERY_REALITY",
                            "operands": [query_params],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                else:
                    instrs.append(
                        {
                            "opcode": "Q_QUERY_REALITY",
                            "operands": [q],
                            "metadata": {"pos": (i, j)},
                        }
                    )
            # Push syntax to explicitly push an entity or literal: PUSH:entityId or PUSH:42
            if s.startswith("PUSH:"):
                val = s.split(":", 1)[1]
                # support PUSH:entity:<id> to indicate pushing an entity by id
                if val.startswith("entity") and ":" in val:
                    _, eid = val.split(":", 1)
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [{"type": "entity_ref", "id": eid}],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                elif val == "entity" and entity is not None:
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [entity],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                else:
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [val],
                            "metadata": {"pos": (i, j)},
                        }
                    )
            # Stack-style pushes: PUSH_STACK:val and POP_STACK
            if s.startswith("PUSH_STACK:"):
                val = s.split(":", 1)[1]
                instrs.append(
                    {
                        "opcode": "Q_PUSH_STACK",
                        "operands": [val],
                        "metadata": {"pos": (i, j)},
                    }
                )
            if s == "POP_STACK":
                instrs.append(
                    {
                        "opcode": "Q_POP_STACK",
                        "operands": [],
                        "metadata": {"pos": (i, j)},
                    }
                )
            # Release/Hormone tokens -> Q_RELEASE_HORMONE
            if s.startswith("RELEASE:") or s.startswith("HORMONE:"):
                rest = s.split(":", 1)[1] if ":" in s else "generic"
                # allow RELEASE:oxytocin:0.8 or RELEASE:oxytocin
                parts = rest.split(":")
                name = parts[0]
                strength = float(parts[1]) if len(parts) > 1 else 1.0
                instrs.append(
                    {
                        "opcode": "Q_RELEASE_HORMONE",
                        "operands": [name, strength],
                        "metadata": {"pos": (i, j)},
                    }
                )
            # Attribute operations: SET:attr=val or INC:attr=val
            if s.startswith("SET:") or s.startswith("INC:"):
                try:
                    kind, rest = s.split(":", 1)
                    attr, val = rest.split("=", 1)
                    op = "set" if kind == "SET" else "inc"
                    instrs.append(
                        {
                            "opcode": "Q_MODIFY_ATTRIBUTE",
                            "operands": [
                                {"attribute": attr, "operation": op, "value": val}
                            ],
                            "metadata": {"pos": (i, j)},
                        }
                    )
                except Exception:
                    pass

    # detect simple arithmetic rows (e.g., ['2', '+', '3'])
    for i in range(rows):
        row_any = matrix[i]
        # only handle lists/tuples here
        if not isinstance(row_any, (list, tuple)):
            continue
        # cast/normalize to a concrete list so mypy understands indexing
        row: list[Any] = list(row_any)  # type: ignore[arg-type]
        for col_idx in range(len(row) - 2):
            # Use a fresh integer-only index name to avoid previous-name type leaks
            aa: Any = row[col_idx]
            op = row[col_idx + 1]
            bb: Any = row[col_idx + 2]
            try:
                if str(op) in "+-*/" and (str(aa).isdigit() or str(bb).isdigit()):
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [aa],
                            "metadata": {"pos": (i, col_idx)},
                        }
                    )
                    instrs.append(
                        {
                            "opcode": "Q_PUSH",
                            "operands": [bb],
                            "metadata": {"pos": (i, col_idx + 2)},
                        }
                    )
                    instrs.append(
                        {
                            "opcode": "Q_MODIFY_ATTRIBUTE",
                            "operands": [
                                {
                                    "attribute": "temp_value",
                                    "operation": _map_operator(str(op)),
                                    "value": bb,
                                }
                            ],
                            "metadata": {"pos": (i, col_idx)},
                        }
                    )
            except Exception:
                continue

    return instrs
