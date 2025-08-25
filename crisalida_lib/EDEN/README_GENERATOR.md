EDEN bytecode generator policy

EDEN code prefers to delegate higher-level semantic compilation to the canonical
EVA generator located at `crisalida_lib.EVA.bytecode_generator.generate_bytecode`.

Reasons:
- Centralises heuristics and reduces duplication.
- Makes it safe to iterate on generator heuristics in EVA without touching EDEN.

Behaviour:
- `crisalida_lib.EDEN.bytecode_generator.compile_intention_to_bytecode` performs a
  lazy import of the EVA generator and delegates when available.
- If EVA generator is unavailable, EDEN provides a conservative, small fallback
  instruction stream.

If you change generator signatures, update both the EVA generator and the EDEN
adapter to keep compatibility, or prefer adding a thin adapter in `crisalida_lib/EVA/compiler_adapter.py`.
