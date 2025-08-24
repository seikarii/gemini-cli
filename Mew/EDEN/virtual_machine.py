from __future__ import annotations

import logging
from typing import Any

"""
Ontological Virtual Machine v9.0 (OVM) for EDEN
=================================================
Executes ontological bytecode, interfacing with the PhysicsEngine to manifest reality.
"""

# Import EVA memory mixin and fixed symbolic language imports
try:
    from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
    from crisalida_lib.EVA.types import (
        OntologicalInstruction,
        OntologicalOpcode,
        SemanticType,
        TypedValue,
    )

    try:
        from crisalida_lib.EVA.types import LivingSymbolRuntime
    except Exception:  # pragma: no cover - allow import even if EVA subsystem missing
        LivingSymbolRuntime = Any  # type: ignore
except Exception:  # pragma: no cover - degrade if EVA subsystem is not present
    EVAMemoryMixin = object  # type: ignore
    OntologicalInstruction = Any  # type: ignore
    OntologicalOpcode = Any  # type: ignore
    SemanticType = Any  # type: ignore
    TypedValue = Any  # type: ignore
    LivingSymbolRuntime = Any  # type: ignore

logger = logging.getLogger(__name__)


class OntologicalVirtualMachine(EVAMemoryMixin):
    """
    Executes ontological bytecode. It translates abstract instructions into
    concrete actions within the physics engine, like manifesting new LivingSymbols.
    Supports both full and tick-based execution.
    """

    def __init__(
        self,
        manifold: Any,
        physics_engine: Any,
        eva_runtime: Any | None = None,
    ):
        super().__init__()  # Initialize the mixin
        self.manifold = manifold
        self.physics_engine = physics_engine
        self.eva_runtime = eva_runtime
        self.stack: list[Any] = []
        self.bytecode: list[OntologicalInstruction] = []
        self.instruction_pointer: int = 0
        self.running: bool = False
        self.execution_history: list[dict[str, Any]] = []
        self.opcode_handlers = {
            "Q_PUSH": self._handle_push,
            "Q_POP": self._handle_pop,
            "MNFT": self._handle_manifest_matter,  # Alias for MANIFEST
            "Q_MANIFEST_MATTER": self._handle_manifest_matter,
        }

    def load_bytecode(self, bytecode: list[OntologicalInstruction]):
        """Loads bytecode into the VM and resets its state for execution."""
        self.bytecode = bytecode
        self.instruction_pointer = 0
        self.stack = []
        self.execution_history.clear()
        self.running = True
        logger.info(f"OVM: Loaded {len(bytecode)} instructions. Ready to tick.")

    def tick(self) -> bool:
        """
        Executes a single instruction from the loaded bytecode.
        Returns True if the VM is still running, False if execution has finished or halted.
        """
        if not self.running or self.instruction_pointer >= len(self.bytecode):
            if self.running:
                logger.info("OVM: End of bytecode reached.")
                self.running = False
            return False

        instruction = self.bytecode[self.instruction_pointer]
        
        try:
            handler = self.opcode_handlers.get(instruction.opcode.name)
            if not handler:
                handler = self.opcode_handlers.get(str(instruction.opcode))

            if handler:
                handler(instruction)
            else:
                logger.warning(f"Unsupported opcode: {instruction.opcode.name}")

            self.instruction_pointer += 1

        except Exception as e:
            error_info = {"error": str(e), "ip": self.instruction_pointer}
            self.execution_history.append(error_info)
            logger.error(f"OVM Execution Error: {e} at IP {self.instruction_pointer}", exc_info=True)
            self.running = False
            return False

        if self.instruction_pointer >= len(self.bytecode):
            logger.info("OVM: End of bytecode reached.")
            self.running = False
        
        return self.running

    def execute_bytecode(self, bytecode: list[Any]) -> dict[str, Any]:
        """
        Executes a bytecode to completion for backward compatibility.
        Internally uses the new tick-based system.
        """
        self.load_bytecode(bytecode)
        
        while self.tick():
            pass

        logger.info("OVM: Full execution finished.")
        
        last_error = next((h for h in reversed(self.execution_history) if "error" in h), None)
        if last_error:
            return {"status": "error", "message": last_error['error']}
        
        return {
            "status": "finished",
            "stack_top": self.stack[-1] if self.stack else None,
            "instructions_executed": self.instruction_pointer
        }

    def _push_typed(self, value: Any, semantic_type: Any | None = None):
        if semantic_type is None:
            if isinstance(value, str): semantic_type = SemanticType.STRING
            elif isinstance(value, (int, float)): semantic_type = SemanticType.NUMBER
            elif isinstance(value, list): semantic_type = SemanticType.LIST
            else: semantic_type = SemanticType.ABSTRACT
        self.stack.append(TypedValue(value=value, semantic_type=semantic_type))

    def _pop_typed(self) -> Any:
        if not self.stack:
            raise ValueError("OVM Stack Underflow: cannot pop from an empty stack.")
        return self.stack.pop()

    def _handle_push(self, instruction: Any):
        val, type_val = instruction.operands[0]
        self._push_typed(val, SemanticType(type_val))

    def _handle_pop(self, instruction: Any):
        self._pop_typed()

    def _handle_manifest_matter(self, instruction: Any):
        """Handles the manifestation of a new LivingSymbol."""
        logger.debug(f"Executing MANIFEST for instruction: {instruction}")
        try:
            if not instruction.operands:
                raise ValueError("OVM Operand Error: MANIFEST requires at least one operand for pattern_type.")

            pattern_type = instruction.operands[0]
            pattern_addresses = instruction.operands[1] if len(instruction.operands) > 1 else []

            new_symbol = self.physics_engine.manifest_living_symbol(
                pattern_addresses=pattern_addresses, pattern_type=pattern_type
            )

            self._push_typed(new_symbol, SemanticType.ENTITY)
            logger.info(f"OVM: Manifested new LivingSymbol {getattr(new_symbol, 'entity_id', 'unknown')}")

        except Exception as e:
            logger.error(f"Error during Q_MANIFEST_MATTER: {e}", exc_info=True)
            raise
