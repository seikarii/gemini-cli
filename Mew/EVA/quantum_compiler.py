"""
QuantumCompilerV7 - Compilador cuántico para lenguaje divino
===========================================================

Transforma matrices simbólicas y AST en bytecode cuántico optimizado para la VM de Prometeo.
Incluye diagnóstico avanzado, optimización de resonancia y trazabilidad.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

# Avoid circular import
from typing import TYPE_CHECKING, Any, cast

# EVA imports
from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    OntologicalInstruction,
    QuantumField,
    QuantumInstruction,
    RealityBytecode,
)
from crisalida_lib.EVA.typequalia import QualiaState

# Opcode constants are defined in local instructions module; keep only ONT_OPCODES import.
from .instructions import ONT_OPCODES, OntologicalBytecode
from .semantics import SemanticAnalyzer
from .symbolic_matrix import Matrix

if TYPE_CHECKING:
    pass


@dataclass
class CompilationResult:
    success: bool
    bytecode: OntologicalBytecode | None = None
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    compilation_time: float = 0.0


class QuantumCompilerV7:
    """
    Compilador cuántico principal para matrices simbólicas y AST.
    Realiza análisis semántico, optimización de resonancia y generación de bytecode.
    Extensión EVA: compila experiencias vivientes, soporta faseo, hooks de entorno y simulación activa.
    """

    # Class-level annotation to help static analysis without executing code at import-time
    _environment_hooks: list

    def __init__(self, unified_field: Any = None, phase: str = "default"):
        self.unified_field = unified_field
        self.semantic_analyzer = SemanticAnalyzer()
        self.last_diagnostics: dict[str, Any] = {}
        # EVA: memoria viviente y runtime de simulación
        self.eva_runtime = LivingSymbolRuntime()
        # Avoid circular dependency - divine_compiler will be set by parent if needed
        self.divine_compiler = None
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def compile_matrix(self, matrix: Matrix) -> CompilationResult:
        start = time.time()
        instructions = []
        diagnostics = {
            "matrix_shape": matrix.get_shape(),
            "sigil_summary": matrix.summary(),
        }
        for i, row in enumerate(matrix.sigils):
            for j, sigil in enumerate(row):
                opcode = self._determine_opcode(sigil)
                operands = self._extract_operands(sigil)
                instr = QuantumInstruction(
                    opcode=opcode,
                    operands=operands,
                    metadata={
                        "row": i,
                        "col": j,
                        "sigil": getattr(sigil, "glyph", str(sigil)),
                        "source_sigil_glyph": getattr(sigil, "glyph", None),
                        "source_sigil_position": (i, j),
                    },
                    instruction_id=str(uuid.uuid4()),
                    quantum_coherence=getattr(sigil, "consciousness_density", 1.0),
                    resonance_frequency=getattr(sigil, "resonance_frequency", 0.0),
                )
                instructions.append(instr)
        # instructions is a list[QuantumInstruction] which is compatible at
        # runtime with OntologicalInstruction; perform a conservative cast
        # to satisfy the static checker without changing runtime behavior.
        ont_instructions: list = list(instructions)
        bytecode = OntologicalBytecode(
            instructions=ont_instructions,
            matrix_signature=str(uuid.uuid4()),
            compilation_timestamp=time.time(),
            source_metadata=diagnostics,
            bytecode_id=str(uuid.uuid4()),
        )
        compilation_time = time.time() - start
        result = CompilationResult(
            success=True,
            bytecode=bytecode,
            diagnostics=diagnostics,
            compilation_time=compilation_time,
        )
        self.last_diagnostics = diagnostics
        return result

    def compile_ast(self, ast: dict[str, Any]) -> CompilationResult:
        start = time.time()
        semantic_intent = self.semantic_analyzer.analyze(ast)
        instructions = self._intent_to_instructions(semantic_intent)
        ont_instructions = list(instructions)
        bytecode = OntologicalBytecode(
            instructions=ont_instructions,
            matrix_signature=str(uuid.uuid4()),
            compilation_timestamp=time.time(),
            source_metadata={"ast": ast, "semantic_intent": semantic_intent},
            bytecode_id=str(uuid.uuid4()),
        )
        compilation_time = time.time() - start
        result = CompilationResult(
            success=True,
            bytecode=bytecode,
            diagnostics={"semantic_intent": semantic_intent},
            compilation_time=compilation_time,
        )
        self.last_diagnostics = result.diagnostics
        return result

    def _determine_opcode(self, sigil: Any) -> str:
        # Determina el opcode según las propiedades del sigil
        if hasattr(sigil, "category"):
            cat = sigil.category
            if cat.name == "CREATOR":
                return ONT_OPCODES["INSTANTIATE"]
            elif cat.name == "TRANSFORMER":
                return ONT_OPCODES["TRANSFORM"]
            elif cat.name == "CONNECTOR":
                return ONT_OPCODES["CONNECT"]
            elif cat.name == "DESTROYER":
                return ONT_OPCODES["INVOKE_CHAOS"]
            elif cat.name == "INFINITE":
                return ONT_OPCODES["TRANSCEND"]
            elif cat.name == "PRESERVER":
                return ONT_OPCODES["STABILIZE"]
            elif cat.name == "OBSERVER":
                return ONT_OPCODES["OBSERVE"]
        return ONT_OPCODES["MANIFEST"]

    def _extract_operands(self, sigil: Any) -> list[Any]:
        # Extrae operandos relevantes del sigil
        operands = []
        # Handle simple string sigils as the primary operand
        if isinstance(sigil, str):
            operands.append(sigil)

        if hasattr(sigil, "glyph"):
            operands.append(sigil.glyph)
        if hasattr(sigil, "amplitude"):
            operands.append(sigil.amplitude)
        if hasattr(sigil, "domains"):
            # domains may be an iterable of strings; extend operands rather than
            # appending the list object to keep a flat operand list and satisfy
            # static typing expectations (list[str]).
            try:
                operands.extend(list(sigil.domains))
            except Exception:
                # fallback: append the representation
                operands.append(str(getattr(sigil, "domains", "")))
        return operands

    def _intent_to_instructions(
        self, semantic_intent: dict[str, Any]
    ) -> list[QuantumInstruction]:
        # Convierte una intención semántica en instrucciones ontológicas
        intent_type = semantic_intent.get("type", "UNKNOWN_INTENT")
        details = semantic_intent.get("details", {})
        instructions = []
        if intent_type == "QUERY_ENTITY_STATUS":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["OBSERVE"],
                    operands=[details.get("entity_id")],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        elif intent_type == "EXECUTE_PROTOCOL":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["FLOW"],
                    operands=[
                        details.get("protocol_name"),
                        details.get("target_entity"),
                    ],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        elif intent_type == "CREATE_ENTITY":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["INSTANTIATE"],
                    operands=[details.get("entity_name")],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        elif intent_type == "MODIFY_ENTITY":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["TRANSFORM"],
                    operands=[details.get("entity_id"), details.get("modification")],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        elif intent_type == "LIST_ENTITIES":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["OBSERVE"],
                    operands=["ALL_ENTITIES"],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        elif intent_type == "HELP_REQUEST":
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["META_REFLECT"],
                    operands=[details.get("topic", "general")],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        else:
            instructions.append(
                QuantumInstruction(
                    opcode=ONT_OPCODES["MANIFEST"],
                    operands=[semantic_intent.get("raw_text", "")],
                    metadata={"intent_type": intent_type, "details": details},
                    instruction_id=str(uuid.uuid4()),
                )
            )
        return instructions

    def get_last_diagnostics(self) -> dict[str, Any]:
        """Devuelve el último diagnóstico de compilación."""
        return self.last_diagnostics

    def optimize_bytecode(self, bytecode: OntologicalBytecode) -> OntologicalBytecode:
        """
        Optimiza el bytecode eliminando redundancias y fusionando instrucciones compatibles.
        """
        optimized_instructions = []
        seen = set()
        for instr in bytecode.instructions:
            key = (instr.opcode, tuple(instr.operands))
            if key not in seen:
                optimized_instructions.append(instr)
                seen.add(key)
        bytecode.instructions = optimized_instructions
        return bytecode

    def compile(self, input_data: Any) -> CompilationResult:
        """
        Compila entrada flexible: matriz simbólica, AST o texto.
        """
        if isinstance(input_data, Matrix):
            return self.compile_matrix(input_data)
        elif isinstance(input_data, dict) and "node_type" in input_data:
            return self.compile_ast(input_data)
        elif isinstance(input_data, str):
            # Asume texto: parsear y analizar semánticamente
            from .parser import DivineParser

            parser = DivineParser()
            ast = parser.parse(input_data)
            return self.compile_ast(ast)
        elif isinstance(input_data, dict):
            return self.compile_ast(input_data)  # Treat dict as a high level AST
        else:
            return CompilationResult(
                success=False,
                errors=["Tipo de entrada no soportado"],
                diagnostics={"input_type": str(type(input_data))},
            )

    # --- EVA: Integración de Memoria Viviente y Simulación ---
    def eva_ingest_matrix_experience(
        self, matrix: Matrix, qualia_state: QualiaState, phase: str | None = None
    ) -> str:
        """
        Compila una matriz simbólica como experiencia viviente y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_MATRIX_EXPERIENCE",
            "matrix": matrix.to_list(),
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        # Defensive: ensure we only call compile_intention when a real compiler is attached.
        bytecode = (
            _dc.compile_intention(intention)
            if (_dc is not None and hasattr(_dc, "compile_intention"))
            else []
        )
        experience_id = f"matrix_{hash(str(matrix.sigils))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_matrix_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una matriz simbólica almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for matrix experience"}
        quantum_field = QuantumField()
        manifestations = []
        # Defensive: reality_bytecode may be an OntologicalBytecode wrapper or a raw list
        if hasattr(reality_bytecode, "instructions"):
            instrs = reality_bytecode.instructions or []
        elif isinstance(reality_bytecode, list) or isinstance(reality_bytecode, tuple):
            # Cast runtime list/tuple to the expected instruction list for typing
            instrs = cast(list[OntologicalInstruction], reality_bytecode)
        else:
            instrs = []
        for instr in instrs:
            # Delegate execution only if the runtime exposes an executor
            _eva = getattr(self, "eva_runtime", None)
            if _eva is None or not hasattr(_eva, "execute_instruction"):
                symbol_manifest = None
            else:
                try:
                    symbol_manifest = _eva.execute_instruction(instr, quantum_field)
                except Exception:
                    symbol_manifest = None
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] QuantumCompiler environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
        }

    def add_matrix_experience_phase(
        self, experience_id: str, phase: str, matrix: Matrix, qualia_state: QualiaState
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de matriz simbólica.
        """
        intention = {
            "intention_type": "ARCHIVE_MATRIX_EXPERIENCE",
            "matrix": matrix.to_list(),
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        if _dc is not None and hasattr(_dc, "compile_intention"):
            bytecode = _dc.compile_intention(intention)
        else:
            bytecode = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self._current_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de matriz."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_matrix_experience": self.eva_ingest_matrix_experience,
            "eva_recall_matrix_experience": self.eva_recall_matrix_experience,
            "add_matrix_experience_phase": self.add_matrix_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


#
