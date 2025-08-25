"""
Interfaz del Demiurgo - El punto de entrada para la interacción del creador.

Permite inicializar el estado del Demiurgo, compilar y ejecutar intenciones ontológicas,
y manifestar materia en el universo simbólico usando la VM y el compilador cuántico.
"""

import asyncio
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbol
from crisalida_lib.EDEN.qualia_engine import PhysicsEngine
from crisalida_lib.EVA.quantum_compiler import CompilationResultV7, FullQuantumCompiler
from crisalida_lib.EVA.symbolic_vm import (
    OntologicalInstruction,
    OntologicalOpcode,
    OntologicalVirtualMachine,
)
from crisalida_lib.EVA.types import QualiaState
from crisalida_lib.utils.eva_memory_mixin import EVAMemoryMixin


class DemiurgeInterface:
    """
    Interfaz principal para la interacción del Demiurgo.
    Permite compilar intenciones, ejecutar bytecode ontológico y manifestar materia.
    """

    def __init__(
        self,
        manifold: Any,
        grammar: Any,
        compiler: FullQuantumCompiler,
        vm: OntologicalVirtualMachine,
        physics_engine: PhysicsEngine,
    ):
        self.manifold = manifold
        self.grammar = grammar
        self.compiler = compiler
        self.vm = vm
        self.physics_engine = physics_engine
        self.demiurge_state: dict[str, Any] = None
        self.demiurge_entity: LivingSymbol = None

    def initialize_demiurge(self, demiurge_id: str) -> dict[str, Any]:
        """
        Inicializa el estado del Demiurgo y su representación en el universo.
        """
        self.demiurge_state = {
            "entity_id": demiurge_id,
            "qualia_state": QualiaState(),
            "abilities": ["compile", "execute", "manifest"],
        }
        position = (25, 25, 25)  # Posición central inicial
        self.demiurge_entity = self.manifold.create_living_symbol(position, "phi")
        print(f"Demiurgo {demiurge_id} inicializado.")
        return self.demiurge_state

    def compile_and_execute_intention(
        self, intention_matrix: list[list[str]]
    ) -> dict[str, Any]:
        """
        Usa el compilador completo y la VM para ejecutar una intención.
        """
        if not self.demiurge_state:
            raise Exception("Demiurgo no inicializado")
        print(f"Compilando intención del Demiurgo: {intention_matrix}")
        # La entidad del demiurgo se pasa al compilador para contexto
        compilation_result: CompilationResultV7 = asyncio.run(
            self.compiler.compile_matrix(intention_matrix, self.demiurge_entity)
        )
        if not compilation_result.success:
            print(f"ERROR de compilación: {compilation_result.errors}")
            return {
                "error": "Falló la compilación",
                "details": compilation_result.errors,
            }
        print(f"Ejecutando {len(compilation_result.bytecode)} instrucciones...")
        execution_result = self.vm.execute_bytecode(compilation_result.bytecode)
        return execution_result

    def manifest_matter(
        self, intention_matrix: list[list[str]], pattern_addresses: list[tuple]
    ) -> dict[str, Any]:
        """
        Manifiesta materia usando el opcode formal.
        """
        if not self.demiurge_state:
            return {"error": "Demiurgo no inicializado"}
        # Primero, se empuja a la pila las direcciones del patrón a colapsar
        push_instruction = OntologicalInstruction(
            opcode=OntologicalOpcode.Q_PUSH, operands=[pattern_addresses]
        )
        # Luego, se añade la instrucción de manifestación
        manifest_instruction = OntologicalInstruction(
            opcode=OntologicalOpcode.Q_MANIFEST_MATTER, operands=[], metadata={}
        )
        bytecode = [push_instruction, manifest_instruction]
        print(
            f"Ejecutando manifestación de materia en {len(pattern_addresses)} puntos."
        )
        result = self.vm.execute_bytecode(bytecode)
        return result


class EVADemiurgeInterface(DemiurgeInterface, EVAMemoryMixin):
    """
    Interfaz avanzada del Demiurgo para integración con EVA.
    Permite compilar, almacenar, simular y recordar intenciones y manifestaciones como experiencias vivientes (RealityBytecode),
    soporta faseo, hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(
        self,
        manifold: Any,
        grammar: Any,
        compiler: FullQuantumCompiler,
        vm: OntologicalVirtualMachine,
        physics_engine: PhysicsEngine,
        phase: str | None = "default",
    ):
        DemiurgeInterface.__init__(
            self, manifold, grammar, compiler, vm, physics_engine
        )
        EVAMemoryMixin.__init__(self, eva_phase=phase)

    def initialize_demiurge(self, demiurge_id: str) -> dict[str, Any]:
        state = super().initialize_demiurge(demiurge_id)
        state["eva_phase"] = self.eva_phase
        return state

    def compile_and_execute_intention(
        self,
        intention_matrix: list[list[str]],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """
        Compila y ejecuta una intención, registrando la experiencia en la memoria EVA.
        """
        execution_result = super().compile_and_execute_intention(intention_matrix)

        experience_data = {
            "demiurge_id": self.demiurge_state["entity_id"],
            "intention_matrix": intention_matrix,
            "execution_result": execution_result,
            "timestamp": execution_result.get("timestamp", 0.0),
        }

        self.eva_ingest_experience(
            intention_type="ARCHIVE_DEMIURGE_INTENTION_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

        return execution_result

    def manifest_matter(
        self,
        intention_matrix: list[list[str]],
        pattern_addresses: list[tuple],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """
        Manifiesta materia y registra la experiencia en la memoria EVA.
        """
        result = super().manifest_matter(intention_matrix, pattern_addresses)

        experience_data = {
            "demiurge_id": self.demiurge_state["entity_id"],
            "intention_matrix": intention_matrix,
            "pattern_addresses": pattern_addresses,
            "manifest_result": result,
            "timestamp": result.get("timestamp", 0.0),
        }

        self.eva_ingest_experience(
            intention_type="ARCHIVE_DEMIURGE_MANIFEST_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

        return result
