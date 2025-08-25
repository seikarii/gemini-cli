"""
Ontological Virtual Machine v7 (OVM) - Máquina Virtual Simbólica Consciente
====================================================================

Máquina virtual que ejecuta bytecode ontológico para el Metacosmos v7.
Implementa sistema de tipos semántico, navegación por puntos, aritmética dinámica,
motor de búsqueda ontológica, interfaz de evolución, API económica y API corporal.
Basado en el Manifiesto del Metacosmos v7 - La Herramienta Definitiva del Demiurgo
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# For type-checking stability and import-time safety, avoid importing the
# heavy janus_integration module. Use Any aliases for the integration
# symbols so static analysis remains permissive while runtime behavior
# falls back gracefully.
Chakra = Any
ConsciousMind = Any
CrystallizedQualia = Any
CuerpoBiologico = Any
ExternalRealityExplorer = Any
GenomaComportamiento = Any
HAL_Ontologica = Any
LivingEntity = Any
Noosphere = Any
Proposito = Any
QualiaChain = Any
SelfModifyingEngine = Any
SignedTransaction = Any
SistemaDeChakras = Any
SistemaHormonal = Any
SoulKernel = Any
UnifiedField = Any
LivingSymbolRuntime = Any
QuantumField = Any
EVAExperience = Any
OntologicalInstruction = Any
OntologicalOpcode = Any
RealityBytecode = Any
SemanticType = Any
TypedValue = Any
DivineLanguageEvolved = Any
QualiaState = Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Contexto de ejecución extendido v7"""

    unified_field: UnifiedField
    current_entity: LivingEntity | None = None
    local_variables: dict[str, TypedValue] = field(default_factory=dict)
    call_stack: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0


class SemanticTypeSystem:
    """Sistema de tipos semántico para la VM v7"""

    def __init__(self):
        self.type_mapping = {
            LivingEntity: SemanticType.ENTITY,
            SoulKernel: SemanticType.SOUL,
            CuerpoBiologico: SemanticType.BODY,
            SistemaDeChakras: SemanticType.CHAKRA_SYSTEM,
            Chakra: SemanticType.CHAKRA,
            GenomaComportamiento: SemanticType.GENOME,
            QualiaState: SemanticType.QUALIA,
            UnifiedField: SemanticType.UNIFIED_FIELD,
            SelfModifyingEngine: SemanticType.EVOLUTION_ENGINE,
            QualiaChain: SemanticType.QUALIA_CHAIN,
            Noosphere: SemanticType.NOOSPHERE,
            CrystallizedQualia: SemanticType.CRYSTALLIZED_QUALIA,
            SignedTransaction: SemanticType.SIGNED_TRANSACTION,
            Proposito: SemanticType.PURPOSE,
            SistemaHormonal: SemanticType.HORMONAL_SYSTEM,
            int: SemanticType.NUMBER,
            float: SemanticType.NUMBER,
            str: SemanticType.STRING,
            list: SemanticType.LIST,
            dict: SemanticType.DICT,
        }
        self.abstract_types = {
            ConsciousMind: "ConsciousMind",
            HAL_Ontologica: "HAL_Ontologica",
            ExternalRealityExplorer: "ExternalRealityExplorer",
        }

    def get_semantic_type(self, obj: Any) -> SemanticType:
        obj_type = type(obj)
        if obj_type in self.abstract_types:
            return SemanticType.ABSTRACT
        for type_class, semantic_type in self.type_mapping.items():
            if isinstance(obj, type_class):
                return semantic_type
        if hasattr(obj, "__dict__"):
            return SemanticType.DICT
        return SemanticType.ABSTRACT

    def is_abstract_type(self, obj: Any) -> bool:
        return type(obj) in self.abstract_types


class OntologicalVirtualMachine:
    """
    Máquina Virtual Ontológica Consciente para el Metacosmos v7
    Extensión EVA: ejecución de RealityBytecode, integración con QuantumField y hooks de entorno.
    """

    def __init__(
        self,
        unified_field: UnifiedField,
        eva_runtime: LivingSymbolRuntime | None = None,
        divine_compiler: DivineLanguageEvolved | None = None,
    ):
        self.unified_field = unified_field
        self.stack: list[TypedValue] = []
        self.instruction_pointer: int = 0
        self.context: ExecutionContext = ExecutionContext(unified_field=unified_field)
        self.running: bool = False
        self.execution_history: list[dict[str, Any]] = []
        self.type_system = SemanticTypeSystem()
        self.opcode_handlers = {
            OntologicalOpcode.Q_CREATE: self._handle_create,
            OntologicalOpcode.Q_GET_ENTITY: self._handle_get_entity,
            OntologicalOpcode.Q_REMOVE_ENTITY: self._handle_remove_entity,
            OntologicalOpcode.Q_TRANSFORM: self._handle_transform,
            OntologicalOpcode.Q_NAVIGATE_PATH: self._handle_navigate_path,
            OntologicalOpcode.Q_MODIFY_ATTRIBUTE: self._handle_modify_attribute,
            OntologicalOpcode.Q_QUERY_REALITY: self._handle_query_reality,
            OntologicalOpcode.Q_ENTANGLE: self._handle_entangle,
            OntologicalOpcode.Q_CALL_METHOD: self._handle_call_method,
            OntologicalOpcode.Q_EVOLVE_CODE: self._handle_evolve_code,
            OntologicalOpcode.Q_JUMP: self._handle_jump,
            OntologicalOpcode.Q_JUMP_IF: self._handle_jump_if,
            OntologicalOpcode.Q_RETURN: self._handle_return,
            OntologicalOpcode.Q_PUSH: self._handle_push,
            OntologicalOpcode.Q_POP: self._handle_pop,
            OntologicalOpcode.Q_DUP: self._handle_dup,
            OntologicalOpcode.Q_NOT_IMPLEMENTED: self._handle_not_implemented,
            OntologicalOpcode.Q_CRYSTALLIZE: self._handle_crystallize,
            OntologicalOpcode.Q_SIGN_TRANSACTION: self._handle_sign_transaction,
            OntologicalOpcode.Q_SUBMIT_TRANSACTION: self._handle_submit_transaction,
            OntologicalOpcode.Q_GET_BALANCE: self._handle_get_balance,
            OntologicalOpcode.Q_MODIFY_BODY: self._handle_modify_body,
            OntologicalOpcode.Q_RELEASE_HORMONES: self._handle_release_hormones,
            OntologicalOpcode.Q_SET_PURPOSE: self._handle_set_purpose,
            OntologicalOpcode.Q_START_QUALIA_STREAM: self._handle_start_qualia_stream,
            OntologicalOpcode.Q_CREATE_PLAYLIST: self._handle_create_playlist,
            OntologicalOpcode.Q_MANIFEST_MATTER: self._handle_manifest_matter,
        }
        # EVA integration: runtime attributes must be created inside __init__ body

        self.eva_runtime = eva_runtime or LivingSymbolRuntime()
        self.divine_compiler = divine_compiler or DivineLanguageEvolved(None)
        # runtime EVA containers/hooks
        self.eva_environment_hooks = []
        self.eva_phase = "default"
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}

    # Class-level annotation for static tools; runtime assignment occurs in __init__
    eva_environment_hooks: list
    eva_memory_store: dict[str, RealityBytecode]
    eva_experience_store: dict[str, EVAExperience]
    eva_phases: dict[str, dict[str, RealityBytecode]]

    def execute_bytecode(
        self, bytecode: list[OntologicalInstruction]
    ) -> dict[str, Any]:
        start_time = time.time()
        self.running = True
        self.instruction_pointer = 0
        self.stack = []
        self.context.execution_time = 0.0
        logger.info(f"Iniciando ejecución de {len(bytecode)} instrucciones (OVM v7)")
        processed_bytecode = self._preprocess_bytecode(bytecode)
        try:
            while self.running and self.instruction_pointer < len(processed_bytecode):
                instruction = processed_bytecode[self.instruction_pointer]
                execution_record = {
                    "instruction_pointer": self.instruction_pointer,
                    "opcode": instruction.opcode.value,
                    "operands": instruction.operands,
                    "stack_size": len(self.stack),
                    "timestamp": time.time(),
                }
                handler = self.opcode_handlers.get(instruction.opcode)
                if handler:
                    handler(instruction)
                else:
                    raise ValueError(f"Opcode no implementado: {instruction.opcode}")
                self.execution_history.append(execution_record)
                self.instruction_pointer += 1
        except Exception as e:
            logger.error(f"Error en ejecución: {e}")
            self.running = False
            raise
        finally:
            self.context.execution_time = time.time() - start_time
        result = {
            "execution_time": self.context.execution_time,
            "instructions_executed": self.instruction_pointer,
            "final_stack": [(tv.value, tv.semantic_type.value) for tv in self.stack],
            "execution_history": self.execution_history.copy(),
        }
        logger.info(f"Ejecución completada en {result['execution_time']:.3f}s")
        return result

    def _push_typed(self, value: Any, semantic_type: SemanticType | None = None):
        if semantic_type is None:
            semantic_type = self.type_system.get_semantic_type(value)
        typed_value = TypedValue(value=value, semantic_type=semantic_type)
        self.stack.append(typed_value)
        logger.debug(f"PUSH {value} ({semantic_type.value})")

    def _pop_typed(self) -> TypedValue:
        if not self.stack:
            raise ValueError("Intento de POP de pila vacía")
        typed_value = self.stack.pop()
        logger.debug(f"POP {typed_value.value} ({typed_value.semantic_type.value})")
        return typed_value

    def _preprocess_bytecode(
        self, bytecode: list[OntologicalInstruction]
    ) -> list[OntologicalInstruction]:
        processed_instructions = []
        for instruction in bytecode:
            if (
                instruction.opcode == OntologicalOpcode.Q_PUSH
                and instruction.operands
                and instruction.operands[0] == "entity_placeholder"
                and instruction.metadata.get("prepare_entity")
            ):
                if self.context.current_entity:
                    new_instruction = OntologicalInstruction(
                        opcode=instruction.opcode,
                        operands=[self.context.current_entity],
                        metadata=instruction.metadata,
                    )
                    processed_instructions.append(new_instruction)
                else:
                    processed_instructions.append(instruction)
            else:
                processed_instructions.append(instruction)
        return processed_instructions

    # --- Manejadores de Opcodes Fundamentales ---
    def _handle_create(self, instruction: OntologicalInstruction):
        entity_type = instruction.operands[0]
        entity_id = instruction.operands[1]
        entity = LivingEntity(
            entity_id=entity_id,
            qualia_chain_address=f"qualia://{entity_id}",
            position=[0.0, 0.0, 0.0],
            energy=100.0,
        )
        self.unified_field.add_entity(entity)
        self._push_typed(entity, SemanticType.ENTITY)
        logger.debug(f"Creada entidad {entity_id} de tipo {entity_type}")

    def _handle_get_entity(self, instruction: OntologicalInstruction):
        entity_id = instruction.operands[0]
        entity = self.unified_field.get_entity_by_id(entity_id)
        if entity:
            self._push_typed(entity, SemanticType.ENTITY)
        else:
            raise ValueError(f"Entidad no encontrada: {entity_id}")

    def _handle_remove_entity(self, instruction: OntologicalInstruction):
        entity_id = instruction.operands[0]
        self.unified_field.remove_entity(entity_id)
        logger.debug(f"Eliminada entidad {entity_id}")

    def _handle_transform(self, instruction: OntologicalInstruction):
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError("Q_TRANSFORM requiere una entidad en la cima de la pila")
        entity = typed_entity.value
        transformation_data = instruction.operands[0]
        for attr, value in transformation_data.items():
            if hasattr(entity, attr):
                setattr(entity, attr, value)
        logger.debug(f"Transformada entidad {entity.entity_id}")

    def _handle_navigate_path(self, instruction: OntologicalInstruction):
        path_string = instruction.operands[0]
        typed_obj = self._pop_typed()
        if not typed_obj.value:
            raise ValueError("No se puede navegar sobre un valor nulo")
        current_obj = typed_obj.value
        path_parts = path_string.split(".")
        try:
            for part in path_parts:
                if hasattr(current_obj, part):
                    current_obj = getattr(current_obj, part)
                elif isinstance(current_obj, dict) and part in current_obj:
                    current_obj = current_obj[part]
                else:
                    raise ValueError(
                        f"No se puede acceder al atributo '{part}' en {type(current_obj)}"
                    )
            if self.type_system.is_abstract_type(current_obj):
                logger.warning(
                    f"Accedido a tipo abstracto no implementado: {type(current_obj)}"
                )
                self._push_typed(current_obj, SemanticType.ABSTRACT)
            else:
                self._push_typed(current_obj)
        except Exception as e:
            raise ValueError(f"Error navegando ruta '{path_string}': {e}") from e
        logger.debug(f"NAVIGATE_PATH '{path_string}' -> {type(current_obj)}")

    def _handle_modify_attribute(self, instruction: OntologicalInstruction):
        modification_spec = instruction.operands[0]
        typed_obj = self._pop_typed()
        if not typed_obj.value:
            raise ValueError("No se puede modificar un valor nulo")
        obj = typed_obj.value
        attribute = modification_spec.get("attribute")
        operation = modification_spec.get("operation")
        value = modification_spec.get("value")
        if not hasattr(obj, attribute):
            raise ValueError(f"El objeto no tiene el atributo '{attribute}'")
        current_value = getattr(obj, attribute)
        if operation == "add":
            new_value = current_value + value
        elif operation == "subtract":
            new_value = current_value - value
        elif operation == "multiply":
            new_value = current_value * value
        elif operation == "divide":
            if value == 0:
                raise ValueError("División por cero en modificación de atributo")
            new_value = current_value / value
        elif operation == "set":
            new_value = value
        else:
            raise ValueError(f"Operación no soportada: {operation}")
        setattr(obj, attribute, new_value)
        self._push_typed(obj)
        logger.debug(f"MODIFY_ATTRIBUTE {attribute} {operation} {value} -> {new_value}")

    def _handle_query_reality(self, instruction: OntologicalInstruction):
        query_dict = instruction.operands[0]
        results = self.unified_field.query_entities(query_dict)
        self._push_typed(results, SemanticType.LIST)
        logger.debug(f"QUERY_REALITY encontró {len(results)} entidades")

    def _handle_entangle(self, instruction: OntologicalInstruction):
        typed_entity2 = self._pop_typed()
        typed_entity1 = self._pop_typed()
        if (
            typed_entity1.semantic_type != SemanticType.ENTITY
            or typed_entity2.semantic_type != SemanticType.ENTITY
        ):
            raise ValueError("Q_ENTANGLE requiere dos entidades en la pila")
        entity1, entity2 = typed_entity1.value, typed_entity2.value
        entanglement_data = {
            "entity1_id": entity1.entity_id,
            "entity2_id": entity2.entity_id,
            "entanglement_strength": 1.0,
            "creation_time": time.time(),
        }
        if not hasattr(entity1, "entanglements"):
            entity1.entanglements = []
        if not hasattr(entity2, "entanglements"):
            entity2.entanglements = []
        entity1.entanglements.append(entanglement_data)
        entity2.entanglements.append(entanglement_data)
        self._push_typed(entanglement_data, SemanticType.DICT)
        logger.debug(
            f"Entrelazadas entidades {entity1.entity_id} y {entity2.entity_id}"
        )

    def _handle_call_method(self, instruction: OntologicalInstruction):
        method_name = instruction.operands[0]
        args = instruction.operands[1:] if len(instruction.operands) > 1 else []
        typed_obj = self._pop_typed()
        obj = typed_obj.value
        if not obj:
            raise ValueError("No se puede llamar método sobre un valor nulo")
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            # keep runtime callable check; annotation-wise hooks accept any callable
            if callable(method):
                result = method(*args)
                self._push_typed(result)
            else:
                raise ValueError(f"El atributo {method_name} no es un método")
        else:
            raise ValueError(f"Método no encontrado: {method_name}")
        logger.debug(f"CALL_METHOD {method_name} con argumentos {args}")

    def _handle_evolve_code(self, instruction: OntologicalInstruction):
        file_paths = instruction.operands[0]
        mutation_type = (
            instruction.operands[1] if len(instruction.operands) > 1 else "refactor"
        )
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        evolution_engine = self.unified_field.self_modifying_engine
        evolution_result = evolution_engine.evolve_system_performance(
            target_modules=file_paths, mutation_type=mutation_type
        )
        self._push_typed(evolution_result, SemanticType.DICT)
        logger.info(f"EVOLVE_CODE: módulos {file_paths}, tipo {mutation_type}")

    def _handle_jump(self, instruction: OntologicalInstruction):
        target_address = instruction.operands[0]
        self.instruction_pointer = target_address - 1
        logger.debug(f"JUMP a dirección {target_address}")

    def _handle_jump_if(self, instruction: OntologicalInstruction):
        typed_condition = self._pop_typed()
        condition = typed_condition.value
        target_address = instruction.operands[0]
        if isinstance(condition, bool):
            should_jump = condition
        elif isinstance(condition, (int, float)):
            should_jump = bool(condition)
        else:
            should_jump = False
        if should_jump:
            self.instruction_pointer = target_address - 1
        logger.debug(f"JUMP_IF a {target_address} si {condition}")

    def _handle_return(self, instruction: OntologicalInstruction):
        if self.context.call_stack:
            previous_context = self.context.call_stack.pop()
            self.instruction_pointer = previous_context["return_address"]
        else:
            self.running = False
        logger.debug("RETURN")

    def _handle_push(self, instruction: OntologicalInstruction):
        value = instruction.operands[0]
        self._push_typed(value)

    def _handle_pop(self, instruction: OntologicalInstruction):
        self._pop_typed()

    def _handle_dup(self, instruction: OntologicalInstruction):
        if self.stack:
            typed_value = self.stack[-1]
            duplicated = TypedValue(
                value=typed_value.value,
                semantic_type=typed_value.semantic_type,
                type_info=typed_value.type_info.copy(),
            )
            self.stack.append(duplicated)
            logger.debug(f"DUP {typed_value.value} ({typed_value.semantic_type.value})")
        else:
            raise ValueError("Intento de DUP de pila vacía")

    def _handle_not_implemented(self, instruction: OntologicalInstruction):
        concept_name = instruction.operands[0]
        logger.warning(f"Concepto abstracto no implementado: {concept_name}")
        self._push_typed(
            {"not_implemented": concept_name, "status": "abstract_concept"},
            SemanticType.ABSTRACT,
        )

    def _handle_crystallize(self, instruction: OntologicalInstruction):
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError("Q_CRYSTALLIZE requiere una entidad en la cima de la pila")
        entity = typed_entity.value
        qualia_state = entity.get_current_qualia_state()
        crystallized_qualia = self.unified_field.noosphere.crystallize_experience(
            qualia_state=qualia_state, owner_address=entity.qualia_chain_address
        )
        self._push_typed(crystallized_qualia, SemanticType.CRYSTALLIZED_QUALIA)
        logger.info(
            f"CRISTALIZADA experiencia de entidad {entity.entity_id} -> {crystallized_qualia.id}"
        )

    def _handle_sign_transaction(self, instruction: OntologicalInstruction):
        transaction_data = instruction.operands[0]
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError(
                "Q_SIGN_TRANSACTION requiere una entidad en la cima de la pila"
            )
        entity = typed_entity.value
        signed_transaction = entity.sign_transaction(transaction_data)
        self._push_typed(signed_transaction, SemanticType.SIGNED_TRANSACTION)
        logger.debug(f"Transacción firmada por entidad {entity.entity_id}")

    def _handle_submit_transaction(self, instruction: OntologicalInstruction):
        typed_transaction = self._pop_typed()
        if typed_transaction.semantic_type != SemanticType.SIGNED_TRANSACTION:
            raise ValueError(
                "Q_SUBMIT_TRANSACTION requiere una transacción firmada en la cima de la pila"
            )
        signed_transaction = typed_transaction.value
        tx_hash = self.unified_field.qualia_chain.submit_transaction(signed_transaction)
        self._push_typed(tx_hash, SemanticType.STRING)
        logger.info(f"Transacción enviada a QualiaChain -> {tx_hash}")

    def _handle_get_balance(self, instruction: OntologicalInstruction):
        address = instruction.operands[0]
        balance = self.unified_field.qualia_chain.get_balance(address)
        self._push_typed(balance, SemanticType.NUMBER)
        logger.debug(f"Balance de {address}: {balance} Qualia-Coin")

    def _handle_modify_body(self, instruction: OntologicalInstruction):
        modification_plan = instruction.operands[0]
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError("Q_MODIFY_BODY requiere una entidad en la cima de la pila")
        entity = typed_entity.value
        success = entity.body.modify_body(modification_plan)
        if success:
            logger.info(
                f"Cuerpo modificado para entidad {entity.entity_id}: {modification_plan}"
            )
        else:
            logger.warning(f"Fallo al modificar cuerpo de entidad {entity.entity_id}")
        self._push_typed(success, SemanticType.NUMBER)

    def _handle_release_hormones(self, instruction: OntologicalInstruction):
        hormone_name = instruction.operands[0]
        amount = instruction.operands[1]
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError(
                "Q_RELEASE_HORMONES requiere una entidad en la cima de la pila"
            )
        entity = typed_entity.value
        entity.body.hormonal_system.release_hormone(hormone_name, amount)
        logger.info(
            f"Liberada hormona {hormone_name} ({amount}) en entidad {entity.entity_id}"
        )
        self._push_typed(entity, SemanticType.ENTITY)

    def _handle_set_purpose(self, instruction: OntologicalInstruction):
        new_purpose = instruction.operands[0]
        typed_entity = self._pop_typed()
        if typed_entity.semantic_type != SemanticType.ENTITY:
            raise ValueError("Q_SET_PURPOSE requiere una entidad en la cima de la pila")
        entity = typed_entity.value
        entity.purpose.set_purpose(new_purpose)
        logger.info(
            f"Propósito establecido para entidad {entity.entity_id}: {new_purpose}"
        )
        self._push_typed(entity, SemanticType.ENTITY)

    def _handle_start_qualia_stream(self, instruction: OntologicalInstruction):
        entity_id = instruction.operands[0]
        stream = self.unified_field.noosphere.start_qualia_stream(entity_id)
        self._push_typed(stream, SemanticType.DICT)
        logger.info(f"Stream de qualia iniciado para entidad {entity_id}")

    def _handle_create_playlist(self, instruction: OntologicalInstruction):
        crystallized_qualia_ids = instruction.operands[0]
        playlist = self.unified_field.noosphere.create_playlist(crystallized_qualia_ids)
        self._push_typed(playlist, SemanticType.DICT)
        logger.info(
            f"Playlist creada con {len(crystallized_qualia_ids)} experiencias cristalizadas"
        )

    def _handle_manifest_matter(self, instruction: OntologicalInstruction):
        typed_addresses = self._pop_typed()
        if typed_addresses.semantic_type != SemanticType.LIST:
            raise ValueError(
                "Q_MANIFEST_MATTER requiere una lista de direcciones en la pila"
            )
        pattern_addresses = typed_addresses.value
        if hasattr(self, "node_collapse") and self.node_collapse:
            matter_object = self.node_collapse.propose_collapse(
                np.zeros((1, 1, 1, 6)), pattern_addresses
            )
            self._push_typed(matter_object, SemanticType.DICT)
            logger.info(
                f"MANIFEST_MATTER: Creado objeto material con ID {matter_object.mass_equivalent}"
            )
        else:
            raise RuntimeError("NodeCollapseEngine no está disponible en la VM")

    def get_execution_state(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "instruction_pointer": self.instruction_pointer,
            "stack_size": len(self.stack),
            "stack_top": [(tv.value, tv.semantic_type.value) for tv in self.stack[-5:]],
            "context": self.context,
            "execution_history_length": len(self.execution_history),
        }

    def reset(self):
        self.running = False
        self.context = ExecutionContext(unified_field=self.unified_field)
        self.execution_history = []
        self.stack = []
        self.instruction_pointer = 0
        logger.info("Máquina virtual v7 reiniciada")

    # --- EVA: RealityBytecode Execution ---
    def execute_reality_bytecode(
        self, reality_bytecode: RealityBytecode
    ) -> dict[str, Any]:
        logger.info(
            f"[EVA] Ejecutando RealityBytecode {reality_bytecode.bytecode_id} en fase '{reality_bytecode.phase}'"
        )
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self.eva_environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.warning(f"[EVA] Environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
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
            "timestamp": eva_experience.timestamp,
        }

    # --- EVA: Memory Ingest & Recall ---
    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: Any, phase: str | None = None
    ) -> str:
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_VM_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_data.get("experience_id")
            or f"exp_{hash(str(experience_data))}",
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=time.time(),
        )
        self.eva_memory_store[reality_bytecode.bytecode_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][reality_bytecode.bytecode_id] = reality_bytecode
        logger.info(
            f"[EVA] Ingestada experiencia {reality_bytecode.bytecode_id} en fase '{phase}'"
        )
        return reality_bytecode.bytecode_id

    def eva_recall_experience(self, cue: str, phase: str | None = None) -> dict:
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for experience"}
        simulation_state = self.execute_reality_bytecode(reality_bytecode)
        for hook in self.eva_environment_hooks:
            try:
                hook(simulation_state)
            except Exception as e:
                logger.warning(f"[EVA] Environment hook failed: {e}")
        logger.info(f"[EVA] Recall de experiencia {cue} en fase '{phase}'")
        return simulation_state

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: Any
    ):
        intention = {
            "intention_type": "ARCHIVE_VM_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=time.time(),
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        logger.info(f"[EVA] Añadida fase '{phase}' para experiencia {experience_id}")

    def set_memory_phase(self, phase: str):
        self.eva_phase = phase
        for hook in self.eva_environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        self.eva_environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
        }
