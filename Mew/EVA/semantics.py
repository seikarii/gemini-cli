"""
Divine Language Semantic Analyzer - Analizador semántico avanzado
=================================================

Extrae intenciones, significado profundo y diagnóstico contextual de expresiones
del lenguaje divino. Incluye validación, enriquecimiento semántico y trazabilidad.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    RealityBytecode,
)
from crisalida_lib.EVA.typequalia import QualiaState

# Avoid circular import
if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_language_evolved import DivineLanguageEvolved

logger = logging.getLogger(__name__)


@dataclass
class SemanticIntent:
    """
    Intención semántica extraída del análisis.
    """

    intent_type: str
    confidence: float
    entities: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    raw_input: str = ""
    error: str | None = None

    def is_valid(self) -> bool:
        return (
            self.confidence > 0.3
            and not self.error
            and self.intent_type not in ("UNKNOWN_INTENT", "SEMANTIC_ERROR")
        )

    def describe(self) -> str:
        return (
            f"Intent: {self.intent_type} | Entities: {self.entities} | "
            f"Params: {self.parameters} | Confidence: {self.confidence:.2f} | Error: {self.error}"
        )


class SemanticAnalyzer:
    """Analizador semántico para el lenguaje divino, con validación y enriquecimiento."""

    def __init__(self):
        self.intent_types = {
            "QUERY_ENTITY_STATUS": {
                "description": "Consultar el estado de una entidad específica",
                "required_params": ["entity_id"],
                "optional_params": ["detailed", "format"],
            },
            "EXECUTE_PROTOCOL": {
                "description": "Ejecutar un protocolo sobre una entidad",
                "required_params": ["protocol_name", "target_entity"],
                "optional_params": ["parameters", "priority"],
            },
            "CREATE_ENTITY": {
                "description": "Crear una nueva entidad",
                "required_params": ["entity_name"],
                "optional_params": ["entity_type", "initial_state", "parameters"],
            },
            "MODIFY_ENTITY": {
                "description": "Modificar una entidad existente",
                "required_params": ["entity_id", "modification"],
                "optional_params": ["force", "validate"],
            },
            "LIST_ENTITIES": {
                "description": "Listar todas las entidades disponibles",
                "required_params": [],
                "optional_params": ["filter", "sort_by", "limit"],
            },
            "HELP_REQUEST": {
                "description": "Solicitar ayuda o información",
                "required_params": [],
                "optional_params": ["topic", "detail_level"],
            },
        }
        self.global_context: dict[str, Any] = {
            "current_session": None,
            "previous_intents": [],
            "known_entities": set(),
            "active_protocols": set(),
        }
        # runtime container for environment hooks (kept as simple list to avoid import-time typing issues)
        self._environment_hooks = []

    # Class-level annotation to keep type-checkers happy without running code at import-time
    _environment_hooks: list

    def analyze(self, ast: dict[str, Any]) -> dict[str, Any]:
        """
        Analiza semánticamente un AST y extrae la intención enriquecida.
        """
        node_type = ast.get("node_type", "UNKNOWN")
        try:
            if node_type == "QUERY_EXPRESSION":
                return self._analyze_query_expression(ast)
            elif node_type == "EXECUTE_EXPRESSION":
                return self._analyze_execute_expression(ast)
            elif node_type == "CREATE_EXPRESSION":
                return self._analyze_create_expression(ast)
            elif node_type == "MODIFY_EXPRESSION":
                return self._analyze_modify_expression(ast)
            elif node_type == "LIST_EXPRESSION":
                return self._analyze_list_expression(ast)
            elif node_type == "HELP_EXPRESSION":
                return self._analyze_help_expression(ast)
            else:
                return self._create_unknown_intent(ast)
        except Exception as e:
            logger.error(f"Error en análisis semántico: {e}")
            return {
                "type": "SEMANTIC_ERROR",
                "details": {"error": str(e), "ast": ast},
                "confidence": 0.0,
                "context": {},
            }

    def _analyze_query_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        entity_name = ast.get("entity_name")
        if not entity_name:
            return {
                "type": "INVALID_QUERY",
                "details": {"error": "Missing entity name"},
                "confidence": 0.0,
                "context": {},
            }
        intent = {
            "type": "QUERY_ENTITY_STATUS",
            "details": {
                "entity_id": entity_name,
                "query_type": "status",
                "detailed": False,
            },
            "confidence": ast.get("confidence", 0.9),
            "context": {
                "source_ast": ast,
                "complexity": "low",
                "requires_validation": True,
            },
        }
        self.global_context["known_entities"].add(entity_name)
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _analyze_execute_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        protocol_name = ast.get("protocol")
        entity_name = ast.get("entity_name")
        if not protocol_name or not entity_name:
            return {
                "type": "INVALID_EXECUTION",
                "details": {"error": "Missing protocol or entity name"},
                "confidence": 0.0,
                "context": {},
            }
        intent = {
            "type": "EXECUTE_PROTOCOL",
            "details": {
                "protocol_name": protocol_name,
                "target_entity": entity_name,
                "parameters": {},
                "priority": "normal",
            },
            "confidence": ast.get("confidence", 0.9),
            "context": {
                "source_ast": ast,
                "complexity": "medium",
                "requires_authorization": True,
            },
        }
        self.global_context["active_protocols"].add(protocol_name)
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _analyze_create_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        entity_name = ast.get("entity_name")
        if not entity_name:
            return {
                "type": "INVALID_CREATION",
                "details": {"error": "Missing entity name"},
                "confidence": 0.0,
                "context": {},
            }
        intent = {
            "type": "CREATE_ENTITY",
            "details": {
                "entity_name": entity_name,
                "entity_type": "default",
                "initial_state": {"energy": 100.0, "status": "active"},
            },
            "confidence": ast.get("confidence", 0.85),
            "context": {
                "source_ast": ast,
                "complexity": "medium",
                "requires_resources": True,
            },
        }
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _analyze_modify_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        entity_name = ast.get("entity_name")
        modification = ast.get("modification")
        if not entity_name or not modification:
            return {
                "type": "INVALID_MODIFICATION",
                "details": {"error": "Missing entity name or modification"},
                "confidence": 0.0,
                "context": {},
            }
        modification_params = self._parse_modification_text(modification)
        intent = {
            "type": "MODIFY_ENTITY",
            "details": {
                "entity_id": entity_name,
                "modification": modification_params,
                "force": False,
                "validate": True,
            },
            "confidence": ast.get("confidence", 0.8),
            "context": {
                "source_ast": ast,
                "complexity": "high",
                "requires_authorization": True,
            },
        }
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _analyze_list_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        intent = {
            "type": "LIST_ENTITIES",
            "details": {"filter": {}, "sort_by": "name", "limit": None},
            "confidence": ast.get("confidence", 0.95),
            "context": {
                "source_ast": ast,
                "complexity": "low",
                "requires_authorization": False,
            },
        }
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _analyze_help_expression(self, ast: dict[str, Any]) -> dict[str, Any]:
        intent = {
            "type": "HELP_REQUEST",
            "details": {"topic": "general", "detail_level": "basic"},
            "confidence": ast.get("confidence", 1.0),
            "context": {
                "source_ast": ast,
                "complexity": "minimal",
                "requires_authorization": False,
            },
        }
        self.global_context["previous_intents"].append(intent["type"])
        return intent

    def _create_unknown_intent(self, ast: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "UNKNOWN_INTENT",
            "details": {
                "raw_text": ast.get("raw_text", ""),
                "suggestion": "Use 'ayuda' para ver comandos disponibles",
            },
            "confidence": 0.0,
            "context": {"source_ast": ast, "complexity": "unknown"},
        }

    def _parse_modification_text(self, modification_text: str) -> dict[str, Any]:
        """
        Parsea texto de modificación a parámetros estructurados.
        """
        modification_text = modification_text.lower().strip()
        params = {}
        if "energía" in modification_text or "energy" in modification_text:
            params["energy"] = (
                "max"
                if "máxima" in modification_text or "max" in modification_text
                else "increase"
            )
        if "estado" in modification_text or "status" in modification_text:
            if "activo" in modification_text or "active" in modification_text:
                params["status"] = "active"
            elif "inactivo" in modification_text or "inactive" in modification_text:
                params["status"] = "inactive"
            elif "pausa" in modification_text or "pause" in modification_text:
                params["status"] = "paused"
        if "nombre" in modification_text or "name" in modification_text:
            parts = modification_text.split("a")
            if len(parts) > 1:
                params["new_name"] = parts[-1].strip()
        return params

    def get_intent_description(self, intent_type: str) -> str | None:
        return cast(
            str | None, self.intent_types.get(intent_type, {}).get("description")
        )

    def validate_intent_parameters(
        self, intent_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        intent_def = self.intent_types.get(intent_type)
        if not intent_def:
            return {"valid": False, "error": f"Unknown intent type: {intent_type}"}
        required_params = intent_def.get("required_params", [])
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}",
                "required": required_params,
                "provided": list(parameters.keys()),
            }
        return {
            "valid": True,
            "message": "All required parameters provided",
            "optional_params": intent_def.get("optional_params", []),
        }

    def update_context(self, new_context: dict[str, Any]) -> None:
        self.global_context.update(new_context)

    def get_context_summary(self) -> dict[str, Any]:
        return {
            "known_entities_count": len(self.global_context["known_entities"]),
            "active_protocols_count": len(self.global_context["active_protocols"]),
            "previous_intents_count": len(self.global_context["previous_intents"]),
            "recent_intents": self.global_context["previous_intents"][-5:],
        }


class EVASemanticAnalyzer(SemanticAnalyzer):
    """
    Analizador semántico extendido para integración con EVA.
    Permite compilar intenciones semánticas en RealityBytecode, almacenarlas y simular su ejecución en el QuantumField.
    Soporta faseo, hooks de entorno y recall activo de experiencias semánticas.
    """

    def __init__(self, phase: str = "default"):
        super().__init__()
        self.eva_runtime = LivingSymbolRuntime()
        self.divine_compiler = DivineLanguageEvolved(None)
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._current_phase: str = phase
        self._environment_hooks: list = []

    def eva_ingest_semantic_intent(
        self,
        semantic_intent: SemanticIntent,
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compila una intención semántica como experiencia y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_SEMANTIC_INTENT_EXPERIENCE",
            "semantic_intent": semantic_intent.describe(),
            "parameters": semantic_intent.parameters,
            "entities": semantic_intent.entities,
            "context": semantic_intent.context,
            "raw_input": semantic_intent.raw_input,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
        experience_id = f"semantic_intent_{hash(semantic_intent.describe())}"
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

    def eva_recall_semantic_intent(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una intención semántica almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for semantic intent"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.warning(
                            f"[EVA] SemanticIntent environment hook failed: {e}"
                        )
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

    def add_semantic_intent_phase(
        self,
        experience_id: str,
        phase: str,
        semantic_intent: SemanticIntent,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de intención semántica.
        """
        intention = {
            "intention_type": "ARCHIVE_SEMANTIC_INTENT_EXPERIENCE",
            "semantic_intent": semantic_intent.describe(),
            "parameters": semantic_intent.parameters,
            "entities": semantic_intent.entities,
            "context": semantic_intent.context,
            "raw_input": semantic_intent.raw_input,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.divine_compiler.compile_intention(intention)
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
                logger.warning(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de intención semántica."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_semantic_intent": self.eva_ingest_semantic_intent,
            "eva_recall_semantic_intent": self.eva_recall_semantic_intent,
            "add_semantic_intent_phase": self.add_semantic_intent_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
