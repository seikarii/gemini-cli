"""
Divine Language Parser - Analizador sintáctico para lenguaje divino
==============================================================

Implementa el análisis léxico y sintáctico para transformar cadenas
de entrada en estructuras de datos AST (Abstract Syntax Tree).
Incluye diagnóstico avanzado, sugerencias y trazabilidad.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ASTNode:
    """Nodo del Árbol de Sintaxis Abstracta"""

    node_type: str
    value: Any = None
    children: list["ASTNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: str | None = None


class DivineParser:
    """Parser principal para el lenguaje divino, con diagnóstico y sugerencias.
    Extendido para integración con EVA: generación de AST enriquecido, hooks de entorno, faseo y simulación.
    """

    def __init__(self, eva_runtime: Any | None = None, phase: str = "default") -> None:
        self.patterns = {
            "query_entity": [
                r"consulta\s+(?:el\s+)?estado\s+(?:de\s+la\s+)?entidad\s+(\w+)",
                r"muestra\s+(?:el\s+)?estado\s+(?:de\s+la\s+)?entidad\s+(\w+)",
                r"ver\s+(?:el\s+)?estado\s+(?:de\s+la\s+)?entidad\s+(\w+)",
            ],
            "execute_protocol": [
                r"ejecuta\s+(?:el\s+)?protocolo\s+(\w+)\s+(?:sobre|en)\s+(?:la\s+)?entidad\s+(\w+)",
                r"corre\s+(?:el\s+)?protocolo\s+(\w+)\s+(?:sobre|en)\s+(?:la\s+)?entidad\s+(\w+)",
                r"inicia\s+(?:el\s+)?protocolo\s+(\w+)\s+(?:sobre|en)\s+(?:la\s+)?entidad\s+(\w+)",
            ],
            "create_entity": [
                r"crea\s+(?:una\s+)?nueva\s+entidad\s+(\w+)",
                r"genera\s+(?:una\s+)?nueva\s+entidad\s+(\w+)",
                r"construye\s+(?:una\s+)?nueva\s+entidad\s+(\w+)",
            ],
            "modify_entity": [
                r"modifica\s+(?:la\s+)?entidad\s+(\w+)\s+(.+)",
                r"cambia\s+(?:la\s+)?entidad\s+(\w+)\s+(.+)",
                r"actualiza\s+(?:la\s+)?entidad\s+(\w+)\s+(.+)",
            ],
            "list_entities": [
                r"lista\s+(?:todas\s+las\s+)?entidades?",
                r"muestra\s+(?:todas\s+las\s+)?entidades?",
                r"ver\s+(?:todas\s+las\s+)?entidades?",
            ],
            "help": [r"ayuda", r"help", r"comandos", r"¿qué\s+puedes\s+hacer?"],
        }
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.patterns.items()
        }
        self.suggestion_map = {
            "query_entity": "Ejemplo: consulta estado de entidad E123",
            "execute_protocol": "Ejemplo: ejecuta protocolo P1 sobre entidad E123",
            "create_entity": "Ejemplo: crea nueva entidad MiEntidad",
            "modify_entity": "Ejemplo: modifica entidad E123 atributo=valor",
            "list_entities": "Ejemplo: lista entidades",
            "help": "Ejemplo: ayuda",
        }
        # EVA: integración de memoria viviente y simulación
        self.eva_runtime = eva_runtime
        self.phase = phase
        # Ensure environment hooks container exists for EVA runtime integrations
        self._environment_hooks = []  # type: list[Callable[..., Any]]

    # class-level annotation to help type-checkers without executing at import time
    _environment_hooks: list[Callable[..., Any]]

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parsea el texto de entrada y genera un AST extendido.
        Si EVA está activo, genera un AST enriquecido con metadata para simulación.
        """
        text = text.strip()
        if not text:
            return {
                "node_type": "EMPTY",
                "error": "Empty input",
                "confidence": 0.0,
                "suggestion": "Proporcione un comando válido.",
            }
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    ast = self._build_ast_node(category, match, text)
                    # EVA: Enriquecer AST con metadata de simulaci3n si runtime activo
                    _eva = getattr(self, "eva_runtime", None)
                    if _eva:
                        ast["eva_phase"] = self.phase
                        ast["eva_hooks"] = [
                            hook.__name__
                            for hook in self._environment_hooks
                            if hasattr(hook, "__name__")
                        ]
                        ast["eva_simulation"] = self._simulate_ast(ast)
                    return ast
        suggestion = self._suggest_command(text)
        return {
            "node_type": "UNKNOWN",
            "raw_text": text,
            "confidence": 0.0,
            "error": "Unrecognized command pattern",
            "suggestion": suggestion,
            "eva_phase": self.phase,
        }

    def _build_ast_node(self, category: str, match, raw_text: str) -> dict[str, Any]:
        """Construye un nodo AST basado en la categoría y el match"""
        if category == "query_entity":
            entity_id = match.group(1).upper()
            return {
                "node_type": "QUERY_EXPRESSION",
                "target": "ENTITY_STATUS",
                "entity_name": entity_id,
                "raw_text": raw_text,
                "confidence": 0.9,
                "suggestion": self.suggestion_map[category],
            }
        elif category == "execute_protocol":
            protocol_name = match.group(1).upper()
            entity_name = match.group(2).upper()
            return {
                "node_type": "EXECUTE_EXPRESSION",
                "protocol": protocol_name,
                "entity_name": entity_name,
                "raw_text": raw_text,
                "confidence": 0.9,
                "suggestion": self.suggestion_map[category],
            }
        elif category == "create_entity":
            entity_name = match.group(1).upper()
            return {
                "node_type": "CREATE_EXPRESSION",
                "entity_name": entity_name,
                "raw_text": raw_text,
                "confidence": 0.85,
                "suggestion": self.suggestion_map[category],
            }
        elif category == "modify_entity":
            entity_name = match.group(1).upper()
            modification = match.group(2).strip()
            return {
                "node_type": "MODIFY_EXPRESSION",
                "entity_name": entity_name,
                "modification": modification,
                "raw_text": raw_text,
                "confidence": 0.8,
                "suggestion": self.suggestion_map[category],
            }
        elif category == "list_entities":
            return {
                "node_type": "LIST_EXPRESSION",
                "target": "ALL_ENTITIES",
                "raw_text": raw_text,
                "confidence": 0.95,
                "suggestion": self.suggestion_map[category],
            }
        elif category == "help":
            return {
                "node_type": "HELP_EXPRESSION",
                "raw_text": raw_text,
                "confidence": 1.0,
                "suggestion": self.suggestion_map[category],
            }
        else:
            return {
                "node_type": "UNKNOWN",
                "raw_text": raw_text,
                "confidence": 0.0,
                "suggestion": self._suggest_command(raw_text),
            }

    def _suggest_command(self, text: str) -> str:
        """Sugiere el comando más cercano basado en similitud simple."""
        text_lower = text.lower()
        for category in self.patterns:
            for pattern in self.patterns[category]:
                if any(word in text_lower for word in pattern.split()):
                    return self.suggestion_map.get(category, "Consulte ayuda.")
        return "Comando no reconocido. Ejemplo: ayuda"

    def get_supported_commands(self) -> list[str]:
        """Devuelve la lista de comandos soportados"""
        return list(self.patterns.keys())

    def validate_syntax(self, text: str) -> dict[str, Any]:
        """
        Valida la sintaxis del texto de entrada sin generar AST completo

        Args:
            text: Texto a validar

        Returns:
            Diccionario con resultados de validación y sugerencias
        """
        ast = self.parse(text)
        return {
            "is_valid": ast.get("node_type") not in ("UNKNOWN", "EMPTY"),
            "confidence": ast.get("confidence", 0.0),
            "error": ast.get("error"),
            "suggested_category": ast.get("node_type"),
            "suggestion": ast.get("suggestion"),
        }

    def _simulate_ast(self, ast: dict) -> dict:
        """
        Simula el AST como experiencia en EVA, generando un RealityBytecode y manifestaciones.
        """
        if not self.eva_runtime:
            return {}
        # Construir intención simbólica básica para EVA
        intention = {
            "intention_type": "PARSE_AST_EXPERIENCE",
            "ast": ast,
            "phase": self.phase,
        }
        # Compilar y ejecutar en EVA
        try:
            bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
            simulation_state = self.eva_runtime.execute_bytecode(bytecode)
            for hook in self._environment_hooks:
                try:
                    hook(simulation_state)
                except Exception as e:
                    simulation_state["eva_hook_error"] = str(e)
            return simulation_state
        except Exception as e:
            return {"error": f"EVA simulation failed: {e}"}

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def set_phase(self, phase: str):
        """Cambia la fase activa de simulación EVA."""
        self.phase = phase

    def get_phase(self) -> str:
        """Devuelve la fase de simulación actual."""
        return self.phase
