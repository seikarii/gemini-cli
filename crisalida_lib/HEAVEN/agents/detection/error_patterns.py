"""
ErrorPatterns - Advanced error pattern matching and fix strategies for Python linting and type checking.

Defines robust patterns for Ruff and MyPy errors, with confidence tracking, prioritization, and adaptive learning.
Ready for autonomous and semi-autonomous code repair, diagnostics, and visualization.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    EVAMemoryManager as EVAMemorySystem,
)
from crisalida_lib.EVA.typequalia import QualiaState

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Types of fix strategies for error resolution."""

    AST_MODIFY = "ast_modify"
    TEXT_REPLACE = "text_replace"
    ADD_IMPORT = "add_import"
    REMOVE_IMPORT = "remove_import"
    ADD_TYPE_ANNOTATION = "add_type_annotation"
    REORDER_IMPORTS = "reorder_imports"
    REMOVE_UNUSED = "remove_unused"
    ADD_DOCSTRING = "add_docstring"
    FORMAT_CODE = "format_code"


@dataclass
class ErrorPattern:
    """Represents a pattern for a specific type of error."""

    error_codes: list[str]  # ruff/mypy error codes that match this pattern
    description: str
    fix_strategy: FixStrategy
    confidence: float  # 0.0 to 1.0, how confident we are this fix will work
    template: str | None = None  # Template for text replacement
    ast_operation: str | None = None  # AST operation description
    conditions: dict[str, Any] | None = field(
        default_factory=dict
    )  # Additional conditions to check
    severity: str = "error"  # error, warning, info


# --- Ruff error patterns ---
RUFF_ERROR_PATTERNS = {
    # Import-related errors
    "F401": ErrorPattern(
        error_codes=["F401"],
        description="Unused import",
        fix_strategy=FixStrategy.REMOVE_IMPORT,
        confidence=0.9,
        ast_operation="remove_import_node",
        severity="warning",
    ),
    "I001": ErrorPattern(
        error_codes=["I001"],
        description="Import block is un-sorted or un-formatted",
        fix_strategy=FixStrategy.REORDER_IMPORTS,
        confidence=0.95,
        ast_operation="reorder_imports",
        severity="info",
    ),
    "E402": ErrorPattern(
        error_codes=["E402"],
        description="Module level import not at top of file",
        fix_strategy=FixStrategy.AST_MODIFY,
        confidence=0.8,
        ast_operation="move_import_to_top",
        severity="warning",
    ),
    # Unused variables
    "F841": ErrorPattern(
        error_codes=["F841"],
        description="Local variable is assigned to but never used",
        fix_strategy=FixStrategy.REMOVE_UNUSED,
        confidence=0.7,
        template="_{variable_name}",  # Prefix with underscore
        severity="warning",
    ),
    # Line length
    "E501": ErrorPattern(
        error_codes=["E501"],
        description="Line too long",
        fix_strategy=FixStrategy.TEXT_REPLACE,
        confidence=0.6,
        template="split_long_line",
        severity="info",
    ),
    # Trailing whitespace
    "W291": ErrorPattern(
        error_codes=["W291"],
        description="Trailing whitespace",
        fix_strategy=FixStrategy.TEXT_REPLACE,
        confidence=0.95,
        template="strip_trailing_whitespace",
        severity="info",
    ),
    # Missing newline at end of file
    "W292": ErrorPattern(
        error_codes=["W292"],
        description="No newline at end of file",
        fix_strategy=FixStrategy.TEXT_REPLACE,
        confidence=0.95,
        template="add_final_newline",
        severity="info",
    ),
    # Docstring-related errors
    "D100": ErrorPattern(
        error_codes=["D100"],
        description="Missing docstring in public module",
        fix_strategy=FixStrategy.ADD_DOCSTRING,
        confidence=0.85,
        template="Add module docstring",
        severity="warning",
    ),
    "D101": ErrorPattern(
        error_codes=["D101"],
        description="Missing docstring in public class",
        fix_strategy=FixStrategy.ADD_DOCSTRING,
        confidence=0.85,
        template="Add class docstring",
        severity="warning",
    ),
    "D102": ErrorPattern(
        error_codes=["D102"],
        description="Missing docstring in public method",
        fix_strategy=FixStrategy.ADD_DOCSTRING,
        confidence=0.85,
        template="Add method docstring",
        severity="warning",
    ),
    # Ambiguous import errors
    "F403": ErrorPattern(
        error_codes=["F403"],
        description="‘from module import *’ used; unable to detect undefined names",
        fix_strategy=FixStrategy.REMOVE_IMPORT,
        confidence=0.7,
        ast_operation="remove_star_import",
        severity="warning",
    ),
}

# --- MyPy error patterns ---
MYPY_ERROR_PATTERNS = {
    "missing-return-type": ErrorPattern(
        error_codes=["missing-return-type"],
        description="Function is missing return type annotation",
        fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
        confidence=0.8,
        template="-> Any",
        severity="warning",
    ),
    "missing-parameter-type": ErrorPattern(
        error_codes=["missing-parameter-type"],
        description="Function parameter is missing type annotation",
        fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
        confidence=0.8,
        template=": Any",
        severity="warning",
    ),
    "arg-type": ErrorPattern(
        error_codes=["arg-type"],
        description="Argument has incompatible type",
        fix_strategy=FixStrategy.AST_MODIFY,
        confidence=0.4,  # Lower confidence since it requires understanding intent
        ast_operation="fix_argument_type",
        severity="error",
    ),
    "attr-defined": ErrorPattern(
        error_codes=["attr-defined"],
        description="Attribute not defined",
        fix_strategy=FixStrategy.AST_MODIFY,
        confidence=0.3,  # Very context-dependent
        ast_operation="add_attribute_check",
        severity="error",
    ),
    "no-untyped-def": ErrorPattern(
        error_codes=["no-untyped-def"],
        description="Function is missing type annotation",
        fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
        confidence=0.7,
        template="def func(...) -> Any:",
        severity="warning",
    ),
}

# --- Ruff error patterns (extendidos) ---
RUFF_ERROR_PATTERNS.update(
    {
        # Formatting
        "E203": ErrorPattern(
            error_codes=["E203"],
            description="Whitespace before ':'",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.95,
            template="strip_whitespace_before_colon",
            severity="info",
        ),
        "E231": ErrorPattern(
            error_codes=["E231"],
            description="Missing whitespace after ','",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.95,
            template="add_whitespace_after_comma",
            severity="info",
        ),
        "E302": ErrorPattern(
            error_codes=["E302"],
            description="Expected 2 blank lines, found 1",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.9,
            template="add_blank_line",
            severity="info",
        ),
        "E305": ErrorPattern(
            error_codes=["E305"],
            description="Expected 2 blank lines after class or function definition",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.9,
            template="add_blank_line_after_def",
            severity="info",
        ),
        # Indentation
        "E111": ErrorPattern(
            error_codes=["E111"],
            description="Indentation is not a multiple of four",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.8,
            template="fix_indentation",
            severity="warning",
        ),
        "E114": ErrorPattern(
            error_codes=["E114"],
            description="Indentation is not a multiple of four (comment)",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.8,
            template="fix_indentation_comment",
            severity="warning",
        ),
        # Syntax
        "E701": ErrorPattern(
            error_codes=["E701"],
            description="Multiple statements on one line (colon)",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.7,
            template="split_statements",
            severity="warning",
        ),
        # Unused expressions
        "F632": ErrorPattern(
            error_codes=["F632"],
            description="Use of ==/!= with constant literals",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.6,
            template="fix_comparison_literal",
            severity="warning",
        ),
        # Unused function argument
        "F822": ErrorPattern(
            error_codes=["F822"],
            description="Undefined name in __all__",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.6,
            template="fix_all_undefined_name",
            severity="warning",
        ),
        # Unused variable (alternative)
        "F823": ErrorPattern(
            error_codes=["F823"],
            description="Local variable referenced before assignment",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.6,
            template="fix_variable_before_assignment",
            severity="warning",
        ),
    }
)

# --- MyPy error patterns (extendidos) ---
MYPY_ERROR_PATTERNS.update(
    {
        "var-annotated": ErrorPattern(
            error_codes=["var-annotated"],
            description="Variable is annotated but not assigned",
            fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
            confidence=0.7,
            template="add_variable_assignment",
            severity="warning",
        ),
        "override-missing": ErrorPattern(
            error_codes=["override-missing"],
            description="Method is missing @override decorator",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.7,
            template="add_override_decorator",
            severity="warning",
        ),
        "redundant-cast": ErrorPattern(
            error_codes=["redundant-cast"],
            description="Redundant cast to type",
            fix_strategy=FixStrategy.TEXT_REPLACE,
            confidence=0.8,
            template="remove_redundant_cast",
            severity="info",
        ),
        "return-value": ErrorPattern(
            error_codes=["return-value"],
            description="Function does not return a value",
            fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
            confidence=0.7,
            template="add_return_type_none",
            severity="warning",
        ),
        "name-defined": ErrorPattern(
            error_codes=["name-defined"],
            description="Name is not defined",
            fix_strategy=FixStrategy.AST_MODIFY,
            confidence=0.5,
            ast_operation="define_missing_name",
            severity="error",
        ),
        "import-error": ErrorPattern(
            error_codes=["import-error"],
            description="Cannot import name",
            fix_strategy=FixStrategy.REMOVE_IMPORT,
            confidence=0.6,
            ast_operation="remove_or_fix_import",
            severity="error",
        ),
        "missing-type-annotation": ErrorPattern(
            error_codes=["missing-type-annotation"],
            description="Missing type annotation for variable",
            fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
            confidence=0.7,
            template="add_variable_type_annotation",
            severity="warning",
        ),
        "incompatible-types": ErrorPattern(
            error_codes=["incompatible-types"],
            description="Incompatible types in assignment",
            fix_strategy=FixStrategy.AST_MODIFY,
            confidence=0.5,
            ast_operation="fix_incompatible_assignment",
            severity="error",
        ),
        "no-untyped-call": ErrorPattern(
            error_codes=["no-untyped-call"],
            description="Call to untyped function",
            fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
            confidence=0.6,
            template="add_function_type_annotation",
            severity="warning",
        ),
        "no-untyped-def": ErrorPattern(
            error_codes=["no-untyped-def"],
            description="Function is missing type annotation",
            fix_strategy=FixStrategy.ADD_TYPE_ANNOTATION,
            confidence=0.7,
            template="def func(...) -> Any:",
            severity="warning",
        ),
        "extra-missing-docstring": ErrorPattern(
            error_codes=["extra-missing-docstring"],
            description="Missing docstring in function/class/module",
            fix_strategy=FixStrategy.ADD_DOCSTRING,
            confidence=0.8,
            template="add_missing_docstring",
            severity="info",
        ),
    }
)

# --- Combined patterns ---
ALL_ERROR_PATTERNS = {**RUFF_ERROR_PATTERNS, **MYPY_ERROR_PATTERNS}


class ErrorPatternMatcher:
    """
    Matches errors to fix patterns, tracks confidence and success rates, and prioritizes fixes.
    Adaptive and ready for integration with autonomous agents.
    """

    def __init__(self):
        self.patterns = ALL_ERROR_PATTERNS
        self.success_history: dict[str, float] = {}  # Track success rates

    def find_pattern(
        self, error_code: str, error_message: str = ""
    ) -> ErrorPattern | None:
        """Find the best matching pattern for an error."""
        # Direct code match
        if error_code in self.patterns:
            return self.patterns[error_code]

        # Fuzzy code match
        for pattern in self.patterns.values():
            if error_code in pattern.error_codes:
                return pattern

            # Check if any keywords from the pattern description match the error message
            if pattern.description.lower() in error_message.lower():
                return pattern

        # Mejorado: fuzzy match por template y ast_operation
        for pattern in self.patterns.values():
            if pattern.template and pattern.template in error_message:
                return pattern
            if pattern.ast_operation and pattern.ast_operation in error_message:
                return pattern

        return None

    def get_fix_confidence(self, error_code: str) -> float:
        """Get confidence level for fixing this error type."""
        pattern = self.find_pattern(error_code)
        if not pattern:
            return 0.0

        base_confidence = pattern.confidence

        # Adjust based on historical success
        historical_success = self.success_history.get(error_code, 0.5)

        # Weighted average: 70% base confidence, 30% historical
        return base_confidence * 0.7 + historical_success * 0.3

    def update_success_rate(self, error_code: str, success: bool):
        """Update the historical success rate for an error type."""
        current_rate = self.success_history.get(error_code, 0.5)
        # Simple moving average with recent events weighted more heavily
        new_rate = current_rate * 0.8 + (1.0 if success else 0.0) * 0.2
        self.success_history[error_code] = new_rate
        logger.debug(f"Updated success rate for {error_code}: {new_rate:.2f}")

    def get_fixable_errors(self, error_codes: list[str]) -> list[str]:
        """Return list of error codes that we can attempt to fix."""
        return [
            code
            for code in error_codes
            if self.find_pattern(code) and self.get_fix_confidence(code) > 0.3
        ]

    def prioritize_errors(self, errors: list[Any]) -> list[Any]:
        """Sort errors by fix priority (confidence * severity)."""
        severity_map = {"error": 1.0, "warning": 0.7, "info": 0.5}

        def error_priority(error):
            code = error.get("code")
            pattern = self.find_pattern(code, error.get("message", ""))
            confidence = self.get_fix_confidence(code)
            severity = severity_map.get(getattr(pattern, "severity", "error"), 1.0)
            # Penalizar errores con baja confianza y baja severidad
            penalty = 0.0
            if confidence < 0.5:
                penalty += 0.1
            if severity < 0.7:
                penalty += 0.05
            return confidence * severity - penalty

        # Mejorado: priorizar errores con dependencias (ej: imports antes que variables)
        def has_dependency(error):
            code = error.get("code")
            return (
                code.startswith("F4")
                or code.startswith("I00")
                or code.startswith("E402")
            )

        sorted_errors = sorted(errors, key=error_priority, reverse=True)
        sorted_errors = sorted(sorted_errors, key=has_dependency, reverse=True)
        return sorted_errors

    def get_pattern_stats(self) -> dict[str, Any]:
        """Returns statistics about patterns and success rates."""
        return {
            "total_patterns": len(self.patterns),
            "success_history": self.success_history.copy(),
            "pattern_confidences": {k: v.confidence for k, v in self.patterns.items()},
        }


class EVAErrorPatternMatcher(ErrorPatternMatcher):
    """
    EVAErrorPatternMatcher - Extensión para integración con la memoria viviente EVA.
    Registra cada patrón de error y su resolución como experiencia viviente, soporta ingestión, recall, benchmarking, hooks y faseo.
    """

    def __init__(
        self, entity_id: str = "ErrorPatternMatcher", eva_phase: str = "default"
    ):
        super().__init__()
        self.eva_memory = EVAMemorySystem(config=None, phase=eva_phase)
        self.eva_phase = eva_phase
        self._environment_hooks: list = []

    def ingest_pattern_experience(
        self,
        error_code: str,
        error_message: str,
        fix_result: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ):
        """
        Compila una experiencia de patrón de error y su resolución en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=1.0 if fix_result.get("success") else 0.3,
            cognitive_complexity=0.7,
            consciousness_density=0.6,
            narrative_importance=0.8,
            energy_level=1.0,
        )
        experience_data = {
            "error_code": error_code,
            "error_message": error_message,
            "fix_result": fix_result,
            "timestamp": fix_result.get("timestamp"),
            "phase": phase,
        }
        self.eva_memory.record_experience(
            entity_id="error_pattern_matcher",
            event_type="error_pattern_resolution",
            data=experience_data,
            qualia_state=qualia_state,
        )
        for hook in self._environment_hooks:
            try:
                hook(experience_data)
            except Exception as e:
                logger.warning(f"[EVA-ERROR-PATTERN] Environment hook failed: {e}")

    def recall_pattern_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de patrón de error almacenada, manifestando la simulación.
        """
        # The EVAMemorySystem.recall_experience only takes experience_id
        # The 'phase' argument is handled internally by EVAMemorySystem
        return self.eva_memory.recall_experience(cue)

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia de patrón de error EVA.
        """
        # EVAMemoryManager no longer has add_experience_phase.
        # This logic should be handled within EVAErrorPatternMatcher if needed.
        # For now, we will just log a warning.
        logger.warning(
            f"Attempted to add experience phase {phase} for {experience_id}, but EVAMemoryManager does not support this directly."
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        self.eva_memory.set_memory_phase(phase)
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-ERROR-PATTERN] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_memory.get_memory_phase()

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de patrón de error EVA."""
        # EVAMemoryManager no longer has get_experience_phases.
        # This logic should be handled within EVAErrorPatternMatcher if needed.
        # For now, we will return an empty list.
        logger.warning(
            f"Attempted to get experience phases for {experience_id}, but EVAMemoryManager does not support this directly."
        )
        return []

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)
        # EVAMemoryManager no longer has add_environment_hook.
        # The hook is added to the local _environment_hooks.

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_pattern_experience": self.ingest_pattern_experience,
            "eva_recall_pattern_experience": self.recall_pattern_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# Singleton instance
error_pattern_matcher = ErrorPatternMatcher()

# Singleton EVA instance
eva_error_pattern_matcher = EVAErrorPatternMatcher()
