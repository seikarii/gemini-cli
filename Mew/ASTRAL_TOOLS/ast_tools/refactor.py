import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crisalida_lib.ASTRAL_TOOLS.ast_tools.models import (
    ModificationOperation,
    ModificationSpec,
)
from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class RenameSymbolParams(BaseModel):
    """
    Parámetros para renombrar un símbolo de forma segura en un ámbito específico de un archivo Python.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: str = Field(
        ..., description="Ruta absoluta al archivo Python a modificar."
    )
    scope_query: dict[str, Any] = Field(
        ...,
        description="Query ASTFinder para localizar el nodo de ámbito (ej: FunctionDef o ClassDef específico).",
    )
    old_name: str = Field(..., description="Nombre antiguo del símbolo a renombrar.")
    new_name: str = Field(..., description="Nuevo nombre para el símbolo.")

    @model_validator(mode="after")
    def validate_names(self):
        if self.old_name == self.new_name:
            raise ValueError("El nombre antiguo y el nuevo deben ser diferentes.")
        if not self.old_name.isidentifier() or not self.new_name.isidentifier():
            raise ValueError(
                "Ambos nombres deben ser identificadores válidos de Python."
            )
        return self


class RenameSymbolTool(BaseTool):
    """
    Herramienta avanzada para renombrar símbolos de forma segura en un ámbito específico de un archivo Python.
    Utiliza ASTModifier y validación de conflictos para evitar colisiones y errores de alcance.
    """

    def _get_name(self) -> str:
        return "rename_symbol"

    def _get_description(self) -> str:
        return (
            "Renombra un símbolo de forma segura en un ámbito específico de un archivo Python, "
            "usando AST y validación avanzada para evitar conflictos."
        )

    def _get_pydantic_schema(self):
        return RenameSymbolParams

    def _get_category(self) -> str:
        return "ast_refactoring"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = RenameSymbolParams(**kwargs)
        # Lazy import to avoid circular import at module import time
        from crisalida_lib.ASTRAL_TOOLS.ast_tools.modifier import ASTModifier

        modifier = ASTModifier()

        mod_spec = ModificationSpec(
            operation=ModificationOperation.RENAME_SYMBOL_SCOPED,
            target_query=params.scope_query,
            attribute=params.old_name,
            value=params.new_name,
            validate_before=True,
            validate_after=True,
            metadata={
                "reason": "safe_rename",
                "timestamp": str(
                    logging.Formatter().formatTime(
                        logging.LogRecord(
                            name="rename_symbol",
                            level=logging.INFO,
                            pathname="",
                            lineno=0,
                            msg="",
                            args=(),
                            exc_info=None,
                        )
                    )
                ),
            },
        )

        return await modifier.execute(
            file_path=params.file_path,
            modifications=[mod_spec],
            output_file_path=params.file_path,
            format_output=True,
            formatter="black",
        )

    async def demo(self):
        """Demonstrate RenameSymbolTool functionality."""
        print("🔄 RENAME SYMBOL TOOL DEMO")
        print("=" * 40)

        print("This tool safely renames symbols within a specific scope")
        print("Example: Rename variable 'old_var' to 'new_var' within a function")

        # Note: This is a demo that would need a temporary file for real operation
        print("✅ Tool initialized and ready for symbol renaming operations")
        print("📝 Use with file_path, scope_query, old_name, and new_name parameters")

        return ToolCallResult(
            command="rename_symbol_demo",
            success=True,
            output="RenameSymbolTool demo completed - tool ready for use",
            execution_time=0.1,
            error_message=None,
        )
