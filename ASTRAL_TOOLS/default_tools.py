"""
DefaultTools - Registro centralizado de herramientas estándar para Crisalida
============================================================================
Este módulo registra todas las herramientas esenciales (filesystem, memoria, shell, web)
en el ToolRegistry, permitiendo su descubrimiento y uso por el sistema principal.

Incluye:
- Herramientas de sistema de archivos (lectura, escritura, búsqueda, reemplazo)
- Herramientas de memoria (guardar, recuperar, limpiar)
- Herramientas de shell y web (comandos, búsqueda, fetch)
- Extensible para nuevas herramientas (AST, generación de código, visualización, etc.)
"""

import logging

from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry
from crisalida_lib.ASTRAL_TOOLS.data_tools import ReturnDataTool

logger = logging.getLogger(__name__)


def register_default_tools(registry: ToolRegistry):
    """Registers all the default tools in the given registry."""
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.finder import ASTFinder
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.modifier import ASTModifier
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.refactor import RenameSymbolTool
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.reader import ASTReader
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.generator import ASTCodeGenerator
    from crisalida_lib.ASTRAL_TOOLS.code_generation.tool import CodeGenerationTool
    from crisalida_lib.ASTRAL_TOOLS.data_visualization import DataVisualizationTool
    from crisalida_lib.ASTRAL_TOOLS.file_system import (
        GlobTool,
        ListDirectoryTool,
        ReadFileTool,
        ReadManyFilesTool,
        ReplaceTool,
        SearchFileContentTool,
        WriteFileTool,
    )
    from crisalida_lib.ASTRAL_TOOLS.memory_tools import (
        ClearMemoryTool,
        RecallMemoryTool,
        SaveMemoryTool,
    )
    from crisalida_lib.ASTRAL_TOOLS.shell_and_web import (
        GoogleWebSearchTool,
        RunShellCommandTool,
        WebFetchTool,
    )

    default_tools = [
        ASTFinder(),
        ASTModifier(),
        RenameSymbolTool(),
        ASTReader(),
        ASTCodeGenerator(),
        CodeGenerationTool(),
        DataVisualizationTool(),
        GlobTool(),
        ListDirectoryTool(),
        ReadFileTool(),
        ReadManyFilesTool(),
        ReplaceTool(),
        SearchFileContentTool(),
        WriteFileTool(),
        ClearMemoryTool(),
        RecallMemoryTool(),
        SaveMemoryTool(),
        GoogleWebSearchTool(),
        RunShellCommandTool(),
        WebFetchTool(),
        ReturnDataTool(),
    ]

    for tool in default_tools:
        registry.register_tool(tool)
