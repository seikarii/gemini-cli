"""
FixGenerators - Generadores avanzados de fixes para AgenteBugs.

Implementa estrategias robustas para corrección de errores de Ruff y MyPy.
Incluye AST, reemplazo de texto, gestión de imports y anotaciones de tipo.
Preparado para diagnóstico, simulación y visualización avanzada.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager
from crisalida_lib.ASTRAL_TOOLS.ast_tools import ASTFinder, ASTModifier
from crisalida_lib.ASTRAL_TOOLS.file_system import (
    ReadFileTool,
    ReplaceTool,
    WriteFileTool,
)
from crisalida_lib.EVA.typequalia import QualiaState

from ..detection.error_patterns import ErrorPattern, FixStrategy

logger = logging.getLogger(__name__)


class FixGenerator:
    """Base para generadores de fixes específicos."""

    def __init__(self):
        self.read_tool = ReadFileTool()
        self.write_tool = WriteFileTool()
        self.replace_tool = ReplaceTool()
        self.ast_finder = ASTFinder()
        self.ast_modifier = ASTModifier()

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def _read_file_content(self, file_path: str) -> str:
        from crisalida_lib.ASTRAL_TOOLS.file_system import ReadFileTool

        read_tool = ReadFileTool()
        result = await read_tool.execute(absolute_path=file_path)
        if not result.success:
            logger.error(f"Failed to read file {file_path}: {result.error_message}")
            return ""
        return result.output

    async def _write_file_content(self, file_path: str, content: str) -> bool:
        result = await self.write_tool.execute(file_path=file_path, content=content)
        if not result.success:
            logger.error(f"Failed to write file {file_path}: {result.error_message}")
        return result.success


class TextReplaceFixGenerator(FixGenerator):
    """Fixes por reemplazo de texto y mejoras de formato."""

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        try:
            content = await self._read_file_content(file_path)
            lines = content.split("\n")
            error_line = error.get("line", 1) - 1
            if error_line >= len(lines):
                return None
            original_line = lines[error_line]
            fixed_line = original_line

            # --- Mejoras de formato y extras ---
            # Eliminar espacios en blanco al final de línea
            # strip_trailing_whitespace
            if pattern.template == "strip_trailing_whitespace":
                fixed_line = original_line.rstrip()
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed trailing whitespace",
                    }

            # Añadir salto de línea final si falta
            elif pattern.template == "add_final_newline":
                if not content.endswith("\n"):
                    new_content = content + "\n"
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Added newline at end of file",
                    }
                return None

            # Dividir línea larga en varias líneas
            elif pattern.template == "split_long_line":
                if len(original_line) > 88:
                    parts = original_line.split(" = ", 1)
                    if len(parts) == 2:
                        var_part, expr_part = parts
                        indent = len(original_line) - len(original_line.lstrip())
                        new_indent = " " * (indent + 4)
                        fixed_line = (
                            f"{var_part} = (\n{new_indent}{expr_part}\n{' ' * indent})"
                        )
                        lines[error_line] = fixed_line
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": "Split long line",
                        }

            # Prefijar variable no usada con guion bajo (F841)
            elif pattern.template and pattern.template.startswith("_"):
                var_match = re.search(
                    r"'([^']+)' is assigned to but never used", error.get("message", "")
                )
                if var_match:
                    var_name = var_match.group(1)
                    fixed_line = original_line.replace(
                        f" {var_name} ", f" _{var_name} "
                    ).replace(f" {var_name}=", f" _{var_name}=")
                    if fixed_line != original_line:
                        lines[error_line] = fixed_line
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": f"Prefixed unused variable with underscore: {var_name}",
                        }

            # Añadir docstring si falta (D100, D101, D102)
            elif pattern.fix_strategy == FixStrategy.ADD_DOCSTRING:
                docstring = '"""TODO: Add docstring."""'
                if "class " in original_line or "def " in original_line:
                    insert_line = error_line + 1
                    lines.insert(insert_line, docstring)
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Added placeholder docstring",
                    }
                elif error_line == 0:
                    lines.insert(0, docstring)
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Added module docstring",
                    }

            # Eliminar espacios antes de dos puntos (E203)
            elif pattern.template == "strip_whitespace_before_colon":
                fixed_line = re.sub(r"\s+:", ":", original_line)
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed whitespace before colon",
                    }

            # Añadir espacio después de coma (E231)
            elif pattern.template == "add_whitespace_after_comma":
                fixed_line = re.sub(r",(\S)", r", \1", original_line)
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Added whitespace after comma",
                    }

            # Añadir línea en blanco antes/después de definición (E302, E305)
            elif pattern.template in ["add_blank_line", "add_blank_line_after_def"]:
                if pattern.template == "add_blank_line" and error_line > 0:
                    if lines[error_line - 1].strip() != "":
                        lines.insert(error_line, "")
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": "Added blank line before definition",
                        }
                elif pattern.template == "add_blank_line_after_def":
                    if (
                        error_line + 1 < len(lines)
                        and lines[error_line + 1].strip() != ""
                    ):
                        lines.insert(error_line + 1, "")
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": "Added blank line after definition",
                        }

            # Corregir indentación (E111, E114)
            elif pattern.template in ["fix_indentation", "fix_indentation_comment"]:
                indent_size = 4
                fixed_line = original_line
                if pattern.template == "fix_indentation":
                    fixed_line = re.sub(
                        r"^( +)",
                        lambda m: " " * (len(m.group(1)) // indent_size * indent_size),
                        original_line,
                    )
                elif pattern.template == "fix_indentation_comment":
                    if original_line.lstrip().startswith("#"):
                        fixed_line = re.sub(
                            r"^( +)",
                            lambda m: " "
                            * (len(m.group(1)) // indent_size * indent_size),
                            original_line,
                        )
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Fixed indentation",
                    }

            # Separar múltiples sentencias en una línea (E701)
            elif pattern.template == "split_statements":
                if ";" in original_line:
                    fixed_line = original_line.replace(";", "\n")
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Split multiple statements on one line",
                    }

            # Corregir comparación con literales constantes (F632)
            elif pattern.template == "fix_comparison_literal":
                fixed_line = re.sub(
                    r"(==|!=)\s*(True|False|None)", r"is \2", original_line
                )
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Fixed comparison with constant literal",
                    }

            # Eliminar nombre indefinido en __all__ (F822)
            elif pattern.template == "fix_all_undefined_name":
                fixed_line = re.sub(r"'([^']+)'", r"", original_line)
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed undefined name from __all__",
                    }

            # Inicializar variable antes de asignación (F823)
            elif pattern.template == "fix_variable_before_assignment":
                var_match = re.search(
                    r"local variable '([^']+)' referenced before assignment",
                    error.get("message", ""),
                )
                if var_match and error_line > 0:
                    var_name = var_match.group(1)
                    lines.insert(error_line, f"{var_name} = None")
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": f"Initialized variable '{var_name}' before assignment",
                    }

            # Eliminar cast redundante (mypy)
            elif pattern.template == "remove_redundant_cast":
                fixed_line = re.sub(r"cast\(([^,]+),\s*\1\)", r"\1", original_line)
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed redundant cast",
                    }

            # Añadir decorador @override (mypy)
            elif pattern.template == "add_override_decorator":
                if error_line > 0 and not lines[error_line - 1].strip().startswith(
                    "@override"
                ):
                    lines.insert(error_line, "@override")
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Added @override decorator",
                    }

            # Asignar variable anotada pero no inicializada (mypy)
            elif pattern.template == "add_variable_assignment":
                var_match = re.search(
                    r"variable '([^']+)' is annotated but not assigned",
                    error.get("message", ""),
                )
                if var_match:
                    var_name = var_match.group(1)
                    lines[error_line] = f"{var_name} = None"
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": f"Assigned variable '{var_name}' to None",
                    }

            # Fix B904: raise from exception
            elif pattern.template == "fix_raise_from_exception":
                if "raise " in original_line:
                    except_var = None
                    # Buscar variable de excepción en el bloque actual
                    for i in range(error_line - 1, max(-1, error_line - 6), -1):
                        line_i = lines[i] if i >= 0 else ""
                        # Detectar 'except ... as var:'
                        except_match = re.search(
                            r"except\s+[^\s]+(?:\s+as\s+(\w+))?:", line_i
                        )
                        if except_match:
                            except_var = except_match.group(1)
                            if except_var:
                                break
                    # Si se encuentra variable, usarla en 'raise ... from var'
                    if except_var:
                        # Detectar tipo de excepción en raise
                        raise_match = re.match(
                            r"\s*raise\s+([^\s]+)(.*)", original_line
                        )
                        if raise_match:
                            exc_type = raise_match.group(1)
                            rest = raise_match.group(2)
                            fixed_line = f"raise {exc_type}{rest} from {except_var}"
                        else:
                            fixed_line = original_line + f" from {except_var}"
                        lines[error_line] = fixed_line
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": f"Fixed B904: added 'from {except_var}' to raise statement",
                        }
                    else:
                        # Si no se encuentra variable, usar 'from None'
                        raise_match = re.match(
                            r"\s*raise\s+([^\s]+)(.*)", original_line
                        )
                        if raise_match:
                            exc_type = raise_match.group(1)
                            rest = raise_match.group(2)
                            fixed_line = f"raise {exc_type}{rest} from None"
                        else:
                            fixed_line = original_line + " from None"
                        lines[error_line] = fixed_line
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": "Fixed B904: added 'from None' to raise statement",
                        }

            # --- Fixes extra y avanzados ---
            # Eliminar líneas duplicadas consecutivas
            elif pattern.template == "remove_duplicate_lines":
                if error_line > 0 and lines[error_line] == lines[error_line - 1]:
                    lines.pop(error_line)
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed consecutive duplicate line",
                    }

            # Normalizar comillas (simples/dobles)
            elif pattern.template == "normalize_quotes":
                fixed_line = original_line.replace("'", '"')
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Normalized quotes to double quotes",
                    }

            # Eliminar comentarios innecesarios
            elif pattern.template == "remove_unnecessary_comment":
                if (
                    original_line.strip().startswith("#")
                    and "TODO" not in original_line
                ):
                    lines.pop(error_line)
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed unnecessary comment",
                    }

            # Reemplazar tabs por espacios
            elif pattern.template == "replace_tabs_with_spaces":
                fixed_line = original_line.replace("\t", "    ")
                if fixed_line != original_line:
                    lines[error_line] = fixed_line
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Replaced tabs with spaces",
                    }

            # Eliminar líneas en blanco extra
            elif pattern.template == "remove_extra_blank_lines":
                if (
                    error_line > 0
                    and lines[error_line].strip() == ""
                    and lines[error_line - 1].strip() == ""
                ):
                    lines.pop(error_line)
                    new_content = "\n".join(lines)
                    return {
                        "type": "file_content_replace",
                        "file_path": file_path,
                        "new_content": new_content,
                        "description": "Removed extra blank line",
                    }

            return None
        except Exception as e:
            logger.error(f"Error generating text fix: {e}")
            return None


class ImportFixGenerator(FixGenerator):
    """Fixes para problemas de imports."""

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        try:
            content = await self._read_file_content(file_path)
            if pattern.fix_strategy == FixStrategy.REMOVE_IMPORT:
                return await self._fix_unused_import(file_path, content, error)
            elif pattern.fix_strategy == FixStrategy.REORDER_IMPORTS:
                return await self._fix_import_order(file_path, content, error)
            elif pattern.fix_strategy == FixStrategy.REMOVE_UNUSED:
                return await self._fix_unused_import(file_path, content, error)
            return None
        except Exception as e:
            logger.error(f"Error generating import fix: {e}")
            return None

    async def _fix_unused_import(
        self, file_path: str, content: str, error: dict[str, Any]
    ) -> dict[str, Any] | None:
        lines = content.split("\n")
        error_line = error.get("line", 1) - 1
        if error_line >= len(lines):
            return None

        # Remove the line with the unused import
        lines.pop(error_line)
        new_content = "\n".join(lines)

        return {
            "type": "file_content_replace",
            "file_path": file_path,
            "new_content": new_content,
            "description": f"Removed unused import at line {error.get('line', 1)}",
        }

    async def _fix_import_order(
        self, file_path: str, content: str, error: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            lines = content.split("\n")
            import_start = None
            import_end = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    if import_start is None:
                        import_start = i
                    import_end = i
                elif import_start is not None and stripped == "":
                    continue
                elif import_start is not None and not stripped.startswith("#"):
                    break
            if import_start is None or import_end is None:
                return None
            import_lines = lines[import_start : import_end + 1]
            stdlib_imports = []
            other_imports = []
            stdlib_modules = [
                "os",
                "sys",
                "json",
                "re",
                "typing",
                "collections",
                "itertools",
                "functools",
            ]
            for line in import_lines:
                stripped = line.strip()
                if any(
                    stripped.startswith(f"import {mod}")
                    or stripped.startswith(f"from {mod}")
                    for mod in stdlib_modules
                ):
                    stdlib_imports.append(line)
                else:
                    other_imports.append(line)
            stdlib_imports.sort()
            other_imports.sort()
            sorted_imports = stdlib_imports
            if stdlib_imports and other_imports:
                sorted_imports.extend([""])
            sorted_imports.extend(other_imports)
            new_lines = lines[:import_start] + sorted_imports + lines[import_end + 1 :]
            new_content = "\n".join(new_lines)
            return {
                "type": "file_content_replace",
                "file_path": file_path,
                "new_content": new_content,
                "description": "Reordered imports alphabetically",
            }
        except Exception as e:
            logger.error(f"Error fixing import order: {e}")
            return None


class TypeAnnotationFixGenerator(FixGenerator):
    """Fixes para anotaciones de tipo."""

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        try:
            content = await self._read_file_content(file_path)
            lines = content.split("\n")
            error_line = error.get("line", 1) - 1
            if error_line >= len(lines):
                return None
            if "missing return type" in error.get("message", "").lower():
                return await self._add_return_type(file_path, lines, error_line, error)
            elif "missing parameter type" in error.get("message", "").lower():
                return await self._add_parameter_type(
                    file_path, lines, error_line, error
                )
            elif (
                "function is missing a type annotation for one or more arguments"
                in error.get("message", "").lower()
            ):
                return await self._add_missing_function_type_annotation(
                    file_path, lines, error_line, error
                )
            return None
        except Exception as e:
            logger.error(f"Error generating type annotation fix: {e}")
            return None

    async def _add_return_type(
        self, file_path: str, lines: list[str], error_line: int, error: dict[str, Any]
    ) -> dict[str, Any] | None:
        line = lines[error_line]
        if "def " in line and ":" in line and "->" not in line:
            new_line = line.replace(":", " -> Any:")
            lines[error_line] = new_line
            has_any_import = any(
                "from typing import" in line and "Any" in line for line in lines[:20]
            )
            if not has_any_import:
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith(
                        "from "
                    ):
                        insert_pos = i + 1
                lines.insert(insert_pos, "from typing import Any")
            new_content = "\n".join(lines)
            return {
                "type": "file_content_replace",
                "file_path": file_path,
                "new_content": new_content,
                "description": "Added return type annotation",
            }
        return None

    async def _add_parameter_type(
        self, file_path: str, lines: list[str], error_line: int, error: dict[str, Any]
    ) -> dict[str, Any] | None:
        line = lines[error_line]
        message = error.get("message", "")
        param_match = re.search(r"parameter '([^']+)'", message)
        if not param_match:
            return None
        param_name = param_match.group(1)
        if f"{param_name}," in line or f"{param_name})" in line:
            new_line = line.replace(f"{param_name},", f"{param_name}: Any,")
            new_line = new_line.replace(f"{param_name})", f"{param_name}: Any)")
            if new_line != line:
                lines[error_line] = new_line
                has_any_import = any(
                    "from typing import" in line and "Any" in line
                    for line in lines[:20]
                )
                if not has_any_import:
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith(
                            "import "
                        ) or line.strip().startswith("from "):
                            insert_pos = i + 1
                    lines.insert(insert_pos, "from typing import Any")
                new_content = "\n".join(lines)
                return {
                    "type": "file_content_replace",
                    "file_path": file_path,
                    "new_content": new_content,
                    "description": f"Added type annotation for parameter: {param_name}",
                }
        return None

    async def _add_missing_function_type_annotation(
        self, file_path: str, lines: list[str], error_line: int, error: dict[str, Any]
    ) -> dict[str, Any] | None:
        line = lines[error_line]
        # Check if it's a function definition and doesn't already have a return type
        if "def " in line and ":" in line and "->" not in line:
            # Find the colon that marks the end of the function signature
            colon_index = line.rfind(":")
            if colon_index != -1:
                new_line = line[:colon_index] + " -> Any:" + line[colon_index + 1 :]
                lines[error_line] = new_line

                # Ensure 'from typing import Any' is present
                has_any_import = any(
                    "from typing import" in line and "Any" in line
                    for line in lines[:20]
                )
                if not has_any_import:
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith(
                            "import "
                        ) or line.strip().startswith("from "):
                            insert_pos = i + 1
                    lines.insert(insert_pos, "from typing import Any")

                new_content = "\n".join(lines)
                return {
                    "type": "file_content_replace",
                    "file_path": file_path,
                    "new_content": new_content,
                    "description": "Added missing type annotation to function signature",
                }
        return None


class ASTFixGenerator(FixGenerator):
    """Fixes usando AST para errores complejos."""

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        try:
            # Ejemplo: mover import al inicio del archivo (E402)
            if pattern.ast_operation == "move_import_to_top":
                content = await self._read_file_content(file_path)
                import_line = error.get("line", 1)
                lines = content.split("\n")
                if 0 < import_line <= len(lines):
                    line_content = lines[import_line - 1]
                    if "import " in line_content:
                        # Remove import from current position
                        lines.pop(import_line - 1)
                        # Insert at top after docstring or comments
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(
                                '"""'
                            ) or line.strip().startswith("#"):
                                insert_pos = i + 1
                            else:
                                break
                        lines.insert(insert_pos, line_content)
                        new_content = "\n".join(lines)
                        return {
                            "type": "file_content_replace",
                            "file_path": file_path,
                            "new_content": new_content,
                            "description": "Moved import to top of file",
                        }
            # Otros casos de AST pueden agregarse aquí
            return None
        except Exception as e:
            logger.error(f"Error generating AST fix: {e}")
            return None


class FixGeneratorFactory:
    """Fábrica para generadores de fixes según estrategia."""

    @staticmethod
    def get_generator(strategy: FixStrategy) -> FixGenerator:
        if strategy in [
            FixStrategy.REMOVE_IMPORT,
            FixStrategy.REORDER_IMPORTS,
            FixStrategy.ADD_IMPORT,
            FixStrategy.REMOVE_UNUSED,
        ]:
            return ImportFixGenerator()
        elif strategy == FixStrategy.ADD_TYPE_ANNOTATION:
            return TypeAnnotationFixGenerator()
        elif strategy == FixStrategy.AST_MODIFY:
            return ASTFixGenerator()
        elif strategy == FixStrategy.ADD_DOCSTRING:
            return TextReplaceFixGenerator()
        else:
            return TextReplaceFixGenerator()


class EVAFixGenerator(FixGenerator):
    """
    FixGenerator extendido para integración con EVA.
    Registra cada fix como experiencia viviente en la memoria EVA, soporta ingestión, recall, faseo, hooks, benchmarking y gestión avanzada.
    """

    def __init__(
        self,
        entity_id: str = "FixAgent",
        eva_phase: str = "default",
        eva_config: EVAConfig | None = None,
    ):
        super().__init__()
        self.eva_config = eva_config or EVAConfig()
        self.eva_phase = eva_phase or self.eva_config.EVA_MEMORY_PHASE
        self.eva_memory = EVAMemoryManager(
            config=self.eva_config,
            phase=self.eva_phase,
        )
        self.max_experiences = self.eva_config.EVA_MEMORY_MAX_EXPERIENCES
        self.retention_policy = self.eva_config.EVA_MEMORY_RETENTION_POLICY
        self.compression_level = self.eva_config.EVA_MEMORY_COMPRESSION_LEVEL
        self.simulation_rate = self.eva_config.EVA_MEMORY_SIMULATION_RATE
        self.multiverse_enabled = self.eva_config.EVA_MEMORY_MULTIVERSE_ENABLED
        self.timeline_count = self.eva_config.EVA_MEMORY_TIMELINE_COUNT
        self.benchmarking_enabled = self.eva_config.EVA_MEMORY_BENCHMARKING_ENABLED
        self.visualization_mode = self.eva_config.EVA_MEMORY_VISUALIZATION_MODE
        self._environment_hooks = self.eva_config.EVA_MEMORY_ENVIRONMENT_HOOKS.copy()

    async def generate_fix(
        self, file_path: str, error: dict[str, Any], pattern: ErrorPattern
    ) -> dict[str, Any] | None:
        fix_result = await super().generate_fix(file_path, error, pattern)
        if fix_result:
            qualia_state = QualiaState(
                emotional_valence=0.8,
                cognitive_complexity=0.7,
                consciousness_density=0.6,
                narrative_importance=1.0,
                energy_level=1.0,
            )
            experience_data = {
                "file_path": file_path,
                "error": error,
                "pattern": pattern.template,
                "fix_result": fix_result,
                "timestamp": fix_result.get("timestamp") if fix_result else None,
                "phase": self.eva_phase,
            }
            self.eva_memory.record_experience(
                entity_id="fix_generator",
                event_type="fix_generation",
                data=experience_data,
                qualia_state=qualia_state,
            )
            self._benchmark_eva_memory()
            self._optimize_eva_memory()
        return fix_result

    def eva_recall_fix_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de fix almacenada, manifestando la simulación.
        """
        # The EVAMemoryManager.recall_experience only takes experience_id.
        # The 'phase' argument is handled internally by EVAMemoryManager.
        return self.eva_memory.recall_experience(cue)

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia de fix EVA.
        """
        # EVAMemoryManager no longer has add_experience_phase.
        # This logic should be handled within EVAFixGenerator if needed.
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
                logger.warning(f"[EVA-FIX-GENERATOR] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_memory.get_memory_phase()

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de fix EVA."""
        # EVAMemoryManager no longer has get_experience_phases.
        # This logic should be handled within EVAFixGenerator if needed.
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

    def _benchmark_eva_memory(self):
        """Realiza benchmarking de la memoria EVA y reporta métricas clave."""
        if self.benchmarking_enabled:
            metrics = {
                "total_experiences": len(self.eva_memory.eva_memory_store),
                "phases": len(self.eva_memory.eva_phases),
                "hooks": len(self._environment_hooks),
                "compression_level": self.compression_level,
                "simulation_rate": self.simulation_rate,
                "multiverse_enabled": self.multiverse_enabled,
                "timeline_count": self.timeline_count,
            }
            logger.info(f"[EVA-FIX-GENERATOR-BENCHMARK] {metrics}")

    def _optimize_eva_memory(self):
        """Optimiza la gestión de memoria EVA, aplicando compresión y limpieza según la política."""
        if len(self.eva_memory.eva_memory_store) > self.max_experiences:
            sorted_exps = sorted(
                self.eva_memory.eva_memory_store.items(),
                key=lambda x: getattr(x[1], "timestamp", 0),
            )
            for exp_id, _ in sorted_exps[
                : len(self.eva_memory.eva_memory_store) - self.max_experiences
            ]:
                del self.eva_memory.eva_memory_store[exp_id]
        # Placeholder para lógica avanzada de compresión si es necesario

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_fix_experience": self.generate_fix,
            "eva_recall_fix_experience": self.eva_recall_fix_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
