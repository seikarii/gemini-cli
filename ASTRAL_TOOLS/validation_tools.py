"""
Herramientas de Validación Avanzada para Crisalida
==================================================
Incluye:
- LinterTool: Linting estructurado con ruff
- TesterTool: Ejecución de tests con pytest
- TypeCheckerTool: Chequeo de tipos con mypy

Optimizado para integración con ToolRegistry y flujos de CI/CD.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class LintResult(BaseModel):
    """Representa un problema de linting detectado por ruff."""

    file: str = Field(..., description="Ruta del archivo.")
    line: int = Field(..., description="Línea.")
    column: int = Field(..., description="Columna.")
    code: str = Field(..., description="Código de la regla (ej: 'E123', 'F401').")
    message: str = Field(..., description="Descripción del problema.")
    severity: str = Field(..., description="Severidad ('error', 'warning', 'info').")


class LinterToolParams(BaseModel):
    """Parámetros para LinterTool."""

    file_path: str = Field(..., description="Ruta absoluta del archivo a lint.")


class LinterTool(BaseTool):
    """Herramienta para linting con ruff y salida estructurada."""

    def _get_name(self) -> str:
        return "linter_tool"

    def _get_description(self) -> str:
        return "Linting avanzado usando ruff, salida JSON estructurada."

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return LinterToolParams

    def _get_category(self) -> str:
        return "validation"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        file_path = kwargs.get("file_path")
        if not file_path:
            return ToolCallResult(
                command="linter_tool",
                success=False,
                output="",
                error_message="file_path es requerido",
                execution_time=0.0,
            )

        logger.info(f"Linting con ruff: {file_path}")
        start_time = asyncio.get_event_loop().time()

        try:
            result = subprocess.run(
                [
                    "/media/seikarii/Nvme/Crisalida/.venv/bin/ruff",
                    "check",
                    file_path,
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            issues = []
            lint_successful = True

            if result.stdout:
                try:
                    ruff_output = json.loads(result.stdout)
                    for item in ruff_output:
                        severity = "error" if item.get("fix") is None else "warning"
                        issues.append(
                            LintResult(
                                file=item.get("filename", file_path),
                                line=item.get("location", {}).get("row", 1),
                                column=item.get("location", {}).get("column", 1),
                                code=item.get("code", "unknown"),
                                message=item.get("message", ""),
                                severity=severity,
                            ).model_dump()
                        )
                except json.JSONDecodeError:
                    logger.warning("Error parseando JSON de ruff")

            if result.returncode != 0 and not result.stdout:
                lint_successful = False
                error_msg = result.stderr or "Error desconocido en ruff"
                return ToolCallResult(
                    command=f"linter_tool(file_path='{file_path}')",
                    success=False,
                    output="",
                    error_message=f"Ruff falló: {error_msg}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            output_json = {
                "file": file_path,
                "issues": issues,
                "lint_successful": lint_successful,
                "error_message": None,
            }

            return ToolCallResult(
                command=f"linter_tool(file_path='{file_path}')",
                success=True,
                output=json.dumps(output_json, indent=2),
                execution_time=asyncio.get_event_loop().time() - start_time,
                error_message=None,
            )

        except subprocess.TimeoutExpired:
            return ToolCallResult(
                command=f"linter_tool(file_path='{file_path}')",
                success=False,
                output="",
                error_message="Ruff timeout (30s)",
                execution_time=30.0,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"linter_tool(file_path='{file_path}')",
                success=False,
                output="",
                error_message=f"Error ejecutando ruff: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def demo(self):
        """Demonstrate the LinterTool's functionality."""
        logger.info(
            "Demonstrating LinterTool: This tool performs static code analysis using ruff."
        )
        # Example usage: lint a dummy file
        dummy_file_content = (
            """import os, sys\n\ndef my_function():\n    x =  1 + 1\n    print(x) \n"""
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(dummy_file_content)
            dummy_file_path = f.name

        try:
            result = await self.execute(file_path=dummy_file_path)
            logger.info(f"LinterTool demo result: {result.output}")
            return result
        finally:
            os.remove(dummy_file_path)


class TestResult(BaseModel):
    """Representa el resultado de un test unitario."""

    test_name: str = Field(..., description="Nombre del test.")
    status: str = Field(..., description="Estado ('passed', 'failed', 'skipped').")
    duration: float = Field(..., description="Duración en segundos.")
    message: str | None = Field(None, description="Mensaje o error.")


class TesterToolParams(BaseModel):
    """Parámetros para TesterTool."""

    test_path: str = Field(
        ..., description="Ruta absoluta del archivo o directorio de tests."
    )


class TesterTool(BaseTool):
    """Herramienta para ejecutar tests con pytest y salida estructurada."""

    def _get_name(self) -> str:
        return "tester_tool"

    def _get_description(self) -> str:
        return "Ejecuta tests con pytest y retorna resultados estructurados."

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return TesterToolParams

    def _get_category(self) -> str:
        return "validation"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        test_path = kwargs.get("test_path")
        if not test_path:
            return ToolCallResult(
                command="tester_tool",
                success=False,
                output="",
                error_message="test_path es requerido",
                execution_time=0.0,
            )
        logger.info(f"Ejecutando tests con pytest: {test_path}")
        start_time = asyncio.get_event_loop().time()
        try:
            result = subprocess.run(
                [
                    "pytest",
                    test_path,
                    "--maxfail=10",
                    "--disable-warnings",
                    "--tb=short",
                    "-v",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            tests_successful = result.returncode == 0
            results = []

            # Parse pytest output
            output_lines = result.stdout.splitlines()
            test_count = 0
            passed_count = 0
            failed_count = 0
            skipped_count = 0

            for line in output_lines:
                if "::" in line and any(
                    status in line for status in ["PASSED", "FAILED", "SKIPPED"]
                ):
                    test_count += 1
                    if "PASSED" in line:
                        passed_count += 1
                        status = "passed"
                    elif "FAILED" in line:
                        failed_count += 1
                        status = "failed"
                    elif "SKIPPED" in line:
                        skipped_count += 1
                        status = "skipped"
                    else:
                        status = "unknown"

                    # Extract test name (before the status)
                    test_name = line.split(" ")[0] if " " in line else line

                    results.append(
                        TestResult(
                            test_name=test_name,
                            status=status,
                            duration=0.0,
                            message=None,
                        ).model_dump()
                    )

            output_json = {
                "test_path": test_path,
                "results": results,
                "tests_run": test_count,
                "tests_passed": passed_count,
                "tests_failed": failed_count,
                "tests_skipped": skipped_count,
                "tests_successful": tests_successful,
                "error_message": result.stderr if result.stderr else None,
            }
            return ToolCallResult(
                command=f"tester_tool(test_path='{test_path}')",
                success=True,
                output=json.dumps(output_json, indent=2),
                execution_time=asyncio.get_event_loop().time() - start_time,
                error_message=None,
            )
        except subprocess.TimeoutExpired:
            return ToolCallResult(
                command=f"tester_tool(test_path='{test_path}')",
                success=False,
                output="",
                error_message="Pytest timeout (60s)",
                execution_time=60.0,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"tester_tool(test_path='{test_path}')",
                success=False,
                output="",
                error_message=f"Error ejecutando pytest: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def demo(self):
        """Demonstrate the tester tool's functionality."""
        print("🧪 TESTER TOOL DEMO")
        print("=" * 40)

        # Create a temporary test file to demonstrate
        import os
        import tempfile

        dummy_test_content = """def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2

def test_multiplication():
    assert 3 * 4 == 12
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.py", delete=False
        ) as f:
            f.write(dummy_test_content)
            test_file_path = f.name

        try:
            result = await self.execute(test_path=test_file_path)
            print(f"Run tests: {result.success}")
            print(
                result.output[:300] + "..."
                if len(result.output) > 300
                else result.output
            )
        finally:
            os.remove(test_file_path)

        print("\n✅ Tester demo completed!")


class TypeCheckResult(BaseModel):
    """Representa un problema de chequeo de tipos (mypy)."""

    file: str = Field(..., description="Ruta del archivo.")
    line: int = Field(..., description="Línea.")
    column: int = Field(..., description="Columna.")
    code: str = Field(..., description="Código de error (ej: 'attr-defined').")
    message: str = Field(..., description="Descripción del problema.")
    severity: str = Field(..., description="Severidad ('error', 'note').")


class TypeCheckerToolParams(BaseModel):
    """Parámetros para TypeCheckerTool."""

    file_path: str = Field(..., description="Ruta absoluta del archivo o directorio.")


class TypeCheckerTool(BaseTool):
    """Herramienta para chequeo de tipos con mypy y salida estructurada."""

    def _get_name(self) -> str:
        return "type_checker_tool"

    def _get_description(self) -> str:
        return "Chequeo de tipos usando mypy, salida JSON estructurada."

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return TypeCheckerToolParams

    def _get_category(self) -> str:
        return "validation"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        file_path = kwargs.get("file_path")
        if not file_path:
            return ToolCallResult(
                command="type_checker_tool",
                success=False,
                output="",
                error_message="file_path es requerido",
                execution_time=0.0,
            )

        logger.info(f"Chequeando tipos con mypy: {file_path}")
        start_time = asyncio.get_event_loop().time()

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".ini", delete=False
            ) as f:
                f.write(
                    """[mypy]\npython_version = 3.12\nwarn_return_any = True\nwarn_unused_configs = True\ndisallow_untyped_defs = True\n"""
                )
                config_file = f.name
            result = subprocess.run(
                [
                    "/media/seikarii/Nvme/Crisalida/.venv/bin/mypy",
                    file_path,
                    "--show-error-codes",
                    "--config-file",
                    config_file,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            issues = []
            type_check_successful = result.returncode == 0
            for line in result.stdout.strip().split("\n"):
                if line and ":" in line:
                    try:
                        parts = line.split(":", 4)
                        if len(parts) >= 4:
                            file_part = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 1
                            col_num = int(parts[2]) if parts[2].isdigit() else 1
                            severity_and_message = parts[3].strip()
                            if severity_and_message.startswith("error:"):
                                severity = "error"
                                message = severity_and_message[6:].strip()
                            elif severity_and_message.startswith("note:"):
                                severity = "note"
                                message = severity_and_message[5:].strip()
                            else:
                                severity = "error"
                                message = severity_and_message
                            code = "unknown"
                            if "[" in message and "]" in message:
                                code_start = message.rfind("[")
                                code_end = message.rfind("]")
                                if code_start < code_end:
                                    code = message[code_start + 1 : code_end]
                                    message = message[:code_start].strip()
                            issues.append(
                                TypeCheckResult(
                                    file=file_part,
                                    line=line_num,
                                    column=col_num,
                                    code=code,
                                    message=message,
                                    severity=severity,
                                ).model_dump()
                            )
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parseando línea mypy: {line} - {e}")
                        continue
            output_json = {
                "file": file_path,
                "issues": issues,
                "type_check_successful": type_check_successful,
                "error_message": result.stderr if result.stderr else None,
            }
            return ToolCallResult(
                command=f"type_checker_tool(file_path='{file_path}')",
                success=True,
                output=json.dumps(output_json, indent=2),
                execution_time=asyncio.get_event_loop().time() - start_time,
                error_message=None,
            )
        except subprocess.TimeoutExpired:
            return ToolCallResult(
                command=f"type_checker_tool(file_path='{file_path}')",
                success=False,
                output="",
                error_message="MyPy timeout (30s)",
                execution_time=30.0,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"type_checker_tool(file_path='{file_path}')",
                success=False,
                output="",
                error_message=f"Error ejecutando mypy: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def demo(self):
        """Demonstrate the TypeCheckerTool's functionality."""
        logger.info(
            "Demonstrating TypeCheckerTool: This tool performs static type checking using mypy."
        )
        # Example usage: type check a dummy file
        dummy_file_content = """def add(a: int, b: int) -> int:\n    return a + b\n\ndef subtract(a: int, b: str) -> int:\n    return a - b # This will cause a type error\n"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(dummy_file_content)
            dummy_file_path = f.name

        try:
            result = await self.execute(file_path=dummy_file_path)
            logger.info(f"TypeCheckerTool demo result: {result.output}")
            return result
        finally:
            os.remove(dummy_file_path)
