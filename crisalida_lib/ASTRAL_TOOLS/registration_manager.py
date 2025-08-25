import logging
import subprocess
import sys
from typing import Any

from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry
from crisalida_lib.ASTRAL_TOOLS.registration import register_all_tools
from crisalida_lib.HEAVEN.sandbox.sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)

_global_tool_registry: ToolRegistry | None = None

# Inicializar el SandboxExecutor (usar el python del .venv principal)
# Usar sys.executable para que funcione en cualquier entorno
_sandbox_executor = SandboxExecutor(python_path=sys.executable)  # NUEVO


def get_global_tool_registry() -> ToolRegistry:
    global _global_tool_registry
    if _global_tool_registry is None:
        _global_tool_registry = ToolRegistry()
        register_all_tools(_global_tool_registry)
        logger.info(f"Registered default tools. Registry content: {list(_global_tool_registry.get_all_tools().keys())}")

        # Register new tools with proper function references
        from datetime import datetime

        from pydantic import BaseModel, Field

        from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

        # Create tool wrappers for proper registration
        class SandboxToolParams(BaseModel):
            code: str = Field(..., description="Python code to execute in sandbox")
            timeout: int = Field(10, description="Timeout in seconds")

        class SandboxTool(BaseTool):
            def _get_name(self) -> str:
                return "execute_code_in_sandbox"

            def _get_description(self) -> str:
                return "Execute Python code in a secure sandbox environment"

            def _get_pydantic_schema(self) -> type[BaseModel]:
                return SandboxToolParams

            def _get_category(self) -> str:
                return "execution"

            async def execute(self, **kwargs) -> ToolCallResult:
                start_time = datetime.now()
                result = _sandbox_executor.execute_python_code(
                    kwargs["code"], kwargs.get("timeout", 10)
                )
                execution_time = (datetime.now() - start_time).total_seconds()

                return ToolCallResult(
                    command="execute_code_in_sandbox",
                    success=result["success"],
                    output=result["stdout"],
                    error_message=result["stderr"] if not result["success"] else None,
                    execution_time=execution_time,
                )

            async def demo(self):
                return await self.execute(code="print('Hello from sandbox!')")

        class IntegrationToolParams(BaseModel):
            file_path: str = Field(..., description="Path where to integrate the file")
            content: str = Field(..., description="Content of the file")
            commit_message: str = Field(..., description="Git commit message")

        class IntegrationTool(BaseTool):
            def _get_name(self) -> str:
                return "integrate_code_into_project"

            def _get_description(self) -> str:
                return "Integrate validated code into the main project"

            def _get_pydantic_schema(self) -> type[BaseModel]:
                return IntegrationToolParams

            def _get_category(self) -> str:
                return "integration"

            async def execute(self, **kwargs) -> ToolCallResult:
                start_time = datetime.now()
                result = integrate_code_into_project(
                    kwargs["file_path"], kwargs["content"], kwargs["commit_message"]
                )
                execution_time = (datetime.now() - start_time).total_seconds()

                return ToolCallResult(
                    command="integrate_code_into_project",
                    success=result["success"],
                    output=result.get("message", ""),
                    error_message=(
                        result.get("error") if not result["success"] else None
                    ),
                    execution_time=execution_time,
                )

            async def demo(self):
                return await self.execute(
                    file_path="demo.py",
                    content="# Demo file\nprint('Demo')",
                    commit_message="Demo integration",
                )

        class PartnerWorkspaceToolParams(BaseModel):
            code: str = Field(..., description="Python code to execute")
            partner_workspace_path: str = Field(
                ..., description="Path to partner workspace"
            )
            timeout: int = Field(30, description="Timeout in seconds")

        class PartnerWorkspaceTool(BaseTool):
            def _get_name(self) -> str:
                return "execute_python_code_in_partner_workspace"

            def _get_description(self) -> str:
                return "Execute code in partner workspace"

            def _get_pydantic_schema(self) -> type[BaseModel]:
                return PartnerWorkspaceToolParams

            def _get_category(self) -> str:
                return "execution"

            async def execute(self, **kwargs) -> ToolCallResult:
                start_time = datetime.now()
                # Import here to avoid import-time cycles; falls back if function missing
                try:
                    from crisalida_lib.ASTRAL_TOOLS.execution_tools import (
                        execute_python_code_in_partner_workspace,
                    )
                except Exception:

                    def execute_python_code_in_partner_workspace(
                        code: str, partner_workspace_path: str, timeout: int = 30
                    ) -> dict:
                        return {
                            "success": False,
                            "stdout": "",
                            "stderr": "execute_python_code_in_partner_workspace not available",
                            "return_code": 1,
                            "error": "NotAvailable",
                        }

                result = execute_python_code_in_partner_workspace(
                    kwargs["code"],
                    kwargs["partner_workspace_path"],
                    kwargs.get("timeout", 30),
                )
                execution_time = (datetime.now() - start_time).total_seconds()

                return ToolCallResult(
                    command="execute_python_code_in_partner_workspace",
                    success=result["success"],
                    output=result["stdout"],
                    error_message=result["stderr"] if not result["success"] else None,
                    execution_time=execution_time,
                )

            async def demo(self):
                return await self.execute(
                    code="print('Demo from partner workspace')",
                    partner_workspace_path="/tmp",
                )

        _global_tool_registry.register_tool(SandboxTool())
        _global_tool_registry.register_tool(IntegrationTool())
        _global_tool_registry.register_tool(PartnerWorkspaceTool())

        logger.info(
            "✅ Crisalida global ToolRegistry initialized and tools registered."
        )
    return _global_tool_registry


def register_all_crisalida_tools() -> ToolRegistry:
    return get_global_tool_registry()


def get_registered_tools() -> dict[str, Any]:
    registry = get_global_tool_registry()
    return registry.get_all_tools_info()


# Herramienta para integrar código (implementación robusta y segura)
def integrate_code_into_project(
    file_path: str, content: str, commit_message: str
) -> dict:
    """
    Integra un archivo de código en el proyecto principal.
    Implementación robusta con validación y backup de seguridad.
    """
    import os
    import shutil
    from datetime import datetime

    try:
        # Validación de entrada
        if not file_path or not content:
            return {"success": False, "error": "file_path and content are required"}

        # Convertir a ruta absoluta y validar que esté dentro del proyecto
        abs_file_path = os.path.abspath(file_path)
        project_root = os.path.abspath(os.getcwd())

        if not abs_file_path.startswith(project_root):
            return {
                "success": False,
                "error": f"File path {file_path} is outside project directory",
            }

        # Crear directorio padre si no existe
        parent_dir = os.path.dirname(abs_file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created directory: {parent_dir}")

        # Crear backup si el archivo ya existe
        backup_path = None
        if os.path.exists(abs_file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{abs_file_path}.backup_{timestamp}"
            shutil.copy2(abs_file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Escribir el contenido al archivo
        with open(abs_file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully wrote content to {abs_file_path}")

        # Intentar operaciones Git si git está disponible
        git_status = "Git operations not performed"
        try:
            # Verificar si estamos en un repositorio git
            result_git_check = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result_git_check.returncode == 0:
                try:
                    # Añadir el archivo al staging area
                    subprocess.run(
                        ["git", "add", abs_file_path], check=True, timeout=10
                    )

                    # Crear commit con el mensaje proporcionado
                    full_commit_message = f"{commit_message}\n\nFile: {file_path}\nIntegrated via Agent MEW"
                    subprocess.run(
                        ["git", "commit", "-m", full_commit_message],
                        check=True,
                        timeout=10,
                    )
                    git_status = "Git add and commit successful"
                    logger.info("Git operations completed successfully")
                except subprocess.CalledProcessError as e:
                    git_status = f"Git operations failed: {e}"
                    logger.warning(f"Git operations failed: {e}")
                except subprocess.TimeoutExpired:
                    git_status = "Git operations timed out"
                    logger.warning("Git operations timed out")
                except Exception as e:
                    git_status = f"Git operations error: {e}"
                    logger.warning(f"Git operations error: {e}")
            else:
                git_status = "Not in a git repository"

        except subprocess.CalledProcessError as e:
            git_status = f"Git check failed: {e}"
            logger.warning(f"Git check failed: {e}")
        except subprocess.TimeoutExpired:
            git_status = "Git check timed out"
            logger.warning("Git check timed out")
        except FileNotFoundError:
            git_status = "Git not available on system"
            logger.info("Git not found on system")
        except Exception as e:
            git_status = f"Git error: {e}"
            logger.warning(f"Git error: {e}")

        return {
            "success": True,
            "message": f"File {file_path} successfully integrated",
            "file_path": abs_file_path,
            "backup_path": backup_path,
            "git_status": git_status,
        }

    except PermissionError as e:
        return {"success": False, "error": f"Permission denied: {e}"}
    except OSError as e:
        return {"success": False, "error": f"File system error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
