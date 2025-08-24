"""
GEMINI-specific tools for agent-to-agent interaction and code modification.
These tools support the auto-evolutionary GEMINI agents system.
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime

from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class ExecutePythonCodeInPartnerWorkspaceArgs(BaseModel):
    """Arguments for execute_python_code_in_partner_workspace tool."""

    code: str = Field(..., description="Python code to execute")
    partner_workspace_path: str = Field(
        ..., description="Absolute path to partner's workspace"
    )
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class ExecutePythonCodeInPartnerWorkspaceTool(BaseTool):
    """Tool to execute Python code in partner agent's workspace for testing changes."""

    def _get_name(self) -> str:
        return "execute_python_code_in_partner_workspace"

    def _get_description(self) -> str:
        return "Execute Python code in a temporary script within partner's workspace for testing"

    def _get_category(self) -> str:
        return "gemini_execution"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return ExecutePythonCodeInPartnerWorkspaceArgs

    async def execute(self, **kwargs) -> ToolCallResult:
        """Execute Python code in partner's workspace."""
        try:
            # Validate input
            args = ExecutePythonCodeInPartnerWorkspaceArgs(**kwargs)
            code = args.code
            partner_workspace_path = args.partner_workspace_path
            timeout = args.timeout

            # Ensure partner workspace path is absolute
            if not os.path.isabs(partner_workspace_path):
                return ToolCallResult(
                    command="execute_python_code_in_partner_workspace",
                    success=False,
                    output="",
                    error_message="partner_workspace_path must be an absolute path",
                    execution_time=0.0,
                )

            # Create temporary script in partner's workspace
            script_path = os.path.join(
                partner_workspace_path,
                f"temp_exec_script_{os.getpid()}_{time.time_ns()}.py",
            )

            try:
                # Write code to temporary file
                with open(script_path, "w") as f:
                    f.write(code)

                # Execute the script
                start_time = datetime.now()
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )

                return ToolCallResult(
                    command="execute_python_code_in_partner_workspace",
                    success=result.returncode == 0,
                    output=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                    error_message=(
                        None
                        if result.returncode == 0
                        else f"Process exited with code {result.returncode}"
                    ),
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

            except subprocess.TimeoutExpired:
                return ToolCallResult(
                    command="execute_python_code_in_partner_workspace",
                    success=False,
                    output="",
                    error_message=f"Execution timed out after {timeout} seconds",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            finally:
                # Clean up temporary file
                if os.path.exists(script_path):
                    os.remove(script_path)

        except Exception as e:
            logger.error(f"Error in execute_python_code_in_partner_workspace: {e}")
            return ToolCallResult(
                command="execute_python_code_in_partner_workspace",
                success=False,
                output="",
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self) -> ToolCallResult:
        """Demo implementation for execute_python_code_in_partner_workspace tool."""
        demo_code = "print('Hello from partner workspace!')"
        demo_workspace = "/tmp"
        return await self.execute(
            code=demo_code, partner_workspace_path=demo_workspace, timeout=10
        )


class IntegrateCodeIntoProjectArgs(BaseModel):
    """Arguments for integrate_code_into_project tool."""

    file_path: str = Field(..., description="Absolute path to the file to modify")
    content: str = Field(..., description="New content for the file")
    commit_message: str = Field(..., description="Git commit message for the change")


class IntegrateCodeIntoProjectTool(BaseTool):
    """Tool to integrate tested code changes into the project."""

    def _get_name(self) -> str:
        return "integrate_code_into_project"

    def _get_description(self) -> str:
        return "Write code changes to a file and optionally commit them with git"

    def _get_category(self) -> str:
        return "gemini_integration"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return IntegrateCodeIntoProjectArgs

    async def execute(self, **kwargs) -> ToolCallResult:
        """Integrate code changes into the project."""
        start_time = datetime.now()
        try:
            # Validate input
            args = IntegrateCodeIntoProjectArgs(**kwargs)
            file_path = args.file_path
            content = args.content
            commit_message = args.commit_message

            # Write the file
            try:
                with open(file_path, "w") as f:
                    f.write(content)

                logger.info(f"Successfully wrote content to {file_path}")

                # For now, we'll just write the file without git operations
                # Git operations can be added later when the system is more mature
                return ToolCallResult(
                    command="integrate_code_into_project",
                    success=True,
                    output=f"File {file_path} written successfully. Commit message: {commit_message}",
                    error_message=None,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

            except Exception as write_error:
                return ToolCallResult(
                    command="integrate_code_into_project",
                    success=False,
                    output="",
                    error_message=f"Failed to write file: {write_error}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

        except Exception as e:
            logger.error(f"Error in integrate_code_into_project: {e}")
            return ToolCallResult(
                command="integrate_code_into_project",
                success=False,
                output="",
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self) -> ToolCallResult:
        """Demo implementation for integrate_code_into_project tool."""
        demo_file = "/tmp/demo_integration.py"
        demo_content = "# Demo integration\nprint('Hello from integrated code!')\n"
        demo_message = "Demo: Add sample integration code"
        return await self.execute(
            file_path=demo_file, content=demo_content, commit_message=demo_message
        )
