"""
Code Generation Tool
"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.code_generation.actions import CodeGenerationActions
from crisalida_lib.ASTRAL_TOOLS.code_generation.models import CodeGenerationParameters

logger = logging.getLogger(__name__)


class CodeGenerationTool(BaseTool):
    """
    Advanced code generation tool with template-driven, semantic context-aware capabilities.
    Integrates workspace analysis, linting, and project best practices.
    """

    def __init__(self):
        super().__init__()
        self.actions = CodeGenerationActions()

    def _get_name(self) -> str:
        return "code_generation"

    def _get_description(self) -> str:
        return (
            "Advanced code generation and templating tool with semantic context awareness.\n"
            "Capabilities:\n"
            "- Generate code from proven templates\n"
            "- Create complete project structures\n"
            "- Analyze code context for intelligent generation\n"
            "- Suggest refactoring improvements\n"
            "- Generate comprehensive documentation\n"
            "- Support multiple programming languages\n"
            "- Semantic understanding of development context\n"
            "- Integrates with workspace analysis and linting tools\n"
        )

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return CodeGenerationParameters

    def _get_category(self) -> str:
        return "code_generation"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Execute code generation operations"""
        start_time = datetime.now()
        try:
            action = kwargs.get("action")
            if action is None:
                return ToolCallResult(
                    command="code_generation",
                    success=False,
                    output="",
                    error_message="Action parameter is required",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            handler = getattr(self.actions, action, None)
            if handler and callable(handler):
                return await handler(**kwargs)
            else:
                return ToolCallResult(
                    command="code_generation",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {action}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

        except Exception as e:
            logger.error(f"Code generation tool execution failed: {e}")
            return ToolCallResult(
                command="code_generation",
                success=False,
                output="",
                error_message=f"Code generation error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrate CodeGenerationTool functionality."""
        print("🚀 CODE GENERATION TOOL DEMO")
        print("=" * 45)

        print("This tool generates code from templates and semantic context")
        print("Available actions: generate_code, analyze_context, create_project")

        # Simple demo without external dependencies
        print("✅ Tool initialized and ready for code generation")
        print("📝 Use with action, language, and template_name parameters")

        return ToolCallResult(
            command="code_generation_demo",
            success=True,
            output="CodeGenerationTool demo completed - tool ready for use",
            execution_time=0.1,
            error_message=None,
        )
