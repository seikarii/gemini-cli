"""
Data Tools - A collection of tools for data manipulation and retrieval.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class ReturnDataParams(BaseModel):
    """Parameters for the ReturnDataTool."""

    data: dict[str, Any] = Field(..., description="The JSON data to be returned.")
    reason: str = Field(
        "", description="An optional reason or context for the data return."
    )


class ReturnDataTool(BaseTool):
    """A simple tool to return structured data from a plan."""

    def _get_name(self) -> str:
        return "return_data"

    def _get_description(self) -> str:
        return "Returns a structured JSON object to the calling agent. Use this as the final step in a plan to provide a result."

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return ReturnDataParams

    def _get_category(self) -> str:
        return "data"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Executes the data return, packaging the data in the output."""
        try:
            data_to_return = kwargs.get("data", {})
            # The primary output is the JSON string of the data.
            output_json = json.dumps(data_to_return)

            return ToolCallResult(
                command=self._get_name(),
                success=True,
                output=output_json,
                error_message=None,
                execution_time=0.0,
            )
        except TypeError as e:
            logger.error(f"Error serializing data in return_data_tool: {e}")
            return ToolCallResult(
                command=self._get_name(),
                success=False,
                output="",
                error_message=f"Data is not JSON serializable: {e}",
                execution_time=0.0,
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred in return_data_tool: {e}")
            return ToolCallResult(
                command=self._get_name(),
                success=False,
                output="",
                error_message=f"An unexpected error occurred: {e}",
                execution_time=0.0,
            )

    async def demo(self):
        """Demonstrates the ReturnDataTool."""
        print("🚀 ReturnDataTool Demo")
        demo_data = {"key": "value", "number": 123}
        print(f"Returning demo data: {demo_data}")
        result = await self.execute(data=demo_data)
        print(f"Execution result: {result}")
        assert result.success
        assert result.output == json.dumps(demo_data)
        print("✅ Demo completed successfully.")
        return result
