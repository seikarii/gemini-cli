import asyncio

from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.file_system import GlobTool


async def demonstrate_glob_tool():
    """Demonstrates the GlobTool's functionality."""
    print("--- Demonstrating GlobTool ---")
    tool = GlobTool()

    # Example 1: Find all python files in the current directory
    print("\n--- Example 1: Find all python files in the current directory ---")
    result = await tool.execute(pattern="*.py")
    print(result.output)

    # Example 2: Find all markdown files recursively
    print("\n--- Example 2: Find all markdown files recursively ---")
    result = await tool.execute(pattern="**/*.md")
    print(result.output)

    return ToolCallResult(
        command="demo_glob",
        success=True,
        output="GlobTool demo completed.",
        execution_time=0.1,
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_glob_tool())
