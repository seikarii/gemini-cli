import asyncio
import os

from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.file_system import ListDirectoryTool


async def demonstrate_list_directory_tool():
    """Demonstrates the ListDirectoryTool's functionality."""
    print("--- Demonstrating ListDirectoryTool ---")
    tool = ListDirectoryTool()

    # Example 1: List the current directory
    print("\n--- Example 1: List the current directory ---")
    current_directory = os.getcwd()
    result = await tool.execute(path=current_directory)
    print(result.output)

    # Example 2: List a subdirectory
    print("\n--- Example 2: List a subdirectory ---")
    subdirectory = os.path.join(current_directory, "crisalida_lib")
    if os.path.exists(subdirectory):
        result = await tool.execute(path=subdirectory)
        print(result.output)
    else:
        print(f"Subdirectory not found: {subdirectory}")

    return ToolCallResult(
        command="demo_list_directory",
        success=True,
        output="ListDirectoryTool demo completed.",
        execution_time=0.1,
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_list_directory_tool())
