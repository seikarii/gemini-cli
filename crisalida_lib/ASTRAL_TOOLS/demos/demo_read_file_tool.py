import asyncio
import os

from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.file_system import ReadFileTool, WriteFileTool


async def demonstrate_read_file_tool():
    """Demonstrates the ReadFileTool's functionality."""
    print("--- Demonstrating ReadFileTool ---")
    read_tool = ReadFileTool()
    write_tool = WriteFileTool()

    # Create a dummy file to read
    dummy_file_path = os.path.abspath("dummy_file_for_read_demo.txt")
    dummy_content = "Hello, this is a test file for the ReadFileTool demo.\n" * 5
    await write_tool.execute(file_path=dummy_file_path, content=dummy_content)

    # Example 1: Read the entire file
    print("\n--- Example 1: Read the entire file ---")
    result = await read_tool.execute(absolute_path=dummy_file_path)
    print(result.output)

    # Example 2: Read a slice of the file
    print("\n--- Example 2: Read a slice of the file ---")
    result = await read_tool.execute(absolute_path=dummy_file_path, offset=1, limit=2)
    print(result.output)

    # Clean up the dummy file
    os.remove(dummy_file_path)

    return ToolCallResult(
        command="demo_read_file",
        success=True,
        output="ReadFileTool demo completed.",
        execution_time=0.1,
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_read_file_tool())
