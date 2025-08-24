import os
import shutil

from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry
from crisalida_lib.ASTRAL_TOOLS.file_system import (
    GlobTool,
    ListDirectoryTool,
    ReadFileTool,
    ReadManyFilesTool,
    ReplaceTool,
    SearchFileContentTool,
    WriteFileTool,
)


async def demo_file_system_tools():
    """Demonstrate file system tools functionality"""
    print("📁 FILE SYSTEM TOOLS DEMO")
    print("=" * 50)
    # Create temporary test files
    test_dir = "test_fs_tools"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, World!\nThis is a test file.\nLine 3 of the test.")
    # Test all tools
    registry = ToolRegistry()
    registry.register_tool(WriteFileTool())
    registry.register_tool(ReadFileTool())
    registry.register_tool(ListDirectoryTool())
    registry.register_tool(GlobTool())
    registry.register_tool(SearchFileContentTool())
    registry.register_tool(ReplaceTool())
    registry.register_tool(ReadManyFilesTool())
    print(f"Registered tools: {registry.list_tools()}")
    # Test read file
    print("\n--- Testing ReadFileTool ---")
    result = await registry.execute_tool(
        "read_file", absolute_path=os.path.abspath(test_file)
    )
    print(f"Result: {result}")
    # Test list directory
    print("\n--- Testing ListDirectoryTool ---")
    result = await registry.execute_tool(
        "list_directory", path=os.path.abspath(test_dir)
    )
    print(f"Result: {result}")
    # Test glob
    print("\n--- Testing GlobTool ---")
    result = await registry.execute_tool(
        "glob", pattern="*.txt", path=os.path.abspath(test_dir)
    )
    print(f"Result: {result}")
    # Test search content
    print("\n--- Testing SearchFileContentTool ---")
    result = await registry.execute_tool(
        "search_file_content", pattern="test", path=os.path.abspath(test_dir)
    )
    print(f"Result: {result}")
    # Test replace
    print("\n--- Testing ReplaceTool ---")
    result = await registry.execute_tool(
        "replace",
        file_path=os.path.abspath(test_file),
        old_string="Hello, World!",
        new_string="Greetings, Universe!",
    )
    print(f"Result: {result}")
    # Clean up
    shutil.rmtree(test_dir)
    print("\n✅ File system tools demo completed!")
