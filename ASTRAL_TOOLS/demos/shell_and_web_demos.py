#!/usr/bin/env python3
import asyncio
import os


async def demo_shell_web_tools():
    """Demuestra las herramientas shell y web"""
    print("🔧 SHELL AND WEB TOOLS DEMO")
    print("=" * 50)
    test_dir = "test_fs_tools"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, World!\nThis is a test file.\nLine 3 of the test.")
    from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry
    from crisalida_lib.ASTRAL_TOOLS.shell_and_web import (
        GoogleWebSearchTool,
        RunShellCommandTool,
        WebFetchTool,
    )

    registry = ToolRegistry()
    registry.register_tool(RunShellCommandTool())
    registry.register_tool(WebFetchTool())
    registry.register_tool(GoogleWebSearchTool())
    print(f"Registered tools: {registry.list_tools()}")
    print("\n--- Testing RunShellCommandTool ---")
    result = await registry.execute_tool(
        "run_shell_command",
        {
            "command": "echo 'Hello from shell!' && ls -la",
            "description": "Test echo and list command",
        },
    )
    print(f"Shell result success: {result.success}")
    print(f"Shell output:\n{result.output}")
    print("\n--- Testing Shell Security (should fail) ---")
    result = await registry.execute_tool(
        "run_shell_command",
        {
            "command": "echo $(whoami)",
            "description": "Test command substitution security",
        },
    )
    print(f"Security test result: {result.success} (should be False)")
    if result.error_message:
        print(f"Security error: {result.error_message}")
    print("\n--- Testing GoogleWebSearchTool ---")
    result = await registry.execute_tool(
        "google_web_search", {"query": "Python programming tutorial"}
    )
    print(f"Search result success: {result.success}")
    print(f"Search output (first 300 chars):\n{result.output[:300]}...")
    print("\n--- Testing WebFetchTool ---")
    result = await registry.execute_tool(
        "web_fetch",
        {
            "prompt": "Please fetch and summarize the content from https://httpbin.org/json and analyze the JSON structure"
        },
    )
    print(f"Fetch result success: {result.success}")
    if result.success:
        print(f"Fetch output (first 500 chars):\n{result.output[:500]}...")
    else:
        print(f"Fetch error: {result.error_message}")
    print("\n✅ Shell and web tools demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_shell_web_tools())
