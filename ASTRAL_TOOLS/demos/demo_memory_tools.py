import asyncio

from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.memory_tools import (
    ClearMemoryTool,
    RecallMemoryTool,
    SaveMemoryTool,
)


class MockMemorySystem:
    def __init__(self):
        self.core_principles = []

    def add_core_principle(self, principle):
        self.core_principles.append(principle)


async def demo_memory_tools():
    """Demonstrates the functionality of memory tools."""
    print("--- Demonstrating Memory Tools ---")
    memory_system = MockMemorySystem()
    save_tool = SaveMemoryTool(agent_memory_system=memory_system)
    recall_tool = RecallMemoryTool(agent_memory_system=memory_system)
    clear_tool = ClearMemoryTool(agent_memory_system=memory_system)

    # 1. Save some facts
    print("\n--- Saving facts ---")
    await save_tool.execute(fact="The user's name is John Doe.")
    await save_tool.execute(fact="The user's favorite color is blue.")
    await save_tool.execute(fact="The user lives in New York.")
    print(f"Memory content: {memory_system.core_principles}")

    # 2. Recall facts
    print("\n--- Recalling facts (keyword search) ---")
    result = await recall_tool.execute(query="user")
    print(result.output)

    print("\n--- Recalling facts (semantic search) ---")
    result = await recall_tool.execute(
        query="What is the user's name?", search_type="semantic"
    )
    print(result.output)

    # 3. Clear a fact
    print("\n--- Clearing a fact ---")
    await clear_tool.execute(fact_pattern="color", confirm=True)
    print(f"Memory content after clearing: {memory_system.core_principles}")

    return ToolCallResult(
        command="demo_memory_tools",
        success=True,
        output="Memory tools demo completed.",
        execution_time=0.1,
    )


if __name__ == "__main__":
    asyncio.run(demo_memory_tools())
