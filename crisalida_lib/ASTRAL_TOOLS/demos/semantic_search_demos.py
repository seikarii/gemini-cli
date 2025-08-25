#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.semantic_search import SemanticSearchTool


async def demo_semantic_search():
    """Demuestra la herramienta de búsqueda semántica"""
    print("🔍 SEMANTIC SEARCH TOOL DEMO")
    print("=" * 40)
    tool = SemanticSearchTool()
    result = await tool.execute(action="index_codebase", query="", path=".")
    print(f"Indexing: {result.success}")
    print(result.output)
    result = await tool.execute(
        action="search_functions", query="execute", max_results=3
    )
    print(f"\nFunction search: {result.success}")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    print("\n✅ Semantic search demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_semantic_search())
