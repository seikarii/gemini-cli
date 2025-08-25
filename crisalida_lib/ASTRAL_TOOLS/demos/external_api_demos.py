#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.external_api import ExternalAPITool


async def demo_external_api():
    """Demo de la herramienta ExternalAPITool"""
    print("🌐 EXTERNAL API TOOL DEMO")
    print("=" * 40)
    tool = ExternalAPITool()
    result = await tool.execute(
        action="test_connection", url="https://httpbin.org/status/200"
    )
    print(f"Connection test: {result.success}")
    print(result.output)
    result = await tool.execute(
        action="make_request", url="https://httpbin.org/json", method="GET"
    )
    print(f"\nGET request: {result.success}")
    print(result.output[:300] + "..." if len(result.output) > 300 else result.output)
    print("\n✅ External API demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_external_api())
