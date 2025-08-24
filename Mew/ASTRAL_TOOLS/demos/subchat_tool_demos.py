#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.subchat_tool import SubchatTool


async def demo_subchat():
    """Demuestra la herramienta subchat"""
    print("💬 SUBCHAT TOOL DEMO")
    print("=" * 40)
    tool = SubchatTool()
    result = await tool.execute(
        action="send_message",
        message="Este es un comentario de prueba sobre la implementación actual",
        priority="normal",
        category="comment",
        context="Demo testing",
    )
    print(f"Send message: {result.success}")
    print(result.output)
    result = await tool.execute(action="get_messages", max_messages=5)
    print(f"\nGet messages: {result.success}")
    print(result.output[:300] + "..." if len(result.output) > 300 else result.output)
    result = await tool.execute(action="get_stats")
    print(f"\nStats: {result.success}")
    print(result.output)
    print("\n✅ Subchat demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_subchat())
