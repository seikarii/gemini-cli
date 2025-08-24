#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.sub_llm_agent import SubLLMAgentTool


async def demo_sub_llm_agent():
    """Demuestra la herramienta sub-LLM agent"""
    print("🤖 SUB-LLM AGENT TOOL DEMO")
    print("=" * 40)
    tool = SubLLMAgentTool()
    result = await tool.execute(action="list_models")
    print(f"List models: {result.success}")
    print(result.output)
    print("\n✅ Sub-LLM agent demo completed!")
    print("Nota: La funcionalidad completa requiere un servidor Ollama en ejecución")


if __name__ == "__main__":
    asyncio.run(demo_sub_llm_agent())
