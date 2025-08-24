#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.validation_tools import (
    LinterTool,
    TesterTool,
    TypeCheckerTool,
)


async def demo_linter_tool():
    print("🔧 LINTER TOOL DEMO")
    print("=" * 50)

    linter = LinterTool()
    result = await linter.safe_execute(
        file_path="crisalida_lib/tools/validation_tools.py"
    )
    print(f"Result: {result}")
    print(f"Output: {result.output}")

    print("\n✅ Linter tool demo completed!")


async def demo_tester_tool():
    print("🧪 TESTER TOOL DEMO")
    print("=" * 50)

    tester = TesterTool()
    result = await tester.safe_execute(test_path="tests/")
    print(f"Result: {result}")
    print(f"Output: {result.output}")

    print("\n✅ Tester tool demo completed!")


async def demo_type_checker_tool():
    print("🔍 TYPE CHECKER TOOL DEMO")
    print("=" * 50)

    type_checker = TypeCheckerTool()
    result = await type_checker.safe_execute(
        file_path="crisalida_lib/tools/validation_tools.py"
    )
    print(f"Result: {result}")
    print(f"Output: {result.output}")

    print("\n✅ Type Checker tool demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_linter_tool())
    asyncio.run(demo_tester_tool())
    asyncio.run(demo_type_checker_tool())
