import logging

from crisalida_lib.ASTRAL_TOOLS.interactive_debugging import InteractiveDebuggingTool

logger = logging.getLogger(__name__)


async def demo_interactive_debugging():
    """Demonstrate the interactive debugging tool"""
    print("🐛 INTERACTIVE DEBUGGING TOOL DEMO")
    print("=" * 50)

    tool = InteractiveDebuggingTool()

    # Start debugging session
    print("\n1. Starting debugging session...")
    result = await tool.execute(action="start_session", session_id="demo_session")
    print(f"Start session: {result.output}")

    # Execute some code
    print("\n2. Executing code...")

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    test_code = """
result = factorial(5)
x = 42
my_list = [1, 2, 3, 4, 5]
"""
    result_exec = await tool.execute(  # Renamed result to result_exec to avoid conflict
        action="execute_code", session_id="demo_session", code=test_code
    )
    print(f"Execute code: {result_exec.output}")

    # Explicitly add variables to the session's execution context for inspection
    # This is a workaround for demo purposes, as exec() scope is isolated
    session = tool.sessions["demo_session"]
    session.execution_context["factorial"] = factorial  # Directly assign the function
    session.execution_context["result"] = eval("result", session.execution_context)
    session.execution_context["x"] = eval("x", session.execution_context)
    session.execution_context["my_list"] = eval("my_list", session.execution_context)

    # Inspect variables
    print("\n3. Inspecting variables...")
    result = await tool.execute(
        action="inspect_variable", session_id="demo_session", variable_name="result"
    )
    print(f"Inspect 'result': {result.output}")

    result = await tool.execute(
        action="inspect_variable", session_id="demo_session", variable_name="factorial"
    )
    print(f"Inspect 'factorial': {result.output}")

    # Get locals
    print("\n4. Getting local variables...")
    result = await tool.execute(action="get_locals", session_id="demo_session")
    print(f"Locals: {result.output}")

    # Evaluate expression
    print("\n5. Evaluating expression...")
    result = await tool.execute(
        action="evaluate_expression",
        session_id="demo_session",
        expression="x * 2 + len(my_list)",
    )
    print(f"Expression result: {result.output}")

    # Get source code
    print("\n6. Getting source code...")
    result = await tool.execute(
        action="get_source", session_id="demo_session", variable_name="factorial"
    )
    print(f"Source code: {result.output}")

    # Test error handling
    print("\n7. Testing error handling...")
    result = await tool.execute(
        action="execute_code", session_id="demo_session", code="undefined_variable + 1"
    )
    print(f"Error test: {result.error_message}")

    # End session
    print("\n8. Ending session...")
    result = await tool.execute(action="end_session", session_id="demo_session")
    print(f"End session: {result.output}")

    print("\n✅ Interactive debugging tool demo completed!")
