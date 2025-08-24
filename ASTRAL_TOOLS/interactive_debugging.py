import dis
import inspect
import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class DebuggingSchema(BaseModel):
    """Schema for interactive debugging tool parameters"""

    action: Literal[
        "start_session",
        "execute_code",
        "set_breakpoint",
        "step_into",
        "step_over",
        "continue",
        "inspect_variable",
        "get_stack",
        "get_locals",
        "get_globals",
        "evaluate_expression",
        "end_session",
        "disassemble",
        "get_source",
        "list_sessions",
    ] = Field(
        ...,
        description="Action to perform: 'start_session', 'execute_code', 'set_breakpoint', 'step_into', 'step_over', 'continue', 'inspect_variable', 'get_stack', 'get_locals', 'get_globals', 'evaluate_expression', 'end_session', 'disassemble', 'get_source', 'list_sessions'",
    )
    session_id: str | None = Field(None, description="Debug session identifier")
    code: str | None = Field(None, description="Python code to execute or debug")
    file_path: str | None = Field(None, description="Path to Python file to debug")
    line_number: int | None = Field(None, description="Line number for breakpoint")
    variable_name: str | None = Field(None, description="Variable name to inspect")
    expression: str | None = Field(None, description="Expression to evaluate")
    depth: int = Field(default=0, description="Stack frame depth (0 = current frame)")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        valid_actions = [
            "start_session",
            "execute_code",
            "set_breakpoint",
            "step_into",
            "step_over",
            "continue",
            "inspect_variable",
            "get_stack",
            "get_locals",
            "get_globals",
            "evaluate_expression",
            "end_session",
            "disassemble",
            "get_source",
            "list_sessions",
        ]
        if v not in valid_actions:
            raise ValueError(f"Action must be one of: {valid_actions}")
        return v


class DebugFrame:
    """Represents a debug frame with context"""

    def __init__(
        self, frame, filename: str, line_number: int, function_name: str
    ) -> None:
        self.frame = frame
        self.filename = filename
        self.line_number = line_number
        self.function_name = function_name
        self.locals = dict(frame.f_locals) if frame else {}
        self.globals = dict(frame.f_globals) if frame else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "locals_count": len(self.locals),
            "globals_count": len(self.globals),
        }


class DebugSession:
    """Manages a debugging session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.current_frame: DebugFrame | None = None
        self.stack_frames: list[DebugFrame] = []
        self.breakpoints: dict[str, list[int]] = {}  # file -> [line_numbers]
        self.execution_context: dict[str, Any] = {}
        self.last_exception: Exception | None = None
        self.is_active = True

    def add_breakpoint(self, file_path: str, line_number: int):
        """Add a breakpoint"""
        if file_path not in self.breakpoints:
            self.breakpoints[file_path] = []
        if line_number not in self.breakpoints[file_path]:
            self.breakpoints[file_path].append(line_number)
            self.breakpoints[file_path].sort()

    def remove_breakpoint(self, file_path: str, line_number: int):
        """Remove a breakpoint"""
        if file_path in self.breakpoints and line_number in self.breakpoints[file_path]:
            self.breakpoints[file_path].remove(line_number)
            if not self.breakpoints[file_path]:
                del self.breakpoints[file_path]

    def get_stack_info(self) -> list[dict[str, Any]]:
        """Get information about the current stack"""
        return [frame.to_dict() for frame in self.stack_frames]

    def update_frames_from_traceback(self, tb):
        """Update frames from a traceback"""
        self.stack_frames = []
        while tb:
            frame = DebugFrame(
                frame=tb.tb_frame,
                filename=tb.tb_frame.f_code.co_filename,
                line_number=tb.tb_lineno,
                function_name=tb.tb_frame.f_code.co_name,
            )
            self.stack_frames.append(frame)
            tb = tb.tb_next

        if self.stack_frames:
            self.current_frame = self.stack_frames[-1]


class InteractiveDebugger:
    """Custom debugger for programmatic control"""

    def __init__(self, session: DebugSession):
        self.session = session
        self.stepping = False
        self.continue_execution = False

    def trace_calls(self, frame, event, arg) -> Any:
        """Trace function for debugging"""
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name

        # Check for breakpoints
        if filename in self.session.breakpoints:
            if line_number in self.session.breakpoints[filename]:
                self.stepping = True
                self.continue_execution = False

        # Update current frame
        debug_frame = DebugFrame(frame, filename, line_number, function_name)
        self.session.current_frame = debug_frame

        if event == "call":
            self.session.stack_frames.append(debug_frame)
        elif event == "return":
            if self.session.stack_frames:
                self.session.stack_frames.pop()

        # If stepping or at breakpoint, pause execution
        if self.stepping and not self.continue_execution:
            # In a real implementation, this would pause execution
            # For this demo, we'll just log the state
            logger.info(f"Paused at {filename}:{line_number} in {function_name}")

        return self.trace_calls


class InteractiveDebuggingTool(BaseTool):
    """Interactive debugging tool for programmatic debugging"""

    def __init__(self):
        super().__init__()
        self.sessions: dict[str, DebugSession] = {}
        self.active_session: str | None = None

    def _get_name(self) -> str:
        return "interactive_debugging"

    def _get_description(self) -> str:
        return "Programmatic debugging interface with breakpoints, stepping, and variable inspection"

    def _get_category(self) -> str:
        return "debugging"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return DebuggingSchema

    async def execute(self, **kwargs) -> ToolCallResult:
        """Execute debugging operation"""
        start_time = datetime.now()

        try:
            action = kwargs.get("action")

            if action == "start_session":
                return await self._start_session(**kwargs)
            elif action == "execute_code":
                return await self._execute_code(**kwargs)
            elif action == "set_breakpoint":
                return await self._set_breakpoint(**kwargs)
            elif action == "inspect_variable":
                return await self._inspect_variable(**kwargs)
            elif action == "get_stack":
                return await self._get_stack(**kwargs)
            elif action == "get_locals":
                return await self._get_locals(**kwargs)
            elif action == "get_globals":
                return await self._get_globals(**kwargs)
            elif action == "evaluate_expression":
                return await self._evaluate_expression(**kwargs)
            elif action == "disassemble":
                return await self._disassemble(**kwargs)
            elif action == "get_source":
                return await self._get_source(**kwargs)
            elif action == "end_session":
                return await self._end_session(**kwargs)
            elif action == "list_sessions":
                return await self._list_sessions(**kwargs)
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Interactive debugging tool error: {e}")
            return ToolCallResult(
                command=f"interactive_debugging({action})",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def _start_session(self, **kwargs) -> ToolCallResult:
        """Start a new debugging session"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", f"debug_session_{len(self.sessions)}")

        if session_id in self.sessions:
            raise ValueError(f"Session '{session_id}' already exists")

        session = DebugSession(session_id)
        self.sessions[session_id] = session
        self.active_session = session_id

        output = f"Started debugging session: {session_id}"
        return ToolCallResult(
            command=f"start_session({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
            },
            error_message=None,
        )

    async def _execute_code(self, **kwargs) -> ToolCallResult:
        """Execute code in the debugging context"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        code_str = kwargs.get("code")

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        if not code_str:
            raise ValueError("Code is required")

        session = self.sessions[session_id]

        # Capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            # Prepare execution context
            exec_globals = session.execution_context.copy()
            exec_locals: dict[str, Any] = {}

            # Execute code with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # First try to compile as expression for eval
                    compiled = compile(code_str, "<debug_session>", "eval")
                    result = eval(compiled, exec_globals, exec_locals)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # If that fails, compile as statement for exec
                    compiled = compile(code_str, "<debug_session>", "exec")
                    exec(compiled, exec_globals, exec_locals)

            # Update session context
            session.execution_context.update(exec_globals)
            session.execution_context.update(exec_locals)

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            output = stdout_output
            if stderr_output:
                output += f"\nSTDERR: {stderr_output}"

            if not output:
                output = "Code executed successfully (no output)"

            return ToolCallResult(
                command=f"execute_code({session_id})",
                success=True,
                output=output,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "session_id": session_id,
                    "code": code_str,
                    "variables_updated": len(exec_locals),
                },
                error_message=None,
            )

        except Exception as e:
            session.last_exception = e

            # Try to get traceback information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback:
                session.update_frames_from_traceback(exc_traceback)

            stderr_output = stderr_capture.getvalue()
            error_msg = str(e)
            if stderr_output:
                error_msg += f"\nSTDERR: {stderr_output}"

            return ToolCallResult(
                command=f"execute_code({session_id})",
                success=False,
                output="",
                error_message=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "session_id": session_id,
                    "code": code_str,
                    "exception_type": exc_type.__name__ if exc_type else "Unknown",
                },
            )

    async def _set_breakpoint(self, **kwargs) -> ToolCallResult:
        """Set a breakpoint"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        file_path = kwargs.get("file_path")
        line_number = kwargs.get("line_number")

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        if not file_path or not line_number:
            raise ValueError("File path and line number are required")

        session = self.sessions[session_id]
        session.add_breakpoint(file_path, line_number)

        output = f"Set breakpoint at {file_path}:{line_number}"
        return ToolCallResult(
            command=f"set_breakpoint({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "session_id": session_id,
                "file_path": file_path,
                "line_number": line_number,
                "total_breakpoints": sum(
                    len(lines) for lines in session.breakpoints.values()
                ),
            },
            error_message=None,
        )

    async def _inspect_variable(self, **kwargs) -> ToolCallResult:
        """Inspect a variable in the current context"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        variable_name = kwargs.get("variable_name")

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        if not variable_name:
            raise ValueError("Variable name is required")

        session = self.sessions[session_id]

        # Look for variable in current context
        context = session.execution_context
        if session.current_frame:
            context = {
                **session.current_frame.locals,
                **session.current_frame.globals,
                **context,
            }

        if variable_name not in context:
            raise ValueError(f"Variable '{variable_name}' not found in current context")

        value = context[variable_name]

        # Get detailed information about the variable
        var_info = {
            "name": variable_name,
            "type": type(value).__name__,
            "value": repr(value),
            "size": len(str(value)),
            "is_callable": callable(value),
        }

        # Additional info for specific types
        if hasattr(value, "__dict__"):
            var_info["attributes"] = list(value.__dict__.keys())

        if hasattr(value, "__len__") and not isinstance(value, str):
            try:
                var_info["length"] = len(value)
            except (TypeError, AttributeError):
                pass

        if inspect.isfunction(value) or inspect.ismethod(value):
            try:
                sig = inspect.signature(value)
                var_info["signature"] = str(sig)
            except (ValueError, TypeError):
                pass

        output = f"Variable '{variable_name}': {var_info['type']} = {var_info['value'][:200]}{'...' if len(var_info['value']) > 200 else ''}"

        return ToolCallResult(
            command=f"inspect_variable({session_id}, {variable_name})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"session_id": session_id, "variable_info": var_info},
            error_message=None,
        )

    async def _get_stack(self, **kwargs) -> ToolCallResult:
        """Get the current call stack"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        session = self.sessions[session_id]
        stack_info = session.get_stack_info()

        output = f"Call stack ({len(stack_info)} frames):"
        for i, frame_info in enumerate(stack_info):
            output += f"\n  {i}: {frame_info['function_name']} at {frame_info['filename']}:{frame_info['line_number']}"

        return ToolCallResult(
            command=f"get_stack({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"session_id": session_id, "stack_frames": stack_info},
            error_message=None,
        )

    async def _get_locals(self, **kwargs) -> ToolCallResult:
        """Get local variables in current frame"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        kwargs.get("depth", 0)

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        session = self.sessions[session_id]

        locals_dict = {}
        if session.current_frame:
            locals_dict = session.current_frame.locals
        elif session.execution_context:
            # Use execution context as fallback
            locals_dict = {
                k: v
                for k, v in session.execution_context.items()
                if not k.startswith("__")
            }

        # Prepare summary
        locals_summary = {}
        for name, value in locals_dict.items():
            if not name.startswith("_"):  # Skip private variables
                locals_summary[name] = {
                    "type": type(value).__name__,
                    "value": repr(value)[:100]
                    + ("..." if len(repr(value)) > 100 else ""),
                }

        output = f"Local variables ({len(locals_summary)} visible):"
        for name, info in list(locals_summary.items())[:20]:  # Limit output
            output += f"\n  {name}: {info['type']} = {info['value']}"

        if len(locals_summary) > 20:
            output += f"\n  ... and {len(locals_summary) - 20} more"

        return ToolCallResult(
            command=f"get_locals({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"session_id": session_id, "locals": locals_summary},
            error_message=None,
        )

    async def _get_globals(self, **kwargs) -> ToolCallResult:
        """Get global variables"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        session = self.sessions[session_id]

        globals_dict = {}
        if session.current_frame:
            globals_dict = session.current_frame.globals

        # Filter to user-defined globals
        user_globals = {
            k: v
            for k, v in globals_dict.items()
            if not k.startswith("__") and not inspect.ismodule(v)
        }

        # Prepare summary
        globals_summary = {}
        for name, value in user_globals.items():
            globals_summary[name] = {
                "type": type(value).__name__,
                "value": repr(value)[:100] + ("..." if len(repr(value)) > 100 else ""),
            }

        output = f"Global variables ({len(globals_summary)} user-defined):"
        for name, info in list(globals_summary.items())[:20]:  # Limit output
            output += f"\n  {name}: {info['type']} = {info['value']}"

        if len(globals_summary) > 20:
            output += f"\n  ... and {len(globals_summary) - 20} more"

        return ToolCallResult(
            command=f"get_globals({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"session_id": session_id, "globals": globals_summary},
            error_message=None,
        )

    async def _evaluate_expression(self, **kwargs) -> ToolCallResult:
        """Evaluate an expression in the current context"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        expression = kwargs.get("expression")

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        if not expression:
            raise ValueError("Expression is required")

        session = self.sessions[session_id]

        try:
            # Prepare context
            context = session.execution_context.copy()
            if session.current_frame:
                context.update(session.current_frame.locals)
                context.update(session.current_frame.globals)

            # Evaluate expression
            result = eval(expression, context)

            output = f"Expression result: {repr(result)}"
            return ToolCallResult(
                command=f"evaluate_expression({session_id})",
                success=True,
                output=output,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "session_id": session_id,
                    "expression": expression,
                    "result": repr(result),
                    "result_type": type(result).__name__,
                },
                error_message=None,
            )

        except Exception as e:
            return ToolCallResult(
                command=f"evaluate_expression({session_id})",
                success=False,
                output="",
                error_message=f"Expression evaluation failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"session_id": session_id, "expression": expression},
            )

    async def _disassemble(self, **kwargs) -> ToolCallResult:
        """Disassemble code for debugging"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        code_str = kwargs.get("code")

        if not code_str:
            raise ValueError("Code is required for disassembly")

        try:
            # Compile the code
            compiled = compile(code_str, "<debug_code>", "exec")

            # Capture disassembly output
            output_capture = StringIO()
            dis.dis(compiled, file=output_capture)
            disassembly = output_capture.getvalue()

            output = f"Disassembly:\n{disassembly}"
            return ToolCallResult(
                command=f"disassemble({session_id})",
                success=True,
                output=output,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"session_id": session_id, "code": code_str},
                error_message=None,
            )

        except Exception as e:
            return ToolCallResult(
                command=f"disassemble({session_id})",
                success=False,
                output="",
                error_message=f"Disassembly failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def _get_source(self, **kwargs) -> ToolCallResult:
        """Get source code of a function or module"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)
        variable_name = kwargs.get("variable_name")

        if not variable_name:
            raise ValueError("Variable name is required")

        session = self.sessions[session_id] if session_id else None

        try:
            # Look for the object
            context = {}
            if session:
                context.update(session.execution_context)
                if session.current_frame:
                    context.update(session.current_frame.locals)
                    context.update(session.current_frame.globals)

            if variable_name not in context:
                # Try to import as module
                try:
                    obj = __import__(variable_name)
                except ImportError as e:
                    raise ValueError(
                        f"'{variable_name}' not found in context or as importable module"
                    ) from e
            else:
                obj = context[variable_name]

            # Get source code
            try:
                source = inspect.getsource(obj)
                lines = source.count("\n") + 1

                output = f"Source code for '{variable_name}' ({lines} lines):\n{source}"
                return ToolCallResult(
                    command=f"get_source({variable_name})",
                    success=True,
                    output=output,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "session_id": session_id,
                        "variable_name": variable_name,
                        "lines": lines,
                        "source": source,
                    },
                    error_message=None,
                )
            except OSError as e:
                raise ValueError(
                    f"Source code not available for '{variable_name}'"
                ) from e

        except Exception as e:
            return ToolCallResult(
                command=f"get_source({variable_name})",
                success=False,
                output="",
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def _end_session(self, **kwargs) -> ToolCallResult:
        """End a debugging session"""
        start_time = datetime.now()
        session_id = kwargs.get("session_id", self.active_session)

        if not session_id or session_id not in self.sessions:
            raise ValueError("Valid session ID required")

        session = self.sessions[session_id]
        session.is_active = False

        if self.active_session == session_id:
            self.active_session = None

        del self.sessions[session_id]

        output = f"Ended debugging session: {session_id}"
        return ToolCallResult(
            command=f"end_session({session_id})",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            error_message=None,
        )

    async def _list_sessions(self, **kwargs) -> ToolCallResult:
        """List active debugging sessions"""
        start_time = datetime.now()
        sessions_info = []
        for session_id, session in self.sessions.items():
            sessions_info.append(
                {
                    "session_id": session_id,
                    "created_at": session.created_at.isoformat(),
                    "is_active": session.is_active,
                    "has_exception": session.last_exception is not None,
                    "breakpoints_count": sum(
                        len(lines) for lines in session.breakpoints.values()
                    ),
                    "context_variables": len(session.execution_context),
                }
            )

        output = f"Active debugging sessions: {len(sessions_info)}"
        if sessions_info:
            for info in sessions_info:
                status = "🟢" if info["is_active"] else "🔴"
                exception_status = "⚠️" if info["has_exception"] else "✅"
                output += f"\n  {status} {info['session_id']} {exception_status} (breakpoints: {info['breakpoints_count']}, vars: {info['context_variables']})"

        return ToolCallResult(
            command="list_sessions",
            success=True,
            output=output,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"sessions": sessions_info, "active_session": self.active_session},
            error_message=None,
        )

    async def demo(self):
        """Demonstrate the interactive debugging tool's functionality."""
        print("🐛 INTERACTIVE DEBUGGING TOOL DEMO")
        print("=" * 40)

        # Create a simple debug session with sample code
        sample_code = """
def buggy_function(x, y):
    result = x / y  # Potential division by zero
    return result * 2

# This will cause an error
buggy_function(10, 0)
"""

        # Start a debug session
        result = await self.execute(
            action="start_session",
            session_id="demo_session",
            script_content=sample_code,
        )
        print(f"Start debug session: {result.success}")
        print(
            result.output[:200] + "..." if len(result.output) > 200 else result.output
        )

        # List sessions
        result = await self.execute(action="list_sessions")
        print(f"\nList sessions: {result.success}")
        print(result.output)

        # Stop the session
        result = await self.execute(action="stop_session", session_id="demo_session")
        print(f"\nStop session: {result.success}")
        print(result.output)

        print("\n✅ Interactive debugging demo completed!")
