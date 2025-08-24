import asyncio
import inspect
from collections.abc import Coroutine
from typing import Any, cast


def run_maybe_awaitable(obj: Any) -> Any:
    """If obj is awaitable, run it synchronously (best-effort). Return result or None on failure."""
    if inspect.isawaitable(obj):
        try:
            # mypy wants a Coroutine for asyncio.run; cast to satisfy static checks
            return asyncio.run(cast(Coroutine[Any, Any, Any], obj))
        except RuntimeError:
            # Event loop already running or other runtime issue; return None conservatively
            return None
        except Exception:
            return None
    return obj


def compile_intention_safe(compiler: Any, intention: Any) -> Any:
    """Call compiler.compile_intention if available, handling awaitables and exceptions."""
    if compiler is None:
        return []
    compile_fn = getattr(compiler, "compile_intention", None)
    if not callable(compile_fn):
        return []
    try:
        res = compile_fn(intention)
    except Exception:
        return []
    return run_maybe_awaitable(res)
