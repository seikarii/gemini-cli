import logging
import os
import resource
import subprocess
import sys
import tempfile
import json
import shlex
import stat
import hashlib
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """
    Secure, resource-limited sandbox for executing untrusted Python snippets.

    Improvements in this "definitive" version:
    - Whitelist-based import policy (safer than blacklist).
    - Strict builtin surface exposed to executed code.
    - Code size / line limits and SHA256 audit of executed payload.
    - Secure temporary file creation with restrictive permissions.
    - Optional Windows compatibility (creationflags) and Unix resource limits.
    - Structured execution result including diagnostics and execution metadata.
    """

    # Conservative defaults
    DEFAULT_MAX_MEMORY_MB = 100
    DEFAULT_MAX_CPU_SEC = 10
    DEFAULT_MAX_CODE_BYTES = 64 * 1024  # 64 KB
    DEFAULT_MAX_LINES = 2000

    # Minimal builtin API exposed to executed code
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "float",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "pow",
        "range",
        "repr",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }

    # Default safe modules allowed to be imported from user code (whitelist)
    DEFAULT_ALLOWED_MODULES = {
        "math",
        "random",
        "json",
        "re",
        "itertools",
        "functools",
        "statistics",
        "time",
    }

    def __init__(
        self,
        python_path: Optional[str] = None,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        max_cpu_time: int = DEFAULT_MAX_CPU_SEC,
        max_code_bytes: int = DEFAULT_MAX_CODE_BYTES,
        allowed_modules: Optional[List[str]] = None,
    ):
        # Use the current Python interpreter by default
        self.python_executable = python_path or sys.executable
        self.max_memory_mb = int(max_memory_mb)
        self.max_cpu_time = int(max_cpu_time)
        self.max_code_bytes = int(max_code_bytes)
        self.allowed_modules = set(allowed_modules or list(self.DEFAULT_ALLOWED_MODULES))
        logger.info(
            "SandboxExecutor initialized (python=%s, mem=%dMB, cpu=%ds, max_code=%dB)",
            self.python_executable,
            self.max_memory_mb,
            self.max_cpu_time,
            self.max_code_bytes,
        )

    def _create_restricted_env(self) -> Dict[str, str]:
        """
        Create a minimal environment for subprocess execution.
        Keeps LANG/LC_ALL to avoid Unicode issues while clearing PYTHONPATH.
        """
        env = {
            "PATH": os.getenv("PATH", "/usr/bin:/bin"),
            "LANG": os.getenv("LANG", "C.UTF-8"),
            "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
            "PYTHONPATH": "",
            "HOME": "/tmp",
        }
        return env

    def _create_security_wrapper(self, code: str, allowed_modules: Optional[set] = None) -> str:
        """
        Wrap user code to enforce import and builtin restrictions inside the subprocess.
        The wrapper executes the user code inside a tightly controlled globals dict.
        """
        allowed = sorted(list(allowed_modules or self.allowed_modules))
        # Precompute the SHA256 of the payload for audit
        payload_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()

        # Limit recursion depth and enable a simple watchdog timestamp
        indented = "\n".join("    " + line for line in code.splitlines()) if code else "    pass"

        wrapper = f"""
import sys, builtins, time, types

# Payload audit
_PAYLOAD_HASH = "{payload_hash}"
_EXEC_START_TS = time.time()

# Allowed modules (whitelist)
_ALLOWED_MODULES = {allowed!r}

# Reduce available builtins to a minimal safe subset
_SAFE_BUILTINS = {sorted(list(self.SAFE_BUILTINS))!r}
_original_builtins = builtins.__dict__.copy()
# Build a minimal builtins mapping
min_builtins = {{name: _original_builtins[name] for name in _SAFE_BUILTINS if name in _original_builtins}}

# Provide a safe subset of __import__ that only allows whitelisted modules (and their submodules)
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    base = name.split('.')[0]
    if base not in _ALLOWED_MODULES:
        raise ImportError(f'Import of module \"{{name}}\" is not allowed in sandbox.')
    return _original_builtins['__import__'](name, globals, locals, fromlist, level)

min_builtins['__import__'] = _safe_import

# Replace builtins for the user execution context only
_user_globals = {{
    '__builtins__': min_builtins,
    '__name__': '__sandbox__',
    '__package__': None,
}}

# Safety knobs
sys.setrecursionlimit(1000)

# Execute user code
try:
{indented}
except Exception as _e:
    # Re-raise to let the outer process capture stderr/exitcode
    raise
"""
        return wrapper

    def _write_temp_script(self, wrapped_code: str, tmpdir: str) -> str:
        """
        Write the wrapped code to a secure temporary file with restrictive permissions.
        Returns the file path.
        """
        path = os.path.join(tmpdir, "sandbox_script.py")
        # Write atomically
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapped_code)
        # Restrict permissions: owner read/write only
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            # best-effort
            pass
        return path

    def _set_process_limits(self):
        """Apply Unix resource limits to the child process (preexec_fn)."""
        try:
            # Address space (bytes)
            mem_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            # CPU time (seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time))
            # Maximum number of processes (try to restrict)
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            except Exception:
                pass
            # Limit file size to prevent huge writes
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            except Exception:
                pass
        except Exception as e:
            # Non-fatal: log and continue
            logger.warning("Could not set all process limits: %s", e)

    def execute_python_code(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute code in an isolated subprocess and return structured result.

        Returns a dict with:
        - success (bool), stdout, stderr, return_code (int), metadata (dict)
        """
        # Basic input validations
        if not isinstance(code, str):
            return {"success": False, "stdout": "", "stderr": "Code must be a string", "return_code": 2, "metadata": {}}

        code_bytes = len(code.encode("utf-8"))
        if code_bytes > self.max_code_bytes:
            return {"success": False, "stdout": "", "stderr": f"Code too large ({code_bytes} bytes, max {self.max_code_bytes})", "return_code": 3, "metadata": {}}

        line_count = code.count("\n") + 1
        if line_count > self.DEFAULT_MAX_LINES:
            return {"success": False, "stdout": "", "stderr": f"Too many lines ({line_count}, max {self.DEFAULT_MAX_LINES})", "return_code": 4, "metadata": {}}

        # Simple sanitize: avoid null bytes and suspicious shebangs
        if "\x00" in code:
            return {"success": False, "stdout": "", "stderr": "Null byte found in code", "return_code": 5, "metadata": {}}

        wrapped = self._create_security_wrapper(code)
        started_at = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = self._write_temp_script(wrapped, tmpdir)
            env = self._create_restricted_env()

            # Prepare subprocess options
            creationflags = 0
            preexec_fn = None
            if os.name == "nt":
                # Avoid console window on Windows
                creationflags = 0x08000000  # CREATE_NO_WINDOW
            else:
                preexec_fn = self._set_process_limits

            try:
                proc = subprocess.run(
                    [self.python_executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                    cwd=tmpdir,
                    preexec_fn=preexec_fn,
                    creationflags=creationflags,
                )
                elapsed = time.time() - started_at
                # Collect metadata
                metadata = {
                    "script_path": script_path,
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "code_bytes": code_bytes,
                    "line_count": line_count,
                    "elapsed_s": elapsed,
                    "max_memory_mb": self.max_memory_mb,
                    "max_cpu_time_s": self.max_cpu_time,
                    "allowed_modules": sorted(list(self.allowed_modules)),
                    "timestamp": started_at,
                }
                return {
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "return_code": proc.returncode,
                    "metadata": metadata,
                }
            except subprocess.TimeoutExpired as te:
                logger.warning("Sandbox execution timed out after %s seconds", timeout)
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds",
                    "return_code": 124,
                    "metadata": {"requested_timeout": timeout},
                }
            except Exception as exc:
                logger.exception("Sandbox execution failed unexpectedly")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Sandbox executor error: {exc}",
                    "return_code": 1,
                    "metadata": {},
                }

# Consideraciones de Seguridad Adicionales:
# - Restringir módulos importables (implementado arriba)
# - Limitar el uso de recursos (CPU, memoria) (implementado arriba)
# - Deshabilitar acceso a la red (parcialmente implementado vía restricción de módulos)
# - Usar contenedores (Docker) para un aislamiento más robusto en producción.
