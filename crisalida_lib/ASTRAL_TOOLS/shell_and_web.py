import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, cast
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field, validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

PRIVATE_IP_RANGES = [
    re.compile(r"^10\."),
    re.compile(r"^127\."),
    re.compile(r"^172\.(1[6-9]|2[0-9]|3[0-1])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^::1$"),
    re.compile(r"^fc00:"),
    re.compile(r"^fe80:"),
]

# FIXED: Enhanced security patterns for command injection detection
COMMAND_SUBSTITUTION_PATTERNS = [
    re.compile(r"\$\("),  # Command substitution $()
    re.compile(r"\$`"),   # Backtick substitution $`
    re.compile(r"`"),     # Backtick substitution
    re.compile(r"\$\{"),  # Parameter expansion that could lead to injection
    re.compile(r"<\("),   # Process substitution <()
    re.compile(r">\("),   # Process substitution >()
    re.compile(r"\|\s*\w+"),  # Pipe to commands (basic detection)
    re.compile(r";\s*\w+"),   # Command chaining with semicolon
    re.compile(r"&&\s*\w+"),  # Command chaining with &&
    re.compile(r"\|\|\s*\w+"), # Command chaining with ||
    re.compile(r">\s*/"),     # Output redirection to filesystem
    re.compile(r">>\s*/"),    # Append redirection to filesystem
    re.compile(r"<\s*/"),     # Input redirection from filesystem
]

# FIXED: Strict whitelist of allowed commands for enhanced security
ALLOWED_COMMANDS = {
    # File operations (safe subset)
    "ls", "cat", "head", "tail", "wc", "file", "stat", "find", "grep", "sort", "uniq",
    # Directory operations
    "pwd", "mkdir", "rmdir", 
    # Text processing
    "echo", "printf", "cut", "awk", "sed",
    # Development tools
    "git", "python", "python3", "pip", "pip3", "npm", "node", "mypy", "pytest", "black", "flake8",
    # System info (read-only)
    "whoami", "id", "uname", "date", "uptime", "df", "du", "ps",
    # Package managers (limited)
    "apt", "yum", "brew",
}

# Commands that require additional validation
RESTRICTED_COMMANDS = {
    "rm", "mv", "cp", "chmod", "chown", "sudo", "su", "systemctl", "service",
    "kill", "killall", "pkill", "nohup", "screen", "tmux",
    "curl", "wget", "ssh", "scp", "rsync", "nc", "netcat",
}

# Commands that are completely forbidden
FORBIDDEN_COMMANDS = {
    "dd", "fdisk", "parted", "mkfs", "mount", "umount", "crontab",
    "iptables", "ufw", "firewall-cmd", "setenforce", "setsid",
    "exec", "eval", "source", ".", "bash", "sh", "zsh", "fish",
}


def is_private_ip(url: str) -> bool:
    """Check if URL contains a private IP address"""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        return any(pattern.search(hostname) for pattern in PRIVATE_IP_RANGES)
    except Exception:
        return False


def detect_command_injection(command: str) -> List[str]:
    """
    Enhanced detection of command injection patterns.
    Returns list of detected threats for detailed error reporting.
    """
    threats = []
    
    # Check for command substitution patterns
    for pattern in COMMAND_SUBSTITUTION_PATTERNS:
        if pattern.search(command):
            threats.append(f"Command injection pattern detected: {pattern.pattern}")
    
    # Check for encoded characters that could bypass filters
    suspicious_encodings = [
        r'\\x[0-9a-fA-F]{2}',  # Hex encoding
        r'\\[0-7]{3}',         # Octal encoding
        r'%[0-9a-fA-F]{2}',    # URL encoding
        r'\$\'[^\']*\'',       # ANSI-C quoting
    ]
    
    for encoding_pattern in suspicious_encodings:
        if re.search(encoding_pattern, command):
            threats.append(f"Suspicious encoding detected: {encoding_pattern}")
    
    # Check for newline injection
    if '\n' in command or '\r' in command:
        threats.append("Newline injection detected")
    
    # Check for null byte injection
    if '\x00' in command:
        threats.append("Null byte injection detected")
    
    return threats


def validate_command_whitelist(command: str) -> Optional[str]:
    """
    Validate command against whitelist and restricted/forbidden lists.
    Returns error message if command is not allowed, None if valid.
    """
    # Parse the command to get the root command
    try:
        # Use shlex to properly parse the command
        parsed_cmd = shlex.split(command)
        if not parsed_cmd:
            return "Empty command not allowed"
        
        root_cmd = parsed_cmd[0]
        
        # Remove path components to get base command
        base_cmd = os.path.basename(root_cmd)
        
        # Check forbidden commands first
        if base_cmd in FORBIDDEN_COMMANDS:
            return f"Command '{base_cmd}' is forbidden for security reasons"
        
        # Check if command is in restricted list (requires additional validation)
        if base_cmd in RESTRICTED_COMMANDS:
            return f"Command '{base_cmd}' is restricted and requires manual approval"
        
        # Check if command is in allowed list
        if base_cmd not in ALLOWED_COMMANDS:
            return f"Command '{base_cmd}' is not in the allowed command whitelist"
        
        return None
        
    except ValueError as e:
        return f"Invalid command syntax: {e}"


def canonicalize_and_validate_directory(directory: str, project_root: str) -> tuple[Optional[str], Optional[str]]:
    """
    Canonicalize and validate directory path with strict containment checking.
    
    Args:
        directory: Relative directory path to validate
        project_root: Absolute path to project root
    
    Returns:
        Tuple of (canonical_path, error_message). If validation fails, path is None.
    """
    try:
        # Convert to Path objects for robust handling
        project_root_path = Path(project_root).resolve()
        
        # Construct target path
        target_path = project_root_path / directory
        
        # Canonicalize the path (resolve symlinks, normalize)
        canonical_path = target_path.resolve()
        
        # Strict containment check: ensure canonical path is within project root
        try:
            canonical_path.relative_to(project_root_path)
        except ValueError:
            return None, f"Directory '{directory}' attempts to escape project root"
        
        # Verify the directory exists
        if not canonical_path.exists():
            return None, f"Directory '{directory}' does not exist"
        
        if not canonical_path.is_dir():
            return None, f"Path '{directory}' is not a directory"
        
        return str(canonical_path), None
        
    except Exception as e:
        return None, f"Directory validation failed: {str(e)}"


def classify_error_type(exit_code: int, stderr: str, command: str) -> str:
    """
    Classify the type of error based on exit code and stderr content.
    """
    if exit_code == 0:
        return "success"
    
    stderr_lower = stderr.lower()
    
    # Permission errors
    if exit_code == 126 or "permission denied" in stderr_lower or "not permitted" in stderr_lower:
        return "permission_denied"
    
    # Command not found
    if exit_code == 127 or "command not found" in stderr_lower or "no such file" in stderr_lower:
        return "command_not_found"
    
    # Timeout/killed
    if exit_code in [-9, 137] or "killed" in stderr_lower or "terminated" in stderr_lower:
        return "timeout_or_killed"
    
    # File system errors
    if "no space left" in stderr_lower or "disk full" in stderr_lower:
        return "disk_full"
    
    # Network errors
    if "network" in stderr_lower or "connection" in stderr_lower or "unreachable" in stderr_lower:
        return "network_error"
    
    # Syntax errors
    if "syntax error" in stderr_lower or "invalid syntax" in stderr_lower:
        return "syntax_error"
    
    # Generic failures
    if exit_code > 0:
        return "command_failed"
    
    return "unknown_error"


def split_commands(command: str) -> List[str]:
    """Split a shell command into individual commands, respecting quotes"""
    commands = []
    current_command = ""
    in_single_quotes = False
    in_double_quotes = False
    i = 0
    while i < len(command):
        char = command[i]
        next_char = command[i + 1] if i + 1 < len(command) else ""
        if char == "\\" and i < len(command) - 1:
            current_command += char + command[i + 1]
            i += 2
            continue
        if char == "'" and not in_double_quotes:
            in_single_quotes = not in_single_quotes
        elif char == '"' and not in_single_quotes:
            in_double_quotes = not in_double_quotes
        if not in_single_quotes and not in_double_quotes:
            if (char == "&" and next_char == "&") or (char == "|" and next_char == "|"):
                commands.append(current_command.strip())
                current_command = ""
                i += 1
            elif char in [";", "&", "|"]:
                commands.append(current_command.strip())
                current_command = ""
            else:
                current_command += char
        else:
            current_command += char
        i += 1
    if current_command.strip():
        commands.append(current_command.strip())
    return [cmd for cmd in commands if cmd]


def get_command_root(command: str) -> Optional[str]:
    """Extract the root command from a shell command string"""
    trimmed_command = command.strip()
    if not trimmed_command:
        return None
    match = re.match(r'^"([^"]+)"|^\'([^\']+)\'|^(\S+)', trimmed_command)
    if match:
        command_root = match.group(1) or match.group(2) or match.group(3)
        if command_root:
            return command_root.split(os.sep)[-1]
    return None


# Pydantic models for tool parameters
class RunShellCommandParams(BaseModel):
    command: str = Field(..., description="Exact bash command to execute")
    directory: Optional[str] = Field(
        None, description="Directory to run the command in (relative to project root)"
    )
    description: Optional[str] = Field(
        None, description="Brief description of the command for the user"
    )
    # FIXED: Configurable timeout with sensible default
    timeout: int = Field(
        30, 
        description="Command timeout in seconds (default: 30, max: 300)",
        ge=1,
        le=300
    )
    # FIXED: Security enforcement options
    strict_whitelist: bool = Field(
        True, 
        description="Enforce strict command whitelist (recommended for security)"
    )
    allow_restricted: bool = Field(
        False,
        description="Allow restricted commands (requires explicit opt-in)"
    )

    @validator("command")
    def validate_command_security(cls, v, values):
        """Enhanced command validation with comprehensive security checks"""
        
        # Check for command injection patterns
        injection_threats = detect_command_injection(v)
        if injection_threats:
            threat_list = "; ".join(injection_threats)
            raise ValueError(
                f"Command injection threats detected: {threat_list}. "
                "Use argument lists instead of shell commands where possible."
            )
        
        # Validate against whitelist if strict mode is enabled
        strict_whitelist = values.get("strict_whitelist", True)
        allow_restricted = values.get("allow_restricted", False)
        
        if strict_whitelist:
            whitelist_error = validate_command_whitelist(v)
            if whitelist_error:
                if "restricted" in whitelist_error and allow_restricted:
                    # Log the restricted command usage
                    logger.warning(f"Restricted command allowed by explicit opt-in: {v}")
                else:
                    raise ValueError(
                        f"{whitelist_error}. "
                        f"Available commands: {', '.join(sorted(ALLOWED_COMMANDS))}. "
                        f"Use allow_restricted=True for restricted commands (with caution)."
                    )
        
        return v

    @validator("directory")
    def validate_directory(cls, v):
        """Enhanced directory validation with canonicalization"""
        if v is not None:
            if os.path.isabs(v):
                raise ValueError("Directory must be relative to project root")
            if ".." in v or v.startswith("/"):
                raise ValueError("Directory traversal not allowed")
            # Additional security: prevent hidden directory access without explicit permission
            if v.startswith(".") and v not in [".", ".git", ".github", ".vscode"]:
                raise ValueError(f"Access to hidden directory '{v}' not allowed")
        return v


class WebFetchParams(BaseModel):
    prompt: str = Field(
        ..., description="Comprehensive prompt with URLs and processing instructions"
    )
    summarize: bool = Field(
        False, description="Automatically summarize extracted content"
    )
    extract_main_content: bool = Field(
        True, description="Extract main content and remove boilerplate"
    )
    max_content_length: int = Field(5000, description="Maximum content length per URL")

    @validator("prompt")
    def validate_prompt_has_url(cls, v):
        url_pattern = r'https?://[^"\']+'
        if not re.search(url_pattern, v):
            raise ValueError(
                "Prompt must contain at least one URL starting with http:// or https://"
            )
        return v


class GoogleWebSearchParams(BaseModel):
    query: str = Field(
        ..., description="The search query to find information on the web"
    )
    max_results: int = Field(
        5, description="Maximum number of results to return (1-10)"
    )
    search_engine: str = Field(
        "duckduckgo", description="Search engine to use: 'duckduckgo' or 'google'"
    )
    safe_search: bool = Field(True, description="Enable safe search filtering")


# Tool implementations
class RunShellCommandTool(BaseTool):
    """
    Executes shell commands with enhanced security validation and error handling.
    
    Security features:
    - Strict command whitelist with forbidden/restricted command detection
    - Advanced command injection pattern detection
    - Path canonicalization and containment validation
    - Configurable timeouts and security enforcement
    
    Following Crisalida repo patterns:
    - Defensive validation with graceful degradation
    - Structured error metadata for intelligent agent responses
    - Comprehensive logging for security auditing
    """

    def _get_name(self) -> str:
        return "run_shell_command"

    def _get_description(self) -> str:
        return (
            "Executes bash commands with comprehensive security validation. "
            "Features strict command whitelisting, injection detection, path validation, "
            "and structured error reporting for intelligent agent responses."
        )

    def _get_pydantic_schema(self):
        return RunShellCommandParams

    def _get_category(self) -> str:
        return "shell"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = RunShellCommandParams(**kwargs)
        start_time = datetime.now()
        
        try:
            # FIXED: Enhanced directory validation with canonicalization
            work_dir = os.getcwd()  # Default to current working directory
            canonical_dir = None
            
            if params.directory:
                canonical_dir, dir_error = canonicalize_and_validate_directory(
                    params.directory, work_dir
                )
                if dir_error:
                    return ToolCallResult(
                        command=f"shell: {params.command}",
                        success=False,
                        output="",
                        error_message=dir_error,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        metadata={
                            "error_type": "directory_validation_failed",
                            "directory": params.directory,
                            "validation_error": dir_error,
                        }
                    )
                work_dir = canonical_dir

            # FIXED: Enhanced command execution with argument list when possible
            try:
                # For simple commands, try to use argument list instead of shell
                parsed_cmd = shlex.split(params.command)
                use_shell = False
                
                # If command contains shell operators, we must use shell
                shell_operators = ['|', '&&', '||', ';', '>', '<', '&']
                if any(op in params.command for op in shell_operators):
                    use_shell = True
                
                if use_shell:
                    # Use shell with explicit bash
                    process = subprocess.Popen(
                        ["bash", "-c", params.command],
                        cwd=work_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,  # Security: prevent signal propagation
                    )
                else:
                    # Use argument list (safer)
                    process = subprocess.Popen(
                        parsed_cmd,
                        cwd=work_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                
            except (ValueError, OSError) as e:
                return ToolCallResult(
                    command=f"shell: {params.command}",
                    success=False,
                    output="",
                    error_message=f"Failed to start command: {str(e)}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "error_type": "command_start_failed",
                        "command": params.command,
                        "start_error": str(e),
                    }
                )

            # FIXED: Execute with configurable timeout
            try:
                stdout, stderr = process.communicate(timeout=params.timeout)
                exit_code = process.returncode
                
            except subprocess.TimeoutExpired:
                # Cleanup on timeout
                process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=5)  # Grace period
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", "Process forcibly terminated"
                
                return ToolCallResult(
                    command=f"shell: {params.command}",
                    success=False,
                    output="",
                    error_message=f"Command timed out after {params.timeout} seconds",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "error_type": "timeout",
                        "command": params.command,
                        "timeout_seconds": params.timeout,
                        "directory": params.directory or "(root)",
                        "exit_code": process.returncode,
                    }
                )

            # FIXED: Enhanced error classification and structured metadata
            success = exit_code == 0
            error_type = classify_error_type(exit_code, stderr, params.command)
            
            tool_output = stdout.strip() if stdout.strip() else "(empty)"
            error_msg = None
            
            if not success:
                error_msg = stderr.strip() if stderr.strip() else f"Command failed with exit code {exit_code}"

            # FIXED: Comprehensive structured metadata for agent intelligence
            metadata = {
                "command": params.command,
                "directory": params.directory if params.directory else "(root)",
                "canonical_directory": canonical_dir,
                "working_directory": work_dir,
                "exit_code": exit_code,
                "error_type": error_type,
                "stderr": stderr.strip() if stderr.strip() else "(empty)",
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "execution_method": "shell" if use_shell else "argument_list",
                "timeout_used": params.timeout,
                "security_settings": {
                    "strict_whitelist": params.strict_whitelist,
                    "allow_restricted": params.allow_restricted,
                },
                # Additional context for agent decision making
                "suggested_actions": self._get_suggested_actions(error_type, stderr, params.command),
            }

            # Security audit logging
            logger.info(
                f"Shell command executed: {params.command} | "
                f"Exit: {exit_code} | Dir: {params.directory or '(root)'} | "
                f"Type: {error_type} | Time: {(datetime.now() - start_time).total_seconds():.2f}s"
            )

            return ToolCallResult(
                command=f"shell: {params.command}",
                success=success,
                output=tool_output,
                error_message=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata=metadata
            )

        except Exception as e:
            logger.exception(f"Unexpected error in RunShellCommandTool: {e}")
            return ToolCallResult(
                command=f"shell: {params.command}",
                success=False,
                output="",
                error_message=f"Unexpected execution error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "error_type": "unexpected_error",
                    "exception": str(e),
                    "command": params.command,
                }
            )

    def _get_suggested_actions(self, error_type: str, stderr: str, command: str) -> List[str]:
        """
        Generate intelligent suggestions based on error type for agent learning.
        Following Crisalida pattern: provide actionable intelligence for agent responses.
        """
        suggestions = []
        
        if error_type == "permission_denied":
            suggestions.extend([
                "Check file/directory permissions with 'ls -la'",
                "Ensure the user has appropriate access rights",
                "Consider if the operation requires elevated privileges"
            ])
        
        elif error_type == "command_not_found":
            cmd_root = get_command_root(command)
            if cmd_root:
                suggestions.extend([
                    f"Verify '{cmd_root}' is installed on the system",
                    f"Check if '{cmd_root}' is in the PATH",
                    f"Consider installing the package containing '{cmd_root}'"
                ])
        
        elif error_type == "timeout_or_killed":
            suggestions.extend([
                "Consider increasing timeout for long-running operations",
                "Break down complex operations into smaller steps",
                "Check if the process requires interactive input"
            ])
        
        elif error_type == "disk_full":
            suggestions.extend([
                "Check available disk space with 'df -h'",
                "Clean up temporary files or logs",
                "Consider moving operation to a different volume"
            ])
        
        elif error_type == "network_error":
            suggestions.extend([
                "Check network connectivity",
                "Verify URLs and hostnames are accessible",
                "Consider proxy or firewall restrictions"
            ])
        
        elif error_type == "syntax_error":
            suggestions.extend([
                "Review command syntax and arguments",
                "Check for proper quoting and escaping",
                "Verify command supports the provided options"
            ])
        
        # Add specific suggestions based on stderr content
        stderr_lower = stderr.lower()
        if "no such file" in stderr_lower:
            suggestions.append("Verify the file or directory path exists")
        if "invalid option" in stderr_lower:
            suggestions.append("Check command help/manual for valid options")
        if "connection refused" in stderr_lower:
            suggestions.append("Ensure the target service is running and accessible")
        
        return suggestions[:5]  # Limit to most relevant suggestions

    async def demo(self):
        """Demonstrates the shell and web tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.shell_and_web_demos import (
            demo_shell_web_tools,
        )

        return await demo_shell_web_tools()


# Web tools remain unchanged - they are not affected by the security vulnerabilities
class WebFetchTool(BaseTool):
    """Extrae y resume contenido de URLs"""

    def _get_name(self) -> str:
        return "web_fetch"

    def _get_description(self) -> str:
        return "Procesa contenido de hasta 20 URLs embebidas en el prompt. Extrae y resume contenido principal."

    def _get_pydantic_schema(self):
        return WebFetchParams

    def _get_category(self) -> str:
        return "web"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = WebFetchParams(**kwargs)
        start_time = datetime.now()
        try:
            url_pattern = r"https?://[^\"' ]+"
            urls = re.findall(url_pattern, params.prompt)
            if len(urls) > 20:
                return ToolCallResult(
                    command="web_fetch",
                    success=False,
                    output="",
                    error_message="Too many URLs (max 20 allowed)",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            fetched_content = []
            async with httpx.AsyncClient(timeout=10.0) as client:
                for i, url in enumerate(urls[:20], 1):
                    try:
                        logger.info(f"Fetching URL {i}/{len(urls)}: {url}")
                        response = await client.get(url, follow_redirects=True)
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "")
                        if "text/html" in content_type:
                            extracted_content = self._extract_html_content(
                                response.text,
                                params.extract_main_content,
                                params.max_content_length,
                            )
                            content = extracted_content["content"]
                            content_preview = extracted_content["preview"]
                        elif content_type.startswith("text/"):
                            content = response.text[: params.max_content_length]
                            content_preview = (
                                content[:200] + "..." if len(content) > 200 else content
                            )
                        elif content_type.startswith("application/json"):
                            try:
                                json_data = response.json()
                                content = json.dumps(json_data, indent=2)[
                                    : params.max_content_length
                                ]
                                content_preview = (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                )
                            except json.JSONDecodeError:
                                content = response.text[: params.max_content_length]
                                content_preview = (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                )
                        else:
                            content_preview = f"Binary content ({content_type}, {len(response.content)} bytes)"
                            content = content_preview
                        if params.summarize and len(content) > 500:
                            summary = self._summarize_content(content)
                            content = f"SUMMARY: {summary}\n\nFULL CONTENT:\n{content}"
                        fetched_content.append(
                            {
                                "url": url,
                                "status": response.status_code,
                                "content_type": content_type,
                                "content": content,
                                "preview": content_preview,
                                "length": len(content),
                                "summarized": params.summarize,
                            }
                        )
                    except httpx.HTTPError as e:
                        fetched_content.append(
                            {
                                "url": url,
                                "status": (
                                    getattr(e.response, "status_code", 0)
                                    if hasattr(e, "response")
                                    else 0
                                ),
                                "content_type": "error",
                                "content": f"Error fetching URL: {str(e)}",
                                "preview": f"Error: {str(e)}",
                            }
                        )
                    except Exception as e:
                        fetched_content.append(
                            {
                                "url": url,
                                "status": 0,
                                "content_type": "error",
                                "content": f"Unexpected error: {str(e)}",
                                "preview": f"Unexpected error: {str(e)}",
                            }
                        )
            if not fetched_content:
                return ToolCallResult(
                    command="web_fetch",
                    success=False,
                    output="",
                    error_message="No URLs could be processed",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            output_parts = [f"Fetched content from {len(fetched_content)} URL(s):\n"]
            successful_fetches = 0
            for item in fetched_content:
                if item["status"] == 200:
                    successful_fetches += 1
                    output_parts.append(f"✅ {item['url']} ({item['content_type']})")
                    output_parts.append(f"   Content preview: {item['preview']}")
                else:
                    output_parts.append(f"❌ {item['url']} (Status: {item['status']})")
                    output_parts.append(f"   Error: {item['content']}")
                output_parts.append("")
            success = successful_fetches > 0
            final_output = "\n".join(output_parts)
            return ToolCallResult(
                command="web_fetch",
                success=success,
                output=final_output,
                error_message=(
                    None if success else "No URLs could be fetched successfully"
                ),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return ToolCallResult(
                command="web_fetch",
                success=False,
                output="",
                error_message=f"Web fetch failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the shell and web tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.shell_and_web_demos import (
            demo_shell_web_tools,
        )

        return await demo_shell_web_tools()

    def _extract_html_content(
        self, html: str, extract_main: bool, max_length: int
    ) -> Dict[str, str]:
        """Extract main content from HTML using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside", "form"]
            ):
                element.decompose()
            if extract_main:
                # Try to find main content areas
                main_content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find(
                        "div", {"class": re.compile(r"content|main|article", re.I)}
                    )
                    or soup.find(
                        "div", {"id": re.compile(r"content|main|article", re.I)}
                    )
                    or soup.body
                    or soup
                )
            else:
                main_content = soup

            if main_content is not None:
                # Extract text content
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = ""  # Fallback if main_content is None
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "..."
            # Create preview
            preview = text[:200] + "..." if len(text) > 200 else text
            return {"content": text, "preview": preview}
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            # Fallback to raw text
            text = html[:max_length]
            preview = text[:200] + "..." if len(text) > 200 else text
            return {"content": text, "preview": preview}

    def _summarize_content(self, content: str) -> str:
        """Create a simple extractive summary of the content."""
        # Simple extractive summarization
        sentences = re.split(r"[.!?]+", content)
        # Filter sentences by length and content quality
        good_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (
                10 <= len(sentence) <= 200
                and not sentence.lower().startswith(
                    ("click", "subscribe", "follow", "share")
                )
                and sentence.count(" ") >= 3
            ):  # At least 4 words
                good_sentences.append(sentence)
        # Take first few sentences up to ~300 chars
        summary_parts = []
        total_length = 0
        for sentence in good_sentences[:5]:  # Max 5 sentences
            if total_length + len(sentence) > 300:
                break
            summary_parts.append(sentence)
            total_length += len(sentence) + 2  # +2 for ". "
        return (
            ". ".join(summary_parts) + "."
            if summary_parts
            else "Content summary not available."
        )


class GoogleWebSearchTool(BaseTool):
    """Realiza búsquedas web estructuradas (DuckDuckGo)"""

    def _get_name(self) -> str:
        return "google_web_search"

    def _get_description(self) -> str:
        return "Realiza búsqueda web usando DuckDuckGo y retorna resultados estructurados (título, URL, snippet)."

    def _get_pydantic_schema(self):
        return GoogleWebSearchParams

    def _get_category(self) -> str:
        return "web"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = GoogleWebSearchParams(**kwargs)
        start_time = datetime.now()
        try:
            if params.search_engine == "duckduckgo":
                results = await self._search_duckduckgo(params)
            else:
                return ToolCallResult(
                    command=f"web_search({params.search_engine})",
                    success=False,
                    output="",
                    error_message=f"Search engine '{params.search_engine}' not supported. Use 'duckduckgo'.",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),
                )
            if not results:
                return ToolCallResult(
                    command=f"web_search: {params.query}",
                    success=True,
                    output=f"No results found for query: '{params.query}'",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,
                )
            # Format results
            output_parts = [f"Search results for: '{params.query}'\n"]
            for i, result in enumerate(results[: params.max_results], 1):
                output_parts.append(f"{i}. {result['title']}")
                output_parts.append(f"   URL: {result['url']}")
                output_parts.append(f"   Snippet: {result['snippet']}")
                output_parts.append("")  # Empty line for readability
            # Create structured output
            search_output = {
                "query": params.query,
                "search_engine": params.search_engine,
                "total_results": len(results),
                "results": results[: params.max_results],
            }
            return ToolCallResult(
                command=f"web_search: {params.query}",
                success=True,
                output="\n".join(output_parts),
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata=search_output,
                error_message=None,
            )
        except Exception as e:
            return ToolCallResult(
                command="web_search",
                success=False,
                output="",
                error_message=f"Search failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the shell and web tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.shell_and_web_demos import (
            demo_shell_web_tools,
        )

        return await demo_shell_web_tools()

    async def _search_duckduckgo(self, params: GoogleWebSearchParams) -> List[Dict[str, str]]:
        """Search using DuckDuckGo instant answers and HTML search"""
        results = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First try instant answers API
            try:
                instant_url = "https://api.duckduckgo.com/"
                instant_params = {
                    "q": params.query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
                response = await client.get(instant_url, params=instant_params)
                response.raise_for_status()
                data = response.json()
                # Extract instant answer if available
                if data.get("Abstract"):
                    results.append(
                        {
                            "title": data.get("Heading", "Instant Answer"),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("Abstract", ""),
                        }
                    )
                # Add related topics
                for topic in data.get("RelatedTopics", [])[:2]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(
                            {
                                "title": topic.get("Text", "").split(" - ")[0],
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                            }
                        )
            except Exception as e:
                logger.warning(f"DuckDuckGo instant answers failed: {e}")
            # If we don't have enough results, try HTML search
            if len(results) < params.max_results:
                try:
                    search_url = "https://html.duckduckgo.com/html/"
                    search_params = {
                        "q": params.query,
                        "b": "",  # No ads
                        "kl": "us-en",  # Language
                        "df": "",  # Date filter
                    }
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = await client.get(
                        search_url, params=search_params, headers=headers
                    )
                    response.raise_for_status()
                    # Parse HTML results
                    soup = BeautifulSoup(response.text, "html.parser")
                    search_results = soup.find_all("div", class_="web-result")
                    for result_div in search_results:
                        if isinstance(result_div, Tag):
                            try:
                                title_elem = result_div.find("a", class_="result__a")
                                snippet_elem = result_div.find(
                                    "a", class_="result__snippet"
                                )
                                if title_elem:
                                    title_elem = cast(Tag, title_elem)
                                    title = title_elem.get_text(strip=True)
                                    url = title_elem.get("href", "")
                                    if isinstance(url, str):
                                        snippet = (
                                            snippet_elem.get_text(strip=True)
                                            if snippet_elem
                                            else ""
                                        )
                                        # Clean up URL (DuckDuckGo sometimes wraps URLs)
                                        if url.startswith("/l/?uddg="):
                                            url = urllib.parse.unquote(
                                                url.split("uddg=")[1].split("&")[0]
                                            )
                                        results.append(
                                            {
                                                "title": title,
                                                "url": url,
                                                "snippet": snippet,
                                            }
                                        )
                                        if len(results) >= params.max_results:
                                            break
                            except Exception as e:
                                logger.warning(f"Error parsing search result: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"DuckDuckGo HTML search failed: {e}")
        return results


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.shell_and_web_demos import (
        demo_shell_web_tools,
    )

    asyncio.run(demo_shell_web_tools())
