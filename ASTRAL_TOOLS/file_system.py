#!/usr/bin/env python3
"""
File System Tools for Prometheus Agent
Python implementation of the TypeScript file system tools from Gemini CLI.

Core features:
- Robust file reading (text, binary, images, PDFs, SVGs)
- Safe writing with auto-backup
- Directory and glob listing with ignore/exclude support
- Regex-based content search
- Advanced replacement (string, regex, line-based)
- Multi-file reading with advanced filters
- Demo/test suite for all tools

Optimized for reliability, extensibility, and integration with the Crisalida ecosystem.
"""

import base64
import fnmatch
import glob as glob_module
import logging
import mimetypes
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES_TEXT_FILE = 2000
MAX_LINE_LENGTH_TEXT_FILE = 2000
DEFAULT_ENCODING = "utf-8"
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB limit


def get_specific_mime_type(file_path: str) -> str | None:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


async def is_binary_file(file_path: str) -> bool:
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        file_size = os.path.getsize(file_path)
        buffer_size = min(4096, file_size)
        with open(file_path, "rb") as f:
            buffer = f.read(buffer_size)
        if not buffer:
            return False
        if b"\x00" in buffer:
            return True
        non_printable_count = 0
        for byte in buffer:
            if byte < 9 or (byte > 13 and byte < 32):
                non_printable_count += 1
        return non_printable_count / len(buffer) > 0.3
    except Exception as e:
        logger.warning(f"Failed to check if file is binary: {file_path} - {e}")
        return False


async def detect_file_type(file_path: str) -> str:
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".ts"):
        return "text"
    if file_path_lower.endswith(".svg"):
        return "svg"
    mime_type = get_specific_mime_type(file_path)
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type == "application/pdf":
            return "pdf"
    binary_extensions = {
        ".zip",
        ".tar",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".class",
        ".jar",
        ".war",
        ".7z",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".bin",
        ".dat",
        ".obj",
        ".o",
        ".a",
        ".lib",
        ".wasm",
        ".pyc",
        ".pyo",
    }
    file_ext = Path(file_path).suffix.lower()
    if file_ext in binary_extensions:
        return "binary"
    if await is_binary_file(file_path):
        return "binary"
    return "text"


class ProcessedFileReadResult:
    def __init__(
        self,
        llm_content: str | dict[str, Any],
        return_display: str,
        error: str | None = None,
        is_truncated: bool | None = None,
        original_line_count: int | None = None,
        lines_shown: tuple[int, int] | None = None,
    ):
        self.llm_content = llm_content
        self.return_display = return_display
        self.error = error
        self.is_truncated = is_truncated
        self.original_line_count = original_line_count
        self.lines_shown = lines_shown


async def process_single_file_content(
    file_path: str,
    root_directory: str,
    offset: int | None = None,
    limit: int | None = None,
) -> ProcessedFileReadResult:
    try:
        if not os.path.exists(file_path):
            return ProcessedFileReadResult(
                llm_content="",
                return_display="File not found.",
                error=f"File not found: {file_path}",
            )
        if os.path.isdir(file_path):
            return ProcessedFileReadResult(
                llm_content="",
                return_display="Path is a directory.",
                error=f"Path is a directory, not a file: {file_path}",
            )
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            return ProcessedFileReadResult(
                llm_content="",
                return_display="File too large.",
                error=f"File size exceeds 20MB limit: {file_path} ({file_size / (1024 * 1024):.2f}MB)",
            )
        file_type = await detect_file_type(file_path)
        relative_path = os.path.relpath(file_path, root_directory).replace("\\", "/")
        if file_type == "binary":
            return ProcessedFileReadResult(
                llm_content=f"Cannot display content of binary file: {relative_path}",
                return_display=f"Skipped binary file: {relative_path}",
            )
        elif file_type == "svg":
            svg_max_size_bytes = 1 * 1024 * 1024
            if file_size > svg_max_size_bytes:
                return ProcessedFileReadResult(
                    llm_content=f"Cannot display content of SVG file larger than 1MB: {relative_path}",
                    return_display=f"Skipped large SVG file (>1MB): {relative_path}",
                )
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ProcessedFileReadResult(
                llm_content=content, return_display=f"Read SVG as text: {relative_path}"
            )
        elif file_type == "text":
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = content.split("\n")
            original_line_count = len(lines)
            start_line = offset or 0
            effective_limit = (
                limit if limit is not None else DEFAULT_MAX_LINES_TEXT_FILE
            )
            end_line = min(start_line + effective_limit, original_line_count)
            actual_start_line = min(start_line, original_line_count)
            selected_lines = lines[actual_start_line:end_line]
            lines_were_truncated_in_length = False
            formatted_lines = []
            for line in selected_lines:
                if len(line) > MAX_LINE_LENGTH_TEXT_FILE:
                    lines_were_truncated_in_length = True
                    formatted_lines.append(
                        line[:MAX_LINE_LENGTH_TEXT_FILE] + "... [truncated]"
                    )
                else:
                    formatted_lines.append(line)
            content_range_truncated = end_line < original_line_count
            is_truncated = content_range_truncated or lines_were_truncated_in_length
            llm_text_content = ""
            if content_range_truncated:
                llm_text_content += f"[File content truncated: showing lines {actual_start_line + 1}-{end_line} of {original_line_count} total lines. Use offset/limit parameters to view more.]\n"
            elif lines_were_truncated_in_length:
                llm_text_content += f"[File content partially truncated: some lines exceeded maximum length of {MAX_LINE_LENGTH_TEXT_FILE} characters.]\n"
            llm_text_content += "\n".join(formatted_lines)
            return ProcessedFileReadResult(
                llm_content=llm_text_content,
                return_display="(truncated)" if is_truncated else "",
                is_truncated=is_truncated,
                original_line_count=original_line_count,
                lines_shown=(actual_start_line + 1, end_line),
            )
        elif file_type in ["image", "pdf", "audio", "video"]:
            with open(file_path, "rb") as f:
                content_buffer = f.read()
            base64_data = base64.b64encode(content_buffer).decode("utf-8")
            mime_type = get_specific_mime_type(file_path) or "application/octet-stream"
            return ProcessedFileReadResult(
                llm_content={
                    "inlineData": {"data": base64_data, "mimeType": mime_type}
                },
                return_display=f"Read {file_type} file: {relative_path}",
            )
        else:
            return ProcessedFileReadResult(
                llm_content=f"Unhandled file type: {file_type}",
                return_display=f"Skipped unhandled file type: {relative_path}",
                error=f"Unhandled file type for {file_path}",
            )
    except Exception as e:
        error_message = str(e)
        display_path = os.path.relpath(file_path, root_directory).replace("\\", "/")
        return ProcessedFileReadResult(
            llm_content=f"Error reading file {display_path}: {error_message}",
            return_display=f"Error reading file {display_path}: {error_message}",
            error=f"Error reading file {file_path}: {error_message}",
        )


# Pydantic models for tool parameters
class ReadFileParams(BaseModel):
    absolute_path: str = Field(..., description="The absolute path to the file to read")
    offset: int | None = Field(
        None, description="0-based line number to start reading from (for text files)"
    )
    limit: int | None = Field(
        None, description="Maximum number of lines to read (for text files)"
    )

    @field_validator("absolute_path")
    def validate_absolute_path(cls, v):
        if not os.path.isabs(v):
            raise ValueError("Path must be absolute")
        return v


class WriteFileParams(BaseModel):
    file_path: str = Field(..., description="The absolute path to the file to write to")
    content: str = Field(..., description="The content to write to the file")
    append: bool = Field(
        False,
        description="If true, content will be appended to the file instead of overwriting it.",
    )

    @field_validator("file_path")
    def validate_file_path(cls, v):
        if not os.path.isabs(v):
            raise ValueError("Path must be absolute")
        return v


class ListDirectoryParams(BaseModel):
    path: str = Field(..., description="The absolute path to the directory to list")
    ignore: list[str] | None = Field(
        None, description="List of glob patterns to ignore"
    )

    @field_validator("path")
    def validate_path(cls, v):
        if not os.path.isabs(v):
            raise ValueError("Path must be absolute")
        return v


class GlobParams(BaseModel):
    pattern: str = Field(..., description="The glob pattern to match against")
    path: str | None = Field(None, description="The absolute path to search within")
    case_sensitive: bool | None = Field(
        False, description="Whether the search should be case-sensitive"
    )

    @field_validator("path")
    def validate_path(cls, v):
        if v is not None and not os.path.isabs(v):
            raise ValueError("Path must be absolute or None")
        return v


class SearchFileContentParams(BaseModel):
    pattern: str = Field(
        ..., description="The regular expression pattern to search for"
    )
    include: str | None = Field(
        None, description="A glob pattern to filter which files are searched"
    )
    path: str | None = Field(
        None,
        description="Optional: The absolute path to the directory to search within. If omitted, searches the current working directory.",
    )


class ReplaceParams(BaseModel):
    file_path: str = Field(..., description="The absolute path to the file to modify")

    # String-based replacement fields
    old_string: str | None = Field(
        None,
        description="The exact literal text to replace (for string-based replacement)",
    )
    new_string: str | None = Field(
        None,
        description="The exact literal text to replace old_string with (for string-based replacement)\nIf is_regex is True, this can contain backreferences like \1, \2.",
    )
    expected_replacements: int | None = Field(
        1, description="Number of replacements expected (for string-based replacement)"
    )
    is_regex: bool = Field(
        False,
        description="Whether old_string should be treated as a regular expression.",
    )

    # Line-based replacement fields
    start_line: int | None = Field(
        None,
        description="The 1-based starting line number for line-based replacement (inclusive)",
    )
    end_line: int | None = Field(
        None,
        description="The 1-based ending line number for line-based replacement (inclusive)",
    )
    new_content: str | None = Field(
        None, description="The new content to insert for line-based replacement"
    )
    expected_old_content: str | None = Field(
        None,
        description="Optional: The expected content of the lines to be replaced (for line-based replacement validation)",
    )
    dry_run: bool = Field(
        False,
        description="If True, perform validation and show what would be changed without actually modifying the file",
    )

    @field_validator("file_path")
    def validate_file_path(cls, v):
        if not os.path.isabs(v):
            raise ValueError("Path must be absolute")
        return v

    @field_validator("start_line")
    def validate_start_line(cls, v, values):
        if v is not None and v < 1:
            raise ValueError("start_line must be 1 or greater")
        if (
            v is not None
            and values.get("end_line") is not None
            and v > values["end_line"]
        ):
            raise ValueError("start_line cannot be greater than end_line")
        return v

    @field_validator("end_line")
    def validate_end_line(cls, v, values):
        if v is not None and v < 1:
            raise ValueError("end_line must be 1 or greater")
        if (
            v is not None
            and values.get("start_line") is not None
            and v < values["start_line"]
        ):
            raise ValueError("end_line cannot be less than start_line")
        return v

    @field_validator("new_content")
    def validate_new_content(cls, v, values):
        if v is not None and values.get("start_line") is None:
            raise ValueError("new_content requires start_line to be specified")
        return v

    @model_validator(
        mode="before"
    )  # Usar pre=True para validar antes de la asignación de campos
    def validate_replacement_params(cls, values):
        is_string_based = (
            values.get("old_string") is not None or values.get("new_string") is not None
        )
        is_line_based = (
            values.get("start_line") is not None
            or values.get("end_line") is not None
            or values.get("new_content") is not None
        )

        if is_string_based and is_line_based:
            raise ValueError(
                "Cannot mix string-based and line-based replacement parameters. Provide either old_string/new_string OR start_line/end_line/new_content."
            )
        if not is_string_based and not is_line_based:
            raise ValueError(
                "Either string-based (old_string/new_string) or line-based (start_line/end_line/new_content) parameters must be provided."
            )

        if is_string_based:
            if values.get("old_string") is None or values.get("new_string") is None:
                raise ValueError(
                    "For string-based replacement, both old_string and new_string must be provided."
                )
            # If is_regex is True, old_string must be a valid regex
            if values.get("is_regex") and values.get("old_string"):
                try:
                    re.compile(values["old_string"])
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex pattern for old_string: {e}"
                    ) from e

        if is_line_based:
            if values.get("start_line") is None or values.get("new_content") is None:
                raise ValueError(
                    "For line-based replacement, start_line and new_content must be provided."
                )

        return values


class ReadManyFilesParams(BaseModel):
    paths: list[str] = Field(
        ..., description="Array of glob patterns or paths relative to target directory"
    )
    recursive: bool | None = Field(True, description="Whether to search recursively")
    use_default_excludes: bool | None = Field(
        True, description="Whether to apply default exclusion patterns"
    )
    exclude: list[str] | None = Field(
        default_factory=lambda: [],
        description="Glob patterns to exclude",
    )
    include: list[str] | None = Field(
        default_factory=lambda: [],
        description="Additional glob patterns to include",
    )
    # Advanced filtering options
    max_file_size: int | None = Field(
        None, description="Maximum file size in bytes (default: no limit)"
    )
    min_file_size: int | None = Field(
        None, description="Minimum file size in bytes (default: 0)"
    )
    modified_after: str | None = Field(
        None,
        description="Only include files modified after this date (ISO format: YYYY-MM-DD)",
    )
    modified_before: str | None = Field(
        None,
        description="Only include files modified before this date (ISO format: YYYY-MM-DD)",
    )
    content_regex: str | None = Field(
        None, description="Only include files whose content matches this regex pattern"
    )
    file_extension: list[str] | None = Field(
        None,
        description="Only include files with these extensions (e.g., ['.py', '.js'])",
    )


# Tool implementations
class ReadFileTool(BaseTool):
    """Read and return the content of a specified file"""

    def _get_name(self) -> str:
        return "read_file"

    def _get_description(self) -> str:
        return "Reads and returns the content of a specified file from the local filesystem. Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, it can read specific line ranges."

    def _get_pydantic_schema(self):
        return ReadFileParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ReadFileParams(**kwargs)
        start_time = datetime.now()
        try:
            # Use current working directory as root for relative path display
            root_directory = os.getcwd()
            result = await process_single_file_content(
                params.absolute_path, root_directory, params.offset, params.limit
            )
            if result.error:
                return ToolCallResult(
                    command=f"read_file {params.absolute_path}",
                    success=False,
                    output="",
                    error_message=result.error,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            return ToolCallResult(
                command=f"read_file {params.absolute_path}",
                success=True,
                output=(
                    result.llm_content
                    if isinstance(result.llm_content, str)
                    else result.return_display
                ),
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"read_file {params.absolute_path}",  # Added command
                success=False,
                output="",
                error_message=f"Failed to read file: {str(e)}",
                execution_time=0.0,
            )

    async def demo(self):
        """Demonstrates the ReadFileTool's functionality by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_read_file_tool import (
            demonstrate_read_file_tool,
        )

        return await demonstrate_read_file_tool()


class WriteFileTool(BaseTool):
    """Write content to a specified file"""

    def _get_name(self) -> str:
        return "write_file"

    def _get_description(self) -> str:
        return "Writes content to a specified file in the local filesystem."

    def _get_pydantic_schema(self):
        return WriteFileParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = WriteFileParams(**kwargs)
        start_time = datetime.now()
        try:
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(params.file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            # Determine file open mode
            mode = "a" if params.append else "w"
            # Write content to file
            with open(params.file_path, mode, encoding="utf-8") as f:
                f.write(params.content)
            relative_path = os.path.relpath(params.file_path, os.getcwd())
            return ToolCallResult(
                command=f"write_file {params.file_path}",
                output=f"Successfully wrote {len(params.content)} characters to {relative_path}",
                success=True,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"write_file {params.file_path}",
                success=False,
                output="",
                error_message=f"Failed to write file: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the file system tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_file_system_tools import (
            demo_file_system_tools,
        )

        return await demo_file_system_tools()


class ListDirectoryTool(BaseTool):
    """List the contents of a directory"""

    def _get_name(self) -> str:
        return "list_directory"

    def _get_description(self) -> str:
        return "Lists the names of files and subdirectories directly within a specified directory path. Can optionally ignore entries matching provided glob patterns."

    def _get_pydantic_schema(self):
        return ListDirectoryParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ListDirectoryParams(**kwargs)
        start_time = datetime.now()
        try:
            if not os.path.exists(params.path):
                return ToolCallResult(
                    command=f"list_directory {params.path}",
                    success=False,
                    output="",
                    error_message=f"Directory not found: {params.path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            if not os.path.isdir(params.path):
                return ToolCallResult(
                    command=f"list_directory {params.path}",
                    success=False,
                    output="",
                    error_message=f"Path is not a directory: {params.path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            # Get directory contents
            entries = []
            try:
                for entry in os.listdir(params.path):
                    entry_path = os.path.join(params.path, entry)
                    # Skip entries matching ignore patterns
                    if params.ignore:
                        should_ignore = False
                        for pattern in params.ignore:
                            if fnmatch.fnmatch(entry, pattern):
                                should_ignore = True
                                break
                        if should_ignore:
                            continue
                    # Determine entry type
                    if os.path.isdir(entry_path):
                        entries.append(f"{entry}/")
                    else:
                        entries.append(entry)
                entries.sort()  # Sort alphabetically
                output = f"Directory listing for {os.path.relpath(params.path, os.getcwd())}:\n"
                if entries:
                    for entry in entries:
                        output += f"  {entry}\n"
                else:
                    output += "  (empty directory)\n"
                return ToolCallResult(
                    command=f"list_directory {params.path}",
                    success=True,
                    output=output.strip(),
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,
                )
            except PermissionError:
                return ToolCallResult(
                    command=f"list_directory {params.path}",
                    success=False,
                    output="",
                    error_message=f"Permission denied accessing directory: {params.path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        except Exception as e:
            return ToolCallResult(
                command=f"list_directory {params.path}",
                success=False,
                output="",
                error_message=f"Failed to list directory: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the ListDirectoryTool's functionality by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_list_directory_tool import (
            demonstrate_list_directory_tool,
        )

        return await demonstrate_list_directory_tool()


class GlobTool(BaseTool):
    """Find files matching glob patterns"""

    def _get_name(self) -> str:
        return "glob"

    def _get_description(self) -> str:
        return "Efficiently finds files matching specific glob patterns (e.g., `src/**/*.ts`, `**/*.md`), returning absolute paths sorted by modification time (newest first). Ideal for quickly locating files based on their name or path structure, especially in large codebases."

    def _get_pydantic_schema(self):
        return GlobParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = GlobParams(**kwargs)
        start_time = datetime.now()
        try:
            search_path = params.path or os.getcwd()
            if not os.path.exists(search_path):
                return ToolCallResult(
                    command=f"glob {params.pattern}",
                    success=False,
                    output="",
                    error_message=f"Search path not found: {search_path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            # Change to search directory for glob operation
            original_cwd = os.getcwd()
            os.chdir(search_path)
            try:
                # Use glob to find matching files
                matches = glob_module.glob(params.pattern, recursive=True)
                # Convert to absolute paths
                absolute_matches = [os.path.abspath(match) for match in matches]
                # Filter out directories (only return files)
                file_matches = [
                    path for path in absolute_matches if os.path.isfile(path)
                ]
                # Sort by modification time (newest first)
                file_matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                if file_matches:
                    output = f"Found {len(file_matches)} files matching pattern '{params.pattern}':\n"
                    for file_path in file_matches:
                        relative_path = os.path.relpath(file_path, original_cwd)
                        output += f"  {relative_path}\n"
                    return ToolCallResult(
                        command=f"glob {params.pattern}",
                        success=True,
                        output=output.strip(),
                        error_message=None,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
                else:
                    output = f"No files found matching pattern '{params.pattern}'"
                    return ToolCallResult(
                        command=f"glob {params.pattern}",
                        success=True,
                        output=output.strip(),
                        error_message=None,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            return ToolCallResult(
                command=f"glob {params.pattern}",
                success=False,
                output="",
                error_message=f"Glob search failed: {str(e)}",
                execution_time=0.0,
            )

    async def demo(self):
        """Demonstrates the GlobTool's functionality by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_glob_tool import (
            demonstrate_glob_tool,
        )

        return await demonstrate_glob_tool()


class SearchFileContentTool(BaseTool):
    """Search for regex patterns within file contents"""

    def _get_name(self) -> str:
        return "search_file_content"

    def _get_description(self) -> str:
        return "Searches for a regular expression pattern within the content of files in a specified directory (or current working directory). Can filter files by a glob pattern. Returns the lines containing matches, along with their file paths and line numbers."

    def _get_pydantic_schema(self):
        return SearchFileContentParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = SearchFileContentParams(**kwargs)
        start_time = datetime.now()
        try:
            search_path = params.path or os.getcwd()
            if not os.path.exists(search_path):
                return ToolCallResult(
                    command=f"search_file_content {params.pattern}",
                    success=False,
                    output="",
                    error_message=f"Search path not found: {search_path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            # Compile regex pattern
            try:
                regex = re.compile(params.pattern)
            except re.error as e:
                return ToolCallResult(
                    command=f"search_file_content {params.pattern}",
                    success=False,
                    output="",
                    error_message=f"Invalid regex pattern: {str(e)}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            matches = []
            files_searched = 0
            # Get files to search
            if params.include:
                # Use glob pattern to filter files
                original_cwd = os.getcwd()
                os.chdir(search_path)
                try:
                    file_paths = glob_module.glob(params.include, recursive=True)
                    file_paths = [
                        os.path.join(search_path, path) for path in file_paths
                    ]
                finally:
                    os.chdir(original_cwd)
            else:
                # Search all files recursively
                file_paths = []
                for root, _dirs, files in os.walk(search_path):
                    for file in files:
                        file_paths.append(os.path.join(root, file))
            logger.debug(f"SearchFileContentTool: Files to search: {file_paths}")
            # Search through files
            for file_path in file_paths:
                if not os.path.isfile(file_path):
                    continue
                try:
                    # Skip binary files
                    is_bin = await is_binary_file(file_path)
                    logger.debug(
                        f"SearchFileContentTool: {file_path} is binary: {is_bin}"
                    )
                    if is_bin:
                        continue
                    files_searched += 1
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                relative_path = os.path.relpath(file_path, os.getcwd())
                                matches.append(
                                    {
                                        "file": relative_path,
                                        "line": line_num,
                                        "content": line.strip(),
                                    }
                                )
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue
            # Format output
            if matches:
                output = f"Found {len(matches)} matches in {files_searched} files:\n\n"
                for match in matches:
                    output += f"{match['file']}:{match['line']} - {match['content']}\n"
                return ToolCallResult(
                    command=f"search_file_content {params.pattern}",
                    success=True,
                    output=output.strip(),
                    error_message=None,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            else:
                output = f"No matches found for pattern '{params.pattern}' in {files_searched} files"
                return ToolCallResult(
                    command=f"search_file_content {params.pattern}",
                    success=True,
                    output=output.strip(),
                    error_message=None,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        except Exception as e:
            return ToolCallResult(
                command=f"search_file_content {params.pattern}",
                success=False,
                output="",
                error_message=f"Content search failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the file system tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_file_system_tools import (
            demo_file_system_tools,
        )

        return await demo_file_system_tools()


class ReplaceTool(BaseTool):
    """Replace text within a file"""

    def _get_name(self) -> str:
        return "replace"

    def _get_description(self) -> str:
        return "Replaces text within a file. By default, replaces a single occurrence, but can replace multiple occurrences when expected_replacements is specified. This tool requires providing significant context around the change to ensure precise targeting."

    def _get_pydantic_schema(self):
        return ReplaceParams

    def _get_category(self) -> str:
        return "file_system"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ReplaceParams(**kwargs)
        start_time = datetime.now()
        try:
            if not os.path.exists(params.file_path):
                return ToolCallResult(
                    command=f"replace {params.file_path}",
                    success=False,
                    output="",
                    error_message=f"File not found: {params.file_path}",
                    execution_time=0.0,
                )
            if os.path.isdir(params.file_path):
                return ToolCallResult(
                    command=f"replace {params.file_path}",
                    success=False,
                    output="",
                    error_message=f"Path is a directory, not a file: {params.file_path}",
                    execution_time=0.0,
                )

            # Read file content
            with open(params.file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
            original_content = "".join(original_lines)  # For string-based replacement

            new_content_to_write = ""
            replacements_made = 0

            if params.start_line is not None and params.new_content is not None:
                # Line-based replacement
                start_idx = params.start_line - 1  # Convert to 0-based index
                end_idx = (
                    params.end_line or params.start_line
                ) - 1  # Convert to 0-based index

                if start_idx < 0 or start_idx >= len(original_lines):
                    return ToolCallResult(
                        command=f"replace {params.file_path}",
                        success=False,
                        output="",
                        error_message=f"start_line {params.start_line} is out of bounds (file has {len(original_lines)} lines).",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
                if end_idx >= len(original_lines):
                    return ToolCallResult(
                        command=f"replace {params.file_path}",
                        success=False,
                        output="",
                        error_message=f"end_line {params.end_line} is out of bounds (file has {len(original_lines)} lines).",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )

                lines_to_replace = original_lines[start_idx : end_idx + 1]
                old_content_block = "".join(lines_to_replace)

                if (
                    params.expected_old_content is not None
                    and old_content_block != params.expected_old_content
                ):
                    return ToolCallResult(
                        command=f"replace {params.file_path}",
                        success=False,
                        output="",
                        error_message=f"Expected old content for lines {params.start_line}-{params.end_line} does not match actual content. Expected: '{params.expected_old_content}', Actual: '{old_content_block}'",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )

                # Construct new content
                new_lines_list = (
                    original_lines[:start_idx]
                    + [line + "\n" for line in params.new_content.splitlines()]
                    + original_lines[end_idx + 1 :]
                )
                new_content_to_write = "".join(new_lines_list)
                replacements_made = 1  # A single block replacement

            elif params.old_string is not None and params.new_string is not None:
                # String-based replacement
                if params.is_regex:
                    try:
                        # Use re.findall to count occurrences for regex
                        occurrences = len(
                            re.findall(params.old_string, original_content)
                        )
                        if occurrences == 0:
                            return ToolCallResult(
                                command=f"replace {params.file_path}",
                                success=False,
                                output="",
                                error_message=f"Regex pattern '{params.old_string}' not found in file.",
                                execution_time=(
                                    datetime.now() - start_time
                                ).total_seconds(),
                            )
                        new_content_to_write = re.sub(
                            params.old_string, params.new_string, original_content
                        )
                    except re.error as e:
                        return ToolCallResult(
                            command=f"replace {params.file_path}",
                            success=False,
                            output="",
                            error_message=f"Invalid regex pattern: {e}",
                            execution_time=(
                                datetime.now() - start_time
                            ).total_seconds(),
                        )
                else:
                    occurrences = original_content.count(params.old_string)
                    if occurrences == 0:
                        return ToolCallResult(
                            command=f"replace {params.file_path}",
                            success=False,
                            output="",
                            error_message=f"String not found in file: '{params.old_string}'. Please ensure the 'old_string' exactly matches the content in the file, including all whitespace and indentation. Consider setting 'is_regex=True' if you intend to use a regular expression.",
                            execution_time=(
                                datetime.now() - start_time
                            ).total_seconds(),
                        )
                    new_content_to_write = original_content.replace(
                        params.old_string, params.new_string
                    )

                # Check expected replacements
                expected = params.expected_replacements or 1
                if occurrences != expected:
                    return ToolCallResult(
                        command=f"replace {params.file_path}",
                        success=False,
                        output="",
                        error_message=f"Expected {expected} occurrences but found {occurrences} of: '{params.old_string}'.",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
                replacements_made = occurrences
            else:
                return ToolCallResult(
                    command=f"replace {params.file_path}",
                    success=False,
                    output="",
                    error_message="Invalid parameters for replacement. Provide either old_string/new_string or start_line/end_line/new_content.",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

            # Handle dry run mode
            if params.dry_run:
                relative_path = os.path.relpath(params.file_path, os.getcwd())
                dry_run_info = {
                    "file": relative_path,
                    "replacements_that_would_be_made": replacements_made,
                    "current_size": len(original_content),
                    "new_size": len(new_content_to_write),
                    "size_change": len(new_content_to_write) - len(original_content),
                }

                return ToolCallResult(
                    command=f"replace {params.file_path} (dry-run)",
                    output=f"DRY RUN: Would replace {replacements_made} occurrence(s) in {relative_path}. Size would change by {dry_run_info['size_change']} characters.",
                    success=True,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata=dry_run_info,
                    error_message=None,
                )

            # Create backup
            backup_path = f"{params.file_path}.backup"
            shutil.copy2(params.file_path, backup_path)
            # Write new content
            with open(params.file_path, "w", encoding="utf-8") as f:
                f.write(new_content_to_write)
            relative_path = os.path.relpath(params.file_path, os.getcwd())
            return ToolCallResult(
                command=f"replace {params.file_path}",
                output=f"Successfully replaced {replacements_made} occurrence(s) in {relative_path}. Backup created at {os.path.basename(backup_path)}",
                success=True,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
            )
        except Exception as e:
            return ToolCallResult(
                command=f"replace {params.file_path}",
                success=False,
                output="",
                error_message=f"Replace operation failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrates the file system tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_file_system_tools import (
            demo_file_system_tools,
        )

        return await demo_file_system_tools()


class ReadManyFilesTool(BaseTool):
    """Read content from multiple files specified by paths or glob patterns"""

    def _get_name(self) -> str:
        return "read_many_files"

    def _get_description(self) -> str:
        return "Reads content from multiple files specified by paths or glob patterns within a configured target directory. For text files, it concatenates their content into a single string. Useful when you need to understand or analyze a collection of files."

    def _get_pydantic_schema(self):
        return ReadManyFilesParams

    def _get_category(self) -> str:
        return "file_system"

    async def demo(self):
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_file_system_tools import (
            demo_file_system_tools,
        )

        await demo_file_system_tools()

    def _passes_advanced_filters(
        self, file_path: str, params: ReadManyFilesParams
    ) -> bool:
        """Apply advanced filtering criteria to a file."""
        try:
            stat = os.stat(file_path)

            # File size filtering
            if params.min_file_size is not None and stat.st_size < params.min_file_size:
                return False
            if params.max_file_size is not None and stat.st_size > params.max_file_size:
                return False

            # File extension filtering
            if params.file_extension:
                file_ext = Path(file_path).suffix.lower()
                if file_ext not in [ext.lower() for ext in params.file_extension]:
                    return False

            # Modification date filtering
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

            if params.modified_after:
                try:
                    after_date = datetime.fromisoformat(params.modified_after).replace(
                        tzinfo=UTC
                    )
                    if file_mtime < after_date:
                        return False
                except ValueError:
                    logger.warning(
                        f"Invalid date format for modified_after: {params.modified_after}"
                    )

            if params.modified_before:
                try:
                    before_date = datetime.fromisoformat(
                        params.modified_before
                    ).replace(tzinfo=UTC)
                    if file_mtime > before_date:
                        return False
                except ValueError:
                    logger.warning(
                        f"Invalid date format for modified_before: {params.modified_before}"
                    )

            # Content regex filtering (only for text files)
            if params.content_regex:
                try:
                    # Use a simpler check for binary files since we're in a sync method
                    try:
                        with open(file_path, encoding="utf-8", errors="strict") as f:
                            content = f.read(1024)  # Read first 1KB to check
                            if not re.search(
                                params.content_regex,
                                content,
                                re.MULTILINE | re.IGNORECASE,
                            ):
                                # Need to read full file if pattern not found in first part
                                f.seek(0)
                                full_content = f.read()
                                if not re.search(
                                    params.content_regex,
                                    full_content,
                                    re.MULTILINE | re.IGNORECASE,
                                ):
                                    return False
                    except UnicodeDecodeError:
                        # Binary file, skip content filtering
                        pass
                except Exception:
                    # If we can't read the file for content filtering, exclude it
                    return False

            return True

        except OSError:
            return False

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ReadManyFilesParams(**kwargs)
        start_time = datetime.now()
        try:
            all_files = set()
            root_directory = os.getcwd()
            # Default excludes if enabled
            default_excludes = (
                [
                    "**/node_modules/**",
                    "**/.git/**",
                    "**/dist/**",
                    "**/build/**",
                    "**/__pycache__/**",
                    "**/*.pyc",
                    "**/*.pyo",
                    "**/venv/**",
                    "**/env/**",
                    "**/.venv/**",
                ]
                if params.use_default_excludes
                else []
            )
            # Combine all exclude patterns
            all_excludes = default_excludes + (params.exclude or [])
            # Process all paths and include patterns
            all_patterns = params.paths + (params.include or [])
            for pattern in all_patterns:
                if params.recursive:
                    matches = glob_module.glob(pattern, recursive=True)
                else:
                    matches = glob_module.glob(pattern)
                for match in matches:
                    abs_path = os.path.abspath(match)
                    if os.path.isfile(abs_path):
                        # Check if file should be excluded
                        relative_path = os.path.relpath(abs_path, root_directory)
                        should_exclude = False
                        for exclude_pattern in all_excludes:
                            if fnmatch.fnmatch(
                                relative_path, exclude_pattern
                            ) or fnmatch.fnmatch(abs_path, exclude_pattern):
                                should_exclude = True
                                break
                        if not should_exclude and self._passes_advanced_filters(
                            abs_path, params
                        ):
                            all_files.add(abs_path)
            if not all_files:
                return ToolCallResult(
                    command="read_many_files",
                    success=True,
                    output="No files found matching the specified patterns",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,
                )
            # Read all files
            file_contents = []
            files_read = 0
            files_skipped = 0
            for file_path in sorted(all_files):
                try:
                    result = await process_single_file_content(
                        file_path, root_directory
                    )
                    if result.error:
                        files_skipped += 1
                    if isinstance(result.llm_content, str):
                        relative_path = os.path.relpath(file_path, root_directory)
                        file_contents.append(f"--- {relative_path} ---")
                        file_contents.append(result.llm_content)
                        file_contents.append("")  # Empty line separator
                        files_read += 1
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    files_skipped += 1
            if not file_contents:
                return ToolCallResult(
                    command="read_many_files",
                    success=True,
                    output=f"Found {len(all_files)} files but none contained readable text content",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error_message=None,
                )
            # Combine all content
            combined_content = "\n".join(file_contents)
            summary = f"Read {files_read} files successfully"
            if files_skipped > 0:
                summary += f", skipped {files_skipped} files"
            return ToolCallResult(
                command="read_many_files",
                success=True,
                output=f"{summary}\n\n{combined_content}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=None,
            )
        except Exception as e:
            return ToolCallResult(
                command="read_many_files",
                success=False,
                output="",
                error_message=f"Failed to read multiple files: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
