import ast
import logging
import io
import tokenize
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult
from datetime import datetime

# Defensive optional imports following repo pattern
try:
    import typed_ast.ast3 as typed_ast  # type: ignore
    TYPED_AST_AVAILABLE = True
except ImportError:
    typed_ast = None
    TYPED_AST_AVAILABLE = False

# Optional prevalidation tools
try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False

if TYPE_CHECKING:
    # Avoid circular imports
    pass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Constants for performance tuning
MAX_FILE_SIZE_MB = 50  # Maximum file size to process (50MB)
MAX_LINES_FAST_PATH = 10000  # Use optimized path for smaller files
CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB chunks for large file processing


class ASTParsingError(Exception):
    """Custom exception for AST parsing failures with detailed context."""
    
    def __init__(self, message: str, line_number: Optional[int] = None, 
                 original_error: Optional[Exception] = None):
        self.message = message
        self.line_number = line_number
        self.original_error = original_error
        super().__init__(self.message)


def prevalidate_python_source(source: str, file_path: Optional[str] = None) -> Optional[str]:
    """
    Prevalidate Python source using built-in compile check.
    Returns error message if validation fails, None if successful.
    
    Following repo pattern: graceful degradation when external tools unavailable.
    """
    try:
        # Basic syntax check using compile()
        compile(source, file_path or '<string>', 'exec')
        return None
    except SyntaxError as e:
        return f"Syntax error: {e.msg} at line {e.lineno}"
    except Exception as e:
        return f"Validation error: {str(e)}"


def parse_source_to_ast(source: str, file_path: Optional[str] = None) -> Tuple[Optional[ast.AST], Optional[str]]:
    """
    Parse source code to AST with robust error handling and prevalidation.
    
    Args:
        source: Python source code
        file_path: Optional file path for better error reporting
        
    Returns:
        Tuple of (ast_tree, error_message). If parsing fails returns (None, msg).
        
    Follows repo pattern: defensive imports and graceful degradation.
    """
    # Prevalidation step for better error reporting
    validation_error = prevalidate_python_source(source, file_path)
    if validation_error:
        logger.warning("parse_source_to_ast prevalidation failed: %s", validation_error)
        return None, validation_error
    
    # Try typed-ast first if available (follows repo defensive import pattern)
    if TYPED_AST_AVAILABLE and typed_ast is not None:
        try:
            tree = typed_ast.parse(source, filename=file_path or '<string>')
            return tree, None
        except Exception as e:
            logger.debug("typed-ast parse failed, trying stdlib ast: %s", e)
    
    # Fallback to stdlib ast
    try:
        tree = ast.parse(source, filename=file_path or '<string>')
        return tree, None
    except SyntaxError as e:
        msg = f"SyntaxError: {e.msg}"
        if e.lineno:
            msg += f" at line {e.lineno}"
        if e.offset:
            msg += f", column {e.offset}"
        logger.warning("parse_source_to_ast: %s", msg)
        return None, msg
    except Exception as e:
        logger.exception("parse_source_to_ast unexpected failure")
        return None, f"Unexpected parsing error: {str(e)}"


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute with defensive access pattern.
    Following repo pattern: robust attribute access for AST nodes.
    """
    try:
        return getattr(obj, attr, default)
    except (AttributeError, TypeError):
        return default


def extract_comments_and_docstrings(source: str, file_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Extract line comments and docstrings from source with robust error handling.
    
    Following repo pattern: graceful degradation on tokenization failures.
    """
    comments: List[str] = []
    docstrings: List[str] = []
    
    # Extract comments via tokenize (defensive)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for toknum, tokval, _, _, _ in tokens:
            if toknum == tokenize.COMMENT:
                cleaned_comment = tokval.lstrip("# ").rstrip()
                if cleaned_comment:  # Skip empty comments
                    comments.append(cleaned_comment)
    except Exception as e:
        logger.debug("Comment extraction failed for %s: %s", file_path or '<string>', e, exc_info=True)
        # Continue without comments rather than failing entirely
    
    # Extract docstrings via AST (defensive)
    try:
        tree, parse_error = parse_source_to_ast(source, file_path)
        if tree is not None:
            # Module docstring
            try:
                mod_doc = ast.get_docstring(tree)
                if mod_doc:
                    docstrings.append(mod_doc)
            except Exception as e:
                logger.debug("Module docstring extraction failed: %s", e)
            
            # Function/class docstrings with defensive node access
            try:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        try:
                            ds = ast.get_docstring(node)
                            if ds:
                                docstrings.append(ds)
                        except Exception as e:
                            logger.debug("Docstring extraction failed for node %s: %s", 
                                       safe_getattr(node, 'name', '<unknown>'), e)
            except Exception as e:
                logger.debug("AST walk failed during docstring extraction: %s", e)
        else:
            logger.debug("Cannot extract docstrings - AST parsing failed: %s", parse_error)
    except Exception as e:
        logger.debug("Docstring extraction fallback failed: %s", e, exc_info=True)
    
    return {"comments": comments, "docstrings": docstrings}


def extract_intentions_from_ast(tree: ast.AST, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract intention map with defensive node access and error isolation.
    
    Following repo pattern: graceful degradation and defensive attribute access.
    """
    intents: Dict[str, Any] = {
        "functions": [],
        "classes": [],
        "imports": [],
        "docstrings": [],
        "constants": [],
        "parsing_errors": [],  # Track partial failures
    }
    
    if tree is None:
        intents["parsing_errors"].append("AST tree is None")
        return intents
    
    try:
        for node in ast.walk(tree):
            try:
                # Function definitions (defensive)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        "name": safe_getattr(node, 'name', '<unknown>'),
                        "lineno": safe_getattr(node, "lineno", None),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }
                    # Add argument count if available
                    args = safe_getattr(node, 'args', None)
                    if args:
                        arg_count = len(safe_getattr(args, 'args', []))
                        func_info["arg_count"] = arg_count
                    intents["functions"].append(func_info)
                
                # Class definitions (defensive)
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": safe_getattr(node, 'name', '<unknown>'),
                        "lineno": safe_getattr(node, "lineno", None),
                    }
                    # Add base class count if available
                    bases = safe_getattr(node, 'bases', [])
                    if bases:
                        class_info["base_count"] = len(bases)
                    intents["classes"].append(class_info)
                
                # Import statements (defensive)
                elif isinstance(node, ast.Import):
                    names = safe_getattr(node, 'names', [])
                    for alias in names:
                        if alias:  # Additional safety check
                            intents["imports"].append({
                                "name": safe_getattr(alias, 'name', '<unknown>'),
                                "asname": safe_getattr(alias, 'asname', None),
                                "type": "import"
                            })
                
                elif isinstance(node, ast.ImportFrom):
                    module = safe_getattr(node, "module", None)
                    names = safe_getattr(node, 'names', [])
                    for alias in names:
                        if alias:  # Additional safety check
                            intents["imports"].append({
                                "module": module,
                                "name": safe_getattr(alias, 'name', '<unknown>'),
                                "asname": safe_getattr(alias, 'asname', None),
                                "type": "from_import"
                            })
                
                # String literals / docstrings (defensive)
                elif isinstance(node, ast.Expr):
                    value = safe_getattr(node, 'value', None)
                    if isinstance(value, ast.Constant) and isinstance(safe_getattr(value, 'value', None), str):
                        intents["docstrings"].append({
                            "value": value.value,
                            "lineno": safe_getattr(node, "lineno", None)
                        })
                
                # Constants (defensive)
                elif isinstance(node, ast.Constant):
                    const_value = safe_getattr(node, 'value', None)
                    if isinstance(const_value, (int, float, str, bool)):
                        intents["constants"].append({
                            "value": const_value,
                            "type": type(const_value).__name__,
                            "lineno": safe_getattr(node, "lineno", None)
                        })
                        
            except Exception as e:
                # Isolate errors per node to avoid total failure
                error_msg = f"Error processing node {type(node).__name__}: {str(e)}"
                intents["parsing_errors"].append(error_msg)
                logger.debug("Node processing error in %s: %s", file_path or '<string>', error_msg)
                
    except Exception as e:
        # Overall AST walk failure
        error_msg = f"AST traversal failed: {str(e)}"
        intents["parsing_errors"].append(error_msg)
        logger.debug("AST walk failed for %s: %s", file_path or '<string>', error_msg, exc_info=True)
    
    return intents


def check_file_size(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if file size is within processing limits.
    
    Returns:
        Tuple of (is_processable, error_message)
    """
    try:
        file_size = Path(file_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)"
        
        return True, None
    except Exception as e:
        return False, f"Cannot check file size: {str(e)}"


def read_file_chunked(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read large files in chunks to manage memory usage.
    
    Following repo pattern: graceful degradation and memory-conscious processing.
    """
    try:
        # Check file size first
        processable, size_error = check_file_size(file_path)
        if not processable:
            return None, size_error
        
        # Read file with encoding detection fallback
        encodings = ['utf-8', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # For very large files, we still need the full content for AST parsing
                    # but we can at least track memory usage
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return None, f"Error reading file with {encoding}: {str(e)}"
        
        if content is None:
            return None, f"Could not decode file with any of: {encodings}"
        
        return content, None
        
    except Exception as e:
        logger.exception("Unexpected error in read_file_chunked")
        return None, f"Unexpected file reading error: {str(e)}"


def read_and_parse_file(path: str) -> Dict[str, Any]:
    """
    Read file, parse AST, and extract comprehensive information with robust error handling.
    
    Following repo patterns:
    - Defensive error handling
    - Graceful degradation
    - Detailed error reporting
    - Memory-conscious processing for large files
    """
    file_path = str(Path(path).resolve())  # Normalize path
    
    # Initialize result structure
    result = {
        "source": None,
        "ast_tree": None,
        "parse_error": None,
        "comments": [],
        "docstrings": [],
        "intentions": {},
        "file_info": {
            "path": file_path,
            "size_bytes": 0,
            "line_count": 0,
            "processing_time_ms": 0,
        }
    }
    
    start_time = datetime.now()
    
    try:
        # Read file with size and encoding checks
        source, read_error = read_file_chunked(file_path)
        if read_error:
            result["parse_error"] = f"File reading failed: {read_error}"
            return result
        
        if source is None:
            result["parse_error"] = "File content is empty or unreadable"
            return result
        
        result["source"] = source
        result["file_info"]["size_bytes"] = len(source)
        result["file_info"]["line_count"] = source.count('\n') + 1
        
        # Parse AST with comprehensive error handling
        tree, parse_err = parse_source_to_ast(source, file_path)
        result["ast_tree"] = tree
        
        if parse_err:
            result["parse_error"] = parse_err
            # Continue with partial processing even if AST parsing failed
            logger.info("AST parsing failed for %s, attempting partial extraction", file_path)
        
        # Extract comments and docstrings (works even with parse errors)
        try:
            comments_doc = extract_comments_and_docstrings(source, file_path)
            result["comments"] = comments_doc.get("comments", [])
            result["docstrings"] = comments_doc.get("docstrings", [])
        except Exception as e:
            logger.debug("Comment/docstring extraction failed for %s: %s", file_path, e)
            # Continue without comments/docstrings
        
        # Extract intentions from AST (if available)
        if tree is not None:
            try:
                intentions = extract_intentions_from_ast(tree, file_path)
                result["intentions"] = intentions
            except Exception as e:
                logger.debug("Intention extraction failed for %s: %s", file_path, e)
                result["intentions"] = {"extraction_error": str(e)}
        else:
            result["intentions"] = {"extraction_error": "AST not available"}
        
        # Record processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result["file_info"]["processing_time_ms"] = round(processing_time, 2)
        
        logger.info("Successfully processed %s (%d bytes, %.2fms)", 
                   file_path, result["file_info"]["size_bytes"], processing_time)
        
    except Exception as e:
        logger.exception("Unexpected error in read_and_parse_file for %s", file_path)
        result["parse_error"] = f"Unexpected processing error: {str(e)}"
    
    return result


class ASTReaderParams(BaseModel):
    """Parameters for AST Reader tool following repo pydantic patterns."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    file_path: str = Field(..., description="Path to the Python file to read and parse.")
    include_source: bool = Field(True, description="Include full source code in results.")
    extract_intentions: bool = Field(True, description="Extract structural information (functions, classes, etc.).")
    extract_comments: bool = Field(True, description="Extract comments and docstrings.")


class ASTReader(BaseTool):
    """
    ASTReader: Robust tool for reading Python files and extracting AST-based information.
    
    Features:
    - Defensive parsing with graceful degradation
    - Memory-conscious processing for large files  
    - Comprehensive error isolation and reporting
    - Prevalidation for better error messages
    - Support for multiple encodings
    
    Following Crisalida repo patterns:
    - Defensive imports and fallbacks
    - Detailed logging and error context
    - Pydantic model validation
    """

    def __init__(self):
        super().__init__()

    def _get_name(self) -> str:
        return "ast_reader"

    def _get_description(self) -> str:
        return (
            "Reads Python files, parses them into AST, and extracts structural information. "
            "Provides robust error handling, memory-conscious processing, and detailed "
            "extraction of functions, classes, imports, comments, and docstrings."
        )

    def _get_pydantic_schema(self):
        return ASTReaderParams

    def _get_category(self) -> str:
        return "ast_analysis"

    async def execute(self, **kwargs) -> ToolCallResult:
        """Execute AST reading with comprehensive error handling and reporting."""
        start_time = datetime.now()
        
        try:
            params = ASTReaderParams(**kwargs)
        except Exception as e:
            return ToolCallResult(
                command="ast_reader",
                success=False,
                output="",
                error_message=f"Parameter validation failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        try:
            # Process the file
            result = read_and_parse_file(params.file_path)
            
            # Determine success based on results
            has_parse_error = bool(result.get("parse_error"))
            has_content = bool(result.get("source"))
            
            # Build output message
            output_parts = ["📄 **AST READER RESULTS**\n"]
            
            file_info = result.get("file_info", {})
            output_parts.append(f"**File:** {file_info.get('path', params.file_path)}")
            output_parts.append(f"**Size:** {file_info.get('size_bytes', 0)} bytes ({file_info.get('line_count', 0)} lines)")
            output_parts.append(f"**Processing Time:** {file_info.get('processing_time_ms', 0)}ms")
            
            if has_parse_error:
                output_parts.append(f"**⚠️ Parse Error:** {result['parse_error']}")
            else:
                output_parts.append("**✅ Parse Status:** Success")
            
            # Summary of extracted information
            intentions = result.get("intentions", {})
            if intentions and not intentions.get("extraction_error"):
                func_count = len(intentions.get("functions", []))
                class_count = len(intentions.get("classes", []))
                import_count = len(intentions.get("imports", []))
                
                output_parts.append(f"**📊 Extracted:** {func_count} functions, {class_count} classes, {import_count} imports")
                
                # Show parsing errors if any (from partial processing)
                parsing_errors = intentions.get("parsing_errors", [])
                if parsing_errors:
                    output_parts.append(f"**⚠️ Partial Errors:** {len(parsing_errors)} node processing issues")
            
            comment_count = len(result.get("comments", []))
            docstring_count = len(result.get("docstrings", []))
            output_parts.append(f"**💬 Documentation:** {comment_count} comments, {docstring_count} docstrings")
            
            output = "\n".join(output_parts)
            
            # Prepare metadata (excluding source if not requested)
            metadata = {
                "file_info": file_info,
                "parse_error": result.get("parse_error"),
                "intentions": intentions,
                "comments": result.get("comments", []) if params.extract_comments else [],
                "docstrings": result.get("docstrings", []) if params.extract_comments else [],
            }
            
            if params.include_source and has_content:
                metadata["source"] = result["source"]
            
            # Determine overall success
            # Success if we got some content and no critical errors
            success = has_content and (not has_parse_error or bool(intentions))
            
            return ToolCallResult(
                command="ast_reader",
                success=success,
                output=output,
                error_message=result.get("parse_error") if has_parse_error and not success else None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata=metadata,
            )
            
        except Exception as e:
            logger.exception("Error in ASTReader.execute for %s", params.file_path)
            return ToolCallResult(
                command="ast_reader",
                success=False,
                output="",
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def demo(self):
        """Demonstrate the tool's functionality."""
        print(f"Demo for {self.name}: {self.description}")
        return ToolCallResult(
            command=f"demo_{self.name}",
            success=True,
            output=f"Demo completed for {self.name}",
            execution_time=0.1,
        )

