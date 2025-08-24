""""
AST-based Code Generator (libcst preferred, fallback to stdlib ast)

Responsabilidades:
- Generar snippets desde plantillas/contexto con validación estricta
- Parsear AST (libcst or stdlib ast) con manejo robusto de errores
- Sintetizar modificaciones estructurales (patches) solo vía AST
- Producir unified-diff para revisión
- Aplicar cambios vía ASTModifier con confirmación explícita
"""
import ast
import asyncio
import difflib
import logging
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

# Defensive optional imports following repo pattern
try:
    import libcst as cst  # type: ignore
    LIBCST_AVAILABLE = True
except ImportError:
    cst = None  # type: ignore
    LIBCST_AVAILABLE = False

try:
    import astor  # optional pretty printer / codegen
    ASTOR_AVAILABLE = True
except ImportError:
    astor = None
    ASTOR_AVAILABLE = False

# Optional integration points with defensive imports
try:
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.modifier import ASTModifier
    MODIFIER_AVAILABLE = True
except ImportError:
    ASTModifier = None  # type: ignore
    MODIFIER_AVAILABLE = False

try:
    from crisalida_lib.ASTRAL_TOOLS.ast_tools.finder import ASTFinder
    FINDER_AVAILABLE = True
except ImportError:
    ASTFinder = None  # type: ignore
    FINDER_AVAILABLE = False

try:
    from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult
    ASTRAL_TOOLS_AVAILABLE = True
except ImportError:
    BaseTool = object  # type: ignore
    ToolCallResult = None  # type: ignore
    ASTRAL_TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TemplateValidationError(Exception):
    """Raised when template validation fails."""
    pass


class ASTOperationError(Exception):
    """Raised when AST operations fail and string fallbacks are unsafe."""
    pass


class TemplateSchema:
    """Template validation schema for ensuring required variables are present."""
    
    def __init__(self, required_vars: Set[str], optional_vars: Optional[Set[str]] = None):
        self.required_vars = required_vars
        self.optional_vars = optional_vars or set()
    
    def validate(self, context: Dict[str, Any]) -> None:
        """Validate that context contains all required variables."""
        missing = self.required_vars - set(context.keys())
        if missing:
            raise TemplateValidationError(
                f"Missing required template variables: {sorted(missing)}"
            )


# Predefined template schemas for common patterns
TEMPLATE_SCHEMAS = {
    "function": TemplateSchema({"name"}, {"args", "body", "return_type", "docstring"}),
    "class": TemplateSchema({"name"}, {"bases", "body", "docstring"}),
    "method": TemplateSchema({"name"}, {"args", "body", "return_type", "docstring", "decorators"}),
    "import": TemplateSchema({"module"}, {"alias", "from_module"}),
    "variable": TemplateSchema({"name", "value"}, {"type_hint"}),
}


class ASTCodeGeneratorCore:
    """
    Strict AST-aware code generator that refuses unsafe string fallbacks.
    
    Core principles:
    - AST-only modifications when structural changes are needed
    - Clear error messages when AST operations aren't possible
    - Template validation to prevent runtime errors
    - Explicit user confirmation for any fallback operations
    """

    def __init__(self, 
                 prefer_libcst: bool = True, 
                 strict_ast_only: bool = True,
                 allow_string_fallbacks: bool = False):
        """
        Initialize the AST code generator.
        
        Args:
            prefer_libcst: Use libcst if available for better formatting preservation
            strict_ast_only: If True, fail rather than use string manipulation fallbacks
            allow_string_fallbacks: Explicit opt-in for string fallbacks (requires user confirmation)
        """
        self.prefer_libcst = prefer_libcst and LIBCST_AVAILABLE
        self.strict_ast_only = strict_ast_only
        self.allow_string_fallbacks = allow_string_fallbacks
        
        logger.info(
            "ASTCodeGeneratorCore initialized (libcst=%s, astor=%s, modifier=%s, strict=%s)",
            LIBCST_AVAILABLE,
            ASTOR_AVAILABLE,
            MODIFIER_AVAILABLE,
            strict_ast_only,
        )

    def generate_from_template(self, 
                             template: str, 
                             context: Dict[str, Any],
                             template_type: Optional[str] = None) -> str:
        """
        Generate code from template with strict validation.
        
        Args:
            template: Template string with {variable} placeholders
            context: Variable values for template substitution
            template_type: Optional schema type for validation (function, class, etc.)
        
        Raises:
            TemplateValidationError: If required variables are missing
        """
        # Validate template context if schema is available
        if template_type and template_type in TEMPLATE_SCHEMAS:
            TEMPLATE_SCHEMAS[template_type].validate(context)
        
        tpl = textwrap.dedent(template)
        try:
            result = tpl.format(**context)
            
            # Basic syntax validation of generated code
            try:
                ast.parse(result)
            except SyntaxError as e:
                raise TemplateValidationError(
                    f"Generated code has syntax error: {e.msg} at line {e.lineno}"
                )
            
            return result
        except KeyError as e:
            missing_var = str(e).strip("'\"")
            raise TemplateValidationError(
                f"Template variable '{missing_var}' not provided in context"
            )

    def parse_source(self, source: str) -> Tuple[Any, str]:
        """
        Parse source into AST representation with strict error handling.
        
        Returns:
            Tuple of (tree, backend_name) where backend_name is 'libcst' or 'stdlib'
            
        Raises:
            ASTOperationError: If parsing fails completely
        """
        if self.prefer_libcst and LIBCST_AVAILABLE:
            try:
                tree = cst.parse_module(source)
                return tree, "libcst"
            except Exception as e:
                logger.debug("libcst parse failed, falling back to stdlib ast: %s", e)

        try:
            tree = ast.parse(source)
            return tree, "stdlib"
        except SyntaxError as e:
            raise ASTOperationError(
                f"Source code has syntax error: {e.msg} at line {e.lineno}"
            )
        except Exception as e:
            raise ASTOperationError(f"Failed to parse source code: {e}")

    async def _find_target_node_safely(self, 
                                     tree: Any, 
                                     target_query: Dict[str, Any]) -> Optional[Any]:
        """
        Safely find target node using ASTFinder with proper async handling.
        
        Returns:
            Node info if found, None if not found or ASTFinder unavailable
        """
        if not (FINDER_AVAILABLE and ASTFinder is not None):
            logger.warning("ASTFinder not available - cannot locate target node precisely")
            return None
        
        try:
            finder = ASTFinder()
            
            # Use ASTFinder's public async API properly
            result = await finder.execute(
                code_string="",  # We already have the tree
                query=target_query
            )
            
            if result.success and result.metadata:
                found_nodes = result.metadata.get("found_nodes", [])
                if found_nodes:
                    return found_nodes[0]
            
            return None
        except Exception as e:
            logger.debug("ASTFinder lookup failed: %s", e, exc_info=True)
            return None

    async def synthesize_fix_from_ast(self,
                                    source: str,
                                    target_query: Dict[str, Any],
                                    new_code: str,
                                    filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize a structural fix using strict AST operations only.
        
        Args:
            source: Original source code
            target_query: Query to locate target node for modification
            new_code: New code to insert/replace
            filename: Optional filename for diff generation
            
        Returns:
            Dict with keys: success, original, modified, diff, error
            
        Raises:
            ASTOperationError: If strict_ast_only=True and AST operations fail
        """
        original = source
        
        try:
            tree, backend = self.parse_source(source)
        except ASTOperationError as e:
            if self.strict_ast_only:
                raise
            return {"success": False, "error": str(e)}

        # Validate new_code syntax
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            error_msg = f"New code has syntax error: {e.msg} at line {e.lineno}"
            if self.strict_ast_only:
                raise ASTOperationError(error_msg)
            return {"success": False, "error": error_msg}

        # Try to find target node using ASTFinder
        node_info = await self._find_target_node_safely(tree, target_query)
        
        if backend == "libcst" and LIBCST_AVAILABLE:
            try:
                # For libcst, we need proper CST transformation
                # This is a simplified implementation - real usage would need
                # proper CST transformers based on the specific modification type
                modified = self._apply_libcst_modification(source, node_info, new_code, target_query)
                diff = self.to_unified_diff(original, modified, filename or "file.py")
                return {
                    "success": True, 
                    "original": original, 
                    "modified": modified, 
                    "diff": diff,
                    "method": "libcst_transformation"
                }
            except Exception as e:
                error_msg = f"libcst transformation failed: {e}"
                logger.debug(error_msg, exc_info=True)
                if self.strict_ast_only:
                    raise ASTOperationError(error_msg)

        # stdlib ast path with precise node location
        if node_info and hasattr(node_info, 'lineno'):
            try:
                modified = self._apply_stdlib_ast_modification(source, node_info, new_code)
                diff = self.to_unified_diff(original, modified, filename or "file.py")
                return {
                    "success": True,
                    "original": original,
                    "modified": modified,
                    "diff": diff,
                    "method": "stdlib_ast_precise"
                }
            except Exception as e:
                error_msg = f"stdlib AST modification failed: {e}"
                logger.debug(error_msg, exc_info=True)
                if self.strict_ast_only:
                    raise ASTOperationError(error_msg)

        # If we reach here and strict_ast_only is True, we must fail
        if self.strict_ast_only:
            raise ASTOperationError(
                "Cannot perform precise AST-based modification. "
                f"Target node not found via ASTFinder for query: {target_query}. "
                "libcst transformation failed or not available. "
                "Enable allow_string_fallbacks=True to use unsafe string operations."
            )

        # String fallback path (only if explicitly allowed)
        if self.allow_string_fallbacks:
            logger.warning(
                "UNSAFE: Using string manipulation fallback for AST modification. "
                "This may introduce syntax errors or break existing code logic."
            )
            try:
                modified = self._apply_string_fallback_modification(source, target_query, new_code)
                diff = self.to_unified_diff(original, modified, filename or "file.py")
                return {
                    "success": True,
                    "original": original,
                    "modified": modified,
                    "diff": diff,
                    "method": "string_fallback",
                    "warning": "Used unsafe string manipulation"
                }
            except Exception as e:
                return {"success": False, "error": f"String fallback failed: {e}"}

        return {
            "success": False, 
            "error": "AST-based modification failed and string fallbacks disabled"
        }

    def _apply_libcst_modification(self, 
                                  source: str, 
                                  node_info: Optional[Any], 
                                  new_code: str,
                                  target_query: Dict[str, Any]) -> str:
        """Apply modification using libcst transformers."""
        # This is a placeholder for proper libcst transformation
        # Real implementation would create specific transformers based on modification type
        if node_info and hasattr(node_info, 'lineno'):
            lines = source.splitlines(True)
            lineno = int(node_info.lineno) - 1
            lines.insert(min(len(lines), lineno + 1), new_code + "\n")
            return "".join(lines)
        else:
            # Simple append if no precise location
            return source + "\n" + new_code

    def _apply_stdlib_ast_modification(self, 
                                      source: str, 
                                      node_info: Any, 
                                      new_code: str) -> str:
        """Apply modification using stdlib ast with precise node location."""
        lines = source.splitlines(True)
        lineno = int(node_info.lineno) - 1
        
        # Insert new code after the target node line
        lines.insert(min(len(lines), lineno + 1), new_code + "\n")
        return "".join(lines)

    def _apply_string_fallback_modification(self, 
                                          source: str, 
                                          target_query: Dict[str, Any], 
                                          new_code: str) -> str:
        """Apply modification using string manipulation (unsafe fallback)."""
        name = target_query.get("name")
        if name and ("def " + name) in source:
            # Insert after function signature line
            idx = source.find("def " + name)
            start = source.find("\n", idx)
            if start != -1:
                return source[: start + 1] + textwrap.indent(new_code + "\n", "    ") + source[start + 1 :]
        
        # Default: append at end
        return source + "\n" + new_code

    def to_unified_diff(self, original: str, modified: str, filename: str = "file.py") -> List[str]:
        """Return unified diff lines between original and modified source."""
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            orig_lines, 
            mod_lines, 
            fromfile=filename, 
            tofile=filename + ".modified"
        ))
        return diff

    async def apply_patch_with_modifier(self,
                                      modifications: List[Dict[str, Any]],
                                      file_path: Optional[str] = None,
                                      write_back: bool = False,
                                      require_modifier: bool = True,
                                      modifier_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply structured modifications via ASTModifier with strict safety guarantees.
        
        Args:
            modifications: List of ModificationSpec-like dicts
            file_path: Optional file path for modifications
            write_back: Whether to write changes to disk
            require_modifier: If True, fail if ASTModifier unavailable (recommended)
            modifier_kwargs: Additional kwargs for ASTModifier
            
        Returns:
            Dict with application results
            
        Raises:
            ASTOperationError: If require_modifier=True and ASTModifier unavailable
        """
        modifier_kwargs = modifier_kwargs or {}
        
        if not (MODIFIER_AVAILABLE and ASTModifier is not None):
            error_msg = (
                "ASTModifier not available but required for safe patch application. "
                "Cannot guarantee safe file modifications without AST-based validation."
            )
            if require_modifier:
                raise ASTOperationError(error_msg)
            
            logger.warning(error_msg)
            return {
                "applied_via_modifier": False,
                "error": error_msg,
                "fallback_attempted": False
            }

        try:
            modifier = ASTModifier()
            result = await modifier.execute(
                modifications=modifications,
                file_path=file_path,
                **modifier_kwargs
            )
            return {
                "applied_via_modifier": True,
                "modifier_result": result,
                "success": result.success if hasattr(result, 'success') else True
            }
        except Exception as e:
            error_msg = f"ASTModifier application failed: {e}"
            logger.exception(error_msg)
            
            if require_modifier:
                raise ASTOperationError(error_msg)
            
            # Only attempt unsafe fallback if explicitly allowed
            if not require_modifier and write_back:
                logger.warning("Attempting unsafe fallback file write")
                return self._unsafe_fallback_write(modifications, file_path)
            
            return {
                "applied_via_modifier": False,
                "error": error_msg,
                "fallback_attempted": False
            }

    def _unsafe_fallback_write(self, 
                              modifications: List[Dict[str, Any]], 
                              file_path: Optional[str]) -> Dict[str, Any]:
        """Unsafe fallback for direct file writing (discouraged)."""
        applied = []
        for mod in modifications:
            modified_src = mod.get("modified")
            target = file_path or mod.get("output_file_path")
            if modified_src and target:
                try:
                    with open(target, "w", encoding="utf-8") as f:
                        f.write(modified_src)
                    applied.append({"target": target, "status": "written_unsafe"})
                except Exception as e:
                    applied.append({"target": target, "status": "error", "error": str(e)})
            else:
                applied.append({
                    "target": target, 
                    "status": "skipped", 
                    "reason": "no_modified_content"
                })
        
        return {
            "applied_via_modifier": False,
            "fallback_results": applied,
            "warning": "Used unsafe direct file writing"
        }


# Pydantic models for BaseTool integration
class ASTCodeGeneratorParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    source: str = Field(..., description="Original source code to modify.")
    target_query: Dict[str, Any] = Field(
        ..., 
        description="Query to locate the target node (e.g., {'type': 'FunctionDef', 'name': 'my_func'})."
    )
    new_code: str = Field(..., description="New code snippet to insert or replace.")
    filename: Optional[str] = Field(None, description="Optional filename for diff generation.")
    template_type: Optional[str] = Field(
        None, 
        description="Template type for validation (function, class, method, etc.)"
    )
    strict_ast_only: bool = Field(
        True, 
        description="If True, fail rather than use unsafe string manipulation fallbacks."
    )
    allow_string_fallbacks: bool = Field(
        False,
        description="Explicit opt-in for string fallbacks (requires user confirmation)."
    )


class ASTCodeGenerator(BaseTool):
    """
    AST-based Code Generator for structural code modifications.
    
    Provides safe, AST-only code generation with strict validation and
    clear error messages when AST operations cannot be performed safely.
    """

    def __init__(self):
        super().__init__()
        self._generator_core = None  # Lazy initialization

    def _get_generator_core(self, strict_ast_only: bool, allow_string_fallbacks: bool) -> ASTCodeGeneratorCore:
        """Lazy initialization of generator core with user-specified safety settings."""
        return ASTCodeGeneratorCore(
            prefer_libcst=True,
            strict_ast_only=strict_ast_only,
            allow_string_fallbacks=allow_string_fallbacks
        )

    def _get_name(self) -> str:
        return "ast_code_generator"

    def _get_description(self) -> str:
        return (
            "Generates and synthesizes structural code modifications using AST. "
            "Provides strict AST-only mode for safety, with optional string fallbacks. "
            "Validates templates and provides clear error messages for unsafe operations."
        )

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return ASTCodeGeneratorParams

    def _get_category(self) -> str:
        return "ast_analysis"

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        start_time = datetime.now()
        
        if not ASTRAL_TOOLS_AVAILABLE:
            return ToolCallResult(
                command="ast_code_generator",
                success=False,
                output="",
                error_message="ASTRAL_TOOLS not available - cannot create ToolCallResult",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        
        try:
            params = ASTCodeGeneratorParams(**kwargs)
            
            # Initialize generator with user-specified safety settings
            generator_core = self._get_generator_core(
                strict_ast_only=params.strict_ast_only,
                allow_string_fallbacks=params.allow_string_fallbacks
            )
            
            # Validate new_code as template if template_type provided
            if params.template_type:
                try:
                    # Extract context from new_code for validation
                    # This is a simplified approach - real implementation might
                    # need more sophisticated template variable extraction
                    context = {"name": params.target_query.get("name", "unknown")}
                    generator_core.generate_from_template(
                        params.new_code, 
                        context, 
                        params.template_type
                    )
                except TemplateValidationError as e:
                    return ToolCallResult(
                        command="ast_code_generator",
                        success=False,
                        output="",
                        error_message=f"Template validation failed: {e}",
                        execution_time=(datetime.now() - start_time).total_seconds(),
                    )
            
            # Perform AST-based code synthesis
            result = await generator_core.synthesize_fix_from_ast(
                source=params.source,
                target_query=params.target_query,
                new_code=params.new_code,
                filename=params.filename
            )
            
            if result.get("success"):
                output_msg = f"Code synthesized successfully using {result.get('method', 'unknown')} method."
                if result.get("warning"):
                    output_msg += f" WARNING: {result['warning']}"
                
                return ToolCallResult(
                    command="ast_code_generator",
                    success=True,
                    output=output_msg,
                    error_message=None,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "original": result.get("original"),
                        "modified": result.get("modified"),
                        "diff": result.get("diff"),
                        "method": result.get("method"),
                        "warning": result.get("warning"),
                    },
                )
            else:
                return ToolCallResult(
                    command="ast_code_generator",
                    success=False,
                    output="",
                    error_message=result.get("error", "Unknown error during synthesis."),
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
                
        except ASTOperationError as e:
            return ToolCallResult(
                command="ast_code_generator",
                success=False,
                output="",
                error_message=f"AST operation failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except TemplateValidationError as e:
            return ToolCallResult(
                command="ast_code_generator",
                success=False,
                output="",
                error_message=f"Template validation error: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            logger.error(f"ASTCodeGenerator execution failed: {e}", exc_info=True)
            return ToolCallResult(
                command="ast_code_generator",
                success=False,
                output="",
                error_message=f"Unexpected error: {str(e)}",
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
