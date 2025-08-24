import ast
import copy
import hashlib
import logging
import subprocess
from datetime import datetime
from typing import Any

import black
from black import FileMode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from crisalida_lib.ASTRAL_TOOLS.ast_tools.exceptions import (
    InvalidModificationTarget,
    SyntaxErrorInNewCode,
)
from crisalida_lib.ASTRAL_TOOLS.ast_tools.finder import ASTFinder
from crisalida_lib.ASTRAL_TOOLS.ast_tools.models import (
    ASTNodeInfo,
    ModificationOperation,
    ModificationSpec,
)
from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class ASTModifierParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: str | None = Field(
        None,
        description="Path to the Python file to parse into an AST for modification.",
    )
    code_string: str | None = Field(
        None, description="Python code string to parse into an AST for modification."
    )
    modifications: list[ModificationSpec] = Field(
        ...,
        description="A list of ModificationSpec objects defining the modifications.",
    )
    output_file_path: str | None = Field(
        None, description="Optional: Path to write the modified code to."
    )
    format_output: bool = Field(
        True, description="Format the output code using the black formatter."
    )
    formatter: str = Field(
        "black", description="Code formatter to use: 'black' or 'ruff'"
    )

    @model_validator(mode="after")
    def check_either_file_path_or_code_string(self):
        if (self.file_path is None and self.code_string is None) or (
            self.file_path is not None and self.code_string is not None
        ):
            raise ValueError(
                "Either 'file_path' or 'code_string' must be provided, but not both."
            )
        return self


class ASTModifier(BaseTool):
    """
    ASTModifier: Herramienta avanzada para modificar AST con rollback, validación y refactorización.
    Soporta operaciones de reemplazo, inserción, borrado, refactorización, extracción y renombrado seguro.

    Example:
        modifier = ASTModifier()
        modifications = [
            ModificationSpec(
                operation=ModificationOperation.REPLACE,
                target_query={"type": "Name", "id": "old_name"},
                new_code="new_name",
            )
        ]
        result = await modifier.execute(
            file_path="my_file.py",
            modifications=modifications,
            output_file_path="my_modified_file.py",
        )
    """

    def __init__(self):
        super().__init__()
        self.finder = ASTFinder()
        self.modification_history: list[dict[str, Any]] = []
        self.backup_trees: dict[str, ast.AST] = {}

    def _get_name(self) -> str:
        return "ast_modifier"

    def _get_description(self) -> str:
        return "Applies modifications to an AST with enhanced safety and rollback capabilities."

    def _get_pydantic_schema(self):
        return ASTModifierParams

    def _get_category(self) -> str:
        return "ast_modification"

    def _establish_parent_references(self, ast_tree: ast.AST):
        for node in ast.walk(ast_tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ASTModifierParams(**kwargs)
        modifications = params.modifications
        start_time = datetime.now()

        code_content = None
        if params.file_path:
            try:
                with open(params.file_path, encoding="utf-8") as f:
                    code_content = f.read()
            except FileNotFoundError:
                return ToolCallResult(
                    command="ast_modifier",
                    success=False,
                    output="",
                    error_message=f"File not found: {params.file_path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            except Exception as e:
                return ToolCallResult(
                    command="ast_modifier",
                    success=False,
                    output="",
                    error_message=f"Error reading file {params.file_path}: {e}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        elif params.code_string:
            code_content = params.code_string

        if not code_content:
            return ToolCallResult(
                command="ast_modifier",
                success=False,
                output="",
                error_message="No code content provided for AST parsing.",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        try:
            ast_tree = ast.parse(code_content)
            self._establish_parent_references(ast_tree)
        except SyntaxError as e:
            return ToolCallResult(
                command="ast_modifier",
                success=False,
                output="",
                error_message=f"Syntax error in code: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return ToolCallResult(
                command="ast_modifier",
                success=False,
                output="",
                error_message=f"Error parsing AST: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        try:
            modified_tree = await self.apply_modifications(ast_tree, modifications)
            if params.output_file_path:
                try:
                    modified_code = ast.unparse(modified_tree)
                    format_warning = None
                    if params.format_output:
                        try:
                            modified_code = self._format_code(
                                modified_code, params.formatter
                            )
                        except ValueError as e:
                            logger.warning(f"Code formatting failed: {e}")
                            format_warning = f"Formatting failed: {e}"

                    with open(params.output_file_path, "w", encoding="utf-8") as f:
                        f.write(modified_code)

                    if format_warning:
                        output_message = f"AST modifications applied successfully. Modified code written to {params.output_file_path}. Warning: {format_warning}"
                    else:
                        output_message = f"AST modifications applied successfully. Modified code written to {params.output_file_path}."
                except Exception as e:
                    output_message = f"AST modifications applied, but failed to write to file {params.output_file_path}: {e}"
                    logger.error(output_message)
            else:
                output_message = "AST modifications applied successfully."

            return ToolCallResult(
                command="ast_modifier",
                success=True,
                output=output_message,
                error_message=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"modified_tree": modified_tree},
            )
        except Exception as e:
            import traceback

            logger.error(f"Error in ASTModifier.execute: {e}\n{traceback.format_exc()}")
            return ToolCallResult(
                command="ast_modifier",
                success=False,
                output="",
                error_message=f"Error applying AST modifications: {e}\n{traceback.format_exc()}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _validate_ast(self, ast_tree: ast.AST) -> dict[str, Any]:
        try:
            code_string = ast.unparse(ast_tree)
            compile(code_string, filename="<ast>", mode="exec")
            return {"valid": True, "errors": []}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def _record_modifications(self, modifications: list[ModificationSpec]) -> None:
        self.modification_history.append(
            {"timestamp": datetime.now(), "modifications": modifications}
        )

    def _parse_code_snippet(
        self,
        code: str,
        expected_type: type | None = None,
    ) -> ast.AST:
        try:
            node: ast.AST
            if expected_type is ast.expr:
                node = ast.parse(code, mode="eval").body
            else:
                node = ast.parse(code).body[0]

            if expected_type and not isinstance(node, expected_type):
                raise ValueError(f"Parsed node is not of expected type {expected_type}")
            return node
        except SyntaxError as e:
            raise SyntaxErrorInNewCode(new_code=code, original_error=e) from e

    def _perform_node_replacement(
        self,
        node_info: ASTNodeInfo,
        new_node: ast.AST,
    ) -> None:
        parent = node_info.parent
        if parent is None:
            raise ValueError("Cannot replace root node")
        if node_info.field_name is not None:
            if node_info.index is not None:
                getattr(parent, node_info.field_name)[node_info.index] = new_node
            else:
                setattr(parent, node_info.field_name, new_node)
        else:
            raise ValueError("Cannot determine field to replace node")

    def _perform_node_insertion(
        self,
        node_info: ASTNodeInfo,
        new_node: ast.AST,
        before: bool,
    ) -> None:
        parent = node_info.parent
        if parent is None or node_info.field_name is None or node_info.index is None:
            raise ValueError("Cannot insert node: missing parent or field/index")
        siblings = getattr(parent, node_info.field_name)
        idx = node_info.index
        if before:
            siblings.insert(idx, new_node)
        else:
            siblings.insert(idx + 1, new_node)

    def _perform_node_deletion(self, node_info: ASTNodeInfo) -> None:
        parent = node_info.parent
        if parent is None or node_info.field_name is None:
            raise ValueError("Cannot delete root node or missing field")
        if node_info.index is not None:
            siblings = getattr(parent, node_info.field_name)
            del siblings[node_info.index]
        else:
            setattr(parent, node_info.field_name, None)

    def _replace_placeholder_in_wrapper(
        self,
        wrapper_node: ast.AST,
        wrapped_node: ast.AST,
    ) -> None:
        class PassReplacer(ast.NodeTransformer):
            def visit_Pass(self, node):
                return wrapped_node

        PassReplacer().visit(wrapper_node)

    def _extract_function(
        self,
        ast_tree: ast.AST,
        node_info: ASTNodeInfo,
        modification: ModificationSpec,
    ) -> None:
        if not modification.extract_name:
            raise ValueError("extract_name is required for extract operation")
        func_def = ast.FunctionDef(
            name=modification.extract_name,
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[copy.deepcopy(node_info.node)],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )  # type: ignore
        module = ast_tree if isinstance(ast_tree, ast.Module) else None
        if module:
            module.body.insert(0, func_def)
        self._perform_node_replacement(
            node_info, ast.Name(id=modification.extract_name, ctx=ast.Load())
        )

    def _refactor_node(
        self,
        ast_tree: ast.AST,
        node_info: ASTNodeInfo,
        modification: ModificationSpec,
    ) -> None:
        if not modification.attribute or modification.value is None:
            raise ValueError("attribute and value required for refactor operation")
        setattr(node_info.node, modification.attribute, modification.value)

    def _rename_symbol_scoped(
        self,
        ast_tree: ast.AST,
        node_info: ASTNodeInfo,
        modification: ModificationSpec,
    ) -> None:
        if not modification.attribute or not modification.value:
            raise ValueError(
                "attribute (old_name) and value (new_name) required for rename operation"
            )

        old_name = modification.attribute
        new_name = modification.value
        scope_node = node_info.node

        if not self._is_safe_rename(scope_node, old_name, new_name):
            raise ValueError(f"Unsafe rename: '{new_name}' would cause conflicts")

        class Renamer(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == old_name:
                    node.id = new_name
                return node

            def visit_arg(self, node: ast.arg) -> ast.arg:
                if node.arg == old_name:
                    node.arg = new_name
                return node

        Renamer().visit(scope_node)

    def _is_safe_rename(
        self, scope_node: ast.AST, old_name: str, new_name: str
    ) -> bool:
        existing_names = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    existing_names.add(node.id)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                existing_names.add(node.name)
                for arg in node.args.args:
                    existing_names.add(arg.arg)
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                existing_names.add(node.name)
                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    existing_names.add(name)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    existing_names.add(name)

        NameCollector().visit(scope_node)
        return new_name not in existing_names or new_name == old_name

    def _add_extract_method_operation(
        self,
        ast_tree: ast.AST,
        target_statements: list[ast.stmt],
        method_name: str,
        return_vars: list[str] | None = None,
    ) -> ast.FunctionDef:
        return_vars = return_vars or []
        used_vars, defined_vars = self._analyze_variable_usage(target_statements)
        params = [var for var in used_vars if var not in defined_vars]
        args = [ast.arg(arg=param, annotation=None) for param in params]
        if return_vars:
            if len(return_vars) == 1:
                return_node = ast.Return(
                    value=ast.Name(id=return_vars[0], ctx=ast.Load())
                )
            else:
                return_node = ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(id=var, ctx=ast.Load()) for var in return_vars],
                        ctx=ast.Load(),
                    )
                )
            target_statements.append(return_node)
        new_function = ast.FunctionDef(
            name=method_name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=target_statements,
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        return new_function

    def _analyze_variable_usage(
        self, statements: list[ast.stmt]
    ) -> tuple[set[str], set[str]]:
        used_vars = set()
        defined_vars = set()

        class VariableAnalyzer(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
                elif isinstance(node.ctx, (ast.Store, ast.Del)):
                    defined_vars.add(node.id)

            def visit_FunctionDef(self, node):
                defined_vars.add(node.name)

            def visit_ClassDef(self, node):
                defined_vars.add(node.name)

        for stmt in statements:
            VariableAnalyzer().visit(stmt)
        return used_vars, defined_vars

    def _format_code(self, code: str, formatter: str = "black") -> str:
        if formatter == "black":
            try:
                return black.format_str(code, mode=FileMode())
            except black.NothingChanged:
                return code
            except Exception as e:
                raise ValueError(f"Black formatting failed: {e}") from e
        elif formatter == "ruff":
            try:
                result = subprocess.run(
                    ["ruff", "format", "--stdin-filename", "temp.py"],
                    input=code,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Ruff formatting failed: {e.stderr}") from e
            except FileNotFoundError as e:
                raise ValueError("Ruff not found. Please install ruff.") from e
        else:
            raise ValueError(f"Unknown formatter: {formatter}")

    async def _apply_modifications_in_single_pass(
        self,
        ast_tree: ast.AST,
        modifications: list[ModificationSpec],
    ) -> ast.AST:
        """Applies all modifications in a single pass using a NodeTransformer."""

        modifications_by_node_id: dict[int, list[ModificationSpec]] = {}
        # Pre-process modifications to group them by target node
        for mod in modifications:
            target_nodes_result = await self.finder.execute(
                code_string=ast.unparse(ast_tree), query=mod.target_query
            )
            if target_nodes_result.success:
                target_nodes = target_nodes_result.metadata.get("nodes", [])
                for node_info in target_nodes:
                    if node_info.node_id not in modifications_by_node_id:
                        modifications_by_node_id[node_info.node_id] = []
                    modifications_by_node_id[node_info.node_id].append(mod)

        class SinglePassModifier(ast.NodeTransformer):
            def __init__(
                self, modifier_instance: "ASTModifier", modifications_by_node_id: dict
            ):
                self.modifier_instance = modifier_instance
                self.modifications_by_node_id = modifications_by_node_id

            def visit(self, node: ast.AST) -> Any:
                node_id = self.modifier_instance.finder._create_node_info(
                    node, ast_tree
                ).node_id
                if node_id in self.modifications_by_node_id:
                    original_node = copy.deepcopy(node)
                    new_node = original_node
                    for mod in self.modifications_by_node_id[node_id]:
                        new_node = self.modifier_instance._apply_modification_to_node(
                            new_node, mod, ast_tree
                        )
                    return self.generic_visit(new_node)
                return self.generic_visit(node)

        transformer = SinglePassModifier(self, modifications_by_node_id)
        modified_tree = transformer.visit(ast_tree)
        self._establish_parent_references(modified_tree)
        return modified_tree

    def _create_backup(self, ast_tree: ast.AST) -> str:
        backup_id = hashlib.md5(str(id(ast_tree)).encode()).hexdigest()[:12]
        self.backup_trees[backup_id] = copy.deepcopy(ast_tree)
        return backup_id

    def _restore_backup(self, backup_id: str) -> ast.AST:
        if backup_id in self.backup_trees:
            return copy.deepcopy(self.backup_trees[backup_id])
        else:
            logger.error(f"Backup {backup_id} not found")
            raise ValueError(f"Cannot restore backup {backup_id}")

    def _handle_replace_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.new_code:
            raise ValueError("new_code is required for replace operation")
        new_node = self._parse_code_snippet(modification.new_code, type(node_info.node))
        self._perform_node_replacement(node_info, new_node)

    def _handle_insert_before_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.new_code:
            raise ValueError("new_code is required for insert operation")
        new_node = self._parse_code_snippet(modification.new_code)
        self._perform_node_insertion(node_info, new_node, before=True)

    def _handle_insert_after_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.new_code:
            raise ValueError("new_code is required for insert operation")
        new_node = self._parse_code_snippet(modification.new_code)
        self._perform_node_insertion(node_info, new_node, before=False)

    def _handle_delete_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        self._perform_node_deletion(node_info)

    def _handle_modify_attribute_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.attribute:
            raise ValueError("attribute is required for modify_attribute operation")
        setattr(node_info.node, modification.attribute, modification.value)

    def _handle_wrap_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.wrapper_template:
            raise ValueError("wrapper_template is required for wrap operation")
        template_code = modification.wrapper_template.replace("{node}", "pass")
        wrapper_node = self._parse_code_snippet(template_code)
        self._replace_placeholder_in_wrapper(wrapper_node, node_info.node)
        self._perform_node_replacement(node_info, wrapper_node)

    def _handle_extract_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        self._extract_function(ast_tree, node_info, modification)

    def _handle_refactor_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        self._refactor_node(ast_tree, node_info, modification)

    def _handle_rename_symbol_scoped_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        self._rename_symbol_scoped(ast_tree, node_info, modification)
        return node_info.node

    def _handle_add_import_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.new_code:
            raise ValueError("new_code is required for add_import operation")
        new_node = self._parse_code_snippet(modification.new_code, ast.Import)
        if hasattr(ast_tree, "body"):
            ast_tree.body.insert(0, new_node)  # type: ignore

    def _handle_add_to_class_bases_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> None:
        if not modification.new_code:
            raise ValueError("new_code is required for add_to_class_bases operation")
        new_node = self._parse_code_snippet(modification.new_code, ast.Name)
        if hasattr(node_info.node, "bases"):
            node_info.node.bases.append(new_node)  # type: ignore

    def _handle_remove_class_methods_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        if not isinstance(node_info.node, ast.ClassDef):
            raise InvalidModificationTarget(
                operation="remove_class_methods",
                target_node_type=type(node_info.node).__name__,
                reason="Target node must be a ClassDef.",
            )

        if not modification.value or not isinstance(modification.value, list):
            raise ValueError("value must be a list of method names to remove.")

        method_names_to_remove = set(modification.value)
        new_body = []
        for item in node_info.node.body:
            if not (
                isinstance(item, ast.FunctionDef)
                and item.name in method_names_to_remove
            ):
                new_body.append(item)
        node_info.node.body = new_body
        return node_info.node

    def _handle_update_method_signature_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        if not isinstance(node_info.node, ast.FunctionDef):
            raise InvalidModificationTarget(
                operation="update_method_signature",
                target_node_type=type(node_info.node).__name__,
                reason="Target node must be a FunctionDef.",
            )

        if not modification.new_code:
            raise ValueError(
                "new_code must be a string representing the new signature."
            )

        try:
            # To parse arguments, we need to wrap it in a function definition
            temp_func_def = f"def temp_func({modification.new_code}): pass"
            parsed_func = ast.parse(temp_func_def).body[0]
            if isinstance(parsed_func, ast.FunctionDef):
                node_info.node.args = parsed_func.args
            else:
                raise ValueError("Failed to parse signature string.")
        except SyntaxError as e:
            raise SyntaxErrorInNewCode(
                new_code=modification.new_code, original_error=e
            ) from e
        return node_info.node

    def _handle_insert_statement_into_function_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        if not isinstance(node_info.node, ast.FunctionDef):
            raise InvalidModificationTarget(
                operation="insert_statement_into_function",
                target_node_type=type(node_info.node).__name__,
                reason="Target node must be a FunctionDef.",
            )

        if not modification.new_code:
            raise ValueError(
                "new_code must be a string representing the new statement."
            )

        new_statement = self._parse_code_snippet(modification.new_code)

        insert_at_end = (modification.metadata or {}).get("insert_at_end", True)

        if insert_at_end:
            node_info.node.body.append(new_statement)  # type: ignore
        else:
            node_info.node.body.insert(0, new_statement)  # type: ignore
        return node_info.node

    def _handle_replace_expression_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        if not isinstance(node_info.node, ast.expr):
            raise InvalidModificationTarget(
                operation="replace_expression",
                target_node_type=type(node_info.node).__name__,
                reason="Target node must be an expression.",
            )

        if not modification.new_code:
            raise ValueError(
                "new_code must be a string representing the new expression."
            )

        new_expression = self._parse_code_snippet(modification.new_code, ast.expr)

        return new_expression

    def _handle_add_class_operation(
        self, ast_tree: ast.AST, node_info: ASTNodeInfo, modification: ModificationSpec
    ) -> ast.AST:
        if not modification.new_code:
            raise ValueError(
                "new_code is required for add_class operation (the class definition)."
            )

        new_class_node = self._parse_code_snippet(modification.new_code, ast.ClassDef)

        # Find a suitable insertion point: after last import or last class definition
        insert_index = 0
        if hasattr(ast_tree, "body"):
            for i, node in enumerate(ast_tree.body):  # type: ignore
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                    insert_index = i + 1
                else:
                    # Stop if we encounter a non-import/non-class statement
                    break

            ast_tree.body.insert(insert_index, new_class_node)  # type: ignore
        return ast_tree

    def _apply_modification_to_node(
        self, node: ast.AST, modification: ModificationSpec, ast_tree: ast.AST
    ) -> ast.AST:
        """Applies a single modification to a single node."""
        node_info = self.finder._create_node_info(node, ast_tree)
        operation_handlers = {
            ModificationOperation.REPLACE: self._handle_replace_operation,
            ModificationOperation.INSERT_BEFORE: self._handle_insert_before_operation,
            ModificationOperation.INSERT_AFTER: self._handle_insert_after_operation,
            ModificationOperation.DELETE: self._handle_delete_operation,
            ModificationOperation.MODIFY_ATTRIBUTE: self._handle_modify_attribute_operation,
            ModificationOperation.WRAP: self._handle_wrap_operation,
            ModificationOperation.EXTRACT: self._handle_extract_operation,
            ModificationOperation.REFACTOR: self._handle_refactor_operation,
            ModificationOperation.RENAME_SYMBOL_SCOPED: self._handle_rename_symbol_scoped_operation,
            ModificationOperation.ADD_IMPORT: self._handle_add_import_operation,
            ModificationOperation.ADD_TO_CLASS_BASES: self._handle_add_to_class_bases_operation,
            ModificationOperation.REMOVE_CLASS_METHODS: self._handle_remove_class_methods_operation,
            ModificationOperation.UPDATE_METHOD_SIGNATURE: self._handle_update_method_signature_operation,
            ModificationOperation.INSERT_STATEMENT_INTO_FUNCTION: self._handle_insert_statement_into_function_operation,
            ModificationOperation.REPLACE_EXPRESSION: self._handle_replace_expression_operation,
            ModificationOperation.ADD_CLASS: self._handle_add_class_operation,
        }

        handler = operation_handlers.get(modification.operation)
        if not handler:
            raise ValueError(
                f"Unsupported modification operation: {modification.operation}"
            )

        result = handler(ast_tree, node_info, modification)
        return result if result is not None else ast_tree

    async def apply_modifications(
        self,
        ast_tree: ast.AST,
        modifications: list[ModificationSpec],
    ) -> ast.AST:
        backup_id = self._create_backup(ast_tree)
        try:
            modified_tree = await self._apply_modifications_in_single_pass(
                ast_tree, modifications
            )
            self._record_modifications(modifications)
            return modified_tree
        except Exception as e:
            logger.error(f"Critical error during AST modification: {e}")
            return self._restore_backup(backup_id)

    async def demo(self):
        """Demonstrate ASTModifier functionality."""
        print("🔧 AST MODIFIER DEMO")
        print("=" * 35)

        # Simple demo showing basic modification
        demo_code = """
def old_function_name():
    print("Hello from old function")
    return "old result"
"""

        print("Original code:")
        print(demo_code)
        print("\n--- Modifying function name ---")

        # Create a simple modification spec to rename the function
        from crisalida_lib.ASTRAL_TOOLS.ast_tools.models import (
            ModificationOperation,
            ModificationSpec,
        )

        modification = ModificationSpec(
            operation=ModificationOperation.REPLACE,
            target_query={"type": "FunctionDef", "name": "old_function_name"},
            new_code="new_function_name",
        )

        result = await self.execute(code_string=demo_code, modifications=[modification])

        if result.success:
            print("✅ Modification successful!")
            print("Modified code preview available in result")
        else:
            print("❌ Modification failed:", result.error_message)

        return ToolCallResult(
            command="ast_modifier_demo",
            success=True,
            output="ASTModifier demo completed",
            execution_time=0.1,
            error_message=None,
        )
