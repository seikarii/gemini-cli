import ast
import hashlib
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crisalida_lib.ASTRAL_TOOLS.ast_tools.models import (
    ASTNodeInfo,
    ComplexQuery,
    QueryOperator,
)
from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class EnhancedXPathParser:
    """Enhanced XPath parser with robust error handling and extended predicate support."""

    def __init__(self):
        self.axis_patterns = {
            "descendant-or-self": r"//(\w+)",
            "child": r"/(\w+)",
            "descendant": r"//(\w+)",
            "parent": r"\.\./(\w+)",
            "self": r"\./?(\w*)",
            "attribute": r"@(\w+)",
            "position": r"\[(\d+)\]",
            "predicate": r"\[([^\]]+)\]",
        }

    def parse(self, xpath: str) -> dict[str, Any]:
        """Parse XPath expressions with support for logical operators and extended predicates."""
        query_parts: dict[str, Any] = {
            "axis": "descendant-or-self" if xpath.startswith("//") else "child",
            "node_test": None,
            "predicates": [],
            "attributes": [],
            "conditions": [],
            "position": None,
            "functions": [],
        }

        # Axis detection
        if xpath.startswith("//"):
            xpath = xpath[2:]
            query_parts["axis"] = "descendant-or-self"
        elif xpath.startswith("../"):
            xpath = xpath[3:]
            query_parts["axis"] = "parent"
        elif xpath.startswith("./"):
            xpath = xpath[2:]
            query_parts["axis"] = "self"
        elif xpath.startswith("/"):
            xpath = xpath[1:]
            query_parts["axis"] = "child"

        # Split by predicates
        parts = re.split(r"(\[[^]]+\])", xpath)
        if parts and parts[0]:
            query_parts["node_test"] = parts[0]

        for part in parts[1:]:
            if part.startswith("[") and part.endswith("]"):
                predicate = part[1:-1]
                self._parse_predicate(predicate, query_parts)

        return query_parts

    def _parse_predicate(self, predicate: str, query_parts: dict[str, Any]):
        """Parse predicates with logical operator support."""
        if " and " in predicate:
            for sub_predicate in predicate.split(" and "):
                self._parse_single_predicate(sub_predicate.strip(), query_parts)
            return
        elif " or " in predicate:
            or_conditions = []
            for sub_predicate in predicate.split(" or "):
                temp_query: dict[str, Any] = {"conditions": []}
                self._parse_single_predicate(sub_predicate.strip(), temp_query)
                or_conditions.extend(temp_query["conditions"])
            for condition in or_conditions:
                condition["logical_op"] = "or"
            query_parts["conditions"].extend(or_conditions)
            return
        self._parse_single_predicate(predicate, query_parts)

    def _parse_single_predicate(self, predicate: str, query_parts: dict[str, Any]):
        """Parse single predicate with extended attribute and function support."""
        position_match = re.match(r"^(\d+)$", predicate)
        if position_match:
            query_parts["position"] = int(position_match.group(1))
            return

        attr_match = re.match(r"""@(\w+)\s*=\s*(['"])([^'"]+)\2""", predicate)
        if attr_match:
            attr_name, _, attr_value = attr_match.groups()
            query_parts["conditions"].append(
                {
                    "type": "attribute",
                    "field": attr_name,
                    "operator": "eq",
                    "value": attr_value,
                    "logical_op": "and",
                }
            )
            return

        attr_neq_match = re.match(r"""@(\w+)\s*!=\s*(['"])([^'"]+)\2""", predicate)
        if attr_neq_match:
            attr_name, _, attr_value = attr_neq_match.groups()
            query_parts["conditions"].append(
                {
                    "type": "attribute",
                    "field": attr_name,
                    "operator": "ne",
                    "value": attr_value,
                    "logical_op": "and",
                }
            )
            return

        contains_match = re.match(
            r"""contains\(\s*@(\w+),\s*(['"])([^'"]+)\2\s*\)""", predicate
        )
        if contains_match:
            attr_name, _, value = contains_match.groups()
            query_parts["conditions"].append(
                {
                    "type": "attribute",
                    "field": attr_name,
                    "operator": "contains",
                    "value": value,
                    "logical_op": "and",
                }
            )
            return

        starts_match = re.match(
            r"""starts-with\(\s*@(\w+),\s*(['"])([^'"]+)\2\s*\)""", predicate
        )
        if starts_match:
            attr_name, _, value = starts_match.groups()
            query_parts["conditions"].append(
                {
                    "type": "attribute",
                    "field": attr_name,
                    "operator": "starts_with",
                    "value": value,
                    "logical_op": "and",
                }
            )
            return

        attr_exists_match = re.match(r"^@(\w+)$", predicate)
        if attr_exists_match:
            attr_name = attr_exists_match.group(1)
            query_parts["conditions"].append(
                {
                    "type": "attribute",
                    "field": attr_name,
                    "operator": "exists",
                    "value": None,
                    "logical_op": "and",
                }
            )
            return

        logger.warning(f"Could not parse predicate: {predicate}")
        query_parts["conditions"].append(
            {"type": "unknown", "raw": predicate, "logical_op": "and"}
        )


class ASTFinderParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: str | None = Field(
        None, description="Path to the Python file to parse into an AST."
    )
    code_string: str | None = Field(
        None, description="Python code string to parse into an AST."
    )
    query: dict | str | ComplexQuery = Field(
        ...,
        description="The query to find nodes (dict, xpath string, or ComplexQuery object).",
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


class ASTFinder(BaseTool):
    """
    ASTFinder: Herramienta avanzada para encontrar nodos en el AST usando queries dict, XPath o ComplexQuery.
    Incluye cache, manejo de errores, y soporte extendido para queries narrativos y sintácticos.
    """

    def __init__(self):
        super().__init__()
        self.xpath_parser = EnhancedXPathParser()
        self.node_cache: dict[str, list[ASTNodeInfo]] = {}

    def _get_name(self) -> str:
        return "ast_finder"

    def _get_description(self) -> str:
        return "Finds nodes in the AST based on a query (dict, xpath string, or ComplexQuery object)."

    def _get_pydantic_schema(self):
        return ASTFinderParams

    def _get_category(self) -> str:
        return "ast_analysis"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = ASTFinderParams(**kwargs)
        query = params.query
        start_time = datetime.now()

        code_content = None
        if params.file_path:
            try:
                with open(params.file_path, encoding="utf-8") as f:
                    code_content = f.read()
            except FileNotFoundError:
                return ToolCallResult(
                    command="ast_finder",
                    success=False,
                    output="",
                    error_message=f"File not found: {params.file_path}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            except Exception as e:
                return ToolCallResult(
                    command="ast_finder",
                    success=False,
                    output="",
                    error_message=f"Error reading file {params.file_path}: {e}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        elif params.code_string:
            code_content = params.code_string

        if not code_content:
            return ToolCallResult(
                command="ast_finder",
                success=False,
                output="",
                error_message="No code content provided for AST parsing.",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        try:
            parsed_ast_tree = ast.parse(code_content)
            self._establish_parent_references(parsed_ast_tree)
        except SyntaxError as e:
            return ToolCallResult(
                command="ast_finder",
                success=False,
                output="",
                error_message=f"Syntax error in code: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return ToolCallResult(
                command="ast_finder",
                success=False,
                output="",
                error_message=f"Error parsing AST: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        try:
            cache_key = self._generate_cache_key(query)
            if cache_key in self.node_cache:
                cached_nodes = self.node_cache[cache_key]
                return ToolCallResult(
                    command="ast_finder",
                    success=True,
                    output=f"Found {len(cached_nodes)} nodes from cache.",
                    error_message=None,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"nodes": cached_nodes},
                )
            found_nodes = []
            if isinstance(query, str):
                found_nodes = self._find_by_xpath(parsed_ast_tree, query)
            elif isinstance(query, ComplexQuery):
                found_nodes = self._find_by_complex_query(parsed_ast_tree, query)
            elif isinstance(query, dict):
                if "xpath" in query:
                    found_nodes = self._find_by_xpath(parsed_ast_tree, query["xpath"])
                elif "custom" in query:
                    found_nodes = self._find_by_custom_function(
                        parsed_ast_tree, query["custom"]
                    )
                else:
                    found_nodes = self._find_by_dict_query(parsed_ast_tree, query)
            else:
                raise ValueError(f"Unsupported query type: {type(query)}")
            self.node_cache[cache_key] = found_nodes
            return ToolCallResult(
                command="ast_finder",
                success=True,
                output=f"Found {len(found_nodes)} nodes.",
                error_message=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"nodes": found_nodes},
            )
        except Exception as e:
            logger.error(f"Error in ASTFinder.execute: {e}")
            return ToolCallResult(
                command="ast_finder",
                success=False,
                output="",
                error_message=f"Error finding AST nodes: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _establish_parent_references(self, ast_tree: ast.AST):
        """Establece referencias parent para todos los nodos del AST."""
        for node in ast.walk(ast_tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore

    def _generate_cache_key(self, query: dict | str | ComplexQuery) -> str:
        """Genera una clave de cache para el query."""
        if isinstance(query, str):
            return f"xpath:{query}"
        elif isinstance(query, dict):
            return f"dict:{hash(str(sorted(query.items())))}"
        elif isinstance(query, ComplexQuery):
            query_str = f"{query.logical_operator.value}:"
            for condition in query.conditions:
                query_str += (
                    f"{condition.field}{condition.operator.value}{condition.value};"
                )
            return f"complex:{hash(query_str)}"

    def _find_by_dict_query(self, ast_tree: ast.AST, query: dict) -> list[ASTNodeInfo]:
        """Encuentra nodos usando queries dict iterativos."""
        found_nodes = []
        nodes_to_visit = [ast_tree]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if self._matches_dict_query(node, query):
                node_info = self._create_node_info(node, ast_tree)
                found_nodes.append(node_info)
            nodes_to_visit.extend(ast.iter_child_nodes(node))
        return found_nodes

    def _matches_dict_query(self, node: ast.AST, query: dict) -> bool:
        """Matching avanzado con soporte para parent y queries anidados."""
        query = query.copy()
        parent_query = query.pop("parent", None)
        if not self._matches_single_level_query(node, query):
            return False
        if parent_query:
            return self._has_ancestor_matching_query(node, parent_query)
        return True

    def _has_ancestor_matching_query(self, node: ast.AST, query: dict) -> bool:
        current = getattr(node, "parent", None)
        while current:
            if self._matches_dict_query(current, query):
                return True
            current = getattr(current, "parent", None)
        return False

    def _matches_single_level_query(self, node: ast.AST, query: dict) -> bool:
        if "type" in query and not self._check_type_match(node, query["type"]):
            return False
        if "name" in query and not self._check_name_match(node, query["name"]):
            return False
        if "func_name" in query and not self._check_func_name_match(
            node, query["func_name"]
        ):
            return False
        if "decorator_names" in query and not self._check_decorator_names_match(
            node, query["decorator_names"]
        ):
            return False
        if "attributes" in query and not self._check_attributes_match(
            node, query["attributes"]
        ):
            return False
        if "has_child" in query and not self._check_has_child_match(
            node, query["has_child"]
        ):
            return False
        if "depth" in query and not self._check_depth_match(node, query["depth"]):
            return False
        return True

    def _check_type_match(self, node: ast.AST, expected_type: str | list[str]) -> bool:
        node_type_name = node.__class__.__name__
        if isinstance(expected_type, list):
            return node_type_name in expected_type
        else:
            return node_type_name == expected_type

    def _check_name_match(self, node: ast.AST, expected_name: str) -> bool:
        node_name = getattr(node, "name", None)
        return self._compare_values(node_name, expected_name, QueryOperator.EQ)

    def _check_func_name_match(self, node: ast.AST, expected_func_name: str) -> bool:
        func_name = self._extract_function_name(node)
        return self._compare_values(func_name, expected_func_name, QueryOperator.EQ)

    def _check_decorator_names_match(
        self, node: ast.AST, expected_decorators: str | list[str]
    ) -> bool:
        decorators = self._extract_decorator_names(node)
        if isinstance(expected_decorators, list):
            return any(dec in decorators for dec in expected_decorators)
        else:
            return expected_decorators in decorators

    def _check_attributes_match(
        self, node: ast.AST, attributes_query: list[dict[str, Any]]
    ) -> bool:
        for attr_spec in attributes_query:
            if isinstance(attr_spec, dict) and "field" in attr_spec:
                field = attr_spec["field"]
                operator = QueryOperator(attr_spec.get("operator", "eq"))
                value = attr_spec["value"]
                actual_value = getattr(node, field, None)
                if not self._compare_values(actual_value, value, operator):
                    return False
            else:
                for attr_name, expected_value in attr_spec.items():
                    actual_value = getattr(node, attr_name, None)
                    if not self._compare_values(
                        actual_value, expected_value, QueryOperator.EQ
                    ):
                        return False
        return True

    def _check_has_child_match(self, node: ast.AST, child_query: dict) -> bool:
        return self._has_matching_child(node, child_query)

    def _check_depth_match(
        self, node: ast.AST, depth_condition: int | dict[str, Any]
    ) -> bool:
        node_depth = self._calculate_node_depth(node)
        if isinstance(depth_condition, dict):
            operator = QueryOperator(depth_condition.get("operator", "eq"))
            value = depth_condition["value"]
            return self._compare_values(node_depth, value, operator)
        else:
            return node_depth == depth_condition

    def _extract_function_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
            elif isinstance(node.func, ast.Subscript):
                return self._extract_function_name(node.func)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
        return None

    def _extract_decorator_names(self, node: ast.AST) -> list[str]:
        decorators = getattr(node, "decorator_list", [])
        decorator_names = []
        for dec in decorators:
            if isinstance(dec, ast.Name):
                decorator_names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorator_names.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorator_names.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorator_names.append(dec.func.attr)
        return decorator_names

    def _has_matching_child(self, node: ast.AST, child_query: dict) -> bool:
        for child in ast.iter_child_nodes(node):
            if self._matches_dict_query(child, child_query):
                return True
        return False

    def _calculate_node_depth(self, node: ast.AST) -> int:
        depth = 0
        current = node
        while hasattr(current, "parent") and current.parent:
            depth += 1
            current = current.parent
        return depth

    def _compare_values(
        self,
        actual: Any,
        expected: Any,
        operator: QueryOperator,
    ) -> bool:
        try:
            if operator == QueryOperator.EQ:
                return actual == expected
            elif operator == QueryOperator.NE:
                return actual != expected
            elif operator == QueryOperator.IN:
                return actual in expected if expected else False
            elif operator == QueryOperator.CONTAINS:
                return str(expected) in str(actual) if actual else False
            elif operator == QueryOperator.STARTS_WITH:
                return str(actual).startswith(str(expected)) if actual else False
            elif operator == QueryOperator.ENDS_WITH:
                return str(actual).endswith(str(expected)) if actual else False
            elif operator == QueryOperator.REGEX_MATCH:
                return bool(re.search(str(expected), str(actual))) if actual else False
            elif operator == QueryOperator.GT:
                return actual > expected if actual is not None else False
            elif operator == QueryOperator.LT:
                return actual < expected if actual is not None else False
            elif operator == QueryOperator.EXISTS:
                return actual is not None
            else:
                logger.warning(f"Unsupported operator: {operator}")
                return False
        except Exception as e:
            logger.error(f"Error comparing values: {e}")
            return False

    def _find_by_xpath(self, ast_tree: ast.AST, xpath: str) -> list[ASTNodeInfo]:
        """Encuentra nodos usando XPath extendido."""
        try:
            query_parts = self.xpath_parser.parse(xpath)
            found_nodes = []
            node_type = query_parts.get("node_test")
            if not node_type:
                return []
            candidate_nodes = []
            if query_parts["axis"] == "descendant-or-self":
                for node in ast.walk(ast_tree):
                    if node.__class__.__name__ == node_type:
                        candidate_nodes.append(node)
            else:
                for child in ast.iter_child_nodes(ast_tree):
                    if child.__class__.__name__ == node_type:
                        candidate_nodes.append(child)
            for node in candidate_nodes:
                if self._matches_xpath_conditions(node, query_parts["conditions"]):
                    node_info = self._create_node_info(node, ast_tree)
                    found_nodes.append(node_info)
            return found_nodes
        except Exception as e:
            logger.error(f"Error parsing XPath '{xpath}': {e}")
            raise ValueError(f"Error parsing XPath '{xpath}': {e}") from e

    def _matches_xpath_conditions(
        self,
        node: ast.AST,
        conditions: list[dict[str, Any]],
    ) -> bool:
        if not conditions:
            return True
        and_conditions = []
        or_conditions = []
        for condition in conditions:
            logical_op = condition.get("logical_op", "and")
            if logical_op == "or":
                or_conditions.append(condition)
            else:
                and_conditions.append(condition)
        and_result = all(
            self._evaluate_single_condition(node, cond) for cond in and_conditions
        )
        or_result = True
        if or_conditions:
            or_result = any(
                self._evaluate_single_condition(node, cond) for cond in or_conditions
            )
        return and_result and or_result

    def _evaluate_single_condition(
        self, node: ast.AST, condition: dict[str, Any]
    ) -> bool:
        if condition["type"] == "attribute":
            field = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            actual_value = getattr(node, field, None)
            try:
                query_op = QueryOperator(operator)
                return self._compare_values(actual_value, expected_value, query_op)
            except ValueError:
                logger.warning(f"Unknown operator: {operator}")
                return False
        elif condition["type"] == "position":
            logger.warning("Position-based XPath predicates not fully implemented")
            return True
        elif condition["type"] == "unknown":
            logger.warning(f"Unknown condition type: {condition.get('raw', 'N/A')}")
            return True
        return False

    def _find_by_complex_query(
        self,
        ast_tree: ast.AST,
        query: ComplexQuery,
    ) -> list[ASTNodeInfo]:
        found_nodes = []
        for node in ast.walk(ast_tree):
            if self._matches_complex_query(node, query):
                node_info = self._create_node_info(node, ast_tree)
                found_nodes.append(node_info)
        return found_nodes

    def _matches_complex_query(self, node: ast.AST, query: ComplexQuery) -> bool:
        condition_results = []
        for condition in query.conditions:
            actual_value = getattr(node, condition.field, None)
            result = self._compare_values(
                actual_value, condition.value, condition.operator
            )
            condition_results.append(result)
        if query.logical_operator == QueryOperator.AND:
            main_result = all(condition_results) if condition_results else True
        elif query.logical_operator == QueryOperator.OR:
            main_result = any(condition_results) if condition_results else False
        else:
            main_result = True
        child_results = []
        for child_query in query.child_queries:
            child_result = self._matches_complex_query(node, child_query)
            child_results.append(child_result)
        if child_results:
            if query.logical_operator == QueryOperator.AND:
                return main_result and all(child_results)
            elif query.logical_operator == QueryOperator.OR:
                return main_result or any(child_results)
        return main_result

    def _find_by_custom_function(
        self,
        ast_tree: ast.AST,
        custom_func: Callable,
    ) -> list[ASTNodeInfo]:
        found_nodes = []
        for node in ast.walk(ast_tree):
            try:
                if custom_func(node):
                    node_info = self._create_node_info(node, ast_tree)
                    found_nodes.append(node_info)
            except Exception as e:
                logger.warning(
                    f"Error in custom function for node {type(node).__name__}: {e}"
                )
        return found_nodes

    def _create_node_info(self, node: ast.AST, ast_tree: ast.AST) -> ASTNodeInfo:
        path = self._get_node_path(node, ast_tree)
        parent = getattr(node, "parent", None)
        field_name = None
        index = None
        depth = self._calculate_node_depth(node)
        if parent:
            field_name, index = self._find_node_location(parent, node)

        # Generate a more robust node ID
        node_repr = ast.dump(node)
        node_id = hashlib.md5(node_repr.encode()).hexdigest()[:12]

        return ASTNodeInfo(
            node=node,
            path=path,
            parent=parent,
            field_name=field_name,
            index=index,
            depth=depth,
            node_id=node_id,
        )

    def _find_node_location(
        self,
        parent: ast.AST,
        target_node: ast.AST,
    ) -> tuple[str | None, int | None]:
        for field_name_iter, value in ast.iter_fields(parent):
            if isinstance(value, list):
                try:
                    index = value.index(target_node)
                    return field_name_iter, index
                except ValueError:
                    continue
            elif value is target_node:
                return field_name_iter, None
        return None, None

    def _get_node_path(self, target_node: ast.AST, ast_tree: ast.AST) -> str:
        path_parts = []
        current = target_node
        while current and hasattr(current, "parent") and current.parent:
            parent = current.parent
            field_name, index = self._find_node_location(parent, current)
            if field_name:
                if index is not None:
                    path_parts.append(f"{field_name}[{index}]")
                else:
                    path_parts.append(field_name)
            current = parent
        path_parts.reverse()
        return "/" + "/".join(path_parts) if path_parts else "/root"

    async def demo(self):
        """Demonstrate ASTFinder functionality."""
        print("🔍 AST FINDER DEMO")
        print("=" * 30)

        # Simple demo code
        demo_code = '''
def hello_world(name="World"):
    """A simple greeting function."""
    return f"Hello, {name}!"

class Greeter:
    def __init__(self, default_name="World"):
        self.default_name = default_name

    def greet(self, name=None):
        return f"Hello, {name or self.default_name}!"
'''

        print("Demo code:")
        print(demo_code)
        print("\n--- Finding all function definitions ---")

        result = await self.execute(
            code_string=demo_code, query={"type": "FunctionDef"}
        )

        print(f"Found {len(result.metadata.get('nodes', []))} functions")
        for node in result.metadata.get("nodes", [])[:2]:  # Show first 2
            print(f"  - {node.node_type} at line {node.line}: {node.value}")

        return ToolCallResult(
            command="ast_finder_demo",
            success=True,
            output="ASTFinder demo completed successfully",
            execution_time=0.1,
            error_message=None,
        )
