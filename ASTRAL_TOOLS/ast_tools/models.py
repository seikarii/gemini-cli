import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ASTNodeInfo:
    """
    Información enriquecida sobre un nodo AST, incluyendo path, parent, campo, índice y profundidad.
    Permite trazabilidad y manipulación avanzada en herramientas AST.
    """

    node: ast.AST
    path: str
    parent: ast.AST | None = None
    field_name: str | None = None
    index: int | None = None
    depth: int = 0
    node_id: str = field(default_factory=lambda: "")


class QueryOperator(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    IN = "in"
    GT = "gt"
    LT = "lt"
    EQ = "eq"
    NE = "ne"
    EXISTS = "exists"


@dataclass
class QueryCondition:
    """
    Representa una condición de búsqueda para queries AST.
    """

    field: str
    operator: QueryOperator
    value: Any


class ComplexQuery:
    """
    Query avanzada para búsqueda y manipulación de nodos AST.
    Permite condiciones lógicas, queries anidados y composición flexible.
    """

    def __init__(
        self,
        conditions: list[QueryCondition] | None = None,
        logical_operator: QueryOperator = QueryOperator.AND,
    ):
        self.conditions: list[QueryCondition] = conditions or []
        self.logical_operator: QueryOperator = logical_operator
        self.child_queries: list[ComplexQuery] = []

    def add_condition(
        self, field: str, operator: QueryOperator, value: Any
    ) -> "ComplexQuery":
        self.conditions.append(QueryCondition(field, operator, value))
        return self

    def add_child_query(
        self, query: "ComplexQuery", operator: QueryOperator = QueryOperator.AND
    ) -> "ComplexQuery":
        query.logical_operator = operator
        self.child_queries.append(query)
        return self


class ModificationOperation(Enum):
    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    DELETE = "delete"
    MODIFY_ATTRIBUTE = "modify_attribute"
    WRAP = "wrap"
    EXTRACT = "extract"
    REFACTOR = "refactor"
    RENAME_SYMBOL_SCOPED = "rename_symbol_scoped"
    EXTRACT_METHOD = "extract_method"
    SPLIT_BLOCK = "split_block"
    MERGE_BLOCKS = "merge_blocks"
    INLINE = "inline"
    MOVE = "move"
    ADD_IMPORT = "add_import"
    ADD_TO_CLASS_BASES = "add_to_class_bases"
    REMOVE_CLASS_METHODS = "remove_class_methods"
    UPDATE_METHOD_SIGNATURE = "update_method_signature"
    INSERT_STATEMENT_INTO_FUNCTION = "insert_statement_into_function"
    REPLACE_EXPRESSION = "replace_expression"
    ADD_CLASS = "add_class"


@dataclass
class ModificationSpec:
    """
    Especificación de una operación de modificación AST.
    Permite reemplazo, inserción, borrado, refactorización y validación avanzada.
    """

    operation: ModificationOperation
    target_query: dict | str | ComplexQuery
    new_code: str | None = None
    attribute: str | None = None
    value: Any = None
    wrapper_template: str | None = None
    extract_name: str | None = None
    validate_before: bool = True
    validate_after: bool = True
    metadata: dict[str, Any] | None = field(default_factory=dict)
