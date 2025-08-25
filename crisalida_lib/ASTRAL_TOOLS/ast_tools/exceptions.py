class ASTToolException(Exception):
    """Base exception for AST tools."""

    pass


class NodeNotFound(ASTToolException):
    """Exception raised when a target AST node is not found."""

    def __init__(self, query: str | dict, file_path: str | None = None):
        self.query = query
        self.file_path = file_path
        message = f"AST node not found for query: {query}"
        if file_path:
            message += f" in file: {file_path}"
        super().__init__(message)


class InvalidModificationTarget(ASTToolException):
    """Exception raised when a modification cannot be applied to the target node."""

    def __init__(self, operation: str, target_node_type: str, reason: str):
        self.operation = operation
        self.target_node_type = target_node_type
        self.reason = reason
        message = (
            f"Invalid modification target for operation '{operation}'. "
            f"Target node type: {target_node_type}. Reason: {reason}"
        )
        super().__init__(message)


class SyntaxErrorInNewCode(ASTToolException):
    """Exception raised when the provided new_code has a syntax error."""

    def __init__(self, new_code: str, original_error: Exception):
        self.new_code = new_code
        self.original_error = original_error
        message = (
            f"Syntax error in provided new_code: '{new_code[:50]}...'. "
            f"Original error: {original_error}"
        )
        super().__init__(message)


class UnsupportedOperation(ASTToolException):
    """Exception raised when an unsupported modification operation is requested."""

    def __init__(self, operation: str):
        self.operation = operation
        message = f"Unsupported modification operation: {operation}"
        super().__init__(message)
