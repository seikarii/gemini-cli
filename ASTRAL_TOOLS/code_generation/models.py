from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TemplateVariableSchema(BaseModel):
    """Base schema for template variables"""
    pass


class FunctionTemplateVariables(TemplateVariableSchema):
    """Variables required for function_basic template"""
    function_name: str = Field(..., description="Name of the function")
    parameters: str = Field(default="", description="Function parameters (e.g., 'x: int, y: str')")
    return_type: str = Field(default="", description="Return type annotation (e.g., ' -> int')")
    description: str = Field(..., description="Function description for docstring")
    args_docs: str = Field(default="", description="Arguments documentation")
    return_docs: str = Field(default="", description="Return value documentation")
    function_body: str = Field(default="    pass", description="Function implementation body")


class ClassTemplateVariables(TemplateVariableSchema):
    """Variables required for class_basic template"""
    class_name: str = Field(..., description="Name of the class")
    base_classes: str = Field(default="", description="Base classes (e.g., 'BaseModel, ABC')")
    description: str = Field(..., description="Class description for docstring")
    class_body: str = Field(default="    pass", description="Class implementation body")


class AsyncFunctionTemplateVariables(TemplateVariableSchema):
    """Variables required for async_function template"""
    function_name: str = Field(..., description="Name of the async function")
    parameters: str = Field(default="", description="Function parameters")
    return_type: str = Field(default="", description="Return type annotation")
    description: str = Field(..., description="Function description")
    function_body: str = Field(default="    pass", description="Async function body")


class ApiEndpointTemplateVariables(TemplateVariableSchema):
    """Variables required for api_endpoint template"""
    endpoint_name: str = Field(..., description="Name of the endpoint function")
    path: str = Field(..., description="API path (e.g., '/users/{user_id}')")
    method: str = Field(default="get", description="HTTP method (get, post, put, delete)")
    description: str = Field(..., description="Endpoint description")
    request_model: str = Field(default="", description="Request model class name")
    response_model: str = Field(default="", description="Response model class name")
    endpoint_body: str = Field(default="    pass", description="Endpoint implementation")


class DataClassTemplateVariables(TemplateVariableSchema):
    """Variables required for data_class template"""
    class_name: str = Field(..., description="Name of the data class")
    description: str = Field(..., description="Data class description")
    fields: str = Field(..., description="Class fields definition")


class TestFunctionTemplateVariables(TemplateVariableSchema):
    """Variables required for test_function template"""
    test_name: str = Field(..., description="Name of the test function")
    description: str = Field(..., description="Test description")
    test_body: str = Field(default="    assert True", description="Test implementation")


# Template variable schema mapping
TEMPLATE_VARIABLE_SCHEMAS: Dict[str, type[TemplateVariableSchema]] = {
    "function_basic": FunctionTemplateVariables,
    "class_basic": ClassTemplateVariables,
    "async_function": AsyncFunctionTemplateVariables,
    "api_endpoint": ApiEndpointTemplateVariables,
    "data_class": DataClassTemplateVariables,
    "test_function": TestFunctionTemplateVariables,
}


class CodeGenerationParameters(BaseModel):
    action: Literal[
        "generate_code",
        "generate_project_structure",
        "analyze_context",
        "suggest_refactoring",
        "generate_documentation",
        "create_template",
        "list_templates",
    ] = Field(..., description="Action to perform")

    language: str = Field(
        "python",
        description="Programming language (python, javascript, typescript, etc.)",
    )
    context: Optional[str] = Field(
        None, description="Context description for semantic understanding"
    )
    template_name: Optional[str] = Field(None, description="Name of template to use")
    custom_template: Optional[str] = Field(None, description="Custom template string")
    variables: Dict[str, Any] = Field(
        default_factory=dict, description="Template variables"
    )
    project_name: Optional[str] = Field(
        None, description="Name of the project to generate"
    )
    project_type: Optional[str] = Field(
        None, description="Type of project (web, api, cli, library, etc.)"
    )
    features: List[str] = Field(
        default_factory=list, description="List of features to include"
    )
    code_snippet: Optional[str] = Field(None, description="Code snippet to analyze")
    file_path: Optional[str] = Field(None, description="Path to code file to analyze")
    
    # FIXED: Clarified output_path behavior per action
    output_path: Optional[str] = Field(
        None, 
        description="Path where to save generated files. Only used by generate_project_structure action. For generate_code, use write_file=True to enable file writing."
    )
    
    # NEW: Explicit file writing control for generate_code
    write_file: bool = Field(
        False, 
        description="If True, write generated code to file specified by output_path (generate_code action only)"
    )
    
    include_tests: bool = Field(True, description="Whether to include test files")
    include_docs: bool = Field(True, description="Whether to include documentation")
    semantic_analysis: bool = Field(
        True, description="Enable semantic context analysis"
    )
    optimization_level: int = Field(2, description="Code optimization level (1-5)")


def get_template_schema(template_name: str) -> Optional[type[TemplateVariableSchema]]:
    """Get the variable schema for a specific template"""
    return TEMPLATE_VARIABLE_SCHEMAS.get(template_name)


def validate_template_variables(template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate template variables against schema and return validated/enhanced variables.
    
    Args:
        template_name: Name of the template
        variables: Variables to validate
        
    Returns:
        Validated and potentially enhanced variables
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    schema_class = get_template_schema(template_name)
    if not schema_class:
        # No schema defined, return variables as-is
        return variables
    
    try:
        # Validate using pydantic
        validated = schema_class(**variables)
        return validated.model_dump()
    except Exception as e:
        # Create comprehensive error message
        if hasattr(e, 'errors'):
            missing_fields = []
            invalid_fields = []
            
            for error in e.errors():
                if error['type'] == 'missing':
                    missing_fields.append(error['loc'][0])
                else:
                    invalid_fields.append(f"{error['loc'][0]}: {error['msg']}")
            
            error_parts = []
            if missing_fields:
                error_parts.append(f"Missing required variables: {', '.join(missing_fields)}")
            if invalid_fields:
                error_parts.append(f"Invalid variables: {'; '.join(invalid_fields)}")
            
            # Add schema documentation
            schema_fields = schema_class.model_fields
            required_docs = []
            optional_docs = []
            
            for field_name, field_info in schema_fields.items():
                desc = field_info.description or "No description"
                default_val = getattr(field_info, 'default', None)
                
                if default_val is None or default_val == ...:  # Required field
                    required_docs.append(f"  {field_name}: {desc}")
                else:
                    optional_docs.append(f"  {field_name}: {desc} (default: {default_val})")
            
            documentation = f"\nTemplate '{template_name}' requires:\n"
            if required_docs:
                documentation += "Required variables:\n" + "\n".join(required_docs) + "\n"
            if optional_docs:
                documentation += "Optional variables:\n" + "\n".join(optional_docs)
            
            raise ValueError(f"{'. '.join(error_parts)}.{documentation}")
        else:
            raise ValueError(f"Template validation failed: {str(e)}")


def get_template_documentation(template_name: str) -> str:
    """Get comprehensive documentation for a template"""
    schema_class = get_template_schema(template_name)
    if not schema_class:
        return f"No schema documentation available for template '{template_name}'"
    
    schema_fields = schema_class.model_fields
    required_fields = []
    optional_fields = []
    
    for field_name, field_info in schema_fields.items():
        desc = field_info.description or "No description"
        default_val = getattr(field_info, 'default', None)
        
        if default_val is None or default_val == ...:  # Required field
            required_fields.append(f"  {field_name}: {desc}")
        else:
            optional_fields.append(f"  {field_name}: {desc} (default: {default_val})")
    
    docs = f"Template '{template_name}' documentation:\n\n"
    if required_fields:
        docs += "Required variables:\n" + "\n".join(required_fields) + "\n\n"
    if optional_fields:
        docs += "Optional variables:\n" + "\n".join(optional_fields) + "\n"
    
    return docs
