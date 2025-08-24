"""
Actions for the Code Generation Tool.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.code_generation.semantic import SemanticAnalyzer
from crisalida_lib.ASTRAL_TOOLS.code_generation.templates import (
    load_code_templates,
    load_project_templates,
)
# FIXED: Import new validation functions
from crisalida_lib.ASTRAL_TOOLS.code_generation.models import (
    validate_template_variables,
    get_template_documentation,
)

logger = logging.getLogger(__name__)


class CodeGenerationActions:
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.code_templates = load_code_templates()
        self.project_templates = load_project_templates()

    async def generate_code(self, **kwargs: Any) -> ToolCallResult:
        """Generate code from templates with semantic context"""
        start_time = datetime.now()
        language = kwargs.get("language", "python")
        template_name = kwargs.get("template_name")
        custom_template = kwargs.get("custom_template")
        variables = kwargs.get("variables", {})
        context = kwargs.get("context")
        semantic_analysis = kwargs.get("semantic_analysis", True)
        
        # FIXED: Clear semantics for file writing
        write_file = kwargs.get("write_file", False)
        output_path = kwargs.get("output_path")

        # Determine template to use
        if custom_template:
            template = custom_template
            actual_template_name = "custom"
        elif template_name:
            if language not in self.code_templates:
                return ToolCallResult(
                    command="generate_code",
                    success=False,
                    output="",
                    error_message=f"Language '{language}' not supported",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

            if template_name not in self.code_templates[language]:
                return ToolCallResult(
                    command="generate_code",
                    success=False,
                    output="",
                    error_message=f"Template '{template_name}' not found for {language}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )

            template = self.code_templates[language][template_name]
            actual_template_name = template_name
        else:
            # Auto-select template based on context
            if context and semantic_analysis:
                template_name = self.semantic_analyzer.select_template_by_context(
                    language, context, self.code_templates
                )
                if template_name:
                    template = self.code_templates[language][template_name]
                    actual_template_name = template_name
                else:
                    template = self.code_templates[language].get(
                        "function_basic", "# TODO: Implement"
                    )
                    actual_template_name = "function_basic"
            else:
                template = self.code_templates[language].get(
                    "function_basic", "# TODO: Implement"
                )
                actual_template_name = "function_basic"

        # FIXED: Template variable validation with comprehensive error messages
        if actual_template_name != "custom":
            try:
                # Enhance variables with semantic analysis first
                if context and semantic_analysis:
                    enhanced_variables = self.semantic_analyzer.enhance_variables_with_context(
                        variables, context, language
                    )
                    variables.update(enhanced_variables)
                
                # Validate template variables against schema
                variables = validate_template_variables(actual_template_name, variables)
                
            except ValueError as e:
                return ToolCallResult(
                    command="generate_code",
                    success=False,
                    output="",
                    error_message=str(e),
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        else:
            # For custom templates, still enhance with semantic analysis
            if context and semantic_analysis:
                enhanced_variables = self.semantic_analyzer.enhance_variables_with_context(
                    variables, context, language
                )
                variables.update(enhanced_variables)

        # Generate code
        try:
            generated_code = template.format(**variables)
        except KeyError as e:
            # This should be rare now due to validation, but provide helpful fallback
            missing_var = str(e).strip("'\"")
            error_msg = f"Missing template variable: {missing_var}"
            
            # Add template documentation if available
            if actual_template_name != "custom":
                doc = get_template_documentation(actual_template_name)
                error_msg += f"\n\n{doc}"
            
            return ToolCallResult(
                command="generate_code",
                success=False,
                output="",
                error_message=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return ToolCallResult(
                command="generate_code",
                success=False,
                output="",
                error_message=f"Failed to render template: {e}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        # FIXED: Implement file writing when requested
        file_written = False
        written_path = ""
        
        if write_file and output_path:
            try:
                output_file = Path(output_path)
                
                # Create parent directories if they don't exist
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the generated code to file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(generated_code)
                
                file_written = True
                written_path = str(output_file.absolute())
                logger.info(f"Generated code written to: {written_path}")
                
            except Exception as e:
                # File writing failed, but we still have the generated code
                logger.error(f"Failed to write file to {output_path}: {e}")
                return ToolCallResult(
                    command="generate_code",
                    success=False,
                    output="",
                    error_message=f"Code generation succeeded but file writing failed: {e}",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={
                        "generated_code": generated_code,
                        "language": language,
                        "template": actual_template_name,
                        "variables": variables,
                    },
                )

        # Format output
        output = "🔧 **CODE GENERATION COMPLETED**\n\n"
        output += f"**Language:** {language}\n"
        output += f"**Template:** {actual_template_name}\n"
        if context:
            output += f"**Context:** {context}\n"
        output += f"**Semantic Analysis:** {'enabled' if semantic_analysis else 'disabled'}\n"
        
        # FIXED: Clear indication of file writing status
        if write_file:
            if file_written:
                output += f"**File Written:** ✅ {written_path}\n"
            else:
                output += f"**File Writing:** ❌ Requested but output_path not provided\n"
        else:
            output += f"**File Writing:** ⏸️ Not requested (use write_file=True to enable)\n"
        
        output += f"\n**Generated Code:**\n```{language}\n{generated_code}\n```"

        return ToolCallResult(
            command="generate_code",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "language": language,
                "template": actual_template_name,
                "variables": variables,
                "generated_code": generated_code,
                "file_written": file_written,
                "written_path": written_path if file_written else None,
            },
        )

    async def list_templates(self, **kwargs: Any) -> ToolCallResult:
        """List available templates with documentation"""
        start_time = datetime.now()
        language = kwargs.get("language")
        show_documentation = kwargs.get("show_documentation", False)

        output = "📋 **AVAILABLE TEMPLATES**\n\n"

        if language:
            if language in self.code_templates:
                output += f"**{language.title()} Templates:**\n"
                for template_name in self.code_templates[language].keys():
                    output += f"• {template_name}\n"
                    
                    # FIXED: Add template documentation
                    if show_documentation:
                        doc = get_template_documentation(template_name)
                        # Indent documentation
                        indented_doc = "\n".join(f"  {line}" for line in doc.split("\n"))
                        output += f"{indented_doc}\n"
                        
            else:
                output += f"No templates found for {language}\n"
        else:
            for lang, templates in self.code_templates.items():
                output += f"**{lang.title()} Templates:**\n"
                for template_name in templates.keys():
                    output += f"• {template_name}\n"
                    
                    # FIXED: Add template documentation if requested
                    if show_documentation:
                        doc = get_template_documentation(template_name)
                        # Indent documentation
                        indented_doc = "\n".join(f"  {line}" for line in doc.split("\n"))
                        output += f"{indented_doc}\n"
                        
                output += "\n"

        output += "\n**Project Templates:**\n"
        for project_type, template in self.project_templates.items():
            output += f"• {project_type}: {template['description']}\n"

        # FIXED: Add usage instructions
        output += "\n**💡 Usage Tips:**\n"
        output += "• Use show_documentation=True to see variable requirements for each template\n"
        output += "• For generate_code action, use write_file=True and output_path to save to file\n"
        output += "• Template variables are validated automatically with helpful error messages\n"

        return ToolCallResult(
            command="list_templates",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "code_templates": self.code_templates,
                "project_templates": self.project_templates,
            },
        )

    async def generate_project_structure(self, **kwargs: Any) -> ToolCallResult:
        """Generate complete project structure"""
        start_time = datetime.now()
        project_name = kwargs.get("project_name")
        project_type = kwargs.get("project_type")
        features = kwargs.get("features", [])
        output_path = kwargs.get("output_path")

        if not project_name:
            return ToolCallResult(
                command="generate_project_structure",
                success=False,
                output="",
                error_message="project_name is required",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        # Auto-detect project type if not specified
        if not project_type:
            project_type = self._detect_project_type(features)

        if project_type not in self.project_templates:
            return ToolCallResult(
                command="generate_project_structure",
                success=False,
                output="",
                error_message=f"Project type '{project_type}' not supported",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        template = self.project_templates[project_type]
        structure = template["structure"]

        # Generate project structure
        created_files = []
        if output_path:
            # Actually create the files
            created_files = self._create_project_files(
                output_path, project_name, structure
            )

        # Format output
        output = "📁 **PROJECT STRUCTURE GENERATED**\n\n"
        output += f"**Project Name:** {project_name}\n"
        output += f"**Project Type:** {project_type}\n"
        output += f"**Description:** {template['description']}\n"
        if features:
            output += f"**Features:** {', '.join(features)}\n"
        if output_path:
            output += f"**Output Path:** {output_path}\n"
        output += "\n**Project Structure:**\n"
        output += self._format_structure_tree(structure)

        if created_files:
            output += f"\n**Created {len(created_files)} files:**\n"
            for file_path in created_files[:10]:  # Show first 10
                output += f"• {file_path}\n"
            if len(created_files) > 10:
                output += f"• ... and {len(created_files) - 10} more files\n"

        return ToolCallResult(
            command="generate_project_structure",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "project_name": project_name,
                "project_type": project_type,
                "created_files": created_files,
                "structure": structure,
            },
        )

    def _detect_project_type(self, features: list[str]) -> str:
        """Auto-detect project type from features"""
        features_lower = [f.lower() for f in features]

        if any(f in features_lower for f in ["api", "rest", "fastapi", "flask"]):
            return "python_api"
        elif any(f in features_lower for f in ["cli", "command", "terminal"]):
            return "python_cli"
        elif any(f in features_lower for f in ["react", "web", "frontend"]):
            return "react_app"
        else:
            return "python_api"  # Default

    def _create_project_files(
        self, base_path: str, project_name: str, structure: dict
    ) -> list[str]:
        """Create actual project files and directories"""
        created_files = []
        project_path = Path(base_path) / project_name

        def create_structure(current_path: Path, struct: dict):
            for name, content in struct.items():
                item_path = current_path / name

                if isinstance(content, dict):
                    # Directory
                    item_path.mkdir(parents=True, exist_ok=True)
                    create_structure(item_path, content)
                else:
                    # File
                    item_path.parent.mkdir(parents=True, exist_ok=True)

                    # Process template variables in content
                    if isinstance(content, str) and "{project_name}" in content:
                        content = content.format(project_name=project_name)

                    with open(item_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    created_files.append(str(item_path))

        try:
            create_structure(project_path, structure)
        except Exception as e:
            logger.error(f"Failed to create project files: {e}")

        return created_files

    def _format_structure_tree(self, structure: dict, indent: int = 0) -> str:
        """Format project structure as a tree"""
        tree = ""
        for name, content in structure.items():
            tree += "  " * indent + f"├── {name}\n"
            if isinstance(content, dict):
                tree += self._format_structure_tree(content, indent + 1)
        return tree

    async def analyze_context(self, **kwargs: Any) -> ToolCallResult:
        """Analyze code context for intelligent suggestions"""
        start_time = datetime.now()
        context = kwargs.get("context")
        code_snippet = kwargs.get("code_snippet")
        language = kwargs.get("language", "python")

        if not context and not code_snippet:
            return ToolCallResult(
                command="analyze_context",
                success=False,
                output="",
                error_message="Either context or code_snippet is required",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        analysis_text = context or code_snippet or ""

        # Perform semantic analysis
        analysis_result = {
            "language": language,
            "context": analysis_text,
            "suggested_patterns": [],
            "recommended_templates": [],
            "complexity_score": 0,
            "architecture_suggestions": [],
        }

        # Analyze patterns
        if self.semantic_analyzer.pattern_vectors is not None:
            try:
                text_vector = self.semantic_analyzer.vectorizer.transform(
                    [analysis_text]
                )
                similarities = cosine_similarity(
                    text_vector, self.semantic_analyzer.pattern_vectors
                )[0]

                # Get top 3 similar patterns
                top_indices = np.argsort(similarities)[-3:][::-1]
                for idx in top_indices:
                    if similarities[idx] > 0.1:
                        analysis_result["suggested_patterns"].append(
                            {
                                "pattern": self.semantic_analyzer.code_patterns[idx],
                                "similarity": float(similarities[idx]),
                                "relevance": (
                                    "high"
                                    if similarities[idx] > 0.5
                                    else "medium" if similarities[idx] > 0.3 else "low"
                                ),
                            }
                        )
            except Exception as e:
                logger.warning(f"Pattern analysis failed: {e}")
                # Fallback analysis without vectorization
                for _i, pattern in enumerate(self.semantic_analyzer.code_patterns):
                    if any(
                        word in analysis_text.lower()
                        for word in pattern.lower().split()[:3]
                    ):
                        analysis_result["suggested_patterns"].append(
                            {
                                "pattern": pattern,
                                "similarity": 0.5,
                                "relevance": "medium",
                            }
                        )
                        if len(analysis_result["suggested_patterns"]) >= 3:
                            break

        # Recommend templates
        if language in self.code_templates:
            for template_name in self.code_templates[language].keys():
                if any(
                    keyword in analysis_text.lower()
                    for keyword in self._get_template_keywords(template_name)
                ):
                    analysis_result["recommended_templates"].append(template_name)

        # Calculate complexity
        complexity = self._calculate_complexity(analysis_text)
        analysis_result["complexity_score"] = complexity

        # Architecture suggestions
        suggestions = self._generate_architecture_suggestions(analysis_text, language)
        analysis_result["architecture_suggestions"] = suggestions

        # Format output
        output = "🔍 **CODE CONTEXT ANALYSIS**\n\n"
        output += f"**Language:** {language}\n"
        output += f"**Complexity Score:** {complexity}/5\n\n"

        if analysis_result["suggested_patterns"]:
            output += "**🎯 Suggested Patterns:**\n"
            for pattern in analysis_result["suggested_patterns"]:
                output += f"• {pattern['pattern']} (relevance: {pattern['relevance']}, similarity: {pattern['similarity']:.3f})\n"
            output += "\n"

        if analysis_result["recommended_templates"]:
            output += "**📋 Recommended Templates:**\n"
            for template in analysis_result["recommended_templates"]:
                output += f"• {template}\n"
            output += "\n"

        if analysis_result["architecture_suggestions"]:
            output += "**🏗️ Architecture Suggestions:**\n"
            for suggestion in analysis_result["architecture_suggestions"]:
                output += f"• {suggestion}\n"

        return ToolCallResult(
            command="analyze_context",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata=analysis_result,
        )

    def _get_template_keywords(self, template_name: str) -> list[str]:
        """Get keywords associated with a template"""
        keyword_map = {
            "class_basic": ["class", "object", "model"],
            "function_basic": ["function", "method", "procedure"],
            "api_endpoint": ["api", "endpoint", "route", "rest"],
            "test_function": ["test", "testing", "unit test"],
            "async_function": ["async", "await", "asynchronous"],
            "data_class": ["data", "model", "structure"],
        }
        return keyword_map.get(template_name, [])

    def _calculate_complexity(self, text: str) -> int:
        """Calculate complexity score from 1-5"""
        complexity_indicators = {
            "simple": ["function", "variable", "if", "print"],
            "moderate": ["class", "loop", "array", "object"],
            "complex": ["async", "api", "database", "service"],
            "advanced": ["microservice", "distributed", "scalable", "optimization"],
            "expert": ["machine learning", "ai", "quantum", "blockchain"],
        }

        text_lower = text.lower()
        max_level = 1

        for level, keywords in enumerate(complexity_indicators.values(), 1):
            if any(keyword in text_lower for keyword in keywords):
                max_level = level

        return min(max_level, 5)

    def _generate_architecture_suggestions(self, text: str, language: str) -> list[str]:
        """Generate architecture suggestions based on context"""
        suggestions = []
        text_lower = text.lower()

        if "api" in text_lower:
            suggestions.append("Consider using RESTful API design principles")
            suggestions.append("Implement proper error handling and status codes")

        if "database" in text_lower:
            suggestions.append("Use ORM for database abstraction")
            suggestions.append("Implement database connection pooling")

        if "test" in text_lower:
            suggestions.append("Follow AAA pattern (Arrange, Act, Assert)")
            suggestions.append("Use mocking for external dependencies")

        if "async" in text_lower:
            suggestions.append("Implement proper async/await patterns")
            suggestions.append("Use connection pools for async operations")

        if language == "python":
            suggestions.append("Follow PEP 8 coding standards")
            suggestions.append("Use type hints for better code documentation")

        return suggestions[:5]  # Limit to top 5 suggestions

    async def suggest_refactoring(self, **kwargs: Any) -> ToolCallResult:
        """Suggest code refactoring improvements"""
        start_time = datetime.now()
        code_snippet = kwargs.get("code_snippet")
        language = kwargs.get("language", "python")

        if not code_snippet:
            return ToolCallResult(
                command="suggest_refactoring",
                success=False,
                output="",
                error_message="code_snippet is required",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        suggestions = []

        # Analyze code for common issues
        if language == "python":
            suggestions.extend(self._analyze_python_code(code_snippet))
        elif language == "javascript":
            suggestions.extend(self._analyze_javascript_code(code_snippet))

        # Generic suggestions
        suggestions.extend(self._analyze_generic_code(code_snippet))

        output = "🔧 **REFACTORING SUGGESTIONS**\n\n"
        output += f"**Language:** {language}\n"
        output += f"**Code Length:** {len(code_snippet)} characters\n\n"

        if suggestions:
            output += "**💡 Suggestions:**\n"
            for i, suggestion in enumerate(suggestions, 1):
                output += f"{i}. {suggestion}\n"
        else:
            output += "**✅ No major refactoring issues found. Code looks good!**\n"

        return ToolCallResult(
            command="suggest_refactoring",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"suggestions": suggestions},
        )

    def _analyze_python_code(self, code: str) -> list[str]:
        """Analyze Python code for refactoring opportunities"""
        suggestions = []

        # Check for long functions
        lines = code.split("\n")
        if len(lines) > 50:
            suggestions.append(
                "Consider breaking down long functions into smaller, focused functions"
            )

        # Check for missing docstrings
        if "def " in code and '""' not in code and "'''" not in code:
            suggestions.append("Add docstrings to functions for better documentation")

        # Check for bare except clauses
        if "except:" in code:
            suggestions.append("Avoid bare except clauses, specify exception types")

        # Check for missing type hints
        if "def " in code and "->" not in code:
            suggestions.append("Consider adding type hints for better code clarity")

        return suggestions

    def _analyze_javascript_code(self, code: str) -> list[str]:
        """Analyze JavaScript code for refactoring opportunities"""
        suggestions = []

        # Check for var usage
        if "var " in code:
            suggestions.append("Replace 'var' with 'let' or 'const' for better scoping")

        # Check for missing semicolons
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        for line in lines:
            if (
                line
                and not line.endswith((";", "{", "}", ")", "]"))
                and not line.startswith("//")
            ):
                suggestions.append("Consider adding semicolons for consistency")
                break

        return suggestions

    def _analyze_generic_code(self, code: str) -> list[str]:
        """Analyze code for generic refactoring opportunities"""
        suggestions = []

        # Check for code duplication
        lines = code.split("\n")
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "//")):
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        duplicates = [line for line, count in line_counts.items() if count > 2]
        if duplicates:
            suggestions.append(
                "Consider extracting duplicated code into reusable functions"
            )

        # Check for long lines
        long_lines = [line for line in lines if len(line) > 120]
        if long_lines:
            suggestions.append("Consider breaking long lines for better readability")

        return suggestions

    async def generate_documentation(self, **kwargs: Any) -> ToolCallResult:
        """Generate comprehensive documentation"""
        start_time = datetime.now()
        code_snippet = kwargs.get("code_snippet")
        language = kwargs.get("language", "python")
        context = kwargs.get("context")

        if not code_snippet:
            return ToolCallResult(
                command="generate_documentation",
                success=False,
                output="",
                error_message="code_snippet is required",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        # Generate documentation
        doc_parts = []

        # Overview
        if context:
            doc_parts.append(f"## Overview\n\n{context}\n")

        # Code analysis
        functions = self._extract_functions(code_snippet, language)
        classes = self._extract_classes(code_snippet, language)

        if functions:
            doc_parts.append("## Functions\n")
            for func in functions:
                doc_parts.append(f"### `{func['name']}`\n")
                doc_parts.append(f"{func.get('description', 'Function description')}\n")
                if func.get("parameters"):
                    doc_parts.append("**Parameters:**\n")
                    for param in func["parameters"]:
                        doc_parts.append(f"- `{param}`: Parameter description\n")
                doc_parts.append("")

        if classes:
            doc_parts.append("## Classes\n")
            for cls in classes:
                doc_parts.append(f"### `{cls['name']}`\n")
                doc_parts.append(f"{cls.get('description', 'Class description')}\n")
                doc_parts.append("")

        # Usage examples
        doc_parts.append(f"## Usage\n\n```{language}\n# Add usage examples here\n```\n")

        documentation = "\n".join(doc_parts)

        output = "📚 **GENERATED DOCUMENTATION**\n\n"
        output += f"**Language:** {language}\n"
        output += f"**Functions Found:** {len(functions)}\n"
        output += f"**Classes Found:** {len(classes)}\n\n"
        output += "**Documentation:**\n\n"
        output += documentation

        return ToolCallResult(
            command="generate_documentation",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "documentation": documentation,
                "functions": functions,
                "classes": classes,
            },
        )

    def _extract_functions(self, code: str, language: str) -> list[dict[str, Any]]:
        """Extract function definitions from code"""
        functions = []

        if language == "python":
            pattern = r"def\s+(\w+)\s*\(([^)]*)\):"
            matches = re.finditer(pattern, code)
            for match in matches:
                name = match.group(1)
                params = [p.strip() for p in match.group(2).split(",") if p.strip()]
                functions.append(
                    {
                        "name": name,
                        "parameters": params,
                        "description": f"Function {name}",
                    }
                )

        elif language == "javascript":
            pattern = r"function\s+(\w+)\s*\(([^)]*)\)"
            matches = re.finditer(pattern, code)
            for match in matches:
                name = match.group(1)
                params = [p.strip() for p in match.group(2).split(",") if p.strip()]
                functions.append(
                    {
                        "name": name,
                        "parameters": params,
                        "description": f"Function {name}",
                    }
                )

        return functions

    def _extract_classes(self, code: str, language: str) -> list[dict[str, Any]]:
        """Extract class definitions from code"""
        classes = []

        if language == "python":
            pattern = r"class\s+(\w+)(?:[^)]*)?:"
            matches = re.finditer(pattern, code)
            for match in matches:
                name = match.group(1)
                classes.append({"name": name, "description": f"Class {name}"})

        elif language == "javascript":
            pattern = r"class\s+(\w+)(?:\s+extends\s+\w+)?"
            matches = re.finditer(pattern, code)
            for match in matches:
                name = match.group(1)
                classes.append({"name": name, "description": f"Class {name}"})

        return classes

    async def create_template(self, **kwargs: Any) -> ToolCallResult:
        """Create a new code template"""
        start_time = datetime.now()
        language = kwargs.get("language", "python")
        template_name = kwargs.get("template_name")
        custom_template = kwargs.get("custom_template")

        if not template_name or not custom_template:
            return ToolCallResult(
                command="create_template",
                success=False,
                output="",
                error_message="template_name and custom_template are required",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        # Add template to collection
        if language not in self.code_templates:
            self.code_templates[language] = {}

        self.code_templates[language][template_name] = custom_template

        output = "✅ **TEMPLATE CREATED**\n\n"
        output += f"**Language:** {language}\n"
        output += f"**Template Name:** {template_name}\n\n"
        output += f"**Template Content:**\n```\n{custom_template}\n```"

        return ToolCallResult(
            command="create_template",
            success=True,
            output=output,
            error_message=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={"language": language, "template_name": template_name},
        )


async def demo_code_generation(tool_instance: BaseTool):
    """Demonstrate code generation tool functionality"""
    print("🔧 CODE GENERATION TOOL DEMO")
    print("=" * 50)

    # Test 1: Generate a simple Python function
    print("\n--- Test 1: Generate Python Function ---")
    result = await tool_instance.execute(
        action="generate_code",
        language="python",
        template_name="function_basic",
        variables={
            "function_name": "calculate_fibonacci",
            "parameters": "n: int",
            "return_type": " -> int",
            "description": "Calculate the nth Fibonacci number",
            "args_docs": "n: The position in the Fibonacci sequence",
            "return_docs": "The nth Fibonacci number",
            "function_body": "    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        },
    )
    print(f"Success: {result.success}")
    if result.success:
        print(result.output)

    # Test 2: Analyze context
    print("\n--- Test 2: Context Analysis ---")
    result = await tool_instance.execute(
        action="analyze_context",
        context="I need to create a REST API endpoint for user authentication with JWT tokens",
        language="python",
    )
    print(f"Success: {result.success}")
    if result.success:
        print(result.output)

    # Test 3: List templates
    print("\n--- Test 3: List Templates ---")
    result = await tool_instance.execute(action="list_templates", language="python")
    print(f"Success: {result.success}")
    if result.success:
        print(result.output)

    print("\n✅ Code generation demo completed!")
