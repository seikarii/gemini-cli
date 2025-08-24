"""
Semantic analysis logic for the Code Generation Tool.

"""

import logging
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            self.code_patterns = self._load_code_patterns()
            self.pattern_vectors = self.vectorizer.fit_transform(self.code_patterns)
        else:
            self.vectorizer = None
            self.code_patterns = []
            self.pattern_vectors = None

    def _load_code_patterns(self) -> list[str]:
        """Load code patterns for semantic analysis."""
        return [
            "REST API endpoint with error handling",
            "Database model with relationships",
            "Authentication and authorization",
            "Data validation and serialization",
            "Asynchronous processing and queues",
            "Caching and performance optimization",
            "Testing and quality assurance",
            "Configuration and environment management",
            "Logging and monitoring",
            "File processing and storage",
            "Web scraping and data extraction",
            "Machine learning and AI integration",
            "Real-time communication and websockets",
            "Payment processing integration",
            "Email and notification systems",
        ]

    def select_template_by_context(
        self, language: str, context: str, code_templates: dict
    ) -> str | None:
        """Select best template based on semantic context analysis"""
        if not SKLEARN_AVAILABLE or self.pattern_vectors is None:
            return self._fallback_template_selection(language, context, code_templates)

        try:
            # Analyze context similarity to patterns
            context_vector = self.vectorizer.transform([context])
            similarities = cosine_similarity(context_vector, self.pattern_vectors)[0]

            # Find best matching pattern
            best_pattern_idx = int(np.argmax(similarities))
            best_similarity = similarities[best_pattern_idx]

            if best_similarity < 0.2:  # Low similarity threshold
                return self._fallback_template_selection(
                    language, context, code_templates
                )

            # Map patterns to templates
            pattern_template_map = {
                0: "api_endpoint",  # REST API
                1: "data_class",  # Database model
                2: "function_basic",  # Auth
                3: "class_basic",  # Data validation
                4: "async_function",  # Async processing
                5: "function_basic",  # Caching
                6: "test_function",  # Testing
                7: "class_basic",  # Configuration
                8: "function_basic",  # Logging
                9: "function_basic",  # File processing
            }

            template_name = pattern_template_map.get(best_pattern_idx, "function_basic")

            # Ensure template exists for language
            if template_name in code_templates[language]:
                return template_name
            else:
                return "function_basic"

        except Exception as e:
            logger.warning(f"Template selection failed: {e}")
            return self._fallback_template_selection(language, context, code_templates)

    def _fallback_template_selection(
        self, language: str, context: str, code_templates: dict
    ) -> str:
        """Fallback template selection without vectorization"""
        context_lower = context.lower()

        if "api" in context_lower or "endpoint" in context_lower:
            return (
                "api_endpoint"
                if "api_endpoint" in code_templates[language]
                else "function_basic"
            )
        elif "test" in context_lower:
            return (
                "test_function"
                if "test_function" in code_templates[language]
                else "function_basic"
            )
        elif "class" in context_lower or "model" in context_lower:
            return (
                "class_basic"
                if "class_basic" in code_templates[language]
                else "function_basic"
            )
        elif "async" in context_lower:
            return (
                "async_function"
                if "async_function" in code_templates[language]
                else "function_basic"
            )
        else:
            return "function_basic"

    def enhance_variables_with_context(
        self, variables: dict[str, Any], context: str, language: str
    ) -> dict[str, Any]:
        """Enhance template variables using semantic context analysis"""
        enhanced = {}

        # Extract common programming concepts from context
        context_lower = context.lower()

        # Auto-generate names if not provided
        if "function_name" not in variables:
            if "api" in context_lower or "endpoint" in context_lower:
                enhanced["function_name"] = "handle_request"
            elif "test" in context_lower:
                enhanced["function_name"] = "test_functionality"
            elif "process" in context_lower:
                enhanced["function_name"] = "process_data"
            else:
                enhanced["function_name"] = "main_function"

        if "class_name" not in variables:
            if "model" in context_lower or "data" in context_lower:
                enhanced["class_name"] = "DataModel"
            elif "service" in context_lower:
                enhanced["class_name"] = "Service"
            elif "handler" in context_lower:
                enhanced["class_name"] = "Handler"
            else:
                enhanced["class_name"] = "MainClass"

        # Auto-generate descriptions
        if "description" not in variables:
            enhanced["description"] = f"Generated for: {context}"

        # Generate method stubs based on context
        if "method_body" not in variables:
            if language == "python":
                if "api" in context_lower:
                    enhanced["method_body"] = "    # Process API request\n    pass"
                elif "data" in context_lower:
                    enhanced["method_body"] = "    # Process data\n    pass"
                else:
                    enhanced["method_body"] = (
                        "    # TODO: Implement functionality\n    pass"
                    )

        if "function_body" not in variables:
            if language == "python":
                enhanced["function_body"] = "    # TODO: Implement\n    pass"
            elif language == "javascript":
                enhanced["function_body"] = "    // TODO: Implement\n    return null;"

        return enhanced
