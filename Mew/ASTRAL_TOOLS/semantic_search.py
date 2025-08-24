#!/usr/bin/env python3
"""
SemanticSearchTool - Búsqueda Semántica y Análisis Contextual del Código
========================================================================
Herramienta avanzada para búsqueda semántica, análisis de dependencias y mapeo de relaciones en bases de código Python.

Características:
- Búsqueda semántica con embeddings y TF-IDF (scikit-learn)
- Indexación eficiente y caching de segmentos de código
- Análisis de dependencias (imports, llamadas, uso de clases)
- Resultados contextuales y filtrados por similitud
- Integración con análisis AST y soporte para métodos async
- Formato de resultados optimizado para revisión rápida
- Demo interactiva para pruebas

Requiere: scikit-learn, numpy, pydantic

"""

import ast
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available - semantic search will use basic text matching"
    )


class SemanticSearchParameters(BaseModel):
    action: Literal[
        "search_semantic",
        "search_functions",
        "search_classes",
        "analyze_structure",
        "find_dependencies",
        "find_similar_code",
        "index_codebase",
    ] = Field(..., description="Tipo de búsqueda/análisis a realizar")
    query: str = Field(..., description="Consulta semántica o patrón de código")
    path: str | None = Field(None, description="Ruta base para búsqueda (default: cwd)")
    file_extensions: list[str] = Field(
        default=[".py"], description="Extensiones de archivo a incluir"
    )
    max_results: int = Field(10, description="Máximo de resultados")
    include_content: bool = Field(
        True, description="Incluir contenido de código en resultados"
    )
    min_similarity: float = Field(0.1, description="Umbral mínimo de similitud (0-1)")
    exclude_dirs: list[str] = Field(
        default=["__pycache__", ".git", "node_modules", ".venv", "venv"],
        description="Directorios a excluir",
    )
    analyze_imports: bool = Field(False, description="Incluir análisis de imports")
    analyze_calls: bool = Field(False, description="Incluir análisis de llamadas")


class CodeSegment(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    content: str
    type: str  # 'function', 'class', 'method', 'variable', 'import'
    name: str
    context: str
    similarity: float = 0.0


class SemanticSearchTool(BaseTool):
    """Herramienta avanzada para búsqueda semántica y análisis de código Python"""

    def __init__(self):
        super().__init__()
        self._code_index: dict[str, list[CodeSegment]] = {}
        self._vectorizer = None
        self._vectors = None

    def _get_name(self) -> str:
        return "semantic_search"

    def _get_description(self) -> str:
        return (
            "Realiza búsqueda semántica y análisis contextual en bases de código Python"
        )

    def _get_category(self) -> str:
        return "code_analysis"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return SemanticSearchParameters

    def _get_python_files(
        self, root_path: Path, exclude_dirs: list[str], file_extensions: list[str]
    ) -> list[str]:
        """Obtiene todos los archivos Python en el árbol de directorios"""
        python_files = []
        # root_path = Path(root_path) # No longer needed
        for file_path in root_path.rglob("*"):
            if any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
            if file_path.suffix in file_extensions and file_path.is_file():
                python_files.append(str(file_path))
        return python_files

    def _parse_python_file(self, file_path: str) -> list[CodeSegment]:
        """Extrae segmentos de código relevantes usando AST"""
        segments: list[CodeSegment] = []
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return segments
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                segment = None
                if isinstance(node, ast.FunctionDef):
                    segment = CodeSegment(
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno + 1),
                        content="\n".join(
                            lines[
                                node.lineno
                                - 1 : getattr(node, "end_lineno", node.lineno)
                            ]
                        ),
                        type="function",
                        name=node.name,
                        context=self._get_context(
                            lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                        ),
                    )
                elif isinstance(node, ast.ClassDef):
                    segment = CodeSegment(
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno + 1),
                        content="\n".join(
                            lines[
                                node.lineno
                                - 1 : getattr(node, "end_lineno", node.lineno)
                            ]
                        ),
                        type="class",
                        name=node.name,
                        context=self._get_context(
                            lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                        ),
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    segment = CodeSegment(
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno + 1),
                        content="\n".join(
                            lines[
                                node.lineno
                                - 1 : getattr(node, "end_lineno", node.lineno)
                            ]
                        ),
                        type="async_function",
                        name=node.name,
                        context=self._get_context(
                            lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                        ),
                    )
                if segment:
                    segments.append(segment)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
        return segments

    def _get_context(
        self, lines: list[str], start: int, end: int, context_lines: int = 3
    ) -> str:
        """Obtiene contexto circundante para un segmento de código"""
        start_idx = max(0, start - context_lines - 1)
        end_idx = min(len(lines), end + context_lines)
        return "\n".join(lines[start_idx:end_idx])

    def _create_search_index(self, segments: list[CodeSegment]) -> None:
        """Crea índice semántico usando TF-IDF"""
        if not SKLEARN_AVAILABLE:
            return
        corpus = []
        for segment in segments:
            text = f"{segment.name} {segment.content} {segment.context}"
            text = re.sub(r"[^\w\s]", " ", text)
            text = " ".join(text.split())
            corpus.append(text)
        if corpus:
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )
            try:
                self._vectors = self._vectorizer.fit_transform(corpus)
                logger.info(f"Created search index with {len(corpus)} code segments")
            except Exception as e:
                logger.warning(f"Failed to create search index: {e}")
                self._vectorizer = None
                self._vectors = None

    def _semantic_search(
        self, query: str, segments: list[CodeSegment], max_results: int
    ) -> list[CodeSegment]:
        """Búsqueda semántica usando similitud TF-IDF"""
        if not SKLEARN_AVAILABLE or not self._vectorizer or self._vectors is None:
            return self._simple_text_search(query, segments, max_results)
        try:
            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._vectors).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            results = []
            for idx in sorted_indices[:max_results]:
                if similarities[idx] > 0:
                    segment = segments[idx]
                    segment.similarity = float(similarities[idx])
                    results.append(segment)
            return results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return self._simple_text_search(query, segments, max_results)

    def _simple_text_search(
        self, query: str, segments: list[CodeSegment], max_results: int
    ) -> list[CodeSegment]:
        """Búsqueda textual simple como fallback"""
        query_lower = query.lower()
        results = []
        for segment in segments:
            score = 0.0
            text_content = f"{segment.name} {segment.content}".lower()
            if query_lower == segment.name.lower():
                score = 1.0
            elif query_lower in segment.name.lower():
                score = 0.8
            elif query_lower in text_content:
                score = 0.5
            else:
                query_words = query_lower.split()
                word_matches = sum(1 for word in query_words if word in text_content)
                if word_matches > 0:
                    score = (word_matches / len(query_words)) * 0.3
            if score > 0:
                segment.similarity = score
                results.append(segment)
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:max_results]

    def _analyze_dependencies(self, segments: list[CodeSegment]) -> dict[str, Any]:
        """Analiza dependencias entre segmentos de código"""
        dependencies: dict[str, Any] = {
            "imports": {},
            "function_calls": {},
            "class_usage": {},
        }
        for segment in segments:
            try:
                tree = ast.parse(segment.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            dependencies["imports"].setdefault(
                                segment.file_path, []
                            ).append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            dependencies["imports"].setdefault(
                                segment.file_path, []
                            ).append(node.module)
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        dependencies["function_calls"].setdefault(
                            segment.name, []
                        ).append(func_name)
            except Exception:
                continue
        return dependencies

    def _format_results(
        self, segments: list[CodeSegment], include_content: bool = True
    ) -> str:
        """Formatea resultados de búsqueda para visualización"""
        if not segments:
            return "No results found."
        output = []
        output.append(f"Found {len(segments)} results:\n")
        for i, segment in enumerate(segments, 1):
            output.append(f"{i}. {segment.type.title()}: {segment.name}")
            output.append(f"   File: {segment.file_path}:{segment.line_start}")
            if hasattr(segment, "similarity") and segment.similarity > 0:
                output.append(f"   Similarity: {segment.similarity:.3f}")
            if include_content:
                content_lines = segment.content.split("\n")[:5]
                output.append("   Content preview:")
                for line in content_lines:
                    if line.strip():
                        output.append(f"     {line}")
                if len(segment.content.split("\n")) > 5:
                    output.append("     ...")
            output.append("")
        return "\n".join(output)

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = asyncio.get_event_loop().time()
        try:
            params = SemanticSearchParameters(**kwargs)
            search_path = params.path or os.getcwd()
            if not os.path.exists(search_path):
                return ToolCallResult(
                    command="semantic_search",
                    success=False,
                    output="",
                    error_message=f"Path does not exist: {search_path}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )
            output = ""
            if params.action == "index_codebase":
                files = self._get_python_files(
                    Path(search_path), params.exclude_dirs, params.file_extensions
                )
                all_segments = []
                for file_path in files:
                    segments = self._parse_python_file(file_path)
                    all_segments.extend(segments)
                self._code_index[search_path] = all_segments
                self._create_search_index(all_segments)
                output = (
                    f"Indexed {len(files)} files with {len(all_segments)} code segments"
                )
            elif params.action == "search_semantic":
                if search_path not in self._code_index:
                    files = self._get_python_files(
                        Path(search_path), params.exclude_dirs, params.file_extensions
                    )
                    all_segments = []
                    for file_path in files:
                        segments = self._parse_python_file(file_path)
                        all_segments.extend(segments)
                    self._code_index[search_path] = all_segments
                    self._create_search_index(all_segments)
                segments = self._code_index[search_path]
                results = self._semantic_search(
                    params.query, segments, params.max_results
                )
                results = [r for r in results if r.similarity >= params.min_similarity]
                output = self._format_results(results, params.include_content)
            elif params.action == "search_functions":
                if search_path not in self._code_index:
                    files = self._get_python_files(
                        Path(search_path), params.exclude_dirs, params.file_extensions
                    )
                    all_segments = []
                    for file_path in files:
                        segments = self._parse_python_file(file_path)
                        all_segments.extend(segments)
                    self._code_index[search_path] = all_segments
                segments = self._code_index[search_path]
                function_segments = [
                    s for s in segments if s.type in ["function", "async_function"]
                ]
                results = self._semantic_search(
                    params.query, function_segments, params.max_results
                )
                results = [r for r in results if r.similarity >= params.min_similarity]
                output = self._format_results(results, params.include_content)
            elif params.action == "search_classes":
                if search_path not in self._code_index:
                    files = self._get_python_files(
                        Path(search_path), params.exclude_dirs, params.file_extensions
                    )
                    all_segments = []
                    for file_path in files:
                        segments = self._parse_python_file(file_path)
                        all_segments.extend(segments)
                    self._code_index[search_path] = all_segments
                segments = self._code_index[search_path]
                class_segments = [s for s in segments if s.type == "class"]
                results = self._semantic_search(
                    params.query, class_segments, params.max_results
                )
                results = [r for r in results if r.similarity >= params.min_similarity]
                output = self._format_results(results, params.include_content)
            elif params.action == "analyze_structure":
                files = self._get_python_files(
                    Path(search_path), params.exclude_dirs, params.file_extensions
                )
                structure_info = {
                    "total_files": len(files),
                    "functions": 0,
                    "classes": 0,
                    "async_functions": 0,
                }
                all_segments = []
                for file_path in files:
                    segments = self._parse_python_file(file_path)
                    all_segments.extend(segments)
                for segment in all_segments:
                    if segment.type == "function":
                        structure_info["functions"] += 1
                    elif segment.type == "class":
                        structure_info["classes"] += 1
                    elif segment.type == "async_function":
                        structure_info["async_functions"] += 1
                output = "Codebase Structure Analysis:\n\n"
                output += f"Total files: {structure_info['total_files']}\n"
                output += f"Total functions: {structure_info['functions']}\n"
                output += (
                    f"Total async functions: {structure_info['async_functions']}\n"
                )
                output += f"Total classes: {structure_info['classes']}\n"
                file_stats = {}
                for segment in all_segments:
                    file_path = segment.file_path
                    if file_path not in file_stats:
                        file_stats[file_path] = {
                            "functions": 0,
                            "classes": 0,
                            "async_functions": 0,
                        }
                    file_stats[file_path][segment.type.replace("async_", "")] += 1
                output += "\nTop files by content:\n"
                sorted_files = sorted(
                    file_stats.items(), key=lambda x: sum(x[1].values()), reverse=True
                )[:10]
                for file_path, stats in sorted_files:
                    relative_path = os.path.relpath(file_path, search_path)
                    total = sum(stats.values())
                    output += f"  {relative_path}: {total} items ({stats})\n"
            elif params.action == "find_dependencies":
                files = self._get_python_files(
                    Path(search_path), params.exclude_dirs, params.file_extensions
                )
                all_segments = []
                for file_path in files:
                    segments = self._parse_python_file(file_path)
                    all_segments.extend(segments)
                deps = self._analyze_dependencies(all_segments)
                output = "Dependency Analysis:\n\n"
                if deps["imports"]:
                    output += "File Imports:\n"
                    for file_path, imports in deps["imports"].items():
                        relative_path = os.path.relpath(file_path, search_path)
                        output += f"  {relative_path}: {', '.join(set(imports))}\n"
                    output += "\n"
                if deps["function_calls"]:
                    output += "Function Call Dependencies:\n"
                    for func_name, calls in deps["function_calls"].items():
                        if calls:
                            output += f"  {func_name} calls: {', '.join(set(calls))}\n"
            elif params.action == "find_similar_code":
                if search_path not in self._code_index:
                    files = self._get_python_files(
                        Path(search_path), params.exclude_dirs, params.file_extensions
                    )
                    all_segments = []
                    for file_path in files:
                        segments = self._parse_python_file(file_path)
                        all_segments.extend(segments)
                    self._code_index[search_path] = all_segments
                    self._create_search_index(all_segments)
                segments = self._code_index[search_path]
                results = self._semantic_search(
                    params.query, segments, params.max_results
                )
                results = [r for r in results if r.similarity >= params.min_similarity]
                output = f"Similar code found for query:\n{params.query[:100]}...\n\n"
                output += self._format_results(results, params.include_content)
            else:
                return ToolCallResult(
                    command="semantic_search",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {params.action}",
                    execution_time=(asyncio.get_event_loop().time() - start_time),
                )
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolCallResult(
                command=f"semantic_search({params.action})",
                success=True,
                output=output,
                error_message=None,
                execution_time=execution_time,
                metadata={
                    "action": params.action,
                    "query": params.query,
                    "search_path": search_path,
                    "sklearn_available": SKLEARN_AVAILABLE,
                },
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Semantic search error: {e}")
            return ToolCallResult(
                command="semantic_search",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def demo(self):
        """Demonstrate the semantic search tool's functionality."""
        print("🔍 SEMANTIC SEARCH TOOL DEMO")
        print("=" * 40)

        # Search for files with specific content
        result = await self.execute(
            action="search_files",
            directory="./crisalida_lib",
            query="import numpy",
            file_extensions=[".py"],
            max_results=3,
        )
        print(f"Search files: {result.success}")
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Search for symbols/functions
        result = await self.execute(
            action="search_symbols",
            directory="./crisalida_lib",
            symbol_name="execute",
            max_results=5,
        )
        print(f"\nSearch symbols: {result.success}")
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        print("\n✅ Semantic search demo completed!")


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.semantic_search_demos import (
        demo_semantic_search,
    )

    asyncio.run(demo_semantic_search())
