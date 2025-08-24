#!/usr/bin/env python3
"""
Base Tool Architecture for Prometheus Agent
===========================================
Foundation for all tool implementations in the Crisalida ecosystem.

- Abstract base class for tools with async execution and pydantic validation.
- Registry system for tool discovery, grouping, and execution.
- Utility for rapid prototyping of new tools.

This module is designed for extensibility, introspection, and robust error handling.
"""

import asyncio
import inspect
import logging
import os
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ToolCallResult(BaseModel):
    """
    Result of a tool call execution.
    Matches the TypeScript Gemini CLI interface, with extended metadata.
    """

    command: str = Field(..., description="Command that was executed")
    success: bool = Field(..., description="Whether the execution was successful")
    output: str = Field(..., description="Output from the tool execution")
    error_message: str | None = Field(
        None, description="Error message if execution failed"
    )
    execution_time: float = Field(0.0, description="Time taken to execute in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the execution occurred"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"ToolCallResult({status} {self.command}, {self.execution_time:.2f}s)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class ToolMetadata:
    """
    Metadatos de una herramienta registrada.
    Attributes:
        name: Nombre único de la herramienta
        description: Descripción de la herramienta
        version: str = "1.0.0"
        author: str = "unknown"
        registration_time: str = field(default_factory=lambda: datetime.now().isoformat())
        usage_count: int = 0
        last_used: str | None = None
        is_dynamic: bool = False
        source_file: str | None = None
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    registration_time: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    last_used: str | None = None
    is_dynamic: bool = False
    source_file: str | None = None


def is_within_root(path_to_check: str, root_directory: str) -> bool:
    """
    Check if a path is within a given root directory.
    Ported from fileUtils.ts isWithinRoot function.
    """
    try:
        normalized_path = os.path.abspath(path_to_check)
        normalized_root = os.path.abspath(root_directory)
        if not normalized_root.endswith(os.sep) and normalized_root != os.sep:
            root_with_separator = normalized_root + os.sep
        else:
            root_with_separator = normalized_root
        return normalized_path == normalized_root or normalized_path.startswith(
            root_with_separator
        )
    except Exception:
        return False


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Prometheus Agent toolkit.
    Defines the interface for async execution, pydantic validation, and introspection.
    """

    def __init__(self) -> None:
        self._name = self._get_name()
        self._description = self._get_description()
        self._pydantic_schema = self._get_pydantic_schema()
        self._category = self._get_category()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    def pydantic_schema(self) -> type[BaseModel]:
        return self._pydantic_schema

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def _get_description(self) -> str:
        pass

    @abstractmethod
    def _get_pydantic_schema(self) -> type[BaseModel]:
        pass

    @abstractmethod
    def _get_category(self) -> str:
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolCallResult:
        """
        Execute the tool with the given parameters.
        Args:
            **kwargs: Tool parameters (validated against pydantic_schema)
        Returns:
            ToolCallResult with execution details
        """
        pass

    def execute_sync(self, **kwargs) -> ToolCallResult:
        """
        Execute the tool synchronously.
        Args:
            **kwargs: Tool parameters (validated against pydantic_schema)
        Returns:
            ToolCallResult with execution details
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute(**kwargs))

    def validate_parameters(self, parameters: dict[str, Any]) -> BaseModel:
        """
        Validate parameters against the tool's pydantic schema.
        Raises ValidationError if invalid.
        """
        try:
            result = self._pydantic_schema(**parameters)
            return result
        except ValidationError as e:
            logger.error(f"Parameter validation failed for {self.name}: {e}")
            raise

    async def safe_execute(self, **kwargs) -> ToolCallResult:
        """
        Execute the tool with automatic parameter validation and error handling.
        Returns ToolCallResult with execution details.
        """
        start_time = datetime.now()
        try:
            validated_params = self.validate_parameters(kwargs)
            result = await self.execute(**validated_params.model_dump())
            if result.execution_time == 0.0:
                result.execution_time = (datetime.now() - start_time).total_seconds()
            return result
        except ValidationError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolCallResult(
                command=f"{self.name} (validation failed)",
                success=False,
                output="",
                error_message=f"Parameter validation failed: {str(e)}",
                execution_time=execution_time,
            )
        except Exception as e:
            import traceback

            logger.error(
                f"Tool {self.name} execution failed: {e}\n{traceback.format_exc()}"
            )
            return ToolCallResult(
                command=f"{self.name}",
                success=False,
                output="",
                error_message=f"Tool execution failed: {str(e)}\n{traceback.format_exc()}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def safe_execute_sync(self, **kwargs) -> ToolCallResult:
        """
        Execute the tool synchronously with automatic parameter validation and error handling.
        Returns ToolCallResult with execution details.
        """
        start_time = datetime.now()
        try:
            validated_params = self.validate_parameters(kwargs)
            result = self.execute_sync(**validated_params.model_dump())
            if result.execution_time == 0.0:
                result.execution_time = (datetime.now() - start_time).total_seconds()
            return result
        except ValidationError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolCallResult(
                command=f"{self.name} (validation failed)",
                success=False,
                output="",
                error_message=f"Parameter validation failed: {str(e)}",
                execution_time=execution_time,
            )
        except Exception as e:
            import traceback

            logger.error(
                f"Tool {self.name} execution failed: {e}\n{traceback.format_exc()}"
            )
            return ToolCallResult(
                command=f"{self.name}",
                success=False,
                output="",
                error_message=f"Tool execution failed: {str(e)}\n{traceback.format_exc()}",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def get_schema_info(self) -> dict[str, Any]:
        """Get information about the tool's schema for introspection"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self._pydantic_schema.model_json_schema(),
        }

    @abstractmethod
    async def demo(self):
        """Demonstrate the tool's functionality."""
        pass


class ToolRegistry:
    """
    Registry for managing tool instances.
    Centralized registration, discovery, grouping, and execution.

    Example:
        registry = ToolRegistry()
        registry.register_tool(my_tool)
        result = await registry.execute_tool("my_tool", {"param1": "value1", "param2": 2})
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.metadata: dict[str, ToolMetadata] = {}
        # Stats may contain numeric counters as well as aggregate lists/dicts
        # Use Any for values to avoid overly strict typing at runtime.
        self.stats: dict[str, Any] = {
            "total_registrations": 0.0,
            "total_deregistrations": 0.0,
            "total_executions": 0.0,
            "failed_executions": 0.0,
            "dynamic_tools_loaded": 0.0,
            "static_tools_count": 0.0,
        }
        self.registration_history: list[dict[str, Any]] = []
        self.max_history_size = 1000
        logger.info("🔧 ToolRegistry initialized")

    def register_tool(
        self,
        tool: BaseTool,
        is_dynamic: bool = False,
        source_file: str | None = None,
    ) -> bool:
        """
        Registra una nueva herramienta en el sistema.
        Args:
            tool: Instancia de la herramienta a registrar
            is_dynamic: Si es una herramienta cargada dinámicamente
            source_file: Archivo fuente (si es dinámica)
        Returns:
            True si el registro fue exitoso, False en caso contrario
        """
        try:
            if not isinstance(tool, BaseTool):
                raise ValueError("La herramienta debe heredar de BaseTool")
            tool_name = tool.name
            if tool_name in self._tools:
                logger.warning(f"⚠️ La herramienta '{tool_name}' ya está registrada")
                return False
            # The BaseTool in this file already ensures execute is callable
            # if not hasattr(tool, "execute") or not callable(tool.execute):
            #     raise ValueError(
            #         "La herramienta debe tener un método 'execute' callable"
            #     )
            self._tools[tool_name] = tool
            metadata = ToolMetadata(
                name=tool_name,
                description=tool.description,
                version=getattr(tool, "version", "1.0.0"),
                author=getattr(tool, "author", "unknown"),
                is_dynamic=is_dynamic,
                source_file=source_file,
            )
            self.metadata[tool_name] = metadata
            self.stats["total_registrations"] += 1
            if is_dynamic:
                self.stats["dynamic_tools_loaded"] += 1
            else:
                self.stats["static_tools_count"] += 1
            self._add_to_history(
                {
                    "operation": "register",
                    "tool_name": tool_name,
                    "success": True,
                    "timestamp": metadata.registration_time,
                    "is_dynamic": is_dynamic,
                }
            )
            logger.info(
                f"✅ Herramienta registrada: {tool_name} "
                f"({'dinámica' if is_dynamic else 'estática'})"
            )
            return True
        except Exception as e:
            logger.error(
                f"❌ Error registrando herramienta {getattr(tool, 'name', 'unknown')}: {e}"
            )
            self._add_to_history(
                {
                    "operation": "register",
                    "tool_name": getattr(tool, "name", "unknown"),
                    "success": False,
                    "timestamp": self._get_timestamp(),
                    "error": str(e),
                }
            )
            return False

    def unregister_tool(self, name: str, force: bool = False) -> bool:
        """
        Desregistra una herramienta del sistema.
        Args:
            tool_name: Nombre de la herramienta a desregistrar
            force: Si se debe forzar el desregistro incluso si está en uso
        Returns:
            True si el desregistro fue exitoso, False en caso contrario
        """
        try:
            if name not in self._tools:
                logger.warning(f"⚠️ La herramienta '{name}' no está registrada")
                return False
            metadata = self.metadata[name]
            if not metadata.is_dynamic and not force:
                logger.warning(
                    f"⚠️ No se puede desregistrar herramienta estática '{name}' sin force=True"
                )
                return False
            del self._tools[name]
            del self.metadata[name]
            self.stats["total_deregistrations"] += 1
            if metadata.is_dynamic:
                self.stats["dynamic_tools_loaded"] = max(
                    0, self.stats["dynamic_tools_loaded"] - 1
                )
            else:
                self.stats["static_tools_count"] = max(
                    0, self.stats["static_tools_count"] - 1
                )
            self._add_to_history(
                {
                    "operation": "unregister",
                    "tool_name": name,
                    "success": True,
                    "timestamp": self._get_timestamp(),
                    "forced": force,
                }
            )
            logger.info(f"✅ Herramienta desregistrada: {name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error desregistrando herramienta '{name}': {e}")
            self._add_to_history(
                {
                    "operation": "unregister",
                    "tool_name": name,
                    "success": False,
                    "timestamp": self._get_timestamp(),
                    "error": str(e),
                    "forced": force,
                }
            )
            return False

    def get_tool(self, name: str) -> BaseTool | None:
        """
        Obtiene una herramienta por su nombre.
        Args:
            tool_name: Nombre de la herramienta
        Returns:
            Instancia de la herramienta o None si no existe
        """
        tool = self._tools.get(name)
        if tool:
            metadata = self.metadata.get(name)
            if metadata:
                metadata.usage_count += 1
                metadata.last_used = self._get_timestamp()
        return tool

    def list_tools(self, filter_dynamic: bool | None = None) -> list[str]:
        """
        Lista todas las herramientas registradas.
        Args:
            filter_dynamic: Si se debe filtrar por tipo (None = todas, True = solo dinámicas, False = solo estáticas)
        Returns:
            Lista de nombres de herramientas
        """
        if filter_dynamic is None:
            return list(self._tools.keys())
        return [
            name
            for name, metadata in self.metadata.items()
            if metadata.is_dynamic == filter_dynamic
        ]

    def get_all_tools(self) -> dict[str, BaseTool]:
        return self._tools.copy()

    def clear(self) -> None:
        count = len(self._tools)
        self._tools.clear()
        logger.info(f"🔧 Cleared {count} tools from registry")

    async def execute_tool(self, tool_name: str, *args, **kwargs) -> ToolCallResult:
        """
        Ejecuta una herramienta por su nombre (async compatible).
        Args:
            tool_name: Nombre de la herramienta
            *args, **kwargs: Parámetros de ejecución
        Returns:
            Resultado de la ejecución
        Raises:
            ValueError: Si la herramienta no existe o falla la ejecución
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolCallResult(
                command=f"{tool_name} (not found)",
                success=False,
                output="",
                error_message=f"Tool '{tool_name}' not found in registry. Available tools: {', '.join(self.list_tools())}",
                execution_time=0.0,
            )
        logger.debug(f"🔧 Ejecutando herramienta: {tool_name}")
        try:
            result = await tool.safe_execute(**kwargs)
            self.stats["total_executions"] += 1
            logger.debug(f"✅ Herramienta {tool_name} ejecutada exitosamente")
            return result
        except Exception as e:
            self.stats["failed_executions"] += 1
            logger.error(f"❌ Error ejecutando herramienta {tool_name}: {e}")
            error_msg = str(e)
            return ToolCallResult(
                command=f"{tool_name}",
                success=False,
                output="",
                error_message=error_msg,  # type: ignore # Mypy seems to misinterpret this valid assignment
                execution_time=0.0,
            )

    def get_registry_info(self) -> dict[str, Any]:
        return {
            "total_tools": len(self._tools),
            "tools": {
                name: tool.get_schema_info() for name, tool in self._tools.items()
            },
        }

    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """
        Obtiene información detallada de una herramienta.
        Returns:
            Diccionario con información de la herramienta o None si no existe
        """
        if tool_name not in self._tools:
            return None
        tool = self._tools[tool_name]
        metadata = self.metadata[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "version": metadata.version,
            "author": metadata.author,
            "parameters_schema": tool.pydantic_schema().model_json_schema(),
            "registration_time": metadata.registration_time,
            "usage_count": metadata.usage_count,
            "last_used": metadata.last_used,
            "is_dynamic": metadata.is_dynamic,
            "source_file": metadata.source_file,
            "tool_class": tool.__class__.__name__,
            "category": tool.category,
        }

    def get_all_tools_info(self) -> dict[str, dict[str, Any]]:
        """
        Obtiene información de todas las herramientas registradas.
        Returns:
            Diccionario con información de todas las herramientas
        """
        return {
            name: info
            for name in self._tools.keys()
            if (info := self.get_tool_info(name)) is not None
        }

    def search_tools(self, query: str) -> list[str]:
        """
        Busca herramientas por nombre o descripción.
        Args:
            query: Término de búsqueda
        Returns:
            Lista de nombres de herramientas que coinciden
        """
        query_lower = query.lower()
        results = []
        for tool_name, tool in self._tools.items():
            if query_lower in tool_name.lower():
                results.append(tool_name)
                continue
            if query_lower in tool.description.lower():
                results.append(tool_name)
        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Devuelve estadísticas del registro.
        Returns:
            Diccionario con estadísticas detalladas
        """
        stats = self.stats.copy()
        total_executions = stats["total_executions"]
        if total_executions > 0:
            stats["success_rate"] = (
                total_executions - stats["failed_executions"]
            ) / total_executions
            stats["failure_rate"] = stats["failed_executions"] / total_executions
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        stats["total_tools"] = len(self._tools)
        stats["dynamic_tools_count"] = self.stats["dynamic_tools_loaded"]
        stats["static_tools_count"] = self.stats["static_tools_count"]
        stats["history_size"] = len(self.registration_history)
        most_used = sorted(
            [item for item in self.metadata.items() if item[1] is not None],
            key=lambda x: x[1].usage_count,
            reverse=True,
        )[:5]
        stats["most_used_tools"] = [
            {"name": name, "usage_count": meta.usage_count} for name, meta in most_used
        ]
        return stats

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Devuelve el historial de operaciones.
        Args:
            limit: Límite de entradas a devolver
        Returns:
            Lista de operaciones del historial
        """
        if limit is None:
            return self.registration_history.copy()
        return self.registration_history[-limit:]

    def clear_history(self) -> None:
        """Limpia el historial de operaciones"""
        self.registration_history.clear()
        logger.info("🧹 Historial de registro limpiado")

    def reset_stats(self) -> None:
        """Reinicia las estadísticas"""
        for metadata in self.metadata.values():
            metadata.usage_count = 0
            metadata.last_used = None
        logger.info("📊 Estadísticas de registro reiniciadas")

    def _add_to_history(self, entry: dict[str, Any]) -> None:
        """Añade una entrada al historial"""
        self.registration_history.append(entry)  # type: ignore
        if len(self.registration_history) > self.max_history_size:
            self.registration_history = self.registration_history[
                -self.max_history_size :
            ]

    def _get_timestamp(self) -> str:
        """Devuelve timestamp actual en formato ISO"""
        return datetime.now().isoformat()

    def validate_registry_integrity(self) -> dict[str, Any]:
        """
        Valida la integridad del registro.
        """
        issues = []
        tool_names = set(self._tools.keys())
        metadata_names = set(self.metadata.keys())
        if tool_names != metadata_names:
            missing_metadata = tool_names - metadata_names
            missing_tools = metadata_names - tool_names
            if missing_metadata:
                issues.append(f"Herramientas sin metadatos: {missing_metadata}")
            if missing_tools:
                issues.append(f"Metadatos sin herramientas: {missing_tools}")
        for tool_name, _ in self._tools.items():
            try:
                # The BaseTool in this file already ensures execute is callable
                # if not hasattr(tool, "execute"):
                #     issues.append(
                #         f"La herramienta '{tool_name}' no tiene método execute"
                #     )
                # elif not callable(tool.execute):
                #     issues.append(f"El método execute de '{tool_name}' no es callable")
                pass  # No need for explicit check here, BaseTool enforces it
            except Exception as e:
                issues.append(f"Error validando herramienta '{tool_name}': {e}")
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "total_tools": len(self._tools),
            "total_metadata": len(self.metadata),
        }

    def get_tools_by_category(self) -> dict[str, list[str]]:
        """
        Group tools by category based on their names.
        Returns a dict of category to tool names.
        """
        categories: dict[str, list[str]] = {
            "file_system": [],
            "shell": [],
            "web": [],
            "memory": [],
            "other": [],
        }
        for name, tool in self._tools.items():
            categories.setdefault(tool.category, []).append(name)
        return {k: v for k, v in categories.items() if v}

    def auto_discover_tools(self, package_name: str = "crisalida_lib.tools") -> int:
        """
        Automatically discover and register all BaseTool subclasses in a package.
        Returns number of tools discovered and registered.
        """
        discovered_count = 0
        try:
            package = __import__(package_name, fromlist=[""])
            package_path = package.__path__
            for _importer, modname, _ispkg in pkgutil.walk_packages(
                package_path, package_name + "."
            ):
                try:
                    if modname.endswith("__init__") or "test" in modname.lower():
                        continue
                    module = __import__(modname, fromlist=[""])
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, BaseTool)
                            and obj is not BaseTool
                            and not inspect.isabstract(obj)
                        ):
                            # Exclude tools that require specific __init__ arguments
                            if name == "LLMManagementTool":
                                logger.warning(
                                    f"⚠️ Skipping auto-discovery for {name} as it requires specific __init__ arguments."
                                )
                                continue
                            try:
                                tool_instance = obj()
                                if tool_instance.name not in self._tools:
                                    self.register_tool(tool_instance)
                                    discovered_count += 1
                                    logger.debug(
                                        f"🔍 Auto-discovered tool: {tool_instance.name}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to instantiate tool {name}: {e}"
                                )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to import module {modname}: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to auto-discover tools in {package_name}: {e}")
        if discovered_count > 0:
            logger.info(f"🔍 Auto-discovered {discovered_count} tools")
        else:
            logger.info("🔍 No new tools discovered")
        return discovered_count

    def get_tools_by_category_enhanced(self) -> dict[str, list[dict[str, str]]]:
        """
        Get tools grouped by their explicit categories with additional info.
        Returns dict of category to list of tool info dicts.
        """
        categories: dict[str, list[dict[str, str]]] = {}
        for name, tool in self._tools.items():
            category = tool.category
            if category not in categories:
                categories[category] = []
            categories[category].append(
                {"name": name, "description": tool.description, "category": category}
            )
        return categories


def create_simple_tool(
    name: str,
    description: str,
    schema_class: type[BaseModel],
    execute_func: Callable[..., Awaitable[ToolCallResult]],
    category: str = "general",
) -> BaseTool:
    """
    Create a simple tool from a function.
    Useful for rapid prototyping without a full class definition.

    Example:
        class MyToolParams(BaseModel):
            param1: str
            param2: int

        async def my_tool_execute(**kwargs) -> ToolCallResult:
            # ... implementation ...
            return ToolCallResult(
                command="my_tool",
                success=True,
                output="...",
            )

        my_tool = create_simple_tool(
            name="my_tool",
            description="My simple tool",
            schema_class=MyToolParams,
            execute_func=my_tool_execute,
            category="custom",
        )
    """

    class SimpleTool(BaseTool):
        def _get_name(self) -> str:
            return name

        def _get_description(self) -> str:
            return description

        def _get_pydantic_schema(self) -> type[BaseModel]:
            return schema_class

        def _get_category(self) -> str:
            return category

        async def execute(self, **kwargs) -> ToolCallResult:
            return await execute_func(**kwargs)

        async def demo(self):
            """Simple demo implementation for dynamically created tools."""
            print(f"Demo for {self.name}: {self.description}")
            return ToolCallResult(
                command=f"demo_{self.name}",
                success=True,
                output=f"Demo completed for {self.name}",
                execution_time=0.1,
            )

    return SimpleTool()


async def demo_base_tools():
    """
    Demonstrate the base tool architecture with a simple test tool.
    """
    print("🔧 BASE TOOL ARCHITECTURE DEMO")
    print("=" * 50)

    class TestParams(BaseModel):
        message: str
        count: int = 1

    async def test_execute(**kwargs) -> ToolCallResult:
        message = kwargs["message"]
        count = kwargs["count"]
        await asyncio.sleep(0.1)
        output = f"Processed '{message}' {count} time(s)"
        return ToolCallResult(
            command="test_tool",
            success=True,
            output=output,
            execution_time=0.1,
            error_message=None,
        )

    test_tool = create_simple_tool(
        name="test_tool",
        description="A simple test tool for demonstration",
        schema_class=TestParams,
        execute_func=test_execute,
    )
    registry = ToolRegistry()
    registry.register_tool(test_tool)
    print(f"Registered tools: {registry.list_tools()}")
    result = await registry.execute_tool("test_tool", {"message": "Hello", "count": 3})
    print(f"Result: {result}")
    print(f"Output: {result.output}")
    result = await registry.execute_tool("test_tool", {"invalid": "param"})
    print(f"Invalid result: {result}")
    print("\n✅ Base tool architecture demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_base_tools())
