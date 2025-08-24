import json
import logging
from datetime import datetime  # Import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class LLMManagementParams(BaseModel):
    """
    Parámetros para la gestión avanzada de conectores LLM.
    """

    action: Literal[
        "status",
        "enable",
        "disable",
        "health_check",
        "toggle_mode",
        "list_models",
        "check_model_availability",
        "generate_text",
    ] = Field(..., description="Acción a realizar")
    target: Literal["brain", "cerebellum", "all"] = Field(
        "all", description="Destino LLM"
    )
    model_name: str | None = Field(None, description="Nombre del modelo LLM")
    prompt: str | None = Field(None, description="Prompt para generación de texto")
    temperature: float | None = Field(None, description="Temperatura para generación")
    max_tokens: int | None = Field(None, description="Máximo de tokens para generación")


class LLMManagementTool(BaseTool):
    """
    Herramienta avanzada para gestionar conectores LLM (brain/cerebellum) en Crisalida.
    Permite habilitar/deshabilitar, consultar estado, listar modelos y generar texto.
    """

    def __init__(self, llm_gateway: Any):
        super().__init__()
        self.llm_gateway = llm_gateway

    def _get_name(self) -> str:
        return "llm_management"

    def _get_description(self) -> str:
        return (
            "Gestiona conectores LLM locales (brain/cerebellum) para desarrollo y operación de Jano. "
            "Permite alternar modos, consultar estado, listar modelos y generar texto."
        )

    def _get_pydantic_schema(self):
        return LLMManagementParams

    def _get_category(self) -> str:
        return "llm_management"

    async def execute(self, **kwargs) -> ToolCallResult:
        params = LLMManagementParams(**kwargs)
        action = params.action
        start_time = datetime.now()  # Added this line

        try:
            if action == "status":
                return await self._get_status()
            elif action == "enable":
                return await self._enable_llm_mode()
            elif action == "disable":
                return await self._disable_llm_mode()
            elif action == "health_check":
                return await self._perform_health_check()
            elif action == "toggle_mode":
                return await self._toggle_llm_mode()
            elif action == "list_models":
                if not self.llm_gateway:
                    return ToolCallResult(
                        command="llm_management list_models",
                        success=False,
                        output="",
                        error_message="LLM Gateway not available",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                models = await self.llm_gateway.list_models()
                return ToolCallResult(
                    command="llm_management list_models",
                    success=True,
                    output=json.dumps(models, indent=2),
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                    error_message=None,  # Added this line
                )
            elif action == "check_model_availability":
                if not self.llm_gateway:
                    return ToolCallResult(
                        command="llm_management check_model_availability",
                        success=False,
                        output="",
                        error_message="LLM Gateway not available",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                if not params.model_name:
                    return ToolCallResult(
                        command="llm_management check_model_availability",
                        success=False,
                        output="",
                        error_message="model_name is required for check_model_availability",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                is_available = await self.llm_gateway.check_model_availability(
                    params.model_name
                )
                return ToolCallResult(
                    command="llm_management check_model_availability",
                    success=True,
                    output=f"Model '{params.model_name}' available: {is_available}",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                    error_message=None,  # Added this line
                )
            elif action == "generate_text":
                if not self.llm_gateway:
                    return ToolCallResult(
                        command="llm_management generate_text",
                        success=False,
                        output="",
                        error_message="LLM Gateway not available",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                if not params.model_name or not params.prompt:
                    return ToolCallResult(
                        command="llm_management generate_text",
                        success=False,
                        output="",
                        error_message="model_name and prompt are required for generate_text",
                        execution_time=(
                            datetime.now() - start_time
                        ).total_seconds(),  # Added this line
                    )
                generate_kwargs: dict[str, Any] = {
                    "model_name": params.model_name,
                    "prompt": params.prompt,
                }
                if params.temperature is not None:
                    generate_kwargs["temperature"] = params.temperature
                if params.max_tokens is not None:
                    generate_kwargs["max_tokens"] = params.max_tokens
                generated_text = await self.llm_gateway.generate_async(
                    **generate_kwargs
                )
                return ToolCallResult(
                    command="llm_management generate_text",
                    success=True,
                    output=generated_text or "No response generated.",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                    error_message=None,  # Added this line
                )
            else:
                return ToolCallResult(
                    command=f"llm_management {action}",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {action}",
                    execution_time=(
                        datetime.now() - start_time
                    ).total_seconds(),  # Added this line
                )
        except Exception as e:
            logger.error(f"LLMManagementTool error: {e}")
            return ToolCallResult(
                command=f"llm_management {action}",
                success=False,
                output="",
                error_message=f"Tool execution failed: {str(e)}",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )

    async def _get_status(self) -> ToolCallResult:
        """Obtiene el estado actual de los LLMs"""
        start_time = datetime.now()  # Added this line
        if not self.llm_gateway:
            return ToolCallResult(
                command="llm_management status",
                success=True,
                output="LLM Gateway not initialized - running in offline mode",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
                error_message=None,  # Added this line
            )
        health_status = await self.llm_gateway.health_monitor.check_all()
        health_summary = self.llm_gateway.health_monitor.get_health_summary()
        output = "🧠 LLM Status Report:\n"
        output += f"Mode: {'Enabled' if self.llm_gateway.is_llm_mode_enabled else 'Disabled (Offline Mode)'}\n"
        output += "Individual Models:\n"
        for role, status in health_status["individual_status"].items():
            emoji = "✅" if status["available"] else "❌"
            output += (
                f"  {emoji} {role.title()}: {status['model']} - {status['status']}\n"
            )
        output += "\nOverall Status:\n"
        output += (
            f"  Any Available: {'Yes' if health_status['any_available'] else 'No'}\n"
        )
        output += (
            f"  All Available: {'Yes' if health_status['all_available'] else 'No'}\n"
        )
        output += f"  Health Grade: {health_summary['overall_health']}\n"
        output += "Availability Rates (last 10 checks):\n"
        for role, rate in health_summary.get("availability_rates", {}).items():
            output += f"  {role.title()}: {rate:.1%}\n"
        return ToolCallResult(
            command="llm_management status",
            success=True,
            output=output.strip(),
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _enable_llm_mode(self) -> ToolCallResult:
        """Habilita el modo LLM"""
        start_time = datetime.now()  # Added this line
        if not self.llm_gateway:
            return ToolCallResult(
                command="llm_management enable",
                success=False,
                output="",
                error_message="LLM Gateway not available",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        await self.llm_gateway.enable_llm_mode()
        return ToolCallResult(
            command="llm_management enable",
            success=True,
            output="🧠 LLM mode enabled - Jano will use brain/cerebellum when available",
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _disable_llm_mode(self) -> ToolCallResult:
        """Deshabilita el modo LLM (offline)"""
        start_time = datetime.now()  # Added this line
        if not self.llm_gateway:
            return ToolCallResult(
                command="llm_management disable",
                success=True,
                output="Already in offline mode",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
                error_message=None,  # Added this line
            )
        await self.llm_gateway.enable_offline_mode()
        return ToolCallResult(
            command="llm_management disable",
            success=True,
            output="🔄 LLM mode disabled - Jano operating independently",
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _perform_health_check(self) -> ToolCallResult:
        """Chequea la salud de los LLMs"""
        start_time = datetime.now()  # Added this line
        if not self.llm_gateway:
            return ToolCallResult(
                command="llm_management health_check",
                success=False,
                output="",
                error_message="LLM Gateway not available",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        health_status = await self.llm_gateway.health_monitor.check_all()
        output = "LLM Health Check Results:\n"
        for role, status in health_status["individual_status"].items():
            output += f"  {role.title()}: {status['model']} - {status['status']}\n"
        return ToolCallResult(
            command="llm_management health_check",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Added this line
            error_message=None,  # Added this line
        )

    async def _toggle_llm_mode(self) -> ToolCallResult:
        """Alterna el modo LLM (habilitar/deshabilitar)"""
        start_time = datetime.now()  # Added this line
        if not self.llm_gateway:
            return ToolCallResult(
                command="llm_management toggle_mode",
                success=False,
                output="",
                error_message="LLM Gateway not available",
                execution_time=(
                    datetime.now() - start_time
                ).total_seconds(),  # Added this line
            )
        if self.llm_gateway.is_llm_mode_enabled:
            return await self._disable_llm_mode()
        else:
            return await self._enable_llm_mode()

    async def demo(self):
        """Demonstrates the LLMManagementTool's functionality by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.demo_llm_management import (
            demonstrate_llm_management,
        )

        return await demonstrate_llm_management()
