"""
Defines the `CerebellumConnector` for integrating with a "Cerebellum" LLM.

This module provides a specialized connector for interacting with a low-level,
tactical LLM (conceptualized as the "Cerebellum"), enabling rapid,
fine-grained control and execution of actions within the Crisalida system.
"""

import logging
import time
from typing import Any

from .base_llm_connector import AbstractLLMConnector, LLMRequest, LLMResponse, LLMRole

logger = logging.getLogger(__name__)


class CerebellumConnector(AbstractLLMConnector):
    """Conector para el 'cerebelo' - tareas granulares y codificación"""

    def __init__(self, ollama_client):
        super().__init__("qwen2.5-coder:0.5b", LLMRole.CEREBELLUM, ollama_client)
        self.system_prompts = {
            "coding": """Eres el 'cerebelo' de Jano. Te especializas en:
1. Generación de código específico y preciso
2. Implementación de pasos granulares
3. Corrección y refinamiento de código
4. Tareas de programación detalladas
Genera código limpio y funcional.""",
            "implementation": """Implementa el paso específico que se te solicita.
Sé preciso y funcional. No expliques demasiado, solo implementa.""",
            "debugging": """Revisa y corrige el código proporcionado.
Identifica errores y proporciona la versión corregida.""",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genera respuesta para tareas granulares/código"""
        start_time = time.time()
        try:
            if not await self.health_check():
                return await self.graceful_fallback(request)

            system_prompt = self.system_prompts.get(
                request.task_type, self.system_prompts["coding"]
            )
            full_prompt = f"{system_prompt}\n\nTarea: {request.prompt}"
            if request.context:
                full_prompt += f"\n\nContexto/Código existente:\n{request.context}"

            response = await self.ollama_client.generate_async(
                model_name=self.model_name, prompt=full_prompt
            )

            if response:
                return LLMResponse(
                    content=response,
                    success=True,
                    model_name=self.model_name,
                    response_time=time.time() - start_time,
                    confidence=0.7,
                )
            else:
                return LLMResponse(
                    content="",
                    success=False,
                    model_name=self.model_name,
                    error_message="LLM did not return a response.",
                    confidence=0.0,
                )
        except Exception as e:
            logger.error(f"CerebellumConnector error: {e}")
            return await self.graceful_fallback(request)

    async def health_check(self) -> bool:
        """Verifica disponibilidad del modelo cerebelo"""
        try:
            is_available = await self.ollama_client.check_model_availability(
                self.model_name
            )
            self.is_available = is_available
            self.health_status = "healthy" if self.is_available else "unhealthy"
            return self.is_available
        except Exception as e:
            logger.error(f"Error during health check for {self.model_name}: {e}")
            self.is_available = False
            self.health_status = "unreachable"
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Retorna las capacidades del LLM"""
        return {
            "role": "cerebellum",
            "specialties": ["coding", "implementation", "debugging", "granular_tasks"],
            "task_types": ["coding", "implementation", "debugging"],
            "max_complexity": "medium",
            "supports_json": False,
            "languages": ["python", "javascript", "html", "css", "bash"],
        }
