"""
BrainConnector - Integración avanzada con LLM estratégico ("Brain")
===================================================================

Conector especializado para razonamiento, planificación y validación de alto nivel.
Incluye diagnóstico extendido, manejo robusto de errores, logging y trazabilidad.
Optimizado para interacción estructurada y generación de prompts para el cerebelo.
"""

import logging
import time
from typing import Any

from .base_llm_connector import AbstractLLMConnector, LLMRequest, LLMResponse, LLMRole

logger = logging.getLogger(__name__)


class BrainConnector(AbstractLLMConnector):
    """Conector para el 'cerebro' - razonamiento estratégico y planificación"""

    def __init__(self, ollama_client, model_name: str) -> None:
        super().__init__(model_name, LLMRole.BRAIN, ollama_client)
        self.system_prompts = {
            "reasoning": (
                "Eres el 'cerebro' de Jano. Tu función es:\n"
                "1. Análisis de alto nivel de problemas complejos\n"
                "2. Planificación estratégica de soluciones\n"
                "3. Generación de prompts específicos para el 'cerebelo'\n"
                "4. Revisión y validación de resultados\n"
                "Responde siempre en JSON estructurado."
            ),
            "planning": (
                "Como cerebro de Jano, descompón tareas complejas en pasos ejecutables.\n"
                "Genera un plan detallado que el cerebelo pueda implementar paso a paso."
            ),
            "review": (
                "Revisa el trabajo del cerebelo. Identifica errores, mejoras y próximos pasos."
            ),
        }
        self.last_prompt: str = ""
        self.last_response: LLMResponse | None = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Genera respuesta de razonamiento de alto nivel con diagnóstico y logging"""
        start_time = time.time()
        try:
            # Verificar disponibilidad primero
            if not await self.health_check():
                return await self.graceful_fallback(request)

            # Seleccionar prompt del sistema apropiado
            system_prompt = self.system_prompts.get(
                request.task_type, self.system_prompts["reasoning"]
            )

            # Construir prompt completo
            full_prompt = f"{system_prompt}\n\nTarea: {request.prompt}"
            if request.context:
                full_prompt += f"\n\nContexto: {request.context}"
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{full_prompt}"

            self.last_prompt = full_prompt

            # Llamada al LLM
            response_content = await self.ollama_client.generate_async(
                model_name=self.model_name,
                prompt=full_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            response_time = time.time() - start_time
            if response_content:
                response = LLMResponse(
                    content=response_content,
                    success=True,
                    model_name=self.model_name,
                    response_time=response_time,
                    confidence=0.85,
                    tokens_used=len(response_content.split()),
                    metadata={
                        "task_type": request.task_type,
                        "prompt": full_prompt,
                        "context": request.context,
                        "response_time": response_time,
                    },
                )
                self.last_response = response
                logger.info(
                    f"BrainConnector: Respuesta generada en {response_time:.2f}s"
                )
                return response
            else:
                logger.warning("BrainConnector: LLM no retornó respuesta.")
                response = LLMResponse(
                    content="",
                    success=False,
                    model_name=self.model_name,
                    error_message="LLM did not return a response.",
                    confidence=0.0,
                    response_time=response_time,
                    metadata={"prompt": full_prompt, "fallback": False},
                )
                self.last_response = response
                return response
        except Exception as e:
            logger.error(f"BrainConnector error: {e}")
            return await self.graceful_fallback(request)

    async def health_check(self) -> bool:
        """Verifica disponibilidad del modelo cerebro con diagnóstico extendido"""
        try:
            is_available = await self.ollama_client.check_model_availability(
                self.model_name
            )
            self.is_available = is_available
            self.health_status = "healthy" if self.is_available else "unhealthy"
            logger.info(f"BrainConnector health: {self.health_status}")
            return self.is_available
        except Exception as e:
            logger.error(f"Error during health check for {self.model_name}: {e}")
            self.is_available = False
            self.health_status = "unreachable"
            self.last_error = str(e)
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Retorna las capacidades extendidas del LLM"""
        capabilities = {
            "role": "brain",
            "specialties": ["reasoning", "planning", "analysis", "strategy", "review"],
            "task_types": list(self.system_prompts.keys()),
            "max_complexity": "high",
            "supports_json": True,
            "model_name": self.model_name,
            "health_status": self.health_status,
        }
        logger.debug(f"BrainConnector capabilities: {capabilities}")
        return capabilities

    def get_last_prompt(self) -> str:
        """Devuelve el último prompt enviado al LLM"""
        return self.last_prompt

    def get_last_response(self) -> LLMResponse | None:
        """Devuelve la última respuesta generada por el LLM"""
        return self.last_response
