"""
BaseLLMConnector - Interfaz estándar para conectores LLM en Crisalida
====================================================================

Define la clase abstracta `BaseLLMConnector` para integración de modelos LLM.
Incluye diagnóstico extendido, manejo robusto de errores, logging y trazabilidad.
Permite extensibilidad y consistencia entre diferentes backends LLM.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMRole(Enum):
    BRAIN = "brain"  # Razonamiento de alto nivel
    CEREBELLUM = "cerebellum"  # Tareas granulares/código
    GENERAL = "general"  # Uso general


@dataclass
class LLMRequest:
    prompt: str
    context: str | None = None
    task_type: str = "general"
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    success: bool
    model_name: str
    tokens_used: int = 0
    response_time: float = 0.0
    error_message: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractLLMConnector(ABC):
    """
    Clase base abstracta para conectores LLM.
    Provee diagnóstico, logging y manejo robusto de errores.
    """

    def __init__(self, model_name: str, role: LLMRole, ollama_client: OllamaClient):
        self.model_name = model_name
        self.role = role
        self.ollama_client = ollama_client
        self.is_available = False
        self.health_status = "unknown"
        self.last_error: str | None = None
        self.last_response: LLMResponse | None = None

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Genera respuesta usando el LLM.
        Debe ser implementado por cada conector específico.
        """
        pass

    async def health_check(self) -> bool:
        """
        Verifica si el LLM está disponible.
        Actualiza el estado interno y retorna True/False.
        """
        try:
            status = await self.ollama_client.ping()
            self.is_available = status
            self.health_status = "healthy" if status else "unavailable"
            logger.info(f"LLM {self.model_name} health: {self.health_status}")
            return status
        except Exception as e:
            self.is_available = False
            self.health_status = "error"
            self.last_error = str(e)
            logger.error(f"LLM {self.model_name} health check error: {e}")
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """
        Retorna las capacidades del LLM.
        Puede ser extendido por conectores concretos.
        """
        capabilities = {
            "model_name": self.model_name,
            "role": self.role.value,
            "max_tokens": 4096,
            "supports_streaming": hasattr(self.ollama_client, "stream"),
            "health_status": self.health_status,
        }
        logger.debug(f"LLM {self.model_name} capabilities: {capabilities}")
        return capabilities

    async def graceful_fallback(self, request: LLMRequest) -> LLMResponse:
        """
        Proporciona respuesta de fallback cuando LLM no disponible.
        Incluye diagnóstico y logging.
        """
        logger.warning(f"LLM {self.model_name} no disponible. Usando fallback.")
        response = LLMResponse(
            content=f"LLM {self.model_name} no disponible. Usando lógica interna.",
            success=False,
            model_name=self.model_name,
            error_message="LLM unavailable - graceful degradation",
            confidence=0.1,
            response_time=0.01,
            metadata={"fallback": True, "prompt": request.prompt},
        )
        self.last_response = response
        return response

    def get_last_error(self) -> str | None:
        """Devuelve el último error registrado."""
        return self.last_error

    def get_last_response(self) -> LLMResponse | None:
        """Devuelve la última respuesta registrada."""
        return self.last_response
