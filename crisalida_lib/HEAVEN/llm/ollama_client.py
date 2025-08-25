"""
EnhancedOllamaClient - Cliente avanzado para integración con Ollama LLMs
=========================================================================

Proporciona interacción asincrónica y robusta con modelos Ollama locales/remotos.
Incluye diagnóstico extendido, manejo de errores, caché, gestión de modelos y trazabilidad.
Listo para integración con sistemas de orquestación, fallback y monitorización.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Cliente avanzado para interacción y gestión de modelos Ollama.
    Soporta generación, chat, listado, pull, diagnóstico y caché.
    """

    def __init__(self, base_url: str = "http://localhost:11434/api"):
        self.base_generate_url = f"{base_url}/generate"
        self.base_chat_url = f"{base_url}/chat"
        self.base_tags_url = f"{base_url}/tags"
        self.base_pull_url = f"{base_url}/pull"
        self.connection_pool = None  # Puede usarse aiohttp.ClientSession persistente
        self.model_cache: dict[str, Any] = {}
        self.last_health_check: dict[str, Any] = {}
        self.last_error: str | None = None
        logger.info(f"OllamaClient initialized with base URL: {base_url}")

    async def generate_async(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 300,
    ) -> str | None:
        """
        Generación asincrónica de texto usando Ollama.
        """
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(
                    self.base_generate_url, json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("response")
        except Exception as e:
            logger.error(f"Error generating with model {model_name}: {e}")
            self.last_error = str(e)
            return None

    async def chat_async(
        self,
        model_name: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 300,
    ) -> str | None:
        """
        Interacción asincrónica tipo chat con Ollama.
        """
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(self.base_chat_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("response")
        except Exception as e:
            logger.error(f"Error in chat with model {model_name}: {e}")
            self.last_error = str(e)
            return None

    async def list_models(self) -> list[dict]:
        """
        Lista todos los modelos disponibles en Ollama.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(self.base_tags_url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("models", [])
        except Exception as e:
            logger.error(f"Error listing models from Ollama: {e}")
            self.last_error = str(e)
            return []

    async def check_model_availability(self, model_name: str) -> bool:
        """
        Verifica si un modelo específico está disponible en Ollama.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(self.base_tags_url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    available_models = [
                        model["name"] for model in data.get("models", [])
                    ]
                    is_available = model_name in available_models
                    self.last_health_check[model_name] = {
                        "available": is_available,
                        "timestamp": time.time(),
                    }
                    return is_available
        except Exception as e:
            logger.error(f"Error checking model availability for {model_name}: {e}")
            self.last_health_check[model_name] = {
                "available": False,
                "timestamp": time.time(),
                "error": str(e),
            }
            self.last_error = str(e)
            return False

    def _check_model_availability_sync(self, model_name: str) -> bool:
        """
        Verificación síncrona de disponibilidad de modelo (fallback).
        """
        try:
            import requests

            response = requests.get(self.base_tags_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
            is_available = model_name in available_models
            self.last_health_check[model_name] = {
                "available": is_available,
                "timestamp": time.time(),
            }
            return is_available
        except Exception as e:
            logger.error(f"Synchronous health check for {model_name} failed: {e}")
            self.last_error = str(e)
            return False

    async def pull_model_if_needed(self, model_name: str) -> bool:
        """
        Descarga un modelo desde Ollama si no está disponible.
        """
        if await self.check_model_availability(model_name):
            return True
        logger.info(f"Model {model_name} not found, attempting to pull...")
        try:
            import aiohttp

            payload = {"name": model_name}
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            ) as session:
                async with session.post(self.base_pull_url, json=payload) as response:
                    # Consume el stream para completar el pull
                    async for _ in response.content.iter_any():
                        pass
                    logger.info(f"Successfully pulled model {model_name}")
                    return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            self.last_error = str(e)
            return False

    def get_health_summary(self) -> dict[str, Any]:
        """
        Devuelve resumen de salud de los modelos monitoreados.
        """
        last_update = (
            max(
                (data["timestamp"] for data in self.last_health_check.values()),
                default=0,
            )
            if self.last_health_check
            else 0
        )
        return {
            "models_checked": list(self.last_health_check.keys()),
            "health_data": self.last_health_check.copy(),
            "last_update": last_update,
            "last_error": self.last_error,
        }

    def clear_cache(self):
        """
        Limpia la caché interna de modelos.
        """
        self.model_cache.clear()
        logger.info("Model cache cleared.")

    def get_cached_models(self) -> list:
        """
        Devuelve lista de modelos en caché.
        """
        return list(self.model_cache.keys())

    async def ping(self) -> bool:
        """
        Verifica conectividad básica con Ollama (health check rápido).
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(self.base_tags_url) as response:
                    response.raise_for_status()
                    return True
        except Exception as e:
            logger.error(f"Ollama ping failed: {e}")
            self.last_error = str(e)
            return False
