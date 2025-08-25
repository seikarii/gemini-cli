import logging
from typing import Any

from crisalida_lib.HEAVEN.llm.cerebellum_connector import CerebellumConnector
from crisalida_lib.HEAVEN.llm.strategic_llm_connector import BrainConnector

from .base_llm_connector import LLMRequest
from .fallback_manager import LLMFallbackManager
from .llm_health_monitor import LLMHealthMonitor

logger = logging.getLogger(__name__)


class LLMGatewayOrchestrator:
    """
    Orquestador principal para la interacción cerebro-cerebelo.
    Gestiona la coordinación entre LLMs, fallback, diagnóstico y modo offline.
    """

    def __init__(self, ollama_client, default_model: str | None = None) -> None:
        self.ollama_client = ollama_client
        self.default_model = default_model
        self.brain = BrainConnector(ollama_client, model_name=default_model or "")
        self.cerebellum = CerebellumConnector(ollama_client)
        self.health_monitor = LLMHealthMonitor([self.brain, self.cerebellum])
        self.fallback_manager = LLMFallbackManager()
        self.is_llm_mode_enabled = True

    async def process_complex_task(
        self, task_description: str, context: str | None = None
    ) -> dict[str, Any]:
        """
        Procesa tarea compleja usando interacción cerebro-cerebelo.
        Incluye diagnóstico extendido y fallback robusto.
        """
        if not self.is_llm_mode_enabled:
            return await self.fallback_manager.handle_offline_mode(
                task_description, context
            )
        try:
            # Fase 1: Análisis y planificación (Cerebro)
            planning_request = LLMRequest(
                prompt=f"Analiza y planifica esta tarea: {task_description}",
                context=context,
                task_type="planning",
                temperature=0.3,
            )
            brain_response = await self.brain.generate(planning_request)
            if not brain_response.success:
                logger.warning("Cerebro no disponible, usando modo degradado")
                return await self.fallback_manager.handle_brain_failure(
                    task_description, context
                )
            plan = self._extract_plan_from_response(brain_response.content)

            # Fase 2: Implementación paso a paso (Cerebelo)
            implementation_results = []
            for step in plan.get("steps", []):
                impl_request = LLMRequest(
                    prompt=step["description"],
                    context=step.get("context", ""),
                    task_type="implementation",
                    temperature=0.1,
                )
                cerebellum_response = await self.cerebellum.generate(impl_request)
                if cerebellum_response.success:
                    implementation_results.append(
                        {
                            "step": step,
                            "result": cerebellum_response.content,
                            "success": True,
                        }
                    )
                else:
                    fallback_result = (
                        await self.fallback_manager.handle_cerebellum_failure(step)
                    )
                    implementation_results.append(
                        {"step": step, "result": fallback_result, "success": False}
                    )

            # Fase 3: Revisión final (Cerebro)
            review_context = self._compile_implementation_results(
                implementation_results
            )
            review_request = LLMRequest(
                prompt="Revisa los resultados de implementación y proporciona feedback",
                context=review_context,
                task_type="review",
                temperature=0.2,
            )
            review_response = await self.brain.generate(review_request)

            return {
                "success": True,
                "plan": plan,
                "implementation_results": implementation_results,
                "review": (
                    review_response.content
                    if review_response.success
                    else "No review available"
                ),
                "llm_mode": True,
                "brain_available": self.brain.is_available,
                "cerebellum_available": self.cerebellum.is_available,
            }
        except Exception as e:
            logger.error(f"LLM orchestration failed: {e}")
            return await self.fallback_manager.handle_complete_failure(
                task_description, context
            )

    def _extract_plan_from_response(self, response_content: str) -> dict[str, Any]:
        """
        Extrae plan estructurado de la respuesta del cerebro.
        Ahora soporta solo texto plano para la descripción del plan.
        """
        logger.info(f"Extracting plan from response: {response_content[:200]}...")
        return {"steps": [{"description": response_content, "context": ""}]}

    async def enable_offline_mode(self):
        """
        Habilita modo offline - Jano funcionará sin LLMs externos.
        """
        self.is_llm_mode_enabled = False
        logger.info("🔄 LLM mode disabled - Jano operating independently")

    async def enable_llm_mode(self):
        """
        Habilita modo LLM si los modelos están disponibles.
        """
        health_status = await self.health_monitor.check_all()
        if health_status.get("any_available", False):
            self.is_llm_mode_enabled = True
            logger.info("🔄 LLM mode enabled")
        else:
            logger.warning("⚠️ Cannot enable LLM mode - no models available")

    def _compile_implementation_results(self, results: list[dict[str, Any]]) -> str:
        """
        Compila los resultados de la implementación para la revisión del cerebro.
        """
        compiled_text = "Resultados de la implementación:\n"
        for res in results:
            status = "Éxito" if res["success"] else "Fallo"
            result_preview = (
                res["result"][:200] + "..."
                if isinstance(res["result"], str)
                else str(res["result"])
            )
            compiled_text += f"- Paso: {res['step']['description']}\n  Estado: {status}\n  Resultado: {result_preview}\n\n"
        return compiled_text
