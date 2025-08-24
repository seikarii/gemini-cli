import asyncio
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - sub-agent delegation will be limited")


class SubAgentTask(BaseModel):
    task_id: str = Field(..., description="Identificador único de la tarea")
    prompt: str = Field(..., description="Prompt para el sub-agente")
    model: str = Field(
        "llama2", description="Modelo a usar (ej: llama2, mistral, phi3)"
    )
    max_tokens: int = Field(500, description="Máximo de tokens en la respuesta")
    temperature: float = Field(0.7, description="Temperatura de generación")
    timeout: int = Field(30, description="Timeout en segundos")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadatos adicionales"
    )


class SubLLMParameters(BaseModel):
    action: Literal[
        "delegate_task",
        "parallel_tasks",
        "list_models",
        "test_model",
        "orchestrate_workflow",
        "aggregate_results",
    ] = Field(..., description="Acción a realizar")
    task: SubAgentTask | None = Field(None, description="Tarea única a delegar")
    tasks: list[SubAgentTask] | None = Field(
        None, description="Lista de tareas para ejecución paralela/secuencial"
    )
    model_name: str | None = Field(
        None, description="Nombre del modelo para pruebas/info"
    )
    ollama_url: str = Field(
        "http://localhost:11434", description="URL del servidor Ollama"
    )
    workflow_type: Literal["sequential", "parallel", "pipeline"] = Field(
        "parallel", description="Tipo de flujo de trabajo"
    )
    aggregation_method: Literal["concatenate", "summarize", "analyze", "vote"] = Field(
        "concatenate", description="Método de agregación"
    )
    aggregation_prompt: str | None = Field(
        None, description="Prompt personalizado para agregación"
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        valid_actions = [
            "delegate_task",
            "parallel_tasks",
            "list_models",
            "test_model",
            "orchestrate_workflow",
            "aggregate_results",
        ]
        if v not in valid_actions:
            raise ValueError(f"Acción inválida: {v}")
        return v


class SubLLMAgentTool(BaseTool):
    """
    Herramienta para delegar tareas a sub-modelos LLM especializados.
    """

    def __init__(self):
        super().__init__()
        self._client = httpx.AsyncClient() if HTTPX_AVAILABLE else None
        self._model_cache = {}
        self._task_results = {}

    def _get_name(self) -> str:
        return "sub_llm_agent"

    def _get_description(self) -> str:
        return "Delegación inteligente de tareas a sub-modelos LLM especializados (Ollama, Mistral, etc.)"

    def _get_category(self) -> str:
        return "llm_delegation"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return SubLLMParameters

    async def _call_ollama_api(
        self,
        endpoint: str,
        data: dict[str, Any],
        base_url: str,
    ) -> dict[str, Any]:
        if not self._client:
            raise Exception("httpx no disponible - no se pueden hacer llamadas API")
        try:
            response = await self._client.post(
                f"{base_url}/api/{endpoint}", json=data, timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise Exception(
                f"Error de conexión con Ollama en {base_url}: {str(e)}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"Error Ollama API: {e.response.status_code} - {e.response.text}"
            ) from e

    async def _generate_response(
        self,
        task: SubAgentTask,
        base_url: str,
    ) -> dict[str, Any]:
        request_data = {
            "model": task.model,
            "prompt": task.prompt,
            "stream": False,
            "options": {
                "num_predict": task.max_tokens,
                "temperature": task.temperature,
            },
        }
        start_time = time.time()
        try:
            timeout = httpx.Timeout(task.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/api/generate", json=request_data
                )
                response.raise_for_status()
                result = response.json()
                execution_time = time.time() - start_time
                return {
                    "success": True,
                    "task_id": task.task_id,
                    "model": task.model,
                    "response": result.get("response", ""),
                    "execution_time": execution_time,
                    "tokens_generated": len(result.get("response", "").split()),
                    "metadata": task.metadata,
                }
        except TimeoutError:
            return {
                "success": False,
                "task_id": task.task_id,
                "error": f"Timeout tras {task.timeout} segundos",
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            return {
                "success": False,
                "task_id": task.task_id,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    async def _list_available_models(self, base_url: str) -> list[dict[str, Any]]:
        try:
            result = await self._call_ollama_api("tags", {}, base_url)
            return result.get("models", [])
        except Exception as e:
            logger.warning(f"Error al listar modelos: {e}")
            return []

    async def _test_model(self, model_name: str, base_url: str) -> dict[str, Any]:
        test_task = SubAgentTask(
            task_id="test",
            prompt="Hello, please respond with 'Model is working' to confirm you are functioning correctly.",
            model=model_name,
            max_tokens=50,
            temperature=0.1,
            timeout=15,
        )
        result = await self._generate_response(test_task, base_url)
        if result["success"]:
            response_text = result["response"].lower()
            is_working = any(
                phrase in response_text
                for phrase in ["working", "functioning", "hello"]
            )
            return {
                "model": model_name,
                "available": True,
                "working": is_working,
                "response": result["response"],
                "execution_time": result["execution_time"],
            }
        else:
            return {
                "model": model_name,
                "available": False,
                "working": False,
                "error": result["error"],
            }

    async def _execute_parallel_tasks(
        self,
        tasks: list[SubAgentTask],
        base_url: str,
    ) -> list[dict[str, Any]]:
        coroutines = [self._generate_response(task, base_url) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        processed_results: list[dict[str, Any]] = []  # Added type hint
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "task_id": tasks[i].task_id,
                        "error": str(result),
                        "execution_time": 0,
                    }
                )
            elif isinstance(result, dict):
                processed_results.append(result)
        return processed_results

    async def _execute_sequential_tasks(
        self,
        tasks: list[SubAgentTask],
        base_url: str,
    ) -> list[dict[str, Any]]:
        results = []
        context = ""
        for task in tasks:
            if context and "{{PREVIOUS_RESULTS}}" in task.prompt:
                task.prompt = task.prompt.replace("{{PREVIOUS_RESULTS}}", context)
            result = await self._generate_response(task, base_url)
            results.append(result)
            if result["success"]:
                context += f"\nTask {task.task_id}: {result['response']}\n"
        return results

    async def _execute_pipeline_tasks(
        self,
        tasks: list[SubAgentTask],
        base_url: str,
    ) -> list[dict[str, Any]]:
        results = []
        current_output = ""
        for i, task in enumerate(tasks):
            if i > 0 and current_output:
                if "{{INPUT}}" in task.prompt:
                    task.prompt = task.prompt.replace("{{INPUT}}", current_output)
                else:
                    task.prompt = f"Previous output: {current_output}\n\n{task.prompt}"
            result = await self._generate_response(task, base_url)
            results.append(result)
            if result["success"]:
                current_output = result["response"]
            else:
                break
        return results

    def _aggregate_results(
        self,
        results: list[dict[str, Any]],
        method: str,
        custom_prompt: str | None = None,
    ) -> str:
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return "No hay resultados exitosos para agregar."
        if method == "concatenate":
            output = "Resultados agregados:\n\n"
            for result in successful_results:
                output += f"Task {result['task_id']} ({result['model']}):\n"
                output += f"{result['response']}\n\n"
            return output
        elif method == "summarize":
            output = "Resumen de resultados:\n\n"
            output += f"Tareas totales: {len(results)}\n"
            output += f"Exitosas: {len(successful_results)}\n"
            output += f"Fallidas: {len(results) - len(successful_results)}\n\n"
            for result in successful_results:
                preview = result["response"][:100]
                if len(result["response"]) > 100:
                    preview += "..."
                output += f"Task {result['task_id']}: {preview}\n"
            return output
        elif method == "analyze":
            output = "Análisis de resultados:\n\n"
            total_tokens = sum(r.get("tokens_generated", 0) for r in successful_results)
            avg_time = sum(
                r.get("execution_time", 0) for r in successful_results
            ) / len(successful_results)
            output += "Métricas de rendimiento:\n"
            output += f"  Tokens generados: {total_tokens}\n"
            output += f"  Tiempo promedio: {avg_time:.2f}s\n"
            output += f"  Tasa de éxito: {len(successful_results)} / {len(results)} ({len(successful_results) / len(results) * 100:.1f}%)\n\n"
            models_used = list({r["model"] for r in successful_results})
            output += f"Modelos usados: {', '.join(models_used)}\n\n"
            output += "Resumen de contenido:\n"
            for result in successful_results:
                word_count = len(result["response"].split())
                output += f"  Task {result['task_id']}: {word_count} palabras\n"
            return output
        elif method == "vote":
            output = "Análisis por votación:\n\n"
            responses = [r["response"] for r in successful_results]
            if custom_prompt:
                output += f"Criterio de votación: {custom_prompt}\n\n"
            output += "Respuestas:\n"
            for i, result in enumerate(successful_results, 1):
                output += (
                    f"{i}. Task {result['task_id']}: {result['response'][:150]}...\n"
                )
            if len(responses) > 1:
                output += f"\nTotal de respuestas: {len(responses)}\n"
                output += "Nota: El análisis avanzado de votación requiere comparación semántica."
            return output
        else:
            return f"Método de agregación desconocido: {method}"

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = asyncio.get_event_loop().time()
        try:
            params = SubLLMParameters(**kwargs)
            if params.action == "list_models":
                models = await self._list_available_models(params.ollama_url)
                if models:
                    output = f"Modelos disponibles ({len(models)}):\n\n"
                    for model in models:
                        name = model.get("name", "Desconocido")
                        size = model.get("size", 0)
                        size_mb = size / (1024 * 1024) if size else 0
                        output += f"  • {name}"
                        if size_mb > 0:
                            output += f" ({size_mb:.1f} MB)"
                        output += "\n"
                else:
                    output = "No se encontraron modelos o fallo de conexión con Ollama."
                return ToolCallResult(
                    command="sub_llm_agent list_models",
                    success=True,
                    output=output,
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                    error_message=None,  # Added this line
                )
            elif params.action == "test_model":
                if not params.model_name:
                    return ToolCallResult(
                        command="sub_llm_agent",
                        success=False,
                        output="",
                        error_message="model_name es requerido para test_model",
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
                test_result = await self._test_model(
                    params.model_name, params.ollama_url
                )
                output = "Resultados de prueba de modelo:\n\n"
                output += f"Modelo: {test_result['model']}\n"
                output += f"Disponible: {'✅' if test_result['available'] else '❌'}\n"
                if test_result["available"]:
                    output += f"Funciona: {'✅' if test_result['working'] else '❌'}\n"
                    output += f"Tiempo: {test_result.get('execution_time', 0):.2f}s\n"
                    output += (
                        f"Respuesta: {test_result.get('response', 'Sin respuesta')}\n"
                    )
                else:
                    output += (
                        f"Error: {test_result.get('error', 'Error desconocido')}\n"
                    )
                return ToolCallResult(
                    command="sub_llm_agent test_model",
                    success=test_result["available"] and test_result["working"],
                    output=output,
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                    error_message=test_result.get("error", None),  # Added this line
                )
            elif params.action == "delegate_task":
                if not params.task:
                    return ToolCallResult(
                        command="sub_llm_agent",
                        success=False,
                        output="",
                        error_message="task es requerido para delegate_task",
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
                result = await self._generate_response(params.task, params.ollama_url)
                if result["success"]:
                    output = "✅ Tarea completada!\n\n"
                    output += f"Task ID: {result['task_id']}\n"
                    output += f"Modelo: {result['model']}\n"
                    output += f"Tiempo: {result['execution_time']:.2f}s\n"
                    output += f"Tokens generados: {result['tokens_generated']}\n\n"
                    output += f"Respuesta:\n{result['response']}"
                    return ToolCallResult(
                        command="sub_llm_agent delegate_task",
                        success=True,
                        output=output,
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                        error_message=None,  # Added this line
                    )
                else:
                    output = "❌ Tarea fallida!\n\n"
                    output += f"Task ID: {result['task_id']}\n"
                    output += f"Error: {result['error']}\n"
                    output += f"Tiempo: {result['execution_time']:.2f}s"
                    return ToolCallResult(
                        command="sub_llm_agent delegate_task",
                        success=False,
                        output=output,
                        error_message=result["error"],
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
            elif params.action == "parallel_tasks":
                if not params.tasks:
                    return ToolCallResult(
                        command="sub_llm_agent",
                        success=False,
                        output="",
                        error_message="tasks es requerido para parallel_tasks",
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
                results = await self._execute_parallel_tasks(
                    params.tasks, params.ollama_url
                )
                successful = sum(1 for r in results if r.get("success", False))
                failed = len(results) - successful
                output = "📊 Resultados de tareas paralelas:\n\n"
                output += f"Tareas totales: {len(results)}\n"
                output += f"Exitosas: {successful}\n"
                output += f"Fallidas: {failed}\n\n"
                for result in results:
                    status = "✅" if result.get("success", False) else "❌"
                    output += f"{status} Task {result.get('task_id', '?')}: "
                    if result.get("success", False):
                        response_preview = result.get("response", "")[:100]
                        if len(result.get("response", "")) > 100:
                            response_preview += "..."
                        output += f"{response_preview}\n"
                    else:
                        output += (
                            f"Error - {result.get('error', 'Error desconocido')}\n"
                        )
                self._task_results[f"parallel_{time.time()}"] = results
                return ToolCallResult(
                    command="sub_llm_agent parallel_tasks",
                    success=True,
                    output=output,
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                    error_message=None,  # Added this line
                )
            elif params.action == "orchestrate_workflow":
                if not params.tasks:
                    return ToolCallResult(
                        command="sub_llm_agent",
                        success=False,
                        output="",
                        error_message="tasks es requerido para orchestrate_workflow",
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
                if params.workflow_type == "sequential":
                    results = await self._execute_sequential_tasks(
                        params.tasks, params.ollama_url
                    )
                elif params.workflow_type == "pipeline":
                    results = await self._execute_pipeline_tasks(
                        params.tasks, params.ollama_url
                    )
                else:
                    results = await self._execute_parallel_tasks(
                        params.tasks, params.ollama_url
                    )
                successful = sum(1 for r in results if r.get("success", False))
                output = f"🔄 Resultados del workflow ({params.workflow_type}):\n\n"
                output += f"Tareas completadas: {successful}/{len(results)}\n\n"
                for i, result in enumerate(results, 1):
                    status = "✅" if result.get("success", False) else "❌"
                    output += f"Step {i} {status}: Task {result.get('task_id', '?')}\n"
                    if result.get("success", False):
                        output += f"  Modelo: {result.get('model', '?')}\n"
                        output += f"  Tiempo: {result.get('execution_time', 0):.2f}s\n"
                        response_preview = result.get("response", "")[:150]
                        if len(result.get("response", "")) > 150:
                            response_preview += "..."
                        output += (
                            "  Resultado: " + response_preview + "\n"
                        )  # Changed this line
                    else:
                        output += (
                            f"  Error: {result.get('error', 'Error desconocido')}\n"
                        )
                    output += "\n"
                self._task_results[f"workflow_{time.time()}"] = results
                return ToolCallResult(
                    command="sub_llm_agent orchestrate_workflow",
                    success=True,
                    output=output,
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                    error_message=None,  # Added this line
                )
            elif params.action == "aggregate_results":
                if not self._task_results:
                    return ToolCallResult(
                        command="sub_llm_agent",
                        success=False,
                        output="",
                        error_message="No hay resultados previos para agregar",
                        execution_time=(
                            asyncio.get_event_loop().time() - start_time
                        ),  # Added this line
                    )
                latest_key = max(self._task_results.keys())
                results = self._task_results[latest_key]
                output = self._aggregate_results(
                    results, params.aggregation_method, params.aggregation_prompt
                )
                return ToolCallResult(
                    command="sub_llm_agent aggregate_results",
                    success=True,
                    output=output,
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                    error_message=None,  # Added this line
                )
            else:
                return ToolCallResult(
                    command="sub_llm_agent",
                    success=False,
                    output="",
                    error_message=f"Acción desconocida: {params.action}",
                    execution_time=(
                        asyncio.get_event_loop().time() - start_time
                    ),  # Added this line
                )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Sub-LLM agent error: {e}")
            return ToolCallResult(
                command="sub_llm_agent",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def demo(self):
        """Demonstrate the sub LLM agent tool's functionality."""
        print("🤖 SUB LLM AGENT TOOL DEMO")
        print("=" * 40)

        # List available models
        result = await self.execute(action="list_models")
        print(f"List models: {result.success}")
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Test a common model (usually available in most Ollama installations)
        result = await self.execute(action="test_model", model_name="llama2")
        print(f"\nTest model: {result.success}")
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        print("\n✅ Sub LLM agent demo completed!")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.sub_llm_agent_demos import demo_sub_llm_agent

    asyncio.run(demo_sub_llm_agent())
