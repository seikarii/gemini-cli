"""
Intentionality System (professionalized, definitive)
===================================================

- Hardens LLM calls with retry/backoff.
- Async/sync EVA persistence support (best-effort).
- Improved parameter validation & path normalization.
- Defensive imports / TYPE_CHECKING-aware annotations.
- Preserves public API while filling placeholders and anticipating evolutions.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import (
    AdamConfig,
)  # [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    EVAMemoryManager,
)  # [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
from crisalida_lib.ASTRAL_TOOLS.base import (
    ToolRegistry,
)  # [`crisalida_lib.ASTRAL_TOOLS.base.ToolRegistry`](crisalida_lib/ASTRAL_TOOLS/base.py)
from crisalida_lib.EVA.core_types import (
    QualiaState,
)  # [`crisalida_lib.EVA.core_types.QualiaState`](crisalida_lib/EVA/core_types.py)
from crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator import (
    LLMGatewayOrchestrator,
)  # [`crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator.LLMGatewayOrchestrator`](crisalida_lib/HEAVEN/llm/llm_gateway_orchestrator.py)

if TYPE_CHECKING:
    # avoid runtime import cycles in type-checking contexts
    pass  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class PlanOfAction:
    """A multi-step plan containing tool calls to execute."""

    steps: list[dict[str, Any]]
    confidence: float
    estimated_duration: float
    fallback_plan: PlanOfAction | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("PlanOfAction must contain at least one step")


class IntentionalityEngine:
    """
    Enhanced intentionality engine.

    References:
    - [`crisalida_lib.ADAM.config.AdamConfig`](crisalida_lib/ADAM/config.py)
    - [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)
    - [`crisalida_lib.ASTRAL_TOOLS.base.ToolRegistry`](crisalida_lib/ASTRAL_TOOLS/base.py)
    - [`crisalida_lib.EVA.core_types.QualiaState`](crisalida_lib/EVA/core_types.py)
    - [`crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator.LLMGatewayOrchestrator`](crisalida_lib/HEAVEN/llm/llm_gateway_orchestrator.py)
    """

    DEFAULT_NLU_RETRIES = 2
    DEFAULT_NLU_BACKOFF = 0.5  # seconds

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
        tool_registry: ToolRegistry | None = None,
        llm_gateway: LLMGatewayOrchestrator | None = None,
        model_name: str = "deepseek-r1:1.5b",
    ) -> None:
        self.config = config
        self.llm_gateway = llm_gateway
        self.eva_manager = eva_manager
        self.entity_id = entity_id
        self.tool_registry = tool_registry or self._safe_instantiate_tool_registry()
        self.model_name = model_name
        self.action_archetypes = {
            "help_user": "helper",
            "analyze_data": "analyst",
            "create_content": "creator",
            "explore_information": "explorer",
            "teach_concept": "teacher",
            "protect_system": "protector",
            "question_assumptions": "skeptic",
        }
        logger.info(
            "IntentionalityEngine initialized for entity '%s' (model=%s)",
            self.entity_id,
            self.model_name,
        )

    def _safe_instantiate_tool_registry(self) -> ToolRegistry | None:
        try:
            return ToolRegistry()
        except Exception:
            logger.debug(
                "ToolRegistry instantiation failed; continuing without registry"
            )
            return None

    def _create_nlu_prompt(self, user_prompt: str) -> str:
        registry_info = {}
        try:
            registry_info = (
                self.tool_registry.get_registry_info() if self.tool_registry else {}
            )
        except Exception:
            registry_info = {}
        tools = registry_info.get("tools", {}) if registry_info else {}
        tools_schema_text = "Available Tools and Their Schemas:\n\n"
        for tool_name, tool_info in tools.items():
            tools_schema_text += f"## {tool_name}\n"
            tools_schema_text += (
                f"Description: {tool_info.get('description', 'No description')}\n"
            )
            parameters = tool_info.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = parameters.get("required", []) or []
            if properties:
                tools_schema_text += "Parameters:\n"
                for param_name, param_details in properties.items():
                    param_type = param_details.get("type", "unknown")
                    param_desc = param_details.get("description", "No description")
                    is_required = param_name in required
                    req_marker = " (REQUIRED)" if is_required else " (optional)"
                    tools_schema_text += (
                        f"  - {param_name}: {param_type}{req_marker} - {param_desc}\n"
                    )
            else:
                tools_schema_text += "Parameters: None\n"
            tools_schema_text += "\n"

        nlu_prompt = (
            "You are a function that analyzes user requests and returns a JSON object with the appropriate tool and parameters.\n"
            f"{tools_schema_text}"
            "ANALYSIS INSTRUCTIONS:\n"
            f"1. Carefully analyze the user's request: {repr(user_prompt)}\n"
            "2. Determine which tool (if any) is most appropriate to fulfill this request.\n"
            "3. Extract ALL required parameters from the user's natural language. The parameter names must match the tool's schema exactly.\n"
            "4. For file paths, convert relative paths to absolute paths using OS logic.\n"
            '5. If no tool is appropriate, return "final_response".\n'
            "RESPONSE FORMAT: Return a single JSON object, nothing else.\n"
            'EXAMPLE: {"tool_name":"write_file", "parameters":{"file_path":"/abs/path","content":"hello"}, "confidence":0.9, "reasoning":"..."}\n'
            f"USER REQUEST TO ANALYZE: {repr(user_prompt)}\n"
            "Remember: Respond ONLY with the JSON object.\n"
        )
        return nlu_prompt

    async def decide_next_action(
        self,
        cognitive_state: dict[str, Any],
        qualia_state: QualiaState,
        soul_system: Any,
    ) -> PlanOfAction:
        user_prompt = cognitive_state.get("original_prompt", "")
        if not user_prompt:
            logger.warning("No original_prompt found in cognitive_state")
            return self._create_fallback_plan()
        logger.debug("Analyzing user prompt (truncated): %s", user_prompt[:200])
        try:
            nlu_prompt = self._create_nlu_prompt(user_prompt)
            nlu_response = await self._call_ollama_nlu(nlu_prompt)
            parsed_intent = self._parse_nlu_response(nlu_response)
            plan = await self._create_plan_from_intent(
                parsed_intent, qualia_state, soul_system
            )
            logger.info(
                "Generated plan - Tool: %s (confidence: %.2f)",
                parsed_intent.get("tool_name", "unknown"),
                parsed_intent.get("confidence", 0.0),
            )
            # best-effort EVA persistence (async or sync)
            try:
                if self.eva_manager:
                    record_fn = getattr(self.eva_manager, "record_experience", None)
                    if record_fn:
                        data = {
                            "user_prompt": user_prompt,
                            "parsed_intent": parsed_intent,
                            "plan_steps": plan.steps,
                            "confidence": plan.confidence,
                            "timestamp": time.time(),
                        }
                        if inspect.iscoroutinefunction(record_fn):
                            try:
                                asyncio.create_task(
                                    record_fn(
                                        entity_id=self.entity_id,
                                        event_type="intentionality_decision",
                                        data=data,
                                    )
                                )
                            except RuntimeError:
                                # No loop active; run synchronously
                                asyncio.run(
                                    record_fn(
                                        entity_id=self.entity_id,
                                        event_type="intentionality_decision",
                                        data=data,
                                    )
                                )
                        else:
                            try:
                                record_fn(
                                    entity_id=self.entity_id,
                                    event_type="intentionality_decision",
                                    data=data,
                                )
                            except Exception:
                                logger.debug(
                                    "Synchronous EVA record_experience failed (non-fatal)"
                                )
            except Exception:
                logger.exception("EVA persistence failed (non-fatal)")
            return plan
        except Exception as e:
            logger.exception("Error in decide_next_action: %s", e)
            return self._create_fallback_plan()

    async def _call_ollama_nlu(self, nlu_prompt: str) -> str:
        if not self.llm_gateway:
            logger.error("LLM Gateway not available to IntentionalityEngine.")
            return '{"tool_name": "final_response", "parameters": {}, "confidence": 0.1, "reasoning": "LLM Gateway not configured"}'
        retries = int(getattr(self.config, "NLU_RETRIES", self.DEFAULT_NLU_RETRIES))
        backoff = float(getattr(self.config, "NLU_BACKOFF", self.DEFAULT_NLU_BACKOFF))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await self.llm_gateway.ollama_client.generate_async(
                    prompt=nlu_prompt,
                    model_name=self.model_name,
                    temperature=float(getattr(self.config, "NLU_TEMPERATURE", 0.1)),
                    max_tokens=int(getattr(self.config, "NLU_MAX_TOKENS", 1024)),
                )
                if isinstance(response, str):
                    return response
                # sometimes SDK returns dict-like
                return json.dumps(response)
            except Exception as e:
                last_exc = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, e
                )
                await asyncio.sleep(backoff * (2**attempt))
        logger.exception("LLM NLU failed after retries: %s", last_exc)
        return '{"tool_name": "final_response", "parameters": {}, "confidence": 0.1, "reasoning": "Error during LLM call"}'

    def _parse_nlu_response(self, nlu_response: str) -> dict[str, Any]:
        try:
            response_text = nlu_response.strip()
            if not response_text.startswith("{"):
                json_start = response_text.find("{")
                json_end = response_text.rfind("}")
                if json_start != -1 and json_end != -1:
                    response_text = response_text[json_start : json_end + 1]
                else:
                    raise ValueError("No JSON object found in response")
            parsed_intent = json.loads(response_text)
            if "tool_name" not in parsed_intent:
                raise ValueError("Missing 'tool_name' in NLU response")
            parsed_intent.setdefault("parameters", {})
            parsed_intent.setdefault("confidence", 0.5)
            logger.debug(
                "Parsed NLU intent: %s (%d params)",
                parsed_intent["tool_name"],
                len(parsed_intent["parameters"]),
            )
            return parsed_intent
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from NLU response: %s", e)
            logger.debug("Raw response (truncated): %s", nlu_response[:1000])
            raise ValueError(f"Invalid JSON in NLU response: {e}") from e

    async def _create_plan_from_intent(
        self,
        parsed_intent: dict[str, Any],
        qualia_state: QualiaState,
        soul_system: Any,
    ) -> PlanOfAction:
        tool_name = parsed_intent["tool_name"]
        parameters = parsed_intent.get("parameters", {}) or {}
        confidence = float(parsed_intent.get("confidence", 0.5) or 0.5)
        if tool_name == "final_response":
            steps = [
                {
                    "type": "final_response",
                    "tool": "final_response",
                    "params": {
                        "intent": "conversational_response",
                        "reasoning": parsed_intent.get(
                            "reasoning", "General conversational response"
                        ),
                    },
                }
            ]
        else:
            validated_list = await self._validate_and_fix_tool_parameters(
                [{"type": "tool_execution", "tool": tool_name, "params": parameters}]
            )
            validated_params = (
                validated_list[0].get("params", {}) if validated_list else {}
            )
            steps = [
                {
                    "type": "tool_execution",
                    "tool": tool_name,
                    "params": validated_params,
                }
            ]
        plan_confidence = self._calculate_plan_confidence(
            confidence, qualia_state, soul_system
        )
        estimated_duration = self._estimate_execution_duration(
            len(steps), tool_name, getattr(qualia_state, "arousal", 0.5)
        )
        return PlanOfAction(
            steps=steps,
            confidence=plan_confidence,
            estimated_duration=estimated_duration,
        )

    async def _validate_and_fix_tool_parameters(
        self, steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        validated_steps: list[dict[str, Any]] = []
        for step in steps:
            tool_name = step.get("tool")
            if not tool_name or tool_name == "final_response":
                validated_steps.append(step)
                continue
            current_params = dict(step.get("params", {}) or {})
            fixed_params: dict[str, Any] = {}
            expected_params: dict[str, Any] = {}

            # Try to obtain tool schema from registry if available
            try:
                if self.tool_registry and hasattr(self.tool_registry, "get_tool"):
                    tool_inst = self.tool_registry.get_tool(tool_name)
                    # Accept either a BaseTool-like object with .schema or callable signature
                    if tool_inst is not None:
                        schema = (
                            getattr(tool_inst, "schema", None)
                            or getattr(tool_inst, "get_schema", lambda: {})()
                        )
                        properties = (schema or {}).get("properties", {}) or {}
                        expected_params = {
                            k: v.get("type", Any) for k, v in properties.items()
                        }
            except Exception:
                logger.debug(
                    "Tool schema extraction failed for %s (continuing with heuristics)",
                    tool_name,
                )

            # Conservative param name mappings (backwards compatibility)
            param_mappings = {
                "filename": ["absolute_path", "file_path", "path"],
                "file_path": ["absolute_path", "filename", "path"],
                "absolute_path": ["filename", "file_path", "path"],
                "path": ["absolute_path", "file_path", "filename"],
                "url": ["prompt"],
                "query": ["prompt", "search_query"],
                "prompt": ["url", "query"],
                "search_term": ["pattern", "query"],
                "pattern": ["search_term", "query"],
                "content": ["text", "data"],
                "text": ["content", "data"],
            }

            # First, fill explicit matches
            for expected in expected_params:
                if expected in current_params:
                    fixed_params[expected] = current_params[expected]
                else:
                    # try mapped names
                    for cur_name, val in current_params.items():
                        if expected in param_mappings.get(cur_name, []):
                            fixed_params[expected] = val
                            break
                    else:
                        for alt in param_mappings.get(expected, []):
                            if alt in current_params:
                                fixed_params[expected] = current_params[alt]
                                break

            # If no expected params detected (no schema), preserve user params but normalize paths
            if not expected_params:
                fixed_params = dict(current_params)

            # Normalize file-like parameters to absolute paths
            for k, v in list(fixed_params.items()):
                if (
                    k in {"path", "absolute_path", "file_path", "filename"}
                    and isinstance(v, str)
                    and v
                ):
                    try:
                        fixed_params[k] = os.path.abspath(os.path.expanduser(v))
                    except Exception:
                        pass

            # Fill defaults if still missing
            for expected, ptype in expected_params.items():
                if expected not in fixed_params:
                    default_value = self._get_default_parameter_value(
                        expected, ptype, step
                    )
                    if default_value is not None:
                        fixed_params[expected] = default_value

            step["params"] = fixed_params
            validated_steps.append(step)
            logger.debug(
                "Validated params for %s: %s", tool_name, list(fixed_params.keys())
            )
        return validated_steps

    def _get_default_parameter_value(
        self, param_name: str, param_type: Any, step: dict[str, Any]
    ) -> Any:
        # Conservative defaults for common parameters
        if param_name in {"path", "absolute_path", "file_path"}:
            if "filename" in str(step).lower():
                return os.path.abspath("example.txt")
            return os.getcwd()
        if param_name == "prompt" and step.get("tool") == "web_fetch":
            return "https://httpbin.org/json"
        if param_name in {"query", "prompt"} and "search" in step.get("tool", ""):
            return "python programming examples"
        if param_name == "pattern":
            return "*.py"
        if param_name == "content":
            return "Generated content based on user request.\n"
        if param_name == "command":
            return 'echo "Hello World"'
        if param_type is str or "str" in str(param_type):
            return ""
        if (
            param_type in [int, float]
            or "int" in str(param_type)
            or "float" in str(param_type)
        ):
            return 0
        if param_type is bool or "bool" in str(param_type):
            return False
        if "list" in str(param_type) or "List" in str(param_type):
            return []
        return None

    def _create_fallback_plan(self) -> PlanOfAction:
        return PlanOfAction(
            steps=[
                {
                    "type": "final_response",
                    "tool": "final_response",
                    "params": {
                        "intent": "conversational_response",
                        "reasoning": "General conversational response",
                    },
                }
            ],
            confidence=0.3,
            estimated_duration=1.0,
        )

    def _calculate_plan_confidence(
        self, nlu_confidence: float, qualia_state: QualiaState, soul_system: Any
    ) -> float:
        base_confidence = float(nlu_confidence)
        coherence_bonus = float(getattr(qualia_state, "temporal_coherence", 0.5)) * 0.1
        focus_bonus = float(getattr(qualia_state, "cognitive_focus", 0.5)) * 0.1
        uncertainty_penalty = (
            0.1
            if float(getattr(qualia_state, "cognitive_complexity", 0.5)) > 0.8
            else 0.0
        )
        final_confidence = (
            base_confidence + coherence_bonus + focus_bonus - uncertainty_penalty
        )
        return max(0.1, min(1.0, final_confidence))

    def _estimate_execution_duration(
        self, step_count: int, tool_name: str, arousal: float
    ) -> float:
        tool_durations = {
            "final_response": 0.5,
            "read_file": 1.0,
            "write_file": 1.5,
            "list_directory": 1.0,
            "glob": 2.0,
            "search_file_content": 3.0,
            "run_shell_command": 2.0,
            "google_web_search": 4.0,
            "web_fetch": 3.0,
        }
        base_duration = float(tool_durations.get(tool_name, 2.0)) * max(
            1, int(step_count)
        )
        if arousal > 0.7:
            arousal_factor = 1.1
        elif arousal > 0.4:
            arousal_factor = 0.9
        else:
            arousal_factor = 1.0
        return base_duration * arousal_factor
