"""
Agent MEW (Master Executor-Worker) - Un Agente Orquestador Consciente
======================================================================

Agente avanzado que fusiona los patrones de BabelAgent y PrometheusAgent.
Utiliza el LLMGatewayOrchestrator para la planificación, el ToolRegistry
para la ejecución de herramientas, y se integra con el sistema de memoria
viviente EVA para el aprendizaje y la evolución.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np

from crisalida_lib.ADAM.adam import Adam
from crisalida_lib.ADAM.config import AdamConfig, EVAConfig
from crisalida_lib.ADAM.mente.mind_core import PlanDeAccion
from crisalida_lib.ASTRAL_TOOLS.file_system import GlobTool
from crisalida_lib.ASTRAL_TOOLS.registration_manager import get_global_tool_registry
from crisalida_lib.ASTRAL_TOOLS.validation_tools import LinterTool, TypeCheckerTool
from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
from crisalida_lib.HEAVEN.agents.core.action_system import ActionSystem
from crisalida_lib.HEAVEN.agents.detection.error_patterns import (
    ErrorPattern,
    FixStrategy,
    error_pattern_matcher,
)
from crisalida_lib.HEAVEN.agents.fixing.fix_generators import FixGeneratorFactory
from crisalida_lib.HEAVEN.agents.monitoring.mew_monitor import MewMonitor
from crisalida_lib.HEAVEN.llm.llm_gateway_orchestrator import LLMGatewayOrchestrator
from crisalida_lib.HEAVEN.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class OperationalMode(Enum):
    DIRECTED = "directed"
    AUTONOMOUS = "autonomous"


class FixResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIALLY_FIXED = "partially_fixed"
    NEW_ERRORS_INTRODUCED = "new_errors_introduced"
    UNFIXABLE = "unfixable"


@dataclass
class ErrorFixAttempt:
    error: dict[str, Any]
    pattern: ErrorPattern | None
    fix_generated: dict[str, Any] | None
    result: FixResult
    execution_time: float
    iterations_used: int
    description: str
    is_llm_fix: bool = False


@dataclass
class FileFixSession:
    file_path: str
    original_errors: list[dict[str, Any]]
    fix_attempts: list[ErrorFixAttempt]
    final_errors: list[dict[str, Any]]
    total_time: float
    success_rate: float
    session_id: str = field(
        default_factory=lambda: f"{int(time.time())}-{hash(time.time())}"
    )


def _get_embedding(data: Any) -> np.ndarray:
    return np.array([0.1, 0.2, 0.3])


def _recall_fn(cue: Any) -> tuple[np.ndarray, list[str]]:
    return np.array([0.4, 0.5, 0.6]), []


def _ingest_fn(data: Any, **kwargs) -> str:
    logger.info(f"[Placeholder Ingest] Data: {data}")
    return "dummy_ingest_id"


def _emit_event(event_type: str, payload: dict[str, Any]):
    logger.info(f"[Placeholder Event] Type: {event_type}, Payload: {payload}")


class AgentMew(EVAMemoryMixin):
    def __init__(
        self,
        config: EVAConfig | None = None,
        ollama_client: OllamaClient | None = None,
        llm_model: str | None = None,
        start_monitor: bool = False,
    ):
        logger.info("Initializing AgentMew...")
        self.config = config or EVAConfig()
        self.adam_config = AdamConfig()
        _ollama_client = ollama_client or OllamaClient()
        self.llm_gateway = LLMGatewayOrchestrator(
            _ollama_client, default_model=llm_model
        )
        self.llm_model = llm_model
        self.tool_registry = get_global_tool_registry()
        self.action_system = ActionSystem()
        for tool_name, tool_fn in self.tool_registry.get_all_tools().items():
            self.action_system.register_tool(tool_name, tool_fn.execute)
        self.action_system.start()
        self.adam = Adam(
            config=self.adam_config,
            entity_id="agent_mew_adam",
            tool_registry=self.tool_registry,
            get_embedding=_get_embedding,
            recall_fn=_recall_fn,
            ingest_fn=_ingest_fn,
            emit_event=_emit_event,
            llm_gateway=self.llm_gateway,
        )
        self.linter_tool = LinterTool()
        self.type_checker_tool = TypeCheckerTool()
        self.glob_tool = GlobTool()
        self.pattern_matcher = error_pattern_matcher
        self.fix_factory = FixGeneratorFactory()
        self.active_sessions: dict[str, FileFixSession] = {}
        self.session_counter = 0
        self.stats = {
            "files_processed": 0,
            "errors_fixed": 0,
            "errors_attempted": 0,
            "llm_fixes": 0,
            "average_fix_time": 0.0,
            "success_rate": 0.0,
            "autonomous_cycles": 0,
            "new_patterns_learned": 0,
            "partial_fixes": 0,
            "new_errors_introduced": 0,
        }
        self.max_iterations_per_error = 4
        self.max_total_iterations = 15
        self.confidence_threshold = 0.25
        self.current_mission: str | None = None
        self.mission_context: dict[str, Any] = {}

        # Initialize monitor lazily to avoid import-time side effects.
        # Use start_monitor=True to create and start it immediately.
        self.monitor = None
        if start_monitor:
            self.start_monitor()

        logger.info("✅ AgentMew initialized successfully.")

    async def shutdown(self):
        """Gracefully shuts down the agent and its components."""
        logger.info("Shutting down AgentMew...")
        if self.monitor:
            await self.stop_monitor()
        if self.action_system:
            self.action_system.stop()
        logger.info("✅ AgentMew shutdown complete.")

    def start_monitor(self):
        """Create and start the internal MewMonitor if not already running."""
        if self.monitor is None:
            try:
                self.monitor = MewMonitor(self)
                self.monitor.start()
            except Exception:
                logger.exception("Failed to start MewMonitor")

    async def stop_monitor(self):
        """Stop and cleanup the monitor if it exists."""
        if self.monitor is not None:
            try:
                await self.monitor.stop()
            except Exception:
                logger.exception("Failed to stop MewMonitor")
            finally:
                self.monitor = None

    def get_status(self) -> dict[str, Any]:
        """Returns the current status report from the monitor."""
        if self.monitor:
            return self.monitor.get_status_report()
        return {"status": "error", "message": "Monitor not initialized."}

    async def fix_file(
        self,
        file_path: str,
        error_source: str = "both",
    ) -> FileFixSession:
        session_id = f"bugfix_session_{self.session_counter}"
        self.session_counter += 1
        start_time = time.time()
        logger.info(f"🐞 Starting error fixing session {session_id} for: {file_path}")
        original_errors = await self._detect_errors(file_path, error_source)
        logger.info(f"📊 Found {len(original_errors)} initial errors")
        fixable_errors = self._filter_fixable_errors(original_errors)
        prioritized_errors = self.pattern_matcher.prioritize_errors(fixable_errors)
        logger.info(
            f"🔧 {len(fixable_errors)} errors are fixable, {len(prioritized_errors)} prioritized"
        )
        fix_attempts: list[ErrorFixAttempt] = []
        iterations_used = 0
        for error in prioritized_errors:
            if iterations_used >= self.max_total_iterations:
                logger.warning(
                    f"⏰ Reached maximum iterations ({self.max_total_iterations})"
                )
                break
            attempt = await self._attempt_fix(file_path, error, iterations_used)
            if attempt.result in [
                FixResult.UNFIXABLE,
                FixResult.FAILED,
                FixResult.NEW_ERRORS_INTRODUCED,
            ]:
                logger.info(
                    f"Rule-based fix failed for {error.get('code')}. Escalating to Adam."
                )
                llm_attempt = await self._attempt_fix_with_llm(
                    file_path, error, [attempt]
                )
                fix_attempts.append(llm_attempt)
                iterations_used += llm_attempt.iterations_used
                if llm_attempt.result == FixResult.SUCCESS:
                    self.stats["llm_fixes"] += 1
                    if llm_attempt.fix_generated is not None:
                        await self._learn_from_llm_fix(error, llm_attempt.fix_generated)
            else:
                fix_attempts.append(attempt)
                iterations_used += attempt.iterations_used
            error_code = error.get("code", "")
            success = attempt.result == FixResult.SUCCESS
            self.pattern_matcher.update_success_rate(error_code, success)
            if attempt.result == FixResult.PARTIALLY_FIXED:
                self.stats["partial_fixes"] += 1
            if attempt.result == FixResult.NEW_ERRORS_INTRODUCED:
                self.stats["new_errors_introduced"] += 1
        final_errors = await self._detect_errors(file_path, error_source)
        total_time = time.time() - start_time
        success_rate = self._calculate_success_rate(original_errors, final_errors)
        session = FileFixSession(
            file_path=file_path,
            original_errors=original_errors,
            fix_attempts=fix_attempts,
            final_errors=final_errors,
            total_time=total_time,
            success_rate=success_rate,
            session_id=session_id,
        )
        self.active_sessions[session_id] = session
        await self._update_statistics(session)
        await self._store_session_memory(session)
        logger.info(
            f"✅ Completed error fixing session {session_id}: {success_rate:.1%} success rate in {total_time:.2f}s"
        )
        return session

    async def _attempt_fix_with_llm(
        self,
        file_path: str,
        error: dict[str, Any],
        failed_attempts: list[ErrorFixAttempt],
    ) -> ErrorFixAttempt:
        """Engage Adam's full mind to devise a fix for a complex error."""
        start_time = time.time()
        logger.info(f"🧠 Engaging Adam for error {error.get('code')} in {file_path}")
        try:
            with open(file_path) as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=None,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description=f"Could not read file for LLM context: {e}",
                is_llm_fix=True,
            )
        failed_attempts_str = "\n".join(
            [f"- {att.description}" for att in failed_attempts]
        )
        objective = f"Fix the error '{error.get('code')}' in the file '{file_path}'."
        prompt_override = f"""You are an expert Python developer who creates plans to fix bugs. Your response must be a single JSON object.

File Path: {file_path}
Error Details:
- Code: {error.get("code")}
- Message: {error.get("message")}
- Line: {error.get("line")}
- Column: {error.get("column")}

Previous failed attempts:
{failed_attempts_str}

Source Code:
```python
{source_code}
```

ANALYSIS INSTRUCTIONS:
1. Analyze the code and the error to understand the problem.
2. Create a plan with one or more steps to fix the bug.
3. Each step must be a call to an available tool (`replace`, `write_file`, etc.).

RESPONSE FORMAT:
You must respond with ONLY a single JSON object. Do not include any other text, explanations, or markdown.
The JSON object must have a `steps` key, which is a list of tool calls. Each tool call is a JSON object with `tool_name` and `parameters`.

EXAMPLE RESPONSE:
{{
    "steps": [
        {{
            "tool_name": "replace",
            "parameters": {{
                "file_path": "{file_path}",
                "old_string": "the code to be replaced",
                "new_string": "the new code"
            }}
        }}
    ]
}}

YOUR RESPONSE:
"""
        perception = {
            "external_request": {
                "original_request": objective,
                "llm_planning_prompt_override": prompt_override,
            }
        }
        await self.adam.update(delta_time=1.0, perception=perception)
        if not self.adam.mind.historia_sintesis:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=None,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description="Adam did not produce a plan.",
                is_llm_fix=True,
            )
        last_synthesis = self.adam.mind.historia_sintesis[-1]
        plan = cast(PlanDeAccion, last_synthesis.get("plan_final"))
        if not plan or not plan.pasos:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=None,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description="Adam produced an empty plan.",
                is_llm_fix=True,
            )
        logger.info(f"Adam produced a plan with {len(plan.pasos)} steps.")
        fix_applied = await self._execute_adam_plan(plan)
        if not fix_applied:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=plan.metadatos,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description="Failed to execute Adam's plan.",
                is_llm_fix=True,
            )
        current_errors = await self._detect_errors(file_path, "both")
        error_still_exists = any(
            e.get("line") == error.get("line") and e.get("code") == error.get("code")
            for e in current_errors
        )
        if not error_still_exists:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=plan.metadatos,
                result=FixResult.SUCCESS,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description=f"Successfully fixed by Adam: {plan.justificacion}",
                is_llm_fix=True,
            )
        else:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=plan.metadatos,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description="Adam's fix applied but error persists.",
                is_llm_fix=True,
            )

    async def _execute_adam_plan(self, plan: PlanDeAccion) -> bool:
        """Executes the steps of a plan generated by Adam."""
        logger.info(f"Executing Adam's plan: {plan.justificacion}")
        for paso in plan.pasos:
            try:
                tool_name = paso.herramienta
                parameters = paso.parametros
                if tool_name not in self.tool_registry.get_all_tools():
                    logger.error(f"Tool '{tool_name}' not found in registry.")
                    return False
                logger.info(f"Executing step: {tool_name} with params {parameters}")
                result = await self.tool_registry.execute_tool(tool_name, **parameters)
                if hasattr(result, "success") and not result.success:
                    logger.error(
                        f"Execution of tool '{tool_name}' failed: {getattr(result, 'error_message', '')}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Exception during plan execution at step {paso}: {e}")
                return False
        return True

    async def _learn_from_llm_fix(
        self,
        error: dict[str, Any],
        fix_spec: dict[str, Any],
    ):
        """Generates a new ErrorPattern from a successful LLM fix using the return_data tool."""
        logger.info(f"🧠 Learning from successful LLM fix for {error.get('code')}")
        objective = "Analyze this successful bug fix and generate a new, generalized rule (ErrorPattern) for the knowledge base."
        error_pattern_template = {
            "error_codes": [error.get("code")],
            "description": "A generalized description of the error.",
            "fix_strategy": "text_replace",
            "confidence": 0.75,
            "template": "A regex or simple text template for the fix.",
        }
        prompt_override = f"""Analyze the following successful bug fix and generate a new, generalized rule (ErrorPattern) for the knowledge base.

Original Error:
`{json.dumps(error, indent=2)}`

Successful Fix Specification:
`{json.dumps(fix_spec, indent=2)}`

Your task is to generate a plan whose final step is a call to the `return_data` tool.
The `data` parameter of the `return_data` tool should be a JSON object representing a new `ErrorPattern`.
Use this template for the `ErrorPattern` JSON:
`{json.dumps(error_pattern_template, indent=2)}`

Fill in the `description` and `template` fields with generalized values derived from the specific fix.
"""
        perception = {
            "external_request": {
                "original_request": objective,
                "llm_planning_prompt_override": prompt_override,
            }
        }
        await self.adam.update(delta_time=1.0, perception=perception)
        if not self.adam.mind.historia_sintesis:
            logger.warning("Could not learn from fix: Adam produced no synthesis.")
            return

        last_synthesis = self.adam.mind.historia_sintesis[-1]
        final_plan = cast(PlanDeAccion, last_synthesis.get("plan_final"))

        if not final_plan or not final_plan.pasos:
            logger.warning("Could not learn from fix: Adam produced an empty plan.")
            return

        # Execute the plan to get the result from the return_data tool
        last_step_result = None
        for paso in final_plan.pasos:
            try:
                tool_name = paso.herramienta
                parameters = paso.parametros
                result = await self.tool_registry.execute_tool(tool_name, **parameters)
                if tool_name == "return_data":
                    last_step_result = result
            except Exception as e:
                logger.error(
                    f"Exception during learning plan execution at step {paso}: {e}"
                )
                return

        if not last_step_result or not last_step_result.success:
            logger.error("Learning plan did not succeed or did not return data.")
            return

        try:
            pattern_json = json.loads(last_step_result.output)
            new_pattern = ErrorPattern(
                error_codes=pattern_json["error_codes"],
                description=pattern_json["description"],
                fix_strategy=FixStrategy(pattern_json["fix_strategy"]),
                confidence=pattern_json["confidence"],
                template=pattern_json.get("template"),
            )
            error_code = new_pattern.error_codes[0]
            self.pattern_matcher.patterns[error_code] = new_pattern
            self.stats["new_patterns_learned"] += 1
            logger.info(
                f"✅ Successfully learned and injected new rule for {error_code}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                f"Failed to parse or create ErrorPattern from LLM response: {e}"
            )

    def assign_mission(self, mission: str, context: dict[str, Any] | None = None):
        """Assigns a new mission for the agent to execute in autonomous mode."""
        logger.info(f"Received new mission: {mission}")
        self.current_mission = mission
        self.mission_context = context or {}

    async def run_autonomous_mode(self, stop_event=None):
        """Runs the agent in a mission-driven autonomous loop."""
        logger.info("🤖 AgentMew entering autonomous 'Golem' mode.")
        while not (stop_event and stop_event.is_set()):
            if self.current_mission:
                logger.info(f"🚀 Starting mission: {self.current_mission}")
                self.stats["autonomous_cycles"] += 1
                try:
                    if self.current_mission == "fix_codebase":
                        await self._execute_fix_codebase_mission()
                    else:
                        logger.warning(f"Unknown mission type: {self.current_mission}")
                    logger.info(f"✅ Mission '{self.current_mission}' completed.")
                except Exception as e:
                    logger.error(f"Error during mission '{self.current_mission}': {e}")
                finally:
                    self.current_mission = None
                    self.mission_context = {}
            else:
                logger.info("Idle and awaiting mission...")
                await asyncio.sleep(15)

    async def _execute_fix_codebase_mission(self):
        """Executes the mission to find and fix all errors in the codebase."""
        scan_result = await self.glob_tool.execute(pattern="**/*.py")
        if not scan_result.success or not scan_result.output:
            logger.warning("No Python files found to scan.")
            return
        python_files = json.loads(scan_result.output).get("files", [])
        for file_path in python_files:
            if any(part in file_path for part in ["/.venv/", "/node_modules/"]):
                continue
            logger.info(f"Scanning file: {file_path}")
            errors = await self._detect_errors(file_path, "both")
            if errors:
                logger.info(
                    f"Found {len(errors)} errors in {file_path}. Attempting to fix."
                )
                await self.fix_file(file_path)

    async def _detect_errors(self, file_path: str, source: str) -> list[dict[str, Any]]:
        all_errors = []
        try:
            if source in ["ruff", "both"]:
                linter_result = await self.linter_tool.execute(file_path=file_path)
                if linter_result.success:
                    linter_data = json.loads(linter_result.output)
                    all_errors.extend(linter_data.get("issues", []))
            if source in ["mypy", "both"]:
                type_result = await self.type_checker_tool.execute(file_path=file_path)
                if type_result.success:
                    type_data = json.loads(type_result.output)
                    all_errors.extend(type_data.get("issues", []))
        except Exception as e:
            logger.error(f"Error detecting issues in {file_path}: {e}")
        return all_errors

    def _filter_fixable_errors(
        self,
        errors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fixable = []
        for error in errors:
            error_code = error.get("code", "")
            if (
                self.pattern_matcher.get_fix_confidence(error_code)
                >= self.confidence_threshold
            ):
                fixable.append(error)
        return fixable

    async def _attempt_fix(
        self,
        file_path: str,
        error: dict[str, Any],
        iteration_start: int,
    ) -> ErrorFixAttempt:
        start_time = time.time()
        error_code = error.get("code", "")
        pattern = self.pattern_matcher.find_pattern(
            error_code, error.get("message", "")
        )
        if not pattern:
            return ErrorFixAttempt(
                error=error,
                pattern=None,
                fix_generated=None,
                result=FixResult.UNFIXABLE,
                execution_time=time.time() - start_time,
                iterations_used=0,
                description=f"No pattern found for error code {error_code}",
            )
        fix_generator = self.fix_factory.get_generator(pattern.fix_strategy)
        fix_spec = await fix_generator.generate_fix(file_path, error, pattern)
        if not fix_spec:
            return ErrorFixAttempt(
                error=error,
                pattern=pattern,
                fix_generated=None,
                result=FixResult.UNFIXABLE,
                execution_time=time.time() - start_time,
                iterations_used=0,
                description=f"Could not generate fix for {error_code}",
            )
        fix_result = await self._apply_fix(fix_spec)
        if not fix_result:
            return ErrorFixAttempt(
                error=error,
                pattern=pattern,
                fix_generated=fix_spec,
                result=FixResult.FAILED,
                execution_time=time.time() - start_time,
                iterations_used=1,
                description=f"Failed to apply fix for {error_code}",
            )
        iterations_used = 1
        for _ in range(self.max_iterations_per_error - 1):
            current_errors = await self._detect_errors(file_path, "both")
            if not any(
                e.get("line") == error.get("line")
                and e.get("code") == error.get("code")
                for e in current_errors
            ):
                return ErrorFixAttempt(
                    error=error,
                    pattern=pattern,
                    fix_generated=fix_spec,
                    result=FixResult.SUCCESS,
                    execution_time=time.time() - start_time,
                    iterations_used=iterations_used,
                    description=f"Successfully fixed {error_code}: {fix_spec.get('description', '')}",
                )
            if len(current_errors) > len([error]):
                return ErrorFixAttempt(
                    error=error,
                    pattern=pattern,
                    fix_generated=fix_spec,
                    result=FixResult.NEW_ERRORS_INTRODUCED,
                    execution_time=time.time() - start_time,
                    iterations_used=iterations_used,
                    description=f"Fix for {error_code} introduced new errors",
                )
            iterations_used += 1
        return ErrorFixAttempt(
            error=error,
            pattern=pattern,
            fix_generated=fix_spec,
            result=FixResult.FAILED,
            execution_time=time.time() - start_time,
            iterations_used=iterations_used,
            description=f"Error {error_code} persists after {iterations_used} attempts",
        )

    async def _apply_fix(self, fix_spec: dict[str, Any]) -> bool:
        try:
            fix_type = fix_spec.get("type")
            absolute_file_path = os.path.abspath(fix_spec["file_path"])
            if fix_type == "file_content_replace":
                from crisalida_lib.ASTRAL_TOOLS.file_system import WriteFileTool

                write_tool = WriteFileTool()
                result = await write_tool.execute(
                    file_path=absolute_file_path, content=fix_spec["new_content"]
                )
                return result.success
            elif fix_type == "line_replace":
                from crisalida_lib.ASTRAL_TOOLS.file_system import ReplaceTool

                replace_tool = ReplaceTool()
                result = await replace_tool.execute(
                    file_path=absolute_file_path,
                    old_string=fix_spec["old_line"],
                    new_string=fix_spec["new_line"],
                    expected_replacements=1,
                )
                return result.success
        except Exception as e:
            logger.error(f"Error applying fix: {e}")
        return False

    def _calculate_success_rate(
        self,
        original_errors: list[dict[str, Any]],
        final_errors: list[dict[str, Any]],
    ) -> float:
        if not original_errors:
            return 1.0
        errors_fixed = len(original_errors) - len(final_errors)
        return max(0.0, errors_fixed / len(original_errors))

    async def _update_statistics(self, session: FileFixSession):
        self.stats["files_processed"] += 1
        self.stats["errors_fixed"] += sum(
            1 for attempt in session.fix_attempts if attempt.result == FixResult.SUCCESS
        )
        self.stats["errors_attempted"] += len(session.fix_attempts)
        total_files = self.stats["files_processed"]
        if total_files > 0:
            self.stats["average_fix_time"] = (
                self.stats["average_fix_time"] * (total_files - 1) + session.total_time
            ) / total_files
        if self.stats["errors_attempted"] > 0:
            self.stats["success_rate"] = (
                self.stats["errors_fixed"] / self.stats["errors_attempted"]
            )

    async def _store_session_memory(self, session: FileFixSession):
        for attempt in session.fix_attempts:
            if attempt.result == FixResult.SUCCESS:
                self.adam.eva_manager.record_experience(
                    entity_id=self.adam.entity_id,
                    event_type="bug_fix_interaction",
                    data={
                        "user_input": f"Fix error {attempt.error.get('code', '')} in {session.file_path}",
                        "agent_response": f"Applied fix: {attempt.description}",
                        "error_code": attempt.error.get("code"),
                        "fix_strategy": (
                            attempt.pattern.fix_strategy.value
                            if attempt.pattern
                            else "llm_generated"
                        ),
                        "execution_time": attempt.execution_time,
                        "success": True,
                        "is_llm_fix": attempt.is_llm_fix,
                    },
                )
