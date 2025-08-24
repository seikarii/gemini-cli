#!/usr/bin/env python3
"""
Basic Agent Interface - Crisalida Agent Management System
========================================================

A basic interface for managing agents in the Crisalida system.
Provides functionality to:
- Start agents with specified configurations
- Send prompts to agents
- Change operational modes (automatic, directed, disconnect)
- Monitor agent status and performance
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

# --- Agent Implementations ---
# Working (Simplified) Agents
# from crisalida_lib.agents.working_babel_agent import WorkingBabelAgent
# from crisalida_lib.agents.working_bugs_agent import OperationalMode as BugsOpMode
# from crisalida_lib.agents.working_bugs_agent import WorkingBugsAgent
from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.HEAVEN.agents.agent_mew import AgentMew

# Real (Complex) Agents
# from crisalida_lib.HEAVEN.agents.agent_bugs import BugFixAgent
# Core Crisalida components
from crisalida_lib.HEAVEN.agents.core.action_system import ActionSystem
from crisalida_lib.HEAVEN.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent operational modes"""

    AUTOMATIC = "automatic"
    DIRECTED = "directed"
    DISCONNECTED = "disconnected"


class AgentStatus(Enum):
    """Agent status states"""

    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentSession:
    """Represents an active agent session"""

    session_id: str
    agent_type: str
    agent_id: str
    status: AgentStatus
    mode: AgentMode
    start_time: float
    last_activity: float
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None


@dataclass
class PromptResponse:
    """Response from agent prompt processing"""

    session_id: str
    agent_id: str
    response: str
    success: bool
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SimpleAgent:
    """
    Simplified agent base class that bypasses complex consciousness system
    for basic interface functionality. Acts as a wrapper/adapter.
    """

    def __init__(self, agent_id: str, agent_type: str, config: EVAConfig | None = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or EVAConfig()
        self.action_system = ActionSystem()
        self.mode = AgentMode.DISCONNECTED
        self.status = AgentStatus.STOPPED
        self.session_id: str | None = None
        self.metrics = {
            "prompts_processed": 0,
            "actions_executed": 0,
            "errors_encountered": 0,
            "total_runtime": 0.0,
            "last_prompt_time": 0.0,
        }

    async def start(self) -> str | None:
        """Start the agent and return session ID"""
        self.session_id = f"{self.agent_type}_{self.agent_id}_{int(time.time())}"
        self.status = AgentStatus.STARTING
        self.mode = AgentMode.DIRECTED
        self.action_system.start()
        self._register_tools()
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.agent_id} started with session {self.session_id}")
        return self.session_id

    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.action_system.stop()
            self.status = AgentStatus.STOPPED
            self.mode = AgentMode.DISCONNECTED
            logger.info(f"Agent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def set_mode(self, mode: AgentMode) -> bool:
        """Change agent operational mode"""
        try:
            old_mode = self.mode
            self.mode = mode
            logger.info(
                f"Agent {self.agent_id} mode changed from {old_mode.value} to {mode.value}"
            )
            return True
        except Exception as e:
            logger.error(f"Error changing mode for agent {self.agent_id}: {e}")
            return False

    async def process_prompt(
        self, prompt: str, metadata: dict[str, Any] | None = None
    ) -> PromptResponse:
        """Process a prompt and return response"""
        start_time = time.time()
        metadata = metadata or {}
        try:
            if self.status != AgentStatus.RUNNING:
                raise ValueError(
                    f"Agent {self.agent_id} is not running (status: {self.status.value})"
                )

            self.metrics["prompts_processed"] += 1
            self.metrics["last_prompt_time"] = start_time

            if self.mode == AgentMode.DISCONNECTED:
                raise ValueError("Agent is disconnected")
            elif self.mode == AgentMode.DIRECTED:
                response = await self._process_directed_prompt(prompt, metadata)
            elif self.mode == AgentMode.AUTOMATIC:
                response = await self._process_automatic_prompt(prompt, metadata)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            execution_time = time.time() - start_time
            self.metrics["total_runtime"] += execution_time
            return PromptResponse(
                session_id=self.session_id if self.session_id is not None else "",
                agent_id=self.agent_id,
                response=response,
                success=True,
                execution_time=execution_time,
                metadata=metadata,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics["errors_encountered"] += 1
            logger.exception(f"Error processing prompt for agent {self.agent_id}")
            return PromptResponse(
                session_id=self.session_id if self.session_id is not None else "",
                agent_id=self.agent_id,
                response=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time,
                metadata=metadata,
            )

    def _register_tools(self):
        """Register basic tools for the agent"""
        self.action_system.register_tool("echo", lambda x: f"Echo: {x}")
        self.action_system.register_tool("status", lambda: self.get_status())

    async def _process_directed_prompt(
        self, prompt: str, metadata: dict[str, Any]
    ) -> str:
        """Process prompt in directed mode - subclasses should override"""
        return f"[DIRECTED] Processed: {prompt}"

    async def _process_automatic_prompt(
        self, prompt: str, metadata: dict[str, Any]
    ) -> str:
        """Process prompt in automatic mode - subclasses should override"""
        return f"[AUTOMATIC] Analyzed and processed: {prompt}"

    def get_status(self) -> dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "metrics": self.metrics.copy(),
            "action_system_stats": (
                self.action_system.get_stats() if self.action_system else {}
            ),
        }


class BasicAgentInterface:
    """
    Basic Agent Interface for managing multiple agents.
    Now supports choosing between 'working' and 'real' agent versions.
    """

    def __init__(self):
        self.agents: dict[str, SimpleAgent] = {}
        self.sessions: dict[str, AgentSession] = {}
        self.available_agent_types = [
            "agent_mew",
            # "bugs_working",
            # "bugs_real",
            # "babel_working",
            # "babel_real",
            # "prometheus",
            # "fallen"
        ]

    def list_available_agents(self) -> list[str]:
        """Get list of available agent types"""
        return self.available_agent_types.copy()

    def list_active_agents(self) -> list[dict[str, Any]]:
        """Get list of active agent sessions"""
        active = []
        for _agent_id, agent in self.agents.items():
            if agent.status != AgentStatus.STOPPED:
                active.append(agent.get_status())
        return active

    async def create_agent(
        self,
        agent_type: str,
        agent_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new agent instance"""
        if agent_type not in self.available_agent_types:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {self.available_agent_types}"
            )

        agent_id = agent_id or f"{agent_type}_{uuid4().hex[:8]}"
        if agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent_id} already exists")

        eva_config = EVAConfig()
        if config:
            for key, value in config.items():
                if hasattr(eva_config, key):
                    setattr(eva_config, key, value)

        # Agent factory
        if agent_type == "agent_mew":
            agent = AgentMewWrapper(agent_id, eva_config, agent_config=config)
        # elif agent_type == "bugs_working":
        #     agent = WorkingBugsAgentWrapper(agent_id, eva_config)
        # elif agent_type == "bugs_real":
        #     agent = RealBugFixAgentWrapper(agent_id, eva_config)
        # elif agent_type == "babel_working":
        #     agent = WorkingBabelAgentWrapper(agent_id, eva_config)
        # elif agent_type == "babel_real":
        #     # Placeholder for the real BabelAgent wrapper
        #     agent = SimpleAgent(agent_id, agent_type, eva_config)
        #     logger.warning("Real BabelAgent not fully implemented in interface yet. Using placeholder.")
        # elif agent_type == "prometheus":
        #     agent = SimplePrometheusAgent(agent_id, eva_config)
        # elif agent_type == "fallen":
        #     agent = SimpleFallenAgent(agent_id, eva_config)
        else:
            agent = SimpleAgent(agent_id, agent_type, eva_config)

        self.agents[agent_id] = agent
        logger.info(f"Created {agent_type} agent with ID: {agent_id}")
        return agent_id

    async def start_agent(self, agent_id: str) -> str | None:
        """Start an agent and return session ID"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        agent = self.agents[agent_id]
        session_id = await agent.start()
        if session_id is None:
            return None  # Agent.start() can return None
        session = AgentSession(
            session_id=session_id,
            agent_type=agent.agent_type,
            agent_id=agent_id,
            status=agent.status,
            mode=agent.mode,
            start_time=time.time(),
            last_activity=time.time(),
            config={"agent_type": agent.agent_type},
        )
        self.sessions[session_id] = session
        return session_id

    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        agent = self.agents[agent_id]
        success = await agent.stop()
        if agent.session_id and agent.session_id in self.sessions:
            self.sessions[agent.session_id].status = agent.status
        return success

    async def set_agent_mode(self, agent_id: str, mode: str | AgentMode) -> bool:
        """Change agent operational mode"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        if isinstance(mode, str):
            mode = AgentMode(mode)
        agent = self.agents[agent_id]
        success = await agent.set_mode(mode)
        if agent.session_id and agent.session_id in self.sessions:
            self.sessions[agent.session_id].mode = agent.mode
            self.sessions[agent.session_id].last_activity = time.time()
        return success

    async def send_prompt(
        self, agent_id: str, prompt: str, metadata: dict[str, Any] | None = None
    ) -> PromptResponse:
        """Send prompt to an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        agent = self.agents[agent_id]
        response = await agent.process_prompt(prompt, metadata)
        if agent.session_id and agent.session_id in self.sessions:
            self.sessions[agent.session_id].last_activity = time.time()
        return response

    def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.agents[agent_id].get_status()

    def get_session_info(self, session_id: str) -> AgentSession | None:
        """Get information about a session"""
        return self.sessions.get(session_id)

    async def call_agent_method(
        self, agent_id: str, method_name: str, **kwargs: Any
    ) -> Any:
        """Call a method on the underlying real agent instance."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent_wrapper = self.agents[agent_id]
        if not hasattr(agent_wrapper, "real_agent") or agent_wrapper.real_agent is None:
            raise AttributeError(
                f"Agent {agent_id} does not have a real_agent instance."
            )

        real_agent_method = getattr(agent_wrapper.real_agent, method_name, None)
        if real_agent_method is None or not callable(real_agent_method):
            raise AttributeError(
                f"Method {method_name} not found or not callable on agent {agent_id}."
            )

        logger.info(
            f"Calling method '{method_name}' on agent '{agent_id}' with kwargs: {kwargs}"
        )
        result = await real_agent_method(**kwargs)
        return result


# --- Agent Wrapper Implementations ---

# class WorkingBabelAgentWrapper(SimpleAgent):
#     """Wrapper for the simplified 'WorkingBabelAgent'."""
#     def __init__(self, agent_id: str, config: EVAConfig):
#         super().__init__(agent_id, "babel_working", config)
#         self.working_agent = WorkingBabelAgent(config)

#     def _register_tools(self):
#         super()._register_tools()
#         self.action_system = self.working_agent.get_action_system()

#     async def _process_directed_prompt(self, prompt: str, metadata: dict[str, Any]) -> str:
#         """Processes library creation requests."""
#         try:
#             # A simple heuristic to decide which method to call
#             if "create" in prompt.lower() and "library" in prompt.lower():
#                 session_id = await self.working_agent.create_library(prompt, **metadata)
#                 return f"[BABEL-WORKING] Library creation session started: {session_id}"
#             else:
#                 return "[BABEL-WORKING] Prompt not understood. Try 'create library ...'"
#         except Exception as e:
#             logger.exception("Error in WorkingBabelAgentWrapper")
#             return f"[BABEL-WORKING-ERROR] {str(e)}"


# class WorkingBugsAgentWrapper(SimpleAgent):
#     """Wrapper for the simplified 'WorkingBugsAgent'."""
#     def __init__(self, agent_id: str, config: EVAConfig):
#         super().__init__(agent_id, "bugs_working", config)
#         self.working_agent = WorkingBugsAgent(config)

#     def _register_tools(self):
#         super()._register_tools()
#         self.action_system = self.working_agent.get_action_system()

#     async def _process_directed_prompt(self, prompt: str, metadata: dict[str, Any]) -> str:
#         """Processes bug fixing requests."""
#         try:
#             # A simple heuristic: if prompt contains 'fix' and a path-like string
#             parts = shlex.split(prompt)
#             if parts[0].lower() == 'fix' and len(parts) > 1:
#                 file_path = parts[1]
#                 session_id = await self.working_agent.process_file_errors(
#                     file_path, mode=BugsOpMode.DIRECTED
#                 )
#                 status = self.working_agent._get_session_status(session_id)
#                 return f"[BUGS-WORKING] Fix session {session_id} started for {file_path}. Status: {status}"
#             else:
#                 return "[BUGS-WORKING] Prompt not understood. Use 'fix /path/to/your/file.py'"
#         except Exception as e:
#             logger.exception("Error in WorkingBugsAgentWrapper")
#             return f"[BUGS-WORKING-ERROR] {str(e)}"


# class RealBugFixAgentWrapper(SimpleAgent):
#     """Wrapper for the real, complex 'BugFixAgent'."""
#     def __init__(self, agent_id: str, config: EVAConfig):
#         super().__init__(agent_id, "bugs_real", config)
#         try:
#             self.real_agent = BugFixAgent()
#             logger.info("Real BugFixAgent instantiated successfully.")
#         except Exception:
#             logger.exception("Failed to instantiate real BugFixAgent.")
#             self.real_agent = None

#     async def _process_directed_prompt(self, prompt: str, metadata: dict[str, Any]) -> str:
#         """Translates a prompt into a call to the real agent's methods."""
#         if not self.real_agent:
#             return "[BUGS-REAL-ERROR] Real agent could not be initialized."
#         try:
#             parts = shlex.split(prompt)
#             if parts[0].lower() == 'fix' and len(parts) > 1:
#                 file_path = parts[1]
#                 logger.info(f"Calling real BugFixAgent to fix {file_path}")
#                 session = await self.real_agent.fix_file(file_path)
#                 response = (
#                     f"Fix session for {file_path} completed.\n"
#                     f"Session ID: {session.session_id}\n"
#                     f"Success Rate: {session.success_rate:.1%}\n"
#                     f"Original Errors: {len(session.original_errors)}\n"
#                     f"Final Errors: {len(session.final_errors)}\n"
#                     f"Time Taken: {session.total_time:.2f}s"
#                 )
#                 return response
#             else:
#                 return "[BUGS-REAL] Prompt not understood. Use 'fix /path/to/your/file.py'"
#         except Exception as e:
#             logger.exception("Error in RealBugFixAgentWrapper")
#             return f"[BUGS-REAL-ERROR] {str(e)}"


# class SimplePrometheusAgent(SimpleAgent):
#     """Simplified Prometheus agent for system orchestration"""
#     def __init__(self, agent_id: str, config: EVAConfig):
#         super().__init__(agent_id, "prometheus", config)

#     async def _process_directed_prompt(self, prompt: str, metadata: dict[str, Any]) -> str:
#         return f"[PROMETHEUS-DIRECTED] Orchestrating: {prompt}"


# class SimpleFallenAgent(SimpleAgent):
#     """Simplified Fallen agent for specialized tasks"""
#     def __init__(self, agent_id: str, config: EVAConfig):
#         super().__init__(agent_id, "fallen", config)

#     async def _process_directed_prompt(self, prompt: str, metadata: dict[str, Any]) -> str:
#         return f"[FALLEN-DIRECTED] Specialized processing: {prompt}"


class AgentMewWrapper(SimpleAgent):
    """Wrapper for the new advanced AgentMew."""

    def __init__(
        self, agent_id: str, config: EVAConfig, agent_config: dict | None = None
    ):
        super().__init__(agent_id, "agent_mew", config)
        agent_config = agent_config or {}
        try:
            # Extract model from the config dict
            ollama_client = OllamaClient()

            # Pass the llm_model to AgentMew so it can use it in LLM calls
            llm_model = agent_config.get("llm_model")
            logger.debug(f"AgentMewWrapper: llm_model received: {llm_model}")
            self.real_agent: AgentMew | None = AgentMew(
                config=config, ollama_client=ollama_client, llm_model=llm_model
            )
            logger.info(
                f"Real AgentMew instantiated successfully (Model: {llm_model or 'default'})."
            )
        except Exception:
            logger.exception("Failed to instantiate real AgentMew.")
            self.real_agent = None

    async def _process_directed_prompt(
        self, prompt: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Translates a simple prompt into a complex request for AgentMew."""
        if not self.real_agent:
            return "[AGENT_MEW-ERROR] Real agent could not be initialized."
        try:
            # Simple command parsing for demonstration
            if prompt.lower().startswith("fix file "):
                file_path = prompt[len("fix file ") :].strip()
                logger.info(f"AgentMewWrapper: Calling fix_file for {file_path}")
                session = await self.real_agent.fix_file(file_path)
                return f"Fix session for {file_path} completed. Success rate: {session.success_rate:.2f}"
            elif prompt.lower().startswith("assign mission "):
                mission = prompt[len("assign mission ") :].strip()
                logger.info(f"AgentMewWrapper: Assigning mission {mission}")
                self.real_agent.assign_mission(mission, metadata)
                # In a real scenario, you might want to run autonomous mode in background
                # asyncio.create_task(self.real_agent.run_autonomous_mode())
                return f"Mission '{mission}' assigned to AgentMew."
            elif prompt.lower() == "run autonomous mode":
                logger.info("AgentMewWrapper: Running autonomous mode.")
                await self.real_agent.run_autonomous_mode()
                return "AgentMew entered autonomous mode."
            else:
                return f"[AGENT_MEW] Prompt not understood: {prompt}. Try 'fix file <path>' or 'assign mission <name>'"
        except Exception as e:
            logger.exception("Error in AgentMewWrapper")
            return f"[AGENT_MEW-ERROR] {str(e)}"
