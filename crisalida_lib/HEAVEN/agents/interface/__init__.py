"""
Agent Interface Module
====================

Basic interface for managing Crisalida agents.
"""

from .basic_agent_interface import (
    AgentMode,
    AgentSession,
    AgentStatus,
    BasicAgentInterface,
    PromptResponse,
    SimpleAgent,
)

__all__ = [
    "BasicAgentInterface",
    "SimpleAgent",
    "AgentMode",
    "AgentStatus",
    "AgentSession",
    "PromptResponse",
]
