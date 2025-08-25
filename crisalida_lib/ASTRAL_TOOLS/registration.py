"""
Tool Registration
=================

This module is responsible for registering all the tools in the ToolRegistry.
"""

import logging

from crisalida_lib.ASTRAL_TOOLS.base import ToolRegistry
from crisalida_lib.ASTRAL_TOOLS.default_tools import register_default_tools

logger = logging.getLogger(__name__)


def register_all_tools(registry: ToolRegistry):
    """Register all default tools."""
    try:
        register_default_tools(registry)
    except Exception as e:
        logger.error(f"Failed to register default tools: {e}")
