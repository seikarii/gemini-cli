"""Bridge to propose AutoGenesisEngine generated modules into SelfModifyingEngine.

Usage: from crisalida_lib.EARTH.age_sme_bridge import propose_module
propose_module(sme_instance, module_info, proposer="age")
"""
from typing import Any
import logging

logger = logging.getLogger(__name__)


def propose_module(sme: Any, module_info: dict[str, Any], proposer: str | None = "age") -> str:
    """Wrap module_info into SME proposal; returns proposal_id.

    This function expects an instance of SelfModifyingEngine (or similar) and a
    module_info dict produced by AutoGenesisEngine._create_and_load_module or
    analyze_memory_and_generate_code pipelines.
    """
    if not hasattr(sme, "submit_generated_module_proposal"):
        raise RuntimeError("SME instance does not support generated module proposals")

    try:
        pid = sme.submit_generated_module_proposal(module_info, proposer=proposer)
        logger.info("Proposed module to SME: %s -> proposal=%s", module_info.get("module_name"), pid)
        return pid
    except Exception as e:
        logger.exception("Failed to propose module to SME: %s", e)
        raise
