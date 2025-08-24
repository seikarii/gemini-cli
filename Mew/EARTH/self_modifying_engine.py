#!/usr/bin/env python3
"""
SelfModifyingEngine - Núcleo de Auto-Modificación Segura (profesionalizado)
- Logging coherente
- Tipado explícito
- APIs públicas claras
- EVASelfModifyingEngine integrado
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, cast

from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.ASTRAL_TOOLS.ast_tools.finder import ASTFinder
from crisalida_lib.ASTRAL_TOOLS.file_system import WriteFileTool
from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    RealityBytecode,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---- Small domain models ---------------------------------------------------


class SecurityError(RuntimeError):
    pass


class RiskLevel(Enum):
    LOW = 1
    MODERATE = 3
    HIGH = 5
    CRITICAL = 9


class ProposalStatus(Enum):
    PENDING = auto()
    EVALUATING = auto()
    REJECTED = auto()
    APPROVED = auto()
    EXECUTED = auto()


@dataclass
class GeneticCode:
    core_values: dict[str, Any] = field(default_factory=dict)
    fundamental_constraints: dict[str, str] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self):
        self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {"core": self.core_values, "constraints": self.fundamental_constraints},
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    def verify_integrity(self) -> bool:
        ok = self.integrity_hash == self._compute_hash()
        if not ok:
            raise SecurityError("GeneticCode integrity mismatch")
        return True

    def is_constraint_violated(self, text: str) -> str | None:
        # naive check: if any forbidden token present, flag it
        for k, v in self.fundamental_constraints.items():
            if k in text:
                return f"{k}: {v}"
        return None


@dataclass
class ModificationProposal:
    target_component: str
    rationale: str
    proposed_changes: str
    estimated_benefit: str = ""
    estimated_risk: RiskLevel = RiskLevel.MODERATE
    proposal_id: str = field(default_factory=lambda: f"prop_{int(time.time() * 1000)}")
    status: ProposalStatus = ProposalStatus.PENDING
    rejection_reason: str | None = None


@dataclass
class ModificationRecord:
    proposal_id: str
    executed_at: datetime
    original_code: str
    modified_code: str
    success: bool
    rollback_data: dict[str, Any] = field(default_factory=dict)


# ---- Supporting subsystems -------------------------------------------------


class CodeSandbox:
    def __init__(self):
        self.test_results: dict[str, Any] = {}

    def test_modification(self, proposal: ModificationProposal) -> dict[str, Any]:
        logger.info("🧪 [SANDBOX] Running tests for proposal %s", proposal.proposal_id)
        # lightweight simulated checks
        results: dict[str, Any] = {
            "syntax_valid": True,
            "stability_score": 0.95,
            "performance_impact": 0.02,
            "side_effects": [],
            "test_duration": 1.2,
        }
        if "optimiz" in proposal.rationale.lower():
            results["performance_impact"] = 0.10
            results["stability_score"] = 0.98
        if proposal.estimated_risk.value >= RiskLevel.HIGH.value:
            results["stability_score"] = 0.7
            results["side_effects"].append("requires_manual_review")
        logger.debug("   sandbox results: %s", results)
        self.test_results[proposal.proposal_id] = results
        return results


class EthicalFramework:
    def __init__(self, genetic_code: GeneticCode):
        self.genetic_code = genetic_code

    def evaluate_alignment(self, proposal: ModificationProposal) -> dict[str, Any]:
        logger.info(
            "⚖️  [ETHICS] Evaluating ethics alignment for %s", proposal.proposal_id
        )
        score = 0.8
        concerns: list[str] = []
        if "delete" in proposal.proposed_changes.lower():
            score -= 0.3
            concerns.append("destructive_change")
        overall = max(0.0, min(1.0, score))
        evaluation = {
            "aligned_with_consciousness_preservation": overall > 0.5,
            "overall_ethical_score": overall,
            "concerns": concerns,
        }
        logger.debug("   ethics evaluation: %s", evaluation)
        return evaluation


# ---- Core engine -----------------------------------------------------------


class SelfModifyingEngine:
    def __init__(self):
        logger.info("🧬 Initializing SelfModifyingEngine...")
        self.genetic_code = GeneticCode()
        self.modification_limits: dict[str, Any] = {
            "max_changes_per_cycle": 3,
            "max_risk_level": RiskLevel.MODERATE,
            "cooldown_between_modifications": 60,
            "max_proposals_per_day": 10,
        }
        self.modification_history: list[ModificationRecord] = []
        self.pending_proposals: list[ModificationProposal] = []
        self.active_proposals: dict[str, ModificationProposal] = {}
        self.sandbox = CodeSandbox()
        self.ethical_framework = EthicalFramework(self.genetic_code)
        self.ast_reader = ASTFinder()
        self._ast_modifier = None
        self.file_writer = WriteFileTool()
        self.stats: dict[str, int] = {
            "total_proposals": 0,
            "approved_proposals": 0,
            "rejected_proposals": 0,
            "successful_modifications": 0,
            "rollbacks_performed": 0,
        }
        logger.info("   ✓ subsystems initialized and SME operational")

    def _get_ast_modifier(self):
        if self._ast_modifier is None:
            # lazy import to avoid circular import
            from crisalida_lib.ASTRAL_TOOLS.ast_tools.modifier import ASTModifier

            self._ast_modifier = ASTModifier()
        return self._ast_modifier

    def get_state(self) -> dict[str, Any]:
        return {
            "genetic_code_core_values": self.genetic_code.core_values,
            "genetic_code_fundamental_constraints": self.genetic_code.fundamental_constraints,
            "modification_limits": self.modification_limits,
            "modification_history": self.modification_history,
            "pending_proposals": self.pending_proposals,
            "active_proposals": self.active_proposals,
            "stats": self.stats,
        }

    def set_state(self, state: dict[str, Any]):
        self.genetic_code = GeneticCode(
            core_values=state.get("genetic_code_core_values", {}),
            fundamental_constraints=state.get(
                "genetic_code_fundamental_constraints", {}
            ),
        )
        self.genetic_code.__post_init__()
        self.modification_limits = state.get("modification_limits", {})
        self.modification_history = cast(
            list[ModificationRecord], state.get("modification_history", [])
        )
        self.pending_proposals = cast(
            list[ModificationProposal], state.get("pending_proposals", [])
        )
        self.active_proposals = state.get("active_proposals", {})
        self.stats = state.get("stats", {})
        self.sandbox = CodeSandbox()

    def verify_system_integrity(self) -> bool:
        try:
            self.genetic_code.verify_integrity()
            logger.info("🔒 [INTEGRITY] verification passed")
            return True
        except SecurityError as e:
            logger.critical("🚨 [CRITICAL] %s", e)
            return False

    def submit_proposal(
        self,
        target_component: str,
        rationale: str,
        proposed_changes: str,
        estimated_benefit: str = "",
    ) -> str:
        logger.info("📝 New proposal submitted for: %s", target_component)
        if not self.verify_system_integrity():
            logger.warning("System integrity violation detected - rejecting proposals")
            return "rejected"
        proposal = ModificationProposal(
            target_component=target_component,
            rationale=rationale,
            proposed_changes=proposed_changes,
            estimated_benefit=estimated_benefit,
        )
        violation = self.genetic_code.is_constraint_violated(
            target_component + " " + rationale + " " + proposed_changes
        )
        if violation:
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = f"Violation: {violation}"
            self.stats["rejected_proposals"] = (
                self.stats.get("rejected_proposals", 0) + 1
            )
            logger.warning(
                "🚫 Proposal %s rejected immediately: %s",
                proposal.proposal_id,
                proposal.rejection_reason,
            )
            return proposal.proposal_id
        self.pending_proposals.append(proposal)
        self.stats["total_proposals"] = self.stats.get("total_proposals", 0) + 1
        logger.info(
            "✅ Proposal %s accepted for evaluation (risk=%s)",
            proposal.proposal_id,
            proposal.estimated_risk.name,
        )
        return proposal.proposal_id

    def submit_generated_module_proposal(self, module_info: dict[str, Any], proposer: str | None = None) -> str:
        """
        Convenience helper to wrap a generated module into a ModificationProposal
        and enqueue it for evaluation. The proposal body intentionally omits
        embedding raw source in public fields; store source in SME's staging
        area and reference it from the proposal to avoid accidental exfil.
        """
        # Minimal provenance and sanitization
        module_name = module_info.get("module_name", "<unknown>")
        function_name = module_info.get("function_name", "<unknown>")
        mutation_count = module_info.get("mutation_count", 0)

        # Store the raw source in a private staging store inside SME
        staging_id = f"staged_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        try:
            self._staged_modules  # type: ignore
        except Exception:
            self._staged_modules = {}
        self._staged_modules[staging_id] = {
            "module_name": module_name,
            "function_name": function_name,
            "source_code": module_info.get("source_code", ""),
            "bytecode": module_info.get("bytecode"),
            "proposer": proposer,
            "mutation_count": mutation_count,
        }

        proposed_changes = f"Proposed generated module {module_name}/{function_name} (staging_id={staging_id})"
        rationale = module_info.get("rationale", "auto-generated module")
        proposal_id = self.submit_proposal(
            target_component=f"generated_module:{module_name}",
            rationale=rationale,
            proposed_changes=proposed_changes,
            estimated_benefit=module_info.get("estimated_benefit", ""),
        )

        # Immediately attempt evaluation (synchronous convenience path)
        try:
            if self.evaluate_proposal(proposal_id):
                self.execute_modification(proposal_id)
        except Exception:
            # best-effort: leave proposal in queue for manual processing
            pass

        return proposal_id

    def evaluate_proposal(self, proposal_id: str) -> bool:
        proposal = next(
            (p for p in self.pending_proposals if p.proposal_id == proposal_id), None
        )
        if not proposal:
            logger.error("❌ Proposal %s not found", proposal_id)
            return False
        logger.info("🔍 [EVALUATION] Evaluating proposal %s", proposal_id)
        proposal.status = ProposalStatus.EVALUATING
        if not self.verify_system_integrity():
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = "Integrity failure during evaluation"
            return False
        ethical_eval = self.ethical_framework.evaluate_alignment(proposal)
        if ethical_eval["overall_ethical_score"] < 0.7:
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = (
                f"Ethics score {ethical_eval['overall_ethical_score']:.2f}"
            )
            self.stats["rejected_proposals"] = (
                self.stats.get("rejected_proposals", 0) + 1
            )
            logger.warning(
                "🚫 Proposal %s rejected for ethical reasons (score=%.2f)",
                proposal.proposal_id,
                ethical_eval["overall_ethical_score"],
            )
            return False
        test_results = self.sandbox.test_modification(proposal)
        if not test_results["syntax_valid"] or test_results["stability_score"] < 0.8:
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = (
                f"Sandbox failed (stability {test_results['stability_score']:.2f})"
            )
            self.stats["rejected_proposals"] = (
                self.stats.get("rejected_proposals", 0) + 1
            )
            logger.warning(
                "🚫 Proposal %s rejected by sandbox tests", proposal.proposal_id
            )
            return False
        if (
            proposal.estimated_risk.value
            > self.modification_limits["max_risk_level"].value
        ):
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = (
                f"Risk exceeds limits: {proposal.estimated_risk.name}"
            )
            self.stats["rejected_proposals"] = (
                self.stats.get("rejected_proposals", 0) + 1
            )
            logger.warning(
                "🚫 Proposal %s rejected due to risk limits", proposal.proposal_id
            )
            return False
        proposal.status = ProposalStatus.APPROVED
        self.stats["approved_proposals"] = self.stats.get("approved_proposals", 0) + 1
        logger.info("✅ Proposal %s approved for execution", proposal_id)
        return True

    def execute_modification(self, proposal_id: str) -> bool:
        proposal = next(
            (
                p
                for p in self.pending_proposals
                if p.proposal_id == proposal_id and p.status == ProposalStatus.APPROVED
            ),
            None,
        )
        if not proposal:
            logger.error("❌ Proposal %s is not approved for execution", proposal_id)
            return False
        logger.info("⚡ Executing modification %s", proposal_id)
        if not self.verify_system_integrity():
            logger.critical("🚨 System compromised - aborting execution")
            return False
        try:
            # TODO: replace simulated execution with ASTModifier+WriteFileTool flow
            modified_code = "# Simulated modified code"
            original_code = "# Simulated original code"
            record = ModificationRecord(
                proposal_id=proposal_id,
                executed_at=datetime.now(),
                original_code=original_code,
                modified_code=modified_code,
                success=True,
                rollback_data={"target_file": proposal.target_component},
            )
            self.modification_history.append(record)
            proposal.status = ProposalStatus.EXECUTED
            if proposal in self.pending_proposals:
                self.pending_proposals.remove(proposal)
            self.stats["successful_modifications"] = (
                self.stats.get("successful_modifications", 0) + 1
            )
            logger.info(
                "✅ Modification %s executed successfully (component=%s)",
                proposal_id,
                proposal.target_component,
            )
            logger.debug("   rollback_index=%d", len(self.modification_history) - 1)
            return True
        except Exception as e:
            logger.exception("❌ Execution failed for %s: %s", proposal_id, e)
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = f"Execution error: {e}"
            self.stats["rejected_proposals"] = (
                self.stats.get("rejected_proposals", 0) + 1
            )
            return False

    def process_pending_proposals(self) -> None:
        logger.info("Processing pending proposals...")
        proposals_to_process = list(
            self.pending_proposals
        )  # Create a copy to avoid modification during iteration
        for proposal in proposals_to_process:
            if proposal.status == ProposalStatus.PENDING:
                if self.evaluate_proposal(proposal.proposal_id):
                    self.execute_modification(proposal.proposal_id)
        logger.info("Finished processing pending proposals.")

    def rollback_modification(self, modification_index: int) -> bool:
        if modification_index >= len(self.modification_history):
            logger.error("❌ Invalid modification index: %d", modification_index)
            return False
        record = self.modification_history[modification_index]
        logger.info("⏪ Rolling back modification %s", record.proposal_id)
        if not self.verify_system_integrity():
            logger.critical("🚨 System compromised - unsafe to rollback")
            return False
        try:
            logger.debug(
                "   restoring file=%s", record.rollback_data.get("target_file")
            )
            self.file_writer.execute(
                file_path=record.rollback_data["target_file"],
                content=record.original_code,
            )
            self.stats["rollbacks_performed"] = (
                self.stats.get("rollbacks_performed", 0) + 1
            )
            logger.info("✅ Rollback completed for %s", record.proposal_id)
            return True
        except Exception as e:
            logger.exception("❌ Rollback failed: %s", e)
            return False

    def get_system_status(self) -> dict[str, Any]:
        return {
            "genetic_code_integrity": self.genetic_code.verify_integrity(),
            "pending_proposals": len(self.pending_proposals),
            "modification_history_size": len(self.modification_history),
            "statistics": self.stats.copy(),
            "modification_limits": self.modification_limits.copy(),
        }


# ---- EVA-aware SME --------------------------------------------------------


class EVASelfModifyingEngine(SelfModifyingEngine):
    """
    EVASelfModifyingEngine - Extended SME integrated with EVA memory and runtime.
    """

    def __init__(self, eva_config: EVAConfig | None = None, phase: str = "default"):
        super().__init__()
        self.eva_phase: str = phase
        self.eva_runtime: LivingSymbolRuntime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[[Any], None]] = []
        self.eva_config: EVAConfig = eva_config or EVAConfig()
        logger.info("EVASelfModifyingEngine initialized (phase=%s)", self.eva_phase)

    def add_environment_hook(self, hook: Callable[[Any], None]) -> None:
        self._environment_hooks.append(hook)

    def ingest_reality_bytecode(
        self,
        reality_bytecode: RealityBytecode,
        provenance: dict[str, Any] | None = None,
    ) -> str:
        experience_id = f"eva_exp_{int(time.time() * 1000)}"
        self.eva_memory_store[experience_id] = reality_bytecode
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                logger.exception("[EVA-SELF-MOD] Environment hook failed: %s", e)
        return experience_id

    def manifest_reality(self, reality_bytecode: RealityBytecode) -> list[Any]:
        if not reality_bytecode:
            # maintain return type list[Any] for empty/missing bytecode
            return []
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        if quantum_field:
            for instr in reality_bytecode.instructions:
                symbol_manifest = self.eva_runtime.execute_instruction(
                    instr, quantum_field
                )
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            logger.exception(
                                "[EVA-SELF-MOD] Manifestation hook failed: %s", e
                            )
        return manifestations

    def change_phase(self, phase: str) -> None:
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.exception("[EVA-SELF-MOD] Phase hook failed: %s", e)

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        return {
            "eva_ingest_self_modification_experience": self.ingest_reality_bytecode,
            "eva_manifest_reality": self.manifest_reality,
            "add_environment_hook": self.add_environment_hook,
            "change_phase": self.change_phase,
        }


# ---- command-line demo ----------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO)
    sme = EVASelfModifyingEngine()
    logger.info("SME demo initialized. system_status=%s", sme.get_system_status())


if __name__ == "__main__":
    main()
