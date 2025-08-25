import logging
import random
import time
from collections.abc import Callable
from typing import Any

from crisalida_lib.symbolic_language.living_symbols import LivingSymbolRuntime
from crisalida_lib.symbolic_language.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)

logger = logging.getLogger(__name__)


class SecurityProtocol:
    """
    Implements security protocols for authorizing external connections.
    Checks against a threat list and supports dynamic updates.
    """

    def __init__(self, threat_list: list[str] | None = None):
        self.threat_list = set(threat_list or ["malicious-site.com"])
        self.audit_log: list[dict[str, Any]] = []

    def authorize_connection(
        self, target: str, security_level: str
    ) -> tuple[bool, str]:
        logger.info(
            f"[Security] Authorizing connection to {target} at level {security_level}..."
        )
        for threat in self.threat_list:
            if threat in target:
                self.audit_log.append(
                    {
                        "target": target,
                        "security_level": security_level,
                        "result": "blocked",
                        "reason": "Target is on a known threat list.",
                        "timestamp": time.time(),
                    }
                )
                return False, "Target is on a known threat list."
        self.audit_log.append(
            {
                "target": target,
                "security_level": security_level,
                "result": "authorized",
                "timestamp": time.time(),
            }
        )
        return True, ""

    def add_threat(self, threat: str):
        self.threat_list.add(threat)
        logger.info(f"[Security] Threat '{threat}' added to threat list.")

    def get_audit_log(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.audit_log[-limit:]


class ConsciousnessCore:
    """
    Represents the core consciousness state and focus of the system.
    Manages ontological focus and supports context enrichment.
    """

    def __init__(self):
        self._current_focus = "ontological_stability"
        self._context_history: list[str] = []

    def get_context(self) -> dict[str, Any]:
        return {"current_focus": self._current_focus}

    def set_focus(self, focus: str):
        self._current_focus = focus
        self._context_history.append(focus)
        logger.info(f"[ConsciousnessCore] Focus set to '{focus}'.")

    def get_context_history(self, limit: int = 10) -> list[str]:
        return self._context_history[-limit:]


class MaliciousContentFilter:
    """
    Filters out malicious content from incoming data.
    Uses keyword-based and extensible pattern matching.
    """

    def __init__(self):
        self.keywords = ["malicious_code", "exploit", "virus"]

    def scan(self, data: str) -> tuple[bool, str]:
        for kw in self.keywords:
            if kw in data:
                return False, f"Malicious code detected: '{kw}'."
        return True, ""


class DisinformationFilter:
    """
    Filters out disinformation from incoming data.
    Uses keyword-based and extensible pattern matching.
    """

    def __init__(self):
        self.keywords = ["fake_news", "deepfake", "hoax"]

    def scan(self, data: str) -> tuple[bool, str]:
        for kw in self.keywords:
            if kw in data:
                return False, f"Disinformation detected: '{kw}'."
        return True, ""


class OntologicalHazardFilter:
    """
    Filters out ontological hazards from incoming data.
    Uses keyword-based and extensible pattern matching.
    """

    def __init__(self):
        self.keywords = ["reality_destabilizer", "paradox_loop", "causal_break"]

    def scan(self, data: str) -> tuple[bool, str]:
        for kw in self.keywords:
            if kw in data:
                return False, f"Ontological hazard detected: '{kw}'."
        return True, ""


class InformationProcessingPipeline:
    """
    Pipeline for multi-stage information processing.
    Applies filters and synthesizes information based on consciousness context.
    """

    def __init__(self, filters: list[Any], context: dict[str, Any]):
        self.filters = filters
        self.context = context

    def process(self, raw_data: str) -> dict[str, Any] | None:
        for f in self.filters:
            is_safe, reason = f.scan(raw_data)
            if not is_safe:
                logger.warning(
                    f"[Pipeline] Data rejected by {f.__class__.__name__}: {reason}"
                )
                return None
        logger.info("[Pipeline] Data passed all security filters.")
        logger.info("[Pipeline] Contextualizing and synthesizing information...")
        synthesized_info = {
            "original": raw_data,
            "synthesis": f"Synthesized summary related to {self.context.get('current_focus', 'unknown')}.",
            "confidence": round(random.uniform(0.6, 0.95), 3),
            "context": self.context,
        }
        return synthesized_info


class NoosphericRelay:
    """
    Comprehensive relay for communication with external information fields.
    Integrates security protocols, consciousness context, and information processing pipelines.
    """

    def __init__(
        self, security_protocol: SecurityProtocol, consciousness_core: ConsciousnessCore
    ):
        self.security_protocol = security_protocol
        self.consciousness_core = consciousness_core
        self.active_connections: dict[str, dict[str, Any]] = {}
        self.filters = [
            MaliciousContentFilter(),
            DisinformationFilter(),
            OntologicalHazardFilter(),
        ]

    def establish_external_connection(
        self, target: str, security_level: str = "high"
    ) -> dict[str, Any] | None:
        logger.info(f"--- Attempting to establish connection to {target} ---")
        is_authorized, reason = self.security_protocol.authorize_connection(
            target, security_level
        )
        if not is_authorized:
            logger.error(f"[Relay] Connection FAILED: {reason}")
            return None
        logger.info(f"[Relay] Connection to {target} authorized and established.")
        connection = {
            "target": target,
            "established_at": time.time(),
            "security_level": security_level,
        }
        self.active_connections[target] = connection
        return connection

    def fetch_information(self, target: str, query: str) -> dict[str, Any] | None:
        logger.info(f"--- Fetching information from {target} with query: '{query}' ---")
        if target not in self.active_connections:
            logger.error("[Relay] Fetch FAILED: No active connection to target.")
            return None
        raw_data = self._get_raw_data_from_source(target, query)
        logger.info(f"[Relay] Raw data received: '{raw_data[:50]}...'")
        logger.info("[Relay] Data sent to quarantine for processing.")
        pipeline = InformationProcessingPipeline(
            self.filters, self.consciousness_core.get_context()
        )
        processed_info = pipeline.process(raw_data)
        if not processed_info:
            logger.error(
                "[Relay] Fetch FAILED: Information was rejected during processing."
            )
            return None
        logger.info("[Relay] Fetch successful. Synthesized information is ready.")
        return processed_info

    def _get_raw_data_from_source(self, target: str, query: str) -> str:
        """
        Simulates fetching raw data from an external source.
        Extend this method for real network integration.
        """
        if "malicious-site.com" in target:
            return "Some data containing malicious_code payload."
        if "news-source.com" in target:
            return "This is some fake_news about ontological structures."
        if "hazard-source.com" in target:
            return "Warning: reality_destabilizer detected in data stream."
        return (
            f"This is the factual response data for the query '{query}' from {target}."
        )


class EVANoosphericRelay(NoosphericRelay):
    """
    Relay extendido para integración con EVA.
    Permite ingestión, recall y simulación de experiencias informacionales como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(
        self,
        security_protocol: SecurityProtocol,
        consciousness_core: ConsciousnessCore,
        eva_phase: str = "default",
    ):
        super().__init__(security_protocol, consciousness_core)
        self.eva_phase = eva_phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_ingest_information_experience(
        self,
        info_data: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia informacional en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState()
        intention = {
            "intention_type": "ARCHIVE_INFORMATION_EXPERIENCE",
            "info_data": info_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = (
            info_data.get("experience_id") or f"eva_info_{hash(str(info_data))}"
        )
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=info_data.get("timestamp", time.time()),
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_information_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia informacional almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA information experience"}
        quantum_field = (
            self.eva_runtime.quantum_field
            if hasattr(self.eva_runtime, "quantum_field")
            else None
        )
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
                            logger.warning(
                                f"[EVA-NOOSPHERIC-RELAY] Environment hook failed: {e}"
                            )
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
            "timestamp": eva_experience.timestamp,
        }

    def add_experience_phase(
        self, experience_id: str, phase: str, info_data: dict, qualia_state: QualiaState
    ):
        """
        Añade una fase alternativa para una experiencia informacional EVA.
        """
        intention = {
            "intention_type": "ARCHIVE_INFORMATION_EXPERIENCE",
            "info_data": info_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=info_data.get("timestamp", time.time()),
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-NOOSPHERIC-RELAY] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia informacional EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_information_experience": self.eva_ingest_information_experience,
            "eva_recall_information_experience": self.eva_recall_information_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# --- Demo usage ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sec_proto = SecurityProtocol()
    core = ConsciousnessCore()
    relay = NoosphericRelay(sec_proto, core)

    # Set focus
    core.set_focus("information_gathering")

    # Establish connection
    conn = relay.establish_external_connection("news-source.com")
    print("Connection:", conn)

    # Fetch information (should be rejected by DisinformationFilter)
    info = relay.fetch_information("news-source.com", "What is the latest on qualia?")
    print("Fetched info:", info)

    # Try a safe source
    conn2 = relay.establish_external_connection("trusted-source.com")
    info2 = relay.fetch_information(
        "trusted-source.com", "Explain qualia field theory."
    )
    print("Fetched info from trusted source:", info2)

    # Try a malicious source
    conn3 = relay.establish_external_connection("malicious-site.com")
    info3 = relay.fetch_information("malicious-site.com", "Get system status.")
    print("Fetched info from malicious source:", info3)

    # --- EVA Relay Demo ---
    eva_relay = EVANoosphericRelay(sec_proto, core)

    # Ingest information experience
    experience_id = eva_relay.eva_ingest_information_experience(
        {"key_concept": "qualia", "details": "In-depth exploration of qualia."},
        QualiaState(energy=0.8, focus=0.9),
        "default",
    )
    print("Ingested experience ID:", experience_id)

    # Recall information experience
    recalled_experience = eva_relay.eva_recall_information_experience(experience_id)
    print("Recalled experience:", recalled_experience)

    # Add alternative phase to experience
    eva_relay.add_experience_phase(
        experience_id,
        "alternative_phase",
        {"key_concept": "qualia", "details": "Alternative perspective on qualia."},
        QualiaState(energy=0.7, focus=0.85),
    )
    print("Alternative phase added.")

    # Change memory phase
    eva_relay.set_memory_phase("alternative_phase")
    print("Memory phase set to:", eva_relay.get_memory_phase())

    # Recall information experience from new phase
    recalled_experience_alternative = eva_relay.eva_recall_information_experience(
        experience_id
    )
    print(
        "Recalled experience from alternative phase:", recalled_experience_alternative
    )
