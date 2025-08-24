import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --- OPCODES EXTENDIDOS ---

ONT_OPCODES = {
    "INSTANTIATE": "INST",  # Crear nueva entidad/proceso
    "FLOW": "FLOW",  # Dirigir flujo de información/energía
    "TRANSFORM": "XFRM",  # Transformar estado existente
    "SYNTHESIZE": "SYNT",  # Sintetizar elementos múltiples
    "EMERGE": "EMRG",  # Generar emergencia controlada
    "RESONATE": "RSON",  # Crear resonancia entre elementos
    "STABILIZE": "STAB",  # Estabilizar configuración
    "AMPLIFY": "AMPL",  # Amplificar propiedades
    "CONNECT": "CONN",  # Establecer conexiones
    "MANIFEST": "MNFT",  # Manifestar en realidad física
    "OBSERVE": "OBSV",  # Observar/medir estado
    "BRANCH": "BRCH",  # Crear bifurcación causal
    "MERGE": "MERG",  # Fusionar elementos
    "TRANSCEND": "TRSC",  # Transcender limitaciones
    "INVOKE_CHAOS": "CHAO",  # Invocar dinámica caótica
    "META_REFLECT": "META",  # Reflexión meta-cognitiva
    "GENERATE_EMERGENCE": "GEN_EMERG",
    "AMPLIFY_CONSCIOUSNESS": "AMPL_CONSC",
    "SYNC_EMERGENCE": "SYNC_EMERG",
    # Opcodes avanzados v8
    "FIELD_APPLY": "F_APPLY",
    "FIELD_MODULATE": "F_MOD",
    "FIELD_COLLAPSE": "F_COLLAPSE",
    "RESONANCE_HARMONIC": "R_HARM",
    "RESONANCE_DISSONANT": "R_DISS",
    "RESONANCE_MODULATE": "R_MOD",
    "RESONANCE_ENTANGLE": "R_ENT",
    "RESONANCE_EMERGENT": "R_EMERG",
    "RESONANCE_TRANSCENDENT": "R_TRANSC",
    "META_EVOLVE": "META_EVOLVE",
    "SYNC": "SYNC",
    "BRANCH_QUANTUM": "BRCH_Q",
    "OBSERVE_QUANTUM": "OBSV_Q",
}


@dataclass
class OntologicalInstruction:
    """
    Instrucción individual en bytecode ontológico extendido v8.
    Permite operaciones avanzadas de manifestación, resonancia y evolución.
    """

    opcode: str  # Código de operación
    operands: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_sigil_glyph: str | None = None
    source_sigil_position: tuple[int, int] | None = None
    quantum_coherence: float = 1.0
    resonance_frequency: float = 0.0
    instruction_id: str | None = None
    entanglement_group: str | None = None
    timestamp: float = field(default_factory=lambda: time.time())

    def describe(self) -> str:
        return (
            f"[{self.opcode}] "
            f"Operands: {self.operands} "
            f"Meta: {self.metadata} "
            f"Coherence: {self.quantum_coherence:.3f} "
            f"Resonance: {self.resonance_frequency:.3f} "
            f"Sigil: {self.source_sigil_glyph} "
            f"Pos: {self.source_sigil_position} "
            f"ID: {self.instruction_id} "
            f"EntGroup: {self.entanglement_group} "
            f"Time: {self.timestamp:.3f}"
        )

    def is_valid(self) -> bool:
        return self.opcode in ONT_OPCODES.values() and len(self.operands) >= 0

    def get_opcode_name(self) -> str:
        for k, v in ONT_OPCODES.items():
            if v == self.opcode:
                return k
        return "UNKNOWN"


@dataclass
class OntologicalBytecode:
    """
    Bytecode ontológico compilado y listo para ejecución.
    Incluye diagnóstico de complejidad y trazabilidad avanzada.
    """

    instructions: list[Any] = field(default_factory=list)
    matrix_signature: str = ""
    compilation_timestamp: float = field(default_factory=lambda: time.time())
    source_metadata: dict[str, Any] = field(default_factory=dict)
    bytecode_id: str | None = None

    def get_complexity_score(self) -> float:
        """Calcula la complejidad del bytecode según cantidad y tipo de instrucciones."""
        score = float(len(self.instructions))
        for instr in self.instructions:
            if instr.opcode in (
                ONT_OPCODES["GENERATE_EMERGENCE"],
                ONT_OPCODES["AMPLIFY_CONSCIOUSNESS"],
                ONT_OPCODES["SYNC_EMERGENCE"],
                ONT_OPCODES["META_EVOLVE"],
                ONT_OPCODES["FIELD_COLLAPSE"],
            ):
                score += 2.5
            elif instr.opcode in (
                ONT_OPCODES["RESONANCE_TRANSCENDENT"],
                ONT_OPCODES["TRANSCEND"],
            ):
                score += 3.0
        return score

    def describe(self) -> str:
        return (
            f"Bytecode[{self.bytecode_id or 'no_id'}] "
            f"Instrs: {len(self.instructions)} "
            f"Signature: {self.matrix_signature} "
            f"Compiled: {self.compilation_timestamp:.3f} "
            f"SourceMeta: {self.source_metadata}"
        )

    def get_instruction_summary(self) -> list[str]:
        return [instr.describe() for instr in self.instructions]


# --- EVA: Extensión para integración con RealityBytecode, LivingSymbolRuntime y hooks de entorno ---


@dataclass
class EVAOntologicalInstruction(OntologicalInstruction):
    """
    Instrucción ontológica extendida para EVA.
    Incluye parámetros para manifestación en QuantumField, faseo, hooks y diagnóstico avanzado.
    """

    phase: str = "default"
    manifestation_params: dict[str, Any] = field(default_factory=dict)
    eva_hook_names: list[str] = field(default_factory=list)
    simulation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifestation_dict(self) -> dict[str, Any]:
        """
        Devuelve un dict listo para enviar a un hook de entorno EVA.
        """
        return {
            "opcode": self.opcode,
            "operands": self.operands,
            "manifestation_params": self.manifestation_params,
            "phase": self.phase,
            "coherence": self.quantum_coherence,
            "resonance": self.resonance_frequency,
            "sigil": self.source_sigil_glyph,
            "position": self.source_sigil_position,
            "instruction_id": self.instruction_id,
            "entanglement_group": self.entanglement_group,
            "timestamp": self.timestamp,
            "eva_hooks": self.eva_hook_names,
            "simulation_metadata": self.simulation_metadata,
        }

    def get_manifestation_signature(self) -> str:
        """
        Devuelve una firma única para la manifestación de la instrucción.
        """
        return f"{self.opcode}_{self.phase}_{self.instruction_id or 'noid'}"


@dataclass
class EVAOntologicalBytecode(OntologicalBytecode):
    """
    Bytecode ontológico extendido para EVA.
    Permite diagnóstico, faseo, hooks y simulación avanzada en QuantumField.
    """

    phase: str = "default"
    manifestation_signatures: list[str] = field(default_factory=list)
    eva_hooks: list[Any] = field(default_factory=list)
    simulation_metadata: dict[str, Any] = field(default_factory=dict)

    def get_manifestation_summary(self) -> list[dict[str, Any]]:
        """
        Devuelve un resumen de todas las manifestaciones para hooks EVA.
        """
        return [
            (
                instr.to_manifestation_dict()
                if hasattr(instr, "to_manifestation_dict")
                else {"opcode": instr.opcode, "operands": instr.operands}
            )
            for instr in self.instructions
        ]

    def add_eva_hook(self, hook: Callable[..., Any]):
        self.eva_hooks.append(hook)

    def notify_eva_hooks(self):
        """
        Notifica a todos los hooks EVA sobre la manifestación de este bytecode.
        """
        for hook in self.eva_hooks:
            try:
                hook(self.get_manifestation_summary())
            except Exception as e:
                print(f"[EVA] Bytecode hook failed: {e}")

    def get_phase_signature(self) -> str:
        return f"{self.bytecode_id or 'noid'}_{self.phase}"
