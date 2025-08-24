from collections.abc import Callable
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode

from .complex_being import ComplexBeing


class DemiurgeAvatar(ComplexBeing):
    """
    This module defines the `DemiurgeAvatar` class, representing the player's
    manifestation within the Janus Metacosmos. It extends `ComplexBeing` and
    includes attributes and methods specific to the player's interaction with
    the simulation, such as power level, known words, and acquired abilities.
    """

    def __init__(self, name="The Demiurge", entity_id=None):
        import uuid

        if entity_id is None:
            entity_id = f"demiurge_{uuid.uuid4()}"
        super().__init__(entity_id=entity_id)
        self.name = name
        self.power_level = 100.0
        self.known_words = set()  # For linguistic engine integration
        self.action_counts: dict[str, int] = {}  # For ability acquisition
        self.stats = {"crises_survived": 0}

    def update(self, ticks) -> Any:
        """The Demiurge's avatar updates its internal state, including power regeneration."""
        # Regenerate power over time
        if self.power_level < 100.0:
            self.power_level += 0.01 * ticks  # Simple linear regeneration
            self.power_level = min(self.power_level, 100.0)

    def get_action_count(self, action_type: str) -> int:
        return self.action_counts.get(action_type, 0)

    def add_ability(self, ability_name: str, description: str):
        print(f"[DEMIURGE] New ability unlocked: {ability_name} - {description}")
        # In a real implementation, this would add to a list of active abilities

    def get_stat(self, stat_name: str) -> Any:
        return self.stats.get(stat_name, 0)


class EVADemiurgeAvatar(DemiurgeAvatar):
    """
    Avatar del Demiurgo extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar manifestaciones, acciones y habilidades como experiencias vivientes (RealityBytecode),
    soporta faseo, hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(
        self,
        name: str = "The Demiurge",
        entity_id: str | None = None,
        phase: str | None = "default",
    ) -> None:
        super().__init__(name=name, entity_id=entity_id)
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_ingest_avatar_experience(
        self,
        action: str,
        details: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de acción/manifestación del avatar en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.8,
            cognitive_complexity=0.9,
            consciousness_density=0.85,
            narrative_importance=1.0,
            energy_level=self.power_level / 100.0,
        )
        experience_data = {
            "avatar_id": self.entity_id,
            "name": self.name,
            "action": action,
            "details": details or {},
            "power_level": self.power_level,
            "known_words": list(self.known_words),
            "action_counts": self.action_counts.copy(),
            "stats": self.stats.copy(),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_DEMIURGE_AVATAR_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = f"eva_avatar_{action}_{hash(str(experience_data))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_avatar_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia del avatar almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA DemiurgeAvatar experience"}
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
                            print(f"[EVA-AVATAR] Environment hook failed: {e}")
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
        self,
        experience_id: str,
        phase: str,
        action: str,
        details: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia del avatar EVA.
        """
        import time

        experience_data = {
            "avatar_id": self.entity_id,
            "name": self.name,
            "action": action,
            "details": details or {},
            "power_level": self.power_level,
            "known_words": list(self.known_words),
            "action_counts": self.action_counts.copy(),
            "stats": self.stats.copy(),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_DEMIURGE_AVATAR_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
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
                print(f"[EVA-AVATAR] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia del avatar EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_avatar_experience": self.eva_ingest_avatar_experience,
            "eva_recall_avatar_experience": self.eva_recall_avatar_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
