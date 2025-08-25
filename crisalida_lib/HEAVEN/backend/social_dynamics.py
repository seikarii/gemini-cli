"""
Janus SocialDynamics - Gestión avanzada de grupos, cohesión, artefactos culturales y dinámica evolutiva.
Incluye métricas de cohesión, historial de artefactos, integración con sistemas de qualia y evolución cultural.
"""

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)


class Group(BaseModel):
    group_id: str
    members: list[str] = Field(default_factory=list)
    group_qualia: dict[str, float] = Field(default_factory=dict)
    cultural_artifacts: list[str] = Field(default_factory=list)
    group_cohesion: float = 0.0
    creation_time: float = Field(default_factory=lambda: time.time())


class SocialDynamics:
    def __init__(self):
        self.groups: dict[str, Group] = {}
        self.artifact_history: dict[str, list[str]] = {}
        self.group_creation_times: dict[str, float] = {}

    def create_group(self, group_id: str, member_ids: list[str]) -> Group | None:
        if group_id in self.groups:
            return None
        group = Group(group_id=group_id, members=member_ids)
        self.groups[group_id] = group
        self.group_creation_times[group_id] = group.creation_time
        return group

    def get_group(self, group_id: str) -> Group | None:
        return self.groups.get(group_id)

    def get_all_groups(self) -> list[dict]:
        return [group.model_dump() for group in self.groups.values()]

    def add_member_to_group(self, group_id: str, member_id: str) -> bool:
        group = self.get_group(group_id)
        if group and member_id not in group.members:
            group.members.append(member_id)
            return True
        return False

    def add_cultural_artifact(self, group_id: str, artifact: str) -> bool:
        group = self.get_group(group_id)
        if group:
            group.cultural_artifacts.append(artifact)
            self.artifact_history.setdefault(group_id, []).append(artifact)
            return True
        return False

    def get_group_cohesion(self, group_id: str) -> float:
        group = self.get_group(group_id)
        if not group or not group.members:
            return 0.0
        # Cohesión basada en número de miembros y artefactos culturales
        cohesion = 0.5 + 0.1 * len(group.cultural_artifacts)
        cohesion += 0.05 * len(group.members)
        return min(1.0, cohesion)

    def get_group_creation_time(self, group_id: str) -> float:
        return self.group_creation_times.get(group_id, 0.0)

    def get_group_artifacts(self, group_id: str) -> list[str]:
        group = self.get_group(group_id)
        return group.cultural_artifacts if group else []

    def get_cultural_artifact_count(self) -> int:
        return sum(len(group.cultural_artifacts) for group in self.groups.values())

    def get_all_cultural_artifacts(self) -> list[str]:
        artifacts = []
        for group in self.groups.values():
            artifacts.extend(group.cultural_artifacts)
        return artifacts

    def get_entity_groups(self, entity_id: str) -> list[str]:
        return [
            group.group_id
            for group in self.groups.values()
            if entity_id in group.members
        ]


class EVAGroup(Group):
    eva_experiences: list[dict] = Field(default_factory=list)
    eva_phase: str = "default"


class EVASocialDynamics(SocialDynamics):
    """
    SocialDynamics extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar experiencias sociales y grupales como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.
    """

    def __init__(self, eva_phase: str = "default"):
        super().__init__()
        self.eva_phase = eva_phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def create_group(self, group_id: str, member_ids: list[str]) -> EVAGroup | None:
        if group_id in self.groups:
            return None
        group = EVAGroup(
            group_id=group_id, members=member_ids, eva_phase=self.eva_phase
        )
        self.groups[group_id] = group
        self.group_creation_times[group_id] = group.creation_time
        return group

    def ingest_group_experience(
        self,
        group_id: str,
        experience_data: dict,
        qualia_state: QualiaState = None,
        phase: str = None,
    ) -> str:
        """
        Compila una experiencia grupal/social en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=experience_data.get("emotional_valence", 0.7),
            cognitive_complexity=experience_data.get("cognitive_complexity", 0.8),
            consciousness_density=experience_data.get("consciousness_density", 0.7),
            narrative_importance=experience_data.get("narrative_importance", 0.8),
            energy_level=experience_data.get("energy_level", 1.0),
        )
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_group_{group_id}_{hash(str(experience_data))}"
        )
        intention = {
            "intention_type": "ARCHIVE_GROUP_EXPERIENCE",
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
            timestamp=experience_data.get("timestamp", time.time()),
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self.eva_experience_store[experience_id] = reality_bytecode
        group = self.get_group(group_id)
        if group and hasattr(group, "eva_experiences"):
            group.eva_experiences.append(
                reality_bytecode.to_dict()
                if hasattr(reality_bytecode, "to_dict")
                else {}
            )
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                print(f"[EVA-SOCIAL] Environment hook failed: {e}")
        return experience_id

    def recall_group_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia grupal almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA group experience"}
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
                            print(f"[EVA-SOCIAL] Manifestation hook failed: {e}")
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
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia grupal EVA.
        """
        import time

        intention = {
            "intention_type": "ARCHIVE_GROUP_EXPERIENCE",
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
            timestamp=experience_data.get("timestamp", time.time()),
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
                print(f"[EVA-SOCIAL] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia grupal EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "ingest_group_experience": self.ingest_group_experience,
            "recall_group_experience": self.recall_group_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
