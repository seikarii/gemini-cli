from collections.abc import Callable
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode


class ProgressionSystem:
    """Manages the Demiurge's level, experience, and skill points."""

    def __init__(self):
        self.level = 1
        self.experience = 0
        self.experience_to_next_level = 100
        self.skill_points = 0

    def add_experience(self, amount: int):
        """Adds experience and checks for level ups."""
        self.experience += amount
        while self.experience >= self.experience_to_next_level:
            self.level_up()

    def level_up(self):
        """Handles the logic for leveling up."""
        self.level += 1
        self.experience -= self.experience_to_next_level
        self.experience_to_next_level = int(self.experience_to_next_level * 1.5)
        self.skill_points += 1
        print(f"[PROGRESSION] Demiurge has reached Level {self.level}!")

    def get_state(self):
        return {
            "level": self.level,
            "experience": self.experience,
            "experience_to_next_level": self.experience_to_next_level,
            "skill_points": self.skill_points,
        }

    def set_state(self, state):
        self.level = state.get("level", 1)
        self.experience = state.get("experience", 0)
        self.experience_to_next_level = state.get("experience_to_next_level", 100)
        self.skill_points = state.get("skill_points", 0)


class AbilityAcquisitionSystem:
    """Gestiona la emergencia de nuevas habilidades para el Demiurgo.
    Las habilidades se desbloquean al cumplir condiciones específicas en el universo.
    """

    def __init__(self, progression_system: ProgressionSystem):
        self.progression_system = progression_system
        self.ability_triggers = self._define_ability_triggers()

    def _define_ability_triggers(self) -> dict[str, Any]:
        """Define las condiciones para que emerjan nuevas habilidades."""
        return {
            "ElementalFocus_Fire": {
                "condition": lambda state: state["fire_actions_taken"] > 100
                and self.progression_system.level >= 2,
                "ability": "Focus Fire: Concentrate fire energy in a single point.",
                "unlocked": False,
            },
            "RealityStabilization_Lv1": {
                "condition": lambda state: state["entropy_crises_survived"] > 5
                and self.progression_system.level >= 3,
                "ability": "Stabilize Reality: Reduce local entropy at a high cost.",
                "unlocked": False,
            },
            "LinguisticInsight": {
                "condition": lambda state: state["new_words_learned"] > 10,
                "ability": "Linguistic Insight: Understand the intent behind unknown words.",
                "unlocked": False,
            },
            # EVA: Habilidad avanzada por memoria viviente
            "EVA_MemoryRecall": {
                "condition": lambda state: state.get("eva_experiences_recalled", 0)
                > 20,
                "ability": "EVA Memory Recall: Manifest and simulate past EVA experiences.",
                "unlocked": False,
            },
            "EVA_PhaseShift": {
                "condition": lambda state: state.get("eva_phases_visited", 0) > 5,
                "ability": "EVA Phase Shift: Change active memory phase/timeline.",
                "unlocked": False,
            },
            "EVA_RealityBytecodeManifest": {
                "condition": lambda state: state.get("eva_bytecodes_executed", 0) > 10,
                "ability": "EVA Reality Manifestation: Execute and manifest RealityBytecode directly.",
                "unlocked": False,
            },
        }

    def check_for_emergent_abilities(
        self, demiurge_avatar, universe_state
    ) -> list[str]:
        """Comprueba el estado del jugador y del universo para ver si se han
        desbloqueado nuevas habilidades.
        """
        newly_unlocked = []
        # Crear un estado resumido para evaluar las condiciones
        evaluation_state = {
            "fire_actions_taken": demiurge_avatar.get_action_count("fire"),
            "entropy_crises_survived": demiurge_avatar.get_stat("crises_survived"),
            "new_words_learned": len(demiurge_avatar.known_words),
            # EVA: Métricas de memoria viviente
            "eva_experiences_recalled": getattr(
                demiurge_avatar, "eva_experiences_recalled", 0
            ),
            "eva_phases_visited": getattr(demiurge_avatar, "eva_phases_visited", 0),
            "eva_bytecodes_executed": getattr(
                demiurge_avatar, "eva_bytecodes_executed", 0
            ),
        }
        for ability_name, trigger in self.ability_triggers.items():
            if not trigger["unlocked"] and trigger["condition"](evaluation_state):
                trigger["unlocked"] = True
                demiurge_avatar.add_ability(ability_name, trigger["ability"])
                newly_unlocked.append(ability_name)
                # Give experience for unlocking an ability
                self.progression_system.add_experience(50)
        return newly_unlocked


class EVAAbilityAcquisitionSystem(AbilityAcquisitionSystem):
    """
    EVAAbilityAcquisitionSystem - Sistema perfeccionado y extendido para integración con EVA.
    Gestiona la emergencia, ingestión y simulación de habilidades como experiencias vivientes (RealityBytecode),
    soporta ingestión/recall, faseo, hooks de entorno y gestión avanzada de memoria viviente EVA.
    """

    def __init__(self, progression_system: ProgressionSystem, phase: str = "default"):
        super().__init__(progression_system)
        # runtime and stores
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []

    def eva_ingest_ability_experience(
        self,
        demiurge_avatar,
        ability_name: str,
        ability_description: str,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de adquisición de habilidad en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=1.0,
            cognitive_complexity=0.8,
            consciousness_density=0.7,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        experience_id = f"eva_ability_{ability_name}_{int(time.time())}"
        experience_data = {
            "avatar_id": getattr(demiurge_avatar, "entity_id", None),
            "ability_name": ability_name,
            "ability_description": ability_description,
            "level": getattr(self.progression_system, "level", 1),
            "skill_points": getattr(self.progression_system, "skill_points", 0),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_ABILITY_ACQUISITION_EXPERIENCE",
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
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self.eva_experience_store[experience_id] = reality_bytecode
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                print(f"[EVA-ABILITY-ACQUISITION] Environment hook failed: {e}")
        return experience_id

    def eva_recall_ability_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de adquisición de habilidad almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA ability acquisition experience"}
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
                            print(
                                f"[EVA-ABILITY-ACQUISITION] Manifestation hook failed: {e}"
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
        self,
        experience_id: str,
        phase: str,
        demiurge_avatar,
        ability_name: str,
        ability_description: str,
        qualia_state: QualiaState = None,
    ):
        """
        Añade una fase alternativa para una experiencia de adquisición de habilidad EVA.
        """
        import time

        experience_data = {
            "avatar_id": getattr(demiurge_avatar, "entity_id", None),
            "ability_name": ability_name,
            "ability_description": ability_description,
            "level": getattr(self.progression_system, "level", 1),
            "skill_points": getattr(self.progression_system, "skill_points", 0),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_ABILITY_ACQUISITION_EXPERIENCE",
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
                print(f"[EVA-ABILITY-ACQUISITION] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de adquisición de habilidad EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_ability_experience": self.eva_ingest_ability_experience,
            "eva_recall_ability_experience": self.eva_recall_ability_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
