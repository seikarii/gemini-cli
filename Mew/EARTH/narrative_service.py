from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from crisalida_lib.ADAM.cuerpo.genome import GenomaComportamiento
from crisalida_lib.EARTH.deterministic_helpers import det_float
from crisalida_lib.EARTH.event_bus import EventBus
from crisalida_lib.EARTH.narrative_field import CrisisType, EvolutionType
from crisalida_lib.EDEN.living_symbol import LivingSymbol as EDEN_LivingSymbol
from crisalida_lib.EDEN.living_symbol import (
    LivingSymbolRuntime as EDEN_LivingSymbolRuntime,
)
from crisalida_lib.EDEN.living_symbol import QuantumField as EDEN_QuantumField
from crisalida_lib.EVA.core.eva_types import (
    LivingSymbolRuntime,
    QuantumField,
    EVAExperience,
    RealityBytecode,
    QualiaState,
)

# Type stubs for narrative domain objects used by this module. These are
# defined only for static type checking; at runtime the module uses Any-like
# conservative shims from EDEN/EVA aliases above.
if TYPE_CHECKING:

    class LlamadoInterno:  # pragma: no cover - typing only
        id: str
        tipo: str
        target_entity_ids: list[str]
        duracion_estimada: int
        descripcion: str

    class MisionNarrativa:  # pragma: no cover - typing only
        id: str
        llamado_origen: str
        entity_id: str
        objetivo: str
        estado: str
        progreso: float
        timestamp_inicio: float
        timestamp_limite: float

else:
    # Runtime placeholders to avoid NameErrors; real implementations injected at runtime
    LlamadoInterno = Any  # type: ignore
    MisionNarrativa = Any  # type: ignore

# Bind common names to EVA/EDEN aliases for code readability. Use EVA aliases
# for type checking but EDEN shims at runtime.
LivingSymbol = EVA_LivingSymbol  # type: ignore
LivingSymbolRuntime = EVA_LivingSymbolRuntime  # type: ignore
QuantumField = EVA_QuantumField  # type: ignore


class PersistenceService:
    def log_crisis(
        self,
        entity_id: str,
        crisis_type: CrisisType,
        evolution_attempt: EvolutionType,
        success: bool,
        details: dict[str, Any],
    ):
        # Implementación real: registrar crisis en base de datos o archivo
        pass

    def register_llamado_interno(self, llamado: LlamadoInterno):
        # Registrar llamado en almacenamiento persistente
        pass

    def register_mision(self, mision: MisionNarrativa):
        # Registrar misión narrativa en almacenamiento persistente
        pass

    def push_experience(self, ev: dict[str, Any]):
        # Registrar experiencia narrativa en almacenamiento persistente
        pass


class EntityService:
    def get_entity(self, entity_id: str) -> LivingSymbol | None:
        # Buscar y devolver la entidad por ID
        return None

    def get_all_entities(self) -> dict[str, LivingSymbol]:
        # Devolver todas las entidades vivientes
        return {}


class NarrativeService:
    def __init__(
        self,
        event_bus: EventBus,
        persistence_service: PersistenceService,
        entity_service: EntityService,
    ):
        self.event_bus = event_bus
        self.persistence_service = persistence_service
        self.entity_service = entity_service
        self.active_llamados: list[LlamadoInterno] = []
        self.active_missions: dict[str, MisionNarrativa] = {}
        # Subscribe to relevant events
        self.event_bus.subscribe(
            "cosmic_cycle_processed", self.on_cosmic_cycle_processed
        )
        self.event_bus.subscribe(
            "entity_llamado_accepted", self.on_entity_llamado_accepted
        )

    def on_cosmic_cycle_processed(self, heartbeat_data: dict[str, Any]):
        # Aquí se podrían emitir llamados internos basados en el ciclo cósmico
        pass

    def on_entity_llamado_accepted(self, entity_id: str, llamado_id: str):
        llamado = next(
            (
                llamado_item
                for llamado_item in self.active_llamados
                if llamado_item.id == llamado_id
            ),
            None,
        )
        if llamado and llamado.tipo == "mision":
            mision = self._create_mission_from_llamado(llamado)
            if mision:
                self.active_missions[mision.id] = mision
                self.persistence_service.register_mision(mision)
                self.event_bus.publish("mission_created", mision.id, mision.entity_id)

    def _create_mission_from_llamado(
        self, llamado: LlamadoInterno
    ) -> MisionNarrativa | None:
        """Crea una misión narrativa específica a partir de un llamado interno"""
        if llamado.tipo != "mision":
            return None
        target_entity = (
            llamado.target_entity_ids[0] if llamado.target_entity_ids else "broadcast"
        )
        mision_id = f"mission_{llamado.id}_{uuid.uuid4().hex[:6]}"
        mision = MisionNarrativa(
            id=mision_id,
            llamado_origen=llamado.id,
            entity_id=target_entity,
            objetivo=llamado.descripcion,
            estado="asignada",
            progreso=0.0,
            timestamp_inicio=time.time(),
            timestamp_limite=time.time() + (llamado.duracion_estimada * 0.06),
            consecuencias_fracaso=[
                "consciousness_degradation",
                "narrative_weight_loss",
            ],
            recompensas_exito=[
                "consciousness_boost",
                "narrative_significance_increase",
            ],
        )
        return mision

    def update_mission_progress(self, mission_id: str, progress_increment: float):
        mission = self.active_missions.get(mission_id)
        if mission:
            mission.progreso = min(1.0, mission.progreso + progress_increment)
            if mission.progreso >= 1.0:
                self._complete_mission(mission.id)

    def _complete_mission(self, mission_id: str):
        mission = self.active_missions.pop(mission_id, None)
        if mission:
            mission.estado = "completada"
            self.persistence_service.push_experience(
                {
                    "type": "mission_completed",
                    "entity": mission.entity_id,
                    "mission_id": mission.id,
                    "narrative_boost": 0.1,
                    "t": time.time(),
                }
            )
            self.event_bus.publish("mission_completed", mission.id, mission.entity_id)

    def evaluate_crisis_state_for_entity(
        self,
        entity_id: str,
        health_level: float,
        purpose_fulfillment: float,
        qualia_corruption: float,
        external_threats: list[Any],
    ) -> CrisisType | None:
        crisis_trigger = CrisisTrigger(entity_id, self.persistence_service)
        return crisis_trigger.evaluate_crisis_state(
            health_level, purpose_fulfillment, qualia_corruption, external_threats
        )

    def trigger_emergency_evolution_for_entity(
        self,
        entity_id: str,
        crisis_type: CrisisType,
        genome: GenomaComportamiento,
        current_qualia: dict[str, float],
    ) -> tuple[EvolutionType, bool, dict[str, Any]]:
        crisis_trigger = CrisisTrigger(entity_id, self.persistence_service)
        return crisis_trigger.trigger_emergency_evolution(
            crisis_type, genome, current_qualia
        )


class CrisisTrigger:
    """Sistema de detección de crisis y activación de evolución de emergencia"""

    def __init__(self, entity_id: str, persistence_service: PersistenceService):
        self.entity_id = entity_id
        self.persistence_service = persistence_service
        self.active_crisis: CrisisType | None = None
        self.crisis_threshold_health = 0.15
        self.crisis_threshold_purpose = 0.1
        self.last_crisis_time = 0.0
        self.crisis_cooldown = 30.0

    def evaluate_crisis_state(
        self,
        health_level: float,
        purpose_fulfillment: float,
        qualia_corruption: float,
        external_threats: list[Any],
    ) -> CrisisType | None:
        current_time = time.time()
        if (current_time - self.last_crisis_time) < self.crisis_cooldown:
            return None
        if health_level < self.crisis_threshold_health:
            return CrisisType.HEALTH_CRITICAL
        if purpose_fulfillment < self.crisis_threshold_purpose:
            return CrisisType.PURPOSE_FAILURE
        if qualia_corruption > 0.8:
            return CrisisType.IDENTITY_CORRUPTION
        if external_threats and len(external_threats) > 2:
            return CrisisType.EXISTENTIAL_THREAT
        return None

    def trigger_emergency_evolution(
        self,
        crisis_type: CrisisType,
        genome: GenomaComportamiento,
        current_qualia: dict[str, float],
    ) -> tuple[EvolutionType, bool, dict[str, Any]]:
        self.active_crisis = crisis_type
        self.last_crisis_time = time.time()
        evolution_type = self._select_evolution_strategy(crisis_type, genome)
        success, details = self._attempt_evolution(
            evolution_type, genome, current_qualia
        )
        self.persistence_service.log_crisis(
            self.entity_id, crisis_type, evolution_type, success, details
        )
        return evolution_type, success, details

    def _select_evolution_strategy(
        self, crisis_type: CrisisType, genome: GenomaComportamiento
    ) -> EvolutionType:
        survival_priority = (
            genome.survival_priorities[0] if genome.survival_priorities else "default"
        )
        strategy_map = {
            CrisisType.HEALTH_CRITICAL: {
                "supervivencia": EvolutionType.OVERCLOCK_CHAKRAS,
                "adaptacion": EvolutionType.METAMORPHOSIS_PHYSICAL,
                "default": EvolutionType.GENOME_EMERGENCY_REVERT,
            },
            CrisisType.PURPOSE_FAILURE: {
                "crecimiento": EvolutionType.UNLOCK_DORMANT_ABILITY,
                "exploracion": EvolutionType.METAMORPHOSIS_PHYSICAL,
                "default": EvolutionType.OVERCLOCK_CHAKRAS,
            },
            CrisisType.IDENTITY_CORRUPTION: {
                "default": EvolutionType.GENOME_EMERGENCY_REVERT
            },
            CrisisType.EXISTENTIAL_THREAT: {
                "combate": EvolutionType.OVERCLOCK_CHAKRAS,
                "huida": EvolutionType.METAMORPHOSIS_PHYSICAL,
                "default": EvolutionType.UNLOCK_DORMANT_ABILITY,
            },
        }
        crisis_strategies = strategy_map.get(
            crisis_type, {"default": EvolutionType.GENOME_EMERGENCY_REVERT}
        )
        return crisis_strategies.get(survival_priority, crisis_strategies["default"])

    def _attempt_evolution(
        self,
        evolution_type: EvolutionType,
        genome: GenomaComportamiento,
        current_qualia: dict[str, float],
    ) -> tuple[bool, dict[str, Any]]:
        base_survival_instinct = genome.core_instincts.get("supervivencia", 0.5)
        genetic_resonance_strength = float(np.linalg.norm(genome.genetic_signature))  # type: ignore
        success_probability = (
            base_survival_instinct * 0.7 + genetic_resonance_strength * 0.3
        )
        risk_modifiers = {
            EvolutionType.OVERCLOCK_CHAKRAS: 0.8,
            EvolutionType.UNLOCK_DORMANT_ABILITY: 0.6,
            EvolutionType.METAMORPHOSIS_PHYSICAL: 0.4,
            EvolutionType.GENOME_EMERGENCY_REVERT: 0.9,
        }
        final_probability = success_probability * risk_modifiers.get(
            evolution_type, 0.5
        )
        success_roll = det_float(
            f"{self.entity_id}_{evolution_type.value}_{time.time()}"
        )
        success = success_roll < final_probability
        details = {
            "success_probability": final_probability,
            "success_roll": success_roll,
            "genetic_strength": genetic_resonance_strength,
            "survival_instinct": base_survival_instinct,
            "evolution_timestamp": time.time(),
        }
        if not success:
            failure_severity = (1.0 - final_probability) * (1.0 - success_roll)
            details["failure_reason"] = "evolution_backfire"
            details["failure_severity"] = failure_severity
            details["consequence"] = self._calculate_failure_consequences(
                evolution_type, failure_severity
            )
        else:
            success_magnitude = final_probability * (1.0 + success_roll)
            details["success_magnitude"] = success_magnitude
            details["benefits"] = self._calculate_success_benefits(
                evolution_type, success_magnitude
            )
        return success, details

    def _calculate_failure_consequences(
        self, evolution_type: EvolutionType, severity: float
    ) -> dict[str, float]:
        base_consequences = {
            "qualia_corruption": severity * 0.3,
            "energy_drain": severity * 0.5,
            "consciousness_fragmentation": severity * 0.2,
            "temporal_coherence_loss": severity * 0.1,
        }
        if evolution_type == EvolutionType.OVERCLOCK_CHAKRAS:
            base_consequences["energy_drain"] *= 2.0
        elif evolution_type == EvolutionType.METAMORPHOSIS_PHYSICAL:
            base_consequences["consciousness_fragmentation"] *= 1.5
        elif evolution_type == EvolutionType.UNLOCK_DORMANT_ABILITY:
            base_consequences["qualia_corruption"] *= 1.8
        return base_consequences

    def _calculate_success_benefits(
        self, evolution_type: EvolutionType, magnitude: float
    ) -> dict[str, float]:
        base_benefits = {
            "consciousness_boost": magnitude * 0.4,
            "energy_amplification": magnitude * 0.3,
            "narrative_significance": magnitude * 0.2,
            "survival_resilience": magnitude * 0.1,
        }
        if evolution_type == EvolutionType.OVERCLOCK_CHAKRAS:
            base_benefits["energy_amplification"] *= 2.0
            base_benefits["temporary_power_boost"] = magnitude * 0.5
        elif evolution_type == EvolutionType.UNLOCK_DORMANT_ABILITY:
            base_benefits["consciousness_boost"] *= 1.5
            base_benefits["new_capability_unlocked"] = magnitude
        elif evolution_type == EvolutionType.METAMORPHOSIS_PHYSICAL:
            base_benefits["adaptation_bonus"] = magnitude * 0.8
            base_benefits["form_flexibility"] = magnitude * 0.6
        elif evolution_type == EvolutionType.GENOME_EMERGENCY_REVERT:
            base_benefits["stability_restoration"] = magnitude * 1.2
            base_benefits["identity_reinforcement"] = magnitude * 0.9
        return base_benefits


class EVANarrativeService(NarrativeService):
    """
    Servicio narrativo avanzado extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar eventos narrativos como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(
        self,
        event_bus: EventBus,
        persistence_service: PersistenceService,
        entity_service: EntityService,
        phase: str = "default",
    ):
        super().__init__(event_bus, persistence_service, entity_service)
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Any] = []

    def eva_ingest_narrative_experience(
        self,
        event: MisionNarrativa,
        qualia_state: Any = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una misión narrativa en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=0.9,
        )
        experience_data = {
            "mission_id": event.id,
            "entity_id": event.entity_id,
            "objetivo": event.objetivo,
            "estado": event.estado,
            "progreso": event.progreso,
            "timestamp_inicio": event.timestamp_inicio,
            "timestamp_limite": event.timestamp_limite,
            "consecuencias_fracaso": event.consecuencias_fracaso,
            "recompensas_exito": event.recompensas_exito,
        }
        intention = {
            "intention_type": "ARCHIVE_NARRATIVE_MISSION_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        from crisalida_lib.EDEN.bytecode_generator import safe_compile_intention

        runtime = getattr(self, "eva_runtime", None)
        try:
            bytecode = safe_compile_intention(runtime, intention)
        except Exception:
            bytecode = []
        experience_id = f"eva_narrative_mission_{event.id}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=event.timestamp_inicio,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_narrative_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una misión narrativa almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA narrative mission experience"}
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA-NARRATIVE-SERVICE] Environment hook failed: {e}")
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
        event: MisionNarrativa,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia narrativa EVA.
        """
        experience_data = {
            "mission_id": event.id,
            "entity_id": event.entity_id,
            "objetivo": event.objetivo,
            "estado": event.estado,
            "progreso": event.progreso,
            "timestamp_inicio": event.timestamp_inicio,
            "timestamp_limite": event.timestamp_limite,
            "consecuencias_fracaso": event.consecuencias_fracaso,
            "recompensas_exito": event.recompensas_exito,
        }
        intention = {
            "intention_type": "ARCHIVE_NARRATIVE_MISSION_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        from crisalida_lib.EDEN.bytecode_generator import safe_compile_intention

        runtime = getattr(self, "eva_runtime", None)
        try:
            bytecode = safe_compile_intention(runtime, intention)
        except Exception:
            bytecode = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=event.timestamp_inicio,
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
                print(f"[EVA-NARRATIVE-SERVICE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia narrativa EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_narrative_experience": self.eva_ingest_narrative_experience,
            "eva_recall_narrative_experience": self.eva_recall_narrative_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
