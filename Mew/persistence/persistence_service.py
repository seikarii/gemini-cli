from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from crisalida_lib.ADAM.cuerpo.genome import GenomaComportamiento
from crisalida_lib.EARTH.event_bus import EventBus
from crisalida_lib.EARTH.narrative_field import (
    CrisisType,
    LlamadoInterno,
    MisionNarrativa,
)

# Constants for paths
BASE_DIR = os.path.abspath("./fonografo_state_v2")
MODULES_DIR = os.path.join(BASE_DIR, "modules")
GENOME_DIR = os.path.join(BASE_DIR, "genomes")
NARRATIVES_DIR = os.path.join(BASE_DIR, "narratives")
STATE_FILE = os.path.join(BASE_DIR, "state.json")
LOG_FILE = os.path.join(BASE_DIR, "logs.txt")

for d in [MODULES_DIR, GENOME_DIR, NARRATIVES_DIR]:
    os.makedirs(d, exist_ok=True)


def log(msg: str) -> Any:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class PersistenceService:
    """
    Servicio de persistencia central para entidades, genomas, llamados internos, misiones y eventos de crisis.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.path = STATE_FILE
        self.state = self._load_state()
        log("PersistenceService initialized.")

    def _load_state(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {
                "id": str(uuid.uuid4()),
                "created": time.time(),
                "modules": {},
                "entities": {},
                "genomes": {},
                "llamados_internos": [],
                "misiones": {},
                "crisis_log": [],
                "qualia_crystals": [],
                "experiences": [],
            }
        return state

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        self.event_bus.publish("state_saved")

    def register_module(self, name: str, meta: dict[str, Any]):
        self.state["modules"][name] = meta
        self._save()
        log(f"persistence: registered module {name}")
        self.event_bus.publish("module_registered", name, meta)

    def register_entity(self, eid: str, meta: dict[str, Any]):
        self.state["entities"][eid] = meta
        self._save()
        log(f"persistence: entity registered {eid}")
        self.event_bus.publish("entity_registered", eid, meta)

    def register_genome(self, genome: GenomaComportamiento):
        """Registra genoma como ancla de identidad y persistencia."""
        self.state["genomes"][genome.entity_id] = genome.to_dict()
        genome_file = os.path.join(GENOME_DIR, f"{genome.entity_id}.json")
        with open(genome_file, "w", encoding="utf-8") as f:
            json.dump(genome.to_dict(), f, indent=2)
        self._save()
        log(f"persistence: genome registered for {genome.entity_id}")
        self.event_bus.publish("genome_registered", genome.entity_id, genome.to_dict())

    def load_genome(self, entity_id: str) -> GenomaComportamiento | None:
        """Carga genoma desde almacenamiento persistente."""
        genome_data = self.state["genomes"].get(entity_id)
        if not genome_data:
            genome_file = os.path.join(GENOME_DIR, f"{entity_id}.json")
            if os.path.exists(genome_file):
                with open(genome_file, encoding="utf-8") as f:
                    genome_data = json.load(f)
        if not genome_data:
            return None
        genome = GenomaComportamiento(
            entity_id=genome_data["entity_id"],
            core_instincts=genome_data["core_instincts"],
            survival_priorities=genome_data["survival_priorities"],
            immutable_truths=set(genome_data["immutable_truths"]),
            genetic_signature=np.array(genome_data["genetic_signature"]),
            creation_timestamp=genome_data["creation_timestamp"],
            backup_qualia_state=genome_data["backup_qualia_state"],
        )
        return genome

    def register_llamado_interno(self, llamado: LlamadoInterno):
        """Registra llamado interno de la mente cósmica."""
        llamado_dict = (
            llamado.to_dict() if hasattr(llamado, "to_dict") else asdict(llamado)
        )
        self.state["llamados_internos"].append(llamado_dict)
        if len(self.state["llamados_internos"]) > 1000:
            self.state["llamados_internos"] = self.state["llamados_internos"][-1000:]
        self._save()
        log(f"persistence: llamado interno {llamado.id} registered")
        self.event_bus.publish("llamado_registered", llamado.id, llamado_dict)

    def register_mision(self, mision: MisionNarrativa):
        """Registra misión narrativa."""
        mision_dict = mision.to_dict() if hasattr(mision, "to_dict") else asdict(mision)
        self.state["misiones"][mision.id] = mision_dict
        self._save()
        log(f"persistence: mission {mision.id} registered for {mision.entity_id}")
        self.event_bus.publish("mission_registered", mision.id, mision_dict)

    def log_crisis(
        self,
        entity_id: str,
        crisis_type: CrisisType,
        evolution_attempt: Any,
        success: bool,
        details: dict[str, Any],
    ):
        """Log de crisis y evolución de emergencia."""
        crisis_record = {
            "timestamp": time.time(),
            "entity_id": entity_id,
            "crisis_type": crisis_type.value,
            "evolution_attempt": (
                evolution_attempt.value
                if hasattr(evolution_attempt, "value")
                else str(evolution_attempt)
            ),
            "success": success,
            "details": details,
        }
        self.state["crisis_log"].append(crisis_record)
        if len(self.state["crisis_log"]) > 500:
            self.state["crisis_log"] = self.state["crisis_log"][-500:]
        self._save()
        log(
            f"persistence: crisis logged for {entity_id}: {crisis_type.value} -> {evolution_attempt} ({'success' if success else 'failure'})"
        )
        self.event_bus.publish("crisis_logged", entity_id, crisis_record)

    def push_experience(self, ev: dict[str, Any]):
        self.state["experiences"].append(ev)
        cap = 8192
        if len(self.state["experiences"]) > cap:
            self.state["experiences"] = self.state["experiences"][-cap:]
        self._save()
        self.event_bus.publish("experience_pushed", ev)

    def push_crystal(self, crystal: dict[str, Any]):
        self.state["qualia_crystals"].append(crystal)
        cap = 16384
        if len(self.state["qualia_crystals"]) > cap:
            self.state["qualia_crystals"] = self.state["qualia_crystals"][-cap:]
        self._save()
        self.event_bus.publish("qualia_crystal_pushed", crystal)

    def get_state(self) -> dict[str, Any]:
        return self.state


class EVAPersistenceService(PersistenceService):
    """
    Servicio de persistencia extendido para EVA.
    Permite almacenar, recuperar y simular experiencias vivientes, fases, hooks y benchmarking de memoria EVA.
    """

    def __init__(self, event_bus: EventBus, eva_phase: str = "default"):
        super().__init__(event_bus)
        self.eva_phase = eva_phase
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list = []

    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str = None
    ) -> str:
        """
        Compila y almacena una experiencia viviente EVA en la memoria local y la persistencia base.
        """
        phase = phase or self.eva_phase
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_exp_{hash(str(experience_data))}"
        )
        eva_record = {
            "experience_id": experience_id,
            "experience_data": experience_data,
            "qualia_state": qualia_state,
            "phase": phase,
            "timestamp": experience_data.get("timestamp", time.time()),
        }
        self.eva_memory_store[experience_id] = eva_record
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        self.eva_experience_store[experience_id] = eva_record
        # Persistir en la base de datos principal
        self.push_experience(eva_record)
        log(f"[EVA-PERSISTENCE] Ingested EVA experience: {experience_id}")
        self.event_bus.publish("eva_experience_ingested", experience_id, eva_record)
        return experience_id

    def eva_recall_experience(self, cue: str, phase: str = None) -> dict:
        """
        Recupera una experiencia viviente EVA por ID y fase desde memoria local o persistencia base.
        """
        phase = phase or self.eva_phase
        eva_record = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if eva_record:
            return eva_record
        # Fallback a experiencias persistidas
        for exp in self.state.get("experiences", []):
            if exp.get("experience_id") == cue and exp.get("phase") == phase:
                return exp
        log(f"[EVA-PERSISTENCE] EVA experience {cue} not found in phase '{phase}'")
        return {"error": "No EVA experience found"}

    def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa para una experiencia EVA.
        """
        eva_record = {
            "experience_id": experience_id,
            "experience_data": experience_data,
            "qualia_state": qualia_state,
            "phase": phase,
            "timestamp": experience_data.get("timestamp", time.time()),
        }
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        log(
            f"[EVA-PERSISTENCE] Added phase '{phase}' for EVA experience {experience_id}"
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                log(f"[EVA-PERSISTENCE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
