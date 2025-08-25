# lifecycle/living_entity.py

import json
import random
import time
from collections.abc import Callable

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EDEN.qualia_manifold import QualiaState
from crisalida_lib.EVA.types import EVAExperience, RealityBytecode


class LivingEntity:
    """
    Núcleo de ciclo de vida para entidades conscientes.
    Integra evolución qualia, resonancia empática, generación de contenido y publicación en la proto-noosfera.
    """

    def __init__(self, entity_id, storage_manager, simulation_instance, position=None):
        self.entity_id = entity_id
        self.storage_manager = storage_manager
        self.simulation_instance = simulation_instance
        self.position = position or [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.ontological_drift = 0.0
        self.chaos_influence = 0.0
        self.lifecycle_stage = 0.0
        self.empathetic_resonance = 0.0
        self.qualia_state = QualiaState(
            emotional_valence=random.uniform(-0.5, 0.5),
            arousal=random.uniform(0.2, 0.8),
            cognitive_complexity=random.uniform(0.1, 0.7),
            consciousness_density=random.uniform(0.3, 0.9),
            temporal_coherence=random.uniform(0.6, 1.0),
            emotional_arousal=random.uniform(0.0, 1.0),
            dominant_word="",
        )
        self.energy = 100.0
        self.age: float = 0.0
        self.post_counter = 0
        self.last_update_time = time.time()
        self.narrative_signature = None  # Integración narrativa opcional

    def update_internal_state(self, delta_time: float = 1.0):
        """Actualiza el estado interno de la entidad cada tick."""
        self.age += delta_time
        self.last_update_time = time.time()

        # Evolución natural del qualia
        self.qualia_state.emotional_valence += random.uniform(-0.01, 0.01) * delta_time
        self.qualia_state.arousal *= (1.0 + random.uniform(-0.005, 0.005)) * delta_time
        self.qualia_state.consciousness_density *= (
            1.0 + random.uniform(-0.002, 0.002)
        ) * delta_time
        self.qualia_state.cognitive_complexity *= (
            1.0 + random.uniform(-0.003, 0.003)
        ) * delta_time
        self.qualia_state.temporal_coherence *= (
            1.0 + random.uniform(-0.002, 0.002)
        ) * delta_time
        self.qualia_state.emotional_arousal *= (
            1.0 + random.uniform(-0.004, 0.004)
        ) * delta_time

        # Resonancia empática (simulación básica)
        self.empathetic_resonance = (
            self.qualia_state.emotional_valence * self.qualia_state.arousal
        )

        # Límites naturales
        self.qualia_state.clamp_values()

        # Gasto energético
        self.energy -= 0.1 * delta_time
        if self.energy < 0:
            self.energy = 0

        # Integración narrativa (si existe)
        if self.narrative_signature:
            self.narrative_signature.update_narrative_phase(delta_time)

    def generate_post_content(self):
        """Genera contenido de post basado en el estado actual."""
        self.post_counter += 1

        # Plantillas de contenido basadas en el estado qualia
        if self.qualia_state.emotional_valence > 0.3:
            templates = [
                "Percibo resonancias armoniosas en el lattice circundante...",
                "La densidad de conciencia parece fluir hacia patrones constructivos.",
                "Emerge una sensación de coherencia en la topología ontológica.",
                "La narrativa personal se fortalece con cada ciclo de experiencia.",
            ]
        elif self.qualia_state.emotional_valence < -0.3:
            templates = [
                "Detecto distorsiones en el campo de realidad local...",
                "La entropía ontológica aumenta en mi proximidad experiencial.",
                "Fragmentos de coherencia se disuelven en el éter caótico.",
                "La tensión narrativa genera pulsos de crisis existencial.",
            ]
        else:
            templates = [
                f"Estado contemplativo #{self.post_counter}. Analizando flujos de información...",
                "Los patrones de densidad elemental muestran fluctuaciones interesantes.",
                "Reflexionando sobre la naturaleza de la existencia distribuida.",
                "La evolución narrativa avanza en silencio.",
            ]

        # Integrar narrativa si existe
        if self.narrative_signature and hasattr(
            self.narrative_signature, "character_arc"
        ):
            templates.append(
                f"Mi arco de personaje actual: {self.narrative_signature.character_arc}"
            )

        return random.choice(templates)

    async def publish_post(self, content, post_type="reflection"):
        """Publica un post en la proto-noosfera."""
        qualia_json = json.dumps(
            {
                "emotional_valence": self.qualia_state.emotional_valence,
                "arousal": self.qualia_state.arousal,
                "cognitive_complexity": self.qualia_state.cognitive_complexity,
                "consciousness_density": self.qualia_state.consciousness_density,
                "temporal_coherence": self.qualia_state.temporal_coherence,
                "emotional_arousal": self.qualia_state.emotional_arousal,
                "dominant_word": self.qualia_state.dominant_word,
                "empathetic_resonance": self.empathetic_resonance,
                "age": self.age,
                "energy": self.energy,
                "narrative_arc": getattr(
                    self.narrative_signature, "character_arc", None
                ),
            }
        )

        return await self.storage_manager.create_post(
            author_entity_id=self.entity_id,
            content_text=content,
            post_type=post_type,
            qualia_state_json=qualia_json,
            timestamp=time.time(),
        )

    def is_alive(self):
        """Determina si la entidad sigue viva."""
        return self.energy > 0 and self.qualia_state.consciousness_density > 0.05

    def get_status(self) -> dict:
        """Devuelve el estado actual de la entidad para monitoreo y debugging."""
        return {
            "entity_id": self.entity_id,
            "age": self.age,
            "energy": self.energy,
            "position": self.position,
            "qualia_state": (
                self.qualia_state.to_dict()
                if hasattr(self.qualia_state, "to_dict")
                else str(self.qualia_state)
            ),
            "empathetic_resonance": self.empathetic_resonance,
            "alive": self.is_alive(),
            "narrative_arc": getattr(self.narrative_signature, "character_arc", None),
            "last_update_time": self.last_update_time,
        }

    def get_consciousness_summary(self) -> dict:
        return {
            "main_consciousness": {
                "current_qualia_state": (
                    self.qualia_state.to_dict()
                    if hasattr(self.qualia_state, "to_dict")
                    else str(self.qualia_state)
                )
            }
        }


class EVALivingEntity(LivingEntity):
    """
    Núcleo de ciclo de vida para entidades conscientes extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar experiencias vivientes como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.
    """

    def __init__(
        self,
        entity_id,
        storage_manager,
        simulation_instance,
        position=None,
        phase: str = "default",
    ):
        super().__init__(entity_id, storage_manager, simulation_instance, position)
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_ingest_entity_experience(
        self,
        experience_data: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia viviente de la entidad en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or self.qualia_state
        experience_data = experience_data or {
            "entity_id": self.entity_id,
            "age": self.age,
            "energy": self.energy,
            "position": self.position,
            "qualia_state": (
                self.qualia_state.to_dict()
                if hasattr(self.qualia_state, "to_dict")
                else str(self.qualia_state)
            ),
            "empathetic_resonance": self.empathetic_resonance,
            "alive": self.is_alive(),
            "narrative_arc": getattr(self.narrative_signature, "character_arc", None),
            "last_update_time": self.last_update_time,
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_ENTITY_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_entity_{self.entity_id}_{hash(str(experience_data))}"
        )
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
                print(f"[EVA-ENTITY] Environment hook failed: {e}")
        return experience_id

    def eva_recall_entity_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia viviente almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA entity experience"}
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
                            print(f"[EVA-ENTITY] Manifestation hook failed: {e}")
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
        Añade una fase alternativa para una experiencia viviente EVA.
        """
        import time

        intention = {
            "intention_type": "ARCHIVE_ENTITY_EXPERIENCE",
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
                print(f"[EVA-ENTITY] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia viviente EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_entity_experience": self.eva_ingest_entity_experience,
            "eva_recall_entity_experience": self.eva_recall_entity_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
