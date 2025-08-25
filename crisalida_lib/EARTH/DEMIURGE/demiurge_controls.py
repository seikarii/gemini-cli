import logging
import re
from collections.abc import Callable
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode

logger = logging.getLogger(__name__)


class DemiurgeControlSystem:
    """
    Sistema de control del Demiurgo, interpretando intenciones desde diversas entradas.
    Implementa las Capas 1 (Control Intuitivo) y 2 (Verbo Divino) de la IMD.
    """

    def __init__(self, unified_field: Any = None):
        self.unified_field = unified_field
        logger.info("DemiurgeControlSystem inicializado.")

    def process_gaze_and_input(
        self, gaze_target: dict[str, Any], modifiers: dict[str, bool]
    ) -> dict[str, Any] | None:
        """
        Procesa la mirada y las entradas de teclado/joystick para generar una intención ontológica.
        """
        target_type = gaze_target.get("type")
        target_id = gaze_target.get("id")
        target_position = gaze_target.get("position")
        if not target_position:
            logger.warning("Gaze target sin posición. No se puede generar intención.")
            return None

        intent_type = "interact_ontologically"
        intent_strength = 0.5
        details: dict[str, Any] = {"target": gaze_target}

        if modifiers.get("Q"):  # Modificador de Creación/Expansión
            intent_type = "manifest_creation"
            intent_strength = 0.7
            details["creation_type"] = "energy_burst"
            logger.info(f"Demiurgo intenta manifestar creación en {target_position}")
        elif modifiers.get("W"):  # Modificador de Conciencia/Armonía
            intent_type = "harmonize_qualia"
            intent_strength = 0.6
            details["harmony_target_id"] = target_id
            logger.info(f"Demiurgo intenta armonizar qualia de {target_id}")
        elif modifiers.get("E"):  # Modificador de Destrucción/Disrupción
            intent_type = "disrupt_coherence"
            intent_strength = 0.75
            details["disruption_target_id"] = target_id
            logger.info(f"Demiurgo intenta disrumpir coherencia de {target_id}")
        elif modifiers.get("R"):  # Modificador de Transmisión/Conocimiento
            intent_type = "transmit_knowledge"
            intent_strength = 0.65
            details["knowledge_target_id"] = target_id
            details["knowledge_concept"] = "ontological_principle_X"
            logger.info(f"Demiurgo intenta transmitir conocimiento a {target_id}")
        else:
            logger.info(
                f"Demiurgo observa {target_type} en {target_position} sin modificadores específicos."
            )

        return {
            "type": intent_type,
            "strength": intent_strength,
            "position": target_position,
            "details": details,
            "source": "demiurge_gaze_input",
        }

    def process_voice_command(self, command_string: str) -> dict[str, Any] | None:
        """
        Procesa un comando de voz de alto nivel para generar una intención ontológica.
        Args:
            command_string (str): La cadena de texto del comando de voz.
        Returns:
            Optional[Dict[str, Any]]: Una intención ontológica procesable por el UnifiedField.
        """
        command_string = command_string.lower().strip()
        logger.info(f"Demiurgo emite comando de voz: '{command_string}'")

        if "manifiesta" in command_string and "firelattice" in command_string:
            density_match = re.search(r"densidad\s+([0-9.]+)", command_string)
            density = float(density_match.group(1)) if density_match else 0.5
            radius_match = re.search(
                r"radio\s+de\s+influencia\s+([0-9.]+)", command_string
            )
            radius = float(radius_match.group(1)) if radius_match else 10.0
            demiurge_position = (
                self.unified_field.player_state["position"]
                if self.unified_field and hasattr(self.unified_field, "player_state")
                else [0, 0, 0]
            )
            return {
                "type": "manifest_lattice",
                "lattice_type": "fire",
                "density": density,
                "position": demiurge_position,
                "radius": radius,
                "source": "demiurge_voice_command",
            }

        elif "transmite" in command_string and "concepto tactico" in command_string:
            concept_match = re.search(r"'(.+?)'", command_string)
            concept = concept_match.group(1) if concept_match else "unknown_concept"
            target_entities_ids = (
                [
                    e.entity_id
                    for e in getattr(self.unified_field, "entities", [])
                    if getattr(e, "is_ally", False)
                ]
                if self.unified_field
                else []
            )
            return {
                "type": "transmit_tactical_concept",
                "concept": concept,
                "target_entities_ids": target_entities_ids,
                "source": "demiurge_voice_command",
            }

        logger.warning(f"Comando de voz no reconocido: '{command_string}'")
        return None

    def _execute_ontological_intention(self, intention: dict[str, Any]):
        """
        Envía la intención procesada al UnifiedField para su ejecución real.
        """
        if self.unified_field and hasattr(
            self.unified_field, "process_demiurge_intention"
        ):
            logger.info(
                f"Enviando intención al UnifiedField: {intention.get('type')} desde {intention.get('source')}"
            )
            self.unified_field.process_demiurge_intention(intention)
        elif not self.unified_field:
            logger.error("UnifiedField no está conectado al DemiurgeControlSystem.")
        else:
            logger.warning(
                "El UnifiedField actual no tiene el método 'process_demiurge_intention'."
            )


class EVADemiurgeControlSystem(DemiurgeControlSystem):
    """
    Sistema de control del Demiurgo extendido para integración con EVA.
    Interpreta intenciones, las compila como RealityBytecode, soporta ingestión/recall, faseo, hooks y benchmarking.
    """

    def __init__(self, unified_field: Any = None, phase: str = "default"):
        super().__init__(unified_field)
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_compile_intention(
        self,
        intention: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una intención ontológica en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.9,
            consciousness_density=0.8,
            narrative_importance=0.8,
            energy_level=1.0,
        )
        experience_data = {
            "intention": intention,
            "timestamp": time.time(),
            "phase": phase,
        }
        eva_intention = {
            "intention_type": "ARCHIVE_DEMIURGE_CONTROL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = []
        runtime = getattr(self, "eva_runtime", None)
        if runtime is not None:
            _dc = getattr(runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(eva_intention)
                except Exception:
                    bytecode = []
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(eva_intention)
            except Exception:
                bytecode = []
        experience_id = f"eva_demiurge_control_{hash(str(experience_data))}"
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

    def eva_recall_intention_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una intención almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {
                "error": "No bytecode found for EVA DemiurgeControlSystem experience"
            }
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
                            logger.warning(
                                f"[EVA-DEMIURGE-CONTROL] Environment hook failed: {e}"
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
        self, experience_id: str, phase: str, intention: dict, qualia_state: QualiaState
    ):
        """
        Añade una fase alternativa para una experiencia de intención EVA.
        """
        import time

        experience_data = {
            "intention": intention,
            "timestamp": time.time(),
            "phase": phase,
        }
        eva_intention = {
            "intention_type": "ARCHIVE_DEMIURGE_CONTROL_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = []
        runtime = getattr(self, "eva_runtime", None)
        if runtime is not None:
            _dc = getattr(runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(eva_intention)
                except Exception:
                    bytecode = []
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(eva_intention)
            except Exception:
                bytecode = []
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
                logger.warning(f"[EVA-DEMIURGE-CONTROL] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de intención EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_compile_intention": self.eva_compile_intention,
            "eva_recall_intention_experience": self.eva_recall_intention_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# Example Usage (for testing this module in isolation)
if __name__ == "__main__":

    class MockUnifiedField:
        def __init__(self):
            self.player_state = {"position": [0, 0, 0]}
            self.entities = []

        def process_demiurge_intention(self, intention):
            print(f"[MockUnifiedField] Recibida intención: {intention}")

    mock_unified_field = MockUnifiedField()
    demiurge_controls = DemiurgeControlSystem(unified_field=mock_unified_field)

    print("\n--- Test Capa 1: Gaze and Modifiers ---")
    gaze_target_entity = {"type": "entity", "id": "entity_A", "position": [5, 5, 5]}
    gaze_target_coords = {"type": "coordinates", "position": [10, 10, 10]}
    intent1 = demiurge_controls.process_gaze_and_input(gaze_target_entity, {})
    if intent1:
        mock_unified_field.process_demiurge_intention(intent1)
    intent2 = demiurge_controls.process_gaze_and_input(gaze_target_coords, {"Q": True})
    if intent2:
        mock_unified_field.process_demiurge_intention(intent2)
    intent3 = demiurge_controls.process_gaze_and_input(gaze_target_entity, {"W": True})
    if intent3:
        mock_unified_field.process_demiurge_intention(intent3)

    print("\n--- Test Capa 2: Voice Commands ---")
    mock_unified_field.player_state["position"] = [1, 2, 3]
    voice_cmd1 = "Manifiesta un FireLattice de densidad 0.9 en mis coordenadas con radio de influencia 20."
    intent4 = demiurge_controls.process_voice_command(voice_cmd1)
    if intent4:
        mock_unified_field.process_demiurge_intention(intent4)
    voice_cmd2 = (
        "HOI, transmite el concepto tactico 'Flanqueo Defensivo Alfa' a mi cabala."
    )
    mock_unified_field.entities.append(
        type("Entity", (), {"entity_id": "ally_1", "is_ally": True})()
    )
    mock_unified_field.entities.append(
        type("Entity", (), {"entity_id": "ally_2", "is_ally": True})()
    )
    intent5 = demiurge_controls.process_voice_command(voice_cmd2)
    if intent5:
        mock_unified_field.process_demiurge_intention(intent5)
    voice_cmd3 = "Haz que llueva oro."
    intent6 = demiurge_controls.process_voice_command(voice_cmd3)
    if intent6:
        mock_unified_field.process_demiurge_intention(
            intent6
        )  # Should be None and log a warning
