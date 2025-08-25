import logging
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import QualiaState, RealityBytecode
from crisalida_lib.unified_field import UnifiedField

logger = logging.getLogger(__name__)


def inject_external_data_to_simulation(
    unified_field: UnifiedField,
    data_type: str,
    payload: dict[str, Any],
    source: str | None = None,
    timestamp: float | None = None,
    priority: int = 1,
) -> bool:
    """
    Inyecta datos externos en la simulación a través del UnifiedField, con trazabilidad y control de prioridad.

    Args:
        unified_field: Instancia de UnifiedField.
        data_type (str): Tipo de dato (ej. "environmental_scan", "user_intent", "real_world_event").
        payload (Dict[str, Any]): Datos a inyectar.
        source (str, opcional): Fuente del dato (ej. "sensor", "api", "user").
        timestamp (float, opcional): Marca de tiempo del evento.
        priority (int): Prioridad de procesamiento (1=normal, 2=alta).

    Returns:
        bool: True si la inyección fue exitosa, False si hubo error.
    """
    try:
        event = {
            "type": data_type,
            "payload": payload,
            "source": source or "external",
            "timestamp": timestamp or unified_field.get_current_time(),
            "priority": priority,
        }
        logger.info(f"Injecting external data: {event}")
        unified_field.inject_external_data(event)
        return True
    except Exception as e:
        logger.error(f"Failed to inject external data: {e}")
        return False


def inject_external_data_to_simulation_eva(
    unified_field: UnifiedField,
    data_type: str,
    payload: dict[str, Any],
    source: str | None = None,
    timestamp: float | None = None,
    priority: int = 1,
    qualia_state: QualiaState | None = None,
    phase: str = "default",
    eva_runtime: LivingSymbolRuntime | None = None,
    eva_memory_store: dict | None = None,
    eva_phases: dict | None = None,
    environment_hooks: list | None = None,
) -> bool:
    """
    Inyecta datos externos en la simulación y los registra como experiencia viviente EVA (RealityBytecode).
    Permite faseo, hooks de entorno y benchmarking de ingestión.
    """
    try:
        event = {
            "type": data_type,
            "payload": payload,
            "source": source or "external",
            "timestamp": timestamp or unified_field.get_current_time(),
            "priority": priority,
        }
        logger.info(f"[EVA] Injecting external data: {event}")
        unified_field.inject_external_data(event)

        # EVA: Registrar experiencia viviente
        eva_runtime = eva_runtime or getattr(unified_field, "eva_runtime", None)
        eva_memory_store = eva_memory_store or getattr(
            unified_field, "eva_memory_store", {}
        )
        eva_phases = eva_phases or getattr(unified_field, "eva_phases", {})
        environment_hooks = environment_hooks or getattr(
            unified_field, "_environment_hooks", []
        )

        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.6,
            cognitive_complexity=0.8,
            consciousness_density=0.7,
            narrative_importance=0.7,
            energy_level=1.0,
        )
        experience_data = {
            "external_event": event,
            "timestamp": event["timestamp"],
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_EXTERNAL_REALITY_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        if eva_runtime:
            bytecode = eva_runtime.divine_compiler.compile_intention(intention)
            experience_id = f"eva_external_{data_type}_{hash(str(experience_data))}"
            reality_bytecode = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=experience_data["timestamp"],
            )
            eva_memory_store[experience_id] = reality_bytecode
            if phase not in eva_phases:
                eva_phases[phase] = {}
            eva_phases[phase][experience_id] = reality_bytecode
            for hook in environment_hooks:
                try:
                    hook(reality_bytecode)
                except Exception as e:
                    logger.warning(
                        f"[EVA-EXTERNAL-REALITY] Environment hook failed: {e}"
                    )
        return True
    except Exception as e:
        logger.error(f"[EVA] Failed to inject external data: {e}")
        return False
