from typing import Any

from crisalida_lib.unified_field import UnifiedField


def get_simulation_render_data(unified_field: UnifiedField) -> dict[str, Any]:
    """Obtiene los datos actuales de la simulación, formateados para renderizado externo.
    Incluye entidades, lattices, estados de conciencia, metadatos narrativos y experiencias EVA si están disponibles.

    Args:
        unified_field: Instancia de UnifiedField con los datos de la simulación.

    Returns:
        dict: Diccionario con entidades, lattices, estados de conciencia, metadatos narrativos y experiencias EVA.
    """
    render_data = unified_field.get_render_data()
    # Enriquecer con estados de conciencia y narrativa si existen
    if hasattr(unified_field, "get_consciousness_states"):
        render_data["consciousness_states"] = unified_field.get_consciousness_states()
    if hasattr(unified_field, "get_narrative_metadata"):
        render_data["narrative_metadata"] = unified_field.get_narrative_metadata()
    # Añadir timestamp y versión de simulación para trazabilidad
    render_data["timestamp"] = getattr(unified_field, "current_time", None)
    render_data["simulation_version"] = getattr(unified_field, "version", "unknown")

    # EVA: Añadir experiencias vivientes y manifestaciones para renderizado avanzado
    eva_runtime = getattr(unified_field, "eva_runtime", None)
    eva_memory_store = getattr(unified_field, "eva_memory_store", {})
    eva_phase = getattr(unified_field, "eva_phase", "default")
    eva_experiences = []
    if eva_memory_store:
        for exp_id, reality_bytecode in eva_memory_store.items():
            if hasattr(reality_bytecode, "to_dict"):
                exp_dict = reality_bytecode.to_dict()
            else:
                exp_dict = {
                    "bytecode_id": getattr(reality_bytecode, "bytecode_id", exp_id),
                    "phase": getattr(reality_bytecode, "phase", eva_phase),
                    "qualia_state": getattr(reality_bytecode, "qualia_state", None),
                    "timestamp": getattr(reality_bytecode, "timestamp", None),
                }
            # Opcional: manifestaciones simuladas para renderizado
            manifestations = []
            quantum_field = (
                getattr(eva_runtime, "quantum_field", None) if eva_runtime else None
            )
            if (
                eva_runtime
                and quantum_field
                and hasattr(reality_bytecode, "instructions")
            ):
                for instr in reality_bytecode.instructions:
                    symbol_manifest = eva_runtime.execute_instruction(
                        instr, quantum_field
                    )
                    if symbol_manifest:
                        manifestations.append(
                            symbol_manifest.to_dict()
                            if hasattr(symbol_manifest, "to_dict")
                            else symbol_manifest
                        )
            exp_dict["manifestations"] = manifestations
            eva_experiences.append(exp_dict)
    render_data["eva_experiences"] = eva_experiences
    render_data["eva_phase"] = eva_phase

    # EVA: Añadir benchmarking de memoria viviente si disponible
    if hasattr(unified_field, "eva_metrics_calculator"):
        metrics = unified_field.eva_metrics_calculator.calculate_eva_aggregated_metrics(
            phase=eva_phase
        )
        render_data["eva_metrics"] = metrics

    return render_data
