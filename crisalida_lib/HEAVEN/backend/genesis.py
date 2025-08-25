"""
Genesis - Entry point for Janus Core backend application.
Initializes PrometheusEvolutionaryOrchestrator, sets up logging, handles graceful shutdown,
and provides diagnostics for startup and shutdown events.
"""

import asyncio
import logging
import signal
import sys
import time

from crisalida_lib.EARTH.orchestrator import PrometheusEvolutionaryOrchestrator
from crisalida_lib.EARTH.sub_simulation_creator import EVASubSimulationCreator
from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EDEN.reality_engine_metrics_calculator import (
    EVARealityEngineMetricsCalculator,
)

sys.path.append("/media/seikarii/Nvme/janus_v13_genesis")

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("JanusGenesis")


def setup_signal_handlers(loop, orchestrator):
    """Configura manejadores de señales para apagado limpio."""

    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        loop.create_task(orchestrator.shutdown())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def setup_eva_signal_handlers(loop, orchestrator, eva_api):
    """Configura manejadores de señales para apagado limpio y persistencia EVA."""

    def shutdown_handler(signum, frame):
        logger.info(
            f"[EVA] Received signal {signum}. Initiating graceful shutdown and EVA memory persist..."
        )
        loop.create_task(orchestrator.shutdown())
        try:
            eva_api.get("save_eva_state", lambda: None)()
            logger.info("[EVA] EVA memory state saved successfully.")
        except Exception as e:
            logger.warning(f"[EVA] Failed to save EVA memory state: {e}")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


async def main():
    logger.info("🚀 Starting PrometheusEvolutionaryOrchestrator...")
    orchestrator = PrometheusEvolutionaryOrchestrator()
    await orchestrator.initialize()
    loop = asyncio.get_running_loop()

    # EVA: Inicialización de memoria viviente y API
    eva_runtime = LivingSymbolRuntime()
    eva_metrics_calculator = EVARealityEngineMetricsCalculator(unified_field=None)
    eva_sub_sim_creator = EVASubSimulationCreator(
        metrics_calculator=eva_metrics_calculator
    )
    eva_api = {
        "eva_runtime": eva_runtime,
        "eva_metrics_calculator": eva_metrics_calculator,
        "eva_sub_sim_creator": eva_sub_sim_creator,
        "save_eva_state": getattr(
            eva_metrics_calculator, "save_eva_state", lambda: None
        ),
    }

    setup_eva_signal_handlers(loop, orchestrator, eva_api)
    try:
        await orchestrator.start_autonomous_evolution()
    except Exception as exc:
        logger.error(f"Fatal error in orchestrator: {exc}", exc_info=True)
    finally:
        logger.info("🛑 Genesis shutdown sequence complete.")
        # EVA: Guardar estado de memoria viviente al finalizar
        try:
            eva_api.get("save_eva_state", lambda: None)()
            logger.info("[EVA] EVA memory state saved successfully on shutdown.")
        except Exception as e:
            logger.warning(f"[EVA] Failed to save EVA memory state on shutdown: {e}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down Genesis...")
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Genesis exited after {elapsed:.2f} seconds.")
