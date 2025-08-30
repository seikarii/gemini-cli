
import asyncio
import logging
import subprocess

from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.EVA.memory_orchestrator import EVAMemoryOrchestrator
from crisalida_lib.HEAVEN.agents.agent_mew import AgentMew
from crisalida_lib.HEAVEN.llm.ollama_client import OllamaClient

# Placeholder for other imports like EDEN, EARTH, etc.
# from crisalida_lib.EDEN.reality_engine import RealityEngine
# from crisalida_lib.EARTH.auto_genesis_engine import AutoGenesisEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

LLM_MODEL = "llama3.1:8b"

async def ensure_ollama_service():
    """Checks if the Ollama service is running and tries to start it if not."""
    client = OllamaClient()
    if await client.ping():
        logger.info("✅ Ollama service is already running.")
        return True

    logger.warning("Ollama service not detected. Attempting to start it...")
    try:
        # This is a simplified approach. A robust solution would use
        # run_shell_command and manage the process.
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(10)  # Wait for the service to initialize
        if await client.ping():
            logger.info("✅ Successfully started Ollama service.")
            return True
        else:
            logger.error("Failed to start Ollama service.")
            return False
    except FileNotFoundError:
        logger.error("❌ 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while starting Ollama: {e}")
        return False

async def ensure_llm_model_is_available(client: OllamaClient, model_name: str):
    """Ensures the specified LLM model is available, pulling it if necessary."""
    is_available = await client.check_model_availability(model_name)
    if is_available:
        logger.info(f"✅ LLM model '{model_name}' is available.")
        # Obtener y loguear información del modelo
        model_info = await client.get_ollama_model_info(model_name)
        if model_info:
            logger.info(f"Model '{model_name}' details: {model_info.get('details', 'N/A')}")
            logger.info(f"Model '{model_name}' size: {model_info.get('size', 'N/A')} bytes")
            logger.info(f"Model '{model_name}' family: {model_info.get('details', {}).get('family', 'N/A')}")
            logger.info(f"Model '{model_name}' parameter_size: {model_info.get('details', {}).get('parameter_size', 'N/A')}")
            logger.info(f"Model '{model_name}' quantization_level: {model_info.get('details', {}).get('quantization_level', 'N/A')}")
        return True

    logger.warning(f"Model '{model_name}' not found. Attempting to pull... (This may take a while)")
    success = await client.pull_model_if_needed(model_name)
    if success:
        logger.info(f"✅ Successfully pulled model '{model_name}'.")
        # Después de un pull exitoso, también obtener y loguear la información del modelo
        model_info = await client.get_ollama_model_info(model_name)
        if model_info:
            logger.info(f"Model '{model_name}' details: {model_info.get('details', 'N/A')}")
            logger.info(f"Model '{model_name}' size: {model_info.get('size', 'N/A')} bytes")
            logger.info(f"Model '{model_name}' family: {model_info.get('details', {}).get('family', 'N/A')}")
            logger.info(f"Model '{model_name}' parameter_size: {model_info.get('details', {}).get('parameter_size', 'N/A')}")
            logger.info(f"Model '{model_name}' quantization_level: {model_info.get('details', {}).get('quantization_level', 'N/A')}")
    else:
        logger.error(f"❌ Failed to pull model '{model_name}'. Please check your internet connection and Ollama setup.")
    return success

async def main():
    """Main orchestration function to initialize and run the Crisalida ecosystem."""
    logger.info("🚀 Starting Crisalida Global Orchestrator...")

    # 1. Ensure Ollama service and model are ready
    if not await ensure_ollama_service():
        return

    ollama_client = OllamaClient()
    if not await ensure_llm_model_is_available(ollama_client, LLM_MODEL):
        return

    # 2. Initialize core components (EVA, EDEN, EARTH)
    logger.info("Initializing core services...")
    config = EVAConfig()
    eva_orchestrator = EVAMemoryOrchestrator(d=256) # Using default dimension
    # eden_engine = RealityEngine() # Placeholder
    # earth_engine = AutoGenesisEngine() # Placeholder
    logger.info("✅ Core services initialized.")

    # 3. Initialize and run AgentMew
    logger.info(f"Initializing AgentMew with model '{LLM_MODEL}'...")
    try:
        agent_mew = AgentMew(
            config=config,
            ollama_client=ollama_client,
            llm_model=LLM_MODEL,
            start_monitor=True
        )
        logger.info("✅ AgentMew initialized successfully.")

        # 4. Start the agent's autonomous loop
        logger.info("Starting AgentMew in autonomous mode...")
        # In a real scenario, you might assign a mission first
        # agent_mew.assign_mission("fix_codebase", {"target_directory": "/path/to/code"})
        await agent_mew.run_autonomous_mode()

    except Exception as e:
        logger.error(f"❌ An error occurred during AgentMew execution: {e}", exc_info=True)
    finally:
        if 'agent_mew' in locals() and agent_mew:
            await agent_mew.shutdown()
        logger.info("Orchestrator shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user.")

