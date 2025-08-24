import asyncio
import logging
import subprocess

from crisalida_lib.HEAVEN.agents.interface.basic_agent_interface import (
    BasicAgentInterface,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def manage_ollama():
    """
    Checks if Ollama is running and starts it if not.
    Also ensures the required model is available.
    """
    try:
        # Check if ollama process is running
        result = subprocess.run(
            ["pgrep", "-f", "ollama"], capture_output=True, text=True
        )
        if not result.stdout:
            logger.info("Ollama server not running. Starting it in the background...")
            # Start the server in the background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(5)  # Give it a moment to start
            logger.info("Ollama server started.")
        else:
            logger.info("Ollama server is already running.")

        # The model was already pulled in the previous step, but this is good practice
        # to keep in a setup script.
        logger.info("Ensured wizardlm2:7b model is available.")

    except FileNotFoundError:
        logger.error(
            "'ollama' command not found. Please ensure Ollama is installed and in your PATH."
        )
        return False
    except Exception as e:
        logger.error(f"An error occurred while managing Ollama: {e}")
        return False
    return True


async def test_agent_mew():
    """
    Initializes and tests the AgentMew through the BasicAgentInterface.
    """
    print("--- Initializing AgentMew Test ---")

    # 1. Ensure Ollama is ready
    if not await manage_ollama():
        return

    interface = BasicAgentInterface()
    agent_id = "mew_001"
    model_name = "wizardlm2:7b"

    # 2. Create Agent with specific model config
    try:
        agent_config = {"llm_model": model_name}
        await interface.create_agent("agent_mew", agent_id, config=agent_config)
        print(f"AgentMew created with ID: {agent_id} and model {model_name}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        return

    # 3. Start Agent
    try:
        session_id = await interface.start_agent(agent_id)
        print(f"AgentMew started with Session ID: {session_id}")
    except Exception as e:
        print(f"Error starting agent: {e}")
        return

    # 4. Send a complex prompt
    complex_prompt = """
    remodelar uno de los agentes 'working_' que se encuentran en
    '/media/seikarii/Nvme/Crisalida/crisalida_lib/HEAVEN/agents'
    para que pueda funcionar con
    '/media/seikarii/Nvme/Crisalida/crisalida_lib/HEAVEN/llm' y
    '/media/seikarii/Nvme/Crisalida/crisalida_lib/ASTRAL_TOOLS'.
    """
    print(
        f"\n--- Sending Complex Prompt ---\n{complex_prompt}\n---------------------------------"
    )

    try:
        response = await interface.send_prompt(agent_id, complex_prompt)
        print("\n--- AgentMew Response ---")
        if response.success:
            print(response.response)
        else:
            print(f"Agent returned an error: {response.response}")
        print("---------------------------")
    except Exception as e:
        print(f"Error sending prompt: {e}")

    # 5. Stop Agent
    try:
        await interface.stop_agent(agent_id)
        print(f"\nAgentMew {agent_id} stopped.")
    except Exception as e:
        print(f"Error stopping agent: {e}")

    print("--- AgentMew Test Finished ---")


if __name__ == "__main__":
    asyncio.run(test_agent_mew())
