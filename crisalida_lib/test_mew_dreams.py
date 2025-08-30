
import asyncio
import logging
import json
import aiofiles
import sys
import os
import pytest

# Add the parent directory to sys.path to allow importing crisalida_lib
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir)

from crisalida_lib.global_orchestrator import ensure_ollama_service, ensure_llm_model_is_available, LLM_MODEL
from crisalida_lib.HEAVEN.agents.agent_mew import AgentMew
from crisalida_lib.ADAM.config import EVAConfig
from crisalida_lib.HEAVEN.llm.ollama_client import OllamaClient

MIRROR_LLM_MODEL = "llama3.2:1b"



async def load_memory(file_path: str):
    try:
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error loading memory from {file_path}: {e}")
        return None

async def save_memory(data: dict, file_path: str):
    try:
        async with aiofiles.open(file_path, mode='w') as f:
            await f.write(json.dumps(data, indent=4))
        logger.info(f"Memory saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving memory to {file_path}: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_mew_dreams():
    """
    Initializes AgentMew, lets it run for a short period,
    and then inspects its memory for dream-like nodes.
    """
    logger.info("🚀 Starting AgentMew Dream Test...")

    MEMORY_FILE = "mew_memory.json"
    initial_memory = None
    try:
        initial_memory = await load_memory(MEMORY_FILE)
        if initial_memory:
            logger.info(f"💾 Loaded initial memory from {MEMORY_FILE}")
        else:
            logger.info(f"🆕 No existing memory found at {MEMORY_FILE}. Starting fresh.")
    except Exception as e:
        logger.error(f"❌ Error loading memory: {e}", exc_info=True)
        logger.info("Continuing with fresh memory.")




    # 1. Setup environment from the orchestrator
    if not await ensure_ollama_service():
        return
    ollama_client_port1 = OllamaClient(base_url="http://localhost:11434/api")
    ollama_client_port2 = ollama_client_port1

    if not await ensure_llm_model_is_available(ollama_client_port1, LLM_MODEL):
        return
    if not await ensure_llm_model_is_available(ollama_client_port1, MIRROR_LLM_MODEL):
        return

    agent_mew = None
    mirror_agent_mew = None
    try:
        # 2. Initialize AgentMew just like the orchestrator does
        logger.info("Initializing AgentMew for the test...")
        config = EVAConfig()
        agent_mew = AgentMew(
            config=config,
            ollama_client=ollama_client_port1,
            llm_model=LLM_MODEL,
            start_monitor=True
        )
        if initial_memory:
            agent_mew.adam.entity_memory.mind.load_from_json(json.dumps(initial_memory))
            logger.info("🧠 AgentMew's mind populated with loaded memory.")
        logger.info("✅ AgentMew initialized. Letting it 'live' for a few seconds to populate its mind...")

        # Initialize Mirror AgentMew
        logger.info("Initializing Mirror AgentMew...")
        mirror_agent_mew = AgentMew(
            config=config,
            ollama_client=ollama_client_port1,
            llm_model=MIRROR_LLM_MODEL,
            start_monitor=False # Mirror agent doesn't need to monitor
        )
        logger.info("✅ Mirror AgentMew initialized.")

        # 3. Let the agent "live" and "dream" by running a few update cycles.
        # The dream() method, which consolidates memory, is called within adam.update().
        async def run_updates():
            for i in range(5):
                logger.info(f"Agent update cycle {i+1}/5...")

                # Main Agent's initial thought/prompt for its LLM
                main_agent_prompt = f"Cycle {i+1}: I am pondering the nature of reality. What are your thoughts on this profound topic?"
                logger.info(f"Mew's LLM prompt: {main_agent_prompt}")

                # Main Agent LLM Call
                main_agent_llm_response = await ollama_client_port1.chat_async(
                    model_name=LLM_MODEL,
                    messages=[{"role": "user", "content": main_agent_prompt}]
                )
                main_agent_output = main_agent_llm_response
                logger.info(f"Mew's LLM response: {main_agent_output}")

                # Main Agent processes its LLM output (update adam with this as a perception)
                main_perception_from_llm = {"type": "llm_output", "content": main_agent_output, "timestamp": asyncio.get_event_loop().time()}
                await agent_mew.adam.update(delta_time=1.0, perception=main_perception_from_llm)

                # Mirror Agent's prompt based on Main Agent's LLM output
                mirror_agent_prompt = f"Mew just said: '{main_agent_output}'. Reflect on this statement. What's your simplified, 'tonto' take?"
                logger.info(f"Mirror's LLM prompt: {mirror_agent_prompt}")

                # Mirror Agent LLM Call
                mirror_agent_llm_response = await ollama_client_port1.chat_async(
                    model_name=MIRROR_LLM_MODEL,
                    messages=[{"role": "user", "content": mirror_agent_prompt}]
                )
                mirror_agent_output = mirror_agent_llm_response
                logger.info(f"Mirror's LLM response: {mirror_agent_output}")

                # Mirror Agent processes its LLM output (update adam with this as a perception)
                mirror_perception_from_llm = {"type": "mirror_llm_output", "content": mirror_agent_output, "timestamp": asyncio.get_event_loop().time()}
                await mirror_agent_mew.adam.update(delta_time=1.0, perception=mirror_perception_from_llm)

                # Main Agent processes Mirror Agent's LLM output
                main_agent_receives_mirror_output = {"type": "mirror_reflection", "content": mirror_agent_output, "timestamp": asyncio.get_event_loop().time()}
                await agent_mew.adam.update(delta_time=1.0, perception=main_agent_receives_mirror_output)

                await asyncio.sleep(2) # Wait a bit between cycles

        update_task = asyncio.create_task(run_updates())
        await update_task

        logger.info("🧠 Peeking into AgentMew's mind to see what it dreams of...")

        # 4. Ask "what does it dream of?" by inspecting its memory for dream nodes
        # These nodes are created by the REM/NREM passes in the MentalLaby.
        dream_nodes = agent_mew.adam.entity_memory.mind.search(
            query="abstract concepts, creative connections, synthesized ideas",
            top_k=5,
            kind="dream" # Filter specifically for nodes created during dream cycles
        )

        if not dream_nodes:
            logger.info("Mew is not dreaming of anything specific right now. Its mind is quiet.")
            # As a fallback, let's check its most important (salient) thoughts.
            salient_nodes = agent_mew.adam.entity_memory.mind.get_salient_nodes(top_k=3)
            if salient_nodes:
                logger.info("However, these are its most salient thoughts:")
                for node in salient_nodes:
                    # The actual content is in the 'data' attribute
                    content = node.data.get('raw', 'an abstract concept')
                    logger.info(f"  - A thought with salience {node.salience:.2f}: {content}")
        else:
            logger.info("Mew seems to be dreaming about:")
            for sim, nid, node in dream_nodes:
                # The content of a dream node often references the path of thoughts that created it
                dream_content = node.data.get('dream_path', node.data.get('raw', 'a fleeting thought'))
                logger.info(f"  - A dream (similarity {sim:.2f}) composed of the ideas: {dream_content}")

    except Exception as e:
        logger.error(f"❌ An error occurred during the test: {e}", exc_info=True)
    finally:
        if agent_mew:
            try:
                await save_memory(json.loads(agent_mew.adam.entity_memory.mind.to_json()), MEMORY_FILE)
                logger.info(f"💾 Saved current memory to {MEMORY_FILE}")
            except Exception as e:
                logger.error(f"❌ Error saving memory: {e}", exc_info=True)
            await agent_mew.shutdown()
        if mirror_agent_mew:
            await mirror_agent_mew.shutdown()
        logger.info("Test finished.")


async def main():
    """Runs the test with a 30-second timeout."""
    logger.info("--- Starting Test: Vitality check and Dream Inquiry ---")
    logger.info("--- The test will be cancelled after 30 seconds. ---")
    try:
        await asyncio.wait_for(test_mew_dreams(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("⏳ Test cancelled after 30 seconds timeout. This is expected.")
    except Exception as e:
        logger.error(f"Test failed with an unexpected error: {e}", exc_info=True)
    finally:
        logger.info("--- Test run complete. ---")


if __name__ == "__main__":
    asyncio.run(main())
