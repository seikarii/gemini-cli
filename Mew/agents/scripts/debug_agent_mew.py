import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx

from crisalida_lib.HEAVEN.agents.interface.basic_agent_interface import (
    BasicAgentInterface,
)

# Add project root to the Python path to resolve imports
project_root = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def _wait_for_ollama_ready(
    model_name: str = "wizardlm2:7b", timeout: int = 120, interval: int = 5
) -> bool:
    """Waits for the Ollama server to be ready and responsive, and for the model to be loaded."""
    ollama_url = "http://localhost:11434"
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                # First, check if the server is up
                response = await client.get(f"{ollama_url}/api/tags", timeout=interval)
                response.raise_for_status()

                # Then, try a small inference request to ensure the model is loaded
                # This will force the model to load into VRAM if it hasn't already
                inference_payload = {
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                }
                inference_response = await client.post(
                    f"{ollama_url}/api/generate",
                    json=inference_payload,
                    timeout=interval * 2,
                )
                inference_response.raise_for_status()

                logger.info(
                    f"Ollama server is responsive and model {model_name} is loaded."
                )
                return True
        except httpx.RequestError as e:
            logger.warning(
                f"Ollama server or model not yet responsive: {e}. Retrying in {interval}s..."
            )
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Ollama server returned error status {e.response.status_code} for {e.request.url}: {e.response.text}. Retrying in {interval}s..."
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error while waiting for Ollama: {e}. Retrying in {interval}s..."
            )
        await asyncio.sleep(interval)
    logger.error(
        f"Ollama server and model {model_name} did not become responsive within {timeout} seconds."
    )
    return False


async def manage_ollama():
    """
    Checks if Ollama is running and starts it if not.
    Also ensures the required model is available.
    """
    try:
        # Check if ollama process is running
        print("Checking if Ollama server is running...")
        result = subprocess.run(
            ["pgrep", "-f", "ollama"], capture_output=True, text=True
        )
        if not result.stdout:
            logger.info("Ollama server not running. Starting it in the background...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Ollama server starting in background. Waiting for it to be ready...")
            if not await _wait_for_ollama_ready(model_name="wizardlm2:7b"):
                logger.error("Ollama server did not become ready in time.")
                return False
            logger.info("Ollama server started and ready.")
            print("Ollama server confirmed started and ready.")
        else:
            logger.info("Ollama server is already running.")
            print("Ollama server is already running.")
            print("Verifying Ollama server readiness...")
            if not await _wait_for_ollama_ready(model_name="wizardlm2:7b"):
                logger.error("Ollama server is running but not responsive.")
                return False
            print("Ollama server confirmed responsive.")

        logger.info("Ensured wizardlm2:7b model is available.")
        # Explicitly pull the model to ensure it's loaded into VRAM
        print("Ensuring wizardlm2:7b model is loaded into VRAM...")
        try:
            pull_result = subprocess.run(
                ["ollama", "pull", "wizardlm2:7b"],
                capture_output=True,
                text=True,
                timeout=300,  # Increased timeout for model pull
            )
            if pull_result.returncode != 0:
                logger.error(f"Failed to pull wizardlm2:7b model: {pull_result.stderr}")
                return False
            print("wizardlm2:7b model pull/load complete.")
        except subprocess.TimeoutExpired:
            logger.error("Ollama model pull timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during Ollama model pull: {e}")
            return False

    except FileNotFoundError:
        logger.error(
            "'ollama' command not found. Please ensure Ollama is installed and in your PATH."
        )
        return False
    except Exception as e:
        logger.error(f"An error occurred while managing Ollama: {e}")
        return False
    return True


async def run_mypy_check(exclude_patterns: list[str]) -> int:
    """Runs mypy and returns the number of errors."""
    mypy_executable = str(Path(project_root) / ".venv" / "bin" / "mypy")
    print(f"Mypy executable path: {mypy_executable}")
    print(f"Project root for mypy: {project_root}")
    command = [mypy_executable, str(Path(project_root) / "tests" / "test_llm_fix.py")]

    # Set PYTHONPATH for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    process = await asyncio.create_subprocess_exec(
        *command,  # Pass the command list directly
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_root,  # Ensure it runs from the project root
        env=env,  # Pass the modified environment
    )
    stdout, stderr = await process.communicate()

    output = stdout.decode().strip()
    error_output = stderr.decode().strip()
    return_code = process.returncode

    print(f"\n--- Raw MyPy Output (Return Code: {return_code}) ---")
    print(output)
    if error_output:
        print("--- Raw MyPy Stderr ---")
        print(error_output)
    print("----------------------------------")

    error_count = 0
    for line in output.splitlines():
        if " error: " in line:
            error_count += 1

    if error_count == 0 and "Found 0 errors in" in output:
        return 0
    elif "Found" in output and "errors in" in output:
        try:
            # Extract error count from the last line like "Found X errors in Y files"
            last_line = output.splitlines()[-1]
            if "errors prevented further checking" in last_line:
                logger.warning(f"Mypy errors prevented further checking: {last_line}")
                # If errors prevent further checking, we can't get an exact count,
                # but we know there are errors. Return a high number or -1 to indicate this.
                return -1
            parts = last_line.split(" ")
            error_count = int(parts[1])
            return error_count
        except Exception as e:
            logger.error(f"Failed to parse mypy output: {e}. Output: {output}")
            return -1  # Indicate parsing error
    return error_count


async def debug_agent_mew():
    """
    Initializes and tests the AgentMew through the BasicAgentInterface.
    """
    print("--- Initializing AgentMew Debug Test ---")

    # 1. Ensure Ollama is ready
    print("Attempting to manage Ollama server...")
    if not await manage_ollama():
        print("Failed to manage Ollama server. Exiting.")
        return
    print("Ollama server managed successfully.")

    interface = BasicAgentInterface()
    agent_id = "debug_mew_001"
    model_name = "wizardlm2:7b"  # Use the model the user expects

    # 2. Create Agent with specific model config
    print(
        f"Attempting to create AgentMew with ID: {agent_id} and model {model_name}..."
    )
    try:
        agent_config = {"llm_model": model_name}
        await interface.create_agent("agent_mew", agent_id, config=agent_config)
        print(f"AgentMew created with ID: {agent_id} and model {model_name}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. Start Agent
    print(f"Attempting to start AgentMew with ID: {agent_id}...")
    try:
        session_id = await interface.start_agent(agent_id)
        print(f"AgentMew started with Session ID: {session_id}")
    except Exception as e:
        print(f"Error starting agent: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Get baseline mypy errors
    print("\n--- Getting baseline MyPy errors ---")
    baseline_errors = await run_mypy_check([])
    print(f"Baseline MyPy errors: {baseline_errors}")

    # 5. Create a synthetic test case to trigger LLM fix
    print("\n--- Attempting to fix a synthetic error to test LLM escalation ---")
    test_file_path = Path(project_root) / "tests" / "test_llm_fix.py"
    # Process only the synthetic test file
    python_files = [str(test_file_path)]

    fixed_files_count = 0
    total_fix_attempts = 0

    for file_path_str in python_files:
        print(f"Attempting to fix: {file_path_str}")
        print(f"Calling agent method 'fix_file' for {file_path_str}...")
        try:
            fix_session = await interface.call_agent_method(
                agent_id, "fix_file", file_path=file_path_str, error_source="mypy"
            )
            print(f"Agent method 'fix_file' call completed for {file_path_str}.")
            total_fix_attempts += len(fix_session.fix_attempts)
            if fix_session.success_rate > 0:
                fixed_files_count += 1
            print(
                f"  Fix session result for {file_path_str}: Success Rate={fix_session.success_rate:.1%}, Fixed={len(fix_session.original_errors) - len(fix_session.final_errors)}"
            )
        except Exception as e:
            print(f"  Error calling fix_file for {file_path_str}: {e}")
            import traceback

            traceback.print_exc()

    print(
        f"\n--- Completed fix attempts for {len(python_files)} files. Total fix attempts: {total_fix_attempts} ---"
    )

    # 6. Re-run mypy to check for reduction
    print("\n--- Re-running MyPy check after fixes ---")
    after_fix_errors = await run_mypy_check([])
    print(f"MyPy errors after fixes: {after_fix_errors}")

    if baseline_errors != -1 and after_fix_errors != -1:
        reduction = baseline_errors - after_fix_errors
        print(f"Total MyPy error reduction: {reduction}")
        if reduction >= 30:
            print("✅ Achieved target reduction of 30 MyPy errors!")
        else:
            print(
                f"❌ Did not achieve target reduction. Reduced by {reduction} errors."
            )
    else:
        print(
            "Could not determine exact reduction due to mypy errors preventing full checking."
        )

    # 7. Stop Agent
    print(f"Attempting to stop AgentMew with ID: {agent_id}...")
    try:
        await interface.stop_agent(agent_id)
        print(f"\nAgentMew {agent_id} stopped.")
    except Exception as e:
        print(f"Error stopping agent: {e}")
        import traceback

        traceback.print_exc()

    print("--- AgentMew Debug Test Finished ---")


if __name__ == "__main__":
    asyncio.run(debug_agent_mew())
