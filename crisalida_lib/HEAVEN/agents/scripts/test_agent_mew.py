import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from crisalida_lib.ADAM.mente.mind_core import PasoAccion, PlanDeAccion, TipoPaso
from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.HEAVEN.agents.agent_mew import AgentMew, FixResult

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Test Setup ---
TEST_FILE_PATH = "/tmp/test_mew_file.py"
TEST_FILE_CONTENT_BROKEN = "import os\n\ndef my_func():\n    print('hello')\n    unused_var = 1 # This should trigger a fixable error\n"
# Let's use an error that is not in the default patterns
TEST_ERROR_CODE = "X999"  # A non-existent error code to force LLM escalation

TEST_FILE_CONTENT_FIXED = "import os\n\ndef my_func():\n    print('hello')\n    # unused_var = 1 # This should trigger a fixable error\n"


async def setup_test_file():
    """Creates a temporary file for testing."""
    with open(TEST_FILE_PATH, "w") as f:
        f.write(TEST_FILE_CONTENT_BROKEN)
    logger.info(f"Created test file at {TEST_FILE_PATH}")


def cleanup_test_file():
    """Removes the temporary test file."""
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)
        logger.info(f"Cleaned up test file at {TEST_FILE_PATH}")


async def test_llm_fix_and_learn():
    """Tests the full LLM fix and learn cycle."""
    logger.info("🚀 Starting LLM Fix & Learn Test...")
    agent_mew = AgentMew()
    agent_mew.confidence_threshold = 0.0  # Allow attempting fixes for unknown errors

    # --- Mocking Adam's Responses ---

    # 1. Mock the plan for the FIX
    fix_plan = PlanDeAccion(
        justificacion="Plan to fix unused variable by commenting it out.",
        pasos=[
            PasoAccion(
                tipo=TipoPaso.MODIFICACION,
                herramienta="replace",
                parametros={
                    "file_path": TEST_FILE_PATH,
                    "old_string": "    unused_var = 1",
                    "new_string": "    # unused_var = 1",
                },
            )
        ],
    )

    # 2. Mock the plan for LEARNING
    new_pattern_json = {
        "error_codes": [TEST_ERROR_CODE],
        "description": "A variable was assigned but never used.",
        "fix_strategy": "text_replace",
        "confidence": 0.8,
        "template": "Comment out the unused variable assignment.",
    }
    learn_plan = PlanDeAccion(
        justificacion="Plan to learn a new pattern.",
        pasos=[
            PasoAccion(
                tipo=TipoPaso.ACCION,
                herramienta="return_data",
                parametros={"data": new_pattern_json},
            )
        ],
    )

    # Mock adam.update to return the correct plan based on the objective
    adam_update_mock = AsyncMock()

    def side_effect(*args, **kwargs):
        perception = kwargs.get("perception", {})
        request = perception.get("external_request", {}).get("original_request", "")
        if "Fix the error" in request:
            agent_mew.adam.mind.historia_sintesis.append({"plan_final": fix_plan})
        elif "Analyze this successful bug fix" in request:
            agent_mew.adam.mind.historia_sintesis.append({"plan_final": learn_plan})
        return MagicMock()

    adam_update_mock.side_effect = side_effect

    with patch.object(
        agent_mew, "_detect_errors", new_callable=AsyncMock
    ) as detect_errors_mock:
        with patch.object(agent_mew.adam, "update", new=adam_update_mock):
            # Initial detection finds the error, subsequent calls find none.
            detect_errors_mock.side_effect = [
                [
                    {
                        "code": TEST_ERROR_CODE,
                        "message": "A completely new and unfixable error",
                        "line": 4,
                        "column": 5,
                    }
                ],  # First call in fix_file
                [],  # Second call in _attempt_fix_with_llm
                [],  # Third call for final_errors in fix_file
            ]

            # --- Execute the fix ---
            session = await agent_mew.fix_file(TEST_FILE_PATH)

            # --- Assertions ---
            logger.info("Asserting test outcomes...")

            # Assert fix was successful
            assert session.success_rate == 1.0, "Fix session should be successful."
            llm_attempt = next((a for a in session.fix_attempts if a.is_llm_fix), None)
            assert llm_attempt is not None, "An LLM attempt should have been made."
            assert (
                llm_attempt.result == FixResult.SUCCESS
            ), "LLM fix attempt should succeed."

            # Assert learning happened
            assert (
                agent_mew.stats["new_patterns_learned"] == 1
            ), "A new pattern should have been learned."
            assert (
                TEST_ERROR_CODE in agent_mew.pattern_matcher.patterns
            ), "New pattern should be in the matcher."
            new_pattern = agent_mew.pattern_matcher.patterns[TEST_ERROR_CODE]
            assert (
                new_pattern.description == new_pattern_json["description"]
            ), "Learned pattern description is incorrect."

            logger.info("✅ LLM Fix & Learn Test Completed Successfully!")


async def test_autonomous_mode():
    """Tests the mission-driven autonomous mode."""
    logger.info("🚀 Starting Autonomous Mode Test...")
    agent_mew = AgentMew()
    agent_mew.confidence_threshold = 0.0  # Allow attempting fixes for unknown errors

    # Mock the glob tool to return our test file
    glob_mock = AsyncMock(
        return_value=ToolCallResult(
            command="glob", success=True, output=json.dumps({"files": [TEST_FILE_PATH]})
        )
    )
    # Mock fix_file to just record the call
    fix_file_mock = AsyncMock()

    with patch.object(agent_mew.glob_tool, "execute", new=glob_mock):
        with patch.object(agent_mew, "fix_file", new=fix_file_mock):
            # Assign the mission
            agent_mew.assign_mission("fix_codebase")

            # Run the autonomous mode for a very short time
            try:
                await asyncio.wait_for(agent_mew.run_autonomous_mode(), timeout=1.0)
            except TimeoutError:
                logger.info("Autonomous mode loop ran as expected.")

            # --- Assertions ---
            fix_file_mock.assert_called_once_with(TEST_FILE_PATH)
            assert (
                agent_mew.current_mission is None
            ), "Mission should be cleared after execution."

            logger.info("✅ Autonomous Mode Test Completed Successfully!")


async def test_monitoring():
    """Tests the MewMonitor integration and functionality."""
    logger.info("🚀 Starting MewMonitor Test...")
    agent_mew = AgentMew()

    # Mock the monitor's collect_metrics to be called immediately for test control
    with patch.object(
        agent_mew.monitor, "collect_metrics", new=AsyncMock()
    ) as mock_collect_metrics:
        # Give the monitor a moment to collect initial metrics (mocked)
        await mock_collect_metrics()

        # Get initial status
        status_report = agent_mew.get_status()
        logger.info(f"Initial monitor status: {status_report}")

        # Assert basic structure and values
        assert status_report["status"] == "running"
        assert "cpu_usage_percent" in status_report
        assert "memory_usage_gb" in status_report
        assert "agent_stats" in status_report
        assert status_report["agent_stats"]["files_processed"] == 0  # Initial state

        # Simulate some activity to change agent_stats
        agent_mew.stats["files_processed"] = 5
        agent_mew.stats["errors_fixed"] = 3
        await agent_mew.monitor.collect_metrics()  # Force monitor to collect new stats

        updated_status_report = agent_mew.get_status()
        logger.info(f"Updated monitor status: {updated_status_report}")
        assert updated_status_report["agent_stats"]["files_processed"] == 5
        assert updated_status_report["agent_stats"]["errors_fixed"] == 3

        # Test shutdown
        await agent_mew.shutdown()
        assert (
            not agent_mew.monitor.is_running
        ), "Monitor should be stopped after shutdown."
        logger.info("✅ MewMonitor Test Completed Successfully!")


async def main():
    try:
        await setup_test_file()
        await test_llm_fix_and_learn()
        # The autonomous mode test needs a fresh file
        await setup_test_file()
        await test_autonomous_mode()
        # Monitoring test
        await test_monitoring()
    finally:
        cleanup_test_file()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 AgentMew tests stopped by user")
    except Exception as e:
        logger.error(f"Fatal error during AgentMew tests: {e}", exc_info=True)
        sys.exit(1)
