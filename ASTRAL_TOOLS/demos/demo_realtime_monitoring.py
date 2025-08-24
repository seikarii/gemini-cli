import asyncio
import logging
import os
import shutil

from crisalida_lib.ASTRAL_TOOLS.realtime_monitoring import RealtimeMonitoringTool

logger = logging.getLogger(__name__)


async def demo_realtime_monitoring():
    """Demonstrate the real-time monitoring tool"""
    print("🔍 REAL-TIME MONITORING TOOL DEMO")
    print("=" * 50)

    tool = RealtimeMonitoringTool()

    # Test file monitoring
    print("\n1. Testing file monitoring...")
    temp_dir = "/tmp/monitor_test"
    os.makedirs(temp_dir, exist_ok=True)

    result = await tool.execute(
        action="start_file_monitor",
        path=temp_dir,
        monitor_id="test_monitor",
        event_types=["created", "modified", "deleted"],
    )
    print(f"Start file monitor: {result.output}")

    # Create a test file
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hello, monitoring!")

    # Wait for events
    await asyncio.sleep(1)

    # Get events
    result = await tool.execute(action="get_events", max_events=10)
    print(f"Get events: {result.output}")
    if result.metadata and "events" in result.metadata:
        for event in result.metadata["events"]:
            print(f"  Event: {event['event_type']} - {event['source_path']}")

    # Stop monitor
    result = await tool.execute(action="stop_monitor", monitor_id="test_monitor")
    print(f"Stop monitor: {result.output}")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n✅ Real-time monitoring tool demo completed!")
