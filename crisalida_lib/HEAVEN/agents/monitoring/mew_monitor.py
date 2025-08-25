"""
MewMonitor - Internal Monitoring System for AgentMew
====================================================

Monitors the internal state, performance, and resource usage of an AgentMew instance.
This runs *inside* the agent's process and provides real-time diagnostics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from crisalida_lib.HEAVEN.agents.agent_mew import AgentMew

logger = logging.getLogger(__name__)


@dataclass
class MonitorMetrics:
    """A snapshot of the agent's metrics at a point in time."""

    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage_gb: float = 0.0
    agent_stats: dict[str, Any] = field(default_factory=dict)


class MewMonitor:
    """Monitors an instance of AgentMew."""

    def __init__(self, agent: "AgentMew", check_interval: int = 10):
        """
        Initializes the monitor.

        Args:
            agent: The AgentMew instance to monitor.
            check_interval: The interval in seconds between metric collections.
        """
        self.agent = agent
        self.check_interval = check_interval
        self.is_running = False
        self._monitor_task: asyncio.Task | None = None
        self._process = psutil.Process()
        self.metrics_history: list[MonitorMetrics] = []
        self.max_history_len = 1000

    def start(self):
        """Starts the monitoring loop as an asyncio task."""
        if self.is_running:
            logger.warning("MewMonitor is already running.")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("MewMonitor started.")

    async def stop(self):
        """Stops the monitoring loop."""
        if not self.is_running or not self._monitor_task:
            logger.warning("MewMonitor is not running.")
            return

        self.is_running = False
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            logger.info("MewMonitor task was successfully cancelled.")
        logger.info("MewMonitor stopped.")

    async def _monitoring_loop(self):
        """The main loop that periodically collects metrics."""
        while self.is_running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break  # Exit loop cleanly on cancellation
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Avoid rapid-fire errors
                await asyncio.sleep(self.check_interval * 2)

    async def collect_metrics(self):
        """Collects the current metrics and stores them."""
        try:
            with self._process.oneshot():
                cpu_usage = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                memory_usage_gb = memory_info.rss / (1024**3)  # Convert bytes to GB

            agent_stats = self.agent.stats.copy()

            metrics = MonitorMetrics(
                cpu_usage=cpu_usage,
                memory_usage_gb=memory_usage_gb,
                agent_stats=agent_stats,
            )

            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_len:
                self.metrics_history.pop(0)

            logger.debug(
                f"Collected new metrics: CPU {cpu_usage}%, Mem {memory_usage_gb:.3f} GB"
            )

        except psutil.NoSuchProcess:
            logger.error("MewMonitor: The agent process no longer exists.")
            self.is_running = False
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}", exc_info=True)

    def get_latest_metrics(self) -> MonitorMetrics | None:
        """Returns the most recently collected metrics."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]

    def get_status_report(self) -> dict[str, Any]:
        """Generates a dictionary with the current status and latest metrics."""
        latest_metrics = self.get_latest_metrics()
        if not latest_metrics:
            return {"status": "initializing", "message": "No metrics collected yet."}

        return {
            "status": "running" if self.is_running else "stopped",
            "timestamp": latest_metrics.timestamp,
            "cpu_usage_percent": latest_metrics.cpu_usage,
            "memory_usage_gb": round(latest_metrics.memory_usage_gb, 3),
            "agent_stats": latest_metrics.agent_stats,
        }
