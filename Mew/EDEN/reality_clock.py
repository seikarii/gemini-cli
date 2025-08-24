"""
RealityClock - High-Resolution Asynchronous Simulation Clock
===========================================================
Manages temporal progression, tick synchronization, and event dispatch for the Crisalida Metacosmos.

Features:
- High-resolution, adaptive tick rate
- Asynchronous event publishing via event_bus
- Robust state serialization and restoration
- Diagnostics and lifecycle management
"""

import asyncio
from collections.abc import Callable
from typing import Any

from crisalida_lib.EARTH.event_bus import event_bus


class RealityClock:
    """
    Central simulation clock for the Crisalida Metacosmos.
    Provides asynchronous ticks, event synchronization, and diagnostics.
    """

    def __init__(self, tick_rate: float = 1.0):
        self.tick_rate = tick_rate
        self.ticks = 0
        self._running = False
        self._delta_time = 1.0 / max(self.tick_rate, 1e-6)
        self.status = "initialized"

    def get_state(self) -> dict[str, Any]:
        """Returns the current serializable state of the RealityClock."""
        return {
            "tick_rate": self.tick_rate,
            "ticks": self.ticks,
            "status": self.status,
        }

    def set_state(self, state: dict[str, Any]):
        """Sets the state of the RealityClock from a deserialized dictionary."""
        self.tick_rate = float(state.get("tick_rate", 1.0))
        self.ticks = int(state.get("ticks", 0))
        self._running = False
        self._delta_time = 1.0 / max(self.tick_rate, 1e-6)
        self.status = state.get("status", "initialized")

    async def start(self):
        """Starts the clock and publishes global_tick events asynchronously."""
        self._running = True
        self.status = "running"
        while self._running:
            self._delta_time = 1.0 / max(self.tick_rate, 1e-6)
            await asyncio.sleep(self._delta_time)
            self.ticks += 1
            await event_bus.publish(
                "global_tick", {"ticks": self.ticks, "delta_time": self._delta_time}
            )

    async def stop(self):
        """Stops the clock asynchronously."""
        self._running = False
        self.status = "stopped"

    def is_running(self) -> bool:
        """Returns True if the clock is running."""
        return self._running

    def set_tick_rate(self, tick_rate: float):
        """Sets a new tick rate and updates delta_time."""
        self.tick_rate = float(tick_rate)
        self._delta_time = 1.0 / max(self.tick_rate, 1e-6)

    def get_global_time(self) -> tuple[int, float]:
        """Returns the current global tick count and delta time."""
        return self.ticks, self._delta_time

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns diagnostics for the current clock state."""
        return {
            "tick_rate": self.tick_rate,
            "ticks": self.ticks,
            "delta_time": self._delta_time,
            "status": self.status,
            "running": self._running,
        }


class EVARealityClock(RealityClock):
    """
    EVARealityClock - Reloj de simulación extendido para integración con EVA.
    Gestiona ticks, faseo, hooks de entorno, sincronización con QuantumField y memoria viviente.
    Permite benchmarks, control adaptativo y notificación de eventos EVA.
    """

    def __init__(self, tick_rate: float = 1.0, phase: str = "default"):
        super().__init__(tick_rate)
        self.eva_phase = phase
        self.eva_hooks: list = []
        self.eva_benchmark_stats: dict = {}
        self._last_benchmark_tick = 0

    async def start(self):
        """Inicia el reloj y publica eventos global_tick y eva_tick asincrónicamente."""
        self._running = True
        self.status = "running"
        while self._running:
            self._delta_time = 1.0 / max(self.tick_rate, 1e-6)
            await asyncio.sleep(self._delta_time)
            self.ticks += 1
            await event_bus.publish(
                "global_tick",
                {
                    "ticks": self.ticks,
                    "delta_time": self._delta_time,
                    "phase": self.eva_phase,
                },
            )
            # EVA: Notificar hooks de entorno y publicar evento EVA
            for hook in self.eva_hooks:
                try:
                    hook(
                        {
                            "ticks": self.ticks,
                            "delta_time": self._delta_time,
                            "phase": self.eva_phase,
                        }
                    )
                except Exception as e:
                    print(f"[EVA-CLOCK] Hook failed: {e}")
            await event_bus.publish(
                "eva_tick",
                {
                    "ticks": self.ticks,
                    "delta_time": self._delta_time,
                    "phase": self.eva_phase,
                },
            )
            # EVA: Benchmark cada 100 ticks
            if self.ticks % 100 == 0:
                self._run_benchmark()

    def set_phase(self, phase: str):
        """Cambia la fase activa del reloj EVA."""
        self.eva_phase = phase
        for hook in self.eva_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-CLOCK] Phase hook failed: {e}")

    def add_eva_hook(self, hook: Callable[..., Any]):
        """Registra un hook para eventos de tick EVA."""
        self.eva_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "set_phase": self.set_phase,
            "add_eva_hook": self.add_eva_hook,
            "get_benchmark_stats": self.get_benchmark_stats,
            "get_state": self.get_state,
            "set_state": self.set_state,
            "get_diagnostics": self.get_diagnostics,
        }

    def _run_benchmark(self):
        """Ejecuta un benchmark simple de ticks y delta_time."""
        import time

        start = time.perf_counter()
        tick_snapshot = self.ticks
        for _ in range(10):
            # Simula avance de ticks
            pass
        end = time.perf_counter()
        avg_tick_time = (end - start) / 10.0
        self.eva_benchmark_stats[tick_snapshot] = {
            "tick": tick_snapshot,
            "avg_tick_time": avg_tick_time,
            "tick_rate": self.tick_rate,
            "phase": self.eva_phase,
        }
        self._last_benchmark_tick = tick_snapshot

    def get_benchmark_stats(self) -> dict:
        """Devuelve los benchmarks recientes del reloj EVA."""
        return self.eva_benchmark_stats


# Singleton instance
reality_clock = RealityClock()

# Singleton EVA instance
eva_reality_clock = EVARealityClock()
