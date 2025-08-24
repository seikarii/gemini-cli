"""
RealityEngine - Core Simulation Manager for Janus Metacosmos
============================================================
Integrates event bus, reality clock, and GPU physics engine to drive the simulation.
Manages lifecycle, diagnostics, and extensible hooks for client/server interaction.
Ready for integration with WebSocket/FastAPI servers and external APIs.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crisalida_lib.EARTH.event_bus import event_bus
from crisalida_lib.EDEN.reality_clock import reality_clock

# Use typing-only imports to avoid importing large runtime modules at import time
if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QualiaState,
        QuantumField,
        RealityBytecode,
    )
else:
    # At runtime, these will be resolved locally where needed to break cycles
    EVAExperience = Any
    LivingSymbolRuntime = Any
    QualiaState = Any
    QuantumField = Any
    RealityBytecode = Any

logger = logging.getLogger(__name__)


class RealityEngine:
    """
    Main simulation engine for the Janus Metacosmos.
    Orchestrates event bus, clock, and physics engine.
    Provides lifecycle management, diagnostics, and extensibility.
    """

    def __init__(self, ctx=None):
        self.event_bus = event_bus
        self.reality_clock = reality_clock
        # Import GPUPhysicsEngine at runtime to avoid importing optional GPU deps
        try:
            from crisalida_lib.EDEN.engines.gpu_physics_engine import GPUPhysicsEngine

            self.physics_engine = GPUPhysicsEngine(ctx=ctx)
        except Exception:
            # Fallback lightweight shim when GPU engine or dependencies are unavailable
            class _FallbackPhysicsEngine:
                def __init__(self, ctx=None):
                    self.status = "fallback"

                async def run(self):
                    # noop loop to integrate with engine lifecycle
                    self.status = "running"
                    while True:
                        await asyncio.sleep(1)

                async def shutdown(self):
                    self.status = "shutdown"

            self.physics_engine = _FallbackPhysicsEngine(ctx=ctx)
        self._running = False
        self._tasks = []

        # defensive defaults used by consumers expecting these attrs
        # keep runtime imports local to avoid large/optional imports at module import time
        self.eva_runtime: LivingSymbolRuntime | None = None
        self.eva_phase: str = getattr(self, "eva_phase", "default")
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list = []

    async def run(self):
        """
        Starts the reality engine and its subsystems.
        Handles lifecycle, diagnostics, and graceful shutdown.
        """
        logger.info("[REALITY_ENGINE] Reality Engine is operational.")
        self._running = True
        try:
            clock_task = asyncio.create_task(self.reality_clock.start())
            physics_task = asyncio.create_task(self.physics_engine.run())
            self._tasks = [clock_task, physics_task]
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("[REALITY_ENGINE] Engine tasks cancelled. Shutting down.")
        except Exception as e:
            logger.error(f"[REALITY_ENGINE] Unhandled exception in run(): {e}")
        finally:
            self._running = False
            await self.shutdown()

    async def shutdown(self):
        """
        Gracefully shuts down the reality engine and its subsystems.
        """
        logger.info("[REALITY_ENGINE] Shutting down Reality Engine...")
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except Exception:
                    pass
        await self.physics_engine.shutdown()
        await self.reality_clock.stop()
        logger.info("[REALITY_ENGINE] Shutdown complete.")

    def is_running(self) -> bool:
        """Returns True if the engine is running."""
        return self._running

    def get_diagnostics(self) -> dict:
        """
        Returns diagnostics for the engine and subsystems.
        """
        return {
            "engine_running": self._running,
            "clock_status": getattr(self.reality_clock, "status", "unknown"),
            "physics_status": getattr(self.physics_engine, "status", "unknown"),
        }


class EVARealityEngine(RealityEngine):
    """
    RealityEngine extendido para integración con EVA.
    Orquesta el event bus, clock, physics engine y la memoria viviente EVA:
    ingestión, recall y simulación de experiencias, faseo, hooks de entorno y QuantumField.
    """

    def __init__(self, ctx=None, phase: str = "default") -> None:
        super().__init__(ctx=ctx)
        # try to construct a minimal runtime if available, otherwise leave None
        try:
            # local import to avoid circular import at module import time
            from crisalida_lib.EVA.core_types import LivingSymbolRuntime

            self.eva_runtime = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None
        self.eva_phase = phase
        # Use runtime-checked/compatible Any types to avoid assigning to typing-only symbols
        # Avoid type annotations on instance assignments inside functions (mypy/Python syntax)
        self.eva_memory_store = {}
        self.eva_experience_store = {}
        self.eva_phases = {}
        self._environment_hooks = []

    def eva_ingest_engine_experience(
        self, experience_data: dict, qualia_state: QualiaState, phase: str | None = None
    ) -> str:
        """
        Compila una experiencia de simulación global en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_ENGINE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        runtime = self.eva_runtime
        if runtime is None:
            raise RuntimeError("EVA runtime not initialized")
        # perform local imports for runtime types used at runtime
        # Ensure a runtime alias variable exists so it can be referenced whether the
        # import succeeds or fails (prevents NameError / mypy name-defined issues).
        # Resolve a runtime alias for RealityBytecode to avoid assigning typing
        # special-forms to module-level names (mypy flags those as non-callable).
        # Resolve runtime RealityBytecode inside the function scope. Use Any
        # as the runtime alias to avoid assigning typing special forms and
        # to make mypy accept callable checks and fallback assignments.
        RealityBytecodeRuntime: Any
        try:
            from crisalida_lib.EVA.core_types import (
                RealityBytecode as _RTRealityBytecode,
            )

            # cast to Any so assigning a typing-only name does not confuse mypy
            RealityBytecodeRuntime = cast(Any, _RTRealityBytecode)  # type: ignore[assignment]
        except Exception:
            # Fallback: use a runtime-resolved name or Any
            RealityBytecodeRuntime = globals().get("RealityBytecode", Any)

        # Guard against runtime or divine_compiler being None in compatibility shims
        _dc = getattr(runtime, "divine_compiler", None)
        if _dc is None:
            bytecode = []
        else:
            compile_fn = getattr(_dc, "compile_intention", None)
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
            else:
                bytecode = []
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_engine_{hash(str(experience_data))}"
        )
        try:
            # Try constructing the runtime RealityBytecode if callable
            if callable(RealityBytecodeRuntime):
                reality_bytecode = RealityBytecodeRuntime(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                )
            else:
                raise TypeError("RealityBytecodeRuntime not callable")
        except Exception:
            from types import SimpleNamespace

            # Cast fallback to Any so assignments to variables typed as
            # RealityBytecode/Any are accepted by mypy.
            reality_bytecode = cast(
                Any,
                SimpleNamespace(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                ),
            )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_engine_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia global almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA engine experience"}
        # local import to avoid module-level circular import
        try:
            from crisalida_lib.EVA.core_types import QuantumField

            quantum_field = QuantumField()
        except Exception:
            quantum_field = QuantumField  # type: ignore
        manifestations = []
        runtime = self.eva_runtime
        if runtime is None:
            raise RuntimeError("EVA runtime not initialized")
        for instr in reality_bytecode.instructions:
            symbol_manifest = None
            try:
                exec_fn = getattr(runtime, "execute_instruction", None)
                if callable(exec_fn):
                    res = exec_fn(instr, quantum_field)
                    # if runtime returns awaitable, avoid awaiting here; mark as pending
                    try:
                        import inspect

                        if inspect.isawaitable(res):
                            symbol_manifest = None
                        else:
                            symbol_manifest = res
                    except Exception:
                        symbol_manifest = res
            except Exception:
                symbol_manifest = None
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.warning(f"[EVA-ENGINE] Environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
        }

    def add_engine_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia global.
        """
        intention = {
            "intention_type": "ARCHIVE_ENGINE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        runtime = self.eva_runtime
        if runtime is None:
            raise RuntimeError("EVA runtime not initialized")
        # Resolve divine_compiler safely and check for callable compile_intention
        # Ensure a runtime alias for RealityBytecode is available in this scope.
        RealityBytecodeRuntime = Any
        try:
            from crisalida_lib.EVA.core_types import RealityBytecode as _RTRealityBytecode

            # cast to Any so mypy does not enforce the typed constructor signature
            RealityBytecodeRuntime = cast(Any, _RTRealityBytecode)  # type: ignore[assignment]
        except Exception:
            from typing import Any as _Any

            RealityBytecodeRuntime = globals().get("RealityBytecode", _Any)
        _dc = getattr(runtime, "divine_compiler", None)
        if _dc is None:
            bytecode = []
        else:
            compile_fn = getattr(_dc, "compile_intention", None)
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
            else:
                bytecode = []
        try:
            # mypy may consider a typing-only RealityBytecode as non-callable; be defensive
            if callable(RealityBytecodeRuntime):
                reality_bytecode = RealityBytecodeRuntime(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                )
            else:
                raise TypeError("RealityBytecodeRuntime not callable")
        except Exception:
            # fallback simple namespace when the runtime RealityBytecode ctor differs
            from types import SimpleNamespace

            reality_bytecode = cast(
                Any,
                SimpleNamespace(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                ),
            )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-ENGINE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia global."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_engine_experience": self.eva_ingest_engine_experience,
            "eva_recall_engine_experience": self.eva_recall_engine_experience,
            "add_engine_experience_phase": self.add_engine_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# Example usage (for testing/demo)
if __name__ == "__main__":

    async def main():
        engine = RealityEngine()
        try:
            await engine.run()
        except KeyboardInterrupt:
            logger.info("[REALITY_ENGINE] Interrupted by user.")
            await engine.shutdown()

    asyncio.run(main())
