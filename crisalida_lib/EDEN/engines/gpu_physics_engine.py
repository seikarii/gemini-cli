"""
GPUPhysicsEngine - High-Performance Physics Simulation Core for Crisalida Metacosmos
===================================================================================
Leverages ModernGL for GPU-accelerated computation of entity and lattice dynamics.
Manages GPU buffers, dispatches compute shaders, and synchronizes simulation data
between CPU and GPU. Ready for integration with advanced simulation pipelines.

Features:
- Dynamic buffer management for entities and lattices
- Uniform buffer for simulation parameters
- Compute shader dispatch with diagnostics
- Safe resource management and extensibility
"""

import logging
import os
import struct
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

try:
    import moderngl
except Exception:
    moderngl = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        RealityBytecode,
    )
    from crisalida_lib.EVA.typequalia import QualiaState
else:
    # Runtime fallbacks to keep module import safe when EVA subsystem isn't
    # available at import time.
    EVAExperience = Any  # type: ignore
    LivingSymbolRuntime = Any  # type: ignore
    RealityBytecode = Any  # type: ignore
    QualiaState = Any  # type: ignore

logger = logging.getLogger(__name__)


class GPUPhysicsEngine:
    """
    Main GPU physics engine for Crisalida Metacosmos.
    Handles buffer management, compute shader dispatch, and simulation data transfer.
    """

    def __init__(
        self,
        ctx: Optional["moderngl.Context"],
        max_entities: int = 10000,
        max_lattices: int = 1000,
    ):
        self.ctx = ctx
        self.max_entities = max_entities
        self.max_lattices = max_lattices

        # Instance attributes with defensive, pydantic-friendly annotations
        self.entity_buffer: moderngl.Buffer | None = None
        self.lattice_buffer: moderngl.Buffer | None = None
        self.uniform_buffer: moderngl.Buffer | None = None
        self.entities_data: np.ndarray | None = None
        self.lattices_data: np.ndarray | None = None

        self.entity_count = 0
        self.lattice_count = 0
        self.simulation_tick = 0
        self.status = "initialized"

        if self.ctx is None:
            raise ValueError("ModernGL context cannot be None")
        shader_path = os.path.join(
            os.path.dirname(__file__), "shaders", "ontological_physics.glsl"
        )
        with open(shader_path) as f:
            shader_source = f.read()
        self.compute_shader = self.ctx.compute_shader(shader_source)

    def _create_entity_buffer(self, entities_data: np.ndarray):
        """Creates or updates the entity buffer on the GPU."""
        if self.ctx is None:
            logger.error("ModernGL context is None. Cannot create entity buffer.")
            return
        if self.entity_buffer:
            self.entity_buffer.release()
        self.entity_buffer = self.ctx.buffer(entities_data.astype("f4").tobytes())
        self.entity_count = len(entities_data)
        logger.debug(f"Created entity buffer with {self.entity_count} entities.")

    def _create_lattice_buffer(self, lattices_data: np.ndarray):
        """Creates or updates the lattice buffer on the GPU."""
        if self.ctx is None:
            logger.error("ModernGL context is None. Cannot create lattice buffer.")
            return
        if self.lattice_buffer:
            self.lattice_buffer.release()
        self.lattice_buffer = self.ctx.buffer(lattices_data.astype("f4").tobytes())
        self.lattice_count = len(lattices_data)
        logger.debug(f"Created lattice buffer with {self.lattice_count} lattices.")

    def _create_uniform_buffer(
        self,
        delta_time: float = 0.016,
        reality_coherence: float = 1.0,
        unified_field_center: tuple = (0.0, 0.0, 0.0),
        chaos_entropy_level: float = 0.0,
        time_dilation_factor: float = 1.0,
    ):
        """Creates or updates the uniform buffer on the GPU."""
        if self.ctx is None:
            logger.error("ModernGL context is None. Cannot create uniform buffer.")
            return
        # Pack fields: delta_time (f), reality_coherence (f),
        # entity_count (I), lattice_count (I), unified_field_center (3f),
        # chaos_entropy_level (f), time_dilation_factor (f), simulation_tick (I)
        fmt = "ffIIfffffI"
        try:
            uniform_data = struct.pack(
                fmt,
                float(delta_time),
                float(reality_coherence),
                int(self.entity_count),
                int(self.lattice_count),
                float(unified_field_center[0]),
                float(unified_field_center[1]),
                float(unified_field_center[2]),
                float(chaos_entropy_level),
                float(time_dilation_factor),
                int(self.simulation_tick),
            )
        except struct.error:
            # Fallback to a simpler, more compatible packing if needed
            uniform_data = struct.pack(
                "ffIIfffI",
                float(delta_time),
                float(reality_coherence),
                int(self.entity_count),
                int(self.lattice_count),
                float(unified_field_center[0]),
                float(unified_field_center[1]),
                float(unified_field_center[2]),
                int(self.simulation_tick),
            )
        if self.uniform_buffer:
            self.uniform_buffer.write(uniform_data)
        else:
            self.uniform_buffer = self.ctx.buffer(uniform_data)
        logger.debug("Updated uniform buffer.")

    def update_buffers(
        self,
        entities_data: "np.ndarray",
        lattices_data: "np.ndarray",
        simulation_tick: int,
        delta_time: float = 0.016,
        reality_coherence: float = 1.0,
        unified_field_center: tuple = (0.0, 0.0, 0.0),
        chaos_entropy_level: float = 0.0,
        time_dilation_factor: float = 1.0,
    ):
        """Updates all GPU buffers with current simulation data."""
        self.simulation_tick = simulation_tick
        self.entities_data = entities_data
        self.lattices_data = lattices_data
        self._create_entity_buffer(entities_data)
        self._create_lattice_buffer(lattices_data)
        self._create_uniform_buffer(
            delta_time,
            reality_coherence,
            unified_field_center,
            chaos_entropy_level,
            time_dilation_factor,
        )

    def compute(self):
        """Dispatches the compute shader to update entity states."""
        if not self.entity_buffer or not self.lattice_buffer or not self.uniform_buffer:
            logger.error("GPU buffers not initialized. Cannot compute.")
            return
        self.entity_buffer.bind_to_storage_buffer(0)
        self.lattice_buffer.bind_to_storage_buffer(1)
        self.uniform_buffer.bind_to_storage_buffer(2)
        num_work_groups = (self.entity_count + 63) // 64
        if num_work_groups == 0:
            logger.warning("No entities to process on GPU.")
            return
        self.compute_shader.run(group_x=num_work_groups)
        if self.ctx:
            self.ctx.finish()
        logger.debug(
            f"Compute shader dispatched for {self.entity_count} entities in {num_work_groups} groups."
        )
        self.status = "computed"

    def add_entity(self, entity_numpy_array: "np.ndarray") -> None:
        """Adds a new entity (as a NumPy array) to the CPU-side data and updates the GPU buffer."""
        if self.entities_data is None:
            self.entities_data = entity_numpy_array.reshape(1, -1)
        else:
            self.entities_data = np.vstack(
                [self.entities_data, entity_numpy_array.reshape(1, -1)]
            )
        self._create_entity_buffer(self.entities_data)
        logger.debug(f"Added entity. Total entities: {self.entity_count}")

    def update_entity_state(
        self, entity_id: str, new_entity_numpy_array: "np.ndarray"
    ) -> None:
        """Updates the state of a specific entity on the CPU-side and then updates the GPU buffer."""
        try:
            idx = int(entity_id.split("_")[-1])
            if self.entities_data is not None and 0 <= idx < self.entity_count:
                self.entities_data[idx] = new_entity_numpy_array.reshape(-1)
                self._create_entity_buffer(self.entities_data)
                logger.debug(f"Updated entity {entity_id} state.")
            else:
                logger.warning(
                    f"Entity ID {entity_id} out of bounds for update or entities_data is None."
                )
        except ValueError:
            logger.warning(f"Could not parse entity ID {entity_id} for update.")

    def get_updated_entities_data(self) -> "np.ndarray":
        """Reads back the updated entity data from the GPU."""
        if not self.entity_buffer:
            return np.array([])
        updated_data_bytes = self.entity_buffer.read()
        # Assuming float32 structure
        # Ensure entities_data is present before using its shape
        if self.entities_data is None:
            return np.frombuffer(updated_data_bytes, dtype=np.float32)
        return np.frombuffer(updated_data_bytes, dtype=np.float32).reshape(
            -1, int(self.entities_data.shape[1])
        )

    def set_state(self, state: dict[str, Any]):
        """Sets the state of the GPUPhysicsEngine from a deserialized dictionary."""
        self.simulation_tick = state.get("simulation_tick", 0)
        entities_data = state.get("entities_data")
        if entities_data is not None:
            self.entities_data = entities_data
            self._create_entity_buffer(entities_data)
        lattices_data = state.get("lattices_data")
        if lattices_data is not None:
            self.lattices_data = lattices_data
            self._create_lattice_buffer(lattices_data)

    async def shutdown(self):
        """Releases all GPU resources asynchronously."""
        if self.entity_buffer:
            self.entity_buffer.release()
        if self.lattice_buffer:
            self.lattice_buffer.release()
        if self.uniform_buffer:
            self.uniform_buffer.release()
        self.status = "shutdown"
        logger.info("GPUPhysicsEngine resources released.")

    def get_diagnostics(self) -> dict[str, Any]:
        """Returns diagnostics for the current GPU physics engine state."""
        return {
            "entity_count": self.entity_count,
            "lattice_count": self.lattice_count,
            "simulation_tick": self.simulation_tick,
            "status": self.status,
        }


class EVAGPUPhysicsEngine(GPUPhysicsEngine):
    """
    GPUPhysicsEngine extendido para integración real con EVA y Adam.
    Compila, almacena, simula y recuerda experiencias físicas como RealityBytecode,
    soporta benchmarking, compresión adaptativa, hooks robustos, faseo, multiverso y API extendida.
    """

    def __init__(
        self,
        ctx: moderngl.Context | None,
        max_entities: int = 10000,
        max_lattices: int = 1000,
        phase: str = "default",
        max_experiences: int = 100000,
        retention_policy: str = "dynamic",
    ):
        super().__init__(ctx, max_entities, max_lattices)
        self.eva_phase = phase
        # Keep eva_runtime optional to avoid import-time side-effects in CI
        self.eva_runtime: LivingSymbolRuntime | None = None
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []
        self.benchmark_log: list = []
        self.max_experiences = max_experiences
        self.retention_policy = retention_policy

    def eva_ingest_physics_experience(
        self,
        metrics: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia física en RealityBytecode y la almacena en la memoria EVA.
        Benchmarking y registro de hooks.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.6,
            cognitive_complexity=0.85,
            consciousness_density=0.7,
            narrative_importance=0.7,
            energy_level=1.0,
        )
        start = time.time()
        ts = float(time.time())
        experience_data = {
            "entity_count": self.entity_count,
            "lattice_count": self.lattice_count,
            "simulation_tick": self.simulation_tick,
            "metrics": metrics or {},
            "diagnostics": self.get_diagnostics(),
            "timestamp": ts,
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_GPU_PHYSICS_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        bytecode: list[Any] = []
        if _eva is not None:
            _dc = getattr(_eva, "divine_compiler", None)
            if _dc is not None and hasattr(_dc, "compile_intention"):
                try:
                    bytecode = _dc.compile_intention(intention)  # type: ignore[assignment]
                except Exception:
                    bytecode = []
        experience_id = f"eva_gpu_physics_{self.simulation_tick}_{uuid.uuid4().hex[:8]}"
        # ensure phase and timestamp are concrete for typed RealityBytecode
        phase = phase or self.eva_phase
        try:
            _phase = str(phase)
        except Exception:
            _phase = "default"
        _ts_src: Any = experience_data.get("timestamp", ts)
        try:
            _ts = float(_ts_src)
        except Exception:
            _ts = float(ts)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=_phase,
            timestamp=_ts,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        end = time.time()
        benchmark = {
            "operation": "ingest",
            "experience_id": experience_id,
            "phase": phase,
            "duration": end - start,
            "timestamp": reality_bytecode.timestamp,
            "entity_count": self.entity_count,
            "lattice_count": self.lattice_count,
            "simulation_tick": self.simulation_tick,
        }
        self.benchmark_log.append(benchmark)
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                logger.warning(f"[EVA-GPU-PHYSICS] Environment hook failed: {e}")
        self._auto_cleanup_eva_memory()
        return experience_id

    def eva_recall_physics_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia física almacenada, manifestando la simulación.
        Benchmarking y registro de hooks.
        """
        phase = phase or self.eva_phase
        start = time.time()
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA GPU physics experience"}
        _eva = getattr(self, "eva_runtime", None)
        quantum_field = (
            getattr(_eva, "quantum_field", None) if _eva is not None else None
        )
        manifestations = []
        if quantum_field:
            # Defensive iteration: reality_bytecode.instructions may be optional or wrapped
            instrs = getattr(reality_bytecode, "instructions", [])
            if instrs is None:
                instrs = []
            _exec = getattr(_eva, "execute_instruction", None)
            for instr in instrs:
                if _exec is None:
                    continue
                try:
                    symbol_manifest = _exec(instr, quantum_field)
                except Exception:
                    symbol_manifest = None
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            logger.warning(
                                f"[EVA-GPU-PHYSICS] Manifestation hook failed: {e}"
                            )
        end = time.time()
        benchmark = {
            "operation": "recall",
            "experience_id": reality_bytecode.bytecode_id,
            "phase": reality_bytecode.phase,
            "duration": end - start,
            "timestamp": reality_bytecode.timestamp,
        }
        self.benchmark_log.append(benchmark)
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
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
            "timestamp": eva_experience.timestamp,
            "benchmark": benchmark,
        }

    def add_experience_phase(
        self, experience_id: str, phase: str, metrics: dict, qualia_state: QualiaState
    ) -> None:
        """Add or update a stored EVA experience phase for an existing experience id.

        This method compiles an archival intention (best-effort) and stores a
        typed RealityBytecode under the given phase.
        """

        experience_data = {
            "entity_count": self.entity_count,
            "lattice_count": self.lattice_count,
            "simulation_tick": self.simulation_tick,
            "metrics": metrics or {},
            "diagnostics": self.get_diagnostics(),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_GPU_PHYSICS_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if (
            _eva is None
            or not hasattr(_eva, "divine_compiler")
            or _eva.divine_compiler is None
        ):
            bytecode: list[Any] = []
        else:
            _dc = _eva.divine_compiler
            bytecode = (
                _dc.compile_intention(intention)  # type: ignore[assignment]
                if hasattr(_dc, "compile_intention")
                else []
            )
        # ensure timestamp is a float for typed RealityBytecode
        _ts_src: Any = experience_data.get("timestamp", time.time())
        try:
            _ts = float(_ts_src)
        except Exception:
            _ts = float(time.time())

        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=_ts,
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self._auto_cleanup_eva_memory()

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-GPU-PHYSICS] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia física EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_benchmark_log(self) -> list:
        """Devuelve el historial de benchmarks de ingestión y recall."""
        return self.benchmark_log

    def optimize_eva_memory(self):
        """
        Compresión adaptativa real: elimina experiencias menos recientes y fusiona duplicados por tick y entidad_count.
        """
        # Eliminar experiencias menos recientes si se supera el máximo
        if len(self.eva_memory_store) > self.max_experiences:
            sorted_exps = sorted(
                self.eva_memory_store.items(),
                key=lambda x: getattr(x[1], "timestamp", 0),
            )
            for exp_id, _ in sorted_exps[
                : len(self.eva_memory_store) - self.max_experiences
            ]:
                del self.eva_memory_store[exp_id]
        # Compresión real: fusiona duplicados por simulation_tick y entity_count
        grouped = {}
        for exp_id, exp in self.eva_memory_store.items():
            key = (
                getattr(exp, "phase", None),
                getattr(exp, "qualia_state", None),
                getattr(exp, "simulation_tick", None),
                getattr(exp, "entity_count", None),
            )
            grouped.setdefault(key, []).append((exp_id, exp))
        for _key, group in grouped.items():
            if len(group) > 1:
                main_id, main_exp = group[0]
                for dup_id, _ in group[1:]:
                    if dup_id in self.eva_memory_store:
                        del self.eva_memory_store[dup_id]

    def _auto_cleanup_eva_memory(self):
        if self.retention_policy == "dynamic":
            self.optimize_eva_memory()

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_physics_experience": self.eva_ingest_physics_experience,
            "eva_recall_physics_experience": self.eva_recall_physics_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "get_benchmark_log": self.get_benchmark_log,
        }


#
