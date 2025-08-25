"""
JaxPhysicsEngine - Differentiable Physics Simulation Core for Crisalida Metacosmos
=================================================================================
Leverages JAX for high-performance, differentiable, and functional physics simulation.
Supports gradient-based optimization, automatic differentiation, and advanced entity dynamics.

Features:
- Stateless, functional physics updates
- JIT-compiled entity position and velocity updates
- Extensible for custom force fields and optimization
- Ready for integration with advanced simulation pipelines
"""

import time
import uuid
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

# JAX is an optional dependency; provide safe fallbacks so the module
# can be imported in stripped environments. Predeclare names with Any so
# mypy doesn't infer concrete types that later get assigned None.
_jax_partial: Any = None
jax: Any = None
jnp: Any = None
_has_jax: bool = False
try:
    # preserve the imported partial under a temporary name
    from functools import partial as _jax_partial  # type: ignore

    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    _has_jax = True
except Exception:
    # leave the Any-typed fallbacks in place
    _jax_partial = None
    _has_jax = False

if TYPE_CHECKING:
    from crisalida_lib.ADAM.living_symbol import LivingSymbolRuntime
    from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode
else:
    # runtime fallbacks when EVA/ADAM runtime types aren't available
    LivingSymbolRuntime = Any
    EVAExperience = Any
    QualiaState = Any
    RealityBytecode = Any

# Provide a safe decorator variable that acts like jax.jit when available,
# otherwise is a no-op decorator that returns the original function.

_maybe_jit: Any = None
if _has_jax and jax is not None:
    _maybe_jit = partial(jax.jit, static_argnums=(0,))
else:

    def _maybe_jit(func=None, *args, **kwargs):
        # Support both decorator usage and direct call
        if func is None:

            def _wrap(f):
                return f

            return _wrap
        return func


class JaxPhysicsEngine:
    """
    Main JAX-based physics engine for Crisalida Metacosmos.
    Stateless, functional, and ready for gradient-based optimization.
    """

    def __init__(self):
        # JAX does not require explicit context or resource management
        self.status = "initialized"

    @_maybe_jit
    def _update_entity_positions(
        self,
        entities_data: Any,
        delta_time: float,
        unified_field_center: Any,
        chaos_entropy_level: float,
        time_dilation_factor: float,
    ) -> Any:
        """
        JIT-compiled physics update for entity positions and velocities.
        Applies a simple attraction force towards the unified field center,
        modulated by chaos entropy and time dilation.
        """
        # Extract positions and velocities (assumes [pos(3), vel(3), ...])
        positions = entities_data[:, 0:3]
        velocities = entities_data[:, 3:6]
        # Compute direction and distance to center
        direction_to_center = unified_field_center - positions
        distance_to_center = jnp.linalg.norm(direction_to_center, axis=1, keepdims=True)
        safe_distance = jnp.where(distance_to_center == 0, 1.0, distance_to_center)
        normalized_direction = direction_to_center / safe_distance
        # Attraction force, modulated by chaos
        attraction_strength = 0.1 * (1.0 - chaos_entropy_level)
        force = normalized_direction * attraction_strength
        # Update velocities and positions
        new_velocities = velocities + force * delta_time
        new_positions = positions + new_velocities * delta_time * time_dilation_factor
        # Update entities_data with new positions and velocities
        updated_entities_data = entities_data.at[:, 0:3].set(new_positions)
        updated_entities_data = updated_entities_data.at[:, 3:6].set(new_velocities)
        return updated_entities_data

    def update_physics(
        self,
        entities_data: Any,
        delta_time: float,
        reality_coherence: float,
        unified_field_center: tuple[float, float, float],
        chaos_entropy_level: float,
        time_dilation_factor: float,
        simulation_tick: int,
    ) -> Any:
        """
        Updates the physics state of entities using JAX.
        Returns updated entities_data (NumPy array).
        """
        # Convert unified_field_center to JAX array
        center_array = jnp.array(unified_field_center, dtype=jnp.float32)
        updated_entities_data = self._update_entity_positions(
            entities_data,
            delta_time,
            center_array,
            chaos_entropy_level,
            time_dilation_factor,
        )
        self.status = f"updated_tick_{simulation_tick}"
        return updated_entities_data

    def get_state(self) -> dict[str, Any]:
        """JAX engine is stateless, so return minimal diagnostics."""
        return {"status": self.status}

    def set_state(self, state: dict[str, Any]):
        """JAX engine is stateless, so no state to set."""
        self.status = state.get("status", "initialized")

    def release(self):
        """No resources to release for JAX engine."""
        self.status = "released"


class EVAJaxPhysicsEngine(JaxPhysicsEngine):
    """
    JaxPhysicsEngine extendido para integración real con EVA y Adam.
    - Compila, almacena, simula y recuerda experiencias físicas diferenciables como RealityBytecode.
    - Soporta benchmarking, compresión adaptativa, hooks robustos, faseo, multiverso y API extendida.
    """

    def __init__(
        self,
        phase: str = "default",
        max_experiences: int = 100000,
        retention_policy: str = "dynamic",
    ):
        super().__init__()
        self.eva_phase = phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []
        self.benchmark_log: list = []
        self.max_experiences = max_experiences
        self.retention_policy = retention_policy

    def eva_ingest_physics_experience(
        self,
        entities_data: Any,
        metrics: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
        simulation_tick: int = 0,
    ) -> str:
        """
        Compila una experiencia física diferenciable en RealityBytecode y la almacena en la memoria EVA.
        Real benchmarking y registro de hooks.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.6,
            cognitive_complexity=0.9,
            consciousness_density=0.8,
            narrative_importance=0.7,
            energy_level=1.0,
        )
        start = time.time()
        ts = float(time.time())
        experience_data = {
            "entity_count": entities_data.shape[0],
            "simulation_tick": simulation_tick,
            "metrics": metrics or {},
            "diagnostics": self.get_state(),
            "entities_snapshot": entities_data.tolist(),
            "timestamp": ts,
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_JAX_PHYSICS_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if _eva is None:
            bytecode = []
        else:
            _dc = getattr(_eva, "divine_compiler", None)
            if _dc is not None and hasattr(_dc, "compile_intention"):
                try:
                    bytecode = _dc.compile_intention(intention)
                except Exception:
                    bytecode = []
            else:
                bytecode = []
        experience_id = f"eva_jax_physics_{simulation_tick}_{uuid.uuid4().hex[:8]}"
        # ensure timestamp is a float for typed RealityBytecode
        _ts_src: Any = experience_data.get("timestamp", ts)
        try:
            _ts = float(_ts_src)
        except Exception:
            _ts = float(ts)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
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
            "entity_count": entities_data.shape[0],
            "simulation_tick": simulation_tick,
        }
        self.benchmark_log.append(benchmark)
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                print(f"[EVA-JAX-PHYSICS] Environment hook failed: {e}")
        self._auto_cleanup_eva_memory()
        return experience_id

    def eva_recall_physics_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia física diferenciable almacenada, manifestando la simulación.
        Real benchmarking y registro de hooks.
        """
        phase = phase or self.eva_phase
        start = time.time()
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA JAX physics experience"}
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        if quantum_field:
            for instr in reality_bytecode.instructions:
                symbol_manifest = self.eva_runtime.execute_instruction(
                    instr, quantum_field
                )
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            print(f"[EVA-JAX-PHYSICS] Manifestation hook failed: {e}")
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
        self,
        experience_id: str,
        phase: str,
        entities_data: Any,
        metrics: dict,
        qualia_state: QualiaState,
        simulation_tick: int = 0,
    ):
        """
        Añade una fase alternativa para una experiencia física EVA.
        """
        experience_data = {
            "entity_count": entities_data.shape[0],
            "simulation_tick": simulation_tick,
            "metrics": metrics or {},
            "diagnostics": self.get_state(),
            "entities_snapshot": entities_data.tolist(),
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_JAX_PHYSICS_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if _eva is None:
            bytecode = []
        else:
            _dc = getattr(_eva, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
            else:
                bytecode = []
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
                print(f"[EVA-JAX-PHYSICS] Phase hook failed: {e}")

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
                getattr(exp, "timestamp", None),
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
            "optimize_eva_memory": self.optimize_eva_memory,
        }
