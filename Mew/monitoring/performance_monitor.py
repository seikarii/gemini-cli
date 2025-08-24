import cProfile
import pstats
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode


class PerformanceMonitor:
    """
    Monitor de rendimiento en tiempo real para el GameEngine, simuladores y workers.
    Incluye:
    - Medición de FPS y timings de frame
    - Estadísticas de workers y profiling avanzado
    - Exportación de métricas y reset dinámico
    """

    def __init__(self, max_frames: int = 120, max_worker_samples: int = 60):
        self.frame_times: deque[float] = deque(maxlen=max_frames)
        self.worker_timings: dict[str, deque[float]] = {}
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.frame_start_time: float | None = None
        self.last_fps: float = 0.0
        self.last_frame_time: float = 0.0
        self.metrics: dict[str, Any] = {}

    def start_frame(self):
        """Marca el inicio de un frame."""
        self.frame_start_time = time.perf_counter()

    def end_frame(self):
        """Marca el fin de un frame y actualiza métricas."""
        if self.frame_start_time is None:
            return
        frame_time = time.perf_counter() - self.frame_start_time
        self.frame_times.append(frame_time)
        self.last_frame_time = frame_time
        self.last_fps = self.get_fps()
        self.metrics["last_frame_time"] = frame_time
        self.metrics["last_fps"] = self.last_fps

    def get_fps(self) -> float:
        """Calcula FPS promedio de los últimos frames."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_frame_time_stats(self) -> dict[str, float]:
        """Devuelve estadísticas de tiempo de frame."""
        if not self.frame_times:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "min": min(self.frame_times),
            "max": max(self.frame_times),
            "avg": sum(self.frame_times) / len(self.frame_times),
        }

    def update_worker_timing(self, worker_name: str, duration: float):
        """Actualiza el timing de un worker."""
        if worker_name not in self.worker_timings:
            self.worker_timings[worker_name] = deque(maxlen=60)
        self.worker_timings[worker_name].append(duration)
        self.metrics[f"{worker_name}_last_duration"] = duration

    def get_worker_stats(self, worker_name: str) -> dict[str, float]:
        """Devuelve estadísticas de timings para un worker."""
        timings = self.worker_timings.get(worker_name, [])
        if not timings:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "min": min(timings),
            "max": max(timings),
            "avg": sum(timings) / len(timings),
        }

    def get_all_worker_stats(self) -> dict[str, dict[str, float]]:
        """Devuelve estadísticas de todos los workers."""
        return {name: self.get_worker_stats(name) for name in self.worker_timings}

    def start_profiling(self):
        """Inicia el profiling avanzado."""
        if not self.is_profiling:
            self.is_profiling = True
            self.profiler.enable()
            print("[DEBUG] Profiling started.")

    def stop_profiling(self, output_file: str = "profile_results.prof"):
        """Detiene el profiling y exporta resultados."""
        if self.is_profiling:
            self.is_profiling = False
            self.profiler.disable()
            stats = pstats.Stats(self.profiler).sort_stats("cumulative")
            stats.dump_stats(output_file)
            print(f"[DEBUG] Profiling stopped. Results saved to {output_file}")

    def export_metrics(self) -> dict[str, Any]:
        """Exporta todas las métricas actuales."""
        metrics = {
            "fps": self.get_fps(),
            "frame_time_stats": self.get_frame_time_stats(),
            "worker_stats": self.get_all_worker_stats(),
            "last_frame_time": self.last_frame_time,
            "last_fps": self.last_fps,
        }
        metrics.update(self.metrics)
        return metrics

    def reset(self):
        """Resetea todas las métricas y timings."""
        self.frame_times.clear()
        self.worker_timings.clear()
        self.last_fps = 0.0
        self.last_frame_time = 0.0
        self.metrics.clear()
        self.frame_start_time = None
        print("[DEBUG] PerformanceMonitor reset.")

    def __str__(self):
        return f"PerformanceMonitor(fps={self.get_fps():.2f}, last_frame={self.last_frame_time:.4f}s)"

    def __repr__(self):
        return


class EVAPerformanceMonitor(PerformanceMonitor):
    """
    Monitor de rendimiento avanzado para EVA.
    Integra benchmarking de simulación de memoria viviente, ingestión/recall de experiencias de rendimiento,
    faseo, hooks de entorno y exportación de métricas para optimización GPU/ECS.
    """

    def __init__(
        self,
        max_frames: int = 120,
        max_worker_samples: int = 60,
        eva_phase: str = "default",
    ):
        super().__init__(max_frames, max_worker_samples)
        self.eva_phase = eva_phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []
        self._gpu_enabled: bool = False
        self._ecs_components: dict[str, Any] = {}

    def enable_gpu_optimization(self, enable: bool = True):
        """Activa la optimización GPU para benchmarking masivo."""
        self._gpu_enabled = enable
        if enable and hasattr(self.eva_runtime, "gpu_physics_engine"):
            self.eva_runtime.gpu_physics_engine.enable_benchmark_mode()

    def register_ecs_component(self, name: str, component: Any):
        """Registra un componente ECS para benchmarking orientado a datos."""
        self._ecs_components[name] = component

    def eva_ingest_performance_experience(
        self, metrics: dict, qualia_state: QualiaState = None, phase: str = None
    ) -> str:
        """
        Compila una experiencia de benchmarking en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.4,
            cognitive_complexity=0.9,
            consciousness_density=0.5,
            narrative_importance=0.3,
            energy_level=1.0,
        )
        experience_data = {
            "metrics": metrics,
            "timestamp": time.time(),
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
        }
        intention = {
            "intention_type": "ARCHIVE_PERFORMANCE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = f"eva_perf_{hash(str(experience_data))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_performance_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de benchmarking almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA performance experience"}
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        start = time.time()
        if quantum_field:
            for instr in reality_bytecode.instructions:
                if self._gpu_enabled and hasattr(
                    self.eva_runtime, "gpu_physics_engine"
                ):
                    symbol_manifest = (
                        self.eva_runtime.gpu_physics_engine.execute_instruction(
                            instr, quantum_field
                        )
                    )
                else:
                    symbol_manifest = self.eva_runtime.execute_instruction(
                        instr, quantum_field
                    )
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            print(f"[EVA-PERF] Environment hook failed: {e}")
        end = time.time()
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        if self._gpu_enabled:
            eva_experience.metadata = {"gpu_recall_time": end - start}
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
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
            "benchmark": (
                eva_experience.metadata if hasattr(eva_experience, "metadata") else {}
            ),
        }

    def add_experience_phase(
        self, experience_id: str, phase: str, metrics: dict, qualia_state: QualiaState
    ):
        """
        Añade una fase alternativa para una experiencia de benchmarking EVA.
        """
        experience_data = {
            "metrics": metrics,
            "timestamp": time.time(),
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
        }
        intention = {
            "intention_type": "ARCHIVE_PERFORMANCE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data["timestamp"],
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-PERF] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de benchmarking EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_performance_experience": self.eva_ingest_performance_experience,
            "eva_recall_performance_experience": self.eva_recall_performance_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "enable_gpu_optimization": self.enable_gpu_optimization,
            "register_ecs_component": self.register_ecs_component,
        }
