import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from crisalida_lib.ADAM.config import DEFAULT_ADAM_CONFIG, AdamConfig
from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode

logger = logging.getLogger(__name__)


class PredictiveMonitor:
    """
    PredictiveMonitor - Monitorización Predictiva Avanzada para Crisalida
    =====================================================================
    Monitorea métricas clave de la simulación (memoria, entidades, CPU) y genera alertas predictivas.
    Optimizado para integración con PerformanceMonitor y sistemas de diagnóstico.
    """

    def __init__(
        self,
        config: AdamConfig = None,
        window_size: int = 120,
        memory_threshold_gb: float = None,
        entity_threshold: int | None = None,
        cpu_threshold: float = None,
    ) -> None:
        """
        Inicializa el monitor predictivo.

        Args:
            config: Configuration instance to use for thresholds
            window_size: Número de puntos para análisis de tendencias.
            memory_threshold_gb: Umbral de memoria en GB para alerta crítica (si None, se usa configuración).
            entity_threshold: Umbral de entidades para alerta.
            cpu_threshold: Umbral de uso de CPU (0.0-1.0) para alerta (si None, se usa configuración).
        """
        self.config = config or DEFAULT_ADAM_CONFIG
        self.memory_usage_history: deque[tuple[float, float]] = deque(
            maxlen=window_size
        )
        self.entity_count_history: deque[tuple[float, int]] = deque(maxlen=window_size)
        self.cpu_usage_history: deque[tuple[float, float]] = deque(maxlen=window_size)
        self.memory_threshold_gb: float = (
            memory_threshold_gb or self.config.MONITORING_MEMORY_THRESHOLD_GB
        )
        self.entity_threshold: int | None = entity_threshold
        self.cpu_threshold: float = (
            cpu_threshold or self.config.MONITORING_CPU_THRESHOLD
        )

    def collect_metrics(
        self,
        current_memory_gb: float | None = None,
        current_entity_count: int | None = None,
        current_cpu: float | None = None,
    ) -> None:
        """
        Recoge métricas actuales de la simulación.

        Args:
            current_memory_gb: Uso actual de memoria en GB (si None, se detecta automáticamente).
            current_entity_count: Número actual de entidades.
            current_cpu: Uso actual de CPU (0.0-1.0, si None, se detecta automáticamente).
        """
        timestamp = time.time()
        if current_memory_gb is None:
            process = psutil.Process()
            current_memory_gb = process.memory_info().rss / (1024**3)
        if current_cpu is None:
            process = psutil.Process()
            current_cpu = process.cpu_percent(interval=0.05) / 100.0
        if current_memory_gb is not None:
            self.memory_usage_history.append((timestamp, current_memory_gb))
        if current_entity_count is not None:
            self.entity_count_history.append((timestamp, current_entity_count))
        if current_cpu is not None:
            self.cpu_usage_history.append((timestamp, current_cpu))

    def analyze_and_alert(self) -> dict[str, Any]:
        """
        Analiza las métricas recogidas y genera alertas predictivas.
        Returns:
            Diccionario con alertas generadas.
        """
        alerts = {}
        mem_alert = self._analyze_memory_usage()
        if mem_alert:
            alerts["memory"] = mem_alert
        entity_alert = self._analyze_entity_count()
        if entity_alert:
            alerts["entity"] = entity_alert
        cpu_alert = self._analyze_cpu_usage()
        if cpu_alert:
            alerts["cpu"] = cpu_alert
        return alerts

    def _analyze_memory_usage(self) -> str | None:
        """Analiza el historial de uso de memoria para tendencias y alertas."""
        if len(self.memory_usage_history) < 2:
            return None

        first_time, first_mem = self.memory_usage_history[0]
        last_time, last_mem = self.memory_usage_history[-1]

        if last_time - first_time > 0:
            memory_increase_rate = (last_mem - first_mem) / (last_time - first_time)
            if (
                memory_increase_rate
                > self.config.MONITORING_MEMORY_INCREASE_RATE_THRESHOLD
            ):
                logger.warning(
                    f"[PREDICTIVE ALERT] Memory usage increasing at {memory_increase_rate:.4f} GB/sec."
                )
                return f"Memory usage increasing at {memory_increase_rate:.4f} GB/sec."

        if last_mem > self.memory_threshold_gb:
            logger.critical(
                f"[PREDICTIVE ALERT] High memory usage: {last_mem:.2f} GB, exceeding threshold of {self.memory_threshold_gb:.2f} GB."
            )
            return f"High memory usage: {last_mem:.2f} GB, exceeding threshold."

        return None

    def _analyze_entity_count(self) -> str | None:
        """Analiza el historial de entidades para tendencias y alertas."""
        if len(self.entity_count_history) < 2:
            return None

        first_time, first_count = self.entity_count_history[0]
        last_time, last_count = self.entity_count_history[-1]

        if last_time - first_time > 0:
            entity_increase_rate = (last_count - first_count) / (last_time - first_time)
            if entity_increase_rate > 0.1:
                logger.warning(
                    f"[PREDICTIVE ALERT] Entity count increasing at {entity_increase_rate:.2f} entities/sec."
                )
                return f"Entity count increasing at {entity_increase_rate:.2f} entities/sec."

        if self.entity_threshold and last_count > self.entity_threshold:
            logger.critical(
                f"[PREDICTIVE ALERT] Entity count {last_count} exceeds threshold {self.entity_threshold}."
            )
            return f"Entity count {last_count} exceeds threshold."

        return None

    def _analyze_cpu_usage(self) -> str | None:
        """Analiza el historial de uso de CPU para tendencias y alertas."""
        if len(self.cpu_usage_history) < 2:
            return None

        first_time, first_cpu = self.cpu_usage_history[0]
        last_time, last_cpu = self.cpu_usage_history[-1]

        if last_time - first_time > 0:
            cpu_increase_rate = (last_cpu - first_cpu) / (last_time - first_time)
            if cpu_increase_rate > 0.01:
                logger.warning(
                    f"[PREDICTIVE ALERT] CPU usage increasing at {cpu_increase_rate:.4f} per sec."
                )
                return f"CPU usage increasing at {cpu_increase_rate:.4f} per sec."

        if last_cpu > self.cpu_threshold:
            logger.critical(
                f"[PREDICTIVE ALERT] High CPU usage: {last_cpu:.2f}, exceeding threshold of {self.cpu_threshold:.2f}."
            )
            return f"High CPU usage: {last_cpu:.2f}, exceeding threshold."

        return None

    def get_metrics_summary(self) -> dict[str, Any]:
        """Devuelve un resumen de las métricas actuales."""
        summary = {}
        if self.memory_usage_history:
            summary["memory_gb"] = self.memory_usage_history[-1][1]
        if self.entity_count_history:
            summary["entity_count"] = self.entity_count_history[-1][1]
        if self.cpu_usage_history:
            summary["cpu"] = self.cpu_usage_history[-1][1]
        return summary

    def reset(self) -> None:
        """Resetea los historiales de métricas."""
        self.memory_usage_history.clear()
        self.entity_count_history.clear()
        self.cpu_usage_history.clear()


class EVAPredictiveMonitor(PredictiveMonitor):
    """
    EVAPredictiveMonitor - Monitorización Predictiva Avanzada para EVA.
    Integra memoria viviente, ingestión/recall de experiencias de monitoreo, faseo, hooks de entorno y benchmarking.
    Permite registrar alertas y métricas como experiencias EVA y simular su manifestación en QuantumField.
    """

    def __init__(
        self,
        config: AdamConfig = None,
        window_size: int = 120,
        memory_threshold_gb: float = None,
        entity_threshold: int | None = None,
        cpu_threshold: float = None,
        eva_phase: str = "default",
    ) -> None:
        super().__init__(
            config, window_size, memory_threshold_gb, entity_threshold, cpu_threshold
        )
        self.eva_phase = eva_phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []

    def eva_ingest_monitoring_experience(
        self,
        metrics: dict,
        alerts: dict,
        qualia_state: QualiaState = None,
        phase: str = None,
    ) -> str:
        """
        Compila una experiencia de monitoreo predictivo en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.5,
            cognitive_complexity=0.8,
            consciousness_density=0.7,
            narrative_importance=0.6,
            energy_level=1.0,
        )
        experience_data = {
            "metrics": metrics,
            "alerts": alerts,
            "timestamp": time.time(),
        }
        intention = {
            "intention_type": "ARCHIVE_MONITORING_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = f"eva_monitor_{hash(str(experience_data))}"
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

    def eva_recall_monitoring_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de monitoreo almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA monitoring experience"}
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
                            logger.warning(
                                f"[EVA-MONITOR] Environment hook failed: {e}"
                            )
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
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        metrics: dict,
        alerts: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa para una experiencia de monitoreo EVA.
        """
        experience_data = {
            "metrics": metrics,
            "alerts": alerts,
            "timestamp": time.time(),
        }
        intention = {
            "intention_type": "ARCHIVE_MONITORING_EXPERIENCE",
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
                logger.warning(f"[EVA-MONITOR] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de monitoreo EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_monitoring_experience": self.eva_ingest_monitoring_experience,
            "eva_recall_monitoring_experience": self.eva_recall_monitoring_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
