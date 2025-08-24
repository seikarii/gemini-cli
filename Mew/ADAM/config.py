"""
Configuration module for Adam's consciousness system.

This module centralizes all threshold values and behavioral parameters
that define Adam's personality and decision-making patterns.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

# Defensive numeric backend pattern (numpy optional)
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class AdamConfig:
    """
    Centralized configuration for Adam's consciousness system.

    This class is intentionally permissive: callers can update parameters
    dynamically using `update_parameter` or `update_from_dict` while the
    built-in validation helpers attempt to keep values in reasonable ranges.
    """

    # === FREQUENCY REGULATION ===
    BASE_HZ: float = 10.0
    MIN_HZ: float = 2.0
    MAX_HZ: float = 60.0

    # === EXECUTIVE PFC PARAMETERS ===
    PFC_INHIBITION_LEVEL: float = 0.2
    PFC_MAX_WORKING_MEMORY: int = 7

    # === PATTERN RECOGNITION ===
    PATTERN_THRESHOLD_NEW: float = 0.75

    # === CRISIS AND AROUSAL THRESHOLDS ===
    CRISIS_AROUSAL_THRESHOLD: float = 0.8
    CRISIS_THREAT_THRESHOLD: float = 0.8
    HIGH_AROUSAL_THRESHOLD: float = 0.85
    SAFE_THRESHOLD_HIGH: float = 0.6
    SAFE_THRESHOLD_MEDIUM: float = 0.4

    # === STAGNATION AND PROGRESS ===
    STAGNATION_THRESHOLD: float = 0.1
    PROGRESS_THRESHOLD_LOW: float = 0.3
    PROGRESS_THRESHOLD_HIGH: float = 0.6

    # === OPPORTUNITY DETECTION ===
    OPPORTUNITY_THRESHOLD_HIGH: float = 0.8
    OPPORTUNITY_THRESHOLD_MEDIUM: float = 0.4

    # === ALFA MIND PARAMETERS ===
    ALFA_MIND_AUDACITY: float = 0.8
    ALFA_MIND_CREATIVITY: float = 0.9
    ALFA_MIND_TOLERANCE_RISK: float = 0.7
    ALFA_MIND_OPTIMISM: float = 0.8
    ALFA_MIND_PRIORITY_ACTION: float = 0.8
    ALFA_MIND_CAUTION: float = 0.3
    ALFA_MIND_RIGOR: float = 0.4
    ALFA_MIND_PRIORITY_ANALYSIS: float = 0.3
    ALFA_MIND_BASE_CONFIDENCE: float = 0.7

    # === OMEGA MIND PARAMETERS ===
    OMEGA_MIND_AUDACITY: float = 0.2
    OMEGA_MIND_CREATIVITY: float = 0.3
    OMEGA_MIND_TOLERANCE_RISK: float = 0.2
    OMEGA_MIND_OPTIMISM: float = 0.4
    OMEGA_MIND_PRIORITY_ACTION: float = 0.2
    OMEGA_MIND_CAUTION: float = 0.9
    OMEGA_MIND_RIGOR: float = 0.9
    OMEGA_MIND_PRIORITY_ANALYSIS: float = 0.9
    OMEGA_MIND_BASE_CONFIDENCE: float = 0.6

    # === JUDGMENT MODULE PARAMETERS ===
    JUDGMENT_BASE_LEARNING_RATE: float = 0.1
    JUDGMENT_MIN_ADJUSTMENT_THRESHOLD: float = 0.05
    JUDGMENT_MAX_ADJUSTMENT_PER_CYCLE: float = 0.2
    JUDGMENT_ANALYSIS_WINDOW_DAYS: int = 7

    # === PERFORMANCE THRESHOLDS ===
    PERFORMANCE_HIGH_QUALITY_THRESHOLD: float = 0.8
    PERFORMANCE_SUCCESS_RATE_THRESHOLD: float = 0.9
    PERFORMANCE_EXECUTION_TIME_THRESHOLD: float = 60.0

    # === HARDWARE ABSTRACTION THRESHOLDS ===
    HARDWARE_ENERGY_POOL_THRESHOLD: float = 0.2
    HARDWARE_PERFORMANCE_METRIC_THRESHOLD: float = 0.15
    HARDWARE_TEMPERATURE_REGULATION_EFFICIENCY: float = 0.8

    # === MONITORING THRESHOLDS ===
    MONITORING_MEMORY_THRESHOLD_GB: float = 0.8
    MONITORING_CPU_THRESHOLD: float = 0.85
    MONITORING_MEMORY_INCREASE_RATE_THRESHOLD: float = 0.01

    # === ALARM TIMEOUTS ===
    ALARM_TIMEOUT_DEFAULT_SECONDS: float = 30.0
    ALARM_CRISIS_TIMEOUT_SECONDS: float = 5.0
    ALARM_STAGNATION_TIMEOUT_SECONDS: float = 15.0
    ALARM_OPPORTUNITY_TIMEOUT_SECONDS: float = 10.0

    # === ADAPTATION PARAMETERS ===
    ADAPTIVE_TUNER_CYCLE_INTERVAL: int = 100
    ADAPTIVE_TUNER_HISTORY_WINDOW: int = 1000
    ADAPTIVE_TUNER_MIN_CONFIDENCE: float = 0.6
    ADAPTIVE_TUNER_ADJUSTMENT_RATE: float = 0.05

    # === BALANCE AND CONTEXT THRESHOLDS ===
    BALANCE_AUDACITY_THRESHOLD: float = 0.6
    BALANCE_CAUTION_THRESHOLD: float = 0.4
    CONTEXT_SURVIVAL_THRESHOLD: float = 0.8
    CONTEXT_CURIOSITY_THRESHOLD: float = 0.8
    CONTEXT_STRESS_THRESHOLD: float = 0.5
    CONTEXT_DOPAMINE_THRESHOLD: float = 0.7

    # === VALIDATION AND QUALITY ===
    VALIDATION_PRIORITY_THRESHOLD: int = 8
    VALIDATION_CONFIDENCE_HIGH: float = 0.85
    VALIDATION_RIGOR_THRESHOLD: float = 0.7
    VALIDATION_COVERAGE_THRESHOLD_HIGH: float = 0.8
    VALIDATION_COVERAGE_THRESHOLD_MEDIUM: float = 0.6

    # === LEARNING AND FEEDBACK ===
    FEEDBACK_SUCCESS_THRESHOLD: float = 0.7
    FEEDBACK_FAILURE_THRESHOLD: float = 0.4
    FEEDBACK_PATTERN_MIN_FREQUENCY: int = 3
    FEEDBACK_PATTERN_CONFIDENCE_THRESHOLD: float = 0.6

    def __post_init__(self) -> None:
        # Basic sanity clamps for well-known ranges
        try:
            self.BASE_HZ = float(self.BASE_HZ)
            self.MIN_HZ = float(self.MIN_HZ)
            self.MAX_HZ = float(self.MAX_HZ)
            if self.MIN_HZ <= 0 or self.MAX_HZ <= 0:
                raise ValueError("Hz thresholds must be positive")
            if self.MIN_HZ > self.MAX_HZ:
                self.MIN_HZ, self.MAX_HZ = self.MAX_HZ, self.MIN_HZ
        except Exception:
            logger.exception("Post-init validation of AdamConfig failed (non-fatal)")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict filtered of callables and private fields."""
        raw = asdict(self)
        return {
            k: v for k, v in raw.items() if not callable(v) and not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AdamConfig:
        """Create a config instance from a mapping (permissive)."""
        safe = {}
        for k, v in config_dict.items():
            if k in cls.__dataclass_fields__:
                safe[k] = v
        return cls(**safe)

    def update_parameter(self, parameter_name: str, new_value: Any) -> bool:
        """Update a single parameter with basic validation where relevant."""
        if parameter_name not in self.__dataclass_fields__:
            return False
        try:
            setattr(self, parameter_name, new_value)
            return True
        except Exception:
            logger.exception("Failed to update parameter %s", parameter_name)
            return False

    def update_from_dict(
        self, config_updates: dict[str, Any], validate: bool = True
    ) -> dict[str, bool]:
        """
        Bulk-update parameters from a dict. Returns a map of parameter -> success flag.
        When `validate` is True, performs light-range validation for common thresholds.
        """
        results: dict[str, bool] = {}
        for k, v in config_updates.items():
            if k not in self.__dataclass_fields__:
                results[k] = False
                continue
            try:
                if validate and isinstance(v, (int, float)):
                    # clamp many probability-like parameters into [0,1]
                    if k.upper().endswith(
                        ("THRESHOLD", "RATE", "FACTOR", "PROBABILITY")
                    ):
                        v = float(max(0.0, min(1.0, float(v))))
                setattr(self, k, v)
                results[k] = True
            except Exception:
                logger.exception("Failed to update %s via update_from_dict", k)
                results[k] = False
        return results

    def export_json(self) -> str:
        try:
            return json.dumps(self.to_dict(), default=str, sort_keys=True)
        except Exception:
            logger.exception("export_json failed")
            return "{}"

    @classmethod
    def import_json(cls, payload: str) -> AdamConfig | None:
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except Exception:
            logger.exception("import_json failed")
            return None


def det_float(seed: str) -> float:
    """Deterministic float in [0, 1) derived from a stable hash of `seed`."""
    try:
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        # convert hex digest to integer and normalize by max SHA1 value
        intval = int(digest, 16)
        max_int = (1 << (len(digest) * 4)) - 1
        return float(intval) / float(max_int)
    except Exception:
        # fallback simple deterministic-ish hash
        try:
            return float(abs(hash(seed)) % 1000) / 1000.0
        except Exception:
            return 0.0


@dataclass
class EVAConfig(AdamConfig):
    """
    Centralized EVA-specific configuration built on top of AdamConfig.

    Keep helper methods that integrate with EVA pipelines; they must use
    logging instead of prints and should be tolerant to failing hooks.
    """

    # === EVA MEMORY SYSTEM ===
    EVA_MEMORY_PHASE: str = "default"
    EVA_MEMORY_MAX_EXPERIENCES: int = 1_000_000
    EVA_MEMORY_RETENTION_POLICY: str = "dynamic"
    EVA_MEMORY_COMPRESSION_LEVEL: float = 0.7
    EVA_MEMORY_SIMULATION_RATE: float = 1.0
    EVA_MEMORY_MULTIVERSE_ENABLED: bool = True
    EVA_MEMORY_TIMELINE_COUNT: int = 12
    EVA_MEMORY_BENCHMARKING_ENABLED: bool = True
    EVA_MEMORY_ENVIRONMENT_HOOKS: list[Callable[..., Any]] = field(default_factory=list)
    EVA_MEMORY_VISUALIZATION_MODE: str = "hybrid"
    EVA_MEMORY_GPU_OPTIMIZATION: bool = True
    EVA_MEMORY_DISTRIBUTED_CACHE: bool = True
    EVA_MEMORY_PERSISTENCE_ENABLED: bool = True
    EVA_MEMORY_PHASE_SWITCH_TIMEOUT: float = 2.0

    # === EVA SIMULATION PARAMETERS ===
    EVA_SIMULATION_MAX_CONCURRENT: int = 128
    EVA_SIMULATION_MULTIVERSE_DEPTH: int = 8
    EVA_SIMULATION_ENTITY_LIMIT: int = 100_000
    EVA_SIMULATION_LIVING_SYMBOLS_LIMIT: int = 500_000
    EVA_SIMULATION_QUANTUMFIELD_ENABLED: bool = True
    EVA_SIMULATION_VISUALIZATION_DETAIL: str = "ultra"
    EVA_SIMULATION_PERFORMANCE_THRESHOLD: float = 0.92
    EVA_SIMULATION_ADAPTIVE_TUNING: bool = True

    # === EVA BENCHMARKING ===
    EVA_BENCHMARK_MEMORY_INTERVAL: int = 120
    EVA_BENCHMARK_SIMULATION_INTERVAL: int = 60
    EVA_BENCHMARK_EXPORT_ENABLED: bool = True
    EVA_BENCHMARK_GPU_METRICS: bool = True
    EVA_BENCHMARK_ECS_METRICS: bool = True

    # === EVA MULTIVERSO PARAMETERS ===
    EVA_MULTIVERSO_ENABLED: bool = True
    EVA_MULTIVERSO_MAX_BRANCHES: int = 24
    EVA_MULTIVERSO_BRANCH_RETENTION: int = 6
    EVA_MULTIVERSO_BRANCH_SWITCH_TIMEOUT: float = 1.5

    # === EVA ENVIRONMENT HOOKS ===
    EVA_ENVIRONMENT_HOOKS: list = field(default_factory=list)
    EVA_ENVIRONMENT_VISUALIZATION_MODE: str = "4D"
    EVA_ENVIRONMENT_EVENT_LOGGING: bool = True

    # === EVA ADVANCED PARAMETERS ===
    EVA_ADVANCED_SYMBOLIC_OPTIMIZATION: bool = True
    EVA_ADVANCED_MEMORY_COMPRESSION: bool = True
    EVA_ADVANCED_PHASEO_ENABLED: bool = True
    EVA_ADVANCED_TIMELINE_MANAGEMENT: bool = True
    EVA_ADVANCED_SIMULATION_SCALING: bool = True

    def add_environment_hook(self, hook: Callable[..., Any]) -> None:
        """Add a runtime hook (defensive)."""
        try:
            if callable(hook):
                self.EVA_ENVIRONMENT_HOOKS.append(hook)
                logger.debug("EVA environment hook added.")
            else:
                logger.warning(
                    "Attempted to add non-callable hook to EVA_ENVIRONMENT_HOOKS"
                )
        except Exception:
            logger.exception("add_environment_hook failed")

    def set_memory_phase(self, phase: str) -> None:
        """Switch EVA memory phase and notify hooks (non-fatal)."""
        old = self.EVA_MEMORY_PHASE
        self.EVA_MEMORY_PHASE = str(phase)
        logger.info("EVA memory phase changed: %s -> %s", old, self.EVA_MEMORY_PHASE)
        for hook in list(self.EVA_ENVIRONMENT_HOOKS):
            try:
                hook({"phase_changed": self.EVA_MEMORY_PHASE})
            except Exception:
                logger.exception("EVA environment hook raised during set_memory_phase")

    def get_memory_phase(self) -> str:
        return self.EVA_MEMORY_PHASE

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        return {
            "add_environment_hook": self.add_environment_hook,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
        }


DEFAULT_ADAM_CONFIG = AdamConfig()
