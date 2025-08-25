"""
deterministic_helpers.py - Utilidades Determinísticas para Simulación Crisalida + EVA
====================================================================================
Funciones para generar valores determinísticos (hashes, seeds, floats, vectores)
a partir de etiquetas o entradas simbólicas. Garantizan reproducibilidad y
consistencia en la simulación, esenciales para la generación de entidades,
asignación de propiedades y sincronización de procesos.

Extensión EVA: integración con memoria viviente, hooks de entorno, benchmarking y faseo.
Inspirado en los principios de El Verbo v7 y BB6 Enhanced.
"""

import hashlib
import struct
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

# EVA imports: keep type-only imports guarded to avoid assigning typing special forms at runtime
if TYPE_CHECKING:
    from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode
else:
    EVAExperience = Any
    QualiaState = Any
    RealityBytecode = Any


class EVADeterministicHelpers:
    """
    Extensión avanzada para EVA: permite registrar cada operación determinística como experiencia viviente,
    soporta hooks de entorno, benchmarking y gestión de memoria viviente EVA.
    """

    def __init__(self, phase: str = "default", eva_runtime: Any = None):
        self.eva_phase = phase
        self.eva_runtime = eva_runtime
        # instance stores
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list = []

    def ingest_experience(
        self,
        operation: str,
        label: str,
        result: Any,
        qualia_state: Any | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Registra la operación determinística como experiencia viviente en la memoria EVA.
        """
        if RealityBytecode is None:
            return ""
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_DETERMINISTIC_OPERATION_EXPERIENCE",
            "operation": operation,
            "label": label,
            "result": result,
            "qualia": qualia_state,
            "phase": phase,
        }
        experience_id = f"eva_det_{operation}_{hash(label)}"
        from crisalida_lib.EDEN.bytecode_generator import safe_compile_intention

        runtime = getattr(self, "eva_runtime", None)
        try:
            instructions = safe_compile_intention(runtime, intention)
        except Exception:
            instructions = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=instructions,
            qualia_state=qualia_state,
            phase=phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self.eva_experience_store[experience_id] = reality_bytecode
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                print(f"[EVA-DETERMINISTIC] Environment hook failed: {e}")
        return experience_id

    def add_environment_hook(self, hook: Callable[..., Any]):
        self._environment_hooks.append(hook)

    def set_memory_phase(self, phase: str):
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-DETERMINISTIC] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def get_eva_api(self) -> dict:
        return {
            "ingest_experience": self.ingest_experience,
            "add_environment_hook": self.add_environment_hook,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
        }


# --- Funciones determinísticas base ---


def det_hash(label: str, bits: int = 256) -> int:
    """
    Genera un hash determinístico entero a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        bits: Número de bits del hash (por defecto 256).
    Returns:
        Entero hash.
    """
    h = hashlib.sha256(label.encode()).hexdigest()
    return int(h, 16) & ((1 << bits) - 1)


def det_seed(label: str) -> int:
    """
    Genera una semilla determinística de 63 bits a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
    Returns:
        Entero semilla.
    """
    return det_hash(label, bits=63)


def det_float(label: str, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Genera un float determinístico en [min_value, max_value) a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        min_value: Valor mínimo.
        max_value: Valor máximo.
    Returns:
        Float determinístico.
    """
    h = det_hash(label)
    frac = (h % (10**10)) / (10**10)
    return min_value + (max_value - min_value) * frac + 1e-12


def det_vector(
    label: str, dim: int = 8, min_value: float = 0.0, max_value: float = 1.0
) -> np.ndarray:
    """
    Genera un vector NumPy determinístico a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        dim: Dimensión del vector.
        min_value: Valor mínimo de cada componente.
        max_value: Valor máximo de cada componente.
    Returns:
        np.ndarray de floats determinísticos.
    """
    h = hashlib.sha256(label.encode()).digest()
    arr = []
    for i in range(dim):
        start = i % len(h)
        chunk = h[start : start + 8]
        if len(chunk) < 8:
            chunk = (chunk + h)[:8]
        val = struct.unpack(">Q", chunk)[0]
        frac = (val % (10**10)) / (10**10)
        arr.append(min_value + (max_value - min_value) * frac + 1e-12)
    return np.array(arr, dtype=np.float64)


def spectral_norm_from_vector(v: np.ndarray) -> float:
    """
    Calcula la norma espectral (media de magnitudes FFT) de un vector.
    Si falla, retorna la norma euclídea.
    Args:
        v: np.ndarray de entrada.
    Returns:
        Float con la norma espectral.
    """
    try:
        mag = np.abs(np.fft.rfftn(v))
        return float(np.mean(mag) + 1e-12)
    except Exception:
        return float(np.linalg.norm(v) + 1e-12)


def det_bool(label: str, threshold: float = 0.5) -> bool:
    """
    Genera un booleano determinístico a partir de una etiqueta y un umbral.
    Args:
        label: Cadena de entrada.
        threshold: Umbral para True (0.0-1.0).
    Returns:
        Booleano determinístico.
    """
    return det_float(label) >= threshold


def det_choice(label: str, options: Sequence[Any]) -> Any:
    """
    Selecciona determinísticamente un elemento de una secuencia a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        options: Secuencia de opciones.
    Returns:
        Elemento seleccionado.
    """
    if not options:
        raise ValueError("Options must not be empty")
    idx = det_hash(label, bits=32) % len(options)
    return options[idx]


def det_color(label: str, as_rgb: bool = True) -> Any:
    """
    Genera un color determinístico a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        as_rgb: Si True, retorna (r, g, b) en [0, 1]; si False, retorna hex.
    Returns:
        Color RGB o hex.
    """
    vec = det_vector(label, dim=3, min_value=0.0, max_value=1.0)
    if as_rgb:
        return tuple(float(min(max(x, 0.0), 1.0)) for x in vec)
    else:
        return "#{:02x}{:02x}{:02x}".format(
            *(int(min(max(x, 0.0), 1.0) * 255) for x in vec)
        )


def det_gaussian(label: str, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Genera un valor gaussiano determinístico a partir de una etiqueta.
    Args:
        label: Cadena de entrada.
        mu: Media.
        sigma: Desviación estándar.
    Returns:
        Float gaussiano determinístico.
    """
    base = det_float(label)
    # Box-Muller transform
    z = (2 * base - 1) * 0.999999
    return mu + sigma * np.sqrt(-2 * np.log(1 - z**2)) * z


# --- EVA: Ejemplo de uso extendido ---
# eva_helpers = EVADeterministicHelpers(phase="default", eva_runtime=my_eva_runtime)
# seed = det_seed("entity_42")
# exp_id = eva_helpers.ingest_experience("det_seed", "entity_42", seed)
# color = det_color("archetype_hero")
# exp_id2
