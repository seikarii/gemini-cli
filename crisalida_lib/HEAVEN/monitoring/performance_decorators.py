"""
performance_decorators.py - Decoradores de Medición y Optimización de Rendimiento
=================================================================================
Incluye decoradores para medir tiempo de ejecución, uso de CPU, memoria y profiling avanzado.
Optimizado para integración con sistemas de monitorización, debugging y análisis de rendimiento Crisalida.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable

import psutil

logger = logging.getLogger(__name__)


def time_execution(func: Callable) -> Callable:
    """
    Decorador que mide el tiempo de ejecución de una función y lo registra.
    Compatible con funciones síncronas y asíncronas.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.info(
                f"[PERF] {func.__name__} ejecutada en {end_time - start_time:.4f}s."
            )
            return result

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(
                f"[PERF] {func.__name__} ejecutada en {end_time - start_time:.4f}s."
            )
            return result

        return sync_wrapper


def profile_resources(cache_duration: float = 1.0) -> Callable:
    """
    Decorador que mide uso de CPU y memoria durante la ejecución de la función.
    Los resultados de la medición se cachean para no sobrecargar el sistema.

    Args:
        cache_duration (float): Duración en segundos para mantener el resultado en caché.
                                La medición solo se ejecutará si ha pasado este tiempo.
    """
    def decorator(func: Callable) -> Callable:
        _cache = {'last_log_time': 0}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            current_time = time.time()
            if current_time - _cache['last_log_time'] > cache_duration:
                try:
                    process = psutil.Process()
                    process.cpu_percent(interval=None)
                    time.sleep(0.1)
                    cpu_usage = process.cpu_percent(interval=None)
                    mem_usage = process.memory_info().rss / (1024 * 1024)
                    logger.info(
                        f"[RES] {func.__name__}: "
                        f"CPU Usage={cpu_usage:.2f}%, MEM Usage={mem_usage:.2f}MB"
                    )
                    _cache['last_log_time'] = current_time
                except psutil.Error as e:
                    logger.warning(f"No se pudo medir los recursos para {func.__name__}: {e}")
            return result
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                current_time = time.time()
                if current_time - _cache['last_log_time'] > cache_duration:
                    try:
                        process = psutil.Process()
                        cpu_after = process.cpu_percent(interval=None)
                        mem_after = process.memory_info().rss
                        logger.info(
                            f"[RES] {func.__name__}: "
                            f"CPU={cpu_after:.2f}%, MEM={mem_after / 1024 / 1024:.2f}MB"
                        )
                        _cache['last_log_time'] = current_time
                    except psutil.Error as e:
                        logger.warning(f"No se pudo medir los recursos para {func.__name__}: {e}")
                return result
            return async_wrapper
        return wrapper
    return decorator


def log_exceptions(func: Callable) -> Callable:
    """
    Decorador que registra excepciones y errores durante la ejecución de la función.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[EXC] {func.__name__} falló: {e}", exc_info=True)
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[EXC] {func.__name__} falló: {e}", exc_info=True)
                raise

        return sync_wrapper


def eva_benchmark(func: Callable) -> Callable:
    """
    Decorador EVA: mide tiempo, uso de CPU, memoria y registra metadata de benchmarking en la memoria viviente EVA.
    Compatible con funciones síncronas y asíncronas.
    """
    from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
    from crisalida_lib.EVA.types import QualiaState, RealityBytecode

    eva_runtime = getattr(func, "_eva_runtime", None) or LivingSymbolRuntime()
    eva_phase = getattr(func, "_eva_phase", "default")

    def _record_eva_benchmark(func_name, duration, cpu_delta, mem_delta):
        qualia_state = QualiaState(
            emotional_valence=0.4,
            cognitive_complexity=0.9,
            consciousness_density=0.5,
            narrative_importance=0.3,
            energy_level=1.0,
        )
        experience_data = {
            "func_name": func_name,
            "duration": duration,
            "cpu_delta": cpu_delta,
            "mem_delta": mem_delta,
            "timestamp": time.time(),
        }
        intention = {
            "intention_type": "ARCHIVE_PERFORMANCE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": eva_phase,
        }
        bytecode = eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = f"eva_perf_{func_name}_{int(experience_data['timestamp'])}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=eva_phase,
            timestamp=experience_data["timestamp"],
        )
        eva_runtime.eva_memory_store[experience_id] = reality_bytecode
        logger.info(f"[EVA-BENCH] Benchmark registrado en EVA: {experience_id}")

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            process = psutil.Process()
            cpu_before = process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"[EVA-BENCH][EXC] {func.__name__} falló: {e}", exc_info=True
                )
                raise
            end_time = time.time()
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss
            duration = end_time - start_time
            cpu_delta = cpu_after - cpu_before
            mem_delta = mem_after - mem_before
            logger.info(
                f"[EVA-BENCH] {func.__name__}: tiempo={duration:.4f}s, CPU={cpu_delta:.2f}%, MEM={mem_delta} bytes"
            )
            _record_eva_benchmark(func.__name__, duration, cpu_delta, mem_delta)
            return result

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process()
            cpu_before = process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"[EVA-BENCH][EXC] {func.__name__} falló: {e}", exc_info=True
                )
                raise
            end_time = time.time()
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss
            duration = end_time - start_time
            cpu_delta = cpu_after - cpu_before
            mem_delta = mem_after - mem_before
            logger.info(
                f"[EVA-BENCH] {func.__name__}: tiempo={duration:.4f}s, CPU={cpu_delta:.2f}%, MEM={mem_delta} bytes"
            )
            _record_eva_benchmark(func.__name__, duration, cpu_delta, mem_delta)
            return result

        return sync_wrapper


def adaptive_tuning(func: Callable) -> Callable:
    """
    Decorador EVA: ajusta dinámicamente la complejidad computacional/visual según el rendimiento medido.
    Reduce la carga si el tiempo de ejecución supera un umbral.
    """
    threshold_seconds = getattr(func, "_eva_adaptive_threshold", 0.05)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        if duration > threshold_seconds:
            logger.warning(
                f"[EVA-ADAPTIVE] {func.__name__}: tiempo={duration:.4f}s supera el umbral {threshold_seconds}s. Activando tuning adaptativo."
            )
            # Aquí se podría reducir la complejidad visual/computacional, ej: bajar LOD, simplificar simulación, etc.
            # Si el objeto tiene método 'reduce_complexity', invócalo.
            if hasattr(result, "reduce_complexity"):
                result.reduce_complexity()
        return result

    return wrapper
