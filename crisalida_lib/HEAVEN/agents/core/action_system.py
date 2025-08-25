"""
Action System - Sistema de Acción Principal
==========================================

Gestiona y ejecuta las herramientas disponibles del agente.
Actúa como un despachador de acciones inteligente, con integración de priorización adaptativa,
registro de uso de herramientas, y soporte para métricas avanzadas de rendimiento.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# from crisalida_lib.symbolic_language.living_symbols import LivingSymbolRuntime
# from crisalida_lib.symbolic_language.types import (
#     EVAExperience,
#     QualiaState,
#     RealityBytecode,
# ) # Commented out due to ModuleNotFoundError

logger = logging.getLogger(__name__)


class ActionPriority(Enum):
    """Prioridades de acciones"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(Enum):
    """Estados de acciones"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Action:
    """
    Representa una acción a ejecutar.

    Attributes:
        id: Identificador único de la acción
        tool_name: Nombre de la herramienta a ejecutar
        parameters: Parámetros para la ejecución
        priority: Prioridad de la acción
        status: Estado actual de la acción
        created_at: Timestamp de creación
        started_at: Timestamp de inicio de ejecución
        completed_at: Timestamp de finalización
        result: Resultado de la ejecución
        error: Error si ocurrió alguno
        metadata: Metadatos adicionales
        callback: Función de callback para notificación de resultado
    """

    id: str
    tool_name: str
    parameters: dict[str, Any]
    priority: ActionPriority = ActionPriority.NORMAL
    status: ActionStatus = ActionStatus.PENDING
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    callback: Callable[[Any], None] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class ActionSystem:
    """
    Sistema de Acción principal para el Agente Prometeo v2.

    Gestiona la ejecución de herramientas, priorización adaptativa,
    registro de uso y métricas avanzadas de rendimiento.
    """

    def __init__(self):
        """Inicializa el Sistema de Acción"""
        self.action_queue: list[Action] = []
        self.active_actions: dict[str, Action] = {}
        self.completed_actions: list[Action] = []
        self.tool_registry: dict[str, Callable] = {}  # Registro de herramientas

        self.config = {
            "max_concurrent_actions": 3,
            "max_queue_size": 100,
            "enable_action_history": True,
            "max_history_size": 1000,
            "auto_cleanup_completed": True,
            "cleanup_interval": 3600,  # 1 hora
            "enable_priority_scheduling": True,
            "default_timeout": 30.0,  # segundos
            "adaptive_priority": True,
        }

        self.stats = {
            "total_actions_created": 0,
            "total_actions_completed": 0,
            "total_actions_failed": 0,
            "total_actions_cancelled": 0,
            "average_execution_time": 0.0,
            "tools_usage": {},
            "priority_stats": {priority.value: 0 for priority in ActionPriority},
            "last_tool_used": None,
        }

        self.is_running = False
        self.last_cleanup_time = None

        logger.info("⚡ Sistema de Acción inicializado")

    def register_tool(self, tool_name: str, tool_fn: Callable) -> None:
        """Registra una herramienta disponible para ejecución."""
        self.tool_registry[tool_name] = tool_fn
        logger.info(f"🔧 Herramienta registrada: {tool_name}")

    def create_action(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        priority: ActionPriority = ActionPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        callback: Callable[[Any], None] | None = None,
    ) -> str:
        """
        Crea una acción sin ejecutarla inmediatamente.

        Args:
            tool_name: Nombre de la herramienta
            parameters: Parámetros para la ejecución
            priority: Prioridad de la acción
            metadata: Metadatos adicionales
            callback: Función de callback para notificación de resultado

        Returns:
            ID de la acción creada
        """
        # Crear acción
        action_id = str(uuid.uuid4())
        action = Action(
            id=action_id,
            tool_name=tool_name,
            parameters=parameters or {},
            priority=priority,
            metadata=metadata or {},
            callback=callback,
        )

        # Añadir a la cola
        self.action_queue.append(action)

        # Actualizar estadísticas
        self.stats["total_actions_created"] += 1
        self.stats["priority_stats"][priority.value] += 1

        # Ordenar cola por prioridad
        if self.config["enable_priority_scheduling"]:
            self._sort_action_queue()

        # Limitar tamaño de la cola
        if len(self.action_queue) > self.config["max_queue_size"]:
            self.action_queue = self.action_queue[-self.config["max_queue_size"] :]

        logger.debug(f"⚡ Acción creada: {action_id} ({tool_name})")
        return action_id

    async def execute_next_action(self) -> str | None:
        """
        Ejecuta la siguiente acción en la cola si hay capacidad.

        Returns:
            ID de la acción ejecutada o None
        """
        if not self.is_running or not self.action_queue:
            return None
        if len(self.active_actions) >= self.config["max_concurrent_actions"]:
            logger.debug("⏳ Límite de acciones concurrentes alcanzado")
            return None

        action = self.action_queue.pop(0)
        action.status = ActionStatus.EXECUTING
        action.started_at = datetime.now().isoformat()
        self.active_actions[action.id] = action
        self.stats["last_tool_used"] = action.tool_name

        try:
            tool_fn = self.tool_registry.get(action.tool_name)
            if not tool_fn:
                raise ValueError(f"Herramienta '{action.tool_name}' no registrada")
            result = await tool_fn(**action.parameters)
            action.result = result
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.now().isoformat()
            self.stats["total_actions_completed"] += 1
            self.stats["tools_usage"].setdefault(action.tool_name, 0)
            self.stats["tools_usage"][action.tool_name] += 1
            if action.callback:
                action.callback(result)
            logger.info(f"✅ Acción ejecutada: {action.id} ({action.tool_name})")
        except Exception as e:
            action.error = str(e)
            action.status = ActionStatus.FAILED
            action.completed_at = datetime.now().isoformat()
            self.stats["total_actions_failed"] += 1
            logger.error(f"❌ Error en acción {action.id}: {e}")

        self.completed_actions.append(self.active_actions.pop(action.id))
        logger.debug(
            f"Action {action.id} status after execution: {action.status.value}"
        )
        self._update_average_execution_time(action)
        return action.id

    def cancel_action(self, action_id: str) -> bool:
        """
        Cancela una acción.

        Args:
            action_id: ID de la acción a cancelar

        Returns:
            True si la acción fue cancelada
        """
        # Buscar en cola
        for i, action in enumerate(self.action_queue):
            if action.id == action_id:
                action.status = ActionStatus.CANCELLED
                action.completed_at = datetime.now().isoformat()
                self.completed_actions.append(self.action_queue.pop(i))
                self.stats["total_actions_cancelled"] += 1
                logger.info(f"⚡ Acción cancelada: {action_id}")
                return True

        # Buscar en acciones activas
        if action_id in self.active_actions:
            action = self.active_actions[action_id]
            action.status = ActionStatus.CANCELLED
            action.completed_at = datetime.now().isoformat()
            del self.active_actions[action_id]
            self.completed_actions.append(action)
            self.stats["total_actions_cancelled"] += 1
            logger.info(f"⚡ Acción activa cancelada: {action_id}")
            return True

        logger.warning(f"⚠️ Acción no encontrada para cancelar: {action_id}")
        return False

    def get_action_status(self, action_id: str) -> dict[str, Any] | None:
        """
        Obtiene el estado de una acción.

        Args:
            action_id: ID de la acción

        Returns:
            Diccionario con estado de la acción o None
        """
        # Buscar en cola
        for action in self.action_queue:
            if action.id == action_id:
                return self._action_to_dict(action)

        # Buscar en activas
        if action_id in self.active_actions:
            return self._action_to_dict(self.active_actions[action_id])

        # Buscar en completadas
        for action in self.completed_actions:
            if action.id == action_id:
                return self._action_to_dict(action)

        return None

    def list_actions(self, status: ActionStatus | None = None) -> list[dict[str, Any]]:
        """
        Lista acciones con filtrado opcional por estado.

        Args:
            status: Estado para filtrar (None = todas)

        Returns:
            Lista de acciones en formato diccionario
        """
        actions = []

        # Acciones en cola
        if status is None or status == ActionStatus.PENDING:
            actions.extend(self.action_queue)

        # Acciones activas
        if status is None or status == ActionStatus.EXECUTING:
            actions.extend(self.active_actions.values())

        # Acciones completadas
        if status is None:
            actions.extend(self.completed_actions)
        elif status in [
            ActionStatus.COMPLETED,
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
        ]:
            actions.extend([a for a in self.completed_actions if a.status == status])

        return [self._action_to_dict(action) for action in actions]

    def start(self) -> None:
        """Inicia el sistema de acción"""
        if self.is_running:
            logger.warning("⚠️ El sistema de acción ya está en ejecución")
            return

        self.is_running = True
        logger.info("⚡ Sistema de Acción iniciado")

    def stop(self) -> None:
        """Detiene el sistema de acción"""
        if not self.is_running:
            logger.warning("⚠️ El sistema de acción no está en ejecución")
            return

        self.is_running = False

        # Cancelar todas las acciones activas
        for action_id in list(self.active_actions.keys()):
            self.cancel_action(action_id)

        logger.info("⚡ Sistema de Acción detenido")

    def _sort_action_queue(self) -> None:
        """Ordena la cola de acciones por prioridad adaptativa"""
        priority_order = {
            ActionPriority.CRITICAL: 0,
            ActionPriority.HIGH: 1,
            ActionPriority.NORMAL: 2,
            ActionPriority.LOW: 3,
        }
        self.action_queue.sort(key=lambda action: priority_order[action.priority])

    def _action_to_dict(self, action: Action) -> dict[str, Any]:
        """Convierte una acción a diccionario"""
        return {
            "id": action.id,
            "tool_name": action.tool_name,
            "parameters": action.parameters,
            "priority": action.priority.value,
            "status": action.status.value,
            "created_at": action.created_at,
            "started_at": action.started_at,
            "completed_at": action.completed_at,
            "has_result": action.result is not None,
            "has_error": action.error is not None,
            "error": action.error,
            "metadata": action.metadata,
        }

    def get_stats(self) -> dict[str, Any]:
        """Devuelve estadísticas del sistema de acción"""
        stats = self.stats.copy()

        # Calcular tasas
        total_completed = stats["total_actions_completed"]
        total_failed = stats["total_actions_failed"]
        total_processed = total_completed + total_failed

        if total_processed > 0:
            stats["success_rate"] = total_completed / total_processed
            stats["failure_rate"] = total_failed / total_processed
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Añadir estado actual
        stats["is_running"] = self.is_running
        stats["queue_size"] = len(self.action_queue)
        stats["active_actions_count"] = len(self.active_actions)
        stats["completed_actions_count"] = len(self.completed_actions)
        stats["last_tool_used"] = self.stats.get("last_tool_used")

        return stats

    def configure(self, config: dict[str, Any]) -> None:
        """Configura parámetros del sistema"""
        self.config.update(config)
        logger.info(f"⚙️ Configuración de sistema de acción actualizada: {config}")

    def reset_stats(self) -> None:
        """Reinicia las estadísticas"""
        self.stats = {
            "total_actions_created": 0,
            "total_actions_completed": 0,
            "total_actions_failed": 0,
            "total_actions_cancelled": 0,
            "average_execution_time": 0.0,
            "tools_usage": {},
            "priority_stats": {priority.value: 0 for priority in ActionPriority},
            "last_tool_used": None,
        }

        # Limpiar acciones completadas
        self.completed_actions.clear()
        logger.info("📊 Estadísticas de sistema de acción reiniciadas")

    def _update_average_execution_time(self, action: Action) -> None:
        """Actualiza la métrica de tiempo promedio de ejecución"""
        try:
            if action.started_at and action.completed_at:
                start = datetime.fromisoformat(action.started_at)
                end = datetime.fromisoformat(action.completed_at)
                elapsed = (end - start).total_seconds()
                total = (
                    self.stats["total_actions_completed"]
                    + self.stats["total_actions_failed"]
                )
                prev_avg = self.stats["average_execution_time"]
                self.stats["average_execution_time"] = (
                    (prev_avg * (total - 1) + elapsed) / total if total > 0 else elapsed
                )
        except Exception as e:
            logger.error(f"Error al actualizar tiempo promedio de ejecución: {e}")


# class EVAActionSystem(ActionSystem):
#     """
#     EVAActionSystem - Sistema de Acción perfeccionado y extendido para integración con EVA.
#     Gestiona la ejecución de herramientas, priorización adaptativa, registro de uso, ingestión/recall de experiencias de acción,
#     benchmarking, hooks de entorno y gestión avanzada de memoria viviente EVA.
#     """

#     def __init__(self, eva_config: EVAConfig = None):
#         super().__init__()
#         self.eva_config = eva_config or EVAConfig()
#         self.eva_phase = self.eva_config.EVA_MEMORY_PHASE
#         self.eva_runtime = LivingSymbolRuntime()
#         self.eva_memory_store: dict[str, RealityBytecode] = {}
#         self.eva_experience_store: dict[str, EVAExperience] = {}
#         self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
#         self._environment_hooks: list = (
#             self.eva_config.EVA_MEMORY_ENVIRONMENT_HOOKS.copy()
#         )
#         self.max_experiences = self.eva_config.EVA_MEMORY_MAX_EXPERIENCES
#         self.retention_policy = self.eva_config.EVA_MEMORY_RETENTION_POLICY
#         self.compression_level = self.eva_config.EVA_MEMORY_COMPRESSION_LEVEL
#         self.simulation_rate = self.eva_config.EVA_MEMORY_SIMULATION_RATE
#         self.multiverse_enabled = self.eva_config.EVA_MEMORY_MULTIVERSE_ENABLED
#         self.timeline_count = self.eva_config.EVA_MEMORY_TIMELINE_COUNT
#         self.benchmarking_enabled = self.eva_config.EVA_MEMORY_BENCHMARKING_ENABLED
#         self.visualization_mode = self.eva_config.EVA_MEMORY_VISUALIZATION_MODE

#     def eva_ingest_action_experience(
#         self, action: Action, qualia_state: QualiaState = None, phase: str = None
#     ) -> str:
#         """
#         Compila una experiencia de acción en RealityBytecode y la almacena en la memoria EVA.
#         """
#         import time

#         phase = phase or self.eva_phase
#         qualia_state = qualia_state or QualiaState(
#             emotional_valence=1.0 if action.status == ActionStatus.COMPLETED else 0.2,
#             cognitive_complexity=0.7,
#             consciousness_density=0.6,
#             narrative_importance=(
#                 1.0 if action.status == ActionStatus.COMPLETED else 0.5
#             ),
#             energy_level=1.0,
#         )
#         experience_id = f"eva_action_{action.id}_{int(time.time())}"
#         experience_data = {
#             "action_id": action.id,
#             "tool_name": action.tool_name,
#             "parameters": dict(action.parameters),
#             "priority": action.priority.value,
#             "status": action.status.value,
#             "created_at": action.created_at,
#             "started_at": action.started_at,
#             "completed_at": action.completed_at,
#             "result": action.result,
#             "error": action.error,
#             "metadata": dict(action.metadata),
#             "timestamp": time.time(),
#             "phase": phase,
#         }
#         intention = {
#             "intention_type": "ARCHIVE_ACTION_EXPERIENCE",
#             "experience": experience_data,
#             "qualia": qualia_state,
#             "phase": phase,
#         }
#         bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
#         reality_bytecode = RealityBytecode(
#             bytecode_id=experience_id,
#             instructions=bytecode,
#             qualia_state=qualia_state,
#             phase=phase,
#             timestamp=experience_data["timestamp"],
#         )
#         self.eva_memory_store[experience_id] = reality_bytecode
#         if phase not in self.eva_phases:
#             self.eva_phases[phase] = {}
#         self.eva_phases[phase][experience_id] = reality_bytecode
#         self.eva_experience_store[experience_id] = reality_bytecode
#         for hook in self._environment_hooks:
#             try:
#                 hook(reality_bytecode)
#             except Exception as e:
#                 logger.warning(f"[EVA-ACTION-SYSTEM] Environment hook failed: {e}")
#         return experience_id

#     def eva_recall_action_experience(self, cue: str, phase: str = None) -> dict:
#         """
#         Ejecuta el RealityBytecode de una experiencia de acción almacenada, manifestando la simulación.
#         """
#         phase = phase or self.eva_phase
#         reality_bytecode = self.eva_phases.get(phase, {}).get(
#             cue
#         ) or self.eva_memory_store.get(cue)
#         if not reality_bytecode:
#             return {"error": "No bytecode found for EVA action experience"}
#         quantum_field = getattr(self.eva_runtime, "quantum_field", None)
#         manifestations = []
#         if quantum_field:
#             for instr in reality_bytecode.instructions:
#                 symbol_manifest = self.eva_runtime.execute_instruction(
#                     instr, quantum_field
#                 )
#                 if symbol_manifest:
#                     manifestations.append(symbol_manifest)
#                     for hook in self._environment_hooks:
#                         try:
#                             hook(symbol_manifest)
#                         except Exception as e:
#                             logger.warning(
#                                 f"[EVA-ACTION-SYSTEM] Manifestation hook failed: {e}"
#                             )
#         eva_experience = EVAExperience(
#             experience_id=reality_bytecode.bytecode_id,
#             bytecode=reality_bytecode,
#             manifestations=manifestations,
#             phase=reality_bytecode.phase,
#             qualia_state=reality_bytecode.qualia_state,
#             timestamp=reality_bytecode.timestamp,
#         )
#         self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
#         return {
#             "experience_id": eva_experience.experience_id,
#             "manifestations": [m.to_dict() for m in manifestations],
#             "phase": eva_experience.phase,
#             "qualia_state": (
#                 eva_experience.qualia_state.to_dict()
#                 if hasattr(eva_experience.qualia_state, "to_dict")
#                 else {}
#             ),
#             "timestamp": eva_experience.timestamp,
#         }

#     def add_experience_phase(
#         self,
#         experience_id: str,
#         phase: str,
#         action: Action,
#         qualia_state: QualiaState = None,
#     ):
#         """
#         Añade una fase alternativa para una experiencia de acción EVA.
#         """
#         import time

#         qualia_state = qualia_state or QualiaState(
#             emotional_valence=1.0 if action.status == ActionStatus.COMPLETED else 0.2,
#             cognitive_complexity=0.7,
#             consciousness_density=0.6,
#             narrative_importance=(
#                 1.0 if action.status == ActionStatus.COMPLETED else 0.5
#             ),
#             energy_level=1.0,
#         )
#         experience_data = {
#             "action_id": action.id,
#             "tool_name": action.tool_name,
#             "parameters": dict(action.parameters),
#             "priority": action.priority.value,
#             "status": action.status.value,
#             "created_at": action.created_at,
#             "started_at": action.started_at,
#             "completed_at": action.completed_at,
#             "result": action.result,
#             "error": action.error,
#             "metadata": dict(action.metadata),
#             "timestamp": time.time(),
#             "phase": phase,
#         }
#         intention = {
#             "intention_type": "ARCHIVE_ACTION_EXPERIENCE",
#             "experience": experience_data,
#             "qualia": qualia_state,
#             "phase": phase,
#         }
#         bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
#         reality_bytecode = RealityBytecode(
#             bytecode_id=experience_id,
#             instructions=bytecode,
#             qualia_state=qualia_state,
#             phase=phase,
#             timestamp=experience_data["timestamp"],
#         )
#         if phase not in self.eva_phases:
#             self.eva_phases[phase] = {}
#         self.eva_phases[phase][experience_id] = reality_bytecode

#     def set_memory_phase(self, phase: str):
#         """Cambia la fase activa de memoria EVA."""
#         self.eva_phase = phase
#         for hook in self._environment_hooks:
#             try:
#                 hook({"phase_changed": phase})
#             except Exception as e:
#                 logger.warning(f"[EVA-ACTION-SYSTEM] Phase hook failed: {e}")

#     def get_memory_phase(self) -> str:
#         """Devuelve la fase de memoria actual."""
#         return self.eva_phase

#     def get_experience_phases(self, experience_id: str) -> list:
#         """Lista todas las fases disponibles para una experiencia de acción EVA."""
#         return [
#             phase for phase, exps in self.eva_phases.items() if experience_id in exps
#         ]

#     from typing import Callable, Any
#     def add_environment_hook(self, hook: Callable[..., Any]):
#         """Registra un hook para manifestación simbólica o eventos EVA."""
#         self._environment_hooks.append(hook)

#     def benchmark_eva_memory(self):
#         """Realiza benchmarking de la memoria EVA y reporta métricas clave."""
#         if self.benchmarking_enabled:
#             metrics = {
#                 "total_experiences": len(self.eva_memory_store),
#                 "phases": len(self.eva_phases),
#                 "hooks": len(self._environment_hooks),
#                 "compression_level": self.compression_level,
#                 "simulation_rate": self.simulation_rate,
#                 "multiverse_enabled": self.multiverse_enabled,
#                 "timeline_count": self.timeline_count,
#             }
#             logger.info(f"[EVA-ACTION-SYSTEM-BENCHMARK] {metrics}")
#             return metrics

#     def optimize_eva_memory(self):
#         """Optimiza la gestión de memoria EVA, aplicando compresión y limpieza según la política."""
#         if len(self.eva_memory_store) > self.max_experiences:
#             sorted_exps = sorted(
#                 self.eva_memory_store.items(),
#                 key=lambda x: getattr(x[1], "timestamp", 0),
#             )
#             for exp_id, _ in sorted_exps[
#                 : len(self.eva_memory_store) - self.max_experiences
#             ]:
#                 del self.eva_memory_store[exp_id]
#         # Placeholder para lógica avanzada de compresión si es necesario

#     def get_eva_api(self) -> dict:
#         return {
#             "eva_ingest_action_experience": self.eva_ingest_action_experience,
#             "eva_recall_action_experience": self.eva_recall_action_experience,
#             "add_experience_phase": self.add_experience_phase,
#             "set_memory_phase": self.set_memory_phase,
#             "get_memory_phase": self.get_memory_phase,
#             "get_experience_phases": self.get_experience_phases,
#             "add_environment_hook": self.add_environment_hook,
#             "benchmark_eva_memory": self.benchmark_eva_memory,
#         }
