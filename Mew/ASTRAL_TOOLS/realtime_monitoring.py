#!/usr/bin/env python3
"""
RealtimeMonitoringTool - Monitorización en Tiempo Real y Feedback Continuo
==========================================================================
Permite a Gemini suscribirse a eventos del sistema (cambios en archivos, logs, procesos)
y reaccionar en tiempo real, sin polling manual.

Características:
- Monitorización eficiente de archivos, logs y procesos (con psutil)
- Suscripción a eventos: creado, modificado, eliminado, movido, rotación de logs, inicio/fin de procesos
- Almacenamiento y consulta de eventos con límite configurable
- API para listar, detener y limpiar monitores activos
- Integración con el sistema de herramientas Crisalida
- Demo interactiva para pruebas

Requiere: watchdog, psutil (para procesos), threading, asyncio
"""

import logging
import os
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any

from pydantic import BaseModel, Field, field_validator
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class MonitoringSchema(BaseModel):
    """Parámetros para la herramienta de monitorización en tiempo real"""

    action: str = Field(
        ...,
        description="Acción: 'start_file_monitor', 'start_log_monitor', 'start_process_monitor', 'stop_monitor', 'get_events', 'clear_events', 'list_monitors'",
    )
    path: str | None = Field(None, description="Ruta a monitorizar (archivos)")
    log_file: str | None = Field(None, description="Archivo de log a monitorizar")
    process_name: str | None = Field(
        None, description="Nombre de proceso a monitorizar"
    )
    monitor_id: str | None = Field(None, description="ID único del monitor")
    event_types: list[str] = Field(
        default=["all"],
        description="Tipos de eventos: 'created', 'modified', 'deleted', 'moved', 'all'",
    )
    recursive: bool = Field(default=True, description="Monitorizar subdirectorios")
    max_events: int = Field(default=100, description="Máximo de eventos en memoria")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        valid_actions = [
            "start_file_monitor",
            "start_log_monitor",
            "start_process_monitor",
            "stop_monitor",
            "get_events",
            "clear_events",
            "list_monitors",
        ]
        if v not in valid_actions:
            raise ValueError(f"Action must be one of: {valid_actions}")
        return v

    @field_validator("event_types")
    @classmethod
    def validate_event_types(cls, v):
        valid_types = ["created", "modified", "deleted", "moved", "all"]
        for event_type in v:
            if event_type not in valid_types:
                raise ValueError(f"Event type must be one of: {valid_types}")
        return v


class MonitorEvent(BaseModel):
    """Evento capturado por la monitorización"""

    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str = Field(..., description="Tipo de evento")
    source_path: str = Field(..., description="Ruta origen")
    dest_path: str | None = Field(None, description="Ruta destino (si aplica)")
    monitor_id: str = Field(..., description="ID del monitor")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadatos adicionales"
    )


class FileEventHandler(FileSystemEventHandler):
    """Handler personalizado para eventos de archivos"""

    def __init__(self, monitor_id: str, event_queue: Queue, event_types: set[str]):
        super().__init__()
        self.monitor_id = monitor_id
        self.event_queue = event_queue
        self.event_types = event_types

    def _should_process_event(self, event_type: str) -> bool:
        return "all" in self.event_types or event_type in self.event_types

    def on_created(self, event: FileSystemEvent):
        if self._should_process_event("created"):
            monitor_event = MonitorEvent(
                event_type="created",
                source_path=str(event.src_path),
                dest_path=None,
                monitor_id=self.monitor_id,
                metadata={"is_directory": event.is_directory},
            )
            self.event_queue.put(monitor_event)

    def on_modified(self, event: FileSystemEvent):
        if self._should_process_event("modified"):
            monitor_event = MonitorEvent(
                event_type="modified",
                source_path=str(event.src_path),
                dest_path=None,
                monitor_id=self.monitor_id,
                metadata={"is_directory": event.is_directory},
            )
            self.event_queue.put(monitor_event)

    def on_deleted(self, event: FileSystemEvent):
        if self._should_process_event("deleted"):
            monitor_event = MonitorEvent(
                event_type="deleted",
                source_path=str(event.src_path),
                dest_path=None,
                monitor_id=self.monitor_id,
                metadata={"is_directory": event.is_directory},
            )
            self.event_queue.put(monitor_event)

    def on_moved(self, event: FileSystemEvent):
        if self._should_process_event("moved"):
            monitor_event = MonitorEvent(
                event_type="moved",
                source_path=str(event.src_path),
                dest_path=(
                    str(getattr(event, "dest_path", None))
                    if hasattr(event, "dest_path")
                    else None
                ),
                monitor_id=self.monitor_id,
                metadata={"is_directory": event.is_directory},
            )
            self.event_queue.put(monitor_event)


class LogFileMonitor:
    """Monitor log files for new entries"""

    def __init__(self, monitor_id: str, log_file: str, event_queue: Queue):
        self.monitor_id = monitor_id
        self.log_file = log_file
        self.event_queue = event_queue
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_position = 0

    def start(self):
        """Start monitoring the log file"""
        if self.running:
            return

        self.running = True
        # Get initial file size
        if os.path.exists(self.log_file):
            self.last_position = os.path.getsize(self.log_file)

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring the log file"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                if os.path.exists(self.log_file):
                    current_size = os.path.getsize(self.log_file)
                    if current_size > self.last_position:
                        # Read new content
                        with open(
                            self.log_file, encoding="utf-8", errors="ignore"
                        ) as f:
                            f.seek(self.last_position)
                            new_content = f.read()

                        if new_content.strip():
                            monitor_event = MonitorEvent(
                                event_type="log_entry",
                                source_path=self.log_file,
                                monitor_id=self.monitor_id,
                                metadata={
                                    "new_content": new_content,
                                    "lines": len(new_content.split("\n")) - 1,
                                    "bytes": len(new_content),
                                },
                            )
                            self.event_queue.put(monitor_event)

                        self.last_position = current_size
                    elif current_size < self.last_position:
                        # File was truncated or rotated
                        self.last_position = 0
                        monitor_event = MonitorEvent(
                            event_type="log_rotated",
                            source_path=self.log_file,
                            monitor_id=self.monitor_id,
                            metadata={"message": "Log file was truncated or rotated"},
                        )
                        self.event_queue.put(monitor_event)

                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                logger.error(f"Error in log monitor loop: {e}")
                time.sleep(1.0)


class ProcessMonitor:
    """Monitor system processes"""

    def __init__(self, monitor_id: str, process_name: str, event_queue: Queue):
        self.monitor_id = monitor_id
        self.process_name = process_name
        self.event_queue = event_queue
        self.running = False
        self.thread: threading.Thread | None = None
        self.known_pids: set[int] = set()

    def start(self):
        """Start monitoring processes"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring processes"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _get_process_pids(self) -> set[int]:
        """Get PIDs of processes matching the name"""
        try:
            import psutil

            pids = set()
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if self.process_name.lower() in proc.info["name"].lower():
                        pids.add(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return pids
        except ImportError:
            # Fallback to basic method if psutil not available
            return set()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                current_pids = self._get_process_pids()

                # Check for new processes
                new_pids = current_pids - self.known_pids
                for pid in new_pids:
                    monitor_event = MonitorEvent(
                        event_type="process_started",
                        source_path=f"process:{self.process_name}",
                        monitor_id=self.monitor_id,
                        metadata={"pid": pid, "process_name": self.process_name},
                    )
                    self.event_queue.put(monitor_event)

                # Check for terminated processes
                terminated_pids = self.known_pids - current_pids
                for pid in terminated_pids:
                    monitor_event = MonitorEvent(
                        event_type="process_terminated",
                        source_path=f"process:{self.process_name}",
                        monitor_id=self.monitor_id,
                        metadata={"pid": pid, "process_name": self.process_name},
                    )
                    self.event_queue.put(monitor_event)

                self.known_pids = current_pids
                time.sleep(2.0)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error in process monitor loop: {e}")
                time.sleep(5.0)


class RealtimeMonitoringTool(BaseTool):
    """Real-time monitoring and continuous feedback tool"""

    def __init__(self):
        super().__init__()
        self.observers: dict[str, Observer] = {}
        self.log_monitors: dict[str, LogFileMonitor] = {}
        self.process_monitors: dict[str, ProcessMonitor] = {}
        self.event_queue: Queue[MonitorEvent] = Queue()
        self.events: list[MonitorEvent] = []
        self.max_events = 1000

    def _get_name(self) -> str:
        return "realtime_monitoring"

    def _get_description(self) -> str:
        return "Monitor files, logs, and processes in real-time for continuous feedback"

    def _get_category(self) -> str:
        return "monitoring"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return MonitoringSchema

    async def execute(self, **kwargs) -> ToolCallResult:
        """Execute monitoring operation"""
        start_time = datetime.now()

        try:
            action = kwargs.get("action")

            if action == "start_file_monitor":
                return await self._start_file_monitor(**kwargs)
            elif action == "start_log_monitor":
                return await self._start_log_monitor(**kwargs)
            elif action == "start_process_monitor":
                return await self._start_process_monitor(**kwargs)
            elif action == "stop_monitor":
                return await self._stop_monitor(**kwargs)
            elif action == "get_events":
                return await self._get_events(**kwargs)
            elif action == "clear_events":
                return await self._clear_events(**kwargs)
            elif action == "list_monitors":
                return await self._list_monitors(**kwargs)
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Real-time monitoring tool error: {e}")
            return ToolCallResult(
                command=f"realtime_monitoring({action})",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def _start_file_monitor(self, **kwargs) -> ToolCallResult:
        """Start monitoring file system changes"""
        path = kwargs.get("path")
        monitor_id = kwargs.get("monitor_id", f"file_monitor_{len(self.observers)}")
        event_types = set(kwargs.get("event_types", ["all"]))
        recursive = kwargs.get("recursive", True)

        if not path:
            raise ValueError("Path is required for file monitoring")

        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")

        if monitor_id in self.observers:
            raise ValueError(f"Monitor with ID '{monitor_id}' already exists")

        # Create observer
        observer = Observer()
        event_handler = FileEventHandler(monitor_id, self.event_queue, event_types)
        observer.schedule(event_handler, path, recursive=recursive)
        observer.start()

        self.observers[monitor_id] = observer

        output = f"Started file monitor '{monitor_id}' for path: {path}"
        return ToolCallResult(
            command=f"start_file_monitor({monitor_id})",
            success=True,
            output=output,
            execution_time=0.1,
            metadata={"monitor_id": monitor_id, "path": path, "recursive": recursive},
            error_message=None,
        )

    async def _start_log_monitor(self, **kwargs) -> ToolCallResult:
        """Start monitoring log file changes"""
        log_file = kwargs.get("log_file")
        monitor_id = kwargs.get("monitor_id", f"log_monitor_{len(self.log_monitors)}")

        if not log_file:
            raise ValueError("Log file path is required for log monitoring")

        if monitor_id in self.log_monitors:
            raise ValueError(f"Log monitor with ID '{monitor_id}' already exists")

        # Create log monitor
        log_monitor = LogFileMonitor(monitor_id, log_file, self.event_queue)
        log_monitor.start()

        self.log_monitors[monitor_id] = log_monitor

        output = f"Started log monitor '{monitor_id}' for file: {log_file}"
        return ToolCallResult(
            command=f"start_log_monitor({monitor_id})",
            success=True,
            output=output,
            execution_time=0.1,
            metadata={"monitor_id": monitor_id, "log_file": log_file},
            error_message=None,
        )

    async def _start_process_monitor(self, **kwargs) -> ToolCallResult:
        """Start monitoring process changes"""
        process_name = kwargs.get("process_name")
        monitor_id = kwargs.get(
            "monitor_id", f"process_monitor_{len(self.process_monitors)}"
        )

        if not process_name:
            raise ValueError("Process name is required for process monitoring")

        if monitor_id in self.process_monitors:
            raise ValueError(f"Process monitor with ID '{monitor_id}' already exists")

        # Create process monitor
        process_monitor = ProcessMonitor(monitor_id, process_name, self.event_queue)
        process_monitor.start()

        self.process_monitors[monitor_id] = process_monitor

        output = f"Started process monitor '{monitor_id}' for process: {process_name}"
        return ToolCallResult(
            command=f"start_process_monitor({monitor_id})",
            success=True,
            output=output,
            execution_time=0.1,
            metadata={"monitor_id": monitor_id, "process_name": process_name},
            error_message=None,
        )

    async def _stop_monitor(self, **kwargs) -> ToolCallResult:
        """Stop a specific monitor"""
        monitor_id = kwargs.get("monitor_id")

        if not monitor_id:
            raise ValueError("Monitor ID is required")

        stopped = False

        # Check file monitors
        if monitor_id in self.observers:
            observer = self.observers[monitor_id]
            if isinstance(observer, Observer):  # type: ignore[arg-type]
                observer.stop()  # type: ignore[attr-defined]
            del self.observers[monitor_id]
            stopped = True

        # Check log monitors
        if monitor_id in self.log_monitors:
            self.log_monitors[monitor_id].stop()
            del self.log_monitors[monitor_id]
            stopped = True

        # Check process monitors
        if monitor_id in self.process_monitors:
            self.process_monitors[monitor_id].stop()
            del self.process_monitors[monitor_id]
            stopped = True

        if not stopped:
            raise ValueError(f"Monitor with ID '{monitor_id}' not found")

        output = f"Stopped monitor: {monitor_id}"
        return ToolCallResult(
            command=f"stop_monitor({monitor_id})",
            success=True,
            output=output,
            execution_time=0.1,
            error_message=None,
        )

    async def _get_events(self, **kwargs) -> ToolCallResult:
        """Get recent monitoring events"""
        max_events = kwargs.get("max_events", 50)

        # Collect new events from queue
        new_events = []
        try:
            while True:
                event = self.event_queue.get_nowait()
                new_events.append(event)
        except Empty:
            pass

        # Add to main events list
        self.events.extend(new_events)

        # Limit events in memory
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

        # Return recent events
        recent_events = self.events[-max_events:] if self.events else []

        events_data = []
        for event in recent_events:
            events_data.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "source_path": event.source_path,
                    "dest_path": event.dest_path,
                    "monitor_id": event.monitor_id,
                    "metadata": event.metadata,
                }
            )

        output = f"Retrieved {len(events_data)} monitoring events"
        return ToolCallResult(
            command="get_events",
            success=True,
            output=output,
            execution_time=0.1,
            metadata={"events": events_data, "total_events": len(self.events)},
            error_message=None,
        )

    async def _clear_events(self, **kwargs) -> ToolCallResult:
        """Clear stored events"""
        count = len(self.events)
        self.events.clear()

        # Also clear queue
        try:
            while True:
                self.event_queue.get_nowait()
        except Empty:
            pass

        output = f"Cleared {count} stored events"
        return ToolCallResult(
            command="clear_events",
            success=True,
            output=output,
            execution_time=0.1,
            error_message=None,
        )

    async def _list_monitors(self, **kwargs) -> ToolCallResult:
        """List active monitors"""
        monitors = {
            "file_monitors": list(self.observers.keys()),
            "log_monitors": list(self.log_monitors.keys()),
            "process_monitors": list(self.process_monitors.keys()),
        }

        total_monitors = (
            len(monitors["file_monitors"])
            + len(monitors["log_monitors"])
            + len(monitors["process_monitors"])
        )
        output = f"Active monitors: {total_monitors} total"

        return ToolCallResult(
            command="list_monitors",
            success=True,
            output=output,
            execution_time=0.1,
            metadata=monitors,
            error_message=None,
        )

    async def demo(self):
        """Demonstrate the realtime monitoring tool's functionality."""
        print("👁️  REALTIME MONITORING TOOL DEMO")
        print("=" * 40)

        # Create a temporary file to monitor
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Initial content\n")
            temp_file = f.name

        try:
            # Start monitoring the temporary file
            result = await self.execute(
                action="monitor_file",
                monitor_id="demo_file_monitor",
                file_path=temp_file,
            )
            print(f"Start file monitoring: {result.success}")
            print(result.output)

            # List active monitors
            result = await self.execute(action="list_monitors")
            print(f"\nList monitors: {result.success}")
            print(result.output)

            # Stop monitoring
            result = await self.execute(
                action="stop_monitor", monitor_id="demo_file_monitor"
            )
            print(f"\nStop monitoring: {result.success}")
            print(result.output)

        finally:
            os.remove(temp_file)

        print("\n✅ Realtime monitoring demo completed!")
