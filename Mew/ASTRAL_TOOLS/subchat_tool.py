import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class SubchatMessage(BaseModel):
    message_id: str = Field(..., description="ID único del mensaje")
    content: str = Field(..., description="Contenido del mensaje")
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        "normal", description="Prioridad"
    )
    category: Literal[
        "comment", "suggestion", "question", "feedback", "instruction", "alert"
    ] = Field("comment", description="Categoría")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Fecha de creación"
    )
    context: str | None = Field(None, description="Contexto o tarea relacionada")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadatos adicionales"
    )
    read: bool = Field(False, description="Leído")
    processed: bool = Field(False, description="Procesado/Respondido")


class SubchatParameters(BaseModel):
    action: Literal[
        "send_message",
        "get_messages",
        "mark_read",
        "mark_processed",
        "clear_messages",
        "get_stats",
        "set_auto_response",
        "process_pending",
    ] = Field(..., description="Acción a realizar")
    message: str | None = Field(None, description="Mensaje a enviar")
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        "normal", description="Prioridad"
    )
    category: Literal[
        "comment", "suggestion", "question", "feedback", "instruction", "alert"
    ] = Field("comment", description="Categoría")
    context: str | None = Field(None, description="Contexto/tarea")
    filter_unread: bool = Field(False, description="Solo mensajes no leídos")
    filter_priority: str | None = Field(None, description="Filtrar por prioridad")
    filter_category: str | None = Field(None, description="Filtrar por categoría")
    max_messages: int = Field(10, description="Máximo de mensajes a retornar")
    message_id: str | None = Field(None, description="ID específico de mensaje")
    message_ids: list[str] | None = Field(None, description="Lista de IDs")
    auto_response_enabled: bool = Field(False, description="Habilitar auto-respuesta")
    auto_response_delay: float = Field(
        5.0, description="Delay de auto-respuesta (segundos)"
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if v not in ["low", "normal", "high", "urgent"]:
            raise ValueError("Prioridad inválida")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        valid = [
            "comment",
            "suggestion",
            "question",
            "feedback",
            "instruction",
            "alert",
        ]
        if v not in valid:
            raise ValueError("Categoría inválida")
        return v


class SubchatTool(BaseTool):
    """Herramienta de comunicación asíncrona y feedback"""

    def __init__(self):
        super().__init__()
        self._messages: dict[str, SubchatMessage] = {}
        self._message_queue: asyncio.Queue[SubchatMessage] = asyncio.Queue()
        self._auto_response_enabled = False
        self._auto_response_delay = 5.0
        self._storage_path = Path.home() / ".crisalida" / "subchat"
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_messages()

    def _get_name(self) -> str:
        return "subchat"

    def _get_description(self) -> str:
        return "Comunicación asíncrona y feedback sin interrupciones"

    def _get_category(self) -> str:
        return "communication"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return SubchatParameters

    def _generate_message_id(self) -> str:
        timestamp = int(time.time() * 1000)
        return f"msg_{timestamp}_{len(self._messages)}"

    def _save_messages(self):
        try:
            messages_file = self._storage_path / "messages.json"
            with open(messages_file, "w", encoding="utf-8") as f:
                messages_dict = {
                    msg_id: {**msg.model_dump(), "timestamp": msg.timestamp.isoformat()}
                    for msg_id, msg in self._messages.items()
                }
                json.dump(messages_dict, f, indent=2)
        except Exception as e:
            logger.warning(f"Error al guardar mensajes: {e}")

    def _load_messages(self):
        try:
            messages_file = self._storage_path / "messages.json"
            if messages_file.exists():
                with open(messages_file, encoding="utf-8") as f:
                    messages_dict = json.load(f)
                for msg_id, msg_data in messages_dict.items():
                    if "timestamp" in msg_data and isinstance(
                        msg_data["timestamp"], str
                    ):
                        msg_data["timestamp"] = datetime.fromisoformat(
                            msg_data["timestamp"]
                        )
                    self._messages[msg_id] = SubchatMessage(**msg_data)
                logger.info(f"Mensajes cargados: {len(self._messages)}")
        except Exception as e:
            logger.warning(f"Error al cargar mensajes: {e}")
            self._messages = {}

    def _add_message(self, message: SubchatMessage) -> str:
        self._messages[message.message_id] = message
        self._save_messages()
        try:
            self._message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Cola de mensajes llena")
        return message.message_id

    def _get_filtered_messages(
        self,
        filter_unread: bool = False,
        filter_priority: str | None = None,
        filter_category: str | None = None,
        max_messages: int = 10,
    ) -> list[SubchatMessage]:
        messages = list(self._messages.values())
        if filter_unread:
            messages = [m for m in messages if not m.read]
        if filter_priority:
            messages = [m for m in messages if m.priority == filter_priority]
        if filter_category:
            messages = [m for m in messages if m.category == filter_category]
        messages.sort(key=lambda m: m.timestamp, reverse=True)
        return messages[:max_messages]

    def _format_messages(self, messages: list[SubchatMessage]) -> str:
        if not messages:
            return "No hay mensajes."
        output = [f"Mensajes ({len(messages)}):\n"]
        for msg in messages:
            read_status = "📖" if msg.read else "📄"
            processed_status = "✅" if msg.processed else "⏳"
            priority_icons = {"low": "🔵", "normal": "⚪", "high": "🟡", "urgent": "🔴"}
            category_icons = {
                "comment": "💬",
                "suggestion": "💡",
                "question": "❓",
                "feedback": "📝",
                "instruction": "📋",
                "alert": "⚠️",
            }
            priority_icon = priority_icons.get(msg.priority, "⚪")
            category_icon = category_icons.get(msg.category, "💬")
            time_str = msg.timestamp.strftime("%H:%M:%S")
            output.append(
                f"{read_status}{processed_status} {priority_icon}{category_icon} [{time_str}] {msg.message_id}"
            )
            if msg.context:
                output.append(f"    Contexto: {msg.context}")
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            output.append(f"    {content}\n")
        return "\n".join(output)

    def _get_stats(self) -> dict[str, Any]:
        total = len(self._messages)
        unread = sum(1 for m in self._messages.values() if not m.read)
        unprocessed = sum(1 for m in self._messages.values() if not m.processed)
        priority_counts: dict[str, int] = {}
        for msg in self._messages.values():
            priority_counts[msg.priority] = priority_counts.get(msg.priority, 0) + 1
        category_counts: dict[str, int] = {}
        for msg in self._messages.values():
            category_counts[msg.category] = category_counts.get(msg.category, 0) + 1
        one_hour_ago = datetime.now().timestamp() - 3600
        recent = sum(
            1 for m in self._messages.values() if m.timestamp.timestamp() > one_hour_ago
        )
        return {
            "total_messages": total,
            "unread": unread,
            "unprocessed": unprocessed,
            "recent_activity": recent,
            "priority_breakdown": priority_counts,
            "category_breakdown": category_counts,
            "auto_response_enabled": self._auto_response_enabled,
        }

    async def _process_auto_responses(self):
        if not self._auto_response_enabled:
            return []
        urgent_messages = [
            msg
            for msg in self._messages.values()
            if not msg.processed and msg.priority in ["high", "urgent"]
        ]
        responses = []
        for msg in urgent_messages:
            if msg.category == "question":
                responses.append(
                    f"Auto-respuesta: Pregunta '{msg.content[:50]}...' recibida y será atendida."
                )
            elif msg.category == "alert":
                responses.append(
                    f"Auto-respuesta: Alerta '{msg.content[:50]}...' reconocida."
                )
            elif msg.category == "instruction":
                responses.append(
                    f"Auto-respuesta: Instrucción '{msg.content[:50]}...' registrada para ejecución."
                )
        return responses

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = asyncio.get_event_loop().time()
        try:
            params = SubchatParameters(**kwargs)
            if params.action == "send_message":
                if not params.message:
                    return ToolCallResult(
                        command="subchat",
                        success=False,
                        output="",
                        error_message="message es requerido para send_message",
                        execution_time=(asyncio.get_event_loop().time() - start_time),
                    )
                message = SubchatMessage(
                    message_id=self._generate_message_id(),
                    content=params.message,
                    priority=params.priority,
                    category=params.category,
                    context=params.context,
                    metadata={"sender": "user"},
                    read=False,
                    processed=False,
                )
                message_id = self._add_message(message)
                output = "✅ Mensaje enviado correctamente!\n\n"
                output += f"ID: {message_id}\nPrioridad: {params.priority}\nCategoría: {params.category}\n"
                if params.context:
                    output += f"Contexto: {params.context}\n"
                output += f"Contenido: {params.message}\n"
                if self._auto_response_enabled and params.priority in [
                    "high",
                    "urgent",
                ]:
                    output += f"\n⚡ Mensaje de alta prioridad - auto-respuesta en {self._auto_response_delay}s"
            elif params.action == "get_messages":
                messages = self._get_filtered_messages(
                    filter_unread=params.filter_unread,
                    filter_priority=params.filter_priority,
                    filter_category=params.filter_category,
                    max_messages=params.max_messages,
                )
                output = self._format_messages(messages)
            elif params.action == "mark_read":
                marked = 0
                if params.message_id:
                    if params.message_id in self._messages:
                        self._messages[params.message_id].read = True
                        marked = 1
                        output = f"✅ Mensaje {params.message_id} marcado como leído"
                    else:
                        output = f"❌ Mensaje {params.message_id} no encontrado"
                elif params.message_ids:
                    for msg_id in params.message_ids:
                        if msg_id in self._messages:
                            self._messages[msg_id].read = True
                            marked += 1
                    output = f"✅ {marked}/{len(params.message_ids)} mensajes marcados como leídos"
                else:
                    for msg in self._messages.values():
                        if not msg.read:
                            msg.read = True
                            marked += 1
                    output = f"✅ {marked} mensajes marcados como leídos"
                self._save_messages()
            elif params.action == "mark_processed":
                marked = 0
                if params.message_id:
                    if params.message_id in self._messages:
                        self._messages[params.message_id].processed = True
                        marked = 1
                        output = (
                            f"✅ Mensaje {params.message_id} marcado como procesado"
                        )
                    else:
                        output = f"❌ Mensaje {params.message_id} no encontrado"
                elif params.message_ids:
                    for msg_id in params.message_ids:
                        if msg_id in self._messages:
                            self._messages[msg_id].processed = True
                            marked += 1
                    output = f"✅ {marked}/{len(params.message_ids)} mensajes marcados como procesados"
                else:
                    for msg in self._messages.values():
                        if not msg.processed:
                            msg.processed = True
                            marked += 1
                    output = f"✅ {marked} mensajes marcados como procesados"
                self._save_messages()
            elif params.action == "clear_messages":
                if params.filter_category or params.filter_priority:
                    to_remove = []
                    for msg_id, msg in self._messages.items():
                        should_remove = True
                        if (
                            params.filter_category
                            and msg.category != params.filter_category
                        ):
                            should_remove = False
                        if (
                            params.filter_priority
                            and msg.priority != params.filter_priority
                        ):
                            should_remove = False
                        if should_remove:
                            to_remove.append(msg_id)
                    for msg_id in to_remove:
                        del self._messages[msg_id]
                    self._save_messages()
                    output = f"🗑️ {len(to_remove)} mensajes filtrados eliminados"
                else:
                    count = len(self._messages)
                    self._messages.clear()
                    self._save_messages()
                    output = f"🗑️ Todos los {count} mensajes eliminados"
            elif params.action == "get_stats":
                stats = self._get_stats()
                output = "📊 Estadísticas Subchat:\n\n"
                output += f"Total: {stats['total_messages']}\nNo leídos: {stats['unread']}\nNo procesados: {stats['unprocessed']}\nActividad reciente (1h): {stats['recent_activity']}\n\n"
                output += "Prioridad:\n"
                for priority, count in stats["priority_breakdown"].items():
                    output += f"  {priority}: {count}\n"
                output += "\nCategoría:\n"
                for category, count in stats["category_breakdown"].items():
                    output += f"  {category}: {count}\n"
                output += f"\nAuto-respuesta: {'Activada' if stats['auto_response_enabled'] else 'Desactivada'}"
            elif params.action == "set_auto_response":
                self._auto_response_enabled = params.auto_response_enabled
                if params.auto_response_delay > 0:
                    self._auto_response_delay = params.auto_response_delay
                status = "activada" if self._auto_response_enabled else "desactivada"
                output = f"⚙️ Auto-respuesta {status}"
                if self._auto_response_enabled:
                    output += f" con delay de {self._auto_response_delay}s"
            elif params.action == "process_pending":
                auto_responses = await self._process_auto_responses()
                output = "🔄 Procesando mensajes pendientes:\n\n"
                if auto_responses:
                    for response in auto_responses:
                        output += f"• {response}\n"
                else:
                    output += "No hay mensajes de alta prioridad pendientes."
                for msg in self._messages.values():
                    if not msg.processed and msg.priority in ["high", "urgent"]:
                        msg.processed = True
                self._save_messages()
            else:
                return ToolCallResult(
                    command="subchat",
                    success=False,
                    output="",
                    error_message=f"Acción desconocida: {params.action}",
                    execution_time=(asyncio.get_event_loop().time() - start_time),
                )
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolCallResult(
                command=f"subchat({params.action})",
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={
                    "action": params.action,
                    "total_messages": len(self._messages),
                    "auto_response_enabled": self._auto_response_enabled,
                },
                error_message=None,
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Subchat error: {e}")
            return ToolCallResult(
                command="subchat",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def demo(self):
        """Demonstrate the subchat tool's functionality."""
        print("💬 SUBCHAT TOOL DEMO")
        print("=" * 40)

        # Send a message
        result = await self.execute(
            action="send_message",
            message="Este es un comentario de prueba sobre la implementación actual",
            priority="normal",
            category="comment",
            context="Demo testing",
        )
        print(f"Send message: {result.success}")
        print(result.output)

        # Get messages
        result = await self.execute(action="get_messages", max_messages=5)
        print(f"\nGet messages: {result.success}")
        print(
            result.output[:300] + "..." if len(result.output) > 300 else result.output
        )

        # Get stats
        result = await self.execute(action="get_stats")
        print(f"\nStats: {result.success}")
        print(result.output)

        print("\n✅ Subchat demo completed!")


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.subchat_tool_demos import demo_subchat

    asyncio.run(demo_subchat())
