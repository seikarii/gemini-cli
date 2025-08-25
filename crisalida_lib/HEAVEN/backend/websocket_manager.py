"""
WebSocketManager - Gestor centralizado de conexiones WebSocket para el backend Janus Metacosmos.
Gestiona clientes, procesa mensajes entrantes, integra con sistemas de entidades y broadcasting.
Incluye trazabilidad avanzada, manejo de errores y soporte para extensibilidad.
"""

import asyncio
import json
import logging
from typing import Any
import websockets

from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime
from crisalida_lib.EVA.types import EVAExperience, QualiaState, RealityBytecode

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self, janus_core: Any | None = None):
        self.websocket_clients: set[Any] = set()
        self.janus_core = janus_core  # Permite integración directa con JanusCore

    async def handle_websocket(
        self, websocket: websockets.WebSocketServerProtocol, path: str | None = None
    ):
        logger.info("Cliente WebSocket conectado.")
        self.websocket_clients.add(websocket)
        try:
            async for message_str in websocket:
                await self.process_client_message(websocket, message_str)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Conexión WebSocket cerrada correctamente.")
        except Exception as e:
            logger.error(f"Error en WebSocket: {e}")
        finally:
            logger.info("Cliente WebSocket desconectado.")
            self.websocket_clients.remove(websocket)

    async def process_client_message(
        self, websocket: websockets.WebSocketServerProtocol, message_str: str | bytes
    ):
        if isinstance(message_str, bytes):
            message_str = message_str.decode("utf-8")

        try:
            message = json.loads(message_str)
            msg_type = message.get("type")
            data = message.get("data", {})

            if msg_type == "chat_message":
                await self.handle_chat_message(data)
                if self.janus_core:
                    # Integrar con almacenamiento y broadcasting real
                    self.janus_core.storage_manager.create_post(
                        entity_id=data.get("author", "Demiurgo"),
                        content=data.get("content", ""),
                        qualia_state=None,
                        ontological_signature=None,
                        post_type="insight_sharing",
                        elemental_resonance=None,
                        consciousness_depth=1.0,
                        meme_potential=0.0,
                        timestamp=data.get("timestamp"),
                    )
                    await self.broadcast_to_clients(
                        {
                            "type": "new_post",
                            "data": {
                                "author": data.get("author", "Demiurgo"),
                                "content": data.get("content", ""),
                                "timestamp": data.get("timestamp"),
                            },
                        }
                    )
            elif msg_type == "player_update":
                logger.debug(f"Player update received: {data}")
                if self.janus_core:
                    self.janus_core.player_state.update(data)
            elif msg_type == "create_entity":
                if self.janus_core:
                    entity_id = data.get("id", None)
                    position = data.get("position", [0, 0, 0])
                    self.janus_core.create_entity(entity_id or "entity_auto", position)
            elif msg_type == "update_entity":
                if self.janus_core:
                    entity_id = data.get("id", None)
                    updates = data.get("updates", {})
                    self.janus_core.update_entity(entity_id, updates)
            elif msg_type == "delete_entity":
                if self.janus_core:
                    entity_id = data.get("id", None)
                    self.janus_core.delete_entity(entity_id)
            elif msg_type == "demiurge_command":
                if self.janus_core:
                    await self.janus_core._handle_demiurge_command(websocket, data)
            else:
                logger.warning(f"Tipo de mensaje desconocido recibido: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Mensaje no JSON recibido: {message_str}")
        except Exception as e:
            logger.error(f"Error procesando mensaje del cliente: {e}")

    async def handle_chat_message(self, data):
        content = data.get("content")
        author = data.get("author", "Demiurgo")
        logger.info(f"Mensaje de chat de '{author}': '{content}'")

    async def broadcast_to_clients(self, message: dict[str, Any]):
        if self.websocket_clients:
            json_message = json.dumps(message)
            await asyncio.wait(
                [
                    asyncio.create_task(client.send(json_message))
                    for client in self.websocket_clients
                ]
            )


class EVAWebSocketManager(WebSocketManager):
    """
    EVAWebSocketManager - Extensión para integración con la memoria viviente EVA.
    Permite ingestión, recall y broadcasting de experiencias EVA, gestión de fases, hooks y benchmarking.
    """

    def __init__(self, janus_core: Any | None = None, eva_api: dict = None):
        super().__init__(janus_core)
        self.eva_api = eva_api or {}
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_phase = (
            getattr(janus_core, "eva_phase", "default") if janus_core else "default"
        )
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []
        self._register_eva_handlers()

    def _register_eva_handlers(self):
        # Placeholder para registrar handlers EVA (puede ser extendido dinámicamente)
        pass

    async def handle_eva_ingest_experience(self, websocket, message: dict) -> Any:
        """
        Ingesta una experiencia viviente EVA desde el cliente y la almacena en la memoria.
        """
        data = message.get("data", {})
        qualia_state = data.get("qualia_state", None)
        phase = data.get("phase", self.eva_phase)
        experience_data = data.get("experience", {})
        if not qualia_state:
            qualia_state = QualiaState()
        exp_id = self.eva_api.get(
            "eva_ingest_janus_experience", self.eva_ingest_experience
        )(experience_data, qualia_state, phase)
        await self.broadcast_eva_experience(
            {"type": "eva_experience_ingested", "experience_id": exp_id}
        )

    async def handle_eva_recall_experience(self, websocket, message: dict) -> Any:
        """
        Recuerda y simula una experiencia viviente EVA por ID y fase.
        """
        data = message.get("data", {})
        cue = data.get("experience_id")
        phase = data.get("phase", self.eva_phase)
        recall_result = self.eva_api.get(
            "eva_recall_janus_experience", self.eva_recall_experience
        )(cue, phase)
        await websocket.send(
            json.dumps({"type": "eva_experience_recalled", "data": recall_result})
        )

    async def handle_eva_set_memory_phase(self, websocket, message: dict) -> Any:
        """
        Cambia la fase activa de memoria EVA.
        """
        data = message.get("data", {})
        phase = data.get("phase", "default")
        self.eva_phase = phase
        self.eva_api.get("set_memory_phase", lambda p: None)(phase)
        await self.broadcast_to_clients({"type": "eva_phase_changed", "phase": phase})

    async def handle_eva_get_memory_phase(self, websocket, message: dict) -> Any:
        """
        Devuelve la fase de memoria actual.
        """
        phase = self.eva_api.get("get_memory_phase", lambda: self.eva_phase)()
        await websocket.send(json.dumps({"type": "eva_phase", "phase": phase}))

    async def handle_eva_add_environment_hook(self, websocket, message: dict) -> Any:
        """
        Registra un hook para manifestación simbólica o eventos EVA.
        """
        # Los hooks deben ser funciones, aquí solo se registra el evento para el ejemplo
        data = message.get("data", {})
        hook_info = data.get("hook_info", {})
        self._environment_hooks.append(hook_info)
        await websocket.send(
            json.dumps({"type": "eva_hook_registered", "hook_info": hook_info})
        )

    async def broadcast_eva_experience(self, experience: dict):
        """
        Broadcast de una experiencia EVA a todos los clientes conectados.
        """
        await self.broadcast_to_clients({"type": "eva_experience", "data": experience})

    def get_eva_client_count(self) -> int:
        """Returns the number of connected EVA WebSocket clients."""
