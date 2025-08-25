"""
WebSocketManager - Advanced WebSocket Communication Core for Janus Metacosmos
=============================================================================

Handles robust connection management, message dispatch, diagnostics, and broadcasting.
Supports extensible message routing, error handling, and integration with backend modules.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from websockets.server import ServerProtocol

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Central manager for WebSocket connections and message routing.
    Provides diagnostics, extensibility, and safe broadcasting.
    """

    def __init__(self):
        self.websocket_clients: set[Any] = set()
        self.message_handlers: dict[str, Callable[[Any, dict], asyncio.Future]] = {}
        self._register_default_handlers()

    async def handle_websocket(
        self, websocket: ServerProtocol, path: str | None = None
    ):
        logger.info("[WebSocketManager] Client connected.")
        self.websocket_clients.add(websocket)
        try:
            async for message_str in websocket:
                await self.process_client_message(websocket, message_str)
        except Exception as e:
            logger.error(f"[WebSocketManager] Error in connection: {e}")
        finally:
            logger.info("[WebSocketManager] Client disconnected.")
            self.websocket_clients.discard(websocket)

    async def process_client_message(
        self, websocket: ServerProtocol, message_str: str | bytes
    ):
        if isinstance(message_str, bytes):
            message_str = message_str.decode("utf-8")
        try:
            message = json.loads(message_str)
            msg_type = message.get("type")
            handler = self.message_handlers.get(msg_type)
            if handler:
                await handler(websocket, message)
            else:
                logger.warning(
                    f"[WebSocketManager] No handler for message type: {msg_type}"
                )
        except json.JSONDecodeError:
            logger.error(f"[WebSocketManager] Non-JSON message received: {message_str}")
        except Exception as e:
            logger.error(f"[WebSocketManager] Error processing client message: {e}")

    def register_handler(
        self, msg_type: str, handler: Callable[[Any, dict], asyncio.Future]
    ):
        """
        Registers a custom handler for a specific message type.
        """
        self.message_handlers[msg_type] = handler
        logger.info(f"[WebSocketManager] Registered handler for type: {msg_type}")

    def _register_default_handlers(self):
        self.register_handler("chat_message", self.handle_chat_message)
        self.register_handler("player_update", self.handle_player_update)

    async def handle_chat_message(self, websocket, message: dict) -> Any:
        data = message.get("data", {})
        content = data.get("content")
        author = data.get("author", "Demiurgo")
        logger.info(f"[WebSocketManager] Chat from '{author}': '{content}'")
        # Optionally broadcast to all clients
        await self.broadcast_to_clients(
            {"type": "chat_message", "data": {"author": author, "content": content}}
        )

    async def handle_player_update(self, websocket, message: dict) -> Any:
        data = message.get("data", {})
        logger.debug(f"[WebSocketManager] Player update received: {data}")
        # Extend: integrate with JanusCore or player manager

    async def broadcast_to_clients(self, message: dict[str, Any]):
        if not self.websocket_clients:
            return
        json_message = json.dumps(message)
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(json_message)
            except Exception as e:
                logger.error(f"[WebSocketManager] Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        self.websocket_clients -= disconnected_clients

    def get_client_count(self) -> int:
        """Returns the number of connected WebSocket clients."""
        return len(self.websocket_clients)

    def get_registered_message_types(self) -> list[str]:
        """Lists all registered message types."""
        return list(self.message_handlers.keys())


class EVAWebSocketManager(WebSocketManager):
    """
    EVAWebSocketManager - Extensión para integración con la memoria viviente EVA.
    Permite ingestión, recall y broadcasting de experiencias EVA, gestión de fases, hooks y benchmarking.
    """

    def __init__(self, eva_api=None):
        super().__init__()
        self.eva_api = eva_api or {}
        self._register_eva_handlers()

    def _register_eva_handlers(self):
        self.register_handler(
            "eva_ingest_experience", self.handle_eva_ingest_experience
        )
        self.register_handler(
            "eva_recall_experience", self.handle_eva_recall_experience
        )
        self.register_handler("eva_set_memory_phase", self.handle_eva_set_memory_phase)
        self.register_handler("eva_get_memory_phase", self.handle_eva_get_memory_phase)
        self.register_handler(
            "eva_add_environment_hook", self.handle_eva_add_environment_hook
        )

    async def handle_eva_ingest_experience(self, websocket, message: dict) -> Any:
        data = message.get("data", {})
        qualia_state = data.get("qualia_state", {})
        phase = data.get("phase")
        if self.eva_api and "eva_ingest_experience" in self.eva_api:
            exp_id = self.eva_api["eva_ingest_experience"](data, qualia_state, phase)
            logger.info(f"[EVAWebSocketManager] EVA experience ingested: {exp_id}")
            await self.broadcast_to_clients(
                {
                    "type": "eva_experience_ingested",
                    "data": {"experience_id": exp_id, "phase": phase},
                }
            )

    async def handle_eva_recall_experience(self, websocket, message: dict) -> Any:
        data = message.get("data", {})
        cue = data.get("experience_id")
        phase = data.get("phase")
        if self.eva_api and "eva_recall_experience" in self.eva_api:
            result = self.eva_api["eva_recall_experience"](cue, phase)
            logger.info(f"[EVAWebSocketManager] EVA experience recalled: {cue}")
            await websocket.send(
                json.dumps({"type": "eva_experience_recalled", "data": result})
            )

    async def handle_eva_set_memory_phase(self, websocket, message: dict) -> Any:
        phase = message.get("data", {}).get("phase")
        if self.eva_api and "set_memory_phase" in self.eva_api:
            self.eva_api["set_memory_phase"](phase)
            logger.info(f"[EVAWebSocketManager] EVA memory phase set: {phase}")
            await self.broadcast_to_clients(
                {"type": "eva_memory_phase_set", "data": {"phase": phase}}
            )

    async def handle_eva_get_memory_phase(self, websocket, message: dict) -> Any:
        if self.eva_api and "get_memory_phase" in self.eva_api:
            phase = self.eva_api["get_memory_phase"]()
            await websocket.send(
                json.dumps({"type": "eva_memory_phase", "data": {"phase": phase}})
            )

    async def handle_eva_add_environment_hook(self, websocket, message: dict) -> Any:
        hook_name = message.get("data", {}).get("hook_name")

        # For demonstration, register a dummy hook
        def dummy_hook(event):
            logger.info(
                f"[EVAWebSocketManager] Dummy hook triggered for event: {event}"
            )

        if self.eva_api and "add_environment_hook" in self.eva_api:
            self.eva_api["add_environment_hook"](dummy_hook)
            logger.info(
                f"[EVAWebSocketManager] EVA environment hook added: {hook_name}"
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "eva_environment_hook_added",
                        "data": {"hook_name": hook_name},
                    }
                )
            )

    async def broadcast_eva_experience(self, experience: dict):
        await self.broadcast_to_clients(
            {"type": "eva_experience_broadcast", "data": experience}
        )

    def get_eva_client_count(self) -> int:
        """Returns the number of connected EVA WebSocket clients."""
        return self.get_client_count()

    def get_eva_registered_message_types(self) -> list[str]:
        """Lists all registered EVA message types."""
        return [k for k in self.message_handlers.keys() if k.startswith("eva_")]
