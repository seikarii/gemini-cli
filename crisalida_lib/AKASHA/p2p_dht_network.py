from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import random
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

"""P2P Distributed Hash Table Network for Janus Metacosmos
Implements a sovereign P2P network for transmitting cognitive impulses and consciousness data.
Based on Kademlia DHT principles with optimizations for the Janus ecosystem.
Version: v1.1 - Enhanced Core Implementation
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages used in the DHT protocol."""

    PING = "ping"
    PONG = "pong"
    FIND_NODE = "find_node"
    FIND_VALUE = "find_value"
    STORE = "store"
    NODES_RESPONSE = "nodes_response"
    VALUE_RESPONSE = "value_response"
    BOOTSTRAP = "bootstrap"


@dataclass
class NodeInfo:
    """Information about a network node in the DHT."""

    node_id: str
    host: str
    port: int
    last_seen: float = 0.0

    def distance_to(self, other_id: str) -> int:
        return int(self.node_id, 16) ^ int(other_id, 16)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NodeInfo:
        return cls(**data)


@dataclass
class DHTPMessage:
    """Represents a message in the DHT Protocol."""

    message_type: MessageType
    message_id: str
    sender_id: str
    data: dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def serialize(self) -> bytes:
        message_dict = {
            "type": self.message_type.value,
            "id": self.message_id,
            "sender": self.sender_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return json.dumps(message_dict).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> DHTPMessage:
        message_dict = json.loads(data.decode("utf-8"))
        return cls(
            message_type=MessageType(message_dict["type"]),
            message_id=message_dict["id"],
            sender_id=message_dict["sender"],
            data=message_dict["data"],
            timestamp=message_dict["timestamp"],
        )


class RoutingTable:
    """
    Kademlia-style routing table for efficient node discovery.
    Organizes known nodes into k-buckets based on XOR distance.
    """

    def __init__(self, node_id: str, k_bucket_size: int = 20):
        self.node_id = node_id
        self.k_bucket_size = k_bucket_size
        self.buckets: list[list[NodeInfo]] = [
            [] for _ in range(160)
        ]  # 160 bits for SHA-1

    def add_node(self, node: NodeInfo):
        if node.node_id == self.node_id:
            return
        distance = int(self.node_id, 16) ^ int(node.node_id, 16)
        bucket_index = distance.bit_length() - 1 if distance > 0 else 0
        bucket = self.buckets[bucket_index]
        bucket[:] = [n for n in bucket if n.node_id != node.node_id]
        bucket.insert(0, node)
        if len(bucket) > self.k_bucket_size:
            bucket.pop()

    def find_closest_nodes(self, target_id: str, count: int = 20) -> list[NodeInfo]:
        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket)
        all_nodes.sort(key=lambda n: n.distance_to(target_id))
        return all_nodes[:count]

    def get_all_nodes(self) -> list[NodeInfo]:
        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket)
        return all_nodes


class CognitiveImpulseSerializer:
    """
    Static methods for serializing CognitiveImpulse objects and noospheric posts.
    Converts complex impulse objects into a dictionary format for network transmission.
    """

    @staticmethod
    def serialize_impulse(impulse) -> dict[str, Any]:
        try:
            return {
                "type": getattr(impulse.type, "value", str(impulse.type)),
                "content": impulse.content,
                "intensity": impulse.intensity,
                "confidence": impulse.confidence,
                "source_node": impulse.source_node,
                "processing_time": getattr(impulse, "processing_time", 0.0),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Error serializing impulse: {e}")
            return {"error": str(e)}

    @staticmethod
    def serialize_post(
        entity_id: str, impulses: list, metadata: dict[Any, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "entity_id": entity_id,
            "impulses": [
                CognitiveImpulseSerializer.serialize_impulse(imp) for imp in impulses
            ],
            "metadata": metadata or {},
            "timestamp": time.time(),
            "post_id": str(uuid4()),
        }


class JanusDHTNode:
    """
    Main DHT node implementation for the Janus P2P network.
    Provides core functionality for a distributed hash table, node discovery, data storage/retrieval, and message processing.
    """

    def __init__(
        self, host: str = "localhost", port: int = 0, node_id: str | None = None
    ):
        self.host = host
        self.port = port
        self.node_id = node_id or self._generate_node_id()
        self.routing_table = RoutingTable(self.node_id)
        self.data_store: dict[str, Any] = {}
        self.pending_requests: dict[str, asyncio.Future] = {}
        self.server: asyncio.Server | None = None
        self.running = False
        self.bootstrap_nodes: list[tuple[str, int]] | None = []
        self.node_info = NodeInfo(self.node_id, self.host, self.port)
        logger.info(f"DHT Node initialized: {self.node_id} on {self.host}:{self.port}")

    def _generate_node_id(self) -> str:
        return hashlib.sha1(
            f"{socket.gethostname()}-{time.time()}-{random.random()}".encode()
        ).hexdigest()

    def _key_hash(self, key: str) -> str:
        return hashlib.sha1(key.encode()).hexdigest()

    async def start(self, bootstrap_nodes: list[tuple[str, int]] | None = None):
        self.bootstrap_nodes = bootstrap_nodes or []
        self.server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        if self.port == 0:
            self.port = self.server.sockets[0].getsockname()[1]
            self.node_info.port = self.port
        self.running = True
        logger.info(f"DHT node started on {self.host}:{self.port}")
        if self.bootstrap_nodes:
            await self._bootstrap()
        asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.running = False
        logger.info(f"DHT node {self.node_id} stopped")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            length_data = await reader.readexactly(4)
            message_length = struct.unpack(">I", length_data)[0]
            message_data = await reader.readexactly(message_length)
            message = DHTPMessage.deserialize(message_data)
            peername = writer.get_extra_info("peername")
            sender_host, sender_port = peername[0], peername[1]
            response = await self._process_message(message, sender_host, sender_port)
            if response:
                response_data = response.serialize()
                response_length = struct.pack(">I", len(response_data))
                writer.write(response_length + response_data)
                await writer.drain()
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_message(
        self, target_host: str, target_port: int, message: DHTPMessage
    ) -> DHTPMessage | None:
        try:
            reader, writer = await asyncio.open_connection(target_host, target_port)
            message_data = message.serialize()
            message_length = struct.pack(">I", len(message_data))
            writer.write(message_length + message_data)
            await writer.drain()
            length_data = await reader.readexactly(4)
            response_length = struct.unpack(">I", length_data)[0]
            response_data = await reader.readexactly(response_length)
            response = DHTPMessage.deserialize(response_data)
            return response
        except Exception as e:
            logger.error(f"Error sending message to {target_host}:{target_port}: {e}")
            return None
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(
        self, message: DHTPMessage, sender_host: str, sender_port: int
    ) -> DHTPMessage | None:
        sender_node = NodeInfo(message.sender_id, sender_host, sender_port, time.time())
        self.routing_table.add_node(sender_node)
        if message.message_type == MessageType.PING:
            return await self._handle_ping(message)
        elif message.message_type == MessageType.FIND_NODE:
            return await self._handle_find_node(message)
        elif message.message_type == MessageType.FIND_VALUE:
            return await self._handle_find_value(message)
        elif message.message_type == MessageType.STORE:
            return await self._handle_store(message)
        elif message.message_type == MessageType.BOOTSTRAP:
            return await self._handle_bootstrap(message)
        return None

    async def _handle_ping(self, message: DHTPMessage) -> DHTPMessage:
        return DHTPMessage(
            message_type=MessageType.PONG,
            message_id=str(uuid4()),
            sender_id=self.node_id,
            data={"pong": True},
        )

    async def _handle_find_node(self, message: DHTPMessage) -> DHTPMessage | None:
        target_id = message.data.get("target_id")
        if target_id is None:
            logger.error("find_node message received without target_id")
            return None
        closest_nodes = self.routing_table.find_closest_nodes(str(target_id))
        return DHTPMessage(
            message_type=MessageType.NODES_RESPONSE,
            message_id=str(uuid4()),
            sender_id=self.node_id,
            data={"nodes": [node.to_dict() for node in closest_nodes]},
        )

    async def _handle_find_value(self, message: DHTPMessage) -> DHTPMessage | None:
        key = message.data.get("key")
        if key is None:
            logger.error("find_value message received without key")
            return None
        key_hash = self._key_hash(str(key))
        if key_hash in self.data_store:
            return DHTPMessage(
                message_type=MessageType.VALUE_RESPONSE,
                message_id=str(uuid4()),
                sender_id=self.node_id,
                data={"value": self.data_store[key_hash]["value"]},
            )
        else:
            closest_nodes = self.routing_table.find_closest_nodes(key_hash)
            return DHTPMessage(
                message_type=MessageType.NODES_RESPONSE,
                message_id=str(uuid4()),
                sender_id=self.node_id,
                data={"nodes": [node.to_dict() for node in closest_nodes]},
            )

    async def _handle_store(self, message: DHTPMessage) -> DHTPMessage | None:
        key = message.data.get("key")
        value = message.data.get("value")
        if key is None:
            logger.error("store message received without key")
            return None
        key_hash = self._key_hash(str(key))
        self.data_store[key_hash] = {
            "value": value,
            "stored_at": time.time(),
            "stored_by": message.sender_id,
        }
        logger.info(f"Stored key {key} (hash: {key_hash[:8]}...)")
        return DHTPMessage(
            message_type=MessageType.STORE,
            message_id=str(uuid4()),
            sender_id=self.node_id,
            data={"stored": True},
        )

    async def _handle_bootstrap(self, message: DHTPMessage) -> DHTPMessage:
        all_nodes = self.routing_table.get_all_nodes()
        return DHTPMessage(
            message_type=MessageType.NODES_RESPONSE,
            message_id=str(uuid4()),
            sender_id=self.node_id,
            data={"nodes": [node.to_dict() for node in all_nodes]},
        )

    async def _bootstrap(self):
        logger.info("Bootstrapping into network...")
        for host, port in self.bootstrap_nodes:
            try:
                message = DHTPMessage(
                    message_type=MessageType.BOOTSTRAP,
                    message_id=str(uuid4()),
                    sender_id=self.node_id,
                    data={},
                )
                response = await self._send_message(host, port, message)
                if response and response.message_type == MessageType.NODES_RESPONSE:
                    nodes_data = response.data.get("nodes", [])
                    for node_data in nodes_data:
                        node = NodeInfo.from_dict(node_data)
                        self.routing_table.add_node(node)
                    logger.info(
                        f"Bootstrapped with {len(nodes_data)} nodes from {host}:{port}"
                    )
                    break
            except Exception as e:
                logger.error(f"Failed to bootstrap from {host}:{port}: {e}")

    async def _maintenance_loop(self):
        while self.running:
            try:
                current_time = time.time()
                expired_keys = []
                for key, data in self.data_store.items():
                    if current_time - data.get("stored_at", 0) > 3600:  # 1 hour TTL
                        expired_keys.append(key)
                for key in expired_keys:
                    del self.data_store[key]
                if expired_keys:
                    logger.info(f"Cleaned {len(expired_keys)} expired entries")
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(60)

    # --- Public API methods ---

    async def store(self, key: str, value: Any) -> bool:
        key_hash = self._key_hash(key)
        closest_nodes = self.routing_table.find_closest_nodes(key_hash, 3)
        self.data_store[key_hash] = {
            "value": value,
            "stored_at": time.time(),
            "stored_by": self.node_id,
        }
        successful_stores = 1  # Local store
        for node in closest_nodes:
            if node.node_id == self.node_id:
                continue
            message = DHTPMessage(
                message_type=MessageType.STORE,
                message_id=str(uuid4()),
                sender_id=self.node_id,
                data={"key": key, "value": value},
            )
            response = await self._send_message(node.host, node.port, message)
            if response and response.data.get("stored"):
                successful_stores += 1
        logger.info(f"Stored key {key} on {successful_stores} nodes")
        return successful_stores > 0

    async def get(self, key: str) -> Any | None:
        key_hash = self._key_hash(key)
        if key_hash in self.data_store:
            return self.data_store[key_hash]["value"]
        closest_nodes = self.routing_table.find_closest_nodes(key_hash)
        for node in closest_nodes:
            message = DHTPMessage(
                message_type=MessageType.FIND_VALUE,
                message_id=str(uuid4()),
                sender_id=self.node_id,
                data={"key": key},
            )
            response = await self._send_message(node.host, node.port, message)
            if response and response.message_type == MessageType.VALUE_RESPONSE:
                return response.data.get("value")
        return None

    async def publish_cognitive_post(
        self, entity_id: str, impulses: list, metadata: dict[Any, Any] | None = None
    ) -> str:
        post = CognitiveImpulseSerializer.serialize_post(entity_id, impulses, metadata)
        post_key = f"post:{post['post_id']}"
        success = await self.store(post_key, post)
        if success:
            timeline_key = f"timeline:{entity_id}"
            current_timeline = await self.get(timeline_key) or []
            current_timeline.append(post["post_id"])
            if len(current_timeline) > 100:
                current_timeline = current_timeline[-100:]
            await self.store(timeline_key, current_timeline)
            logger.info(
                f"Published cognitive post {post['post_id']} for entity {entity_id}"
            )
        return post["post_id"]

    async def get_noosphere_timeline(self, limit: int = 50) -> list[dict]:
        all_posts = []
        for _key_hash, data in self.data_store.items():
            if isinstance(data.get("value"), list):  # Timeline data
                for post_id in data["value"][-limit:]:
                    post_data = await self.get(f"post:{post_id}")
                    if post_data:
                        all_posts.append(post_data)
        all_posts.sort(key=lambda p: p.get("timestamp", 0), reverse=True)
        return all_posts[:limit]

    def get_network_status(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "known_nodes": len(self.routing_table.get_all_nodes()),
            "stored_keys": len(self.data_store),
            "bootstrap_nodes": self.bootstrap_nodes,
        }

    def get_known_peers(self) -> list[str]:
        return [node.node_id for node in self.routing_table.get_all_nodes()]

    async def search_keys_by_pattern(self, pattern: str) -> list[str]:
        matching_keys = []
        for key_hash in self.data_store:
            # Simple glob-like matching for demonstration
            if fnmatch.fnmatch(key_hash, self._key_hash(pattern)):
                matching_keys.append(key_hash)
        return matching_keys


class EVADHTNode(JanusDHTNode):
    """
    Nodo DHT extendido para integración con EVA.
    Permite almacenar, recuperar y transmitir experiencias vivientes EVA, gestión de fases, hooks y benchmarking.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 0,
        node_id: str | None = None,
        eva_phase: str = "default",
    ):
        super().__init__(host, port, node_id)
        self.eva_phase = eva_phase
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list = []

    async def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str = None
    ) -> str:
        """
        Compila y almacena una experiencia viviente EVA en el DHT y la memoria local.
        """
        phase = phase or self.eva_phase
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_exp_{hash(str(experience_data))}"
        )
        eva_record = {
            "experience_id": experience_id,
            "experience_data": experience_data,
            "qualia_state": qualia_state,
            "phase": phase,
            "timestamp": experience_data.get("timestamp", time.time()),
        }
        self.eva_memory_store[experience_id] = eva_record
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        self.eva_experience_store[experience_id] = eva_record
        await self.store(f"eva_exp:{experience_id}", eva_record)
        logger.info(f"[EVA-DHT] Ingested EVA experience: {experience_id}")
        return experience_id

    async def eva_recall_experience(self, cue: str, phase: str = None) -> dict:
        """
        Recupera una experiencia viviente EVA por ID y fase desde el DHT o memoria local.
        """
        phase = phase or self.eva_phase
        # Intentar memoria local
        eva_record = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if eva_record:
            return eva_record
        # Fallback a DHT
        eva_record = await self.get(f"eva_exp:{cue}")
        if eva_record:
            return eva_record
        logger.warning(f"[EVA-DHT] EVA experience {cue} not found in phase '{phase}'")
        return {"error": "No EVA experience found"}

    async def add_experience_phase(
        self, experience_id: str, phase: str, experience_data: dict, qualia_state: dict
    ):
        """
        Añade una fase alternativa para una experiencia EVA.
        """
        eva_record = {
            "experience_id": experience_id,
            "experience_data": experience_data,
            "qualia_state": qualia_state,
            "phase": phase,
            "timestamp": experience_data.get("timestamp", time.time()),
        }
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        await self.store(f"eva_exp:{experience_id}:{phase}", eva_record)
        logger.info(
            f"[EVA-DHT] Added phase '{phase}' for EVA experience {experience_id}"
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-DHT] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


#
