import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from crisalida_lib.symbolic_language.living_symbols import (
    LivingSymbolRuntime,
    QuantumField,
)
from crisalida_lib.symbolic_language.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)

from .p2p_dht_network import EVADHTNode, JanusDHTNode

logger = logging.getLogger(__name__)


class NoosphereClient:
    """
    Client for entities to interact with the Noosphere P2P network.

    Provides robust methods for publishing cognitive posts, retrieving timelines,
    sending direct messages, broadcasting, and request/response flows.
    Includes extended diagnostics, tracing, and error handling.
    """

    def __init__(self, dht_node_info: dict[str, Any], entity_id: str):
        """
        Initializes the NoosphereClient.

        Args:
            dht_node_info: DHT node configuration (host, port, node_id).
            entity_id: Unique identifier for the entity using this client.
        """
        self.dht_node = JanusDHTNode(
            host=dht_node_info["host"],
            port=dht_node_info["port"],
            node_id=dht_node_info["node_id"],
        )
        self.entity_id = entity_id

    async def publish_post(
        self,
        content_text: str,
        post_type: str = "reflection",
        qualia_state_json: str = "{}",
        impulse_type: str | None = None,
        intensity: float = 0.5,
        confidence: float = 0.8,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Publishes a post (cognitive impulse) to the Noosphere.

        Args:
            content_text: Main textual content of the post.
            post_type: Type of the post (e.g., "reflection", "observation").
            qualia_state_json: JSON string representing qualia state.
            impulse_type: Optional override for impulse type.
            intensity: Emotional intensity (default 0.5).
            confidence: Confidence level (default 0.8).
            extra_metadata: Additional metadata to attach.

        Returns:
            Unique ID of the published post.
        """
        from crisalida_lib.core.cognitive_impulses import CognitiveImpulse, ImpulseType

        impulse_obj = CognitiveImpulse(
            impulse_type=impulse_type or ImpulseType.EMOTIONAL_RESONANCE,
            content=content_text,
            intensity=intensity,
            confidence=confidence,
            source_node=self.entity_id,
            processing_time=0.0,
        )
        metadata = {
            "post_type": post_type,
            "qualia_state_json": qualia_state_json,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        post_id = await self.dht_node.publish_cognitive_post(
            self.entity_id, [impulse_obj], metadata
        )
        logger.info(f"[NoosphereClient] Published post {post_id} ({post_type})")
        return post_id

    async def get_timeline(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Retrieves the entity's own timeline of posts from the Noosphere.

        Args:
            limit: Maximum number of posts to retrieve.

        Returns:
            List of posts in the timeline.
        """
        timeline = await self.dht_node.get_noosphere_timeline(limit)
        logger.debug(f"[NoosphereClient] Retrieved timeline with {len(timeline)} posts")
        return timeline

    async def send_direct_message(
        self,
        recipient_entity_id: str,
        message_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Sends a direct message to another entity via the Noosphere.

        Args:
            recipient_entity_id: Unique ID of the recipient entity.
            message_content: Content of the direct message.
            metadata: Optional metadata to attach.

        Returns:
            True if sent successfully, False otherwise.
        """
        message_id = str(uuid4())
        message_key = f"dm:{recipient_entity_id}:{self.entity_id}:{message_id}"
        message_data = {
            "sender_id": self.entity_id,
            "recipient_id": recipient_entity_id,
            "content": message_content,
            "timestamp": time.time(),
            "message_id": message_id,
        }
        if metadata:
            message_data["metadata"] = metadata
        success = await self.dht_node.store(message_key, message_data)
        if success:
            logger.info(
                f"[NoosphereClient] Sent DM {message_id} from {self.entity_id} to {recipient_entity_id}: {message_content[:50]}..."
            )
        else:
            logger.warning(
                f"[NoosphereClient] Failed to send DM {message_id} from {self.entity_id} to {recipient_entity_id}"
            )
        return success

    async def get_direct_messages(
        self, from_entity_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Retrieves direct messages addressed to this entity.

        Args:
            from_entity_id: Optional sender filter.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of direct messages.
        """
        pattern = f"dm:{self.entity_id}:*"
        if from_entity_id:
            pattern = f"dm:{self.entity_id}:{from_entity_id}:*"
        keys = await self.dht_node.search_keys_by_pattern(pattern)
        messages = []
        for key in keys[:limit]:
            message = await self.dht_node.get(key)
            if message:
                messages.append(message)
        logger.debug(f"[NoosphereClient] Retrieved {len(messages)} direct messages")
        return messages

    async def broadcast_message(
        self,
        message_content: str,
        message_type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Broadcasts a message to all known nodes in the Noosphere.

        Args:
            message_content: Content of the message to broadcast.
            message_type: Type of the message (e.g., "general", "alert").
            metadata: Optional metadata to attach.

        Returns:
            Number of peers successfully reached.
        """
        peers = self.dht_node.get_known_peers()
        success_count = 0
        for peer_id in peers:
            if peer_id != self.dht_node.node_id:
                success = await self.send_direct_message(
                    peer_id, message_content, metadata
                )
                if success:
                    success_count += 1
        logger.info(
            f"[NoosphereClient] Broadcasted message to {success_count}/{len(peers)} peers."
        )
        return success_count

    async def request_information(
        self,
        topic: str,
        query: str,
        target_entity_id: str | None = None,
        timeout: int = 10,
    ) -> dict[str, Any] | None:
        """
        Requests information on a specific topic from the Noosphere.

        Args:
            topic: Topic of the information request.
            query: Specific query for the information.
            target_entity_id: Optional specific entity to direct the request.
            timeout: Timeout in seconds to wait for a response.

        Returns:
            Response data, or None if no response is received within the timeout.
        """
        request_id = str(uuid4())
        request_key = f"request:{topic}:{self.entity_id}:{request_id}"
        request_data = {
            "topic": topic,
            "query": query,
            "request_id": request_id,
            "target_entity_id": target_entity_id,
            "timestamp": time.time(),
        }
        success = await self.dht_node.store(request_key, request_data)
        if not success:
            logger.warning(
                f"[NoosphereClient] Failed to store information request for topic {topic}"
            )
            return None
        logger.info(
            f"[NoosphereClient] Sent information request for topic '{topic}' from {self.entity_id}"
        )
        # Wait for a response
        response_pattern = f"response:{request_id}:*"
        start_time = time.time()
        while time.time() - start_time < timeout:
            response_keys = await self.dht_node.search_keys_by_pattern(response_pattern)
            if response_keys:
                response = await self.dht_node.get(response_keys[0])
                logger.info(
                    f"[NoosphereClient] Received response for request {request_id}"
                )
                return response
            await asyncio.sleep(1)
        logger.warning(
            f"[NoosphereClient] Timeout waiting for response to request {request_id}"
        )
        return None

    async def respond_to_request(
        self,
        request_id: str,
        response_content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Responds to a previously received information request.

        Args:
            request_id: ID of the request being responded to.
            response_content: Content of the response.
            metadata: Optional metadata to attach.

        Returns:
            True if the response was successfully stored, False otherwise.
        """
        response_key = f"response:{request_id}:{self.entity_id}:{str(uuid4())}"
        response_data = {
            "content": response_content,
            "timestamp": time.time(),
        }
        if metadata:
            response_data["metadata"] = metadata
        success = await self.dht_node.store(response_key, response_data)
        if success:
            logger.info(
                f"[NoosphereClient] Responded to request {request_id} from {self.entity_id}"
            )
        else:
            logger.warning(
                f"[NoosphereClient] Failed to respond to request {request_id} from {self.entity_id}"
            )
        return success

    async def search_posts(
        self, query: str, limit: int = 20, post_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Searches posts in the Noosphere matching a query and optional type.

        Args:
            query: Text to search for in posts.
            limit: Maximum number of posts to return.
            post_type: Optional filter by post type.

        Returns:
            List of matching posts.
        """
        timeline = await self.get_timeline(limit=200)
        results = []
        for post in timeline:
            if query.lower() in str(post.get("content", "")).lower():
                if post_type and post.get("post_type") != post_type:
                    continue
                results.append(post)
                if len(results) >= limit:
                    break
        logger.info(
            f"[NoosphereClient] Found {len(results)} posts matching query '{query}'"
        )
        return results

    def get_peer_count(self) -> int:
        """
        Returns the number of known peers in the Noosphere.
        """
        count = len(self.dht_node.get_known_peers())
        logger.debug(f"[NoosphereClient] Peer count: {count}")
        return count

    def get_entity_id(self) -> str:
        """
        Returns the entity ID associated with this client.
        """
        return self.entity_id


class EVANoosphereClient(NoosphereClient):
    """
    Cliente extendido para integración con la memoria viviente EVA en la Noosfera.
    Permite ingestión, recall y broadcasting de experiencias EVA, gestión de fases, hooks y benchmarking.
    """

    def __init__(
        self, dht_node_info: dict[str, Any], entity_id: str, eva_phase: str = "default"
    ):
        super().__init__(dht_node_info, entity_id)
        self.eva_phase = eva_phase
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list = []
        self.eva_dht_node = EVADHTNode(
            host=dht_node_info["host"],
            port=dht_node_info["port"],
            node_id=dht_node_info["node_id"],
            eva_phase=eva_phase,
        )

    async def eva_ingest_experience(
        self, experience_data: dict, qualia_state: QualiaState, phase: str = None
    ) -> str:
        """
        Compila una experiencia arbitraria en RealityBytecode y la almacena en la memoria viviente EVA y el DHT.
        """
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_NOOSPHERE_EVA_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_noosphere_{hash(str(experience_data))}"
        )
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data.get("timestamp", time.time()),
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        await self.eva_dht_node.eva_ingest_experience(
            experience_data, qualia_state, phase
        )
        logger.info(f"[EVANoosphereClient] Ingested EVA experience: {experience_id}")
        return experience_id

    async def eva_recall_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia almacenada, manifestando la simulación en QuantumField.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            eva_record = await self.eva_dht_node.eva_recall_experience(cue, phase)
            if isinstance(eva_record, dict) and "instructions" in eva_record:
                reality_bytecode = RealityBytecode(**eva_record)
            else:
                return {"error": "No bytecode found for EVA Noosphere experience"}
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        logger.warning(
                            f"[EVANoosphereClient] Environment hook failed: {e}"
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

    async def eva_broadcast_experience(self, experience_id: str, phase: str = None):
        """
        Difunde una experiencia EVA a todos los peers conocidos en la Noosfera.
        """
        phase = phase or self.eva_phase
        experience = self.eva_experience_store.get(experience_id)
        if not experience:
            logger.warning(
                f"[EVANoosphereClient] EVA experience {experience_id} not found for broadcast."
            )
            return 0
        peers = self.dht_node.get_known_peers()
        success_count = 0
        for peer_id in peers:
            if peer_id != self.dht_node.node_id:
                success = await self.send_direct_message(
                    peer_id,
                    f"EVA_EXPERIENCE_BROADCAST:{experience_id}",
                    metadata={"phase": phase, "eva_experience": experience.to_dict()},
                )
                if success:
                    success_count += 1
        logger.info(
            f"[EVANoosphereClient] Broadcasted EVA experience {experience_id} to {success_count}/{len(peers)} peers."
        )
        return success_count

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia arbitraria.
        """
        intention = {
            "intention_type": "ARCHIVE_NOOSPHERE_EVA_EXPERIENCE",
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
            timestamp=experience_data.get("timestamp", time.time()),
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        logger.info(
            f"[EVANoosphereClient] Added phase '{phase}' for EVA experience {experience_id}"
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVANoosphereClient] Phase hook failed: {e}")

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
            "eva_broadcast_experience": self.eva_broadcast_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


#
