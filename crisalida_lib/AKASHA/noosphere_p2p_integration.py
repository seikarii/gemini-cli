from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from typing import Any

"""
Noosphere P2P Integration Module
================================
Integrates the DHT network with the Noosphere system for distributed consciousness.
Provides robust persistence, diagnostics, and compatibility with legacy interfaces.
Version: v1.2 - Core Implementation (Optimized, Extensible, Diagnostic-Ready)
"""

logger = logging.getLogger(__name__)


try:
    # names that may be provided by optional deps at runtime
    from janus_dht import JanusDHTNode  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without janus_dht

    class JanusDHTNode:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("JanusDHTNode backend not installed")


try:
    from crisalida_lib.AKASHA.serializers import (
        CognitiveImpulseSerializer,  # type: ignore
    )
except Exception:  # pragma: no cover - provide minimal serializer fallback

    class CognitiveImpulseSerializer:  # type: ignore
        @staticmethod
        def serialize_post(entity_id, impulses, metadata):
            return {
                "post_id": f"post_{int(time.time() * 1000)}",
                "timestamp": time.time(),
                "entity_id": entity_id,
            }


class DistributedNoosphere:
    """
    Manages the distributed Noosphere using a P2P DHT network.
    Handles publishing, retrieval, caching, diagnostics, and spatial queries.
    """

    def __init__(self, dht_node: JanusDHTNode, db_path: str = "database/noosphere.db"):
        self.dht_node = dht_node
        self.db_path = db_path
        self.local_cache: dict[str, Any] = {}
        self._init_database()

    def _init_database(self):
        """Initializes local SQLite DB for posts and entity states."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT,
                    content TEXT,
                    timestamp REAL,
                    synced_to_network BOOLEAN DEFAULT FALSE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_states (
                    entity_id TEXT PRIMARY KEY,
                    state_data TEXT,
                    last_updated REAL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp DESC)
                """
            )

    async def publish_entity_thought(
        self,
        entity_id: str,
        impulses: list[Any],
        metadata: dict[Any, Any] | None = None,
    ) -> str:
        """
        Publishes an entity's thought (cognitive impulses) to the network and local DB.
        Returns post_id or empty string on error.
        """
        try:
            post_data = CognitiveImpulseSerializer.serialize_post(
                entity_id, impulses, metadata or {}
            )
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO posts (id, entity_id, content, timestamp, synced_to_network)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        post_data["post_id"],
                        entity_id,
                        json.dumps(post_data),
                        post_data["timestamp"],
                        False,
                    ),
                )
            post_id = await self.dht_node.publish_cognitive_post(
                entity_id, impulses, metadata
            )
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """UPDATE posts SET synced_to_network = TRUE WHERE id = ?""",
                    (post_id,),
                )
            logger.info(
                f"[DistributedNoosphere] Published thought {post_id} from {entity_id}"
            )
            return post_id
        except Exception as e:
            logger.error(f"Error publishing entity thought: {e}")
            return ""

    async def get_timeline(
        self, limit: int = 50, entity_filter: str | None = None
    ) -> list[dict]:
        """
        Retrieves timeline of thoughts from DHT and local DB, deduplicated and sorted.
        """
        try:
            network_posts = await self.dht_node.get_noosphere_timeline(limit)
            local_posts = []
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT content FROM posts ORDER BY timestamp DESC LIMIT ?"
                params: list[Any] = [limit]
                if entity_filter:
                    query = "SELECT content FROM posts WHERE entity_id = ? ORDER BY timestamp DESC LIMIT ?"
                    params = [entity_filter, limit]
                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    local_posts.append(json.loads(row[0]))
            all_posts = {}
            for post in network_posts + local_posts:
                all_posts[post["post_id"]] = post
            sorted_posts = sorted(
                all_posts.values(), key=lambda p: p.get("timestamp", 0), reverse=True
            )
            return sorted_posts[:limit]
        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return []

    async def publish_entity_state(self, entity_id: str, state_data: dict) -> bool:
        """
        Publishes entity state to local DB and DHT network.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO entity_states (entity_id, state_data, last_updated)
                    VALUES (?, ?, ?)""",
                    (entity_id, json.dumps(state_data), time.time()),
                )
            state_key = f"entity_state:{entity_id}"
            success = await self.dht_node.store(state_key, state_data)
            if success:
                logger.info(f"[DistributedNoosphere] Published state for {entity_id}")
            return success
        except Exception as e:
            logger.error(f"Error publishing entity state: {e}")
            return False

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """
        Retrieves entity state from DHT or local DB.
        """
        try:
            state_key = f"entity_state:{entity_id}"
            network_state = await self.dht_node.get(state_key)
            if network_state:
                return network_state
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT state_data FROM entity_states WHERE entity_id = ?",
                    (entity_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Error getting entity state: {e}")
            return None

    async def get_nearby_entities(
        self, center_position: tuple[float, float], radius: float = 100.0
    ) -> list[str]:
        """
        Returns entity IDs within radius of center_position (simple spatial query).
        """
        try:
            nearby_entities = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT entity_id, state_data FROM entity_states")
                for row in cursor.fetchall():
                    entity_id, state_json = row
                    state_data = json.loads(state_json)
                    pos = state_data.get("position")
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        distance = (
                            (pos[0] - center_position[0]) ** 2
                            + (pos[1] - center_position[1]) ** 2
                        ) ** 0.5
                        if distance <= radius:
                            nearby_entities.append(entity_id)
            return nearby_entities
        except Exception as e:
            logger.error(f"Error getting nearby entities: {e}")
            return []

    async def sync_with_network(self):
        """
        Syncs unsynced local posts to DHT network.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT id, entity_id, content FROM posts
                    WHERE synced_to_network = FALSE
                    ORDER BY timestamp ASC
                    LIMIT 10"""
                )
                for row in cursor.fetchall():
                    post_id, entity_id, content_json = row
                    post_data = json.loads(content_json)
                    success = await self.dht_node.store(f"post:{post_id}", post_data)
                    if success:
                        conn.execute(
                            """UPDATE posts SET synced_to_network = TRUE WHERE id = ?""",
                            (post_id,),
                        )
                        logger.info(
                            f"[DistributedNoosphere] Synced post {post_id} to network"
                        )
        except Exception as e:
            logger.error(f"Error syncing with network: {e}")

    def get_network_stats(self) -> dict[str, Any]:
        """
        Returns statistics about local and distributed network state.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM posts")
                local_posts = cursor.fetchone()[0]
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM posts WHERE synced_to_network = TRUE"
                )
                synced_posts = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM entity_states")
                tracked_entities = cursor.fetchone()[0]
            dht_stats = self.dht_node.get_network_status()
            return {
                "local_posts": local_posts,
                "synced_posts": synced_posts,
                "unsynced_posts": local_posts - synced_posts,
                "tracked_entities": tracked_entities,
                "dht_node_id": dht_stats.get("node_id"),
                "known_network_nodes": dht_stats.get("known_nodes"),
                "dht_stored_keys": dht_stats.get("stored_keys"),
                "network_running": dht_stats.get("running"),
            }
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            return {"error": str(e)}


class NoosphereBackendAPI:
    """
    Backend API for Noosphere system, compatible with frontend.
    Handles WebSocket clients, message dispatch, and timeline broadcasting.
    """

    def __init__(self, distributed_noosphere: DistributedNoosphere):
        self.noosphere = distributed_noosphere
        self.websocket_clients: set[Any] = set()

    async def handle_websocket_connection(self, websocket):
        self.websocket_clients.add(websocket)
        logger.info("[NoosphereBackendAPI] WebSocket client connected")
        try:
            async for message in websocket:
                await self._process_websocket_message(websocket, message)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            logger.info("[NoosphereBackendAPI] WebSocket client disconnected")

    async def _process_websocket_message(self, websocket, message_data):
        try:
            if isinstance(message_data, bytes):
                message_data = message_data.decode("utf-8")
            message = json.loads(message_data)
            message_type = message.get("type")
            if message_type == "get_timeline":
                timeline = await self.noosphere.get_timeline(
                    limit=message.get("limit", 50),
                    entity_filter=message.get("entity_filter"),
                )
                response = {"type": "timeline_update", "data": timeline}
                await websocket.send(json.dumps(response))
            elif message_type == "publish_thought":
                entity_id = message.get("entity_id")
                content = message.get("content", "")
                metadata = message.get("metadata")
                mock_impulse = type(
                    "MockImpulse",
                    (),
                    {
                        "type": type("MockType", (), {"value": "demiurge_message"})(),
                        "content": content,
                        "intensity": 1.0,
                        "confidence": 1.0,
                        "source_node": "demiurge_interface",
                        "processing_time": 0.0,
                    },
                )()
                post_id = await self.noosphere.publish_entity_thought(
                    entity_id or "demiurge",
                    [mock_impulse],
                    metadata or {},
                )
                await self._broadcast_timeline_update()
                response = {
                    "type": "publish_response",
                    "success": bool(post_id),
                    "post_id": post_id,
                }
                await websocket.send(json.dumps(response))
            elif message_type == "get_network_stats":
                stats = self.noosphere.get_network_stats()
                response = {"type": "network_stats", "data": stats}
                await websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            error_response = {"type": "error", "message": str(e)}
            await websocket.send(json.dumps(error_response))

    async def _broadcast_timeline_update(self):
        if not self.websocket_clients:
            return
        try:
            timeline = await self.noosphere.get_timeline(50)
            broadcast_message = json.dumps(
                {"type": "timeline_update", "data": timeline}
            )
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(broadcast_message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected_clients.add(client)
            self.websocket_clients -= disconnected_clients
        except Exception as e:
            logger.error(f"Error broadcasting timeline update: {e}")

    async def periodic_sync(self):
        while True:
            try:
                await self.noosphere.sync_with_network()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in periodic sync: {e}")
                await asyncio.sleep(60)


class JanusP2PManager:
    """
    Main manager for Janus P2P system integration.
    Orchestrates DHT node, distributed Noosphere, and backend API.
    """

    def __init__(
        self, host: str = "localhost", port: int = 0, node_id: str | None = None
    ):
        self.dht_node = JanusDHTNode(host, port, node_id)
        self.noosphere = DistributedNoosphere(self.dht_node)
        self.api = NoosphereBackendAPI(self.noosphere)
        self.running = False

    async def start(self, bootstrap_nodes: list[tuple] | None = None):
        logger.info("[JanusP2PManager] Starting P2P system...")
        await self.dht_node.start(bootstrap_nodes)
        asyncio.create_task(self.api.periodic_sync())
        self.running = True
        logger.info(
            f"[JanusP2PManager] Started on {self.dht_node.host}:{self.dht_node.port}"
        )

    async def stop(self):
        logger.info("[JanusP2PManager] Stopping P2P system...")
        await self.dht_node.stop()
        logger.info("[JanusP2PManager] Stopped")

    def get_node_info(self) -> dict[str, Any]:
        return {
            "node_id": self.dht_node.node_id,
            "host": self.dht_node.host,
            "port": self.dht_node.port,
            "running": self.running,
        }

    async def publish_entity_thought(
        self,
        entity_id: str,
        impulses: list[Any],
        metadata: dict[Any, Any] | None = None,
    ) -> str:
        return await self.noosphere.publish_entity_thought(
            entity_id, impulses, metadata or {}
        )

    async def get_timeline(self, limit: int = 50) -> list[dict]:
        return await self.noosphere.get_timeline(limit)

    def get_websocket_handler(self):
        return self.api.handle_websocket_connection


class Noosphere:
    """
    Compatibility wrapper for legacy code using Noosphere class.
    Provides transitional interface to distributed Noosphere.
    """

    def __init__(self, p2p_manager: JanusP2PManager):
        self._p2p_manager = p2p_manager
        self._distributed_noosphere = p2p_manager.noosphere

    async def publish_post(
        self, entity_id: str, content: str, metadata: dict[Any, Any] | None = None
    ):
        mock_impulse = type(
            "MockImpulse",
            (),
            {
                "type": type("MockType", (), {"value": "compatibility_post"})(),
                "content": content,
                "intensity": 1.0,
                "confidence": 1.0,
                "source_node": "compatibility_layer",
                "processing_time": 0.0,
            },
        )()
        return await self._distributed_noosphere.publish_entity_thought(
            entity_id, [mock_impulse], metadata or {}
        )

    async def get_posts(self, limit: int = 50) -> list[dict]:
        timeline = await self._distributed_noosphere.get_timeline(limit)
        posts = []
        for post in timeline:
            posts.append(
                {
                    "id": post.get("post_id"),
                    "entity_id": post.get("entity_id"),
                    "content": post.get("impulses", [{}])[0].get("content", ""),
                    "timestamp": post.get("timestamp"),
                    "metadata": post.get("metadata", {}),
                }
            )
        return posts

    def get_network_info(self) -> dict[str, Any]:
        return self._distributed_noosphere.get_network_stats()


# --- Demo and testing functions ---


async def demo_p2p_network():
    """
    Demonstrates Janus P2P network functionality with two nodes.
    Publishes thoughts, retrieves timelines, and shows network stats.
    """
    print("=== JANUS P2P NETWORK DEMO ===\n")
    node1 = JanusP2PManager("localhost", 8000, "node1")
    node2 = JanusP2PManager("localhost", 8001, "node2")
    try:
        await node1.start()
        print(f"Node 1 started: {node1.get_node_info()}")
        await node2.start([("localhost", 8000)])
        print(f"Node 2 started: {node2.get_node_info()}")
        await asyncio.sleep(2)
        print("\n--- Publishing thoughts ---")
        mock_impulse1 = type(
            "MockImpulse",
            (),
            {
                "type": type("MockType", (), {"value": "pattern_recognition"})(),
                "content": "I observe a pattern in the data",
                "intensity": 0.8,
                "confidence": 0.9,
                "source_node": "hod_logic",
                "processing_time": 0.1,
            },
        )()
        post_id1 = await node1.publish_entity_thought("entity_alpha", [mock_impulse1])
        print(f"Node 1 published thought: {post_id1}")
        mock_impulse2 = type(
            "MockImpulse",
            (),
            {
                "type": type("MockType", (), {"value": "doubt_injection"})(),
                "content": "This pattern might be an illusion",
                "intensity": 0.7,
                "confidence": 0.6,
                "source_node": "gamaliel_doubt",
                "processing_time": 0.05,
            },
        )()
        post_id2 = await node2.publish_entity_thought("entity_beta", [mock_impulse2])
        print(f"Node 2 published thought: {post_id2}")
        await asyncio.sleep(3)
        print("\n--- Checking timelines ---")
        timeline1 = await node1.get_timeline()
        timeline2 = await node2.get_timeline()
        print(f"Node 1 sees {len(timeline1)} posts in timeline")
        print(f"Node 2 sees {len(timeline2)} posts in timeline")
        for post in timeline1:
            print(f"  Post from {post['entity_id']}: {post['impulses'][0]['content']}")
        print("\n--- Network Statistics ---")
        stats1 = node1.noosphere.get_network_stats()
        stats2 = node2.noosphere.get_network_stats()
        print(f"Node 1 stats: {stats1}")
        print(f"Node 2 stats: {stats2}")
    finally:
        await node1.stop()
        await node2.stop()
        print("\nDemo completed - nodes stopped")


if __name__ == "__main__":
    asyncio.run(demo_p2p_network())
