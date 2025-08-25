import logging
import sqlite3
from collections.abc import Callable
from typing import Any

import redis

from crisalida_lib.AKASHA.websocket_manager import DEFAULT_DB_NAME, DEFAULT_HOST

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Gestor avanzado de almacenamiento para posts, patrones culturales y caché distribuida.
    Integra SQLite para persistencia local y Redis para cacheo rápido y sincronización P2P.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_NAME,
        p2p_node: Any | None = None,
        redis_host: str = DEFAULT_HOST,
        redis_port: int = 6379,
        use_redis_cache: bool = True,
        tidb_host: str = DEFAULT_HOST,
        tidb_port: int = 4000,
        tidb_user: str = "root",
        tidb_password: str = "",
        tidb_db: str = "noosphere_db",
    ):
        self.db_path = db_path
        self.p2p_node = p2p_node
        self.use_redis_cache = use_redis_cache

        # Inicializar Redis
        if self.use_redis_cache:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, db=0, decode_responses=True
                )
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except redis.exceptions.ConnectionError as e:
                logger.warning(
                    f"Could not connect to Redis: {e}. Caching will be disabled."
                )
                self.use_redis_cache = False
                self.redis_client = None
        else:
            self.redis_client = None

        # Inicializar SQLite (TiDB/MySQL pendiente de integración)
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        """Inicializa la base de datos SQLite y crea tablas si no existen."""
        try:
            self.conn: sqlite3.Connection | None = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            # Crear tabla de posts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    author_id TEXT,
                    content TEXT,
                    timestamp REAL,
                    post_type TEXT,
                    parent_id TEXT,
                    reality_level REAL,
                    qualia_state_json TEXT
                )
                """
            )
            # Crear tabla de patrones culturales
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cultural_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    data TEXT,
                    detected_at REAL
                )
                """
            )
            self.conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn = None

    def create_post(
        self,
        post_id: str,
        author_id: str,
        content: str,
        timestamp: float,
        post_type: str,
        parent_id: str | None = None,
        reality_level: float = 1.0,
        qualia_state_json: str | None = None,
    ) -> None:
        """Crea un nuevo post en la base de datos y lo cachea en Redis."""
        if not self.conn:
            logger.error("Database connection not available.")
            return
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO posts (id, author_id, content, timestamp, post_type, parent_id, reality_level, qualia_state_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        author_id,
                        content,
                        timestamp,
                        post_type,
                        parent_id,
                        reality_level,
                        qualia_state_json,
                    ),
                )
            logger.info(f"Post {post_id} created by {author_id}.")
            # Cachear en Redis
            if self.use_redis_cache and self.redis_client:
                post_key = f"post:{post_id}"
                post_data = {
                    "id": post_id,
                    "author_id": author_id,
                    "content": content,
                    "timestamp": timestamp,
                    "post_type": post_type,
                    "parent_id": parent_id or "",
                    "reality_level": reality_level,
                    "qualia_state_json": qualia_state_json or "",
                }
                self.redis_client.hset(post_key, mapping=post_data)
                self.redis_client.zadd("timeline", {post_id: timestamp})
                self.redis_client.zremrangebyrank("timeline", 0, -1001)
        except sqlite3.IntegrityError:
            logger.warning(f"Post with ID {post_id} already exists.")
        except Exception as e:
            logger.error(f"Failed to create post {post_id}: {e}")

    def get_post_by_id(self, post_id: str) -> dict[str, Any] | None:
        """Recupera un post por su ID, primero desde cache y luego desde la base de datos."""
        # Intentar cache Redis
        if self.use_redis_cache and self.redis_client:
            post_key = f"post:{post_id}"
            cached_post = self.redis_client.hgetall(post_key)
            if cached_post:
                cached_post["timestamp"] = float(cached_post["timestamp"])
                cached_post["reality_level"] = float(cached_post["reality_level"])
                return cached_post
        # Fallback a base de datos
        if not self.conn:
            logger.error("Database connection not available.")
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, author_id, content, timestamp, post_type, parent_id, reality_level, qualia_state_json FROM posts WHERE id = ?",
                (post_id,),
            )
            row = cursor.fetchone()
            if row:
                post = dict(row)
                # Cachear resultado
                if self.use_redis_cache and self.redis_client:
                    post_key = f"post:{post_id}"
                    post_data_for_cache = {k: str(v) for k, v in post.items()}
                    self.redis_client.hset(post_key, mapping=post_data_for_cache)
                return post
        except Exception as e:
            logger.error(f"Failed to retrieve post {post_id}: {e}")
        return None

    def get_recent_posts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recupera los posts más recientes, usando cache si está disponible."""
        posts = []
        # Intentar cache Redis
        if self.use_redis_cache and self.redis_client:
            try:
                post_ids = self.redis_client.zrevrange("timeline", 0, limit - 1)
                for post_id in post_ids:
                    post = self.get_post_by_id(post_id)
                    if post:
                        posts.append(post)
                if posts:
                    return posts
            except redis.exceptions.RedisError as e:
                logger.warning(f"Redis error when fetching recent posts: {e}")
        # Fallback a base de datos
        if not self.conn:
            logger.error("Database connection not available.")
            return []
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, author_id, content, timestamp, post_type, parent_id, reality_level, qualia_state_json FROM posts ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve recent posts: {e}")
            return []

    def store_cultural_pattern(
        self, pattern_id: str, pattern_type: str, data: str, detected_at: float
    ) -> None:
        """Almacena un patrón cultural detectado en la base de datos."""
        if not self.conn:
            logger.error("Database connection not available.")
            return
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO cultural_patterns (pattern_id, pattern_type, data, detected_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (pattern_id, pattern_type, data, detected_at),
                )
            logger.info(f"Stored cultural pattern: {pattern_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Pattern with ID {pattern_id} already exists.")
        except Exception as e:
            logger.error(f"Failed to store cultural pattern {pattern_id}: {e}")

    def get_all_posts_for_analysis(self) -> list[tuple[str, str]]:
        """Recupera todos los posts para análisis cultural."""
        if not self.conn:
            logger.error("Database connection not available.")
            return []
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, content FROM posts")
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve all posts for analysis: {e}")
            return []

    def save_universe_state(self, serialized_state: bytes) -> None:
        """Saves the complete state of the universe."""
        if not self.conn:
            logger.error("Database connection not available.")
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO posts (id, content, post_type) VALUES (?, ?, ?)",
                    ("universe_state", serialized_state, "universe_state"),
                )
            logger.info("Universe state saved.")
        except Exception as e:
            logger.error(f"Failed to save universe state: {e}")

    def load_universe_state(self) -> bytes | None:
        """Loads the complete state of the universe."""
        if not self.conn:
            logger.error("Database connection not available.")
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT content FROM posts WHERE id = ?", ("universe_state",)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        except Exception as e:
            logger.error(f"Failed to load universe state: {e}")
        return None

    def save_being_state(self, entity_id: str, serialized_state: bytes) -> None:
        """Saves the state of a single ConsciousBeing."""
        if not self.conn:
            logger.error("Database connection not available.")
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO posts (id, content, post_type) VALUES (?, ?, ?)",
                    (entity_id, serialized_state, "being_state"),
                )
            logger.info(f"State for being {entity_id} saved.")
        except Exception as e:
            logger.error(f"Failed to save state for being {entity_id}: {e}")

    def load_being_state(self, entity_id: str) -> bytes | None:
        """Loads the state of a single ConsciousBeing."""
        if not self.conn:
            logger.error("Database connection not available.")
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT content FROM posts WHERE id = ?", (entity_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
        except Exception as e:
            logger.error(f"Failed to load state for being {entity_id}: {e}")
        return None

    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")


class EVAStorageManager(StorageManager):
    """
    Gestor EVA extendido para almacenamiento de experiencias vivientes, patrones culturales y memoria distribuida.
    Integra SQLite y Redis, soporta ingestión/recall de experiencias EVA, faseo, hooks de entorno y benchmarking.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_NAME,
        p2p_node: Any | None = None,
        redis_host: str = DEFAULT_HOST,
        redis_port: int = 6379,
        use_redis_cache: bool = True,
        tidb_host: str = DEFAULT_HOST,
        tidb_port: int = 4000,
        tidb_user: str = "root",
        tidb_password: str = "",
        tidb_db: str = "noosphere_db",
        eva_phase: str = "default",
    ):
        super().__init__(
            db_path=db_path,
            p2p_node=p2p_node,
            redis_host=redis_host,
            redis_port=redis_port,
            use_redis_cache=use_redis_cache,
            tidb_host=tidb_host,
            tidb_port=tidb_port,
            tidb_user=tidb_user,
            tidb_password=tidb_password,
            tidb_db=tidb_db,
        )
        self.eva_phase = eva_phase
        self.eva_memory_store: dict[str, Any] = {}
        self.eva_experience_store: dict[str, Any] = {}
        self.eva_phases: dict[str, dict[str, Any]] = {}
        self._environment_hooks: list = []

    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: dict, phase: str = None
    ) -> str:
        """
        Compila y almacena una experiencia viviente EVA en la base de datos y cache distribuida.
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
            "timestamp": experience_data.get("timestamp", 0.0),
        }
        self.eva_memory_store[experience_id] = eva_record
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        # Persistir en SQLite como post especial
        self.create_post(
            post_id=experience_id,
            author_id=experience_data.get("author_id", "eva"),
            content=str(experience_data),
            timestamp=eva_record["timestamp"],
            post_type="eva_experience",
            parent_id=None,
            reality_level=experience_data.get("reality_level", 1.0),
            qualia_state_json=str(qualia_state),
        )
        # Cachear en Redis
        if self.use_redis_cache and self.redis_client:
            self.redis_client.hset(f"eva_exp:{experience_id}", mapping=eva_record)
            self.redis_client.zadd(
                "eva_timeline", {experience_id: eva_record["timestamp"]}
            )
            self.redis_client.zremrangebyrank("eva_timeline", 0, -1001)
        logger.info(f"[EVA-STORAGE] Ingested EVA experience: {experience_id}")
        return experience_id

    def eva_recall_experience(self, cue: str, phase: str = None) -> dict:
        """
        Recupera una experiencia viviente EVA por ID y fase, desde cache o base de datos.
        """
        phase = phase or self.eva_phase
        # Intentar cache Redis
        if self.use_redis_cache and self.redis_client:
            eva_key = f"eva_exp:{cue}"
            cached_exp = self.redis_client.hgetall(eva_key)
            if cached_exp:
                return cached_exp
        # Fallback a memoria local
        eva_record = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if eva_record:
            return eva_record
        # Fallback a base de datos
        post = self.get_post_by_id(cue)
        if post and post.get("post_type") == "eva_experience":
            return {
                "experience_id": post["id"],
                "experience_data": post["content"],
                "qualia_state": post.get("qualia_state_json", ""),
                "phase": phase,
                "timestamp": post["timestamp"],
            }
        logger.warning(
            f"[EVA-STORAGE] EVA experience {cue} not found in phase '{phase}'"
        )
        return {"error": "No EVA experience found"}

    def add_experience_phase(
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
            "timestamp": experience_data.get("timestamp", 0.0),
        }
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = eva_record
        logger.info(
            f"[EVA-STORAGE] Added phase '{phase}' for EVA experience {experience_id}"
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logger.warning(f"[EVA-STORAGE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases de una experiencia EVA."""
        return [
            {"phase": phase, "data": data}
            for phase, data in self.eva_phases.items()
            if data.get(experience_id)
        ]

    def register_environment_hook(self, hook: Callable[..., Any]):
        """
        Registra un hook de entorno que será llamado en cambios de fase.
        """
        if hook not in self._environment_hooks:
            self._environment_hooks.append(hook)
            logger.info(f"[EVA-STORAGE] Registered new environment hook: {hook}")

    def unregister_environment_hook(self, hook: Callable[..., Any]):
        """
        Elimina un hook de entorno registrado.
        """
        if hook in self._environment_hooks:
            self._environment_hooks.remove(hook)
            logger.info(f"[EVA-STORAGE] Unregistered environment hook: {hook}")

    def benchmark_storage_performance(self, iterations: int = 100):
        """
        Benchmark para evaluar el rendimiento de almacenamiento y recuperación.
        """
        import time

        # Ingesta
        start_time = time.time()
        for i in range(iterations):
            self.eva_ingest_experience(
                {
                    "experience_id": f"bench_exp_{i}",
                    "content": "Benchmarking EVA storage",
                },
                {"state": "benchmark"},
                phase="benchmark",
            )
        ingestion_time = time.time() - start_time
        logger.info(
            f"[EVA-STORAGE] Ingestion benchmark completed in {ingestion_time:.4f} seconds"
        )

        # Recall
        start_time = time.time()
        for i in range(iterations):
            self.eva_recall_experience(f"bench_exp_{i}", phase="benchmark")
        recall_time = time.time() - start_time
        logger.info(
            f"[EVA-STORAGE] Recall benchmark completed in {recall_time:.4f} seconds"
        )
