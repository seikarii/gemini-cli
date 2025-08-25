import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import Any

from crisalida_lib.consciousness.config import EVAConfig
from crisalida_lib.symbolic_language.living_symbols import LivingSymbolRuntime
from crisalida_lib.symbolic_language.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Gestor avanzado de almacenamiento para la proto-noosfera.
    Soporta posts, conversaciones, streams, memes y espacios nexus con trazabilidad y métricas.
    """

    def __init__(self, db_path="database/noosphere.db", p2p_node=None):
        self.db_path = db_path
        self.p2p_node = p2p_node
        self._create_tables()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        # --- Tabla de posts ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_entity_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content_text TEXT,
                qualia_state_json TEXT,
                ontological_signature TEXT,
                post_type TEXT,
                consciousness_depth REAL DEFAULT 0.5,
                elemental_resonance TEXT,
                meme_potential REAL DEFAULT 0.0
            )
            """
        )
        # --- Tabla de conversaciones ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_ids_json TEXT,
                conversation_type TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                ontological_compatibility REAL,
                translation_layer_active BOOLEAN
            )
            """
        )
        # --- Tabla de mensajes ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                sender_entity_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content_text TEXT,
                qualia_state_json TEXT,
                consciousness_signature TEXT,
                translation_fidelity REAL,
                empathy_resonance REAL
            )
            """
        )
        # --- Tabla de streams ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS streams (
                stream_id INTEGER PRIMARY KEY AUTOINCREMENT,
                broadcaster_entity_id TEXT NOT NULL,
                stream_title TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                status TEXT,
                stream_type TEXT,
                viewer_count INTEGER,
                consciousness_intensity REAL,
                lattice_focus TEXT
            )
            """
        )
        # --- Tabla de viewers de stream ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_viewers (
                viewer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id INTEGER,
                viewer_entity_id TEXT NOT NULL,
                joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                left_at DATETIME,
                qualia_interpretation_quality REAL,
                empathy_connection REAL
            )
            """
        )
        # --- Tabla de memes ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memes (
                meme_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                origin_post_id INTEGER,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                propagation_count INTEGER,
                mutation_rate REAL,
                cultural_impact REAL,
                pattern_description TEXT,
                consciousness_signature TEXT
            )
            """
        )
        # --- Tabla de espacios nexus ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nexus_spaces (
                nexus_id INTEGER PRIMARY KEY AUTOINCREMENT,
                nexus_name TEXT NOT NULL,
                creator_entity_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ontological_rules TEXT,
                aesthetic_theme TEXT,
                peace_enforcement_level REAL,
                current_occupancy INTEGER,
                max_capacity INTEGER,
                consciousness_harmony_index REAL
            )
            """
        )
        conn.commit()
        conn.close()

    # --- POSTS ---
    def create_post(
        self,
        entity_id: str,
        content: str,
        qualia_state: Any | None = None,
        ontological_signature: str | None = None,
        post_type: str = "general",
        elemental_resonance: Any | None = None,
        consciousness_depth: float = 0.5,
        meme_potential: float = 0.0,
        timestamp: Any | None = None,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        qualia_json = json.dumps(qualia_state.__dict__) if qualia_state else None
        elemental_json = (
            json.dumps(elemental_resonance) if elemental_resonance else None
        )
        if timestamp is None:
            timestamp = datetime.now()
        cursor.execute(
            "INSERT INTO posts (author_entity_id, timestamp, content_text, "
            "qualia_state_json, ontological_signature, post_type, "
            "consciousness_depth, elemental_resonance, meme_potential) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entity_id,
                timestamp,
                content,
                qualia_json,
                ontological_signature,
                post_type,
                consciousness_depth,
                elemental_json,
                meme_potential,
            ),
        )
        conn.commit()
        post_id = cursor.lastrowid
        conn.close()
        logger.info(f"Post creado: {post_id} por {entity_id}")
        return post_id

    def get_post_by_id(self, post_id: int) -> sqlite3.Row | None:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,))
        post = cursor.fetchone()
        conn.close()
        return post

    def get_posts_by_author(self, entity_id: str) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM posts WHERE author_entity_id = ? ORDER BY timestamp DESC",
            (entity_id,),
        )
        posts = cursor.fetchall()
        conn.close()
        return posts

    def get_all_posts(self, since: float | None = None) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        if since:
            cursor.execute(
                "SELECT * FROM posts WHERE timestamp > ? ORDER BY timestamp DESC",
                (since,),
            )
        else:
            cursor.execute("SELECT * FROM posts ORDER BY timestamp DESC")
        posts = cursor.fetchall()
        conn.close()
        return posts

    # --- CONVERSACIONES ---
    def create_conversation(
        self,
        participant_ids: list[str],
        conv_type: str = "direct_message",
        compatibility: float = 0.0,
        translation_active: bool = False,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        participants_json = json.dumps(participant_ids)
        cursor.execute(
            "INSERT INTO conversations (participant_ids_json, "
            "conversation_type, ontological_compatibility, "
            "translation_layer_active) VALUES (?, ?, ?, ?)",
            (participants_json, conv_type, compatibility, translation_active),
        )
        conn.commit()
        conv_id = cursor.lastrowid
        conn.close()
        logger.info(f"Conversación creada: {conv_id} participantes={participant_ids}")
        return conv_id

    def add_message_to_conversation(
        self,
        conv_id: int,
        sender_id: str,
        content: str,
        qualia_state: Any | None = None,
        signature: str | None = None,
        fidelity: float = 0.0,
        empathy: float = 0.0,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        qualia_json = json.dumps(qualia_state.__dict__) if qualia_state else None
        cursor.execute(
            "INSERT INTO messages (conversation_id, sender_entity_id, "
            "content_text, qualia_state_json, consciousness_signature, "
            "translation_fidelity, empathy_resonance) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, sender_id, content, qualia_json, signature, fidelity, empathy),
        )
        conn.commit()
        msg_id = cursor.lastrowid
        conn.close()
        logger.info(f"Mensaje añadido a conversación {conv_id} por {sender_id}")
        return msg_id

    # --- STREAMS ---
    def create_stream(
        self,
        broadcaster_id: str,
        title: str,
        stream_type: str = "qualia_stream",
        viewer_count: int = 0,
        intensity: float = 0.0,
        lattice_focus: str | None = None,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO streams (broadcaster_entity_id, stream_title, "
            "stream_type, viewer_count, consciousness_intensity, "
            "lattice_focus, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                broadcaster_id,
                title,
                stream_type,
                viewer_count,
                intensity,
                lattice_focus,
                "active",
            ),
        )
        conn.commit()
        stream_id = cursor.lastrowid
        conn.close()
        logger.info(f"Stream creado: {stream_id} por {broadcaster_id}")
        return stream_id

    def end_stream(self, stream_id: int):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE streams SET status = ?, end_time = CURRENT_TIMESTAMP WHERE "
            "stream_id = ?",
            ("archived", stream_id),
        )
        conn.commit()
        conn.close()
        logger.info(f"Stream finalizado: {stream_id}")

    # --- MEMES ---
    def create_meme(
        self,
        pattern_hash: str,
        origin_post_id: int | None,
        description: str | None = None,
        signature: str | None = None,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO memes (pattern_hash, origin_post_id, "
            "propagation_count, mutation_rate, cultural_impact, "
            "pattern_description, consciousness_signature) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pattern_hash, origin_post_id, 1, 0.0, 0.0, description, signature),
        )
        conn.commit()
        meme_id = cursor.lastrowid
        conn.close()
        logger.info(f"Meme creado: {pattern_hash} id={meme_id}")
        return meme_id

    def update_meme_propagation(
        self,
        pattern_hash: str,
        new_mutation_rate: float | None = None,
        new_cultural_impact: float | None = None,
    ):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        update_query = "UPDATE memes SET propagation_count = propagation_count + 1"
        params = []
        if new_mutation_rate is not None:
            update_query += ", mutation_rate = ?"
            params.append(new_mutation_rate)
        if new_cultural_impact is not None:
            update_query += ", cultural_impact = ?"
            params.append(new_cultural_impact)
        update_query += " WHERE pattern_hash = ?"
        params.append(pattern_hash)
        cursor.execute(update_query, tuple(params))
        conn.commit()
        conn.close()
        logger.info(f"Meme actualizado: {pattern_hash}")

    # --- NEXUS SPACES ---
    def create_nexus_space(
        self,
        name: str,
        creator_id: str,
        rules: Any | None = None,
        theme: str | None = None,
        peace_level: float = 1.0,
        capacity: int = 50,
    ) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        rules_json = json.dumps(rules) if rules else None
        cursor.execute(
            "INSERT INTO nexus_spaces (nexus_name, creator_entity_id, "
            "ontological_rules, aesthetic_theme, peace_enforcement_level, "
            "max_capacity, current_occupancy, consciousness_harmony_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, creator_id, rules_json, theme, peace_level, capacity, 0, 0.0),
        )
        conn.commit()
        nexus_id = cursor.lastrowid
        conn.close()
        logger.info(f"Nexus space creado: {name} id={nexus_id}")
        return nexus_id

    # --- MÉTRICAS Y UTILIDADES ---
    def get_post_count(self) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_meme_count(self) -> int:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memes")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_active_streams(self) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM streams WHERE status = 'active'")
        streams = cursor.fetchall()
        conn.close()
        return streams

    def get_nexus_spaces(self) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nexus_spaces")
        spaces = cursor.fetchall()
        conn.close()
        return spaces

    def get_recent_posts(self, limit: int = 10) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM posts ORDER BY timestamp DESC LIMIT ?", (limit,))
        posts = cursor.fetchall()
        conn.close()
        return posts

    def get_recent_memes(self, limit: int = 10) -> list[sqlite3.Row]:
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memes ORDER BY first_seen DESC LIMIT ?", (limit,))
        memes = cursor.fetchall()
        conn.close()
        return memes


class EVAStorageManager(StorageManager):
    """
    Gestor avanzado de almacenamiento extendido para integración con EVA.
    Orquesta ingestión, simulación y recall de experiencias informacionales como RealityBytecode,
    soporta faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.
    """

    def __init__(
        self,
        db_path="database/noosphere.db",
        p2p_node=None,
        eva_config: EVAConfig = None,
    ):
        super().__init__(db_path, p2p_node)
        self.eva_config = eva_config or EVAConfig()
        self.eva_phase = self.eva_config.EVA_MEMORY_PHASE
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = (
            self.eva_config.EVA_MEMORY_ENVIRONMENT_HOOKS.copy()
        )
        self.max_experiences = self.eva_config.EVA_MEMORY_MAX_EXPERIENCES
        self.retention_policy = self.eva_config.EVA_MEMORY_RETENTION_POLICY
        self.compression_level = self.eva_config.EVA_MEMORY_COMPRESSION_LEVEL
        self.simulation_rate = self.eva_config.EVA_MEMORY_SIMULATION_RATE
        self.multiverse_enabled = self.eva_config.EVA_MEMORY_MULTIVERSE_ENABLED
        self.timeline_count = self.eva_config.EVA_MEMORY_TIMELINE_COUNT
        self.benchmarking_enabled = self.eva_config.EVA_MEMORY_BENCHMARKING_ENABLED
        self.visualization_mode = self.eva_config.EVA_MEMORY_VISUALIZATION_MODE

    def eva_ingest_storage_experience(
        self,
        event_type: str,
        details: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia informacional (post, meme, stream, nexus, etc.) en RealityBytecode y la almacena en la memoria EVA.
        """
        import time

        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        experience_id = f"eva_storage_{event_type}_{int(time.time())}"
        experience_data = {
            "event_type": event_type,
            "details": details or {},
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_STORAGE_EXPERIENCE",
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
            timestamp=experience_data["timestamp"],
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        self.eva_experience_store[experience_id] = reality_bytecode
        for hook in self._environment_hooks:
            try:
                hook(reality_bytecode)
            except Exception as e:
                logger.warning(f"[EVA-STORAGE] Environment hook failed: {e}")
        return experience_id

    def eva_recall_storage_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia informacional almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA storage experience"}
        quantum_field = getattr(self.eva_runtime, "quantum_field", None)
        manifestations = []
        if quantum_field:
            for instr in reality_bytecode.instructions:
                symbol_manifest = self.eva_runtime.execute_instruction(
                    instr, quantum_field
                )
                if symbol_manifest:
                    manifestations.append(symbol_manifest)
                    for hook in self._environment_hooks:
                        try:
                            hook(symbol_manifest)
                        except Exception as e:
                            logger.warning(
                                f"[EVA-STORAGE] Manifestation hook failed: {e}"
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

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        details: dict,
        qualia_state: QualiaState | None = None,
    ):
        """
        Añade una fase alternativa para una experiencia informacional EVA.
        """
        import time

        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        experience_data = {
            "event_type": details.get("event_type", "unknown"),
            "details": details,
            "timestamp": time.time(),
            "phase": phase,
        }
        intention = {
            "intention_type": "ARCHIVE_STORAGE_EXPERIENCE",
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
            timestamp=experience_data["timestamp"],
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

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
        """Lista todas las fases disponibles para una experiencia informacional EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[[], Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def benchmark_eva_memory(self):
        """Realiza benchmarking de la memoria EVA y reporta métricas clave."""
        if self.benchmarking_enabled:
            metrics = {
                "total_experiences": len(self.eva_memory_store),
                "phases": len(self.eva_phases),
                "hooks": len(self._environment_hooks),
                "compression_level": self.compression_level,
                "simulation_rate": self.simulation_rate,
                "multiverse_enabled": self.multiverse_enabled,
                "timeline_count": self.timeline_count,
            }
            logger.info(f"[EVA-STORAGE-BENCHMARK] {metrics}")
            return metrics

    def optimize_eva_memory(self):
        """Optimiza la gestión de memoria EVA, aplicando compresión y limpieza según la política."""
        if len(self.eva_memory_store) > self.max_experiences:
            sorted_exps = sorted(
                self.eva_memory_store.items(),
                key=lambda x: getattr(x[1], "timestamp", 0),
            )
            for exp_id, _ in sorted_exps[
                : len(self.eva_memory_store) - self.max_experiences
            ]:
                del self.eva_memory_store[exp_id]
        # Placeholder para lógica avanzada de compresión si es necesario

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_storage_experience": self.eva_ingest_storage_experience,
            "eva_recall_storage_experience": self.eva_recall_storage_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "benchmark_eva_memory": self.benchmark_eva_memory,
            "optimize_eva_memory": self.optimize_eva_memory,
        }
