import json
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from crisalida_lib.consciousness.config import EVAConfig
from crisalida_lib.symbolic_language.living_symbols import LivingSymbolRuntime
from crisalida_lib.symbolic_language.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)

logger = logging.getLogger(__name__)


class CulturalEvolutionObserver:
    """
    Observa y analiza la emergencia cultural en la proto-noosfera.
    Integra detección de memes, evolución lingüística, análisis de grafos sociales y feedback adaptativo para el SelfModifyingEngine.
    """

    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        self.meme_pool = {}
        self.linguistic_evolution_data = defaultdict(Counter)
        self.social_graph_data = defaultdict(set)
        self.last_observed_timestamp = 0.0

    async def observe_cultural_emergence(self):
        """Monitorea y analiza patrones culturales emergentes."""
        logger.info("Observando emergencia cultural...")
        all_posts = self.storage_manager.get_all_posts(
            since=self.last_observed_timestamp
        )
        if not all_posts:
            return

        self.last_observed_timestamp = max(
            post.get("timestamp", time.time()) for post in all_posts
        )

        self._detect_and_track_memes(all_posts)
        self._analyze_linguistic_evolution(all_posts)
        self._analyze_social_structures(all_posts)

    def _detect_and_track_memes(self, posts):
        """Detecta y rastrea memes basados en el contenido de los posts."""
        word_counts: Counter = Counter()
        meme_contexts: defaultdict = defaultdict(list)
        for post_row in posts:
            post = dict(post_row) if isinstance(post_row, sqlite3.Row) else post_row
            content = post.get("content_text", "")
            words = [w.strip(".,!?") for w in content.lower().split()]
            word_counts.update(words)
            for word in words:
                meme_contexts[word].append(post.get("author_entity_id", "unknown"))

        for word, count in word_counts.items():
            if count > 5 and len(word) > 3:
                if word not in self.meme_pool:
                    logger.info(f"Nuevo meme detectado: '{word}' (frecuencia: {count})")
                    self.meme_pool[word] = {
                        "first_seen": time.time(),
                        "propagation_count": count,
                        "cultural_impact": 0.1,
                        "spreaders": set(meme_contexts[word]),
                    }
                    self.storage_manager.create_meme(
                        word,
                        post_type="word_meme",
                        description=f"Palabra clave '{word}'",
                    )
                else:
                    self.meme_pool[word]["propagation_count"] = count
                    self.meme_pool[word]["cultural_impact"] = min(
                        1.0, self.meme_pool[word]["cultural_impact"] + 0.01
                    )
                    self.meme_pool[word]["spreaders"].update(meme_contexts[word])
                    self.storage_manager.update_meme_propagation(
                        word,
                        new_cultural_impact=self.meme_pool[word]["cultural_impact"],
                    )

    def _analyze_linguistic_evolution(self, posts):
        """Analiza la evolución lingüística real."""
        for post_row in posts:
            post = dict(post_row) if isinstance(post_row, sqlite3.Row) else post_row
            author = post.get("author_entity_id", "unknown")
            content = post.get("content_text", "")
            words = [w.strip(".,!?") for w in content.lower().split()]
            for word in words:
                self.linguistic_evolution_data[author][word] += 1

    def _analyze_social_structures(self, posts):
        """Analiza la formación de estructuras sociales reales."""
        for post_row in posts:
            post = dict(post_row) if isinstance(post_row, sqlite3.Row) else post_row
            author = post.get("author_entity_id", "unknown")
            mentions = post.get("mentions", [])
            if isinstance(mentions, str):
                try:
                    mentions = json.loads(mentions)
                except Exception:
                    mentions = []
            for mentioned in mentions:
                self.social_graph_data[author].add(mentioned)
                self.social_graph_data[mentioned].add(author)

    async def provide_feedback_to_self_modifying_engine(self):
        """Genera y envía un reporte adaptativo para el SelfModifyingEngine."""
        logger.info("Generando feedback cultural para el SelfModifyingEngine...")
        feedback = []
        # Diversidad de memes
        if len(self.meme_pool) < 5:
            feedback.append(
                {
                    "type": "linguistic_optimization_request",
                    "target": "LinguisticEngine",
                    "rationale": "Baja diversidad de memes culturales. Sugerir mayor creatividad.",
                    "parameters": {"creativity_boost": 0.1},
                }
            )
        # Detección de estancamiento lingüístico
        stagnant_authors = [
            author
            for author, counter in self.linguistic_evolution_data.items()
            if len(counter) < 3
        ]
        if stagnant_authors:
            feedback.append(
                {
                    "type": "author_stagnation_alert",
                    "target": "SelfModifyingEngine",
                    "rationale": f"Autores con baja diversidad léxica: {stagnant_authors}",
                    "parameters": {"target_authors": stagnant_authors},
                }
            )
        # Análisis de clusters sociales
        clusters = self._detect_social_clusters()
        if clusters and len(clusters) > 1:
            feedback.append(
                {
                    "type": "social_cluster_analysis",
                    "target": "SelfModifyingEngine",
                    "rationale": f"Detectados {len(clusters)} clusters sociales.",
                    "parameters": {"clusters": clusters},
                }
            )
        for fb in feedback:
            logger.info(f"Feedback cultural: {fb}")
        # Aquí se enviaría el feedback al SelfModifyingEngine (integración pendiente)

    def _detect_social_clusters(self):
        """Detecta clusters sociales usando el grafo de menciones."""
        visited = set()
        clusters = []
        for node in self.social_graph_data:
            if node not in visited:
                cluster = set()
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        stack.extend(self.social_graph_data[current] - visited)
                clusters.append(list(cluster))
        return clusters


class CulturalEvolutionObserverEVA(CulturalEvolutionObserver):
    """
    Observador y analizador de emergencia cultural extendido para integración con EVA.
    Registra cada evento cultural, meme, evolución lingüística y cluster social como experiencia viviente en la memoria EVA,
    soporta ingestión/recall, faseo, hooks de entorno, benchmarking y gestión avanzada de memoria viviente EVA.
    """

    def __init__(self, storage_manager, eva_config: EVAConfig = None) -> None:
        super().__init__(storage_manager)
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

    async def observe_cultural_emergence(self):
        await super().observe_cultural_emergence()
        # EVA: Ingesta experiencia cultural agregada
        qualia_state = QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        experience_data = {
            "timestamp": time.time(),
            "meme_pool": dict(self.meme_pool),
            "linguistic_evolution_data": {
                k: dict(v) for k, v in self.linguistic_evolution_data.items()
            },
            "social_graph_data": {
                k: list(v) for k, v in self.social_graph_data.items()
            },
            "phase": self.eva_phase,
        }
        self.eva_ingest_cultural_experience(
            experience_data, qualia_state, phase=self.eva_phase
        )

    def eva_ingest_cultural_experience(
        self,
        experience_data: dict,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia cultural agregada en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.eva_phase
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        experience_id = f"eva_cultural_{int(time.time())}"
        intention = {
            "intention_type": "ARCHIVE_CULTURAL_EXPERIENCE",
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
                logger.warning(f"[EVA-CULTURAL] Environment hook failed: {e}")
        return experience_id

    def eva_recall_cultural_experience(
        self, cue: str, phase: str | None = None
    ) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia cultural almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA cultural experience"}
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
                                f"[EVA-CULTURAL] Manifestation hook failed: {e}"
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
        experience_data: dict,
        qualia_state: QualiaState = None,
    ):
        """
        Añade una fase alternativa para una experiencia cultural EVA.
        """
        qualia_state = qualia_state or QualiaState(
            emotional_valence=0.7,
            cognitive_complexity=0.8,
            consciousness_density=0.6,
            narrative_importance=1.0,
            energy_level=1.0,
        )
        intention = {
            "intention_type": "ARCHIVE_CULTURAL_EXPERIENCE",
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
            timestamp=time.time(),
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
                logger.warning(f"[EVA-CULTURAL] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia cultural EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
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
            logger.info(f"[EVA-CULTURAL-BENCHMARK] {metrics}")
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
            "eva_ingest_cultural_experience": self.eva_ingest_cultural_experience,
            "eva_recall_cultural_experience": self.eva_recall_cultural_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "benchmark_eva_memory": self.benchmark_eva_memory,
            "optimize_eva_memory": self.optimize_eva_memory,
        }
