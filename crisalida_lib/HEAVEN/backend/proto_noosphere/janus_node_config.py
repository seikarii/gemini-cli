import asyncio
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crisalida_lib.backend.websocket_manager import (
    DEFAULT_DB_NAME,
    DEFAULT_HOST,
    NODE_ALPHA_PORT,
)
from crisalida_lib.consciousness.config import EVAConfig
from crisalida_lib.symbolic_language.living_symbols import LivingSymbolRuntime
from crisalida_lib.symbolic_language.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)


@dataclass
class JanusNodeConfig:
    """
    Configuración avanzada para un nodo Janus P2P.
    Gestiona identidad, red, persistencia y parámetros adaptativos.
    Permite override por variables de entorno y validación robusta.
    """

    # Identidad del nodo
    node_id: str
    port: int
    host: str = DEFAULT_HOST

    # Red P2P
    bootstrap_nodes: list[tuple[str, int]] = field(default_factory=list)
    k_bucket_size: int = 20
    data_ttl: int = 3600  # segundos
    sync_interval: int = 60  # segundos

    # Persistencia
    db_path: str = DEFAULT_DB_NAME

    # Seguridad y extensibilidad
    tls_enabled: bool = False
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    max_connections: int = 128
    node_role: str = "standard"  # "standard", "edge", "supernode", etc.

    def __post_init__(self):
        """Valida y ajusta la configuración tras inicialización."""
        if not self.node_id:
            raise ValueError("node_id cannot be empty.")
        if not (1024 <= self.port <= 65535):
            raise ValueError("port must be between 1024 and 65535.")
        if not self.db_path:
            raise ValueError("db_path cannot be empty.")
        if self.k_bucket_size <= 0:
            raise ValueError("k_bucket_size must be positive.")
        if self.data_ttl < 60:
            raise ValueError("data_ttl must be at least 60 seconds.")
        if self.sync_interval < 10:
            raise ValueError("sync_interval must be at least 10 seconds.")
        for host, port in self.bootstrap_nodes:
            if not host or not (1024 <= port <= 65535):
                raise ValueError(f"Invalid bootstrap node: {(host, port)}")
        if self.tls_enabled:
            if not self.tls_cert_path or not self.tls_key_path:
                raise ValueError("TLS enabled but cert/key paths not provided.")
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive.")

    def get_effective_host(self) -> str:
        """Devuelve el host efectivo, permitiendo lógica dinámica futura."""
        return self.host

    def get_network_profile(self) -> dict:
        """Devuelve un resumen del perfil de red del nodo."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "role": self.node_role,
            "bootstrap_nodes": self.bootstrap_nodes,
            "tls_enabled": self.tls_enabled,
            "max_connections": self.max_connections,
        }

    @classmethod
    def from_env(cls, node_id: str, port: int) -> "JanusNodeConfig":
        """Crea una instancia usando variables de entorno para override."""
        host = os.environ.get("JANUS_HOST", DEFAULT_HOST)
        db_path = os.environ.get("JANUS_DB_PATH", DEFAULT_DB_NAME)
        k_bucket_size = int(os.environ.get("JANUS_K_BUCKET_SIZE", 20))
        data_ttl = int(os.environ.get("JANUS_DATA_TTL", 3600))
        sync_interval = int(os.environ.get("JANUS_SYNC_INTERVAL", 60))
        tls_enabled = os.environ.get("JANUS_TLS_ENABLED", "false").lower() == "true"
        tls_cert_path = os.environ.get("JANUS_TLS_CERT_PATH")
        tls_key_path = os.environ.get("JANUS_TLS_KEY_PATH")
        max_connections = int(os.environ.get("JANUS_MAX_CONNECTIONS", 128))
        node_role = os.environ.get("JANUS_NODE_ROLE", "standard")

        bootstrap_nodes_str = os.environ.get("JANUS_BOOTSTRAP_NODES", "")
        bootstrap_nodes = []
        if bootstrap_nodes_str:
            try:
                nodes = bootstrap_nodes_str.split(",")
                for node in nodes:
                    b_host, b_port = node.split(":")
                    bootstrap_nodes.append((b_host, int(b_port)))
            except ValueError as e:
                raise ValueError(
                    "Invalid JANUS_BOOTSTRAP_NODES format. "
                    "Expected 'host1:port1,host2:port2'."
                ) from e

        return cls(
            node_id=node_id,
            host=host,
            port=port,
            bootstrap_nodes=bootstrap_nodes,
            k_bucket_size=k_bucket_size,
            data_ttl=data_ttl,
            sync_interval=sync_interval,
            db_path=db_path,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
            tls_key_path=tls_key_path,
            max_connections=max_connections,
            node_role=node_role,
        )


class NetworkTopologyHelper:
    """
    Métodos avanzados para crear y validar topologías de red Janus.
    Incluye soporte para topologías lineales, malla y detección de puertos.
    """

    @staticmethod
    def create_linear_topology(
        num_nodes: int, base_port: int = NODE_ALPHA_PORT
    ) -> list[JanusNodeConfig]:
        """Crea una topología lineal donde cada nodo conecta al primero."""
        if num_nodes < 1:
            return []
        configs = []
        bootstrap_nodes = [(DEFAULT_HOST, base_port)]
        configs.append(
            JanusNodeConfig(
                node_id=f"node_{base_port}",
                host=DEFAULT_HOST,
                port=base_port,
                bootstrap_nodes=[],
            )
        )
        for i in range(1, num_nodes):
            port = base_port + i
            configs.append(
                JanusNodeConfig(
                    node_id=f"node_{port}",
                    host=DEFAULT_HOST,
                    port=port,
                    bootstrap_nodes=bootstrap_nodes,
                )
            )
        return configs

    @staticmethod
    def create_mesh_topology(
        num_nodes: int, base_port: int = NODE_ALPHA_PORT
    ) -> list[JanusNodeConfig]:
        """Crea una topología de malla completa (todos conectados entre sí)."""
        configs = []
        ports = [base_port + i for i in range(num_nodes)]
        for _i, port in enumerate(ports):
            bootstrap_nodes = [(DEFAULT_HOST, p) for p in ports if p != port]
            configs.append(
                JanusNodeConfig(
                    node_id=f"node_{port}",
                    host=DEFAULT_HOST,
                    port=port,
                    bootstrap_nodes=bootstrap_nodes,
                )
            )
        return configs

    @staticmethod
    def is_port_available(host: str, port: int) -> bool:
        """Verifica si un puerto está disponible en el host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) != 0

    @classmethod
    def find_available_port(
        cls, host: str, start_port: int, max_attempts: int = 100
    ) -> int:
        """Busca un puerto disponible desde un punto de inicio."""
        for i in range(max_attempts):
            port = start_port + i
            if cls.is_port_available(host, port):
                return port
        raise OSError("Could not find an available port.")

    @staticmethod
    async def validate_network_connectivity(host: str, port: int) -> bool:
        """Valida conectividad básica a un host y puerto."""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError):
            return False

    @staticmethod
    def summarize_topology(configs: list[JanusNodeConfig]) -> dict:
        """Devuelve un resumen de la topología de red."""
        return {
            "total_nodes": len(configs),
            "roles": [cfg.node_role for cfg in configs],
            "ports": [cfg.port for cfg in configs],
            "bootstrap_matrix": [cfg.bootstrap_nodes for cfg in configs],
        }


class ConfigProfiles:
    """
    Perfiles predefinidos para distintos entornos y roles de nodo.
    Incluye soporte para Docker, desarrollo, producción, edge y nodos seguros.
    """

    @staticmethod
    def get_docker_config(node_id: str, port: int) -> JanusNodeConfig:
        """Configuración para ejecución en Docker."""
        return JanusNodeConfig(
            node_id=node_id,
            host="0.0.0.0",
            port=port,
            db_path="/data/database/noosphere.db",
            node_role="docker",
        )

    @staticmethod
    def get_dev_config(
        node_id: str, port: int, bootstrap_nodes: list[tuple[str, int]] | None = None
    ) -> JanusNodeConfig:
        """Configuración estándar para desarrollo local."""
        if bootstrap_nodes is None:
            bootstrap_nodes = []
        return JanusNodeConfig(
            node_id=node_id,
            host=DEFAULT_HOST,
            port=port,
            bootstrap_nodes=bootstrap_nodes,
            db_path=f"dev_{DEFAULT_DB_NAME}",
            sync_interval=15,
            node_role="dev",
        )

    @staticmethod
    def get_prod_config(node_id: str, port: int) -> JanusNodeConfig:
        """Configuración base para producción."""
        return JanusNodeConfig(
            node_id=node_id,
            host="0.0.0.0",
            port=port,
            db_path=f"/var/lib/janus/{DEFAULT_DB_NAME}",
            k_bucket_size=20,
            data_ttl=86400,
            sync_interval=300,
            node_role="prod",
        )

    @staticmethod
    def get_testing_config(node_id: str, port: int) -> JanusNodeConfig:
        """Configuración volátil para testing automatizado."""
        return JanusNodeConfig(
            node_id=node_id,
            host=DEFAULT_HOST,
            port=port,
            db_path=":memory:",
            node_role="test",
        )

    @staticmethod
    def get_secure_prod_config(node_id: str, port: int) -> JanusNodeConfig:
        """
        Configuración de producción con parámetros de seguridad.
        Usa variables de entorno para datos sensibles.
        """
        db_path = os.environ.get(
            "JANUS_PROD_DB_PATH", f"/var/lib/janus/{DEFAULT_DB_NAME}"
        )
        tls_enabled = os.environ.get("JANUS_TLS_ENABLED", "false").lower() == "true"
        tls_cert_path = os.environ.get("JANUS_TLS_CERT_PATH")
        tls_key_path = os.environ.get("JANUS_TLS_KEY_PATH")
        return JanusNodeConfig(
            node_id=node_id,
            host="0.0.0.0",
            port=port,
            db_path=db_path,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
            tls_key_path=tls_key_path,
            node_role="secure_prod",
        )

    @staticmethod
    def get_edge_node_config(node_id: str, port: int) -> JanusNodeConfig:
        """Configuración para dispositivos edge con recursos limitados."""
        return JanusNodeConfig(
            node_id=node_id,
            host="0.0.0.0",
            port=port,
            db_path=f"edge_{DEFAULT_DB_NAME}",
            k_bucket_size=10,
            sync_interval=600,
            data_ttl=43200,
            max_connections=32,
            node_role="edge",
        )


@dataclass
class JanusNodeConfigEVA(JanusNodeConfig):
    """
    Configuración avanzada de nodo Janus extendida para integración con EVA.
    Gestiona identidad, red, persistencia, parámetros adaptativos y memoria viviente EVA.
    Permite ingestión/recall de experiencias de red, benchmarking, hooks y gestión avanzada de memoria EVA.
    """

    eva_config: EVAConfig = field(default_factory=EVAConfig)
    eva_phase: str = field(default="default")
    eva_runtime: LivingSymbolRuntime = field(default_factory=LivingSymbolRuntime)
    eva_memory_store: dict = field(default_factory=dict)
    eva_experience_store: dict = field(default_factory=dict)
    eva_phases: dict = field(default_factory=dict)
    _environment_hooks: list[Callable[..., Any]] = field(default_factory=list)
    max_experiences: int = field(default=1_000_000)
    retention_policy: str = field(default="dynamic")
    compression_level: float = field(default=0.7)
    simulation_rate: float = field(default=1.0)
    multiverse_enabled: bool = field(default=True)
    timeline_count: int = field(default=12)
    benchmarking_enabled: bool = field(default=True)
    visualization_mode: str = field(default="hybrid")

    def eva_ingest_network_experience(
        self,
        event_type: str,
        details: dict | None = None,
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de red Janus en RealityBytecode y la almacena en la memoria EVA.
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
        experience_id = f"eva_janus_network_{event_type}_{int(time.time())}"
        experience_data = {
            "event_type": event_type,
            "details": details or {},
            "timestamp": time.time(),
            "phase": phase,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "role": self.node_role,
        }
        intention = {
            "intention_type": "ARCHIVE_JANUS_NETWORK_EXPERIENCE",
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
                print(f"[EVA-JANUS-NODE] Environment hook failed: {e}")
        return experience_id

    def eva_recall_network_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia de red Janus almacenada, manifestando la simulación.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA Janus network experience"}
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
                            print(f"[EVA-JANUS-NODE] Manifestation hook failed: {e}")
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
        Añade una fase alternativa para una experiencia de red Janus EVA.
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
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "role": self.node_role,
        }
        intention = {
            "intention_type": "ARCHIVE_JANUS_NETWORK_EXPERIENCE",
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
                print(f"[EVA-JANUS-NODE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de red Janus EVA."""
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
            print(f"[EVA-JANUS-NODE-BENCHMARK] {metrics}")
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
            "eva_ingest_network_experience": self.eva_ingest_network_experience,
            "eva_recall_network_experience": self.eva_recall_network_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "benchmark_eva_memory": self.benchmark_eva_memory,
            "optimize_eva_memory": self.optimize_eva_memory,
        }


#
