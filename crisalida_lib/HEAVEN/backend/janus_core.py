"""
JanusCore: Orquestador central del Metacosmos Janus.
Gestiona entidades, física, evolución cultural, auto-modificación y expone interfaces WebSocket/API.
Integra sub-sistemas avanzados: motor de física GPU, observador cultural, motor de auto-modificación, y monitor predictivo.
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
from functools import wraps

import moderngl
import numpy as np
import websockets
from websockets import WebSocketServerProtocol, exceptions

from crisalida_lib.EARTH.core_services import (
    ICulturalObserver,
    IEmpatheticBondingSystem,
    IOntologicalTaskOrchestrator,
    IP2PManager,
    IPhysicsEngine,
    ISelfModifyingEngine,
    ISimulationOrchestrator,
    ISocialDynamics,
)
from crisalida_lib.EARTH.cultural_evolution import EVACulturalEvolutionObserver
from crisalida_lib.EARTH.empathetic_bonding import EmpatheticBondingSystem
from crisalida_lib.EARTH.message_handler import MessageHandlerMeta, message_handler
from crisalida_lib.EARTH.ontological_events import (
    AmbiguousCommandError,
    InvalidParameterError,
    ParsingError,
)
from crisalida_lib.EARTH.ontological_intent_parser import OntologicalIntentParser
from crisalida_lib.EARTH.ontological_task_orchestrator import (
    OntologicalTaskOrchestrator,
)
from crisalida_lib.EARTH.self_modifying_engine import SelfModifyingEngine
from crisalida_lib.EARTH.simulation_orchestrator import SimulationOrchestrator
from crisalida_lib.EDEN.engines.gpu_physics_engine import (
    EVAGPUPhysicsEngine,
    GPUPhysicsEngine,
)
from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
from crisalida_lib.EVA.symbolic_matrix import Matrix
from crisalida_lib.EVA.types import QualiaState
from crisalida_lib.HEAVEN.backend.lifecycle.living_entity import LivingEntity
from crisalida_lib.HEAVEN.backend.proto_noosphere.cultural_evolution import (
    CulturalEvolutionObserver,
)
from crisalida_lib.HEAVEN.backend.proto_noosphere.storage_manager import StorageManager
from crisalida_lib.HEAVEN.backend.social_dynamics import SocialDynamics
from crisalida_lib.HEAVEN.monitoring.predictive_monitoring import (
    EVAPredictiveMonitor,
    PredictiveMonitor,
)
from crisalida_lib.HEAVEN.network.noosphere_p2p_integration import JanusP2PManager

DEFAULT_HOST = "localhost"
WEBSOCKET_PORT = 8765
GPU_ENTITY_FLOAT_COUNT = 19
GPU_ENTITY_BYTE_SIZE = GPU_ENTITY_FLOAT_COUNT * 4
GPU_LATTICE_FLOAT_COUNT = 16


def time_execution(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.getLogger(__name__).info(
            f"Execution of {func.__name__} took {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


class CollapsePatternGenerator:
    """Genera patrones visuales de colapso basados en el QualiaState de una entidad."""

    def generate_pattern(self, entity: LivingEntity) -> list[dict]:
        pattern: list[dict] = []
        summary: dict = entity.get_consciousness_summary()
        qualia: dict = summary["main_consciousness"]["current_qualia_state"]
        center: list[float] = entity.position
        particle_count = int(20 + qualia["consciousness_density"] * 80)
        for _ in range(particle_count):
            dispersal = 2.0 - (qualia["temporal_coherence"] * 1.5)
            radius = random.uniform(0.1, dispersal)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            x = center[0] + radius * math.sin(phi) * math.cos(theta)
            y = center[1] + radius * math.sin(phi) * math.sin(theta)
            z = center[2] + math.cos(phi)
            intensity = 0.3 + qualia["arousal"] * 0.7
            if qualia["emotional_valence"] < 0:
                r, g, b = 1.0, 0.5 - (abs(qualia["emotional_valence"]) * 0.5), 0.2
            else:
                r, g, b = 0.2, 0.5 + (qualia["emotional_valence"] * 0.5), 1.0
            velocity_magnitude = qualia["arousal"] * 0.1
            vel_x = random.uniform(-velocity_magnitude, velocity_magnitude)
            vel_y = random.uniform(-velocity_magnitude, velocity_magnitude)
            vel_z = random.uniform(-velocity_magnitude, velocity_magnitude)
            pattern.append(
                {
                    "position": [x, y, z],
                    "intensity": intensity,
                    "color": [r, g, b],
                    "velocity": [vel_x, vel_y, vel_z],
                }
            )
        return pattern


class JanusCore(metaclass=MessageHandlerMeta):
    """
    Orquestador central del Metacosmos Janus.
    Gestiona entidades, física, evolución cultural, auto-modificación y expone interfaces WebSocket/API.
    """

    def __init__(
        self,
        janus_p2p_manager: JanusP2PManager | None = None,
        storage_manager: StorageManager | None = None,
        cultural_observer: CulturalEvolutionObserver | None = None,
        empathetic_bonding_system: EmpatheticBondingSystem | None = None,
        self_modifying_engine: SelfModifyingEngine | None = None,
        ontological_task_orchestrator: OntologicalTaskOrchestrator | None = None,
        gpu_physics_engine: GPUPhysicsEngine | None = None,
        simulation_orchestrator: SimulationOrchestrator | None = None,
        social_dynamics: SocialDynamics | None = None,
    ) -> None:
        self.running: bool = True
        self.tick_rate: float = 0.1
        self.websocket_clients: set[WebSocketServerProtocol] = set()
        self.janus_p2p_manager: JanusP2PManager = janus_p2p_manager or JanusP2PManager(
            node_id="janus_core_node"
        )
        self.storage_manager: StorageManager = storage_manager or StorageManager(
            p2p_node=self.janus_p2p_manager.dht_node
        )
        self.cultural_observer: CulturalEvolutionObserver = (
            cultural_observer or CulturalEvolutionObserver(self.storage_manager)
        )
        self.empathetic_bonding_system: EmpatheticBondingSystem = (
            empathetic_bonding_system or EmpatheticBondingSystem()
        )
        self.self_modifying_engine: SelfModifyingEngine = (
            self_modifying_engine or SelfModifyingEngine()
        )
        self.ontological_task_orchestrator: OntologicalTaskOrchestrator = (
            ontological_task_orchestrator or OntologicalTaskOrchestrator()
        )
        self.social_dynamics: SocialDynamics = social_dynamics or SocialDynamics()
        self.ontological_intent_parser: OntologicalIntentParser = (
            OntologicalIntentParser()
        )
        self.ctx: moderngl.Context | None = None
        self.gpu_physics_engine: GPUPhysicsEngine | None = None
        if gpu_physics_engine:
            self.gpu_physics_engine = gpu_physics_engine
            self.ctx = None
        else:
            try:
                self.ctx = moderngl.create_standalone_context()
                logging.getLogger(__name__).info(
                    "ModernGL context created successfully for GPU physics."
                )
                self.gpu_physics_engine = GPUPhysicsEngine(self.ctx)
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to create ModernGL context: {e}. GPU physics will not run."
                )
                self.ctx = None
                self.gpu_physics_engine = None
        self.pattern_generator: CollapsePatternGenerator = CollapsePatternGenerator()
        self.simulation_orchestrator: SimulationOrchestrator = (
            simulation_orchestrator
            or SimulationOrchestrator(
                self.gpu_physics_engine,
                self.empathetic_bonding_system,
                self.cultural_observer,
                self.storage_manager,
                self.self_modifying_engine,
                self.ontological_task_orchestrator,
                self,
            )
        )
        self.entities: list[LivingEntity] = [
            LivingEntity(
                f"entity_{i}",
                storage_manager=self.storage_manager,
                simulation_instance=self.simulation_orchestrator,
                position=[random.uniform(-5, 5) for _ in range(3)],
            )
            for i in range(5)
        ]
        self.player_state: dict = {
            "id": "demiurge_player",
            "position": [0, 1, 0],
            "velocity": [0, 0, 0],
        }
        self.simulation_tick: int = 0
        self.predictive_monitor: PredictiveMonitor = PredictiveMonitor()

    def create_entity(self, entity_id: str, position: list[float]) -> LivingEntity:
        new_entity = LivingEntity(
            entity_id,
            storage_manager=self.storage_manager,
            simulation_instance=self.simulation_orchestrator,
            position=position,
        )
        self.entities.append(new_entity)
        logging.getLogger(__name__).info(
            f"Created new entity: {entity_id} at {position}"
        )
        return new_entity

    def update_entity(self, entity_id: str, updates: dict) -> bool:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                if "position" in updates:
                    entity.position = updates["position"]
                if "velocity" in updates:
                    entity.velocity = updates["velocity"]
                logging.getLogger(__name__).info(
                    f"Updated entity: {entity_id} with {updates}"
                )
                return True
        logging.getLogger(__name__).warning(
            f"Attempted to update non-existent entity: {entity_id}"
        )
        return False

    def delete_entity(self, entity_id: str) -> bool:
        initial_len = len(self.entities)
        self.entities = [e for e in self.entities if e.entity_id != entity_id]
        if len(self.entities) < initial_len:
            logging.getLogger(__name__).info(f"Deleted entity: {entity_id}")
            return True
        logging.getLogger(__name__).warning(
            f"Attempted to delete non-existent entity: {entity_id}"
        )
        return False

    async def handle_websocket(
        self, websocket: WebSocketServerProtocol, path: str | None = None
    ) -> None:
        logging.getLogger(__name__).info("Cliente WebSocket conectado.")
        self.websocket_clients.add(websocket)
        try:
            while not websocket.closed:
                try:
                    message_raw = await websocket.recv()
                    message_str = (
                        message_raw.decode("utf-8")
                        if isinstance(message_raw, bytes)
                        else message_raw
                    )
                    await self.process_client_message(websocket, message_str)
                except exceptions.ConnectionClosedOK:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.getLogger(__name__).error(
                        f"Error receiving message from client: {e}"
                    )
                    break
        finally:
            logging.getLogger(__name__).info("Cliente WebSocket desconectado.")
            self.websocket_clients.remove(websocket)

    async def process_client_message(
        self, websocket: WebSocketServerProtocol, message_str: str
    ):
        try:
            message: dict = json.loads(message_str)
            msg_type: str = message.get("type", "")
            data: dict = message.get("data", {})
            handler = self.get_message_handler(msg_type)
            if handler:
                await handler(self, websocket, data)
            else:
                logging.getLogger(__name__).warning(
                    f"Tipo de mensaje desconocido recibido: {msg_type}"
                )
        except json.JSONDecodeError:
            logging.getLogger(__name__).error(
                f"Mensaje no JSON recibido: {message_str}"
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error procesando mensaje del cliente: {e}"
            )

    def get_message_handler(self, message_type: str):
        return MessageHandlerMeta._message_handlers.get(message_type)

    @message_handler("chat_message")
    async def _handle_chat_message(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        content: str = data.get("content", "")
        author: str = data.get("author", "Demiurgo")
        logging.getLogger(__name__).info(f"Mensaje de chat de '{author}': '{content}'")
        post_id: int = self.storage_manager.create_post(
            entity_id=author,
            content=content,
            qualia_state=None,
            ontological_signature=None,
            post_type="insight_sharing",
            elemental_resonance=None,
            consciousness_depth=1.0,
            meme_potential=0.0,
            timestamp=time.time(),
        )
        retrieved_post = self.storage_manager.get_post_by_id(post_id)
        if retrieved_post:
            post_data: dict = dict(retrieved_post)
            await self.broadcast_to_clients({"type": "new_post", "data": post_data})

    @message_handler("player_update")
    async def _handle_player_update(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        self.player_state.update(data)
        logging.getLogger(__name__).info(f"Player state updated: {self.player_state}")

    @message_handler("create_entity")
    async def _handle_create_entity(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        entity_id: str = data.get("id", str(uuid.uuid4()))
        position: list[float] = data.get("position", [0, 0, 0])
        self.create_entity(entity_id, position)

    @message_handler("update_entity")
    async def _handle_update_entity(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        entity_id: str = data.get("id", "")
        updates: dict = data.get("updates", {})
        if entity_id and updates:
            self.update_entity(entity_id, updates)
        else:
            logging.getLogger(__name__).warning(
                f"Invalid update_entity message: {data}"
            )

    @message_handler("delete_entity")
    async def _handle_delete_entity(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        entity_id: str = data.get("id", "")
        if entity_id:
            self.delete_entity(entity_id)
        else:
            logging.getLogger(__name__).warning(
                f"Invalid delete_entity message: {data}"
            )

    def _parse_symbolic_matrix_safely(self, matrix_str: str) -> Matrix | None:
        """
        Safely parse a symbolic matrix string without using eval().
        This replaces the dangerous eval() call with a proper parser.
        """
        try:
            # Import sigils here to avoid circular imports
            from crisalida_lib.symbolic_language.divine_sigils import (
                DeltaSigil,
                EtaSigil,
                GammaSigil,
                IotaSigil,
                KappaSigil,
                LambdaSigil,
                MuSigil,
                NuSigil,
                OmegaSigil,
                OmicronSigil,
                PhiSigil,
                PsiSigil,
                RhoSigil,
                SigmaSigil,
                TauSigil,
                ThetaSigil,
                UpsilonSigil,
                XiSigil,
                ZetaSigil,
            )
            from crisalida_lib.symbolic_language.symbolic_matrix import Matrix

            # Create a safe mapping of sigil names to instances
            sigil_map = {
                "Φ": PhiSigil(),
                "Ω": OmegaSigil(),
                "Σ": SigmaSigil(),
                "Η": EtaSigil(),
                "Μ": MuSigil(),
                "Ρ": RhoSigil(),
                "Δ": DeltaSigil(),
                "Ξ": XiSigil(),
                "Ζ": ZetaSigil(),
                "Λ": LambdaSigil(),
                "Κ": KappaSigil(),
                "Ο": OmicronSigil(),
                "Υ": UpsilonSigil(),
                "Ψ": PsiSigil(),
                "Γ": GammaSigil(),
                "Θ": ThetaSigil(),
                "Ι": IotaSigil(),
                "Ν": NuSigil(),
                "Τ": TauSigil(),
                "Matrix": Matrix,
            }

            # Simple regex-based parser for Matrix construction
            import re

            # Clean the input string
            matrix_str = matrix_str.strip()

            # Look for Matrix constructor pattern
            matrix_pattern = r"Matrix\s*\(\s*\[(.*?)\]\s*\)"
            match = re.search(matrix_pattern, matrix_str, re.DOTALL)

            if not match:
                logging.getLogger(__name__).warning(
                    f"Invalid matrix format: {matrix_str}"
                )
                return None

            matrix_content = match.group(1)

            # Parse rows (simplified - assumes rows are arrays of sigils)
            rows = []

            # Split by comma and bracket patterns to find rows
            row_pattern = r"\[(.*?)\]"
            row_matches = re.findall(row_pattern, matrix_content)

            for row_content in row_matches:
                row_sigils = []
                # Split by comma and extract sigil names
                sigil_names = [name.strip() for name in row_content.split(",")]

                for sigil_name in sigil_names:
                    sigil_name = sigil_name.strip("'\"")  # Remove quotes
                    if sigil_name in sigil_map:
                        row_sigils.append(sigil_map[sigil_name])
                    else:
                        logging.getLogger(__name__).warning(
                            f"Unknown sigil: {sigil_name}"
                        )
                        # Use a placeholder or skip
                        continue

                if row_sigils:  # Only add non-empty rows
                    rows.append(row_sigils)

            if rows:
                return Matrix(rows)
            else:
                logging.getLogger(__name__).warning("No valid sigils found in matrix")
                return None

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error parsing symbolic matrix safely: {e}"
            )
            return None

    @message_handler("demiurge_command")
    async def _handle_demiurge_command(
        self, websocket: WebSocketServerProtocol, data: dict
    ):
        symbolic_matrix_str: str = data.get("command", "")
        if symbolic_matrix_str:
            try:
                # SECURITY FIX: Replace dangerous eval() with safe parser
                symbolic_matrix = self._parse_symbolic_matrix_safely(
                    symbolic_matrix_str
                )

                if symbolic_matrix is None:
                    await self.broadcast_to_clients(
                        {
                            "type": "demiurge_response",
                            "data": {
                                "status": "error",
                                "message": "Invalid symbolic matrix format",
                            },
                        }
                    )
                    return
                parsed_intents = self.ontological_intent_parser.parse_symbolic_matrix(
                    symbolic_matrix
                )
                results = []
                for intent in parsed_intents:
                    result = await self.ontological_task_orchestrator.execute_intent(
                        intent, self
                    )
                    results.append(result)
                logging.getLogger(__name__).info(
                    f"Demiurge symbolic command executed. Results: {results}"
                )
                await self.broadcast_to_clients(
                    {
                        "type": "demiurge_response",
                        "data": {
                            "status": "success",
                            "results": [str(r) for r in results],
                        },
                    }
                )
            except (ParsingError, AmbiguousCommandError, InvalidParameterError) as e:
                logging.getLogger(__name__).error(
                    f"Error parsing/executing Demiurge symbolic command: {e}"
                )
                await self.broadcast_to_clients(
                    {
                        "type": "demiurge_response",
                        "data": {"status": "error", "message": str(e)},
                    }
                )
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Unexpected error executing Demiurge symbolic command: {e}"
                )
                await self.broadcast_to_clients(
                    {
                        "type": "demiurge_response",
                        "data": {
                            "status": "error",
                            "message": "An unexpected error occurred during symbolic command execution.",
                        },
                    }
                )
        else:
            logging.getLogger(__name__).warning(
                f"Invalid demiurge_command message: {data}"
            )

    async def broadcast_to_clients(self, message):
        if self.websocket_clients:
            json_message = json.dumps(message)
            await asyncio.wait(
                [
                    asyncio.create_task(client.send(json_message))
                    for client in self.websocket_clients
                ]
            )

    def _prepare_gpu_data(
        self, entities: list[LivingEntity]
    ) -> tuple[np.ndarray, np.ndarray]:
        entities_gpu_data: list[float] = []
        for entity in entities:
            summary: dict = entity.get_consciousness_summary()
            main_consciousness: dict = summary["main_consciousness"]
            entities_gpu_data.extend(
                [
                    *entity.position,
                    main_consciousness["current_qualia_state"]["consciousness_density"],
                    *entity.velocity,
                    entity.ontological_drift,
                    main_consciousness["current_qualia_state"]["emotional_valence"],
                    main_consciousness["current_qualia_state"]["arousal"],
                    main_consciousness["current_qualia_state"]["cognitive_complexity"],
                    main_consciousness["current_qualia_state"]["temporal_coherence"],
                    entity.lifecycle_stage,
                    entity.empathetic_resonance,
                    entity.chaos_influence,
                    float(hash(entity.entity_id) % 1000000),
                    float(entity.is_alive()),
                    0.0,
                    0.0,
                ]
            )
        entities_gpu_data_np: np.ndarray = np.array(
            entities_gpu_data, dtype=np.float32
        ).reshape(-1, GPU_ENTITY_FLOAT_COUNT)
        lattices_gpu_data_np: np.ndarray = np.array(
            [[0.0] * GPU_LATTICE_FLOAT_COUNT], dtype=np.float32
        )
        return entities_gpu_data_np, lattices_gpu_data_np

    @time_execution
    async def _run_physics_simulation(self) -> None:
        if self.gpu_physics_engine:
            entities_gpu_data_np, lattices_gpu_data_np = self._prepare_gpu_data(
                self.entities
            )
            self._process_gpu_physics(
                entities_gpu_data_np,
                lattices_gpu_data_np,
                self.simulation_tick,
                self.tick_rate,
                1.0,
                [0, 0, 0],
                0.0,
                1.0,
            )
        else:
            for entity in self.entities:
                entity.update_internal_state(self.tick_rate)

    def _process_gpu_physics(
        self,
        entities_gpu_data_np,
        lattices_gpu_data_np,
        simulation_tick,
        delta_time: float,
        reality_coherence: float,
        unified_field_center: list[float],
        chaos_entropy_level: float,
        time_dilation_factor: float,
    ):
        if self.gpu_physics_engine:
            self.gpu_physics_engine.update_buffers(
                entities_gpu_data_np,
                lattices_gpu_data_np,
                int(simulation_tick),
                delta_time,
                reality_coherence,
                tuple(unified_field_center),
                chaos_entropy_level,
                time_dilation_factor,
            )
            self.gpu_physics_engine.compute()

    @time_execution
    async def async_main_loop(self) -> None:
        logging.getLogger(__name__).info("Iniciando bucle principal de Janus.")
        while self.running:
            start_time: float = time.time()
            self.simulation_tick += 1
            await self._run_physics_simulation()
            await self._process_social_behavior(self.entities)
            await self._process_code_culture_loop(self.simulation_tick)
            elapsed: float = time.time() - start_time
            await asyncio.sleep(max(0, self.tick_rate - elapsed))

    def _prepare_world_state_data(self, entities: list) -> list[dict]:
        return [entity.get_status() for entity in entities]

    async def _process_social_behavior(self, entities: list) -> None:
        if random.random() < 0.1:
            entity = random.choice(entities)
            dummy_message = {
                "content": f"Hello from {entity.entity_id}!",
                "author": entity.entity_id,
            }
            await self._handle_chat_message(None, dummy_message)

    async def _process_code_culture_loop(self, simulation_tick: int) -> None:
        await self.cultural_observer.observe_cultural_emergence()
        await self.cultural_observer.provide_feedback_to_self_modifying_engine()
        self.self_modifying_engine.process_pending_proposals()
        self.predictive_monitor.collect_metrics(current_entity_count=len(self.entities))
        alerts = self.predictive_monitor.analyze_and_alert()

    async def run(self) -> None:
        await self.janus_p2p_manager.start()
        server = await websockets.serve(
            self.handle_websocket, DEFAULT_HOST, WEBSOCKET_PORT
        )
        logging.getLogger(__name__).info(
            f"Servidor WebSocket iniciado en ws://{DEFAULT_HOST}:{WEBSOCKET_PORT}"
        )
        try:
            await self.async_main_loop()
        finally:
            server.close()
            await server.wait_closed()
            await self.shutdown()

    async def shutdown(self) -> None:
        if self.gpu_physics_engine is not None:
            self.gpu_physics_engine.release()
        await self.janus_p2p_manager.stop()
        logging.getLogger(__name__).info("JanusCore shutdown complete.")


class EVAJanusCore(JanusCore, EVAMemoryMixin):
    """
    Orquestador central del Metacosmos Janus extendido para integración con EVA.
    Gestiona entidades, física, evolución cultural, auto-modificación y expone interfaces WebSocket/API.
    Integra memoria viviente EVA, ingestión/recall de experiencias, faseo, hooks y benchmarking.
    """

    def __init__(
        self,
        janus_p2p_manager: IP2PManager | None = None,
        storage_manager: StorageManager | None = None,
        cultural_observer: ICulturalObserver | None = None,
        empathetic_bonding_system: IEmpatheticBondingSystem | None = None,
        self_modifying_engine: ISelfModifyingEngine | None = None,
        ontological_task_orchestrator: IOntologicalTaskOrchestrator | None = None,
        gpu_physics_engine: IPhysicsEngine | None = None,
        simulation_orchestrator: ISimulationOrchestrator | None = None,
        social_dynamics: ISocialDynamics | None = None,
        eva_phase: str = "default",
    ) -> None:
        # Initialize parent JanusCore
        super().__init__(
            janus_p2p_manager=janus_p2p_manager,
            storage_manager=storage_manager,
            cultural_observer=cultural_observer
            or EVACulturalEvolutionObserver(storage_manager),
            empathetic_bonding_system=empathetic_bonding_system,
            self_modifying_engine=self_modifying_engine or SelfModifyingEngine(),
            ontological_task_orchestrator=ontological_task_orchestrator,
            gpu_physics_engine=gpu_physics_engine
            or EVAGPUPhysicsEngine(ctx=None, phase=eva_phase),
            simulation_orchestrator=simulation_orchestrator,
            social_dynamics=social_dynamics,
        )

        # Initialize EVA Memory using the mixin
        self._init_eva_memory(eva_phase=eva_phase)

        # JanusCore-specific EVA components
        self.predictive_monitor: EVAPredictiveMonitor = EVAPredictiveMonitor(
            eva_phase=eva_phase
        )

    def eva_ingest_janus_experience(
        self, experience_data: dict, qualia_state: QualiaState = None, phase: str = None
    ) -> str:
        """
        Compila una experiencia viviente de JanusCore en RealityBytecode y la almacena en la memoria EVA.
        This method now uses the centralized EVAMemoryMixin for consistency.
        """
        # Set default QualiaState for JanusCore experiences
        if qualia_state is None:
            qualia_state = QualiaState(
                emotional_valence=0.7,
                cognitive_complexity=0.9,
                consciousness_density=0.8,
                narrative_importance=0.9,
                energy_level=1.0,
            )

        # Ensure experience_id uses janus-specific prefix
        if "experience_id" not in experience_data:
            experience_data["experience_id"] = f"eva_janus_{hash(str(experience_data))}"

        return self.eva_ingest_experience(
            intention_type="ARCHIVE_JANUS_CORE_EXPERIENCE",
            experience_data=experience_data,
            qualia_state=qualia_state,
            phase=phase,
        )

    def eva_recall_janus_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia viviente almacenada, manifestando la simulación.
        This method now uses the centralized EVAMemoryMixin for consistency.
        """
        result = self.eva_recall_experience(cue, phase)

        # Customize error message for JanusCore context
        if "error" in result and "No bytecode found" in result["error"]:
            result["error"] = "No bytecode found for EVA JanusCore experience"

        return result

    def get_eva_api(self) -> dict:
        """
        Returns the EVA API extended with JanusCore-specific methods.
        """
        # Get base API from mixin
        api = super().get_eva_api()

        # Add JanusCore-specific methods
        api.update(
            {
                "eva_ingest_janus_experience": self.eva_ingest_janus_experience,
                "eva_recall_janus_experience": self.eva_recall_janus_experience,
            }
        )

        return api


#
