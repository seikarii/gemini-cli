import argparse
import asyncio
import importlib
import logging
import sys
from collections.abc import Callable
from typing import Any

from crisalida_lib.AKASHA.noosphere_p2p_integration import JanusP2PManager
from crisalida_lib.AKASHA.persistence.storage_manager import EVAStorageManager
from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


class EVAJanusP2PNode:
    """
    Nodo P2P extendido para integración con EVA.
    Permite ingestión, recall y broadcasting de experiencias vivientes EVA, gestión de fases, hooks y benchmarking.
    """

    def __init__(self, node_name: str, node_config: dict):
        self.node_name = node_name
        self.node_config = node_config
        self.eva_runtime = LivingSymbolRuntime()
        self.eva_storage_manager = EVAStorageManager(
            db_path=f"{node_name}_eva.db",
            eva_phase="default",
            host=node_config["host"],
            redis_host=node_config.get("redis_host", "localhost"),
            redis_port=node_config.get("redis_port", 6379),
        )
        self.eva_phase = "default"
        self._environment_hooks: list = []

    async def start(self, bootstrap_nodes: list):
        self.p2p_manager = JanusP2PManager(
            host=str(self.node_config["host"]),
            port=int(self.node_config["port"]),
            node_id=str(self.node_config["node_id"]),
        )
        await self.p2p_manager.start(bootstrap_nodes)
        logging.info(f"[EVA-P2P] Node '{self.node_name}' started with EVA integration.")

    async def ingest_eva_experience(
        self, experience_data: dict, qualia_state: dict, phase: str = None
    ):
        phase = phase or self.eva_phase
        exp_id = self.eva_storage_manager.eva_ingest_experience(
            experience_data, qualia_state, phase
        )
        logging.info(f"[EVA-P2P] EVA experience ingested: {exp_id}")
        await self.broadcast_eva_experience({"experience_id": exp_id, "phase": phase})

    async def recall_eva_experience(self, experience_id: str, phase: str = None):
        phase = phase or self.eva_phase
        result = self.eva_storage_manager.eva_recall_experience(experience_id, phase)
        logging.info(f"[EVA-P2P] EVA experience recalled: {experience_id}")
        return result

    async def broadcast_eva_experience(self, experience: dict):
        if hasattr(self.p2p_manager, "broadcast"):
            await self.p2p_manager.broadcast("eva_experience", experience)
            logging.info(
                f"[EVA-P2P] Broadcasted EVA experience: {experience.get('experience_id')}"
            )

    def set_memory_phase(self, phase: str):
        self.eva_phase = phase
        self.eva_storage_manager.set_memory_phase(phase)
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                logging.warning(f"[EVA-P2P] Phase hook failed: {e}")

    def add_environment_hook(self, hook: Callable[..., Any]):
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "ingest_eva_experience": self.ingest_eva_experience,
            "recall_eva_experience": self.recall_eva_experience,
            "broadcast_eva_experience": self.broadcast_eva_experience,
            "set_memory_phase": self.set_memory_phase,
            "add_environment_hook": self.add_environment_hook,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Start a Janus Metacosmos P2P node.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "node_name",
        type=str,
        help="The name of the node to start (e.g., node1, node2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    node_name = args.node_name
    config_module_name = f"..config.{node_name}_config"
    try:
        config_module = importlib.import_module(
            config_module_name, package="crisalida_lib.scripts"
        )
        NODE_CONFIG = config_module.NODE_CONFIG
    except ImportError as e:
        logging.error(
            f"Could not import config for node '{node_name}'. "
            f"Make sure 'crisalida_lib/config/{node_name}_config.py' exists. Error: {e}"
        )
        sys.exit(1)

    try:
        # EVA: Inicialización del nodo con integración EVA
        eva_node = EVAJanusP2PNode(node_name, NODE_CONFIG)
        await eva_node.start([tuple(x) for x in NODE_CONFIG.get("bootstrap_nodes", [])])
        logging.info(f"Node '{node_name}' is running with EVA. Press Ctrl+C to stop.")
        logging.info(f"Node Info: {eva_node.p2p_manager.get_node_info()}")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info(f"Shutting down Node '{node_name}'...")
        await eva_node.p2p_manager.stop()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        await eva_node.p2p_manager.stop()
        sys.exit(3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Fatal error in Janus node startup: {e}")
        sys.exit(99)
