import importlib.util
import logging
import os
import random  # For initial position
from typing import TYPE_CHECKING, Any, Union

from crisalida_lib.ADAM.config import AdamConfig  # Import AdamConfig
from crisalida_lib.EDEN.living_symbol import MovementPattern  # Import MovementPattern

# Defer heavy EVA imports to TYPE_CHECKING to avoid runtime import cycles
if TYPE_CHECKING:
    from crisalida_lib.EDEN.qualia_manifold import QualiaField
    from crisalida_lib.EVA.core_types import LivingSymbolRuntime
    from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
else:
    # Runtime fallbacks when EVA isn't available at import time
    EVAMemoryMixin = object  # type: ignore
    LivingSymbolRuntime = Any  # type: ignore

logger = logging.getLogger(__name__)


class CosmicLattice(EVAMemoryMixin):
    """
    Red cósmica de nodos Sephirot/Qliphoth que influye activamente en la realidad.
    Permite carga dinámica, agregación de influencia, evolución adaptativa y diagnóstico.
    """

    def __init__(
        self,
        manifold: "QualiaField",
        config: AdamConfig = AdamConfig(),
        eva_runtime: Union["LivingSymbolRuntime", None] = None,
    ):
        super().__init__()
        self.manifold_ref = manifold
        self.config = config

        # Default path to the TREE directory next to this file
        self.base_tree_dir = os.path.join(os.path.dirname(__file__), "TREE")

        # Define the mapping of node names to their type (Sephirot or Qliphoth)
        self.node_type_mapping = {
            "keter": "sephirot",
            "chokmah": "sephirot",
            "binah": "sephirot",
            "chesed": "sephirot",
            "gebura": "sephirot", # Assuming typo for Geburah
            "tiferet": "sephirot",
            "netzach": "sephirot",
            "hod": "sephirot",
            "yesod": "sephirot",
            "malkuth": "sephirot",
            "daat": "sephirot",
            "thaumiel": "qliphoth",
            "ghagiel": "qliphoth",
            "satariel": "qliphoth",
            "gamchicoth": "qliphoth",
            "golachab": "qliphoth",
            "togaririm": "qliphoth",
            "harab_serapel": "qliphoth",
            "samael": "qliphoth",
            "gamaliel": "qliphoth",
            "lilith": "qliphoth",
        }

        # containers
        self.nodes: dict[str, Any] = getattr(self, "nodes", {}) or {}
        self.sephirot_nodes: dict[str, Any] = getattr(self, "sephirot_nodes", {}) or {}
        self.qliphoth_nodes: dict[str, Any] = getattr(self, "qliphoth_nodes", {}) or {}
        self._last_aggregate: dict[str, Any] = {}

        # Load nodes based on the mapping
        for node_name, node_type in self.node_type_mapping.items():
            file_path = os.path.join(self.base_tree_dir, f"{node_name}.py")
            self._load_node_from_file(file_path, node_type, self.manifold_ref, self.config)

        logger.info("CosmicLattice initialized with %d nodes.", len(self.nodes))

    def _load_node_from_file(
        self, file_path: str, node_type: str, manifold: "QualiaField", config: AdamConfig
    ):
        """Carga dinámicamente un módulo de nodo desde un archivo."""
        if not os.path.isfile(file_path):
            logger.debug("Node file does not exist: %s", file_path)
            return

        module_name = os.path.basename(file_path)[:-3]

        try:
            # Compute a stable module fullname so relative imports work.
            # Use the package root 'crisalida_lib.EDEN.TREE' plus an optional
            # subfolder name if the files live in 'sephirot' or 'qliphoth'.
            pkg_base = "crisalida_lib.EDEN.TREE"
            fullname = f"{pkg_base}.{module_name}"

            spec = importlib.util.spec_from_file_location(fullname, file_path)
            if not spec or not spec.loader:
                logger.debug("Could not create spec for %s", file_path)
                return
            module = importlib.util.module_from_spec(spec)

            # Set a package context so `from .cosmic_node import ...` works
            module.__package__ = fullname.rpartition(".")[0]

            # Register in sys.modules temporarily so intra-package imports resolve
            import sys

            already_registered = fullname in sys.modules
            if not already_registered:
                sys.modules[fullname] = module

            try:
                spec.loader.exec_module(module)
            except Exception:
                # If execution fails, ensure we don't leave a broken module in sys.modules
                if not already_registered and fullname in sys.modules:
                    try:
                        del sys.modules[fullname]
                    except Exception:
                        pass
                raise

            # Discover a Node class inside the module: any class whose name ends with 'Node'
            node_class = None
            for attr_name, attr_val in vars(module).items():
                if isinstance(attr_val, type) and attr_name.endswith("Node"):
                    node_class = attr_val
                    break

            if node_class is None:
                logger.debug("No Node class found in %s", module_name)
                return

            # Build a safe initial position using manifold dimensions (fallback to zeros)
            try:
                dims = getattr(manifold, "dimensions", None) or (1, 1, 1)
                initial_position = (
                    random.randint(0, max(1, int(dims[0])) - 1),
                    random.randint(0, max(1, int(dims[1])) - 1),
                    random.randint(0, max(1, int(dims[2])) - 1),
                )
            except Exception:
                initial_position = (0, 0, 0)

            # Try to instantiate with a generous kwarg set, then fall back on simpler signatures
            kwargs = {
                "entity_id": f"{node_type}_{module_name}",
                "manifold": manifold,
                "initial_position": initial_position,
                "config": config,
                "movement_pattern": MovementPattern.STATIC,
                "node_name": module_name,
            }

            instance = None
            try:
                instance = node_class(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                # Fallback to positional minimal constructor
                try:
                    instance = node_class(kwargs["entity_id"], manifold, initial_position)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("Failed to instantiate node %s with fallbacks", module_name)
                    return

            if instance is None:
                return

            # annotate and register
            setattr(instance, "node_type", node_type)
            setattr(instance, "node_name", module_name)
            self.nodes[module_name] = instance
            if node_type == "sephirot":
                self.sephirot_nodes[module_name] = instance
            else:
                self.qliphoth_nodes[module_name] = instance

            logger.info("Loaded node %s (%s) at %s", module_name, node_type, initial_position)

        except Exception:
            logger.exception("ERROR: Could not load node %s from %s", module_name, file_path)

    def get_total_influence(
        self, perception: dict[str, Any]
    ) -> tuple[float, float, dict[str, float]]:
        """
        Obtiene la influencia agregada de todos los nodos sobre una percepción.
        Devuelve influencia Sephirot, Qliphoth y un dict de influencias por nodo.
        """
        total_sephirot_influence = 0.0
        total_qliphoth_influence = 0.0
        node_influences: dict[str, float] = {}
        for node_name, node_instance in self.nodes.items():
            # Nodes now analyze their local qualia, not a global perception dict
            if hasattr(node_instance, "analyze") and callable(node_instance.analyze):
                try:
                    impulses = node_instance.analyze(perception)
                except TypeError:
                    impulses = node_instance.analyze()
                node_instance.pulse(impulses)  # Make node pulse its influence

                # For now, we'll just sum up the total strength of impulses
                node_influence = sum(
                    getattr(impulse, "intensity", 0.0) for impulse in impulses
                )
                node_influences[node_name] = node_influence
                if getattr(node_instance, "node_type", "qliphoth") == "sephirot":
                    total_sephirot_influence += node_influence
                else:
                    total_qliphoth_influence += node_influence
        num_sephirot = max(1, len(self.sephirot_nodes))
        num_qliphoth = max(1, len(self.qliphoth_nodes))
        return (
            total_sephirot_influence / num_sephirot,
            total_qliphoth_influence / num_qliphoth,
            node_influences,
        )

    def calculate_total_influence(
        self, perception_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculates the total cosmic influence that should be applied to the QualiaManifold.
        This aggregates influences from all Sephirot and Qliphoth nodes.
        """
        # Now, this method will trigger the nodes to analyze their local qualia and pulse
        # The actual influence on the manifold happens within node.pulse()
        # This method will just return a summary of the overall cosmic state
        sephirot_influence, qliphoth_influence, node_influences = (
            self.get_total_influence(perception_context)
        )  # perception_context is unused here now

        influence_strength = abs(sephirot_influence - qliphoth_influence) * 0.1
        influence_type = "order" if sephirot_influence > qliphoth_influence else "chaos"

        return {
            "strength": influence_strength,
            "type": influence_type,
            "sephirot_influence": sephirot_influence,
            "qliphoth_influence": qliphoth_influence,
            "node_influences": node_influences,
            "pattern": None,  # Placeholder for now
            "active_nodes": len(self.nodes),
            "cosmic_balance": sephirot_influence / (qliphoth_influence + 1e-6),
        }

    def aggregate_node_resonances(self) -> dict[str, Any]:
        """
        Query each registered node for local qualia resonance (prefer node.perceive_local_qualia())
        and produce compact aggregate metrics used by the lattice controller.
        """
        per_node = []
        for node in getattr(self, "nodes", {}).values():
            try:
                if hasattr(node, "perceive_local_qualia") and callable(node.perceive_local_qualia):
                    r = node.perceive_local_qualia()
                elif hasattr(node, "sense_local_field") and callable(node.sense_local_field):
                    r = node.sense_local_field()
                else:
                    r = {"resonance": {"coherence": 0.0, "intensity": 0.0, "memory_influence_score": 0.0}}

                res = r.get("resonance", {}) if isinstance(r, dict) else {}
                per_node.append({
                    "node_name": getattr(node, "node_name", getattr(node, "entity_id", "unknown")),
                    "resonance": res,
                })
            except Exception:
                logger.exception(
                    "Error collecting resonance from node %s",
                    getattr(node, "node_name", getattr(node, "entity_id", "unknown")),
                )

        # aggregate numeric metrics robustly
        count = len(per_node)
        agg = {"count": count, "mean_coherence": 0.0, "mean_intensity": 0.0, "mean_memory_influence": 0.0}

        if count:
            coherence_vals = [float(n["resonance"].get("coherence", 0.0)) for n in per_node]
            intensity_vals = [float(n["resonance"].get("intensity", 0.0)) for n in per_node]
            mem_vals = [float(n["resonance"].get("memory_influence_score", 0.0)) for n in per_node]
            agg["mean_coherence"] = sum(coherence_vals) / count
            agg["mean_intensity"] = sum(intensity_vals) / count
            agg["mean_memory_influence"] = sum(mem_vals) / count

        result = {"aggregate": agg, "per_node": per_node}
        self._last_aggregate = result
        return result


# --- Bridge Function ---


def apply_cosmic_influence_to_manifold(
    cosmic_lattice: "CosmicLattice",
    qualia_manifold: "QualiaField",
    perception_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Gets the total influence from the CosmicLattice and applies it to the QualiaManifold.
    This function acts as a bridge between the two systems.
    """
    # 1. Get influence from the Cosmic Lattice (this now triggers nodes to pulse)
    influence_data = cosmic_lattice.calculate_total_influence(perception_context)

    # 2. The actual application to QualiaManifold happens inside node.pulse()
    #    This function now primarily orchestrates the nodes to act.
    #    We might still want a global influence application here if the CosmicLattice itself has a global effect.
    #    For now, we'll assume node.pulse() handles the direct manifold interaction.

    # 3. Return a summary of the interaction
    return {
        "applied_influence_type": influence_data.get("type"),
        "applied_influence_strength": influence_data.get("strength"),
        "cosmic_balance": influence_data.get("cosmic_balance"),
    }
