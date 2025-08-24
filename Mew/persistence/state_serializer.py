import copy
import io
import pickle
import zlib
from typing import Any

import h5py
import numpy as np

from crisalida_lib.ADAM.adam import LivingEntity  # Import LivingEntity


class StateSerializer:
    """
    StateSerializer - Protocolo de serialización optimizado para estados de simulación grandes.
    Combina HDF5 para arrays masivos de NumPy y pickle para objetos complejos.
    Integración con LivingEntity y soporte para versiones futuras.
    """

    PROTOCOL_VERSION = 3  # Increment version due to Protocol Buffer integration

    def _serialize_lattices(self, game_state: dict) -> tuple[dict, bytes]:
        """
        Extrae y serializa los datos de las lattices elementales usando HDF5.
        Elimina los arrays de la estructura temporal para evitar duplicados.
        """
        lattice_data_to_serialize = {}
        temp_game_state = copy.deepcopy(game_state)
        universe_state = temp_game_state.get("universe_state")
        if universe_state and isinstance(universe_state, dict):
            elemental_lattices = universe_state.get("elemental_lattices")
            if elemental_lattices and isinstance(elemental_lattices, dict):
                for name, lattice in elemental_lattices.items():
                    if hasattr(lattice, "lattice_data") and isinstance(
                        lattice.lattice_data, np.ndarray
                    ):
                        lattice_data_to_serialize[name] = lattice.lattice_data
                        lattice.lattice_data = None  # Remove array for pickle
        hdf5_buffer = io.BytesIO()
        with h5py.File(hdf5_buffer, "w") as f:
            for name, data in lattice_data_to_serialize.items():
                f.create_dataset(name, data=data)
        return temp_game_state, hdf5_buffer.getvalue()

    def _serialize_living_entities(self, game_state: dict) -> tuple[dict, bytes]:
        """
        Serializa los objetos LivingEntity usando su método to_updatable_state.
        Reemplaza los objetos por sus IDs en la estructura temporal.
        """
        temp_game_state = copy.deepcopy(game_state)
        living_entities_proto_bytes = {}
        living_entities = temp_game_state.get("living_entities")
        if living_entities and isinstance(living_entities, dict):
            for entity_id, entity in living_entities.items():
                if isinstance(entity, LivingEntity):
                    living_entities_proto_bytes[entity_id] = entity.to_updatable_state()
            temp_game_state["living_entities"] = {
                entity_id: entity_id for entity_id in living_entities
            }
        return temp_game_state, pickle.dumps(living_entities_proto_bytes)

    def _build_package(
        self, hdf5_bytes: bytes, living_entities_bytes: bytes, object_bytes: bytes
    ) -> bytes:
        """
        Construye el paquete final con header, datos HDF5, entidades y el resto del estado.
        Comprime el resultado para optimizar espacio.
        """
        header = {
            "protocol_version": self.PROTOCOL_VERSION,
            "hdf5_bytes_len": len(hdf5_bytes),
            "living_entities_bytes_len": len(living_entities_bytes),
            "object_bytes_len": len(object_bytes),
        }
        header_bytes = pickle.dumps(header)
        header_len_bytes = len(header_bytes).to_bytes(4, "big")
        full_package = (
            header_len_bytes
            + header_bytes
            + hdf5_bytes
            + living_entities_bytes
            + object_bytes
        )
        return zlib.compress(full_package, level=6)

    def serialize(self, game_state: dict) -> bytes:
        """
        Serializa el estado completo del juego en un stream de bytes.
        """
        temp_game_state_lattices, hdf5_bytes = self._serialize_lattices(game_state)
        temp_game_state_entities, living_entities_bytes = (
            self._serialize_living_entities(temp_game_state_lattices)
        )
        object_bytes = pickle.dumps(temp_game_state_entities)
        return self._build_package(hdf5_bytes, living_entities_bytes, object_bytes)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """
        Reconstruye el estado del juego desde un stream de bytes comprimido.
        Restaura arrays de lattices y objetos LivingEntity en la estructura.
        """
        full_package = zlib.decompress(data)
        # Read the header length
        header_len = int.from_bytes(full_package[:4], "big")
        header_bytes = full_package[4 : 4 + header_len]
        header = pickle.loads(header_bytes)
        # Extract HDF5, LivingEntity, and other object data
        hdf5_start = 4 + header_len
        hdf5_end = hdf5_start + header["hdf5_bytes_len"]
        hdf5_bytes = full_package[hdf5_start:hdf5_end]
        living_entities_start = hdf5_end
        living_entities_end = (
            living_entities_start + header["living_entities_bytes_len"]
        )
        living_entities_bytes = full_package[living_entities_start:living_entities_end]
        object_start = living_entities_end
        object_bytes = full_package[object_start:]
        # Deserialize lattice data from HDF5
        hdf5_buffer = io.BytesIO(hdf5_bytes)
        lattice_data_deserialized = {}
        with h5py.File(hdf5_buffer, "r") as f:
            for name in f.keys():
                lattice_data_deserialized[name] = f[name][()]
        # Deserialize LivingEntity states
        living_entities_proto_bytes = pickle.loads(living_entities_bytes)
        living_entities_deserialized = {}
        for entity_id, proto_bytes in living_entities_proto_bytes.items():
            living_entities_deserialized[entity_id] = LivingEntity.from_updatable_state(
                proto_bytes
            )
        # Deserialize the rest of the game state
        game_state = pickle.loads(object_bytes)
        # Reinsert the numpy arrays into the object structure
        if (
            "universe_state" in game_state
            and isinstance(game_state["universe_state"], dict)
            and "elemental_lattices" in game_state["universe_state"]
            and isinstance(game_state["universe_state"]["elemental_lattices"], dict)
        ):
            for name, lattice_array in lattice_data_deserialized.items():
                if name in game_state["universe_state"]["elemental_lattices"]:
                    lattice_obj = game_state["universe_state"]["elemental_lattices"][
                        name
                    ]
                    if hasattr(lattice_obj, "lattice_data"):
                        lattice_obj.lattice_data = lattice_array
        # Reinsert LivingEntity objects
        if "living_entities" in game_state:
            game_state["living_entities"] = living_entities_deserialized
        return game_state


class EVAStateSerializer(StateSerializer):
    """
    EVAStateSerializer - Protocolo de serialización extendido para la memoria viviente EVA.
    Soporta serialización y restauración de experiencias, fases, manifestaciones y hooks de entorno.
    Optimizado para grandes volúmenes de RealityBytecode y EVAExperience.
    """

    EVA_PROTOCOL_VERSION = 1

    def _serialize_eva_memory(
        self, eva_memory: dict, eva_phases: dict, eva_experience_store: dict
    ) -> bytes:
        """
        Serializa la memoria viviente EVA (experiencias, fases, manifestaciones).
        """
        eva_package = {
            "eva_memory": eva_memory,
            "eva_phases": eva_phases,
            "eva_experience_store": eva_experience_store,
        }
        return zlib.compress(pickle.dumps(eva_package), level=6)

    def _deserialize_eva_memory(self, data: bytes) -> dict:
        """
        Reconstruye la memoria viviente EVA desde el stream de bytes comprimido.
        """
        eva_package = pickle.loads(zlib.decompress(data))
        return eva_package

    def serialize(
        self,
        game_state: dict,
        eva_memory: dict = None,
        eva_phases: dict = None,
        eva_experience_store: dict = None,
    ) -> bytes:
        """
        Serializa el estado completo del juego y la memoria viviente EVA en un stream de bytes.
        """
        base_bytes = super().serialize(game_state)
        eva_bytes = b""
        if (
            eva_memory is not None
            and eva_phases is not None
            and eva_experience_store is not None
        ):
            eva_bytes = self._serialize_eva_memory(
                eva_memory, eva_phases, eva_experience_store
            )
        header = {
            "protocol_version": self.PROTOCOL_VERSION,
            "eva_protocol_version": self.EVA_PROTOCOL_VERSION,
            "base_bytes_len": len(base_bytes),
            "eva_bytes_len": len(eva_bytes),
        }
        header_bytes = pickle.dumps(header)
        header_len_bytes = len(header_bytes).to_bytes(4, "big")
        full_package = header_len_bytes + header_bytes + base_bytes + eva_bytes
        return zlib.compress(full_package, level=6)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """
        Reconstruye el estado del juego y la memoria viviente EVA desde un stream de bytes comprimido.
        """
        full_package = zlib.decompress(data)
        header_len = int.from_bytes(full_package[:4], "big")
        header_bytes = full_package[4 : 4 + header_len]
        header = pickle.loads(header_bytes)
        base_start = 4 + header_len
        base_end = base_start + header["base_bytes_len"]
        base_bytes = full_package[base_start:base_end]
        eva_bytes = full_package[base_end : base_end + header.get("eva_bytes_len", 0)]
        game_state = super().deserialize(base_bytes)
        eva_memory = {}
        eva_phases = {}
        eva_experience_store = {}
        if eva_bytes:
            eva_package = self._deserialize_eva_memory(eva_bytes)
            eva_memory = eva_package.get("eva_memory", {})
            eva_phases = eva_package.get("eva_phases", {})
            eva_experience_store = eva_package.get("eva_experience_store", {})
        game_state["eva_memory_store"] = eva_memory
        game_state["eva_phases"] = eva_phases
        game_state["eva_experience_store"] = eva_experience_store
        return game_state

    def serialize_eva_only(
        self, eva_memory: dict, eva_phases: dict, eva_experience_store: dict
    ) -> bytes:
        """
        Serializa solo la memoria viviente EVA (sin estado base).
        """
        return self._serialize_eva_memory(eva_memory, eva_phases, eva_experience_store)

    def deserialize_eva_only(self, data: bytes) -> dict:
        """
        Reconstruye solo la memoria viviente EVA desde el stream de bytes comprimido.
        """
        return self._deserialize_eva_memory(data)
