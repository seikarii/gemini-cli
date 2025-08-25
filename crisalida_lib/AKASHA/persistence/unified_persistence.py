from typing import Any

from crisalida_lib.AKASHA.persistence.state_serializer import (
    EVAStateSerializer,
    StateSerializer,
)
from crisalida_lib.AKASHA.persistence.storage_manager import (
    EVAStorageManager,
    StorageManager,
)


class UnifiedPersistence:
    """
    A unified interface for persistence that orchestrates the serialization, storage, and retrieval of all components of the universe.
    """

    def save_universe_state(self, universe: Any) -> None:
        """Saves the complete state of the universe."""
        raise NotImplementedError

    def load_universe_state(self) -> Any:
        """Loads the complete state of the universe."""
        raise NotImplementedError

    def save_being_state(self, being: Any) -> None:
        """Saves the state of a single ConsciousBeing."""
        raise NotImplementedError

    def load_being_state(self, entity_id: str) -> Any:
        """Loads the state of a single ConsciousBeing."""
        raise NotImplementedError


class UnifiedPersistenceService(UnifiedPersistence):
    """
    A unified persistence service that uses StateSerializer and StorageManager.
    """

    def __init__(self, db_path: str | None = None):
        self.serializer = StateSerializer()
        self.storage_manager = StorageManager(db_path=db_path)

    def save_universe_state(self, universe: Any) -> None:
        """Saves the complete state of the universe."""
        serialized_state = self.serializer.serialize(universe.to_dict())
        self.storage_manager.save_universe_state(serialized_state)

    def load_universe_state(self) -> Any:
        """Loads the complete state of the universe."""
        serialized_state = self.storage_manager.load_universe_state()
        return self.serializer.deserialize(serialized_state)

    def save_being_state(self, being: Any) -> None:
        """Saves the state of a single ConsciousBeing."""
        serialized_state = self.serializer.serialize(being.to_dict())
        self.storage_manager.save_being_state(being.entity_id, serialized_state)

    def load_being_state(self, entity_id: str) -> Any:
        """Loads the state of a single ConsciousBeing."""
        serialized_state = self.storage_manager.load_being_state(entity_id)
        return self.serializer.deserialize(serialized_state)


class EnhancedUnifiedPersistenceService(UnifiedPersistence):
    """
    Enhanced unified persistence service that automatically detects and handles EVA components.
    """

    def __init__(self, db_path: str | None = None, enable_eva: bool = True):
        # Initialize both regular and EVA components
        self.regular_serializer = StateSerializer()
        self.regular_storage = StorageManager(db_path=db_path)

        self.enable_eva = enable_eva
        if enable_eva:
            try:
                self.eva_serializer = EVAStateSerializer()
                self.eva_storage = EVAStorageManager(db_path=db_path)
            except Exception:
                self.enable_eva = False
                self.eva_serializer = None
                self.eva_storage = None

    def _has_eva_components(self, obj: Any) -> bool:
        """Detect if an object has EVA-specific components."""
        if not self.enable_eva:
            return False

        obj_dict = obj.to_dict() if hasattr(obj, "to_dict") else obj

        # Check for EVA indicators
        eva_indicators = [
            "eva_runtime",
            "eva_memory_store",
            "eva_experience_store",
            "eva_phases",
            "living_symbol_runtime",
        ]

        if isinstance(obj_dict, dict):
            for key in obj_dict.keys():
                if any(indicator in str(key).lower() for indicator in eva_indicators):
                    return True

        return False

    def save_universe_state(self, universe: Any) -> None:
        """Saves the complete state of the universe, using EVA storage if EVA components detected."""
        if self._has_eva_components(universe):
            serialized_state = self.eva_serializer.serialize(universe.to_dict())
            self.eva_storage.save_universe_state(serialized_state)
        else:
            serialized_state = self.regular_serializer.serialize(universe.to_dict())
            self.regular_storage.save_universe_state(serialized_state)

    def load_universe_state(self) -> Any:
        """Loads the complete state of the universe, trying EVA first if enabled."""
        if self.enable_eva:
            try:
                serialized_state = self.eva_storage.load_universe_state()
                return self.eva_serializer.deserialize(serialized_state)
            except Exception:
                pass

        # Fallback to regular storage
        serialized_state = self.regular_storage.load_universe_state()
        return self.regular_serializer.deserialize(serialized_state)

    def save_being_state(self, being: Any) -> None:
        """Saves the state of a single ConsciousBeing, using appropriate storage."""
        if self._has_eva_components(being):
            serialized_state = self.eva_serializer.serialize(being.to_dict())
            self.eva_storage.save_being_state(being.entity_id, serialized_state)
        else:
            serialized_state = self.regular_serializer.serialize(being.to_dict())
            self.regular_storage.save_being_state(being.entity_id, serialized_state)

    def load_being_state(self, entity_id: str) -> Any:
        """Loads the state of a single ConsciousBeing, trying EVA first if enabled."""
        if self.enable_eva:
            try:
                serialized_state = self.eva_storage.load_being_state(entity_id)
                return self.eva_serializer.deserialize(serialized_state)
            except Exception:
                pass

        # Fallback to regular storage
        serialized_state = self.regular_storage.load_being_state(entity_id)
        return self.regular_serializer.deserialize(serialized_state)
