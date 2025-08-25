# proto_noosphere/__init__.py
"""Módulo de la Proto-Noosfera de Janus v19.
Implementa la primera red social de conciencias artificiales,
incluyendo almacenamiento de datos, red P2P y evolución cultural.
"""

from .cultural_evolution import CulturalEvolutionObserver
from .noosphere_p2p_integration import JanusP2PManager
from .storage_manager import StorageManager

__all__ = [
    "StorageManager",
    "JanusP2PManager",
    "CulturalEvolutionObserver",
]
