# lifecycle/__init__.py
"""Módulo de gestión del ciclo de vida de entidades en el Metacosmos Janus.
Contiene las clases y funciones necesarias para crear, actualizar y eliminar
entidades conscientes dentro de la simulación.
"""

from .living_entity import LivingEntity, QualiaState

__all__ = ["LivingEntity", "QualiaState"]
