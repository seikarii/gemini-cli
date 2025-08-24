"""
Contratos de datos para el subsistema EDEN.
"""

import time
from dataclasses import dataclass, field


@dataclass
class IntentionMatrix:
    """Matriz de intención que representa el programa interno de un símbolo vivo."""

    matrix: list[list[str]]
    description: str = ""
    coherence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self):
        """Valida y calcula propiedades de la matriz."""
        if not self.matrix or not all(self.matrix):
            raise ValueError("La matriz de intención no puede estar vacía")

        if not isinstance(self.matrix, list) or not all(
            isinstance(row, list) for row in self.matrix
        ):
            raise ValueError("La matriz debe ser una lista de listas")

        if self.matrix:
            first_row_len = len(self.matrix[0])
            if not all(len(row) == first_row_len for row in self.matrix):
                raise ValueError(
                    "Todas las filas de la matriz deben tener el mismo tamaño"
                )
