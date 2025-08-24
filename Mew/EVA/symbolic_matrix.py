from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from crisalida_lib.EVA.core_types import (
    EVAExperience,
    LivingSymbolRuntime,
    RealityBytecode,
)
from crisalida_lib.EVA.typequalia import QualiaState

# Avoid circular import
if TYPE_CHECKING:
    pass


class Matrix:
    """
    Representa una matriz simbólica, un arreglo de Sigils que define una intención ontológica.
    Soporta estructura de cuadrícula, pero está diseñada para ser extendida a representaciones
    esféricas, topológicas y multidimensionales. Incluye métodos avanzados para manipulación,
    diagnóstico y transformación de matrices simbólicas.
    Extensión EVA: integración con memoria viviente, simulación, faseo y hooks de entorno.
    """

    # type: ignore[attr-defined]
    _environment_hooks: list[Callable[..., Any]]

    def __init__(self, sigils: list[list[Any]]):
        # Basic matrix data
        self.sigils = sigils
        self.rows = len(sigils)
        self.cols = len(sigils[0]) if self.rows > 0 else 0

        # EVA: memoria viviente y runtime de simulación
        self.eva_runtime = LivingSymbolRuntime()
        # Avoid circular dependency - divine_compiler will be set externally if needed
        self.divine_compiler = None

        # EVA storage and phases (runtime assignments; keep type-comments to satisfy static tools)
        self.eva_memory_store = {}  # type: dict[str, RealityBytecode]
        self.eva_experience_store = {}  # type: dict[str, EVAExperience]
        self.eva_phases = {}  # type: dict[str, dict[str, RealityBytecode]]
        self._current_phase = "default"
        self._environment_hooks = []  # type: list[Callable[..., Any]]

    def get_sigil_at(self, row: int, col: int) -> Any | None:
        """Devuelve el sigilo en la posición dada, o None si fuera de rango."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.sigils[row][col]
        return None

    def set_sigil_at(self, row: int, col: int, sigil: Any) -> bool:
        """Establece el sigilo en la posición dada. Devuelve True si exitoso."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.sigils[row][col] = sigil
            return True
        return False

    def get_shape(self) -> tuple[int, int]:
        """Devuelve la forma (filas, columnas) de la matriz."""
        return self.rows, self.cols

    def transpose(self) -> "Matrix":
        """Devuelve una nueva matriz transpuesta."""
        transposed = [list(row) for row in zip(*self.sigils, strict=False)]
        return Matrix(transposed)

    def map(self, func: Callable[[Any], Any]) -> "Matrix":
        """Aplica una función a cada sigilo y devuelve una nueva matriz."""
        mapped = [[func(s) for s in row] for row in self.sigils]
        return Matrix(mapped)

    def iter_sigils(self):
        """Itera sobre todos los sigilos de la matriz."""
        for row in self.sigils:
            yield from row

    def find(self, predicate: Callable[[Any], bool]) -> list[tuple[int, int]]:
        """Devuelve las posiciones (fila, col) donde el predicado es True."""
        positions = []
        for i, row in enumerate(self.sigils):
            for j, sigil in enumerate(row):
                if predicate(sigil):
                    positions.append((i, j))
        return positions

    def get_neighbors(self, row: int, col: int, diagonal: bool = False) -> list[Any]:
        """Devuelve los sigilos vecinos (ortogonales y opcionalmente diagonales)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append(self.sigils[nr][nc])
        return neighbors

    def to_list(self) -> list[list[Any]]:
        """Devuelve la matriz como lista de listas."""
        return self.sigils

    def __str__(self) -> str:
        return "\n".join(
            [" ".join([getattr(s, "name", str(s)) for s in row]) for row in self.sigils]
        )

    def __repr__(self) -> str:
        return f"Matrix(shape={self.get_shape()}, sigils={self.rows * self.cols})"

    def summary(self) -> dict:
        """Devuelve un resumen diagnóstico de la matriz."""
        unique = set(self.iter_sigils())
        return {
            "shape": self.get_shape(),
            "total_sigils": self.rows * self.cols,
            "unique_sigils": len(unique),
            "sigil_names": [getattr(s, "name", str(s)) for s in unique],
        }

    # --- EVA: Integración de Memoria Viviente y Simulación ---
    def eva_ingest_matrix_experience(
        self, qualia_state: QualiaState, phase: str | None = None
    ) -> str:
        """
        Compila la matriz simbólica como experiencia y la almacena en la memoria EVA.
        """
        phase = phase or self._current_phase
        intention = {
            "intention_type": "ARCHIVE_MATRIX_EXPERIENCE",
            "matrix": self.to_list(),
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        if _dc is not None:
            bytecode = _dc.compile_intention(intention)
        else:
            bytecode = []
        experience_id = f"matrix_{hash(str(self.sigils))}"
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_matrix_experience(self, cue: str, phase: str | None = None) -> dict:
        """
        Ejecuta el RealityBytecode de una matriz simbólica almacenada, manifestando la simulación.
        """
        phase = phase or self._current_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for matrix experience"}
        _eva = getattr(self, "eva_runtime", None)
        quantum_field = (
            getattr(_eva, "quantum_field", None) if _eva is not None else None
        )
        manifestations = []
        for instr in reality_bytecode.instructions:
            _exec = getattr(_eva, "execute_instruction", None)
            if _exec is None:
                continue
            symbol_manifest = _exec(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] Matrix environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
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
        }

    def add_matrix_experience_phase(
        self,
        experience_id: str,
        phase: str,
        matrix: list[list[Any]],
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de matriz simbólica.
        """
        intention = {
            "intention_type": "ARCHIVE_MATRIX_EXPERIENCE",
            "matrix": matrix,
            "qualia": qualia_state,
            "phase": phase,
        }
        _dc = getattr(self, "divine_compiler", None)
        if _dc is not None:
            bytecode = _dc.compile_intention(intention)
        else:
            bytecode = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self._current_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self._current_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de matriz."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_matrix_experience": self.eva_ingest_matrix_experience,
            "eva_recall_matrix_experience": self.eva_recall_matrix_experience,
            "add_matrix_experience_phase": self.add_matrix_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
        }
