from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Importaciones de tipos (con fallbacks)
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore

try:
    from crisalida_lib.EDEN.constants import DEFAULT_QUALIA_DIM
except ImportError:
    DEFAULT_QUALIA_DIM = 16

if TYPE_CHECKING:
    from crisalida_lib.EVA.types import QualiaSignature
else:
    try:
        from crisalida_lib.EVA.types import QualiaSignature
    except Exception:

        @dataclass
        class QualiaSignature:
            fingerprint: str
            vector: list[float]
            metadata: dict[str, Any] | None = None


if TYPE_CHECKING:
    from crisalida_lib.EVA.typequalia import QualiaState
else:
    try:
        from crisalida_lib.EVA.typequalia import QualiaState
    except Exception:

        @dataclass
        class QualiaState:
            emotional_valence: float = 0.5
            cognitive_complexity: float = 0.5
            consciousness_density: float = 0.5
            narrative_importance: float = 0.5
            energy_level: float = 1.0


if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import RealityBytecode
else:
    try:
        from crisalida_lib.EVA.core_types import RealityBytecode
    except Exception:

        @dataclass
        class RealityBytecode:
            bytecode_id: str
            instructions: list[Any]
            qualia_state: QualiaState | None = None
            phase: str | None = None
            timestamp: float | None = None


logger = logging.getLogger(__name__)


class EVAMemoryHelper:
    """
    Clase auxiliar que contiene la lógica central de gestión de memoria de EVA.
    Diseñada para ser usada por composición por EVAMemoryMixin.
    """

    def __init__(self, owner: Any):
        self.owner = owner
        self.init_containers()

    def init_containers(self, eva_runtime: Any | None = None) -> None:
        """Inicializa los contenedores de memoria en el objeto propietario."""
        if not hasattr(self.owner, "eva_phase"):
            self.owner.eva_phase = "default"
        if not hasattr(self.owner, "eva_memory_store"):
            self.owner.eva_memory_store = {}
        if not hasattr(self.owner, "eva_experience_store"):
            self.owner.eva_experience_store = {}
        if not hasattr(self.owner, "eva_phases"):
            self.owner.eva_phases = {}
        if not hasattr(self.owner, "_environment_hooks"):
            self.owner._environment_hooks = []
        if not hasattr(self.owner, "eva_signature_index"):
            self.owner.eva_signature_index = {}
        if eva_runtime and not hasattr(self.owner, "eva_runtime"):
            self.owner.eva_runtime = eva_runtime
        logger.debug(f"EVAMemoryHelper initialized for {self.owner.__class__.__name__}")

    def _compile_intention_to_bytecode(
        self, intention: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            canon = json.dumps(
                intention,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
            )
        except Exception:
            canon = str(intention)
        fingerprint = hashlib.sha1(canon.encode("utf-8")).hexdigest()
        dim = int(DEFAULT_QUALIA_DIM)
        if np is not None:
            try:
                seed = int(fingerprint[:16], 16) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.normal(0.0, 1.0, size=dim).astype(float)
                max_abs = float(np.max(np.abs(vec))) or 1.0
                vec = (vec / max_abs).clip(-1.0, 1.0)
                vector = vec.tolist()
            except Exception:
                vector = [
                    ((int(fingerprint[i % len(fingerprint)], 16) / 15.0) * 2.0) - 1.0
                    for i in range(dim)
                ]
        else:
            vector = [
                ((int(fingerprint[i % len(fingerprint)], 16) / 15.0) * 2.0) - 1.0
                for i in range(dim)
            ]
        qualia_sig = QualiaSignature(
            fingerprint=fingerprint,
            vector=vector,
            metadata={
                "intention_type": intention.get("intention_type"),
                "phase": intention.get("phase"),
                "has_experience": "experience" in intention,
                "ts": time.time(),
            },
        )
        symbolic_instruction = {"OP": "QUALIA_SIGNATURE", "fp": fingerprint, "dim": dim}
        return {"instructions": [symbolic_instruction], "qualia_signature": qualia_sig}

    def eva_ingest_experience(
        self,
        intention_type: str,
        experience_data: dict[str, Any],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        phase = (getattr(self.owner, "eva_phase", None) or phase) or "default"
        qualia_state = qualia_state or QualiaState()
        intention = {
            "intention_type": intention_type,
            "experience": experience_data,
            "qualia": getattr(qualia_state, "to_dict", lambda: qualia_state)(),
            "phase": phase,
        }
        compiled = self._compile_intention_to_bytecode(intention)
        instructions = list(compiled.get("instructions") or [])
        from typing import cast

        qualia_sig: QualiaSignature = cast(
            QualiaSignature, compiled.get("qualia_signature")
        )
        experience_id = experience_data.get("experience_id") or str(uuid.uuid4())
        rb = RealityBytecode(
            bytecode_id=experience_id,
            instructions=instructions,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data.get("timestamp", time.time()),
        )
        self.owner.eva_memory_store[experience_id] = rb
        if phase not in self.owner.eva_phases:
            self.owner.eva_phases[phase] = {}
        self.owner.eva_phases[phase][experience_id] = rb
        self.owner.eva_signature_index[experience_id] = qualia_sig
        for hook in getattr(self.owner, "_environment_hooks", []):
            try:
                hook(rb)
            except Exception as e:
                logger.debug(f"[EVA] Environment hook failed during ingest: {e}")
        return experience_id

    def eva_recall_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        phase = (getattr(self.owner, "eva_phase", None) or phase) or "default"
        reality_bytecode: RealityBytecode | None = self.owner.eva_phases.get(
            phase, {}
        ).get(cue) or self.owner.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {
                "error": "No bytecode found for EVA experience",
                "cue": cue,
                "phase": phase,
            }
        # ... resto de la lógica usando self.owner
        return {
            "experience_id": reality_bytecode.bytecode_id,
            "phase": reality_bytecode.phase,
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        intention_type: str,
        experience_data: dict[str, Any],
        qualia_state: QualiaState,
    ) -> bool:
        """
        Añade una fase alternativa para una experiencia EVA (timeline paralela).
        """
        self.init_containers()
        try:
            phase = phase or "default"
            intention = {
                "intention_type": intention_type,
                "experience": experience_data,
                "qualia": getattr(qualia_state, "to_dict", lambda: qualia_state)(),
                "phase": phase,
            }
            compiled = self._compile_intention_to_bytecode(intention)
            instructions = list(compiled.get("instructions") or [])
            from typing import cast

            qualia_sig: QualiaSignature = cast(
                QualiaSignature, compiled.get("qualia_signature")
            )

            rb = RealityBytecode(
                bytecode_id=experience_id,
                instructions=instructions,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=experience_data.get("timestamp", time.time()),
            )

            if phase not in self.owner.eva_phases:
                self.owner.eva_phases[phase] = {}
            self.owner.eva_phases[phase][experience_id] = rb
            self.owner.eva_memory_store[experience_id] = rb
            try:
                self.owner.eva_signature_index[experience_id] = qualia_sig
            except Exception:
                pass
            return True
        except Exception as e:
            logger.exception("[EVA] add_experience_phase failed: %s", e)
            return False

    def register_environment_hook(self, hook: Callable[[Any], None]) -> None:
        """Registra un hook de entorno en el objeto propietario."""
        if not hasattr(self.owner, "_environment_hooks"):
            self.owner._environment_hooks = []
        self.owner._environment_hooks.append(hook)

    def set_memory_phase(self, phase: str) -> None:
        """Establece la fase de memoria en el objeto propietario."""
        self.owner.eva_phase = phase

    def get_memory_phase(self) -> str:
        """Obtiene la fase de memoria del objeto propietario."""
        return getattr(self.owner, "eva_phase", "default")

    def get_eva_api(self) -> dict[str, Any]:
        """Return a minimal description of the helper-backed EVA API for runtime introspection.

        This keeps compatibility with callers that expected a delegate object with a
        `get_eva_api()` method.
        """
        return {
            "name": "EVAMemoryHelper",
            "methods": [
                "eva_ingest_experience",
                "eva_recall_experience",
            ],
        }
