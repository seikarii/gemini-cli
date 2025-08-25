"""
Centralized service for all interactions with the EVA Memory System.
This file now contains the full, functional implementation based on EVAMemoryMixin.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from threading import RLock
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

if TYPE_CHECKING:
    # Import real types for static analysis only.
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QuantumField,
        RealityBytecode,
    )
    from crisalida_lib.EVA.typequalia import QualiaState
else:
    # Attempt to import runtime types; fall back to lightweight placeholders if unavailable.
    try:
        from crisalida_lib.EVA.core_types import (
            EVAExperience,
            LivingSymbolRuntime,
            QuantumField,
            RealityBytecode,
        )
        from crisalida_lib.EVA.typequalia import QualiaState
    except Exception as e:
        # Define placeholder classes if imports fail, allowing the module to be loaded.
        logging.warning(
            f"EVA/ADAM type imports failed ({e}). Using placeholder classes for EVAMemoryManager."
        )

        class LivingSymbolRuntime:
            def __init__(self):
                self.divine_compiler = self
                self.quantum_field = None  # Placeholder

            def compile_intention(self, intention: dict) -> list:
                # Return a simple instruction list as placeholder bytecode
                return [{"op": "placeholder_compile", "details": str(intention)}]

            def execute_instruction(self, instr: dict, field: Any) -> dict | None:
                # Placeholder execution simulates a manifestation object
                return {"manifestation": "placeholder", "instruction": instr}

        class QuantumField:
            pass

        class QualiaState:
            def __init__(self, **kwargs):
                self._payload = dict(kwargs)

            def to_dict(self) -> dict:
                return dict(self._payload)

        class RealityBytecode:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class EVAExperience:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)


logger = logging.getLogger(__name__)


class _SimpleEntityMemory:
    """
    Lightweight internal EntityMemory fallback used if no external entity memory is provided.

    - Stores arbitrary documents keyed by generated IDs.
    - Supports minimal ingest() and recall() API used by EVAMemoryManager.
    """

    def __init__(self, max_items: int = 1000) -> None:
        self._store: dict[str, dict] = {}
        self._lock = RLock()
        self._max_items = int(max_items)

    def ingest(self, data: dict[str, Any], kind: str = "generic") -> str:
        with self._lock:
            doc_id = (
                f"ent_{int(time.time() * 1000)}_{hash(str((kind, data))) & 0xFFFFFFFF}"
            )
            self._store[doc_id] = {"kind": kind, "data": data, "timestamp": time.time()}
            # simple eviction policy
            if len(self._store) > self._max_items:
                # pop oldest
                oldest = sorted(self._store.items(), key=lambda kv: kv[1]["timestamp"])[
                    0
                ][0]
                self._store.pop(oldest, None)
            return doc_id

    def recall(self, cue: str) -> dict[str, Any] | None:
        # cue may be doc_id or a substring; best-effort search
        with self._lock:
            if cue in self._store:
                return self._store[cue]
            # fallback: search by substring
            for doc_id, doc in self._store.items():
                if cue in doc_id or cue in str(doc.get("data", "")):
                    return doc
        return None

    def stats(self) -> dict:
        with self._lock:
            return {"items": len(self._store)}


class EVAMemoryManager:
    """
    Manages all read/write operations to EVA's memory space.

    Enhancements in this "definitive" version:
    - Thread-safe (RLock) and bounded stores with simple eviction.
    - Best-effort async/sync integration with various `divine_compiler` backends.
    - Optional internal entity memory with simple ingest/recall API.
    - Environment hooks invoked for side-effects (manifestation, logging).
    """

    DEFAULT_MAX_MEMORY = 2000

    def __init__(
        self,
        config: Any = None,
        phase: str = "default",
        eva_runtime: LivingSymbolRuntime | None = None,
        entity_memory: Any | None = None,
        max_memory_items: int = DEFAULT_MAX_MEMORY,
    ) -> None:
        # This logic is from the mixin's _init_eva_memory
        self._lock = RLock()
        self.config = config
        self.eva_phase = phase
        self.eva_runtime: LivingSymbolRuntime = eva_runtime or LivingSymbolRuntime()
        # Use provided entity memory or create a simple internal one
        self.entity_memory = entity_memory or _SimpleEntityMemory(
            max_items=max_memory_items
        )
        if getattr(self.eva_runtime, "divine_compiler", None) is None:
            # lazy import to avoid heavy dependency at module import
            try:
                from crisalida_lib.EVA.divine_language_evolved import (
                    DivineLanguageEvolved,
                    UnifiedField,
                )

                self.eva_runtime.divine_compiler = DivineLanguageEvolved(UnifiedField())
            except Exception:
                # keep placeholder in eva_runtime
                logger.debug(
                    "Could not initialize full DivineLanguageEvolved; keeping placeholder compiler"
                )

        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []
        self._max_memory_items = int(max_memory_items)

    # ----------------------------
    # Core persistence operations
    # ----------------------------
    def record_experience(
        self,
        entity_id: str,
        event_type: str,
        data: dict[str, Any],
        qualia_state: QualiaState | None = None,
    ) -> str:
        """
        Records a new experience. Compatible with both sync and async `divine_compiler`.
        Returns generated experience id (empty string on failure).
        """
        with self._lock:
            try:
                # Ingest into internal entity memory (best-effort)
                try:
                    ingest_id = self.entity_memory.ingest(data=data, kind=event_type)
                    logger.debug("Entity memory ingested id=%s", ingest_id)
                except Exception:
                    logger.debug(
                        "Entity memory ingest failed (non-fatal)", exc_info=True
                    )

                phase = self.eva_phase
                qualia_state = qualia_state or QualiaState()

                experience_id = f"exp_{entity_id}_{int(time.time() * 1000)}_{hash(str(data)) & 0xFFFFFFFF}"
                experience_document = {
                    "experience_id": experience_id,
                    "entity_id": entity_id,
                    "event_type": event_type,
                    "payload": data,
                    "timestamp": time.time(),
                }

                intention = {
                    "intention_type": event_type.upper(),
                    "experience": experience_document,
                    "qualia": qualia_state,
                    "phase": phase,
                }

                bytecode: list[Any] = []
                _eva = getattr(self, "eva_runtime", None)
                if (
                    _eva is not None
                    and hasattr(_eva, "divine_compiler")
                    and _eva.divine_compiler is not None
                ):
                    try:
                        compiled = _eva.divine_compiler.compile_intention(intention)
                        # compiled may be coroutine or immediate result
                        if asyncio.iscoroutine(compiled):
                            # schedule and don't block; we keep an empty bytecode for now
                            asyncio.ensure_future(compiled)
                            bytecode = []
                        else:
                            bytecode = compiled or []
                    except Exception:
                        logger.exception(
                            "divine_compiler.compile_intention failed (non-fatal)"
                        )
                        bytecode = []
                else:
                    bytecode = []

                # Normalize timestamp
                ts_val = experience_document.get("timestamp", time.time())
                try:
                    timestamp = float(
                        cast("SupportsFloat | str | bytes | int | float", ts_val)
                    )
                except Exception:
                    timestamp = float(time.time())

                reality_bytecode = RealityBytecode(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=timestamp,
                )

                # store with eviction policy
                self._store_bytecode(reality_bytecode)

                # invoke environment hooks asynchronously (best-effort)
                self._invoke_environment_hooks("record_experience", experience_document)

                logger.info("[EVAMemoryManager] RECORDED: %s", experience_id)
                return experience_id
            except Exception:
                logger.exception("[EVAMemoryManager] Failed to record experience")
                return ""

    def recall_experience(self, experience_id: str) -> dict[str, Any]:
        """
        Recalls an experience, executes its bytecode (best-effort) and returns a manifest summary.
        """
        with self._lock:
            try:
                phase = self.eva_phase
                # First, try to recall from the internal entity memory as hint
                try:
                    internal_recall_result = self.entity_memory.recall(
                        cue=experience_id
                    )
                    if internal_recall_result:
                        logger.debug(
                            "[EVAMemoryManager] Internal memory recall hint: %s",
                            experience_id,
                        )
                except Exception:
                    internal_recall_result = None

                reality_bytecode = self.eva_phases.get(phase, {}).get(
                    experience_id
                ) or self.eva_memory_store.get(experience_id)
                if not reality_bytecode:
                    logger.warning(
                        "[EVAMemoryManager] No bytecode found for %s", experience_id
                    )
                    return {
                        "error": f"No bytecode found for EVA experience ID: {experience_id}"
                    }

                quantum_field = getattr(self.eva_runtime, "quantum_field", None)
                manifestations = []

                instrs = getattr(reality_bytecode, "instructions", []) or []
                _exec = getattr(self.eva_runtime, "execute_instruction", None)
                for instr in instrs:
                    if _exec is None:
                        continue
                    try:
                        res = _exec(instr, quantum_field)
                        if asyncio.iscoroutine(res):
                            # schedule and skip waiting (best-effort)
                            asyncio.ensure_future(res)
                            continue
                        symbol_manifest = res
                    except Exception:
                        logger.exception(
                            "execute_instruction failed for instr (continuing)"
                        )
                        symbol_manifest = None
                    if symbol_manifest:
                        manifestations.append(symbol_manifest)

                eva_experience = EVAExperience(
                    experience_id=getattr(
                        reality_bytecode, "bytecode_id", experience_id
                    ),
                    bytecode=reality_bytecode,
                    manifestations=manifestations,
                    phase=getattr(reality_bytecode, "phase", phase),
                    qualia_state=getattr(
                        reality_bytecode, "qualia_state", QualiaState()
                    ),
                    timestamp=getattr(reality_bytecode, "timestamp", time.time()),
                )
                self.eva_experience_store[
                    getattr(reality_bytecode, "bytecode_id", experience_id)
                ] = eva_experience

                # invoke hooks (manifestation)
                self._invoke_environment_hooks(
                    "recall_experience",
                    {
                        "experience_id": experience_id,
                        "manifestations_count": len(manifestations),
                    },
                )

                logger.info("[EVAMemoryManager] RECALLED: %s", experience_id)

                return {
                    "experience_id": eva_experience.experience_id,
                    "manifestations": [
                        m.to_dict() if hasattr(m, "to_dict") else str(m)
                        for m in manifestations
                    ],
                    "phase": eva_experience.phase,
                    "qualia_state": (
                        eva_experience.qualia_state.to_dict()
                        if hasattr(eva_experience.qualia_state, "to_dict")
                        else {}
                    ),
                    "timestamp": eva_experience.timestamp,
                }
            except Exception as e:
                logger.exception("[EVAMemoryManager] Failed to recall experience")
                return {"error": f"Failed to recall experience: {e}"}

    # ----------------------------
    # Phase / metadata helpers
    # ----------------------------
    def set_memory_phase(self, phase: str) -> None:
        """Changes the active memory phase (timeline)."""
        with self._lock:
            self.eva_phase = phase
            logger.info("[EVAMemoryManager] Phase set to: %s", phase)

    def get_memory_phase(self) -> str:
        """Returns the current memory phase."""
        with self._lock:
            return self.eva_phase

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        data: dict[str, Any],
        qualia_state: QualiaState,
    ) -> None:
        """
        Adds an alternative phase for an EVA cognitive impulse experience.
        """
        with self._lock:
            experience_data = {
                "impulse": data,
                "phase": phase,
                "timestamp": time.time(),
            }
            intention = {
                "intention_type": "ARCHIVE_IMPULSE_EXPERIENCE",
                "experience": experience_data,
                "qualia": qualia_state,
                "phase": phase,
            }
            bytecode: list[Any] = []
            _eva = getattr(self, "eva_runtime", None)
            if (
                _eva is not None
                and hasattr(_eva, "divine_compiler")
                and _eva.divine_compiler is not None
            ):
                try:
                    compiled = _eva.divine_compiler.compile_intention(intention)
                    if asyncio.iscoroutine(compiled):
                        asyncio.ensure_future(compiled)
                    else:
                        bytecode = compiled or []
                except Exception:
                    logger.exception(
                        "divine_compiler.compile_intention failed (non-fatal)"
                    )
                    bytecode = []

            reality_bytecode = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=experience_data["timestamp"],
            )
            if phase not in self.eva_phases:
                self.eva_phases[phase] = {}
            self.eva_phases[phase][experience_id] = reality_bytecode
            logger.info(
                "[EVAMemoryManager] Added experience %s to phase %s",
                experience_id,
                phase,
            )

    def get_experience_phases(self, experience_id: str) -> list[str]:
        """Lists all available phases for an EVA cognitive impulse experience."""
        with self._lock:
            return [
                phase
                for phase, exps in self.eva_phases.items()
                if experience_id in exps
            ]

    # ----------------------------
    # Hooks / environment
    # ----------------------------
    def add_environment_hook(self, hook: Callable[..., Any]) -> None:
        """Registers a hook for symbolic manifestation or EVA events."""
        with self._lock:
            self._environment_hooks.append(hook)
            logger.info(
                "[EVAMemoryManager] Added environment hook: %s",
                getattr(hook, "__name__", str(hook)),
            )

    def _invoke_environment_hooks(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        # Best-effort, non-blocking invocation of hooks
        for hook in list(self._environment_hooks):
            try:
                res = hook(event_type, payload)
                if asyncio.iscoroutine(res):
                    asyncio.ensure_future(res)
            except Exception:
                logger.exception("Environment hook failed (non-fatal)")

    # ----------------------------
    # Utility / storage
    # ----------------------------
    def _store_bytecode(self, reality_bytecode: RealityBytecode) -> None:
        """
        Store RealityBytecode with a simple bounded-memory eviction policy.
        """
        with self._lock:
            key = getattr(
                reality_bytecode, "bytecode_id", f"bc_{int(time.time()*1000)}"
            )
            self.eva_memory_store[key] = reality_bytecode
            # maintain phase index
            phase = getattr(reality_bytecode, "phase", self.eva_phase)
            if phase not in self.eva_phases:
                self.eva_phases[phase] = {}
            self.eva_phases[phase][key] = reality_bytecode
            # evict if needed
            if len(self.eva_memory_store) > self._max_memory_items:
                # remove oldest by timestamp if available, otherwise pop arbitrary
                try:
                    oldest_key = min(
                        self.eva_memory_store.keys(),
                        key=lambda k: getattr(
                            self.eva_memory_store[k], "timestamp", time.time()
                        ),
                    )
                    self.eva_memory_store.pop(oldest_key, None)
                    # also remove from phases
                    for ph in list(self.eva_phases.keys()):
                        self.eva_phases[ph].pop(oldest_key, None)
                except Exception:
                    # fallback: simple pop
                    self.eva_memory_store.pop(next(iter(self.eva_memory_store)), None)

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        """Returns a dictionary of EVA API methods exposed by the manager."""
        return {
            "record_experience": self.record_experience,
            "recall_experience": self.recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
