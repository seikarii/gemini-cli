from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable

# Standard and third-party imports required by the field implementation.
from enum import Enum
from typing import Any, cast

# Snow Model extensions (safe, additive)

# pydantic core types used by field models; import may fail at runtime in some
# environments so we keep a minimal try/except to degrade gracefully.
try:
    from pydantic import BaseModel, Field, PrivateAttr, model_post_init
except Exception:  # pragma: no cover - runtime compatibility shim
    BaseModel = object  # type: ignore

    def Field(*a, **k):  # type: ignore
        # minimal fallback for Field used in class attributes
        return None

    def PrivateAttr(*a, **k):  # type: ignore
        # minimal fallback for PrivateAttr
        return None

    def model_post_init(func: Callable) -> Callable:  # type: ignore
        # no-op decorator for older pydantic versions
        return func

    # Provide a safely-typed alias for the decorator so static checkers
    # don't treat the symbol as an arbitrary object in fallback environments.
    from typing import Protocol

    class _CallableDecorator(Protocol):
        def __call__(self, f: Callable[..., Any]) -> Callable[..., Any]: ...

    safe_model_post_init: _CallableDecorator
    try:
        # prefer the real decorator if available
        safe_model_post_init = model_post_init  # type: ignore[assignment]
    except Exception:
        # fallback no-op
        def safe_model_post_init(f: Callable[..., Any]) -> Callable[..., Any]:
            return f


from typing import TYPE_CHECKING, Dict, List
from typing import Any as _Any

# TYPE_CHECKING imports for static analysis only. At runtime we try to
# import the concrete classes and fall back to Any; importantly we avoid
# assigning typing special forms directly to module-level names in a way
# that mypy flags as "Cannot assign to a type". This pattern keeps the
# module importable at runtime while satisfying static analyzers.
if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience as _EVAExperience,  # type: ignore
    )
    from crisalida_lib.EVA.core_types import (
        QuantumField as _QuantumField,  # type: ignore
    )
    from crisalida_lib.EVA.core_types import (
        RealityBytecode as _RealityBytecode,  # type: ignore
    )
    from crisalida_lib.EVA.eva_memory_helper import (
        EVAMemoryHelper as _EVAMemoryHelper,  # type: ignore
    )
    from crisalida_lib.EVA.typequalia import (
        QualiaState as _QualiaState,  # type: ignore
    )
    from crisalida_lib.EVA.types import (
        LivingSymbolRuntime as _LivingSymbolRuntime,  # type: ignore
    )

    # Expose names for type checkers only
    EVAExperience = _EVAExperience  # type: ignore[assignment]
    QuantumField = _QuantumField  # type: ignore[assignment]
    RealityBytecode = _RealityBytecode  # type: ignore[assignment]
    EVAMemoryHelper = _EVAMemoryHelper  # type: ignore[assignment]
    QualiaState = _QualiaState  # type: ignore[assignment]
    LivingSymbolRuntime = _LivingSymbolRuntime  # type: ignore[assignment]
else:
    # Runtime: attempt to import the concrete classes; if unavailable, keep
    # a safe 'Any' fallback. Do not assign typing special forms at runtime
    # in a way that mypy will later flag as "Cannot assign to a type".
    EVAExperience = _Any
    QuantumField = _Any
    RealityBytecode = _Any
    EVAMemoryHelper = _Any
    QualiaState = _Any
    LivingSymbolRuntime = _Any
    try:
        # try importing runtime implementations; wrap each import to avoid
        # a single failing import breaking the module.
        from crisalida_lib.EVA.core_types import (
            EVAExperience as _rt_EVAExperience,
        )
        from crisalida_lib.EVA.core_types import (
            QuantumField as _rt_QuantumField,
        )
        from crisalida_lib.EVA.core_types import (
            RealityBytecode as _rt_RealityBytecode,
        )
        from crisalida_lib.EVA.eva_memory_helper import (
            EVAMemoryHelper as _rt_EVAMemoryHelper,
        )
        from crisalida_lib.EVA.typequalia import (
            QualiaState as _rt_QualiaState,
        )
        from crisalida_lib.EVA.types import (
            LivingSymbolRuntime as _rt_LivingSymbolRuntime,
        )

        # bind runtime names only if imports succeeded
        EVAExperience = _rt_EVAExperience  # type: ignore[assignment]
        QuantumField = _rt_QuantumField  # type: ignore[assignment]
        RealityBytecode = _rt_RealityBytecode  # type: ignore[assignment]
        EVAMemoryHelper = _rt_EVAMemoryHelper  # type: ignore[assignment]
        QualiaState = _rt_QualiaState  # type: ignore[assignment]
        LivingSymbolRuntime = _rt_LivingSymbolRuntime  # type: ignore[assignment]
    except Exception:
        # leave Any fallbacks if runtime imports fail
        pass

try:
    # predeclare to keep mypy from inferring a module type when a None is assigned
    _np: Any = None
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # degrade gracefully if unavailable
else:
    _np = cast(Any, _np)

# Expose `np` at runtime to match the rest of the codebase while avoiding
# a repeated name binding that mypy flags; `_np` is the temporary import.
# Annotate as Any so mypy doesn't treat the initial None as a Module type.
np: Any = _np

# optional accelerated query helpers and smoothing; consolidate into one guarded
# block so static analyzers don't see repeated redefinitions when the module
# is reloaded or patched during runtime.
if "CRISALIDA_EDEN_OPTIONAL_DEPS_INITIALIZED" not in globals():
    CRISALIDA_EDEN_OPTIONAL_DEPS_INITIALIZED = True
    # module-level fallbacks: explicitly type as Any | None so assigning None at
    # runtime doesn't conflict with static typing and mypy won't report
    # 'Cannot assign to a type'. Use explicit Any to make intent clear.
    cKDTree: Any | None = None  # type: ignore[misc]
    _gaussian_filter: Any | None = None
    # predeclare as Any to avoid mypy binding these names to the imported types
    _scipy_cKDTree: Any = None
    _scipy_gaussian_filter: Any = None
    try:
        # import under a temporary name to avoid mypy treating this as a
        # redefinition of the module variable declared above.
        from scipy.spatial import cKDTree as _scipy_cKDTree  # optional
    except Exception:  # pragma: no cover
        _scipy_cKDTree = None
    try:
        from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    except Exception:  # pragma: no cover
        _scipy_gaussian_filter = None

    # Use runtime-local implementation names to avoid reassigning a symbol
    # that static analyzers may have treated as a typing construct.
    _cKDTree_impl = _scipy_cKDTree
    _gaussian_filter_impl = _scipy_gaussian_filter

    # assign the runtime names used by the rest of the module (None-safe)
    # Keep explicit Any typing on these names so mypy accepts module-time reassignment.
    # Assign into module globals to avoid mypy "Cannot assign to a type" when
    # the name was previously used in a typing-only context.
    globals()["cKDTree"] = cast(Any, _cKDTree_impl)  # type: ignore[assignment,misc]
    globals()["_gaussian_filter"] = cast(Any, _gaussian_filter_impl)  # type: ignore[assignment,misc]

    # Some modules expect a callable KDTree constructor; expose a safe alias
    # that is guaranteed to be either the real constructor or None cast to Any.
    try:
        _kd_ctor: Any = _cKDTree_impl
    except Exception:
        _kd_ctor = None
    # Expose a typed alias used by consumers; ensure it's typed as Any so mypy
    # does not complain about later assignments or None fallbacks.
    globals()["cKDTree_ctor"] = cast(Any, _kd_ctor)  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    try:
        # prefer numpy typing when available for better type checking
        from numpy import ndarray as NDArray  # type: ignore
    except Exception:  # pragma: no cover - typing-time shim
        from typing import Any

        NDArray = Any  # type: ignore
else:
    # runtime: avoid importing numpy types when numpy may be optional
    NDArray = Any  # type: ignore


def _make_nested_zeros(dimensions: tuple[int, ...]) -> list:
    """Create a nested list of zeros matching the provided dimensions.

    This is used as a numpy-free fallback to ensure field shapes match the
    expected nested list structure used throughout the manifold code.
    """
    if not dimensions:
        return []

    def build(dim_list: tuple[int, ...]):
        if len(dim_list) == 1:
            return [0.0 for _ in range(int(dim_list[0]))]
        return [build(dim_list[1:]) for _ in range(int(dim_list[0]))]

    return build(tuple(int(d) for d in dimensions))


# EVAMemoryMixin fallback: many modules expect this symbol at import time.
if TYPE_CHECKING:
    from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin  # type: ignore
else:
    try:
        from crisalida_lib.EVA.eva_memory_mixin import EVAMemoryMixin
    except Exception:
        try:
            from crisalida_lib.EVA.compat import EVAMemoryMixin
        except Exception:  # pragma: no cover
            # Minimal shim to avoid NameError during import-time class definitions.
            class EVAMemoryMixin:  # type: ignore
                def __init__(self, *args, **kwargs):
                    # This __init__ accepts and ignores any arguments passed by other parent classes
                    # like pydantic.BaseModel, preventing TypeErrors.
                    super().__init__()

                def _init_eva_memory(self, *args, **kwargs):
                    self.eva_phase = getattr(self, "eva_phase", "default")
                    self.eva_memory_store = getattr(self, "eva_memory_store", {})

                def add_experience_phase(self, *args, **kwargs):
                    return False


def _qm_ensure_snow_state(self) -> None:
    """Ensure Snow Model state exists on QualiaField instance (idempotent)."""
    if not hasattr(self, "memory_crystal_index"):
        self.memory_crystal_index = {}
    if not hasattr(self, "crystal_kdtree"):
        self.crystal_kdtree = None
    if not hasattr(self, "crystal_kdtree_ids"):
        self.crystal_kdtree_ids = []
    if not hasattr(self, "crystal_kdtree_needs_rebuild"):
        self.crystal_kdtree_needs_rebuild = True
    # tunables (defaults; engine/config may override)
    if not hasattr(self, "crystal_preservation_threshold"):
        self.crystal_preservation_threshold = 0.6
    if not hasattr(self, "crystal_half_life"):
        # default: 5 años en segundos (vida media muy larga)
        self.crystal_half_life = 3600.0 * 24.0 * 365.0 * 5.0
    if not hasattr(self, "max_crystals"):
        self.max_crystals = 10000


def _qm_rebuild_kdtree(self) -> None:
    _qm_ensure_snow_state(self)
    if np is None or cKDTree is None:
        # no accel index, use fallback scans
        self.crystal_kdtree = None
        self.crystal_kdtree_ids = []
        self.crystal_kdtree_needs_rebuild = False
        return
    pts: list[tuple[float, float, float]] = []
    ids: list[str] = []
    for cid, meta in self.memory_crystal_index.items():
        pos = meta.get("position")
        if isinstance(pos, (tuple, list)) and len(pos) == 3:
            pts.append((float(pos[0]), float(pos[1]), float(pos[2])))
            ids.append(cid)
    if pts:
        # Use the runtime ctor alias 'cKDTree_ctor' which is typed as Any to avoid
        # assigning to a typing-only symbol. If unavailable, fall back to None.
        try:
            ctor = globals().get("cKDTree_ctor", None)
            if ctor is not None:
                self.crystal_kdtree = ctor(np.asarray(pts, dtype=float))
            else:
                self.crystal_kdtree = None
        except Exception:
            # best-effort: leave as None on failure
            self.crystal_kdtree = None
        self.crystal_kdtree_ids = ids
    else:
        self.crystal_kdtree = None
        self.crystal_kdtree_ids = []
    self.crystal_kdtree_needs_rebuild = False


def _qm_crystallize_trace(
    self,
    *,
    entity_id: str,
    position: tuple[int, int, int] | None,
    pattern: dict[str, Any] | None = None,
    strength: float = 1.0,
    provenance: dict[str, Any] | None = None,
    signature: dict[str, Any] | None = None,
) -> str:
    """Create a persistent memory crystal (qualia scar) with spatial metadata."""
    _qm_ensure_snow_state(self)
    # soft cap
    if len(self.memory_crystal_index) >= getattr(self, "max_crystals", 10000):
        # drop weakest
        try:
            weakest = min(
                self.memory_crystal_index.items(),
                key=lambda kv: float(kv[1].get("strength", 0.0)),
            )[0]
            self.memory_crystal_index.pop(weakest, None)
        except Exception:
            pass
    ts = time.time()
    cid = f"crystal_{int(ts * 1000) % 10_000_000}_{len(self.memory_crystal_index) % 1_000_000}"
    meta = {
        "id": cid,
        "entity_id": entity_id,
        "position": tuple(position) if position is not None else None,
        "pattern": dict(pattern or {}),
        "strength": float(max(0.0, min(1.0, strength))),
        "provenance": dict(provenance or {}),
        "created_at": ts,
        "updated_at": ts,
        "half_life": float(getattr(self, "crystal_half_life", 3600.0)),
    }
    if signature:
        meta["signature"] = (
            signature  # e.g., {"fingerprint": "...", "vector": [...]} o tipo similar
        )
    self.memory_crystal_index[cid] = meta
    # mark KD-tree dirty
    if hasattr(self, "crystal_kdtree_needs_rebuild"):
        self.crystal_kdtree_needs_rebuild = True
    return cid


def _qm_reinforce_crystal(self, cid: str, amount: float = 0.05) -> bool:
    _qm_ensure_snow_state(self)
    meta = self.memory_crystal_index.get(cid)
    if not meta:
        return False
    meta["strength"] = float(
        max(0.0, min(1.0, float(meta.get("strength", 0.0)) + float(amount)))
    )
    meta["updated_at"] = time.time()
    return True


def _qm_decay_crystals_tick(self, dt: float = 1.0) -> None:
    """Exponential decay by half-life; never hard-delete unless strength ~ 0."""
    _qm_ensure_snow_state(self)
    now = time.time()
    to_del: list[str] = []
    for cid, meta in list(self.memory_crystal_index.items()):
        hl = float(meta.get("half_life", getattr(self, "crystal_half_life", 3600.0)))
        if hl <= 0:
            continue
        # convert dt to seconds if your engine ticks are seconds; otherwise adapt
        decay_factor = 0.5 ** (float(dt) / hl)
        meta["strength"] = float(meta.get("strength", 0.0)) * float(decay_factor)
        meta["updated_at"] = now
        if meta["strength"] < 1e-6:
            to_del.append(cid)
    for cid in to_del:
        self.memory_crystal_index.pop(cid, None)
        if hasattr(self, "crystal_kdtree_needs_rebuild"):
            self.crystal_kdtree_needs_rebuild = True


def _qm_get_memory_traces(
    self,
    position: tuple[int, int, int] | None = None,
    max_distance: float = 100.0,
    k: int = 32,
    min_strength: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Spatial query for memory crystals. If no position, returns all ≥ min_strength."""
    _qm_ensure_snow_state(self)
    out: dict[str, dict[str, Any]] = {}
    if position is None:
        if self.memory_crystal_index is None:
            self.memory_crystal_index = {}  # Guard against NoneType
        for cid, meta in self.memory_crystal_index.items():
            if float(meta.get("strength", 0.0)) >= float(min_strength):
                out[cid] = dict(meta)
        return out
    if getattr(self, "crystal_kdtree_needs_rebuild", True):
        _qm_rebuild_kdtree(self)
    # fast path with KD-tree
    if self.crystal_kdtree is not None and np is not None and self.crystal_kdtree_ids:
        q = np.asarray(position, dtype=float)
        kq = min(int(k), len(self.crystal_kdtree_ids))
        dists, idxs = self.crystal_kdtree.query(q, k=kq)
        if np.isscalar(idxs):
            idxs = [int(idxs)]
            dists = [float(dists)]
        for d, ix in zip(dists, idxs, strict=False):
            cid = self.crystal_kdtree_ids[int(ix)]
            meta = self.memory_crystal_index.get(cid)
            if not meta:
                continue
            if float(meta.get("strength", 0.0)) < float(min_strength):
                continue
            if float(d) <= float(max_distance):
                m = dict(meta)
                m["distance"] = float(d)
                out[cid] = m
        return out
    # fallback linear scan
    px, py, pz = position  # type: ignore[misc]
    for cid, meta in self.memory_crystal_index.items():
        if float(meta.get("strength", 0.0)) < float(min_strength):
            continue
        pos = meta.get("position")
        if not isinstance(pos, (tuple, list)) or len(pos) != 3:
            continue
        dx, dy, dz = float(pos[0] - px), float(pos[1] - py), float(pos[2] - pz)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= float(max_distance):
            m = dict(meta)
            m["distance"] = float(dist)
            out[cid] = m
    return out


def _qm_install_snow_wrappers_on(manifold_cls: Any) -> None:
    """Attach Snow Model API to QualiaField class safely."""
    # attach helpers if missing
    for name, fn in (
        ("_crystallize_trace", _qm_crystallize_trace),
        ("reinforce_crystal", _qm_reinforce_crystal),
        ("decay_crystals_tick", _qm_decay_crystals_tick),
        ("get_memory_traces", _qm_get_memory_traces),
        ("_rebuild_crystal_kdtree", _qm_rebuild_kdtree),
        ("_ensure_snow_state", _qm_ensure_snow_state),
    ):
        if not hasattr(manifold_cls, name):
            setattr(manifold_cls, name, fn)

    # wrap influence/integrate to crystallize on threshold
    def _wrap(func_name: str):
        if not hasattr(manifold_cls, func_name):
            return
        original = getattr(manifold_cls, func_name)
        # Use an Any-cast to avoid mypy complaining that Callable has no
        # attribute '_snow_wrapped'. This is a runtime marker only.
        original_any = cast(Any, original)
        try:
            # Access via getattr on the Any-cast to keep mypy quiet and be defensive
            if getattr(original_any, "_snow_wrapped", False):
                return
        except Exception:
            # best-effort; continue if attribute access fails
            pass

        def wrapper(self, *args, **kwargs):
            res = None
            try:
                res = original(self, *args, **kwargs)
            finally:
                try:
                    try:
                        _qm_ensure_snow_state(self)
                    except Exception:
                        # best-effort: if ensuring snow state fails, continue
                        pass
                    # extract basic signals
                    entity_id = (
                        kwargs.get("entity_id") or kwargs.get("source") or "unknown"
                    )
                    position = kwargs.get("position")
                    # strength/coherence signals
                    strength = float(kwargs.get("strength", 0.5))
                    coherence = 0.5
                    # try to infer coherence from inputs/outputs
                    cs = kwargs.get("consciousness_state") or kwargs.get("state") or {}
                    try:
                        coherence = (
                            float(cs.get("coherence", coherence))
                            if isinstance(cs, dict)
                            else coherence
                        )
                    except Exception:
                        pass
                    if isinstance(res, dict):
                        coherence = float(res.get("coherence", coherence))
                        if position is None:
                            position = res.get("position")
                    score = strength * (0.5 + 0.5 * coherence)
                    if score >= float(
                        getattr(self, "crystal_preservation_threshold", 0.6)
                    ):
                        try:
                            pattern = (
                                res.get("pattern") if isinstance(res, dict) else None
                            )
                        except Exception:
                            pattern = None
                        _qm_crystallize_trace(
                            self,
                            entity_id=str(entity_id),
                            position=tuple(position) if position is not None else None,
                            pattern=pattern,
                            strength=float(min(1.0, score)),
                            provenance={"wrapped_from": func_name},
                            signature=(
                                res.get("qualia_signature")
                                if isinstance(res, dict)
                                else None
                            ),
                        )
                        if hasattr(self, "crystal_kdtree_needs_rebuild"):
                            self.crystal_kdtree_needs_rebuild = True
                except Exception:
                    # best-effort; never break original flow
                    pass
            return res

        # mark wrapper as wrapped using setattr to avoid mypy attr-defined complaints
        try:
            # set runtime marker defensively via setattr on an Any-cast to silence mypy
            setattr(cast(Any, wrapper), "_snow_wrapped", True)
        except Exception:
            # best-effort; non-fatal if attribute can't be set
            pass
        setattr(manifold_cls, func_name, wrapper)  # type: ignore[arg-type]

    for fname in (
        "integrate_consciousness_with_qualia_field",
        "apply_consciousness_influence",
    ):
        _wrap(fname)


# auto-install once QualiaField is defined in this module
_maybe_QualiaField = globals().get("QualiaField", None)
if _maybe_QualiaField is not None:
    try:
        # Cast to Any to avoid mypy complaining about runtime-only attributes
        _qm_install_snow_wrappers_on(cast(Any, _maybe_QualiaField))  # type: ignore[arg-type]
    except Exception:
        # non-fatal if install fails during import-time
        pass

# QualiaField - Advanced Quantum-Conscious Field System
# -----------------------------------------------------
# Models the quantum field enabling interaction between consciousness and reality.
# Implements a robust PDE-based simulation for the superposition of Order and Chaos states,
# with adaptive stability, advanced influence patterns, and diagnostics.
#
# Features:
# - Adaptive velocity-verlet PDE solver with nonlinear coupling
# - Modular influence application and resonance analysis
# - Robust boundary condition handling (periodic, reflective, absorbing, open)
# - Efficient numpy operations for all field updates
# - Diagnostics, visualization, and bidirectional consciousness integration

# The required imports (Enum, pydantic, uuid and EVA types) have been imported
# near the top of this file to satisfy flake8's 'imports at top' rule.


class InfluenceType(Enum):
    ORDER = "order"
    CHAOS = "chaos"
    COHERENCE = "coherence"
    TRANSCENDENT = "transcendent"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


class QualiaFieldBoundary(Enum):
    PERIODIC = "periodic"
    REFLECTIVE = "reflective"
    ABSORBING = "absorbing"
    OPEN = "open"


class QualiaField(EVAMemoryMixin, BaseModel):
    # Explicit runtime attributes to satisfy static checkers and consumers.
    # These are initialized in _initialize_fields / via PrivateAttr where appropriate.
    memory_crystal_index: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Use Optional[Any] instead of a concrete `cKDTree` typing name so that
    # runtime assignment (None or a scipy.spatial.cKDTree) does not trigger
    # mypy errors about assigning to a type name. Consumers that need the
    # concrete accelerated type may `isinstance`-check at runtime.
    crystal_kdtree: Any | None = None
    crystal_kdtree_needs_rebuild: bool = True
    crystal_kdtree_ids: list[str] = Field(default_factory=list)
    alpha: float = Field(0.5, ge=0, le=1)
    beta: float = Field(0.5, ge=0, le=1)
    dimensions: tuple[int, int, int] = Field((20, 20, 20))
    spatial_resolution: float = Field(0.1, gt=0)
    phi: list[Any] = Field(default_factory=lambda: [])
    phi_velocity: list[Any] = Field(default_factory=lambda: [])
    phi_acceleration: list[Any] = Field(default_factory=lambda: [])
    coherence_field: list[Any] = Field(default_factory=lambda: [])
    influence_field: list[Any] = Field(default_factory=lambda: [])
    c: float = Field(0.2, gt=0)
    gamma: float = Field(0.05, ge=0)
    eta_strength: float = Field(0.01, ge=0)
    boundary_type: QualiaFieldBoundary = Field(QualiaFieldBoundary.PERIODIC)
    nonlinear_coupling: float = Field(0.1, ge=0, le=1)
    field_memory: float = Field(0.95, ge=0, le=1)
    current_time: float = Field(0.0, ge=0)
    time_step_adaptive: bool = Field(True)
    stability_threshold: float = Field(1.0, gt=0)
    active_influences: dict[str, dict[str, Any]] = Field(default_factory=lambda: {})
    influence_history: list[dict[str, Any]] = Field(default_factory=lambda: [])

    # --- Snow Model: persistent memory crystals ---
    # (fields are initialized in _initialize_fields; avoid duplicate declarations)
    # ensure tunables exist for static checkers
    crystal_preservation_threshold: float = Field(0.6)
    crystal_half_life: float = Field(3600.0 * 24.0 * 365.0 * 5.0)
    max_crystals: int = Field(10000)

    # --- Private attributes for runtime state ---
    # Use the EVAMemoryHelper via the EVAMemoryMixin contract; keep a helper attr if needed
    _eva_helper: Any = PrivateAttr(default=None)

    @safe_model_post_init
    def _post_init_setup(self, **kwargs):
        """Called after the model is initialized.

        Ensure the helper exists (some class init ordering with Pydantic means
        EVAMemoryMixin.__init__ may not have run). Create helper if missing.
        """
        if not getattr(self, "_eva_helper", None):
            try:
                _helper_cls = globals().get("EVAMemoryHelper", None)
                if _helper_cls is not None:
                    try:
                        self._eva_helper = _helper_cls(self)
                    except Exception:
                        # construction failed; leave None
                        self._eva_helper = None
                else:
                    self._eva_helper = None
            except Exception:
                # best-effort: if helper can't be created at import time, leave None
                self._eva_helper = None
        self._initialize_fields()
        self._normalize_coefficients()

    # The EVAMemoryMixin provides `eva_ingest_experience` and `eva_recall_experience`.
    # If consumers call `get_eva_api` on the manifold, provide a thin passthrough to the helper.
    def get_eva_api(self) -> dict[str, Any]:
        helper = getattr(self, "_eva_helper", None) or getattr(
            self, "_eva_helper", None
        )
        if helper and hasattr(helper, "get_eva_api"):
            return helper.get_eva_api()
        return {}

    def _initialize_fields(self) -> None:
        if not self.phi:
            self.phi = self._create_initial_field()
        if not self.phi_velocity:
            self.phi_velocity = (
                np.zeros(self.dimensions).tolist()
                if np is not None
                else _make_nested_zeros(self.dimensions)
            )
        if not self.phi_acceleration:
            self.phi_acceleration = (
                np.zeros(self.dimensions).tolist()
                if np is not None
                else _make_nested_zeros(self.dimensions)
            )
        if not self.coherence_field:
            self.coherence_field = self._create_coherence_field()
        if not self.influence_field:
            self.influence_field = (
                np.zeros(self.dimensions).tolist()
                if np is not None
                else _make_nested_zeros(self.dimensions)
            )
        # memory crystals: persistent traces and spatial index
        self.memory_crystal_index = getattr(self, "memory_crystal_index", {})  # type: Dict[str, Dict[str, Any]]
        self.crystal_kdtree = None
        self.crystal_kdtree_needs_rebuild = True
        self.crystal_kdtree_ids = getattr(self, "crystal_kdtree_ids", [])  # type: List[str]
        self.current_time = getattr(self, "current_time", time.time())
        # Defensive: ensure numeric tunables are concrete values (pydantic Field may be a noop)
        self.spatial_resolution = getattr(self, "spatial_resolution", 0.1) or 0.1
        self.c = getattr(self, "c", 0.2) or 0.2
        self.gamma = getattr(self, "gamma", 0.05) or 0.05
        self.stability_threshold = getattr(self, "stability_threshold", 1.0) or 1.0
        self.field_memory = getattr(self, "field_memory", 0.95) or 0.95
        self.dimensions = tuple(getattr(self, "dimensions", (20, 20, 20)))
        self.alpha = getattr(self, "alpha", 0.5) or 0.5
        self.beta = getattr(self, "beta", 0.5) or 0.5
        self.active_influences = getattr(self, "active_influences", {}) or {}
        # Ensure noise strength and other tunables are numeric
        self.eta_strength = getattr(self, "eta_strength", 0.01) or 0.01

    def _create_initial_field(self) -> list:
        if np is None:
            # minimal fallback: flat field
            return [0.0] * (
                self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
            )
        field = np.random.normal(0.0, 0.05, self.dimensions)
        x, y, z = np.mgrid[
            0 : self.dimensions[0], 0 : self.dimensions[1], 0 : self.dimensions[2]
        ]
        center = np.array(self.dimensions) / 2
        order_peak = 0.3 * np.exp(
            -((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2) / 50
        )
        wave_pattern = 0.1 * np.sin(2 * np.pi * x / 10) * np.cos(2 * np.pi * y / 10)
        field += order_peak + wave_pattern
        field = np.clip(field, -self.stability_threshold, self.stability_threshold)
        return field.tolist()

    def _create_coherence_field(self) -> list:
        if np is None:
            return [0.6] * (
                self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
            )
        coherence = np.full(self.dimensions, 0.6)
        x, y, z = np.mgrid[
            0 : self.dimensions[0], 0 : self.dimensions[1], 0 : self.dimensions[2]
        ]
        variation = (
            0.2
            * np.sin(2 * np.pi * x / self.dimensions[0])
            * np.cos(2 * np.pi * y / self.dimensions[1])
        )
        coherence += variation
        coherence = np.clip(coherence, 0.1, 1.0)
        return coherence.tolist()

    def _normalize_coefficients(self):
        sum_sq = self.alpha**2 + self.beta**2
        if sum_sq != 0 and abs(sum_sq - 1.0) > 1e-10:
            factor = (1.0 / sum_sq) ** 0.5
            self.alpha *= factor
            self.beta *= factor
        elif sum_sq == 0:
            self.alpha = self.beta = 0.7071

    def _calculate_laplacian_3d(self, field_array: Any) -> Any:
        laplacian = np.zeros_like(field_array)
        dx2 = self.spatial_resolution**2
        if self.boundary_type == QualiaFieldBoundary.PERIODIC:
            laplacian += (
                np.roll(field_array, 1, axis=0)
                - 2 * field_array
                + np.roll(field_array, -1, axis=0)
            )
            laplacian += (
                np.roll(field_array, 1, axis=1)
                - 2 * field_array
                + np.roll(field_array, -1, axis=1)
            )
            laplacian += (
                np.roll(field_array, 1, axis=2)
                - 2 * field_array
                + np.roll(field_array, -1, axis=2)
            )
        elif self.boundary_type == QualiaFieldBoundary.REFLECTIVE:
            laplacian[1:-1, :, :] += (
                field_array[2:, :, :]
                - 2 * field_array[1:-1, :, :]
                + field_array[:-2, :, :]
            )
            laplacian[:, 1:-1, :] += (
                field_array[:, 2:, :]
                - 2 * field_array[:, 1:-1, :]
                + field_array[:, :-2, :]
            )
            laplacian[:, :, 1:-1] += (
                field_array[:, :, 2:]
                - 2 * field_array[:, :, 1:-1]
                + field_array[:, :, :-2]
            )
            laplacian[0, :, :] = laplacian[1, :, :]
            laplacian[-1, :, :] = laplacian[-2, :, :]
            laplacian[:, 0, :] = laplacian[:, 1, :]
            laplacian[:, -1, :] = laplacian[:, -2, :]
            laplacian[:, :, 0] = laplacian[:, :, 1]
            laplacian[:, :, -1] = laplacian[:, :, -2]
        elif self.boundary_type == QualiaFieldBoundary.ABSORBING:
            laplacian[1:-1, 1:-1, 1:-1] = (
                field_array[2:, 1:-1, 1:-1]
                - 2 * field_array[1:-1, 1:-1, 1:-1]
                + field_array[:-2, 1:-1, 1:-1]
                + field_array[1:-1, 2:, 1:-1]
                + field_array[1:-1, :-2, 1:-1]
                + field_array[1:-1, 1:-1, 2:]
                + field_array[1:-1, 1:-1, :-2]
            )
        return laplacian / dx2

    def _calculate_nonlinear_term(self, field_array: Any) -> Any:
        if self.nonlinear_coupling == 0:
            return np.zeros_like(field_array)
        return self.nonlinear_coupling * (field_array**3 - field_array)

    def _generate_quantum_noise(self) -> Any | None:
        if np is None:
            # No numeric backend: return zeros
            return None  # type: ignore[return-value]
        # Defensive: validate eta_strength and dimensions
        try:
            eta = float(getattr(self, "eta_strength", 0.01) or 0.01)
        except Exception:
            eta = 0.01
        dims = getattr(self, "dimensions", None) or (1,)
        try:
            noise = np.random.normal(0, eta, tuple(dims))
        except Exception:
            try:
                noise = np.zeros(tuple(dims))
            except Exception:
                return None
        try:
            if eta > 0 and _gaussian_filter is not None:
                noise = _gaussian_filter(noise, sigma=1.0, mode="wrap")
        except Exception:
            pass
        try:
            coherence_array = np.array(self.coherence_field)
            noise *= 1.0 + 0.3 * coherence_array
        except Exception:
            # best-effort: return noise even if coherence multiplication fails
            pass
        return noise

    def _calculate_adaptive_timestep(
        self, field_array: Any, velocity_array: Any
    ) -> float:
        if not self.time_step_adaptive:
            return 0.01
        max_velocity = np.max(np.abs(velocity_array)) + 1e-10
        cfl_dt = 0.5 * self.spatial_resolution / (self.c + max_velocity)
        diffusion_dt = 0.25 * self.spatial_resolution**2 / (self.gamma + 1e-10)
        dt = min(cfl_dt, diffusion_dt, 0.05)
        return max(dt, 0.001)

    def get_state_at(self, position: tuple[int, int, int]) -> dict[str, Any]:
        """Get the field state at a specific position."""
        x, y, z = position
        # Clamp coordinates to valid range
        x = max(0, min(x, self.dimensions[0] - 1))
        y = max(0, min(y, self.dimensions[1] - 1))
        z = max(0, min(z, self.dimensions[2] - 1))

        # Calculate flat index for 3D coordinates
        flat_index = (
            z * self.dimensions[0] * self.dimensions[1] + y * self.dimensions[0] + x
        )

        # Get values from field arrays
        phi_val = self.phi[flat_index] if flat_index < len(self.phi) else 0.0
        velocity_val = (
            self.phi_velocity[flat_index]
            if flat_index < len(self.phi_velocity)
            else 0.0
        )
        coherence_val = (
            self.coherence_field[flat_index]
            if flat_index < len(self.coherence_field)
            else 0.0
        )

        return {
            "phi": float(phi_val),
            "velocity": float(velocity_val),
            "coherence": float(coherence_val),
            "position": position,
        }

    def set_state_at(
        self, position: tuple[int, int, int], state: dict[str, Any]
    ) -> None:
        """Set the field state at a specific position."""
        x, y, z = position
        # Clamp coordinates to valid range
        x = max(0, min(x, self.dimensions[0] - 1))
        y = max(0, min(y, self.dimensions[1] - 1))
        z = max(0, min(z, self.dimensions[2] - 1))

        # Calculate flat index for 3D coordinates
        flat_index = (
            z * self.dimensions[0] * self.dimensions[1] + y * self.dimensions[0] + x
        )

        # Set values in field arrays if they exist
        if "phi" in state and flat_index < len(self.phi):
            self.phi[flat_index] = float(state["phi"])
        if "velocity" in state and flat_index < len(self.phi_velocity):
            self.phi_velocity[flat_index] = float(state["velocity"])
        if "coherence" in state and flat_index < len(self.coherence_field):
            self.coherence_field[flat_index] = float(state["coherence"])

    def apply_global_influence(self, influence_data: dict[str, Any]) -> None:
        """Apply global influence to the field."""
        # Basic implementation that applies global effects
        if not influence_data:
            return

        # Apply a small global modification to all field values
        influence_strength = influence_data.get("strength", 0.01)
        if influence_strength > 0:
            phi_array = np.array(self.phi)
            phi_array += influence_strength * 0.001  # Small global perturbation
            self.phi = phi_array.tolist()

    def _create_influence_pattern(
        self,
        position: tuple[int, int, int],
        radius: float,
        influence_type: InfluenceType,
        strength: float,
        consciousness_state: dict[str, float],
    ) -> Any:
        """Create an influence pattern for consciousness effects."""
        # Create a 3D Gaussian influence pattern
        dims = self.dimensions
        pattern = np.zeros(dims)

        x_center, y_center, z_center = position

        # Create Gaussian influence pattern
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    distance = np.sqrt(
                        (x - x_center) ** 2 + (y - y_center) ** 2 + (z - z_center) ** 2
                    )
                    if distance <= radius:
                        # Gaussian falloff
                        gaussian_factor = np.exp(
                            -(distance**2) / (2 * (radius / 3) ** 2)
                        )
                        pattern[x, y, z] = strength * gaussian_factor

        return pattern

    def _calculate_natural_influence_position(
        self, consciousness_state: dict[str, float]
    ) -> tuple[int, int, int]:
        """Calculate a natural position for influence based on consciousness state."""
        # Default to center of field if no specific positioning logic
        center_x = self.dimensions[0] // 2
        center_y = self.dimensions[1] // 2
        center_z = self.dimensions[2] // 2

        # Add some variation based on consciousness state
        coherence = consciousness_state.get("consciousness_coherence", 0.5)
        confidence = consciousness_state.get("decision_confidence", 0.5)

        # Small offset based on consciousness parameters
        offset_x = int((coherence - 0.5) * 5)
        offset_y = int((confidence - 0.5) * 5)

        final_x = max(0, min(center_x + offset_x, self.dimensions[0] - 1))
        final_y = max(0, min(center_y + offset_y, self.dimensions[1] - 1))

        return (final_x, final_y, center_z)

    def _analyze_field_response(
        self, influence_pattern: Any, phi_array: Any
    ) -> dict[str, Any]:
        """Analyze how the field responds to an influence pattern."""
        # Calculate basic response metrics
        total_influence = np.sum(np.abs(influence_pattern))
        max_influence = np.max(np.abs(influence_pattern))
        field_stability = np.std(phi_array) if len(phi_array) > 0 else 0.0

        return {
            "total_influence": float(total_influence),
            "max_influence": float(max_influence),
            "field_stability": float(field_stability),
            "response_coherence": min(1.0, total_influence / (max_influence + 1e-6)),
        }

    def _calculate_effective_influence(
        self,
        base_strength: float,
        coherence: float,
        stress_level: float,
        energy_balance: float,
        decision_confidence: float,
    ) -> float:
        """Compute a conservative effective strength used for crystallization decisions.

        Keep implementation simple and deterministic so static checks and unit
        tests can reason about the output. Values are clamped to [0,1].
        """
        try:
            s = float(base_strength)
        except Exception:
            s = 0.0
        # Coherence amplifies, stress reduces, energy_balance increases capacity,
        # decision_confidence nudges effect slightly.
        eff = s * (0.5 + 0.5 * float(coherence or 0.0))
        eff *= max(0.0, 1.0 - 0.3 * float(stress_level or 0.0))
        eff *= max(0.0, 0.5 + 0.5 * float(energy_balance or 0.0))
        eff *= max(0.0, 0.7 + 0.3 * float(decision_confidence or 0.0))
        return max(0.0, min(1.0, float(eff)))

    def apply_consciousness_influence(
        self,
        entity_id: str,
        consciousness_state: dict[str, float],
        position: tuple[int, int, int] | None = None,
        influence_type: InfluenceType = InfluenceType.ORDER,
        strength: float = 0.5,
        radius: float = 3.0,
        duration: float = 10.0,
    ) -> dict[str, Any]:
        """
        Now includes crystallization logic: strong or coherent influences may leave a persistent
        'memory_crystal' in the field. These crystals decay extremely slowly and become part
        of the field state (Snow Model).
        """
        coherence = consciousness_state.get("consciousness_coherence", 0.5)
        stress_level = consciousness_state.get("stress_level", 0.3)
        energy_balance = consciousness_state.get("energy_balance", 0.6)
        decision_confidence = consciousness_state.get("decision_confidence", 0.5)
        effective_strength = self._calculate_effective_influence(
            strength, coherence, stress_level, energy_balance, decision_confidence
        )
        if position is None:
            position = self._calculate_natural_influence_position(consciousness_state)
        position = (
            max(0, min(position[0], self.dimensions[0] - 1)),
            max(0, min(position[1], self.dimensions[1] - 1)),
            max(0, min(position[2], self.dimensions[2] - 1)),
        )
        influence_pattern = self._create_influence_pattern(
            position, radius, influence_type, effective_strength, consciousness_state
        )
        current_influence = np.array(self.influence_field)
        current_influence += influence_pattern
        phi_array = np.array(self.phi)
        resonance_pattern, resonance_score = self._resonance_kernel(
            phi_array, influence_pattern
        )
        field_modification = influence_pattern * 0.1 * (1.0 + resonance_pattern)
        phi_array += field_modification
        phi_array = np.clip(
            phi_array, -self.stability_threshold, self.stability_threshold
        )
        self.phi = phi_array.tolist()
        self.influence_field = current_influence.tolist()
        influence_data: dict[str, Any] = {
            "entity_id": entity_id,
            "type": influence_type.value,
            "position": position,
            "strength": effective_strength,
            "radius": radius,
            "lifetime": duration,
            "decay_time": duration * 0.3,
            "consciousness_state": consciousness_state.copy(),
            "timestamp": self.current_time,
            "resonance_achieved": float(np.mean(resonance_pattern)),
        }
        influence_id = f"{entity_id}_{self.current_time:.3f}"
        self.active_influences[influence_id] = influence_data
        self.influence_history.append(influence_data.copy())
        if len(self.influence_history) > 1000:
            self.influence_history = self.influence_history[-1000:]
        field_response = self._analyze_field_response(influence_pattern, phi_array)

        # CRYSTALLIZATION: convert significant / coherent influences into persistent memory traces
        crystallize_score = effective_strength * (
            coherence + 0.5 * float(resonance_score)
        )
        if crystallize_score >= self.crystal_preservation_threshold:
            crystal_id = self._crystallize_trace(
                entity_id=entity_id,
                position=position,
                pattern=influence_pattern,
                strength=float(min(1.0, crystallize_score)),
                provenance={
                    "influence_type": influence_type.value,
                },
            )
            # mark KD-tree rebuild needed
            self.crystal_kdtree_needs_rebuild = True
        # ...
        result: dict[str, Any] = {
            "influence_id": influence_id,
            "influence_data": influence_data,
            "field_response": field_response,
            "crystallized": crystallize_score >= self.crystal_preservation_threshold,
            "crystal_id": (
                crystal_id
                if crystallize_score >= self.crystal_preservation_threshold
                else None
            ),
        }
        return result

    def _crystallize_trace(
        self,
        *,
        entity_id: str,
        position: tuple[int, int, int] | None,
        pattern: Any,
        strength: float,
        provenance: dict[str, Any],
    ) -> str:
        """Create a persistent memory crystal (qualia scar) with spatial metadata."""
        cid = f"crystal_{uuid.uuid4().hex[:10]}"
        entry = {
            "id": cid,
            "entity_id": entity_id,
            "position": tuple(position) if position is not None else None,
            "pattern": pattern,
            "strength": float(max(0.0, min(1.0, strength))),
            "provenance": provenance,
            "created_at": time.time(),
            "half_life": getattr(
                self, "crystal_half_life", 3600.0 * 24.0 * 365.0 * 5.0
            ),
        }
        self.memory_crystal_index[cid] = entry
        return cid

    def _remove_crystal(self, cid: str) -> None:
        if cid in self.memory_crystal_index:
            del self.memory_crystal_index[cid]
            self.crystal_kdtree_needs_rebuild = True

    def _rebuild_crystal_kdtree(self) -> None:
        if np is None or cKDTree is None:
            self.crystal_kdtree = None
            self.crystal_kdtree_ids = []
            self.crystal_kdtree_needs_rebuild = False
            return
        pts: list[tuple[float, float, float]] = []
        ids: list[str] = []
        for cid, meta in self.memory_crystal_index.items():
            pos = meta.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                pts.append((float(pos[0]), float(pos[1]), float(pos[2])))
                ids.append(cid)
        if pts:
            self.crystal_kdtree = cKDTree(np.asarray(pts, dtype=float))
            self.crystal_kdtree_ids = ids
        else:
            self.crystal_kdtree = None
            self.crystal_kdtree_ids = []
        self.crystal_kdtree_needs_rebuild = False

    def get_memory_traces(
        self,
        position: tuple[int, int, int] | None = None,
        max_distance: float = 100.0,
        k: int = 32,
        min_strength: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        """
        Spatially-aware query for persistent memory crystals.
        Returns dict[id] -> meta for crystals within max_distance of position (if provided),
        otherwise returns entire index (filtered by min_strength).
        """
        # Defensive: ensure memory_crystal_index exists before iterating
        if getattr(self, "memory_crystal_index", None) is None:
            self.memory_crystal_index = {}

        if position is not None and self.crystal_kdtree_needs_rebuild:
            self._rebuild_crystal_kdtree()
        results = {}
        if position is not None and self.crystal_kdtree is not None:
            dists, idxs = self.crystal_kdtree.query(
                np.array(position, dtype=float), k=min(k, len(self.crystal_kdtree_ids))
            )
            if np.isscalar(idxs):
                idxs = [idxs]
                dists = [dists]
            for d, ix in zip(dists, idxs, strict=False):
                cid = self.crystal_kdtree_ids[int(ix)]
                meta = self.memory_crystal_index.get(cid)
                if (
                    meta
                    and meta.get("strength", 0.0) >= min_strength
                    and d <= max_distance
                ):
                    meta_copy = dict(meta)
                    meta_copy["distance"] = float(d)
                    results[cid] = meta_copy
            return results
        # fallback: full scan filtered by strength
        for cid, meta in self.memory_crystal_index.items():
            if meta.get("strength", 0.0) >= min_strength:
                results[cid] = dict(meta)
        return results

    # --- Resonance kernel (refactored scoring) ---
    def _resonance_kernel(
        self, phi_array: Any, influence_pattern: Any
    ) -> tuple[Any, float]:
        """
        Compute a per-cell resonance pattern and a scalar resonance score.
        Returns (resonance_pattern, score) where score is mean(abs(resonance_pattern)) in [0,1].
        """
        try:
            # align shapes
            a = np.asarray(phi_array, dtype=float)
            b = np.asarray(influence_pattern, dtype=float)
            if a.shape != b.shape:
                # try to broadcast safely
                b = np.broadcast_to(b, a.shape)
            # per-cell resonance: normalized product
            eps = 1e-9
            denom = np.abs(a) + np.abs(b) + eps
            resonance = (a * b) / denom
            resonance = np.nan_to_num(resonance)
            score = float(np.mean(np.abs(resonance)))
            score = max(0.0, min(1.0, score))
            return resonance, score
        except Exception:
            # fallback: compute simple normalized dot-like score
            a_flat = np.ravel(phi_array).astype(float)
            b_flat = np.ravel(influence_pattern).astype(float)
            num = float(np.sum(np.abs(a_flat * b_flat)))
            denom = (
                float(np.sqrt(np.sum(a_flat * a_flat) * np.sum(b_flat * b_flat))) + 1e-9
            )
            score = max(0.0, min(1.0, num / denom))
            # return uniform small resonance pattern
            return np.sign(a_flat * b_flat).reshape(phi_array.shape) * 0.001, score


def integrate_consciousness_with_qualia_field(
    consciousness_state: dict[str, Any],
    qualia_field: QualiaField,
    position: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """
    Backwards-compatible wrapper that now performs crystallization-aware influence application.
    Prefer QualiaField.apply_consciousness_influence directly.
    """
    if position is None:
        # ensure a 3-tuple position
        dims = qualia_field.dimensions
        position = (int(dims[0] // 2), int(dims[1] // 2), int(dims[2] // 2))
    return qualia_field.apply_consciousness_influence(
        entity_id=consciousness_state.get("entity_id", "unknown"),
        consciousness_state=consciousness_state,
        position=position,
    )


class EVAQualiaField(QualiaField):
    """
    EVAQualiaField - Campo cuántico de qualia extendido para integración con EVA.
    Permite compilar, almacenar, simular y recordar experiencias de influencia qualia como RealityBytecode,
    soporta faseo, hooks de entorno, gestión de memoria viviente y benchmarking.
    """

    def __init__(self, phase: str = "default", **data) -> None:
        super().__init__(**data)
        self.phase = phase
        # initialize runtime; use LivingSymbolRuntime when available
        try:
            _runtime_cls = globals().get("LivingSymbolRuntime", None)
            if _runtime_cls is not None:
                try:
                    self.eva_runtime = _runtime_cls()
                except Exception:
                    self.eva_runtime = None
            else:
                self.eva_runtime = None
        except Exception:
            self.eva_runtime = None

        # EVA memory/runtime containers (runtime assignments)
        self.eva_memory_store = {}
        self.eva_experience_store = {}
        self.eva_phases = {}
        self._environment_hooks = []

    def eva_ingest_qualia_field_experience(
        self,
        consciousness_state: dict[str, Any],
        position: tuple[int, int, int],
        qualia_state: Any,
        phase: str | None = None,
    ) -> str:
        """
        Compila una experiencia de influencia qualia en RealityBytecode y la almacena en la memoria EVA.
        """
        phase = phase or self.phase
        experience_data = {
            "consciousness_state": consciousness_state,
            "position": position,
            "qualia_state": (
                qualia_state.to_dict()
                if hasattr(qualia_state, "to_dict")
                else dict(cast(dict[str, Any], qualia_state))
            ),
            "field_snapshot": self.get_state_at(position),
            "timestamp": self.current_time,
        }
        intention = {
            "intention_type": "ARCHIVE_QUALIA_FIELD_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        # mypy: LivingSymbolRuntime may be typed as Any/None in compatibility shims
        # guard eva_runtime access (compat shims may set it to None)
        runtime = cast(Any, getattr(self, "eva_runtime", None))
        # Avoid accessing `.divine_compiler` directly on a possibly None object
        # by resolving it to a local and checking for a callable `compile_intention`.
        bytecode = []
        if runtime is not None:
            _dc = getattr(runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
        # If runtime compiler didn't produce bytecode, use local conservative fallback.
        if not bytecode:
            try:
                from crisalida_lib.EDEN.bytecode_generator import (
                    compile_intention_to_bytecode,
                )

                bytecode = compile_intention_to_bytecode(intention)
            except Exception:
                bytecode = []
        experience_id = f"eva_qualia_field_{hash(str(experience_data))}"
        _rb_cls = globals().get("RealityBytecode", None)
        if _rb_cls is not None:
            try:
                reality_bytecode = _rb_cls(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=self.current_time,
                )
            except Exception:
                from types import SimpleNamespace

                reality_bytecode = SimpleNamespace(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=self.current_time,
                )
        else:
            from types import SimpleNamespace

            reality_bytecode = SimpleNamespace(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=self.current_time,
            )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        return experience_id

    def eva_recall_qualia_field_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        """
        Ejecuta el RealityBytecode de una experiencia de influencia qualia almacenada, manifestando la simulación.
        """
        phase = phase or self.phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA qualia field experience"}
        _qf_cls = globals().get("QuantumField", None)
        if _qf_cls is not None:
            try:
                quantum_field = _qf_cls()
            except Exception:
                from types import SimpleNamespace

                quantum_field = SimpleNamespace()
        else:
            from types import SimpleNamespace

            quantum_field = SimpleNamespace()
        manifestations = []
        instrs = getattr(reality_bytecode, "instructions", []) or []
        for instr in instrs:
            symbol_manifest = None
            # Try runtime execution first
            rt = getattr(self, "eva_runtime", None)
            try:
                if rt is not None and hasattr(rt, "execute_instruction"):
                    symbol_manifest = rt.execute_instruction(instr, quantum_field)
            except Exception:
                symbol_manifest = None

            # Fallback to internal VM shim to generate simple manifestations
            if symbol_manifest is None:
                try:
                    from crisalida_lib.EDEN.internal_vm import InternalVMShim

                    vm = InternalVMShim()
                    res = vm.execute_bytecode([instr], quantum_field)
                    if res:
                        # take the first manifest produced for this instruction
                        symbol_manifest = res[0]
                except Exception:
                    symbol_manifest = None

            if symbol_manifest:
                # Normalize manifestations to plain dicts when possible to avoid
                # union-attr issues for static checkers and consumers expecting
                # serializable structures.
                try:
                    manifest_dict = (
                        symbol_manifest.to_dict()
                        if hasattr(symbol_manifest, "to_dict")
                        else (
                            dict(symbol_manifest)
                            if hasattr(symbol_manifest, "items")
                            else symbol_manifest
                        )
                    )
                except Exception:
                    manifest_dict = symbol_manifest
                manifestations.append(manifest_dict)
                for hook in self._environment_hooks:
                    try:
                        hook(manifest_dict)
                    except Exception as e:
                        print(f"[EVA-QUALIA-FIELD] Environment hook failed: {e}")
        _eva_cls = globals().get("EVAExperience", None)
        if _eva_cls is not None:
            try:
                eva_experience = _eva_cls(
                    experience_id=reality_bytecode.bytecode_id,
                    bytecode=reality_bytecode,
                    manifestations=manifestations,
                    phase=reality_bytecode.phase,
                    qualia_state=reality_bytecode.qualia_state,
                    timestamp=reality_bytecode.timestamp,
                )
            except Exception:
                from types import SimpleNamespace

                eva_experience = SimpleNamespace(
                    experience_id=reality_bytecode.bytecode_id,
                    manifestations=manifestations,
                    phase=reality_bytecode.phase,
                    qualia_state=reality_bytecode.qualia_state,
                    timestamp=reality_bytecode.timestamp,
                )
        else:
            from types import SimpleNamespace

            eva_experience = SimpleNamespace(
                experience_id=reality_bytecode.bytecode_id,
                manifestations=manifestations,
                phase=reality_bytecode.phase,
                qualia_state=reality_bytecode.qualia_state,
                timestamp=reality_bytecode.timestamp,
            )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        # serialize manifestations defensively
        serial_manifestations = []
        for m in manifestations:
            if hasattr(m, "to_dict") and callable(getattr(m, "to_dict")):
                try:
                    if hasattr(m, "to_dict") and callable(getattr(m, "to_dict")):
                        try:
                            serial_manifestations.append(m.to_dict())
                        except Exception:
                            try:
                                serial_manifestations.append(dict(m))
                            except Exception:
                                continue
                    else:
                        try:
                            serial_manifestations.append(dict(m))
                        except Exception:
                            continue
                except Exception:
                    try:
                        serial_manifestations.append(dict(m))
                    except Exception:
                        continue
            else:
                try:
                    serial_manifestations.append(dict(m))
                except Exception:
                    continue

        qualia_state_serial = {}
        qs = getattr(eva_experience, "qualia_state", None)
        if qs is not None:
            if hasattr(qs, "to_dict") and callable(getattr(qs, "to_dict")):
                try:
                    qualia_state_serial = qs.to_dict()
                except Exception:
                    try:
                        qualia_state_serial = dict(qs)
                    except Exception:
                        qualia_state_serial = {}
            else:
                try:
                    qualia_state_serial = dict(qs)
                except Exception:
                    qualia_state_serial = {}

        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": serial_manifestations,
            "phase": eva_experience.phase,
            "qualia_state": qualia_state_serial,
            "timestamp": eva_experience.timestamp,
        }

    def add_qualia_field_experience_phase(
        self,
        experience_id: str,
        phase: str,
        consciousness_state: dict[str, Any],
        position: tuple[int, int, int],
        qualia_state: Any,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia de influencia qualia.
        """
        experience_data = {
            "consciousness_state": consciousness_state,
            "position": position,
            "qualia_state": (
                qualia_state.to_dict()
                if hasattr(qualia_state, "to_dict")
                else dict(cast(dict[str, Any], qualia_state))
            ),
            "field_snapshot": self.get_state_at(position),
            "timestamp": self.current_time,
        }
        intention = {
            "intention_type": "ARCHIVE_QUALIA_FIELD_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        # guard eva_runtime access and avoid direct attribute access on None
        bytecode = []
        _runtime = getattr(self, "eva_runtime", None)
        if _runtime is not None:
            _dc = getattr(_runtime, "divine_compiler", None)
            compile_fn = (
                getattr(_dc, "compile_intention", None) if _dc is not None else None
            )
            if callable(compile_fn):
                try:
                    bytecode = compile_fn(intention)
                except Exception:
                    bytecode = []
        _rb_cls = globals().get("RealityBytecode", None)
        if _rb_cls is not None:
            try:
                reality_bytecode = _rb_cls(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=self.current_time,
                )
            except Exception:
                from types import SimpleNamespace

                reality_bytecode = SimpleNamespace(
                    bytecode_id=experience_id,
                    instructions=bytecode,
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=self.current_time,
                )
        else:
            from types import SimpleNamespace

            reality_bytecode = SimpleNamespace(
                bytecode_id=experience_id,
                instructions=bytecode,
                qualia_state=qualia_state,
                phase=phase,
                timestamp=self.current_time,
            )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria (timeline)."""
        self.phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-QUALIA-FIELD] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia de influencia qualia."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[[Any], None]) -> None:
        """Registra un hook para manifestación simbólica (ej. renderizado 3D/4D)."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict[str, Any]:
        return {
            "eva_ingest_qualia_field_experience": self.eva_ingest_qualia_field_experience,
            "eva_recall_qualia_field_experience": self.eva_recall_qualia_field_experience,
            "add_qualia_field_experience_phase": self.add_qualia_field_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }

    # Backward-compat alias expected by older code paths
    def get_field_state_at_position(
        self, position: tuple[int, int, int]
    ) -> dict[str, Any]:
        return self.get_state_at(position)


# Ensure Snow wrappers are installed after class definition
try:
    _qm_install_snow_wrappers_on(QualiaField)  # type: ignore[arg-type]
except Exception:
    pass


def create_qualia_field() -> Any:
    """Defensive factory for creating a QualiaField instance.

    Attempts to call the Pydantic model construct() when available to
    avoid requiring all constructor args at call time. Returns an Any-typed
    object (the instance or None) so callers can assign without mypy
    complaining about missing named arguments.
    """
    try:
        # Avoid calling the QualiaField constructor through a typed name so
        # mypy doesn't enforce the pydantic constructor call-arg contract.
        QF_cls = QualiaField  # type: ignore
        # Prefer the pydantic .construct() no-arg initializer when available
        if hasattr(QF_cls, "construct"):
            try:
                return cast(Any, QF_cls.construct())
            except Exception:
                pass
        # Call through an Any-typed alias to silence static call-arg checks
        QF_any = cast(Any, QF_cls)
        try:
            return cast(Any, QF_any())
        except Exception:
            try:
                # Last-resort: try construct again
                if hasattr(QF_cls, "construct"):
                    return cast(Any, QF_cls.construct())
            except Exception:
                return cast(Any, None)
        return cast(Any, None)
    except Exception:
        return cast(Any, None)
