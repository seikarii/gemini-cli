"""
PhysicsEngine - Core Physics Orchestrator for the Metacosmos
================================================================
Orchestrates the entire physics simulation loop, integrating the QualiaManifold,
LivingSymbols, CosmicLattice, and the OntologicalVirtualMachine.
"""

# Standard imports
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

# Typing-only imports to avoid import-time cycles; populate real symbols at runtime in __init__
if TYPE_CHECKING:
    from crisalida_lib.EARTH.self_modifying_engine import EVASelfModifyingEngine
    from crisalida_lib.EDEN.living_symbol import (
        DivineSignature,
        LivingSymbol,
        MovementPattern,
    )
    from crisalida_lib.EDEN.qualia_manifold import QualiaField
    from crisalida_lib.EVA.language.grammar import eva_grammar_engine
    from crisalida_lib.EVA.typequalia import QualiaState
    from crisalida_lib.EVA.types import LivingSymbolRuntime

    from .cosmic_lattice import CosmicLattice
    from .virtual_machine import OntologicalVirtualMachine
else:
    eva_grammar_engine = Any
    QualiaState = Any
    LivingSymbolRuntime = Any
    EVASelfModifyingEngine = Any
    DivineSignature = Any
    LivingSymbol = Any
    MovementPattern = Any
    QualiaField = Any
    CosmicLattice = Any
    OntologicalVirtualMachine = Any

# Runtime alias to avoid mypy enforcing typing-only QualiaField constructor requirements
QualiaFieldRuntime = QualiaField  # type: ignore

# EVAMemoryMixin may be provided by runtime modules; to avoid mypy
# incompatible-import and redefinition warnings we keep a typing-friendly
# module-level name but resolve the concrete runtime class into
# `EVAMemoryMixin_runtime` which is used at runtime when available.
EVAMemoryMixin: Any = Any
EVAMemoryMixin_runtime: Any = None
try:
    # Use importlib to perform a runtime import so static analyzers don't infer
    # a concrete type and raise incompatible-import errors across compatibility
    # shim modules.
    import importlib

    _mod = importlib.import_module("crisalida_lib.EVA.eva_memory_mixin")
    _rt_EVAMemoryMixin = getattr(_mod, "EVAMemoryMixin", None)
    EVAMemoryMixin_runtime = cast(Any, _rt_EVAMemoryMixin)
except Exception:
    try:
        import importlib

        _mod = importlib.import_module("crisalida_lib.EVA.compat")
        _rt_EVAMemoryMixin = getattr(_mod, "EVAMemoryMixin", None)
        EVAMemoryMixin_runtime = cast(Any, _rt_EVAMemoryMixin)
    except Exception:
        EVAMemoryMixin_runtime = None


# For static analysis, create an Any-typed alias instead of reassigning the
# imported name (reassigning the type name confuses mypy). Use
# `EVAMemoryMixinAny` where an Any-typed reference is needed.


if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import AdamConfig, EVAConfig
    from crisalida_lib.EARTH.self_modifying_engine import EVASelfModifyingEngine
    from crisalida_lib.EDEN.living_symbol import (
        DivineSignature,
        LivingSymbol,
        MovementPattern,
    )
    from crisalida_lib.EDEN.qualia_manifold import QualiaField
    from crisalida_lib.EVA.types import LivingSymbolRuntime

    from .cosmic_lattice import CosmicLattice
    from .virtual_machine import OntologicalVirtualMachine
else:
    # runtime fallbacks to avoid import-time coupling with heavy deps
    AdamConfig = Any
    EVAConfig = Any
    EVASelfModifyingEngine = Any
    DivineSignature = Any
    LivingSymbol = Any
    MovementPattern = Any
    QualiaField = Any
    LivingSymbolRuntime = Any
    CosmicLattice = Any
    OntologicalVirtualMachine = Any

logger = logging.getLogger(__name__)


# Avoid using an Any-typed variable as a base class (mypy rejects this).
# Instead initialize the EVAMemoryMixin at runtime inside __init__ if available.
class QualiaEngine:
    """
    The central orchestrator for the physics simulation.
    Manages the simulation loop and the interaction between all core components.
    """

    def __init__(
        self,
        dimensions: tuple = (100, 100, 100),
        dt: float = 0.016,
        eva_phase: str = "default",
        logger: logging.Logger | None = None,
    ):
        # Try to initialize EVAMemoryMixin behavior if the real mixin is available.
        try:
            # If a runtime mixin implementation was resolved, call its __init__ on this instance.
            if isinstance(EVAMemoryMixin_runtime, type):
                init_fn = getattr(EVAMemoryMixin_runtime, "__init__", None)
                if callable(init_fn):
                    init_fn(cast(Any, self))  # type: ignore[arg-type]
        except Exception:
            # best-effort: continue even if mixin init fails
            pass
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing QualiaEngine...")

        self.dimensions = dimensions
        self.dt = float(dt)

        # Instantiate core physics components
        self.adam_config = AdamConfig()
        # runtime may not provide a LivingSymbolRuntime; annotate as Optional
        # Construct runtime instance defensively and cast to Any for mypy.
        self.living_symbol_runtime: LivingSymbolRuntime | None = None
        try:
            try:
                lsr = LivingSymbolRuntime()
            except Exception:
                lsr = None
            # cast to Any to avoid mypy complaining about assignment to LivingSymbolRuntime
            self.living_symbol_runtime = cast(Any, lsr)
        except Exception:
            self.living_symbol_runtime = None
        # QualiaField does not accept 'eva_runtime' in its constructor across
        # compatibility shims; instantiate and assign runtime afterwards to
        # avoid call-arg mypy errors and import-time incompatibilities.
        # Cast the instantiated manifold to Any for static checkers.
        # Use the defensive factory to create the QualiaField without requiring
        # all pydantic constructor args. This keeps the import-time init light
        # and prevents mypy call-arg errors.
        try:
            from crisalida_lib.EDEN.qualia_manifold import create_qualia_field

            self.manifold = cast(Any, create_qualia_field())
        except Exception:
            try:
                self.manifold = cast(Any, None)
            except Exception:
                self.manifold = cast(Any, None)
        try:
            # attach living symbol runtime if the manifold supports it
            if self.manifold is not None and hasattr(self.manifold, "eva_runtime"):
                self.manifold.eva_runtime = self.living_symbol_runtime
        except Exception:
            # best-effort; don't fail QualiaEngine init
            pass
        # Grammar engine (rules -> fuerzas/efectos)
        # self.grammar_engine = eva_grammar_engine
        # operational knobs
        self.symbolic_interaction_radius = float(
            getattr(self.adam_config, "SYMBOLIC_INTERACTION_RADIUS", 6.0)
        )
        self._max_creations_per_tick = int(
            getattr(self.adam_config, "MAX_CREATIONS_PER_TICK", 2)
        )
        self._creations_this_tick = 0
        # Configure snow-model persistence parameters from AdamConfig (fall back to defaults)
        try:
            self.manifold.crystal_preservation_threshold = float(
                getattr(
                    self.adam_config,
                    "CRYSTAL_PRESERVATION_THRESHOLD",
                    self.manifold.crystal_preservation_threshold,
                )
            )
            self.manifold.crystal_half_life = float(
                getattr(
                    self.adam_config,
                    "CRYSTAL_HALF_LIFE",
                    self.manifold.crystal_half_life,
                )
            )
            self.manifold.max_crystals = int(
                getattr(
                    self.adam_config,
                    "MAX_CRYSTALS",
                    getattr(self.manifold, "max_crystals", 10000),
                )
            )
        except Exception:
            self.logger.debug(
                "Could not apply AdamConfig crystal parameters; using manifold defaults"
            )
        self.cosmic_lattice = CosmicLattice(
            manifold=self.manifold, eva_runtime=self.living_symbol_runtime
        )
        self.vm = OntologicalVirtualMachine(
            manifold=self.manifold,
            physics_engine=self,
            eva_runtime=self.living_symbol_runtime,
        )

        # Use the EVA-aware SelfModifyingEngine so proposals integrate with EVA memory
        eva_config = EVAConfig()
        self.self_modifying_engine = EVASelfModifyingEngine(
            eva_config=eva_config, phase=eva_phase
        )
        # Provide the runtime to the self-mod engine if supported
        try:
            if hasattr(self.self_modifying_engine, "eva_runtime"):
                # assign defensively via setattr and Any-cast to avoid mypy
                # incompatible assignment errors when living_symbol_runtime is Optional
                if self.living_symbol_runtime is not None:
                    setattr(cast(Any, self.self_modifying_engine), "eva_runtime", cast(Any, self.living_symbol_runtime))
        except Exception:
            self.logger.debug("Failed to attach runtime to EVASelfModifyingEngine")

        self.living_symbols: dict[str, LivingSymbol] = {}
        self._running = False
        self.ticks = 0

        # Initialize EVA mixin runtime linkage
        # Call _init_eva_memory only if available on the mixin/shim to avoid
        # static and runtime AttributeErrors on compatibility layers.
        if hasattr(self, "_init_eva_memory") and callable(self._init_eva_memory):
            try:
                self._init_eva_memory(eva_runtime=self.living_symbol_runtime)
            except Exception:
                self.logger.debug(
                    "_init_eva_memory not available during QualiaEngine init"
                )
        self.logger.info("QualiaEngine initialized successfully.")

    def manifest_living_symbol(
        self, pattern_addresses: list[tuple[int, int, int]], pattern_type: str
    ) -> LivingSymbol:
        """Creates a new LivingSymbol and adds it to the simulation."""
        entity_id = f"symbol_{len(self.living_symbols) + 1}"

        # The new LivingSymbol constructor needs a single initial position.
        # We'll use the centroid of the pattern addresses as the initial position.
        if not pattern_addresses:
            # materialize a concrete 3-tuple of ints for initial position
            dims_list: list[int] = [int(d) for d in list(self.dimensions)]
            initial_position_list: list[int] = [int(d) // 2 for d in dims_list]
            # mypy expects List[int] here; keep explicit ints
            initial_position: tuple[int, int, int] = (
                initial_position_list[0],
                initial_position_list[1],
                initial_position_list[2],
            )  # Default to center
        else:
            # Convert to numpy array and calculate centroid
            addresses_array = np.array(pattern_addresses)
            if addresses_array.ndim == 1:
                # Single address - ensure floats but convert to ints for the LivingSymbol constructor
                initial_position_list = [int(float(x)) for x in addresses_array.tolist()]
                initial_position = (initial_position_list[0], initial_position_list[1], initial_position_list[2])
            else:
                # Multiple addresses - calculate centroid and coerce to floats
                centroid = np.mean(addresses_array, axis=0)
                centroid_list: list[float] = [float(x) for x in centroid]
                # convert centroid floats to ints for consistent positioning
                initial_position = (int(centroid_list[0]), int(centroid_list[1]), int(centroid_list[2]))

        # Construir una DivineSignature determinística a partir del “pattern_type”
        # para que la entidad encarne su identidad simbólica.
        try:
            import hashlib

            seed = int(hashlib.sha1(pattern_type.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            base_vec = rng.normal(0.0, 1.0, size=16).astype(float)
            n = float(np.linalg.norm(base_vec))
            if n > 1e-9:
                base_vec = base_vec / n
        except Exception:
            base_vec = np.zeros(16, dtype=float)
        divine_sig = DivineSignature(
            category=pattern_type,
            base_vector=base_vec,
            metadata={"sigil": pattern_type},
        )

        # Config dict mínima para evitar accesos .get fallidos
        cfg = {
            "signature_alpha": 0.7,
            "interaction_radius": float(self.symbolic_interaction_radius),
            "learning_rate": 0.02,
            "velocity_damping": 0.98,
            "qualia_decay": 0.9995,
        }

        symbol = LivingSymbol(
            entity_id=entity_id,
            manifold=self.manifold,
            initial_position=initial_position,
            divine_signature=divine_sig,
            movement_pattern=MovementPattern.STATIC,  # Default movement
            config=cfg,
        )
        # Exponer un alias conveniente para la gramática si aplica (sigil visible)
        try:
            # Avoid mypy complaints about missing 'sigil' attribute on LivingSymbol
            # by setting the attribute via setattr on an Any-cast to avoid mypy
            # treating this as a method/descriptor assignment.
            setattr(cast(Any, symbol), "sigil", pattern_type)
        except Exception:
            pass
        self.living_symbols[entity_id] = symbol
        self.logger.info(
            "Manifested new LivingSymbol: %s at %s", entity_id, initial_position
        )
        return symbol

    # Small no-op EVA ingestion API to satisfy static checkers and older callsites.
    def eva_ingest_experience(self, *args, **kwargs):
        # This method is intentionally permissive and non-blocking.
        return None

    def _perform_dream_reinforcement(self) -> None:
        """
        Periodically called to reinforce memory crystals that are relevant to living symbols.
        Reinforcement happens when a symbol is spatially near a crystal or when the field coherence suggests recall.
        """
        try:
            if not self.living_symbols:
                return
            # Query crystals via public API (KD-tree backed)
            # Query only nearby/salient crystals to reinforce efficiently
            crystals = self.manifold.get_memory_traces(position=None, min_strength=0.0)
            if not crystals:
                return
            # Compute global coherence as heuristic (fallbacks handled)
            coherence_field = getattr(self.manifold, "coherence_field", None)
            global_coherence = (
                float(np.mean(coherence_field))
                if coherence_field is not None and len(coherence_field)
                else 0.5
            )
            reinforce_prob_base = min(0.5, max(0.01, global_coherence * 0.2))

            for cid, crystal in list(crystals.items()):
                # If crystal disabled for reinforcement skip
                if not crystal.get("reinforce_on_recall", True):
                    continue
                # Determine proximity to any living symbol
                pos = crystal.get("position")
                reinforce_amount = 0.0
                if pos is not None:
                    pos_arr = np.array(tuple(map(float, pos)), dtype=float)
                    for symbol in self.living_symbols.values():
                        sym_pos = getattr(symbol.kinetic_state, "position", None)
                        if sym_pos is None:
                            continue
                        try:
                            dist = float(
                                np.linalg.norm(np.array(sym_pos, dtype=float) - pos_arr)
                            )
                        except Exception:
                            continue
                        # Influence decreases with distance (tuned thresholds)
                        if dist < 1.0:
                            reinforce_amount = max(reinforce_amount, 0.06)
                        elif dist < 3.0:
                            reinforce_amount = max(reinforce_amount, 0.03)
                        elif dist < 6.0:
                            reinforce_amount = max(reinforce_amount, 0.01)
                # Small random reinforcement based on coherence
                if reinforce_amount == 0.0 and np.random.rand() < reinforce_prob_base:
                    reinforce_amount = 0.005
                if reinforce_amount > 0.0:
                    try:
                        # call manifold reinforcement via supported public API
                        if hasattr(self.manifold, "reinforce_crystal"):
                            self.manifold.reinforce_crystal(
                                cid, amount=reinforce_amount
                            )
                        elif hasattr(self.manifold, "_reinforce_crystal"):
                            # last resort internal API
                            self.manifold._reinforce_crystal(
                                cid, amount=reinforce_amount
                            )
                        else:
                            self.logger.debug(
                                "No reinforcement API available on manifold for %s", cid
                            )
                    except Exception:
                        self.logger.debug(
                            "Failed to reinforce crystal %s", cid, exc_info=True
                        )
        except Exception:
            self.logger.exception("Error during dream reinforcement")

    def _extract_sigil(self, symbol: LivingSymbol) -> str | None:
        """
        Obtiene un sigilo textual del símbolo si existe; si no, intenta derivarlo
        de su DivineSignature (metadata/ categoría).
        """
        for attr in ("sigil", "divine_sigil", "signature_sigil"):
            v = getattr(symbol, attr, None)
            if isinstance(v, str) and v:
                return v
        try:
            div = getattr(symbol, "divine", None)
            if div:
                meta = getattr(div, "metadata", {}) or {}
                if isinstance(meta, dict):
                    if isinstance(meta.get("sigil"), str) and meta.get("sigil"):
                        return meta["sigil"]
                    if isinstance(meta.get("glyph"), str) and meta.get("glyph"):
                        return meta["glyph"]
                cat = getattr(div, "category", "")
                if isinstance(cat, str) and 0 < len(cat) <= 3:
                    return cat
        except Exception:
            return None
        return None

    # --- Topological pattern helpers (Fase 3) ---
    def _get_positions_by_sigil(self) -> dict[str, list[tuple[float, float, float]]]:
        pos_by_sigil: dict[str, list[tuple[float, float, float]]] = {}
        for s in self.living_symbols.values():
            sig = self._extract_sigil(s)
            pos = getattr(getattr(s, "kinetic_state", None), "position", None)
            if not sig or pos is None:
                continue
            try:
                pos_list = [float(x) for x in pos]
                # ensure fixed-length 3-tuple for mypy and consumers
                if len(pos_list) >= 3:
                    p = (pos_list[0], pos_list[1], pos_list[2])
                else:
                    padded = pos_list + [0.0] * (3 - len(pos_list))
                    p = (padded[0], padded[1], padded[2])
            except Exception:
                continue
            # ensure list is concrete type for mypy
            lst = pos_by_sigil.setdefault(sig, [])
            lst.append(p)
        return pos_by_sigil

    def _infer_spatial_pattern(
        self, pts: list[tuple[float, float, float]]
    ) -> tuple[str | None, float]:
        """
        Devuelve (pattern_name, confidence) con pattern in {"línea","triángulo","círculo","espiral"}.
        Best‑effort con PCA y métricas simples. confidence en [0,1].
        """
        try:
            if len(pts) < 3:
                return (None, 0.0)
            X = np.asarray(pts, dtype=float)
            C = np.mean(X, axis=0)
            Y = X - C
            # SVD para planicidad/linealidad
            U, S, Vt = np.linalg.svd(Y, full_matrices=False)
            s1, s2, s3 = (S.tolist() + [0.0, 0.0, 0.0])[:3]
            s1 = float(s1 or 1e-9)
            float((s1 - s2) / (s1 + 1e-9))
            float((s2 - s3) / (s2 + 1e-9)) if s2 > 0 else 0.0
            planarity = float(1.0 - (s3 / (s1 + 1e-9)))
            linearity = float(1.0 - (s2 / (s1 + 1e-9)))
            # Línea: alta linealidad
            if linearity > 0.85:
                return ("línea", min(1.0, linearity))
            # Proyección a plano principal para círculo/espiral
            Vt[2] if Vt.shape[0] >= 3 else np.array([0, 0, 1.0])
            # Base ortonormal del plano
            a = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
            b = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
            XY = np.c_[Y @ a, Y @ b]  # coords 2D en el plano
            r = np.linalg.norm(XY, axis=1)
            r_mean = float(np.mean(r) or 1e-9)
            r_std = float(np.std(r))
            radius_cv = float(r_std / (r_mean + 1e-9))
            # Círculo: buena planicidad y radio casi constante
            if planarity > 0.9 and radius_cv < 0.18:
                conf = float(min(1.0, (planarity * 0.6 + (1.0 - radius_cv) * 0.4)))
                return ("círculo", conf)
            # Triángulo: no lineal, no círculo, área significativa del triángulo máximo
            tri_conf = 0.0
            if len(pts) >= 3:
                # usar los extremos por PCA como base
                proj = XY
                # escoger 3 puntos: minX, maxX, maxDist a la recta
                idx_min = int(np.argmin(proj[:, 0]))
                idx_max = int(np.argmax(proj[:, 0]))

                def tri_area(p, q, r):
                    return abs(
                        0.5
                        * np.linalg.det(
                            np.array(
                                [[p[0] - r[0], p[1] - r[1]], [q[0] - r[0], q[1] - r[1]]]
                            )
                        )
                    )

                base = (proj[idx_min], proj[idx_max])
                idx_far = int(
                    np.argmax([tri_area(base[0], base[1], pr) for pr in proj])
                )
                A = tri_area(base[0], base[1], proj[idx_far])
                span = max(1e-9, np.ptp(proj[:, 0]) * np.ptp(proj[:, 1]) + 1e-9)
                tri_conf = float(
                    min(1.0, (A / (span + 1e-9)) * 4.0)
                )  # normalización heurística
                if tri_conf > 0.25 and linearity < 0.8:
                    return ("triángulo", min(1.0, tri_conf * planarity))
            # Espiral: planaria, radio crece con ángulo
            if planarity > 0.85:
                angles = np.arctan2(XY[:, 1], XY[:, 0])
                # ordenar por ángulo desenrollado
                order = np.argsort(angles)
                ang = angles[order]
                rad = r[order]
                # desenrollar discontinuidad
                ang_un = np.unwrap(ang)
                # correlación ángulo-radio
                if np.std(ang_un) > 1e-6 and np.std(rad) > 1e-6:
                    corr = float(np.corrcoef(ang_un, rad)[0, 1])
                else:
                    corr = 0.0
                if corr > 0.5:
                    conf = float(min(1.0, (planarity * 0.5 + corr * 0.5)))
                    return ("espiral", conf)
            # Sin patrón claro
            return (None, 0.0)
        except Exception:
            return (None, 0.0)

    def _pattern_confidence_meets(
        self, required: str, pts: list[tuple[float, float, float]], threshold: float
    ) -> tuple[bool, float]:
        name, conf = self._infer_spatial_pattern(pts)
        return (name == required, conf) if name else (False, conf)

    def _detect_topological_patterns(self) -> None:
        """
        Detect emergent grammar properties from active sigils and their spatial topology.
        If a property's required sigils are present AND the positions match its spatial_pattern
        with sufficient confidence, trigger its effects.
        """
        try:
            if not self.living_symbols:
                return
            pos_by_sigil = self._get_positions_by_sigil()
            active_sigils = set(pos_by_sigil.keys())
            if not active_sigils:
                return

            from crisalida_lib.EVA.language.grammar import (
                eva_grammar_engine,  # Moved import to break circular dependency
            )

            # Evaluar cada propiedad emergente declarada en la gramática
            for prop in eva_grammar_engine.emergent_properties.values():
                if not prop.required_sigils.issubset(active_sigils):
                    continue
                # Reunir puntos de los sigilos requeridos
                pts: list[tuple[float, float, float]] = []
                for s in prop.required_sigils:
                    pts.extend(pos_by_sigil.get(s, []))
                if len(pts) < 3:
                    continue
                # Verificar patrón espacial
                required_pattern = getattr(prop, "spatial_pattern", "") or ""
                ok, conf = (True, 1.0)
                if required_pattern:
                    ok, conf = self._pattern_confidence_meets(
                        required_pattern, pts, getattr(prop, "emergence_threshold", 0.8)
                    )
                    if not ok or conf < getattr(prop, "emergence_threshold", 0.8):
                        continue

                # Disparar efectos: registrar en EVA y aplicar mapa de efectos
                try:
                    self.eva_ingest_experience(
                        intention_type="EMERGENT_PROPERTY_DETECTED",
                        experience_data={
                            "property": prop.name,
                            "effects": prop.effects,
                            "required": list(prop.required_sigils),
                            "spatial_pattern": required_pattern or "any",
                            "confidence": conf,
                        },
                        qualia_state=QualiaState(
                            consciousness=min(1.0, 0.5 + conf * 0.5),
                            energy=0.5,
                            importance=0.8,
                        ),
                    )
                except Exception:
                    self.logger.debug(
                        "Failed to record emergent property to EVA", exc_info=True
                    )

                effects = getattr(prop, "effects", {}) or {}
                # Coherencia local: reforzar cristales cercanos
                if (
                    "consciencia_unificada" in effects
                    or "inteligencia_colectiva" in effects
                ) and hasattr(self.manifold, "memory_crystal_index"):
                    try:
                        for cid, _meta in list(
                            getattr(self.manifold, "memory_crystal_index", {}).items()
                        )[:8]:
                            try:
                                if hasattr(self.manifold, "reinforce_crystal"):
                                    self.manifold.reinforce_crystal(
                                        cid, amount=min(0.05, 0.01 + conf * 0.02)
                                    )
                            except Exception:
                                continue
                    except Exception:
                        pass
                # Nueva existencia: crear símbolo emergente
                if (
                    "nueva_existencia" in effects or "potencial_manifestado" in effects
                ) and self._creations_this_tick < self._max_creations_per_tick:
                    try:
                        # Spawn en el centroide del patrón detectado (mejor correlato físico)
                        arr = np.asarray(pts, dtype=float)
                        centroid_list = [float(x) for x in np.mean(arr, axis=0).tolist()]
                        centroid = tuple(centroid_list)
                        # ensure spawn is a concrete 3-tuple[int, int, int]
                        dims = list(self.dimensions)
                        def _clamp(idx: int) -> int:
                            try:
                                val = int(centroid_list[idx]) if idx < len(centroid_list) else 0
                            except Exception:
                                val = 0
                            maxd = int(dims[idx]) - 1 if idx < len(dims) else 0
                            return int(max(0, min(val, maxd)))

                        spawn = (_clamp(0), _clamp(1), _clamp(2))
                        self.manifest_living_symbol(
                            pattern_addresses=[spawn],
                            pattern_type=f"emergent:{prop.name}",
                        )
                        self._creations_this_tick += 1
                        self.logger.info(
                            "Topological emergent pattern '%s' (%.2f) -> creation at %s",
                            prop.name,
                            conf,
                            spawn,
                        )
                    except Exception:
                        self.logger.debug(
                            "Failed topology-driven creation", exc_info=True
                        )
        except Exception:
            self.logger.exception("Error detecting topological grammar patterns")

    def tick(self, bytecode: list | None = None):
        """
        Executes a single tick of the physics simulation.
        This implements the core physics simulation loop as specified in the requirements.
        """
        self.logger.debug("--- Physics Tick %d ---", self.ticks)

        # 1. Execute incoming LSM bytecode via the VM
        if bytecode:
            self.logger.debug("Executing %d bytecode instructions", len(bytecode))
            try:
                vm_result = self.vm.execute_bytecode(bytecode)
                self.logger.debug("VM execution result: %s", vm_result)
            except Exception as e:
                self.logger.warning("VM execution failed: %s", e, exc_info=True)

        # 2. Apply influences from the CosmicLattice to the manifold
        try:
            # Get cosmic influence data from CosmicLattice
            cosmic_influence_data = self.cosmic_lattice.calculate_total_influence(
                perception_context={} # TODO: Pass a real perception context
            )
            self.logger.debug("Cosmic lattice influence computed: %s", cosmic_influence_data)

            # Calculate QualiaState influence from cosmic data
            qualia_influence_state = self._calculate_cosmic_qualia_influence(cosmic_influence_data)
            self.logger.debug("QualiaState influence from CosmicLattice: %s", qualia_influence_state)

            # Apply this QualiaState influence to the manifold
            # The manifold's apply_consciousness_influence expects a consciousness_state dict
            # We can map the QualiaState attributes to a consciousness_state dict
            consciousness_state_for_manifold = {
                "consciousness_coherence": qualia_influence_state.consciousness,
                "stress_level": 1.0 - qualia_influence_state.emotional, # Inverse emotional valence for stress
                "energy_balance": qualia_influence_state.energy,
                "decision_confidence": qualia_influence_state.importance,
            }
            self.manifold.apply_consciousness_influence(
                entity_id="cosmic_lattice_influence",
                consciousness_state=consciousness_state_for_manifold,
                influence_type=cosmic_influence_data.get("type", "ORDER").upper(), # Use influence type from cosmic lattice
                strength=cosmic_influence_data.get("strength", 0.0),
            )
            self.logger.debug("Cosmic QualiaState influence applied to manifold.")
        except Exception as e:
            self.logger.warning(
                "Error applying cosmic lattice influence: %s", e, exc_info=True
            )

        # 2.b Procesar Interacciones Simbólicas (Fase 3: Gramática -> Fuerzas/Eventos)
        try:
            # reset per-tick creation counter
            self._creations_this_tick = 0
            # self._process_symbolic_interactions()
            self._detect_topological_patterns()
        except Exception:
            self.logger.exception(
                "Error in symbolic interactions / topological detection"
            )

        # 3. Update each LivingSymbol
        symbols_to_remove = []
        for symbol_id, symbol in self.living_symbols.items():
            try:
                # The LivingSymbol update method handles all internal and external interactions
                # wrap updates to avoid a single bad symbol breaking the tick
                symbol.update(self.dt, self.self_modifying_engine)
                self.logger.debug(
                    "Updated symbol %s at position %s",
                    symbol_id,
                    getattr(symbol.kinetic_state, "position", "unknown"),
                )
            except Exception as e:
                self.logger.warning(
                    "Error updating symbol %s: %s", symbol_id, e, exc_info=True
                )
                # Mark for removal if update fails critically
                if hasattr(symbol, "health") and getattr(symbol, "health", 100) <= 0:
                    symbols_to_remove.append(symbol_id)

        # Remove any symbols that have been marked for removal
        for symbol_id in symbols_to_remove:
            self.logger.info(f"Removing symbol {symbol_id} due to critical failure")
            del self.living_symbols[symbol_id]

        # DREAM / SLEEP reinforcement: periodically reinforce memory crystals
        try:
            # frequency configurable; here a conservative periodic reinforcement
            if self.ticks % 50 == 0:
                self._perform_dream_reinforcement()
        except Exception:
            self.logger.exception("Error performing periodic dream reinforcement")

        # 4. Update the manifold dynamics
        try:
            dynamics_result = self.manifold.update_dynamics(self.dt)
            self.logger.debug(f"Manifold dynamics update: {dynamics_result}")
        except Exception as e:
            self.logger.warning(f"Error updating manifold dynamics: {e}")

        # 5. Record this tick in EVA memory
        # Record this tick in EVA memory (best-effort, non-critical)
        try:
            tick_experience = {
                "tick_number": self.ticks,
                "active_symbols": len(self.living_symbols),
                "lattice_influence": (
                    lattice_influence if "lattice_influence" in locals() else {}
                ),
                "manifold_state": {
                    "field_stability": (
                        dynamics_result.get("field_stability", 0.0)
                        if "dynamics_result" in locals()
                        else 0.0
                    ),
                    "total_coherence": (
                        dynamics_result.get("total_coherence", 0.0)
                        if "dynamics_result" in locals()
                        else 0.0
                    ),
                },
                "timestamp": self.ticks * self.dt,
            }

            from crisalida_lib.EVA.typequalia import (
                QualiaState,  # Moved import to break circular dependency
            )

            qualia_state = QualiaState(
                consciousness=(
                    dynamics_result.get("total_coherence", 0.5)
                    if "dynamics_result" in locals()
                    else 0.5
                ),
                energy=(
                    dynamics_result.get("mean_field_energy", 0.5)
                    if "dynamics_result" in locals()
                    else 0.5
                ),
                importance=(
                    dynamics_result.get("field_stability", 0.5)
                    if "dynamics_result" in locals()
                    else 0.5
                ),
            )

            # Best-effort ingest via EVAMemoryMixin/helper; failures should not stop the simulation
            try:
                # Prefer mixin API which delegates to the helper internally
                if hasattr(self, "eva_ingest_experience"):
                    self.eva_ingest_experience(
                        intention_type="PHYSICS_TICK",
                        experience_data=tick_experience,
                        qualia_state=qualia_state,
                    )
                else:
                    # Fallback to the helper if present (older instances)
                    helper = getattr(self, "_eva_helper", None)
                    if helper is not None:
                        helper.eva_ingest_experience(
                            intention_type="PHYSICS_TICK",
                            experience_data=tick_experience,
                            qualia_state=qualia_state,
                        )
            except Exception:
                self.logger.debug(
                    "eva_ingest_experience failed for tick %d",
                    self.ticks,
                    exc_info=True,
                )
        except Exception:
            self.logger.exception("Error preparing tick experience")

        self.ticks += 1
        self.logger.debug("--- End of Tick %d ---", self.ticks)

    def run(self, simulation_duration_ticks: int = 1000):
        """
        Runs the simulation for a given number of ticks.
        """
        self.logger.info(
            "Starting physics simulation for %d ticks.", simulation_duration_ticks
        )
        self._running = True
        try:
            for _ in range(simulation_duration_ticks):
                if not self._running:
                    break
                self.tick()
        finally:
            self._running = False
            self.logger.info("Physics simulation finished.")

    def shutdown(self):
        """
        Gracefully shuts down the physics engine.
        """
        self.logger.info("Shutting down QualiaEngine...")
        self._running = False

    # Context manager helpers for convenience in demos/tests
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def get_diagnostics(self) -> dict:
        """
        Returns diagnostics for the engine and its components.
        """
        try:
            active_influences = len(getattr(self.manifold, "active_influences", {}))
        except Exception:
            active_influences = 0
        return {
            "engine_running": self._running,
            "simulation_ticks": self.ticks,
            "living_symbols_count": len(self.living_symbols),
            "total_symbol_interactions": sum(
                len(getattr(s, "interaction_history", []))
                for s in self.living_symbols.values()
            ),
            "consciousness_influences_active": active_influences,
        }

    def move_living_symbol(
        self, symbol_id: str, new_position: tuple[float, float, float]
    ):
        """Directly moves a LivingSymbol to a new position."""
        if symbol_id in self.living_symbols:
            self.living_symbols[symbol_id].kinetic_state.position = new_position
            self.logger.info(f"Moved symbol {symbol_id} to {new_position}")
            return True
        return False

    def delete_living_symbol(self, symbol_id: str):
        """Directly deletes a LivingSymbol from the simulation."""
        if symbol_id in self.living_symbols:
            del self.living_symbols[symbol_id]
            self.logger.info(f"Deleted symbol {symbol_id}")
            return True
        return False

    def _apply_snow_model_params(self):
        mf = getattr(self, "manifold", None)
        cfg = getattr(self, "adam_config", None)
        if not mf or not cfg:
            return
        for k, attr, default in (
            ("CRYSTAL_PRESERVATION_THRESHOLD", "crystal_preservation_threshold", 0.6),
            ("CRYSTAL_HALF_LIFE", "crystal_half_life", 3600.0 * 24.0 * 365.0 * 5.0),
            ("MAX_CRYSTALS", "max_crystals", 10000),
        ):
            try:
                setattr(mf, attr, float(getattr(cfg, k, default)))
            except Exception:
                setattr(mf, attr, default)

    def _calculate_cosmic_qualia_influence(self, cosmic_influence_data: dict[str, Any]) -> QualiaState:
        """
        Calculates a QualiaState based on the aggregated influence from CosmicLattice.
        Adapts the logic from the legacy QualiaInfluenceCalculator.
        """
        sephirot_influence = cosmic_influence_data.get("sephirot_influence", 0.0)
        qliphoth_influence = cosmic_influence_data.get("qliphoth_influence", 0.0)
        influence_strength = cosmic_influence_data.get("strength", 0.0)
        influence_type = cosmic_influence_data.get("type", "neutral")

        # Initialize QualiaState
        qualia = QualiaState()

        # Map CosmicLattice influences to QualiaState attributes
        # Emotional Valence: Higher Sephirot -> positive, higher Qliphoth -> negative
        qualia.emotional = (sephirot_influence - qliphoth_influence) * 0.5

        # Arousal: Higher overall strength
        qualia.arousal = influence_strength * 0.8

        # Cognitive Complexity: Tension between Sephirot and Qliphoth
        qualia.complexity = abs(sephirot_influence - qliphoth_influence) * 0.5

        # Sensory Clarity: Higher Sephirot influence
        qualia.sensory_clarity = sephirot_influence * 0.6

        # Temporal Flow: Overall influence strength
        qualia.temporal = influence_strength * 0.5

        # Add some basic clamping (QualiaState's clamp_values will handle full clamping)
        qualia.emotional = max(-1.0, min(1.0, qualia.emotional))
        qualia.arousal = max(0.0, min(1.0, qualia.arousal))
        qualia.complexity = max(0.0, min(1.0, qualia.complexity))
        qualia.sensory_clarity = max(0.0, min(1.0, qualia.sensory_clarity))
        qualia.temporal = max(0.0, min(1.0, qualia.temporal))

        return qualia


# Store the original __init__ method
_original_qualia_engine_init = QualiaEngine.__init__

def _qe_init_wrap(self, *args, **kwargs):
    """Wrapper executed after QualiaEngine.__init__ to apply snow model params."""
    # Call the original __init__ if available
    if callable(_original_qualia_engine_init):
        _original_qualia_engine_init(self, *args, **kwargs)
    try:
        self._apply_snow_model_params()
    except Exception:
        pass


# Use setattr to avoid direct reassignment on the class object which some
# static checkers flag; only wrap if attribute exists.
if hasattr(QualiaEngine, "__init__"):
    try:
        setattr(QualiaEngine, "__init__", _qe_init_wrap)
    except Exception:
        # fallback: if setattr is restricted, assign directly as last resort
        QualiaEngine.__init__ = _qe_init_wrap
