from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType
from crisalida_lib.ADAM.mente.cognitive_node import CognitiveNode
from crisalida_lib.EDEN.living_symbol import (  # Living entity base
    LivingSymbol,
    MovementPattern,
)

if TYPE_CHECKING:
    from crisalida_lib.EDEN.qualia_manifold import QualiaField  # Qualia field fabric

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CosmicNode(LivingSymbol, CognitiveNode):
    """
    Embodied Cosmic Lattice node (Sephirot / Qliphoth).

    A TITANIC entity that physically inhabits the QualiaField.
    Its influence is the deformation its qualia-mass imposes on the manifold,
    not a mere abstract signal. Deformation is applied as a localized imprint
    (Gaussian-like). Higher mass yields deeper, longer-lived scars.
    """

    def __init__(
        self,
        entity_id: str,
        manifold: Any | None = None,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        node_type: str = "sephirot",  # "sephirot" or "qliphoth"
        node_name: str = "",  # e.g., "malkuth_reality"
        config: AdamConfig = AdamConfig(),
        movement_pattern: MovementPattern = MovementPattern.STATIC,
        mass: float = 1.0,
        activation_threshold: float = 0.5,
        influence_radius: float = 3.0,
        divine_signature: Any | None = None,
    ):
        # LivingSymbol initialization (kinematic + embodied layers)
        # Forward 'divine_signature' when available so nodes can supply
        # their pre-built signatures and avoid per-node fallbacks.
        try:
            super().__init__(
                entity_id=entity_id,
                manifold=manifold,
                initial_position=initial_position,
                pattern_type=node_name,
                movement_pattern=movement_pattern,
                config=config,
                divine_signature=divine_signature,
            )
        except TypeError:
            # Older LivingSymbol/CosmicNode variants may not accept divine_signature
            # — fall back to the older signature preserving compatibility.
            super().__init__(
                entity_id=entity_id,
                manifold=manifold,
                initial_position=initial_position,
                pattern_type=node_name,
                movement_pattern=movement_pattern,
                config=config,
            )

        # Initialize CognitiveNode
        CognitiveNode.__init__(self, node_name, activation_threshold)

        # Embodiment
        self.node_type = node_type
        self.node_name = node_name
        self.mass: float = max(0.0, float(mass))
        self.inertia: float = self._compute_inertia()
        # Use a widened annotation so assignments from numpy operations don't
        # cause mypy incompatibility errors in environments with optional numpy
        # typings. At runtime this is a numpy array.
        self.momentum: Any = np.zeros(3, dtype=float)
        self.influence_radius: float = float(max(0.5, influence_radius))

        # Cognitive/activation
        self.activation_level: float = 0.0  # current coherence/activation
        self.pulse_strength: float = 0.1  # baseline multiplier for impulses

        logger.debug(
            "CosmicNode %s initialized: type=%s, mass=%.3f, radius=%.2f",
            self.entity_id,
            self.node_type,
            self.mass,
            self.influence_radius,
        )

    # ------- Physical presence API -------
    def _compute_inertia(self) -> float:
        """Simple inertia model proportional to mass (can be refined)."""
        return max(1e-6, 0.5 * self.mass)

    def apply_force(self, force: Iterable[float], dt: float = 1.0) -> None:
        """Apply an external force to change momentum and kinetic_state."""
        f = np.asarray(list(force), dtype=float)
        self.momentum += f * float(dt)
        if self.mass > 0:
            # ensure velocity stored as a plain 3-tuple of floats (not ndarray)
            vel_arr = (self.momentum / self.mass).tolist()
            self.kinetic_state.velocity = (
                float(vel_arr[0]),
                float(vel_arr[1]),
                float(vel_arr[2]),
            )

    # ------- Perception: resonant field sensing -------
    def perceive_local_qualia(self) -> dict[str, Any]:
        # Minimal, well-indented version preserving original behavior
        # Convert position to integer tuple
        # keep runtime semantics but satisfy typing: tuple of floats
        pos = cast(
            tuple[float, float, float],
            tuple(
                float(x)
                for x in getattr(self.kinetic_state, "position", (0.0, 0.0, 0.0))
            ),
        )

        if not hasattr(self.manifold_ref, "dimensions"):
            return {
                "error": "manifold-unavailable",
                "node_position": pos,
                "node_type": self.node_type,
                "node_name": self.node_name,
            }

        dims = self.manifold_ref.dimensions
        clamped = tuple(
            max(0, min(int(p), d - 1)) for p, d in zip(pos, dims, strict=False)
        )

        r = max(1, int(math.ceil(self.influence_radius)))
        vals: list[float] = []
        stride = 1 if r <= 3 else max(1, r // 3)
        for dx in range(-r, r + 1, stride):
            for dy in range(-r, r + 1, stride):
                for dz in range(-r, r + 1, stride):
                    cx, cy, cz = clamped[0] + dx, clamped[1] + dy, clamped[2] + dz
                    if 0 <= cx < dims[0] and 0 <= cy < dims[1] and 0 <= cz < dims[2]:
                        try:
                            s = self.manifold_ref.get_state_at((cx, cy, cz))
                        except Exception:
                            s = None
                        try:
                            if s is None:
                                vals.append(0.0)
                            elif isinstance(s, dict):
                                if "valence" in s or "arousal" in s:
                                    vals.append(
                                        (
                                            float(s.get("valence", 0.0))
                                            + float(s.get("arousal", 0.0))
                                        )
                                        / 2.0
                                    )
                                elif "field_value" in s:
                                    vals.append(float(s.get("field_value", 0.0)))
                                else:
                                    vals.append(
                                        float(
                                            sum(hash(k + str(v)) for k, v in s.items())
                                            % 1000
                                        )
                                        / 1000.0
                                        - 0.5
                                    )
                            elif hasattr(s, "as_vector") and callable(s.as_vector):
                                vec = s.as_vector()
                                vals.append(
                                    float(np.mean(np.asarray(vec, dtype=float)))
                                )
                            elif hasattr(s, "__dict__"):
                                v = 0.0
                                for attr in (
                                    "consciousness_density",
                                    "temporal_coherence",
                                    "valence",
                                    "arousal",
                                ):
                                    if hasattr(s, attr):
                                        v += float(getattr(s, attr, 0.0))
                                vals.append(v / 4.0)
                            elif isinstance(s, (int, float, np.number)):
                                vals.append(float(s))
                            elif isinstance(s, np.ndarray):
                                vals.append(float(np.mean(s)))
                            else:
                                vals.append(0.0)
                        except Exception:
                            vals.append(0.0)

        arr = np.asarray(vals, dtype=float) if vals else np.zeros(1, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        sample_count = int(arr.size)

        # Optional dynamic signature
        node_sig_vec = None
        try:
            if hasattr(self, "compute_dynamic_signature") and callable(
                self.compute_dynamic_signature
            ):
                sig = self.compute_dynamic_signature()
                vec = getattr(sig, "vector", None) or getattr(sig, "embedding", None)
                if vec is not None:
                    # Cast to Any to avoid mypy NDArray shape/type mismatches
                    node_sig_vec = cast(Any, np.asarray(vec, dtype=float).flatten())
                    try:
                        n = float(np.linalg.norm(node_sig_vec))
                        if n > 1e-9:
                            node_sig_vec = node_sig_vec / n
                    except Exception:
                        pass
        except Exception:
            node_sig_vec = None

        proximal_crystals: list[dict[str, Any]] = []
        mem_score = 0.0
        weighted_alignment_sum = 0.0
        try:
            if hasattr(self.manifold_ref, "get_memory_traces"):
                crystals = self.manifold_ref.get_memory_traces(
                    position=clamped,
                    max_distance=max(1.0, r * 2.0),
                    k=32,
                    min_strength=0.0,
                )
            else:
                crystals = getattr(self.manifold_ref, "memory_crystal_index", {})

            if isinstance(crystals, dict):
                items_iter = list(crystals.items())
            elif isinstance(crystals, (list, tuple)):
                # materialize into a concrete list to satisfy static analysers
                items_iter = [(m.get("id", f"c{i}"), m) for i, m in enumerate(crystals)]
            else:
                items_iter = []

            now = getattr(self.manifold_ref, "current_time", None) or time.time()
            for cid, meta in items_iter:
                pos_c = meta.get("position")
                strength = float(meta.get("strength", 0.0))
                if not pos_c or not isinstance(pos_c, (tuple, list)):
                    continue

                ddx = float(pos_c[0] - clamped[0])
                ddy = float(pos_c[1] - clamped[1])
                ddz = float(pos_c[2] - clamped[2])
                dist = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                sigma = max(1.0, r / 2.0)
                k_spatial = math.exp(-(dist * dist) / (2.0 * (sigma * sigma)))

                created_at = float(meta.get("created_at", now))
                half_life = float(meta.get("half_life", 0.0))
                age = max(0.0, float(now - created_at))
                if half_life and half_life > 0.0:
                    k_time = math.exp(-age / half_life)
                else:
                    k_time = 1.0

                influence = strength * k_spatial * k_time
                if influence <= 1e-6:
                    continue

                alignment = None
                try:
                    if node_sig_vec is not None:
                        c_sig = meta.get("signature")
                        c_vec = c_sig.get("vector") if isinstance(c_sig, dict) else None
                        if c_vec is not None:
                            c_vec_np = cast(Any, np.asarray(c_vec, dtype=float).flatten())
                            cn = float(np.linalg.norm(c_vec_np))
                            if cn > 1e-9:
                                c_vec_np = c_vec_np / cn
                                # cast intermediate numpy arrays to Any to avoid strict ndarray typing mismatches
                                c_vec_any = cast(Any, c_vec_np)
                                node_vec_any = cast(Any, node_sig_vec)
                                alignment = float(
                                    np.clip(np.dot(node_vec_any, c_vec_any), -1.0, 1.0)
                                )
                                weighted_alignment_sum += alignment * influence
                except Exception:
                    alignment = None

                mem_score += influence
                proximal_crystals.append(
                    {
                        "id": cid,
                        "strength": strength,
                        "distance": float(dist),
                        "age": float(age),
                        "t_decay": float(k_time),
                        "influence": float(influence),
                        **({"alignment": alignment} if alignment is not None else {}),
                        "provenance": meta.get("provenance"),
                    }
                )
            proximal_crystals.sort(key=lambda x: x["influence"], reverse=True)
            proximal_crystals = proximal_crystals[:16]
        except Exception:
            proximal_crystals = []
            mem_score = 0.0
            weighted_alignment_sum = 0.0

        eps = 1e-6
        coherence = float(max(0.0, 1.0 - (std / (abs(mean) + eps))))
        intensity = float(min(1.0, abs(mean)))
        memory_influence_score = float(min(1.0, mem_score))
        diversity = float(min(1.0, len(proximal_crystals) / max(1.0, 8.0)))
        signature_alignment = None
        if node_sig_vec is not None and mem_score > eps:
            signature_alignment = float(
                max(-1.0, min(1.0, weighted_alignment_sum / mem_score))
            )

        resonance = {
            "coherence": coherence,
            "intensity": intensity,
            "memory_influence_score": memory_influence_score,
            "diversity": diversity,
            "sample_count": sample_count,
            **(
                {"signature_alignment": signature_alignment}
                if signature_alignment is not None
                else {}
            ),
        }

        raw_patch = arr.tolist() if sample_count <= 125 else None

        return {
            "node_position": clamped,
            "node_type": self.node_type,
            "node_name": self.node_name,
            "patch_stats": {
                "mean": mean,
                "std": std,
                "min": mn,
                "max": mx,
                "count": sample_count,
            },
            "resonance": resonance,
            "proximal_crystals": proximal_crystals,
            "raw_patch": raw_patch,
        }

    # ------- Emission: embodied pulse and deformation -------
    def pulse(self, impulses: list[CognitiveImpulse]) -> str | None:
        """
        Emit embodied impulses and imprint a mass-based deformation on the manifold.
        Returns the id of a crystallized trace when available.
        """
        if not impulses and self.activation_level <= self.activation_threshold:
            return None

        total_intensity = (
            sum(float(i.intensity) * float(i.confidence) for i in impulses)
            if impulses
            else 0.0
        )
        mass_factor = math.log1p(1.0 + self.mass)
        effective_strength = (
            float(self.pulse_strength)
            * (total_intensity + 0.1)
            * mass_factor
            * max(0.01, self.activation_level)
        )

        consciousness_state = {
            "consciousness_coherence": float(self.activation_level),
            "stress_level": 0.0,
            "energy_balance": float(min(1.0, mass_factor / 10.0)),
            "decision_confidence": float(min(1.0, total_intensity)),
            "embodied_mass": float(self.mass),
        }

        deformed_id: str | None = None

        # Preferred: dedicated mass deformation API
        if hasattr(self.manifold_ref, "apply_mass_deformation"):
            try:
                deformed_id = self.manifold_ref.apply_mass_deformation(
                    entity_id=self.entity_id,
                    mass=self.mass,
                    position=tuple(
                        int(x)
                        for x in getattr(self.kinetic_state, "position", (0, 0, 0))
                    ),
                    strength=effective_strength,
                    radius=self.influence_radius,
                    provenance={"node": self.node_name, "type": self.node_type},
                )
            except Exception:
                logger.exception(
                    "apply_mass_deformation failed; trying compatibility fallbacks"
                )
                deformed_id = None

        # Fallback 1: generic consciousness influence
        if deformed_id is None and hasattr(
            self.manifold_ref, "apply_consciousness_influence"
        ):
            try:
                resp = self.manifold_ref.apply_consciousness_influence(
                    entity_id=self.entity_id,
                    consciousness_state=consciousness_state,
                    position=tuple(
                        int(x)
                        for x in getattr(self.kinetic_state, "position", (0, 0, 0))
                    ),
                    influence_type=(
                        "order" if self.node_type == "sephirot" else "chaos"
                    ),
                    strength=effective_strength,
                    radius=self.influence_radius,
                    duration=max(0.5, float(self.influence_radius) * 0.2),
                )
                deformed_id = resp.get("crystal_id") if isinstance(resp, dict) else None
            except Exception:
                logger.exception("Fallback apply_consciousness_influence failed")

        # Fallback 2: local Gaussian imprint (direct field write)
        if deformed_id is None:
            try:
                deformed_id = self._imprint_gaussian_local_field(
                    amplitude=effective_strength, radius=float(self.influence_radius)
                )
            except Exception:
                logger.exception("Local Gaussian imprint failed")

        # Record in node history
        self.interaction_history.append(
            {
                "type": "embodied_pulse",
                "mass": float(self.mass),
                "effective_strength": float(effective_strength),
                "impulses_count": len(impulses) if impulses else 0,
                "crystal_id": deformed_id,
                "time": getattr(self, "age", 0.0),
            }
        )
        return deformed_id

    def imprint_mass_deformation(self) -> str | None:
        """
        Explicitly imprint this node's mass as a deformation without impulses.
        Useful for static giants continuously shaping space.
        """
        return self.pulse([])

    # ------- Cognitive behavior (minimal by default) -------
    def analyze(self, data: Any) -> list[CognitiveImpulse]:
        """
        Basic implementation: CosmicNodes primarily act through embodied deformation.
        Emit a minimal impulse when activation exceeds threshold.
        """
        if self.activation_level > self.activation_threshold:
            impulse = CognitiveImpulse(
                impulse_type=ImpulseType.PERCEPTION_INTUITION,
                content={"cosmic_awareness": True, "node_type": self.node_type},
                intensity=float(self.activation_level),
                confidence=1.0,
                source_node=getattr(self, "entity_id", self.node_name),
            )
            return [impulse]
        return []

    # ------- Internal helpers -------
    def _imprint_gaussian_local_field(
        self, amplitude: float, radius: float
    ) -> str | None:
        """
        Best-effort local imprint if manifold lacks high-level APIs.
        Writes a Gaussian bump around the node's integer position into a generic 'field_value'
        channel (or augments dict states), then attempts to crystallize a trace if SnowModel APIs exist.
        """
        if not hasattr(self.manifold_ref, "get_state_at") or not hasattr(
            self.manifold_ref, "set_state_at"
        ):
            return None

        center = tuple(int(x) for x in self.kinetic_state.position)
        dims = getattr(self.manifold_ref, "dimensions", None)
        if dims is None:
            return None

        r = int(max(1, round(radius)))
        sigma = max(1.0, radius / 2.0)
        amp = float(max(0.0, amplitude))

        # Cap workload for safety
        r_cap = min(r, 8)
        for dx in range(-r_cap, r_cap + 1):
            for dy in range(-r_cap, r_cap + 1):
                for dz in range(-r_cap, r_cap + 1):
                    x, y, z = center[0] + dx, center[1] + dy, center[2] + dz
                    if (
                        x < 0
                        or y < 0
                        or z < 0
                        or x >= dims[0]
                        or y >= dims[1]
                        or z >= dims[2]
                    ):
                        continue
                    dist2 = float(dx * dx + dy * dy + dz * dz)
                    k = math.exp(-(dist2) / (2.0 * sigma * sigma))
                    if k <= 1e-9:
                        continue
                    try:
                        s = self.manifold_ref.get_state_at((x, y, z))
                    except Exception:
                        s = None
                    # Update state conservatively
                    try:
                        if s is None:
                            self.manifold_ref.set_state_at(
                                (x, y, z), {"field_value": amp * k}
                            )
                        elif isinstance(s, dict):
                            fv = float(s.get("field_value", 0.0)) + (amp * k)
                            s["field_value"] = fv
                            self.manifold_ref.set_state_at((x, y, z), s)
                        elif isinstance(s, (int, float, np.number)):
                            self.manifold_ref.set_state_at(
                                (x, y, z), float(s) + (amp * k)
                            )
                        elif isinstance(s, np.ndarray):
                            arr = np.asarray(s, dtype=float)
                            arr.flat[0] = float(arr.flat[0]) + (amp * k)
                            self.manifold_ref.set_state_at((x, y, z), arr)
                        else:
                            # Fallback to dict channel
                            self.manifold_ref.set_state_at(
                                (x, y, z), {"field_value": amp * k}
                            )
                    except Exception:
                        # continue best-effort
                        continue

        # Try to register a persistent crystal (Snow Model helpers if present)
        try:
            if hasattr(self.manifold_ref, "_qm_crystallize_trace"):
                cid = self.manifold_ref._qm_crystallize_trace(
                    entity_id=self.entity_id,
                    position=center,
                    pattern=None,
                    strength=min(1.0, amp),
                    provenance={
                        "node": self.node_name,
                        "type": self.node_type,
                        "mode": "gaussian_local_imprint",
                    },
                    signature=None,
                )
                return cid
        except Exception:
            pass
        return None
