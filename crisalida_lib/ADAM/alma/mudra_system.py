"""
Mudra System - definitive, hardened implementation.

Improvements:
- Stronger typing and defensive imports.
- EVA (best-effort) recording of activations (sync/async-aware).
- Extensible Mudra registry, serialization, safe metrics and diagnostics.
- Clearer invariants: clamps, cooldowns, bounded histories.
- Utilities: register/deregister custom mudras, get statistics, reset metrics.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Defensive imports / TYPE_CHECKING to avoid circulars in runtime
if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import AdamConfig  # type: ignore
    from crisalida_lib.ADAM.enums import ChakraType  # type: ignore
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )
else:
    AdamConfig = Any
    EVAMemoryManager = Any
    ChakraType = Any

# local chakra helpers (consumer expectation)
from crisalida_lib.ADAM.config import AdamConfig  # runtime default import
from crisalida_lib.ADAM.enums import ChakraType
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import EVAMemoryManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MudraEffect:
    """Represents a temporal effect produced by a Mudra activation."""

    effect_type: str
    magnitude: float
    duration: float
    target_position: tuple[float, float, float] | None = None
    creation_time: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, now: float | None = None) -> bool:
        now = now or time.time()
        elapsed = now - self.creation_time
        if elapsed >= self.duration:
            self.is_active = False
            return False
        # gentle exponential-like decay to avoid numeric issues
        lifetime_frac = max(0.0, min(1.0, elapsed / max(1e-6, self.duration)))
        self.magnitude *= max(0.0, 1.0 - 0.5 * lifetime_frac)
        return True


class Mudra(ABC):
    """Abstract base for all mudras. Implementers should be idempotent and robust."""

    def __init__(
        self,
        name: str,
        required_chakras: list[ChakraType],
        base_energy_cost: float = 0.1,
        cooldown_time: float = 1.0,
        max_object_mass: float = 0.0,
        max_distance: float = 0.0,
        accuracy_base: float = 0.5,
    ) -> None:
        self.name = name
        self.required_chakras = list(required_chakras)
        self.base_energy_cost = float(max(0.0, base_energy_cost))
        self.cooldown_time = float(max(0.0, cooldown_time))
        self.last_activation_time = 0.0
        self.activation_count = 0
        self.success_rate = 0.8
        self.active_effects: list[MudraEffect] = []
        self.max_object_mass = float(max_object_mass)
        self.max_distance = float(max_distance)
        self.accuracy_base = float(max(0.0, min(1.0, accuracy_base)))
        # bounded history for diagnostics
        self._history: list[dict[str, Any]] = []

    # implementors must compute realistic energy cost and not exceed 1.0
    @abstractmethod
    def calculate_energy_cost(self, chakra_system: Any, **kwargs: Any) -> float:
        raise NotImplementedError

    # implementors produce a MudraEffect (or None on soft-failure)
    @abstractmethod
    def execute_effect(
        self, physics_engine: Any, chakra_system: Any, **kwargs: Any
    ) -> MudraEffect | None:
        raise NotImplementedError

    def can_activate(self, chakra_system: Any) -> tuple[bool, str]:
        now = time.time()
        if now - self.last_activation_time < self.cooldown_time:
            remaining = self.cooldown_time - (now - self.last_activation_time)
            return False, f"Cooldown activo: {remaining:.1f}s restantes"
        if not getattr(chakra_system, "chakras", None):
            return False, "Chakra system invalido"
        for chakra_type in self.required_chakras:
            chakra = chakra_system.chakras.get(chakra_type)
            if chakra is None:
                return (
                    False,
                    f"Chakra {getattr(chakra_type, 'value', str(chakra_type))} no disponible",
                )
            if chakra.blockage_level >= 0.85:
                return False, f"Chakra {chakra_type.value} está severamente bloqueado"
            min_energy_needed = self.base_energy_cost / max(
                1, len(self.required_chakras)
            )
            if chakra.current_energy < min_energy_needed:
                return False, f"Energía insuficiente en {chakra_type.value}"
        return True, "OK"

    def _record_local_history(self, entry: dict[str, Any]) -> None:
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def activate(
        self, chakra_system: Any, physics_engine: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        ok, reason = self.can_activate(chakra_system)
        now = time.time()
        if not ok:
            entry = {"timestamp": now, "success": False, "reason": reason}
            self._record_local_history(entry)
            return {"success": False, "reason": reason, "energy_consumed": 0.0}

        try:
            energy_cost = float(
                max(0.0, min(1.0, self.calculate_energy_cost(chakra_system, **kwargs)))
            )
            per_chakra = energy_cost / max(1, len(self.required_chakras))
            consumed = 0.0
            for ct in self.required_chakras:
                chakra = chakra_system.chakras[ct]
                # consume energy; chakra.consume_energy returns bool
                if chakra.consume_energy(per_chakra, purpose=self.name):
                    consumed += per_chakra
                else:
                    entry = {
                        "timestamp": now,
                        "success": False,
                        "reason": f"Insufficient energy in {ct.value}",
                    }
                    self._record_local_history(entry)
                    return {
                        "success": False,
                        "reason": f"No se pudo consumir energía de {ct.value}",
                        "energy_consumed": 0.0,
                    }

            # success chance depends on system alignment and mudra success_rate
            alignment = getattr(chakra_system, "get_overall_alignment", lambda: 0.5)()
            success_chance = self.success_rate * (0.5 + 0.5 * float(alignment))
            if random.random() < success_chance:
                effect = self.execute_effect(physics_engine, chakra_system, **kwargs)
                if effect:
                    self.active_effects.append(effect)
                self.last_activation_time = now
                self.activation_count += 1
                self.success_rate = min(0.98, self.success_rate + 0.001)
                entry = {
                    "timestamp": now,
                    "success": True,
                    "energy_consumed": consumed,
                    "effect": getattr(effect, "effect_type", None),
                }
                self._record_local_history(entry)
                logger.info("Mudra '%s' activated (energy=%0.3f)", self.name, consumed)
                return {
                    "success": True,
                    "energy_consumed": consumed,
                    "effect": effect,
                    "alignment": alignment,
                }
            else:
                # partial cost on failure (half)
                loss = consumed * 0.5
                entry = {
                    "timestamp": now,
                    "success": False,
                    "reason": "Concentración insuficiente",
                    "energy_consumed": loss,
                }
                self._record_local_history(entry)
                self.success_rate = max(0.05, self.success_rate - 0.01)
                logger.warning("Mudra '%s' failed to execute", self.name)
                return {
                    "success": False,
                    "reason": "Ejecución fallida - concentración insuficiente",
                    "energy_consumed": loss,
                }
        except Exception as exc:
            logger.exception("Error activating mudra %s: %s", self.name, exc)
            return {
                "success": False,
                "reason": f"Error interno: {exc}",
                "energy_consumed": 0.0,
            }

    def update_effects(self, delta_time: float, physics_engine: Any = None) -> None:
        now = time.time()
        keep: list[MudraEffect] = []
        for eff in list(self.active_effects):
            if eff.update(now):
                keep.append(eff)
                # continuous effect application hook (optional)
                try:
                    self._apply_continuous_effect(eff, physics_engine, delta_time)
                except Exception:
                    logger.debug(
                        "Continuous effect hook failed for %s",
                        eff.effect_type,
                        exc_info=True,
                    )
        self.active_effects = keep

    @abstractmethod
    def _apply_continuous_effect(
        self, effect: MudraEffect, physics_engine: Any, delta_time: float
    ) -> None:
        raise NotImplementedError

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        cooldown_remaining = max(
            0.0, self.cooldown_time - (now - self.last_activation_time)
        )
        return {
            "name": self.name,
            "activation_count": int(self.activation_count),
            "success_rate": float(self.success_rate),
            "cooldown_remaining": float(cooldown_remaining),
            "active_effects": len(self.active_effects),
            "history_len": len(self._history),
            "required_chakras": [
                getattr(c, "value", str(c)) for c in self.required_chakras
            ],
            "base_energy_cost": float(self.base_energy_cost),
        }

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation_count": self.activation_count,
            "success_rate": self.success_rate,
            "last_activation_time": self.last_activation_time,
            "active_effects": [
                {
                    "effect_type": e.effect_type,
                    "magnitude": e.magnitude,
                    "duration": e.duration,
                    "target_position": e.target_position,
                    "is_active": e.is_active,
                }
                for e in self.active_effects
            ],
        }


# --- Concrete Mudras ----------------------------------------------------
class TelekinesisMudra(Mudra):
    def __init__(self) -> None:
        super().__init__(
            name="Telequinesis",
            required_chakras=[ChakraType.SACRO, ChakraType.PLEXO_SOLAR],
            base_energy_cost=0.15,
            cooldown_time=2.0,
            max_object_mass=10.0,
            max_distance=15.0,
            accuracy_base=0.7,
        )
        self.force_multiplier = 100.0

    def calculate_energy_cost(
        self,
        chakra_system: Any,
        target_mass: float = 1.0,
        distance: float = 5.0,
        force_magnitude: float = 1.0,
        **_,
    ) -> float:
        mass_factor = (
            1.0
            + (min(target_mass, self.max_object_mass) / max(1.0, self.max_object_mass))
            * 0.5
        )
        dist_factor = (
            1.0 + (min(distance, self.max_distance) / max(1.0, self.max_distance)) * 0.3
        )
        force_factor = 1.0 + float(force_magnitude) * 0.2
        # chakra efficiency heuristic
        sac = chakra_system.chakras.get(ChakraType.SACRO)
        plex = chakra_system.chakras.get(ChakraType.PLEXO_SOLAR)
        efficiency = 1.0
        if sac and plex:
            avg_e = (sac.current_energy + plex.current_energy) / 2.0
            avg_b = (sac.blockage_level + plex.blockage_level) / 2.0
            efficiency = max(0.05, avg_e * (1.0 - avg_b * 0.5))
        total = (
            self.base_energy_cost
            * mass_factor
            * dist_factor
            * force_factor
            * (2.0 - efficiency)
        )
        return float(min(max(0.0, total), 1.0))

    def execute_effect(
        self,
        physics_engine: Any,
        chakra_system: Any,
        target_object_id: str = "",
        target_position: tuple[float, float, float] | None = None,
        force_vector: tuple[float, float, float] = (0, 0, 1),
        **_,
    ) -> MudraEffect | None:
        sac_infl = chakra_system.get_chakra_influence(
            ChakraType.SACRO, "telekinesis_power"
        )
        plex_infl = chakra_system.get_chakra_influence(
            ChakraType.PLEXO_SOLAR, "energy_manipulation"
        )
        effective = self.force_multiplier * max(0.0, (sac_infl + plex_infl) / 2.0)
        effect = MudraEffect(
            effect_type="telekinesis",
            magnitude=effective,
            duration=5.0,
            target_position=target_position,
        )
        if physics_engine and hasattr(physics_engine, "apply_force_to_object"):
            try:
                physics_engine.apply_force_to_object(
                    object_id=target_object_id,
                    force_vector=(
                        force_vector[0] * effective,
                        force_vector[1] * effective,
                        force_vector[2] * effective,
                    ),
                    duration=effect.duration,
                )
            except Exception as exc:
                logger.warning("Physics engine force application failed: %s", exc)
                effect.is_active = False
        return effect

    def _apply_continuous_effect(
        self, effect: MudraEffect, physics_engine: Any, delta_time: float
    ) -> None:
        # Basic decay and optional continuous force application
        if (
            physics_engine
            and hasattr(physics_engine, "apply_continuous_force")
            and effect.target_position is not None
        ):
            try:
                physics_engine.apply_continuous_force(
                    position=effect.target_position,
                    magnitude=effect.magnitude * 0.01 * delta_time,
                )
            except Exception:
                logger.debug("apply_continuous_force failed (ignored)", exc_info=True)
        effect.magnitude *= max(0.0, 1.0 - 0.05 * delta_time)


class BlinkMudra(Mudra):
    def __init__(self) -> None:
        super().__init__(
            name="Blink",
            required_chakras=[ChakraType.TERCER_OJO, ChakraType.CORONA],
            base_energy_cost=0.2,
            cooldown_time=1.0,
            max_distance=20.0,
            accuracy_base=0.8,
        )

    def calculate_energy_cost(
        self,
        chakra_system: Any,
        distance: float = 10.0,
        dimensional_complexity: float = 1.0,
        **_,
    ) -> float:
        dist_factor = (
            1.0 + (min(distance, self.max_distance) / max(1.0, self.max_distance)) * 1.2
        )
        comp_factor = 1.0 + float(dimensional_complexity) * 0.5
        t = chakra_system.chakras.get(ChakraType.TERCER_OJO)
        c = chakra_system.chakras.get(ChakraType.CORONA)
        efficiency = 1.0
        if t and c:
            efficiency = max(
                0.05,
                (
                    (t.current_energy * (1.0 - t.blockage_level))
                    + (c.current_energy * (1.0 - c.blockage_level))
                )
                / 2.0,
            )
        total = (
            self.base_energy_cost * dist_factor * comp_factor * (2.5 - efficiency * 1.5)
        )
        return float(min(max(0.0, total), 1.0))

    def execute_effect(
        self,
        physics_engine: Any,
        chakra_system: Any,
        target_position: tuple[float, float, float] | None = None,
        entity_id: str | None = None,
        dimensional_complexity: float = 1.0,
        **_,
    ) -> MudraEffect | None:
        if target_position is None:
            raise ValueError("target_position is required for BlinkMudra")
        t_infl = chakra_system.get_chakra_influence(
            ChakraType.TERCER_OJO, "blink_accuracy"
        )
        c_infl = chakra_system.get_chakra_influence(
            ChakraType.CORONA, "reality_perception"
        )
        accuracy = max(0.0, min(1.0, self.accuracy_base * ((t_infl + c_infl) / 2.0)))
        # compute jitter based on accuracy
        x, y, z = target_position
        max_err = (1.0 - accuracy) * 5.0
        final = (
            x + random.uniform(-max_err, max_err),
            y + random.uniform(-max_err, max_err),
            z + random.uniform(-max_err, max_err),
        )
        effect = MudraEffect(
            effect_type="blink", magnitude=1.0, duration=0.1, target_position=final
        )
        if physics_engine and hasattr(physics_engine, "teleport_entity"):
            try:
                ok = physics_engine.teleport_entity(
                    entity_id=entity_id,
                    new_position=final,
                    dimensional_complexity=dimensional_complexity,
                )
                if not ok:
                    effect.is_active = False
            except Exception:
                logger.warning("Teleport failed", exc_info=True)
                effect.is_active = False
        return effect

    def _apply_continuous_effect(
        self, effect: MudraEffect, physics_engine: Any, delta_time: float
    ) -> None:
        # Blink is instantaneous; keep small decay for bookkeeping
        effect.magnitude *= max(0.0, 1.0 - 0.1 * delta_time)


# --- System wrapper -------------------------------------------------------
class SistemaDeMudras:
    """Registry and runtime for mudra management and metrics."""

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config = config or AdamConfig()
        self.eva_manager = eva_manager
        self.entity_id = entity_id

        # Default registry - can be extended at runtime
        self.available_mudras: dict[str, Mudra] = {
            "telekinesis": TelekinesisMudra(),
            "blink": BlinkMudra(),
        }

        self.total_mudra_activations = 0
        self.successful_mudra_activations = 0
        self._activation_history: list[dict[str, Any]] = []

    def register_mudra(self, name: str, mudra: Mudra) -> None:
        self.available_mudras[name] = mudra

    def deregister_mudra(self, name: str) -> None:
        self.available_mudras.pop(name, None)

    def activate_mudra(
        self,
        mudra_name: str,
        chakra_system: Any,
        physics_engine: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if mudra_name not in self.available_mudras:
            return {
                "success": False,
                "reason": f"Mudra '{mudra_name}' no está disponible",
                "available": list(self.available_mudras.keys()),
            }
        mudra = self.available_mudras[mudra_name]
        result = mudra.activate(chakra_system, physics_engine, **kwargs)
        self.total_mudra_activations += 1
        if result.get("success"):
            self.successful_mudra_activations += 1
        entry = {
            "timestamp": time.time(),
            "mudra": mudra_name,
            "result": {
                "success": bool(result.get("success")),
                "energy_consumed": float(result.get("energy_consumed", 0.0)),
            },
        }
        self._activation_history.append(entry)
        if len(self._activation_history) > 500:
            self._activation_history = self._activation_history[-500:]
        # best-effort EVA recording
        self._record_eva_event(mudra_name, result)
        return result

    def update_active_mudras(
        self, delta_time: float, physics_engine: Any = None
    ) -> None:
        for mudra in self.available_mudras.values():
            try:
                mudra.update_effects(delta_time, physics_engine)
            except Exception:
                logger.debug(
                    "update_effects failed for mudra %s", mudra.name, exc_info=True
                )

    def get_mudra_capabilities(self, chakra_system: Any) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        for name, mudra in self.available_mudras.items():
            status = mudra.get_status()
            if name == "telekinesis":
                sac = chakra_system.get_chakra_influence(
                    ChakraType.SACRO, "telekinesis_power"
                )
                plex = chakra_system.get_chakra_influence(
                    ChakraType.PLEXO_SOLAR, "energy_manipulation"
                )
                status["power_modifier"] = float((sac + plex) / 2.0)
                status["estimated_max_mass"] = float(
                    mudra.max_object_mass * status["power_modifier"]
                )
            elif name == "blink":
                t = chakra_system.get_chakra_influence(
                    ChakraType.TERCER_OJO, "blink_accuracy"
                )
                c = chakra_system.get_chakra_influence(
                    ChakraType.CORONA, "reality_perception"
                )
                status["accuracy_modifier"] = float((t + c) / 2.0)
                status["estimated_max_distance"] = float(
                    mudra.max_distance * status["accuracy_modifier"]
                )
            caps[name] = status
        return caps

    def get_statistics(self) -> dict[str, Any]:
        success_rate = self.successful_mudra_activations / max(
            1, self.total_mudra_activations
        )
        return {
            "total_activations": int(self.total_mudra_activations),
            "successful_activations": int(self.successful_mudra_activations),
            "global_success_rate": float(success_rate),
            "registered_mudras": list(self.available_mudras.keys()),
            "recent_activations": list(self._activation_history[-20:]),
        }

    def reset_metrics(self) -> None:
        self.total_mudra_activations = 0
        self.successful_mudra_activations = 0
        self._activation_history = []
        for m in self.available_mudras.values():
            m.activation_count = 0
            m._history = []

    def to_serializable(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "timestamp": time.time(),
            "mudras": {
                name: mudra.to_serializable()
                for name, mudra in self.available_mudras.items()
            },
            "stats": self.get_statistics(),
        }

    def load_serializable(self, data: dict[str, Any]) -> None:
        try:
            mudras = data.get("mudras", {})
            for name, state in mudras.items():
                if name in self.available_mudras:
                    # apply state to existing mudra where sensible
                    m = self.available_mudras[name]
                    m.activation_count = int(
                        state.get("activation_count", m.activation_count)
                    )
                    m.success_rate = float(state.get("success_rate", m.success_rate))
                    m.last_activation_time = float(
                        state.get("last_activation_time", m.last_activation_time)
                    )
                    # active effects restored only minimally
                    effs = state.get("active_effects", [])
                    m.active_effects = [
                        MudraEffect(
                            effect_type=e.get("effect_type", "unknown"),
                            magnitude=float(e.get("magnitude", 0.0)),
                            duration=float(e.get("duration", 0.0)),
                            target_position=(
                                tuple(e.get("target_position"))
                                if e.get("target_position")
                                else None
                            ),
                            is_active=bool(e.get("is_active", True)),
                        )
                        for e in effs
                    ]
        except Exception:
            logger.exception("Failed to load SistemaDeMudras serializable data")

    # --- EVA helpers (best-effort, sync/async aware) -----------------
    def _record_eva_event(self, mudra_name: str, result: dict[str, Any]) -> None:
        if not self.eva_manager:
            return
        try:
            rec = getattr(self.eva_manager, "record_experience", None)
            if rec is None:
                return
            experience_id = f"mudra:{self.entity_id}:{mudra_name}:{int(time.time())}"
            payload = {
                "entity_id": self.entity_id,
                "mudra": mudra_name,
                "result": {
                    "success": bool(result.get("success")),
                    "energy": float(result.get("energy_consumed", 0.0)),
                },
                "timestamp": time.time(),
            }
            coro_or_res = rec(
                entity_id=self.entity_id,
                event_type="mudra_activation",
                data=payload,
                experience_id=experience_id,
            )
            if hasattr(coro_or_res, "__await__"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(coro_or_res)
                    else:
                        loop.run_until_complete(coro_or_res)
                except Exception:
                    logger.debug(
                        "Could not schedule async EVA record for mudra", exc_info=True
                    )
        except Exception:
            logger.exception("Failed to record mudra event to EVA")
