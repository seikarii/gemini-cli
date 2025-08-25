"""
Hormonal System
===============

This module defines the SistemaHormonal class, which manages the neuroendocrine
regulation of the entity.

This file has been refactored and completed with the full implementation.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from threading import RLock
from typing import TYPE_CHECKING, Any

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.enums import NeurotransmitterType

if TYPE_CHECKING:
    from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
        EVAMemoryManager,  # type: ignore
    )
    from crisalida_lib.EVA.types import QualiaState  # type: ignore
else:
    EVAMemoryManager = Any  # runtime fallback
    QualiaState = Any  # runtime fallback

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _clamp01(v: float) -> float:
    try:
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.0


class SistemaHormonal:
    """
    SistemaHormonal - Sistema avanzado de regulación neuroendocrina para SO-Ser.

    Improvements in this "definitive" version:
      - Thread-safety with RLock.
      - Async-aware EVA recording (best-effort).
      - Serialization / snapshot helpers.
      - Tick() for incremental simulation using dt seconds.
      - Safe handling of QualiaState as mapping or object.
      - Utility methods for calibration, reset and trend analysis.
    """

    def __init__(
        self,
        config: AdamConfig | None = None,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
    ) -> None:
        self.config: AdamConfig = config or AdamConfig()  # type: ignore
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id

        self._lock = RLock()

        # initialize maps keyed by enum members
        self.niveles: dict[NeurotransmitterType, float] = dict.fromkeys(
            NeurotransmitterType, 0.5
        )
        self.niveles_base: dict[NeurotransmitterType, float] = dict.fromkeys(
            NeurotransmitterType, 0.5
        )
        self.synthesis_rates: dict[NeurotransmitterType, float] = dict.fromkeys(
            NeurotransmitterType, 0.1
        )
        self.degradation_rates: dict[NeurotransmitterType, float] = dict.fromkeys(
            NeurotransmitterType, 0.1
        )

        self._initialize_defaults()

        self.historial_niveles: list[dict[str, Any]] = []
        self.last_update: float = time.time()
        self.desequilibrios_activos: dict[str, Any] = {}
        self.modulacion_externa: dict[str, float] = {
            "estres_cronico": 0.0,
            "ejercicio": 0.0,
            "nutricion": 0.0,
            "sueno": 0.0,
            "interaccion_social": 0.0,
            "estimulacion_cognitiva": 0.0,
        }

    def _initialize_defaults(self) -> None:
        with self._lock:
            # sensible physiological defaults (kept from original but clamped)
            base_values = {
                NeurotransmitterType.DOPAMINE: 0.5,
                NeurotransmitterType.SEROTONIN: 0.6,
                NeurotransmitterType.CORTISOL: 0.15,
                NeurotransmitterType.ADRENALINE: 0.05,
                NeurotransmitterType.OXYTOCIN: 0.4,
                NeurotransmitterType.GABA: 0.7,
                NeurotransmitterType.ACETYLCHOLINE: 0.6,
                NeurotransmitterType.NOREPINEPHRINE: 0.25,
            }
            for nt, v in base_values.items():
                self.niveles_base[nt] = _clamp01(v)
                self.niveles[nt] = _clamp01(v)
            # synthesis / degradation defaults (conservative)
            synth = {
                NeurotransmitterType.DOPAMINE: 0.08,
                NeurotransmitterType.SEROTONIN: 0.06,
                NeurotransmitterType.CORTISOL: 0.05,
                NeurotransmitterType.ADRENALINE: 0.15,
                NeurotransmitterType.OXYTOCIN: 0.07,
                NeurotransmitterType.GABA: 0.09,
                NeurotransmitterType.ACETYLCHOLINE: 0.12,
                NeurotransmitterType.NOREPINEPHRINE: 0.10,
            }
            degr = {
                NeurotransmitterType.DOPAMINE: 0.06,
                NeurotransmitterType.SEROTONIN: 0.05,
                NeurotransmitterType.CORTISOL: 0.03,
                NeurotransmitterType.ADRENALINE: 0.20,
                NeurotransmitterType.OXYTOCIN: 0.08,
                NeurotransmitterType.GABA: 0.07,
                NeurotransmitterType.ACETYLCHOLINE: 0.15,
                NeurotransmitterType.NOREPINEPHRINE: 0.12,
            }
            for nt, v in synth.items():
                self.synthesis_rates[nt] = max(0.0, float(v))
            for nt, v in degr.items():
                self.degradation_rates[nt] = max(0.0, float(v))

    # ------------------ Runtime API ------------------
    def tick(self, dt: float = 1.0) -> dict[str, Any]:
        """
        Incremental simulation step. dt is seconds elapsed since previous tick.
        Returns the emitted hormonal state snapshot.
        """
        with self._lock:
            try:
                now = time.time()
                delta_time = max(0.0, float(dt))
                # natural decay + homeostatic synthesis nudging toward baseline
                for nt in list(self.niveles.keys()):
                    decay = self.degradation_rates.get(nt, 0.05) * delta_time * 0.01
                    self.niveles[nt] = _clamp01(self.niveles[nt] - decay)
                    base = self.niveles_base.get(nt, 0.5)
                    diff = base - self.niveles[nt]
                    # small homeostatic synthesis proportional to diff and synthesis rate
                    synth = min(
                        abs(diff),
                        self.synthesis_rates.get(nt, 0.05) * delta_time * 0.01,
                    )
                    if diff > 0:
                        self.niveles[nt] = _clamp01(self.niveles[nt] + synth)
                # apply slow external modulation decay (return toward 0)
                for k in list(self.modulacion_externa.keys()):
                    self.modulacion_externa[k] *= max(0.0, 1.0 - 0.01 * delta_time)
                # interactions and detection
                self._handle_neurotransmitter_interactions()
                desequilibrios = self._detect_imbalances()
                self._update_history_snapshot()
                snapshot = self.get_state()
                # async best-effort snapshot into EVA if available but non-blocking
                if self.eva_manager:
                    asyncio.ensure_future(
                        self._record_eva_async("hormonal_tick", snapshot)
                    )
                return snapshot
            except Exception:
                logger.exception("tick failed")
                return {}

    def update(
        self, event: str, intensity: float = 1.0, duration: float = 1.0
    ) -> dict[str, Any]:
        """
        Compatibility wrapper that mirrors older API: apply event-driven changes and return state.
        """
        with self._lock:
            current_time = time.time()
            delta_time = max(0.0, current_time - self.last_update)
            self.last_update = current_time

            niveles_previos = copy.deepcopy(self.niveles)

            self._apply_natural_degradation(delta_time)
            event_changes = self._process_specific_event(event, intensity, duration)
            self._apply_external_modulations(delta_time)
            self._handle_neurotransmitter_interactions()
            desequilibrios = self._detect_imbalances()
            self._update_history(niveles_previos)
            estado_hormonal = self._calculate_hormonal_state()

            current_state = {
                "niveles_actuales": {
                    k.value: float(v) for k, v in self.niveles.items()
                },
                "cambios_por_evento": {
                    k.value: float(v) for k, v in event_changes.items()
                },
                "desequilibrios": desequilibrios,
                "estado_hormonal": estado_hormonal,
                "coherencia_sistema": self._calculate_system_coherence(),
                "recomendaciones": self._generate_balance_recommendations(),
                "timestamp": time.time(),
            }

            # Best-effort EVA recording (async-aware)
            if self.eva_manager:
                try:
                    recorder = getattr(self.eva_manager, "record_experience", None)
                    if callable(recorder):
                        res = recorder(
                            entity_id=self.entity_id,
                            event_type="hormonal_state_update",
                            data=current_state,
                        )
                        if hasattr(res, "__await__"):
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(res)
                            else:
                                loop.run_until_complete(res)
                except Exception:
                    logger.exception(
                        "Failed to record hormonal state to EVA (non-fatal)"
                    )

            return current_state

    def inject_qualia_event(self, qualia_state: QualiaState | None) -> dict[str, Any]:
        """
        Injects a QualiaState (mapping or object) to modulate neurotransmitter levels.
        Returns a report with applied modifications and coherence metrics.
        """
        if qualia_state is None:
            return {"error": "no qualia_state provided"}

        with self._lock:
            # helper to read qualia values from mapping or object-like
            def _qget(key: str, default: float = 0.0) -> float:
                try:
                    if isinstance(qualia_state, dict):
                        return float(qualia_state.get(key, default))
                    return float(getattr(qualia_state, key, default))
                except Exception:
                    return float(default)

            consciousness_density = _qget("consciousness_density", 0.5)
            temporal_coherence = _qget("temporal_coherence", 0.5)
            emotional_valence = _qget("emotional_valence", 0.0)
            arousal = _qget("arousal", 0.0)
            cognitive_complexity = _qget("cognitive_complexity", 0.0)

            qualia_modifications: dict[NeurotransmitterType, float] = {}

            # heuristics (kept and slightly smoothed)
            if consciousness_density > 0.7:
                qualia_modifications[NeurotransmitterType.ACETYLCHOLINE] = 0.05
            elif consciousness_density < 0.3:
                qualia_modifications[NeurotransmitterType.ACETYLCHOLINE] = -0.03
            if temporal_coherence > 0.8:
                qualia_modifications[NeurotransmitterType.GABA] = 0.04
            elif temporal_coherence < 0.2:
                qualia_modifications[NeurotransmitterType.GABA] = -0.05
            if emotional_valence > 0.5:
                qualia_modifications[NeurotransmitterType.DOPAMINE] = (
                    emotional_valence * 0.06
                )
                qualia_modifications[NeurotransmitterType.SEROTONIN] = (
                    emotional_valence * 0.04
                )
            elif emotional_valence < -0.5:
                qualia_modifications[NeurotransmitterType.CORTISOL] = (
                    abs(emotional_valence) * 0.08
                )
            if arousal > 0.6:
                qualia_modifications[NeurotransmitterType.ADRENALINE] = arousal * 0.10
                qualia_modifications[NeurotransmitterType.NOREPINEPHRINE] = (
                    arousal * 0.06
                )
            if cognitive_complexity > 0.8:
                qualia_modifications[NeurotransmitterType.ACETYLCHOLINE] = (
                    qualia_modifications.get(NeurotransmitterType.ACETYLCHOLINE, 0.0)
                    + 0.05
                )
                qualia_modifications[NeurotransmitterType.CORTISOL] = (
                    qualia_modifications.get(NeurotransmitterType.CORTISOL, 0.0) + 0.02
                )

            for nt, change in qualia_modifications.items():
                self.niveles[nt] = _clamp01(self.niveles.get(nt, 0.0) + float(change))

            qualia_feedback = self._calculate_qualia_feedback()
            coherence = self._calculate_mind_body_coherence(qualia_state)

            report = {
                "modificaciones_aplicadas": {
                    k.value: float(v) for k, v in qualia_modifications.items()
                },
                "qualia_feedback": qualia_feedback,
                "coherencia_mente_cuerpo": float(coherence),
                "estado_post_inyeccion": {
                    k.value: float(v) for k, v in self.niveles.items()
                },
                "timestamp": time.time(),
            }

            # record to EVA best-effort
            if self.eva_manager:
                try:
                    recorder = getattr(self.eva_manager, "record_experience", None)
                    if callable(recorder):
                        res = recorder(
                            entity_id=self.entity_id,
                            event_type="qualia_injection",
                            data=report,
                        )
                        if hasattr(res, "__await__"):
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(res)
                            else:
                                loop.run_until_complete(res)
                except Exception:
                    logger.exception(
                        "Failed to record qualia injection to EVA (non-fatal)"
                    )

            return report

    # ------------------ Internal mechanics (original logic preserved, hardened) ------------------
    def _apply_natural_degradation(self, delta_time: float) -> None:
        for neurotransmisor in list(self.niveles.keys()):
            degradacion = self.degradation_rates.get(neurotransmisor, 0.05) * delta_time
            self.niveles[neurotransmisor] = _clamp01(
                self.niveles[neurotransmisor] - degradacion
            )
            nivel_base = self.niveles_base.get(neurotransmisor, 0.5)
            diferencia = nivel_base - self.niveles[neurotransmisor]
            homeostasis_rate = 0.02 * delta_time
            if diferencia > 0:
                sintesis = min(
                    diferencia,
                    self.synthesis_rates.get(neurotransmisor, 0.05)
                    * delta_time
                    * homeostasis_rate,
                )
                self.niveles[neurotransmisor] = _clamp01(
                    self.niveles[neurotransmisor] + sintesis
                )

    def _process_specific_event(
        self, event: str, intensity: float, duration: float
    ) -> dict[NeurotransmitterType, float]:
        cambios: dict[NeurotransmitterType, float] = {}
        base_intensity = min(2.0, max(0.0, float(intensity)))
        # event mapping preserved; apply scaled by duration
        event_map = {
            "success": {
                NeurotransmitterType.DOPAMINE: 0.15,
                NeurotransmitterType.SEROTONIN: 0.10,
                NeurotransmitterType.CORTISOL: -0.08,
                NeurotransmitterType.OXYTOCIN: 0.05,
            },
            "failure": {
                NeurotransmitterType.CORTISOL: 0.20,
                NeurotransmitterType.DOPAMINE: -0.10,
                NeurotransmitterType.SEROTONIN: -0.08,
                NeurotransmitterType.NOREPINEPHRINE: 0.12,
            },
            "stress": {
                NeurotransmitterType.CORTISOL: 0.25,
                NeurotransmitterType.ADRENALINE: 0.30,
                NeurotransmitterType.NOREPINEPHRINE: 0.20,
                NeurotransmitterType.GABA: -0.15,
                NeurotransmitterType.SEROTONIN: -0.10,
            },
            "social_bonding": {
                NeurotransmitterType.OXYTOCIN: 0.20,
                NeurotransmitterType.SEROTONIN: 0.12,
                NeurotransmitterType.DOPAMINE: 0.08,
                NeurotransmitterType.CORTISOL: -0.05,
            },
            "learning": {
                NeurotransmitterType.ACETYLCHOLINE: 0.18,
                NeurotransmitterType.DOPAMINE: 0.10,
                NeurotransmitterType.NOREPINEPHRINE: 0.08,
            },
            "relaxation": {
                NeurotransmitterType.GABA: 0.15,
                NeurotransmitterType.SEROTONIN: 0.10,
                NeurotransmitterType.CORTISOL: -0.12,
                NeurotransmitterType.ADRENALINE: -0.10,
            },
            "danger": {
                NeurotransmitterType.ADRENALINE: 0.40,
                NeurotransmitterType.NOREPINEPHRINE: 0.35,
                NeurotransmitterType.CORTISOL: 0.30,
                NeurotransmitterType.GABA: -0.20,
            },
            "creative_flow": {
                NeurotransmitterType.DOPAMINE: 0.12,
                NeurotransmitterType.ACETYLCHOLINE: 0.10,
                NeurotransmitterType.NOREPINEPHRINE: 0.08,
                NeurotransmitterType.GABA: 0.05,
            },
        }
        if event in event_map:
            for nt, base_change in event_map[event].items():
                delta = base_change * base_intensity * float(duration)
                cambios[nt] = delta
        # Apply with clamping
        for neurotransmisor, cambio in cambios.items():
            self.niveles[neurotransmisor] = _clamp01(
                self.niveles.get(neurotransmisor, 0.0) + cambio
            )
        return cambios

    def _apply_external_modulations(self, delta_time: float) -> None:
        for factor, valor in list(self.modulacion_externa.items()):
            if valor != 0.0:
                self._apply_specific_modulation(factor, valor, delta_time)

    def _apply_specific_modulation(
        self, factor: str, valor: float, delta_time: float
    ) -> None:
        modulation_strength = float(valor) * max(0.0, delta_time)
        if factor == "ejercicio" and valor > 0:
            self.niveles[NeurotransmitterType.DOPAMINE] = _clamp01(
                self.niveles[NeurotransmitterType.DOPAMINE] + 0.05 * modulation_strength
            )
            self.niveles[NeurotransmitterType.SEROTONIN] = _clamp01(
                self.niveles[NeurotransmitterType.SEROTONIN]
                + 0.04 * modulation_strength
            )
            self.niveles[NeurotransmitterType.CORTISOL] = _clamp01(
                self.niveles[NeurotransmitterType.CORTISOL] - 0.03 * modulation_strength
            )
        elif factor == "estres_cronico" and valor > 0:
            self.niveles[NeurotransmitterType.CORTISOL] = _clamp01(
                self.niveles[NeurotransmitterType.CORTISOL] + 0.08 * modulation_strength
            )
            self.niveles[NeurotransmitterType.SEROTONIN] = _clamp01(
                self.niveles[NeurotransmitterType.SEROTONIN]
                - 0.05 * modulation_strength
            )
            self.niveles[NeurotransmitterType.GABA] = _clamp01(
                self.niveles[NeurotransmitterType.GABA] - 0.04 * modulation_strength
            )
        elif factor == "sueno" and valor < 0:
            self.niveles[NeurotransmitterType.CORTISOL] = _clamp01(
                self.niveles[NeurotransmitterType.CORTISOL]
                + 0.06 * abs(modulation_strength)
            )
            self.niveles[NeurotransmitterType.ACETYLCHOLINE] = _clamp01(
                self.niveles[NeurotransmitterType.ACETYLCHOLINE]
                - 0.05 * abs(modulation_strength)
            )
            self.niveles[NeurotransmitterType.GABA] = _clamp01(
                self.niveles[NeurotransmitterType.GABA]
                - 0.04 * abs(modulation_strength)
            )
        elif factor == "nutricion" and valor > 0:
            for neurotransmisor in list(self.synthesis_rates.keys()):
                self.synthesis_rates[neurotransmisor] = max(
                    0.0,
                    self.synthesis_rates.get(neurotransmisor, 0.05)
                    * (1.0 + 0.1 * modulation_strength),
                )
        elif factor == "interaccion_social" and valor > 0:
            self.niveles[NeurotransmitterType.OXYTOCIN] = _clamp01(
                self.niveles[NeurotransmitterType.OXYTOCIN] + 0.06 * modulation_strength
            )
            self.niveles[NeurotransmitterType.SEROTONIN] = _clamp01(
                self.niveles[NeurotransmitterType.SEROTONIN]
                + 0.04 * modulation_strength
            )
        elif factor == "estimulacion_cognitiva" and valor > 0:
            self.niveles[NeurotransmitterType.ACETYLCHOLINE] = _clamp01(
                self.niveles[NeurotransmitterType.ACETYLCHOLINE]
                + 0.05 * modulation_strength
            )
            self.niveles[NeurotransmitterType.DOPAMINE] = _clamp01(
                self.niveles[NeurotransmitterType.DOPAMINE] + 0.03 * modulation_strength
            )
        # clamp after modulations
        for neurotransmisor in list(self.niveles.keys()):
            self.niveles[neurotransmisor] = _clamp01(self.niveles[neurotransmisor])

    def _handle_neurotransmitter_interactions(self) -> None:
        # Dopamine-serotonin balancing
        if (
            self.niveles.get(NeurotransmitterType.DOPAMINE, 0.0) > 0.8
            and self.niveles.get(NeurotransmitterType.SEROTONIN, 0.0) < 0.3
        ):
            self.niveles[NeurotransmitterType.SEROTONIN] = _clamp01(
                self.niveles[NeurotransmitterType.SEROTONIN] + 0.02
            )
        # Cortisol reduces GABA gradually
        if self.niveles.get(NeurotransmitterType.CORTISOL, 0.0) > 0.7:
            self.niveles[NeurotransmitterType.GABA] = _clamp01(
                self.niveles[NeurotransmitterType.GABA] * 0.95
            )
        # Norepinephrine <> Acetylcholine heuristic
        if self.niveles.get(NeurotransmitterType.NOREPINEPHRINE, 0.0) > 0.8:
            if self.niveles.get(NeurotransmitterType.ACETYLCHOLINE, 0.0) < 0.7:
                self.niveles[NeurotransmitterType.ACETYLCHOLINE] = _clamp01(
                    self.niveles[NeurotransmitterType.ACETYLCHOLINE] + 0.01
                )
            else:
                self.niveles[NeurotransmitterType.ACETYLCHOLINE] = _clamp01(
                    self.niveles[NeurotransmitterType.ACETYLCHOLINE] - 0.02
                )
        # Oxytocin moderates cortisol
        if self.niveles.get(NeurotransmitterType.OXYTOCIN, 0.0) > 0.6:
            self.niveles[NeurotransmitterType.CORTISOL] = _clamp01(
                self.niveles[NeurotransmitterType.CORTISOL] * 0.98
            )

    def _detect_imbalances(self) -> dict[str, Any]:
        desequilibrios: dict[str, Any] = {}
        rangos_saludables = {
            NeurotransmitterType.DOPAMINE: (0.4, 0.8),
            NeurotransmitterType.SEROTONIN: (0.5, 0.9),
            NeurotransmitterType.CORTISOL: (0.05, 0.4),
            NeurotransmitterType.ADRENALINE: (0.0, 0.3),
            NeurotransmitterType.OXYTOCIN: (0.2, 0.7),
            NeurotransmitterType.GABA: (0.5, 0.9),
            NeurotransmitterType.ACETYLCHOLINE: (0.4, 0.8),
            NeurotransmitterType.NOREPINEPHRINE: (0.1, 0.5),
        }
        for neurotransmisor, nivel in self.niveles.items():
            rango_min, rango_max = rangos_saludables.get(neurotransmisor, (0.0, 1.0))
            if nivel < rango_min:
                desequilibrios[neurotransmisor.value] = {
                    "tipo": "deficiencia",
                    "severidad": float(
                        (rango_min - nivel) / rango_min if rango_min else 0.0
                    ),
                    "recomendacion": self._get_deficiency_recommendation(
                        neurotransmisor
                    ),
                }
            elif nivel > rango_max:
                sever = (nivel - rango_max) / max(1e-6, (1.0 - rango_max))
                desequilibrios[neurotransmisor.value] = {
                    "tipo": "exceso",
                    "severidad": float(sever),
                    "recomendacion": self._get_excess_recommendation(neurotransmisor),
                }
        patrones_complejos = self._detect_complex_patterns()
        if patrones_complejos:
            desequilibrios["patrones_complejos"] = patrones_complejos
        return desequilibrios

    def _detect_complex_patterns(self) -> dict[str, Any]:
        patrones: dict[str, Any] = {}
        # Depression-like pattern
        if (
            self.niveles.get(NeurotransmitterType.DOPAMINE, 0.0) < 0.3
            and self.niveles.get(NeurotransmitterType.SEROTONIN, 0.0) < 0.4
            and self.niveles.get(NeurotransmitterType.CORTISOL, 0.0) > 0.6
        ):
            patrones["depression_pattern"] = {
                "confidence": 0.8,
                "description": "Patrón neuroquímico consistente con estado depresivo",
            }
        # Anxiety-like pattern
        if (
            self.niveles.get(NeurotransmitterType.CORTISOL, 0.0) > 0.7
            and self.niveles.get(NeurotransmitterType.GABA, 0.0) < 0.4
            and self.niveles.get(NeurotransmitterType.NOREPINEPHRINE, 0.0) > 0.6
        ):
            patrones["anxiety_pattern"] = {
                "confidence": 0.75,
                "description": "Patrón neuroquímico consistente con estado ansioso",
            }
        # Chronic stress
        if (
            self.niveles.get(NeurotransmitterType.CORTISOL, 0.0) > 0.8
            and len(
                [
                    h
                    for h in self.historial_niveles[-10:]
                    if h.get("niveles", {}).get(NeurotransmitterType.CORTISOL.value, 0)
                    > 0.7
                ]
            )
            > 7
        ):
            patrones["chronic_stress_pattern"] = {
                "confidence": 0.9,
                "description": "Patrón de estrés crónico sostenido",
            }
        # Flow/optimal
        if (
            self.niveles.get(NeurotransmitterType.DOPAMINE, 0.0) > 0.6
            and self.niveles.get(NeurotransmitterType.ACETYLCHOLINE, 0.0) > 0.7
            and self.niveles.get(NeurotransmitterType.GABA, 0.0) > 0.6
            and self.niveles.get(NeurotransmitterType.CORTISOL, 0.0) < 0.3
        ):
            patrones["flow_pattern"] = {
                "confidence": 0.85,
                "description": "Patrón neuroquímico óptimo para estado de flow",
            }
        return patrones

    def _get_deficiency_recommendation(
        self, neurotransmisor: NeurotransmitterType
    ) -> str:
        recommendations = {
            NeurotransmitterType.DOPAMINE: "Incrementar actividades de recompensa, ejercicio, logros pequeños",
            NeurotransmitterType.SEROTONIN: "Mejorar rutina de sueño, exposición solar, actividades sociales",
            NeurotransmitterType.GABA: "Técnicas de relajación, meditación, reducir estimulantes",
            NeurotransmitterType.ACETYLCHOLINE: "Estimulación cognitiva, aprendizaje activo, concentración",
            NeurotransmitterType.OXYTOCIN: "Contacto social, actos de bondad, conexión emocional",
        }
        return recommendations.get(neurotransmisor, "Consultar con especialista")

    def _get_excess_recommendation(self, neurotransmisor: NeurotransmitterType) -> str:
        recommendations = {
            NeurotransmitterType.CORTISOL: "Reducir estrés, técnicas de relajación, descanso",
            NeurotransmitterType.ADRENALINE: "Evitar estimulantes, practicar calma, respiración profunda",
            NeurotransmitterType.NOREPINEPHRINE: "Reducir arousal, ambiente tranquilo, actividades calmantes",
        }
        return recommendations.get(neurotransmisor, "Moderar actividad relacionada")

    def _calculate_hormonal_state(self) -> dict[str, float]:
        bienestar = (
            self.niveles.get(NeurotransmitterType.SEROTONIN, 0.0)
            + self.niveles.get(NeurotransmitterType.OXYTOCIN, 0.0)
        ) / 2.0
        motivacion = self.niveles.get(NeurotransmitterType.DOPAMINE, 0.0)
        estres = (
            self.niveles.get(NeurotransmitterType.CORTISOL, 0.0)
            + self.niveles.get(NeurotransmitterType.ADRENALINE, 0.0)
        ) / 2.0
        concentracion = self.niveles.get(NeurotransmitterType.ACETYLCHOLINE, 0.0)
        calma = self.niveles.get(NeurotransmitterType.GABA, 0.0)
        alerta = self.niveles.get(NeurotransmitterType.NOREPINEPHRINE, 0.0)
        balance_general = (
            bienestar + motivacion + calma + concentracion
        ) / 4.0 - estres * 0.5
        return {
            "bienestar": float(bienestar),
            "motivacion": float(motivacion),
            "estres": float(estres),
            "concentracion": float(concentracion),
            "calma": float(calma),
            "alerta": float(alerta),
            "balance_general": float(balance_general),
        }

    def _calculate_system_coherence(self) -> float:
        deviations = []
        for n in list(self.niveles.keys()):
            base = self.niveles_base.get(n, 1e-6)
            deviations.append(
                abs(self.niveles.get(n, 0.0) - base) / max(abs(base), 1e-6)
            )
        avg_deviation = sum(deviations) / max(1, len(deviations))
        return max(0.0, 1.0 - avg_deviation)

    def _generate_balance_recommendations(self) -> list[str]:
        recommendations = []
        estado = self._calculate_hormonal_state()
        if estado["estres"] > 0.7:
            recommendations.append("Implementar técnicas de manejo del estrés")
            recommendations.append("Considerar períodos de descanso regulares")
        if estado["bienestar"] < 0.4:
            recommendations.append("Aumentar actividades que generen bienestar")
            recommendations.append("Mejorar rutinas de autocuidado")
        if estado["motivacion"] < 0.3:
            recommendations.append("Establecer metas pequeñas y alcanzables")
            recommendations.append("Buscar actividades estimulantes y recompensantes")
        if estado["concentracion"] < 0.4:
            recommendations.append("Minimizar distracciones en el entorno")
            recommendations.append("Implementar ejercicios de atención plena")
        if estado["calma"] < 0.4:
            recommendations.append("Practicar técnicas de relajación")
            recommendations.append("Crear espacios de tranquilidad")
        if not recommendations:
            recommendations.append("Sistema hormonal en balance saludable")
        return recommendations

    def _update_history(
        self, niveles_previos: dict[NeurotransmitterType, float]
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "niveles": {k.value: float(v) for k, v in self.niveles.items()},
            "cambios": {
                k.value: float(self.niveles[k] - niveles_previos.get(k, 0.0))
                for k in self.niveles.keys()
            },
            "estado_general": self._calculate_hormonal_state(),
        }
        self.historial_niveles.append(entry)
        if len(self.historial_niveles) > 200:
            self.historial_niveles = self.historial_niveles[-200:]

    def _update_history_snapshot(self) -> None:
        # lightweight snapshot used by tick
        entry = {
            "timestamp": time.time(),
            "niveles": {k.value: float(v) for k, v in self.niveles.items()},
        }
        self.historial_niveles.append(entry)
        if len(self.historial_niveles) > 200:
            self.historial_niveles = self.historial_niveles[-200:]

    # ------------------ Public utilities ------------------
    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "niveles": {k.value: float(v) for k, v in self.niveles.items()},
                "estado_hormonal": self._calculate_hormonal_state(),
                "coherencia": float(self._calculate_system_coherence()),
                "modulacion_externa": copy.deepcopy(self.modulacion_externa),
                "timestamp": time.time(),
            }

    def set_external_modulation(self, factor: str, valor: float) -> None:
        with self._lock:
            if factor in self.modulacion_externa:
                self.modulacion_externa[factor] = float(
                    max(-1.0, min(1.0, float(valor)))
                )

    def get_neurotransmitter_trends(self, window_size: int = 10) -> dict[str, str]:
        with self._lock:
            if len(self.historial_niveles) < window_size:
                return {nt.value: "insufficient_data" for nt in NeurotransmitterType}
            trends: dict[str, str] = {}
            recent_history = self.historial_niveles[-window_size:]
            for neurotransmisor in NeurotransmitterType:
                valores = [
                    entry.get("niveles", {}).get(neurotransmisor.value, 0.0)
                    for entry in recent_history
                ]
                half = max(1, len(valores) // 2)
                first_half = sum(valores[:half]) / half
                second_half = sum(valores[half:]) / max(1, len(valores) - half)
                diff = second_half - first_half
                if abs(diff) < 0.02:
                    trends[neurotransmisor.value] = "stable"
                elif diff > 0.02:
                    trends[neurotransmisor.value] = "increasing"
                else:
                    trends[neurotransmisor.value] = "decreasing"
            return trends

    def calibrate_baselines(self, factor_map: dict[str, float]) -> None:
        """
        Adjust baseline target levels (niveles_base). Accepts mapping of enum.name or enum.value to factor.
        """
        with self._lock:
            for key, factor in factor_map.items():
                for nt in NeurotransmitterType:
                    if key == nt.name or key == nt.value:
                        self.niveles_base[nt] = _clamp01(float(factor))
                        break

    def reset(self) -> None:
        with self._lock:
            for nt in list(self.niveles.keys()):
                self.niveles[nt] = self.niveles_base.get(nt, 0.5)
            self.historial_niveles.clear()
            self.modulacion_externa = dict.fromkeys(self.modulacion_externa.keys(), 0.0)

    # ------------------ Serialization / EVA helpers ------------------
    def to_serializable(self) -> dict[str, Any]:
        with self._lock:
            return {
                "niveles": {k.value: float(v) for k, v in self.niveles.items()},
                "niveles_base": {
                    k.value: float(v) for k, v in self.niveles_base.items()
                },
                "synthesis_rates": {
                    k.value: float(v) for k, v in self.synthesis_rates.items()
                },
                "degradation_rates": {
                    k.value: float(v) for k, v in self.degradation_rates.items()
                },
                "historial_niveles_len": len(self.historial_niveles),
                "modulacion_externa": copy.deepcopy(self.modulacion_externa),
                "last_update": float(self.last_update),
            }

    def load_serializable(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        with self._lock:
            try:
                for k, v in data.get("niveles", {}).items():
                    # map by enum value string
                    for nt in NeurotransmitterType:
                        if nt.value == k:
                            self.niveles[nt] = _clamp01(float(v))
                for k, v in data.get("niveles_base", {}).items():
                    for nt in NeurotransmitterType:
                        if nt.value == k:
                            self.niveles_base[nt] = _clamp01(float(v))
                self.modulacion_externa.update(data.get("modulacion_externa", {}))
                self.last_update = float(data.get("last_update", time.time()))
            except Exception:
                logger.exception("load_serializable failed", exc_info=True)

    async def _record_eva_async(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Async helper for EVA recording: best-effort, logs on failure.
        """
        if not self.eva_manager:
            return
        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                return
            res = recorder(entity_id=self.entity_id, event_type=event_type, data=data)
            if hasattr(res, "__await__"):
                try:
                    await res
                except Exception:
                    logger.exception("Async EVA recorder failed for %s", event_type)
        except Exception:
            logger.exception("Failed to record EVA experience (async helper)")

    # ------------------ Feedback & qualia metrics (kept and hardened) ------------------
    def _calculate_qualia_feedback(self) -> dict[str, float]:
        estado = self._calculate_hormonal_state()
        return {
            "consciousness_density_feedback": estado["concentracion"] * 0.8
            + estado["calma"] * 0.2,
            "temporal_coherence_feedback": estado["calma"] * 0.6
            + estado["balance_general"] * 0.4,
            "emotional_valence_feedback": estado["bienestar"] * 0.7
            - estado["estres"] * 0.5,
            "arousal_feedback": estado["alerta"] * 0.6 + estado["motivacion"] * 0.4,
            "cognitive_clarity_feedback": estado["concentracion"] * 0.9
            - estado["estres"] * 0.3,
        }

    def _calculate_mind_body_coherence(self, qualia_state: QualiaState) -> float:
        qualia_feedback = self._calculate_qualia_feedback()
        coherences = []
        for qualia_aspect, hormonal_equivalent in [
            ("consciousness_density", "consciousness_density_feedback"),
            ("temporal_coherence", "temporal_coherence_feedback"),
            ("emotional_valence", "emotional_valence_feedback"),
            ("arousal", "arousal_feedback"),
        ]:
            try:
                if isinstance(qualia_state, dict):
                    qv = qualia_state.get(qualia_aspect, None)
                else:
                    qv = getattr(qualia_state, qualia_aspect, None)
                if qv is None:
                    continue
                qualia_value = float(qv)
                qualia_norm = (
                    (qualia_value + 1.0) / 2.0 if qualia_value < -0.1 else qualia_value
                )
                coherence = 1.0 - abs(
                    qualia_norm - qualia_feedback.get(hormonal_equivalent, 0.0)
                )
                coherences.append(_clamp01(float(coherence)))
            except Exception:
                continue
        return float(sum(coherences) / len(coherences)) if coherences else 0.5
