"""
Behavioral Genome - FULL IMPLEMENTATION (professionalized)

- Removes legacy typing unions to use typing module forms for compatibility.
- Fixes missing state (eva_manager, entity_id) and duplicate methods.
- Adds robust EVA recording (sync/async-aware), add_new_action_type helper,
  safer import/validation paths and small API improvements for evolution/simulation.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any

from crisalida_lib.ADAM.enums import (
    ActionCategory,
)  # [`crisalida_lib.ADAM.enums.ActionCategory`](crisalida_lib/ADAM/enums.py)
from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    EVAMemoryManager,
)  # [`crisalida_lib.ADAM.eva_integration.eva_memory_manager.EVAMemoryManager`](crisalida_lib/ADAM/eva_integration/eva_memory_manager.py)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ActionDescriptor:
    name: str
    category: ActionCategory
    description: str
    energy_cost: float = 0.1
    prerequisites: set[str] = field(default_factory=set)
    awakening_level: int = 0
    complexity: float = 0.5
    success_probability_base: float = 0.8
    cooldown_cycles: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    last_used: float | None = None
    usage_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.energy_cost <= 1.0:
            raise ValueError("energy_cost must be between 0.0 and 1.0")
        if not 0.0 <= self.complexity <= 1.0:
            raise ValueError("complexity must be between 0.0 and 1.0")
        if not 0.0 <= self.success_probability_base <= 1.0:
            raise ValueError("success_probability_base must be between 0.0 and 1.0")
        if self.awakening_level not in (0, 1, 2, 3):
            raise ValueError("awakening_level must be 0, 1, 2, or 3")


class ActionRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._actions: dict[str, ActionDescriptor] = {}
        self._categories: dict[ActionCategory, set[str]] = {
            cat: set() for cat in ActionCategory
        }
        self._action_history: list[dict[str, Any]] = []
        self._locked_actions: set[str] = set()
        self._initialize_core_actions()

    def _initialize_core_actions(self) -> None:
        core_actions: list[ActionDescriptor] = [
            ActionDescriptor(
                "observe_environment",
                ActionCategory.PHYSICAL,
                "Observar el entorno circundante",
                energy_cost=0.05,
                success_probability_base=0.95,
                metadata={"core_action": True},
            ),
            ActionDescriptor(
                "think",
                ActionCategory.MENTAL,
                "Procesar información y generar pensamientos",
                energy_cost=0.08,
                success_probability_base=0.9,
            ),
            ActionDescriptor(
                "remember",
                ActionCategory.MENTAL,
                "Acceder a memorias almacenadas",
                energy_cost=0.06,
                success_probability_base=0.85,
            ),
            ActionDescriptor(
                "communicate_basic",
                ActionCategory.COMMUNICATION,
                "Comunicación básica con otros seres",
                energy_cost=0.1,
                success_probability_base=0.8,
            ),
            ActionDescriptor(
                "survive",
                ActionCategory.SURVIVAL,
                "Respuesta instintiva de supervivencia",
                energy_cost=0.15,
                metadata={"core_action": True, "instinctive": True},
                success_probability_base=0.8,
            ),
            ActionDescriptor(
                "explore",
                ActionCategory.PHYSICAL,
                "Explorar nuevas áreas o conceptos",
                energy_cost=0.12,
                prerequisites={"observe_environment"},
                success_probability_base=0.75,
            ),
            ActionDescriptor(
                "meditate",
                ActionCategory.SPIRITUAL,
                "Práctica de meditación y autoconocimiento",
                energy_cost=0.2,
                awakening_level=1,
                success_probability_base=0.7,
                metadata={"core_action": True, "awakening_related": True},
            ),
            ActionDescriptor(
                "create_simple",
                ActionCategory.CREATIVE,
                "Creación simple de objetos o conceptos",
                energy_cost=0.18,
                prerequisites={"think"},
                success_probability_base=0.65,
            ),
            ActionDescriptor(
                "lucid_dream",
                ActionCategory.SPIRITUAL,
                "Control consciente de los sueños en el Metacosmos",
                energy_cost=0.3,
                prerequisites={"meditate"},
                success_probability_base=0.6,
                cooldown_cycles=5,
                metadata={"core_action": True, "awakening_ability": True},
            ),
            ActionDescriptor(
                "genetic_self_modification",
                ActionCategory.MANIPULATION,
                "Auto-modificación genética voluntaria",
                energy_cost=0.5,
                awakening_level=2,
                prerequisites={"lucid_dream", "meditate"},
                complexity=0.9,
                success_probability_base=0.4,
                cooldown_cycles=10,
                metadata={"core_action": True, "high_level_ability": True},
            ),
            ActionDescriptor(
                "autonomous_decision",
                ActionCategory.MENTAL,
                "Toma de decisiones completamente autónomas",
                energy_cost=0.25,
                awakening_level=3,
                prerequisites={"genetic_self_modification"},
                complexity=0.8,
                metadata={"core_action": True, "ultimate_ability": True},
            ),
            ActionDescriptor(
                "no_action",
                ActionCategory.PHYSICAL,
                "Decisión consciente de no actuar",
                energy_cost=0.02,
                success_probability_base=1.0,
                metadata={"core_action": True, "special": True},
            ),
            ActionDescriptor(
                "diagnose_self",
                ActionCategory.DIAGNOSTIC,
                "Auto-diagnóstico de estado interno",
                energy_cost=0.07,
                success_probability_base=0.95,
                metadata={"meta": True},
            ),
            ActionDescriptor(
                "meta_reflect",
                ActionCategory.META,
                "Reflexión meta-cognitiva sobre acciones y estados",
                energy_cost=0.09,
                success_probability_base=0.9,
                metadata={"meta": True},
            ),
            ActionDescriptor(
                "emotional_regulation",
                ActionCategory.EMOTIONAL,
                "Regulación consciente de estados emocionales",
                energy_cost=0.12,
                success_probability_base=0.8,
                metadata={"meta": True},
            ),
        ]
        for action in core_actions:
            self.register_action(action, lock=True)

    def register_action(self, action: ActionDescriptor, lock: bool = False) -> bool:
        with self._lock:
            if action.name in self._locked_actions:
                logger.warning("Cannot register locked action: %s", action.name)
                return False
            for prereq in action.prerequisites:
                if prereq not in self._actions:
                    logger.warning(
                        "Prerequisite '%s' not found for action '%s'",
                        prereq,
                        action.name,
                    )
                    return False
            self._actions[action.name] = action
            self._categories[action.category].add(action.name)
            if lock:
                self._locked_actions.add(action.name)
            self._action_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operation": "register",
                    "action_name": action.name,
                    "category": action.category.value,
                    "locked": lock,
                }
            )
            logger.info("Action '%s' registered successfully", action.name)
            return True

    def unregister_action(self, action_name: str) -> bool:
        with self._lock:
            if action_name in self._locked_actions:
                logger.warning("Cannot unregister locked action: %s", action_name)
                return False
            if action_name not in self._actions:
                logger.warning("Action '%s' not found", action_name)
                return False
            dependents = self.get_dependent_actions(action_name)
            if dependents:
                logger.warning(
                    "Cannot remove '%s': dependencies exist in %s",
                    action_name,
                    dependents,
                )
                return False
            action = self._actions[action_name]
            del self._actions[action_name]
            self._categories[action.category].discard(action_name)
            self._action_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operation": "unregister",
                    "action_name": action_name,
                }
            )
            logger.info("Action '%s' unregistered successfully", action_name)
            return True

    def modify_action(self, action_name: str, **modifications: Any) -> bool:
        with self._lock:
            if action_name not in self._actions:
                logger.warning("Action '%s' not found", action_name)
                return False
            if action_name in self._locked_actions:
                logger.warning("Cannot modify locked action: %s", action_name)
                return False
            action = self._actions[action_name]
            old_values: dict[str, Any] = {}
            for attr, value in modifications.items():
                if hasattr(action, attr):
                    old_values[attr] = getattr(action, attr)
                    setattr(action, attr, value)
            self._action_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operation": "modify",
                    "action_name": action.name,
                    "modifications": modifications,
                    "old_values": old_values,
                }
            )
            logger.info("Action '%s' modified successfully", action.name)
            return True

    def get_action(self, action_name: str) -> ActionDescriptor | None:
        with self._lock:
            return self._actions.get(action_name)

    def get_available_actions(
        self,
        awakening_level: int = 0,
        category: ActionCategory | None = None,
        prerequisites_met: set[str] | None = None,
    ) -> list[str]:
        with self._lock:
            if prerequisites_met is None:
                prerequisites_met = set()
            available: list[str] = []
            for name, action in self._actions.items():
                if action.awakening_level > awakening_level:
                    continue
                if category is not None and action.category != category:
                    continue
                if not action.prerequisites.issubset(prerequisites_met):
                    continue
                available.append(name)
            return available

    def get_actions_by_category(self, category: ActionCategory) -> list[str]:
        with self._lock:
            return list(self._categories.get(category, set()))

    def get_dependent_actions(self, action_name: str) -> list[str]:
        with self._lock:
            return [
                name
                for name, action in self._actions.items()
                if action_name in action.prerequisites
            ]

    def get_all_action_names(self) -> list[str]:
        with self._lock:
            return list(self._actions.keys())

    def get_registry_stats(self) -> dict[str, Any]:
        with self._lock:
            actions = list(self._actions.values())
            total = len(actions)
            avg_energy = (sum(a.energy_cost for a in actions) / total) if total else 0.0
            avg_complexity = (
                (sum(a.complexity for a in actions) / total) if total else 0.0
            )
            return {
                "total_actions": total,
                "locked_actions": len(self._locked_actions),
                "actions_by_category": {
                    cat.value: len(actions) for cat, actions in self._categories.items()
                },
                "actions_by_awakening_level": {
                    level: len([a for a in actions if a.awakening_level == level])
                    for level in range(4)
                },
                "average_energy_cost": avg_energy,
                "average_complexity": avg_complexity,
            }

    def export_to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actions": {
                    name: {
                        "name": action.name,
                        "category": action.category.value,
                        "description": action.description,
                        "energy_cost": action.energy_cost,
                        "prerequisites": list(action.prerequisites),
                        "awakening_level": action.awakening_level,
                        "complexity": action.complexity,
                        "success_probability_base": action.success_probability_base,
                        "cooldown_cycles": action.cooldown_cycles,
                        "metadata": action.metadata,
                        "last_used": action.last_used,
                        "usage_count": action.usage_count,
                    }
                    for name, action in self._actions.items()
                },
                "locked_actions": list(self._locked_actions),
                "history": self._action_history[-100:],
            }

    def import_from_dict(self, data: dict[str, Any], merge: bool = True) -> bool:
        with self._lock:
            try:
                if not merge:
                    self._actions.clear()
                    self._categories = {category: set() for category in ActionCategory}
                    self._locked_actions.clear()
                    self._action_history.clear()
                if "actions" in data:
                    for name, action_data in data["actions"].items():
                        action = ActionDescriptor(
                            name=action_data["name"],
                            category=ActionCategory(action_data["category"]),
                            description=action_data["description"],
                            energy_cost=action_data.get("energy_cost", 0.1),
                            prerequisites=set(action_data.get("prerequisites", [])),
                            awakening_level=int(action_data.get("awakening_level", 0)),
                            complexity=float(action_data.get("complexity", 0.5)),
                            success_probability_base=float(
                                action_data.get("success_probability_base", 0.5)
                            ),
                            cooldown_cycles=int(action_data.get("cooldown_cycles", 0)),
                            metadata=action_data.get("metadata", {}),
                            last_used=action_data.get("last_used"),
                            usage_count=int(action_data.get("usage_count", 0)),
                        )
                        self.register_action(
                            action, lock=name in data.get("locked_actions", [])
                        )
                if "locked_actions" in data and not merge:
                    self._locked_actions = set(data["locked_actions"])
                if "history" in data and not merge:
                    self._action_history = data["history"]
                logger.info("ActionRegistry import completed successfully")
                return True
            except Exception as e:
                logger.error("Failed to import ActionRegistry data: %s", str(e))
                return False

    # convenience helper to add new actions programmatically
    def add_new_action_type(self, action: ActionDescriptor, lock: bool = False) -> bool:
        return self.register_action(action, lock=lock)


@dataclass
class InstinctProfile:
    base_instincts: dict[str, float] = field(
        default_factory=lambda: {
            "supervivencia": 0.9,
            "curiosidad": 0.8,
            "orden": 0.7,
            "sociabilidad": 0.6,
            "creatividad": 0.5,
            "dominancia": 0.4,
            "cautela": 0.6,
            "empathy": 0.5,
            "meta_reflexion": 0.4,
            "diagnostico": 0.5,
            "regulacion_emocional": 0.5,
        }
    )
    learned_modifiers: dict[str, float] = field(default_factory=dict)
    environmental_factors: dict[str, float] = field(default_factory=dict)

    def get_effective_instinct(
        self,
        instinct_name: str,
        environmental_context: dict[str, float] | None = None,
    ) -> float:
        base = self.base_instincts.get(instinct_name, 0.0)
        learned = self.learned_modifiers.get(instinct_name, 0.0)
        environmental = 0.0
        if environmental_context:
            for factor, strength in environmental_context.items():
                if factor in self.environmental_factors:
                    environmental += self.environmental_factors[factor] * float(
                        strength
                    )
        effective = base + learned + environmental * 0.1
        return max(0.0, min(1.0, effective))

    def evolve_instinct(self, instinct_name: str, experience_outcome: float) -> None:
        if instinct_name not in self.learned_modifiers:
            self.learned_modifiers[instinct_name] = 0.0
        change = float(experience_outcome) * 0.01
        self.learned_modifiers[instinct_name] = max(
            -0.5, min(0.5, self.learned_modifiers[instinct_name] + change)
        )


class GenomaComportamiento:
    def __init__(
        self,
        eva_manager: EVAMemoryManager | None,
        entity_id: str,
        action_registry: ActionRegistry | None = None,
    ) -> None:
        self._lock = RLock()
        self.eva_manager: EVAMemoryManager | None = eva_manager
        self.entity_id: str = entity_id
        self.action_registry: ActionRegistry = (
            action_registry if action_registry else ActionRegistry()
        )
        self.instinct_profile: InstinctProfile = InstinctProfile()
        self.instinct_action_mappings: dict[str, list[str]] = {
            "supervivencia": ["survive", "observe_environment", "no_action"],
            "curiosidad": ["explore", "observe_environment", "think"],
            "orden": ["think", "remember", "create_simple"],
            "sociabilidad": ["communicate_basic"],
            "creatividad": ["create_simple", "lucid_dream"],
            "dominancia": ["autonomous_decision", "genetic_self_modification"],
            "cautela": ["no_action", "observe_environment"],
            "empathy": ["communicate_basic", "meditate"],
            "meta_reflexion": ["meta_reflect", "diagnose_self"],
            "diagnostico": ["diagnose_self"],
            "regulacion_emocional": ["emotional_regulation"],
        }
        self.action_history: list[dict[str, Any]] = []
        self.success_patterns: dict[str, float] = {}

    @property
    def instintos(self) -> dict[str, float]:
        return self.instinct_profile.base_instincts.copy()

    def get_available_actions(
        self,
        awakening_level: int = 0,
        environmental_context: dict[str, Any] | None = None,
    ) -> list[str]:
        known_actions: set[str] = set(self.action_registry.get_all_action_names())
        if self.action_history:
            attempted_actions = {
                action["action_name"] for action in self.action_history
            }
            known_actions.update(attempted_actions)
        return self.action_registry.get_available_actions(
            awakening_level=awakening_level, prerequisites_met=known_actions
        )

    def get_instinct_driven_actions(
        self,
        awakening_level: int = 0,
        environmental_context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        action_scores: dict[str, float] = {}
        available_actions = self.get_available_actions(
            awakening_level=awakening_level, environmental_context=environmental_context
        )
        for instinct_name in self.instinct_profile.base_instincts.keys():
            effective_strength = self.instinct_profile.get_effective_instinct(
                instinct_name, environmental_context or {}
            )
            preferred_actions = self.instinct_action_mappings.get(instinct_name, [])
            for action_name in preferred_actions:
                if action_name in available_actions:
                    action_scores[action_name] = (
                        action_scores.get(action_name, 0.0) + effective_strength
                    )
        if action_scores:
            max_score = max(action_scores.values())
            if max_score > 0:
                for action in list(action_scores.keys()):
                    action_scores[action] = action_scores[action] / max_score
        return action_scores

    def select_action(
        self,
        awakening_level: int = 0,
        environmental_context: dict[str, Any] | None = None,
        randomness: float = 0.1,
    ) -> str | None:
        action_scores = self.get_instinct_driven_actions(
            awakening_level=awakening_level, environmental_context=environmental_context
        )
        if not action_scores:
            return None
        for action_name in action_scores:
            success_rate = self.success_patterns.get(action_name, 0.5)
            action_scores[action_name] *= 1.0 + success_rate * 0.2
        if random.random() < float(randomness):
            return random.choice(list(action_scores.keys()))
        else:
            return max(action_scores.items(), key=lambda x: x[1])[0]

    def register_action_result(
        self,
        action_name: str,
        success: bool,
        outcome_details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.action_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action_name": action_name,
                    "success": bool(success),
                    "details": outcome_details or {},
                }
            )
            if action_name not in self.success_patterns:
                self.success_patterns[action_name] = 0.5
            current_pattern = self.success_patterns[action_name]
            new_outcome = 1.0 if success else 0.0
            self.success_patterns[action_name] = float(
                current_pattern * 0.9 + new_outcome * 0.1
            )
            self._evolve_instincts_from_action(action_name, success, outcome_details)
            action = self.action_registry.get_action(action_name)
            if action:
                action.usage_count += 1
                action.last_used = datetime.now().timestamp()

    def _evolve_instincts_from_action(
        self, action_name: str, success: bool, outcome_details: dict[str, Any] | None
    ) -> None:
        experience_outcome = 0.5 if success else -0.3
        for instinct_name, actions in self.instinct_action_mappings.items():
            if action_name in actions:
                self.instinct_profile.evolve_instinct(instinct_name, experience_outcome)

    def get_genome_state(self) -> dict[str, Any]:
        return {
            "instinct_profile": {
                "base_instincts": self.instinct_profile.base_instincts.copy(),
                "learned_modifiers": self.instinct_profile.learned_modifiers.copy(),
                "environmental_factors": self.instinct_profile.environmental_factors.copy(),
            },
            "instinct_action_mappings": self.instinct_action_mappings.copy(),
            "action_history": self.action_history[-1000:],
            "success_patterns": self.success_patterns.copy(),
            "action_registry_stats": self.action_registry.get_registry_stats(),
        }

    def analyze_behavioral_patterns(self) -> dict[str, Any]:
        if len(self.action_history) < 5:
            return {
                "insufficient_data": True,
                "message": "Need at least 5 actions for analysis",
            }
        recent_actions = self.action_history[-20:]
        action_frequency: dict[str, int] = {}
        success_by_action: dict[str, list[bool]] = {}
        for record in recent_actions:
            action_name = record["action_name"]
            success = bool(record["success"])
            action_frequency[action_name] = action_frequency.get(action_name, 0) + 1
            success_by_action.setdefault(action_name, []).append(success)
        most_used_actions = sorted(
            action_frequency.items(), key=lambda x: x[1], reverse=True
        )[:5]
        action_success_rates = {
            action_name: (sum(successes) / len(successes))
            for action_name, successes in success_by_action.items()
        }
        instinct_trends = self._analyze_instinct_trends()
        behavioral_patterns = self._detect_behavioral_patterns(recent_actions)
        return {
            "analysis_period": len(recent_actions),
            "most_used_actions": most_used_actions,
            "action_success_rates": action_success_rates,
            "overall_success_rate": (
                (sum(1 for r in recent_actions if r["success"]) / len(recent_actions))
                if recent_actions
                else 0.0
            ),
            "instinct_trends": instinct_trends,
            "behavioral_patterns": behavioral_patterns,
            "recommendations": self._generate_behavioral_recommendations(
                action_success_rates, instinct_trends
            ),
        }

    def _analyze_instinct_trends(self) -> dict[str, Any]:
        trends: dict[str, Any] = {}
        for instinct_name, base_value in self.instinct_profile.base_instincts.items():
            learned_modifier = self.instinct_profile.learned_modifiers.get(
                instinct_name, 0.0
            )
            effective_value = base_value + learned_modifier
            if abs(learned_modifier) < 0.05:
                trend = "stable"
            elif learned_modifier > 0.05:
                trend = "strengthening"
            else:
                trend = "weakening"
            trends[instinct_name] = {
                "base_value": base_value,
                "learned_modifier": learned_modifier,
                "effective_value": effective_value,
                "trend": trend,
            }
        return trends

    def _detect_behavioral_patterns(
        self, recent_actions: list[dict[str, Any]]
    ) -> list[str]:
        patterns: list[str] = []
        action_sequence = [r["action_name"] for r in recent_actions]
        for pattern_length in (2, 3):
            pattern_counts: dict[tuple, int] = {}
            for i in range(len(action_sequence) - pattern_length + 1):
                pattern = tuple(action_sequence[i : i + pattern_length])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            for pattern, count in pattern_counts.items():
                if count >= 3:
                    patterns.append(
                        f"Repetitive sequence: {' -> '.join(pattern)} (occurs {count} times)"
                    )
        success_sequence = [r["success"] for r in recent_actions]
        current_streak = 0
        max_success_streak = 0
        max_failure_streak = 0
        current_failure_streak = 0
        for success in success_sequence:
            if success:
                current_streak += 1
                current_failure_streak = 0
                max_success_streak = max(max_success_streak, current_streak)
            else:
                current_failure_streak += 1
                current_streak = 0
                max_failure_streak = max(max_failure_streak, current_failure_streak)
        if max_success_streak >= 5:
            patterns.append(
                f"Strong success streak detected (max: {max_success_streak})"
            )
        if max_failure_streak >= 3:
            patterns.append(
                f"Concerning failure streak detected (max: {max_failure_streak})"
            )
        action_types_by_time: list[str] = []
        for record in recent_actions:
            action_name = record["action_name"]
            action_descriptor = self.action_registry.get_action(action_name)
            if action_descriptor:
                action_types_by_time.append(action_descriptor.category.value)
        if len(action_types_by_time) >= 2:
            first_half = action_types_by_time[: len(action_types_by_time) // 2]
            second_half = action_types_by_time[len(action_types_by_time) // 2 :]
            first_half_counts = {t: first_half.count(t) for t in set(first_half)}
            second_half_counts = {t: second_half.count(t) for t in set(second_half)}
            for action_type in set(first_half_counts.keys()) | set(
                second_half_counts.keys()
            ):
                first_freq = (
                    (first_half_counts.get(action_type, 0) / len(first_half))
                    if first_half
                    else 0.0
                )
                second_freq = (
                    (second_half_counts.get(action_type, 0) / len(second_half))
                    if second_half
                    else 0.0
                )
                if second_freq - first_freq > 0.2:
                    patterns.append(f"Increasing focus on {action_type} actions")
                elif first_freq - second_freq > 0.2:
                    patterns.append(f"Decreasing focus on {action_type} actions")
        return patterns

    def _generate_behavioral_recommendations(
        self, success_rates: dict[str, float], instinct_trends: dict[str, Any]
    ) -> list[str]:
        recommendations: list[str] = []
        if success_rates:
            best_action = max(success_rates.items(), key=lambda x: x[1])
            worst_action = min(success_rates.items(), key=lambda x: x[1])
            if best_action[1] > 0.8:
                recommendations.append(
                    f"Continue leveraging '{best_action[0]}' - high success rate ({best_action[1]:.1%})"
                )
            if worst_action[1] < 0.3:
                recommendations.append(
                    f"Consider avoiding '{worst_action[0]}' - low success rate ({worst_action[1]:.1%})"
                )
        strengthening_instincts = [
            name
            for name, data in instinct_trends.items()
            if data["trend"] == "strengthening"
        ]
        weakening_instincts = [
            name
            for name, data in instinct_trends.items()
            if data["trend"] == "weakening"
        ]
        if strengthening_instincts:
            recommendations.append(
                f"Emerging strengths in: {', '.join(strengthening_instincts[:3])}"
            )
        if weakening_instincts:
            recommendations.append(
                f"Consider reinforcing: {', '.join(weakening_instincts[:2])}"
            )
        action_diversity = len(success_rates.keys())
        if action_diversity < 5:
            recommendations.append(
                "Consider expanding behavioral repertoire - low action diversity"
            )
        return recommendations

    def evolve_genome(self, evolution_pressure: dict[str, float]) -> dict[str, Any]:
        with self._lock:
            evolution_report: dict[str, Any] = {
                "pre_evolution_state": self.get_genome_state(),
                "pressures_applied": evolution_pressure.copy(),
                "changes_made": [],
            }
            for pressure_type, intensity in evolution_pressure.items():
                if intensity < 0.1:
                    continue
                change = float(intensity) * 0.1
                if pressure_type == "survival_pressure":
                    self.instinct_profile.base_instincts["supervivencia"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("supervivencia", 0.5)
                        + change,
                    )
                    self.instinct_profile.base_instincts["cautela"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("cautela", 0.5)
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Strengthened survival instincts by {change:.3f}"
                    )
                elif pressure_type == "creativity_pressure":
                    self.instinct_profile.base_instincts["creatividad"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("creatividad", 0.5)
                        + change,
                    )
                    self.instinct_profile.base_instincts["curiosidad"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("curiosidad", 0.5)
                        + change * 0.5,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced creativity by {change:.3f}"
                    )
                elif pressure_type == "social_pressure":
                    self.instinct_profile.base_instincts["sociabilidad"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("sociabilidad", 0.5)
                        + change,
                    )
                    self.instinct_profile.base_instincts["empathy"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("empathy", 0.5)
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced social instincts by {change:.3f}"
                    )
                elif pressure_type == "order_pressure":
                    self.instinct_profile.base_instincts["orden"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("orden", 0.5) + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Strengthened order instinct by {change:.3f}"
                    )
                elif pressure_type == "autonomy_pressure":
                    self.instinct_profile.base_instincts["dominancia"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("dominancia", 0.5)
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced autonomy drives by {change:.3f}"
                    )
                elif pressure_type == "meta_pressure":
                    self.instinct_profile.base_instincts["meta_reflexion"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("meta_reflexion", 0.4)
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced meta-reflection by {change:.3f}"
                    )
                elif pressure_type == "diagnostic_pressure":
                    self.instinct_profile.base_instincts["diagnostico"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get("diagnostico", 0.5)
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced diagnostic instinct by {change:.3f}"
                    )
                elif pressure_type == "emotional_pressure":
                    self.instinct_profile.base_instincts["regulacion_emocional"] = min(
                        1.0,
                        self.instinct_profile.base_instincts.get(
                            "regulacion_emocional", 0.5
                        )
                        + change,
                    )
                    evolution_report["changes_made"].append(
                        f"Enhanced emotional regulation by {change:.3f}"
                    )
            evolution_report["post_evolution_state"] = self.get_genome_state()
            return evolution_report

    def simulate_genetic_modification(
        self, target_changes: dict[str, float], awakening_level: int = 2
    ) -> dict[str, Any]:
        if awakening_level < 2:
            return {
                "success": False,
                "error": "Genetic self-modification requires Segundo Despertar (level 2+)",
            }
        if (
            "genetic_self_modification"
            not in self.action_registry.get_available_actions(
                awakening_level=awakening_level
            )
        ):
            return {
                "success": False,
                "error": "genetic_self_modification action not available",
            }
        modification_success = random.random() < 0.4
        if not modification_success:
            self.register_action_result(
                "genetic_self_modification",
                False,
                {
                    "attempted_changes": target_changes,
                    "failure_reason": "Modification process failed",
                },
            )
            return {
                "success": False,
                "message": "Genetic modification attempt failed",
                "energy_cost": 0.5,
                "attempted_changes": target_changes,
            }
        changes_applied: dict[str, dict[str, float]] = {}
        for instinct_name, desired_change in target_changes.items():
            actual_change = max(-0.2, min(0.2, float(desired_change)))
            if instinct_name in self.instinct_profile.base_instincts:
                old_value = self.instinct_profile.base_instincts[instinct_name]
                new_value = max(0.0, min(1.0, old_value + actual_change))
                self.instinct_profile.base_instincts[instinct_name] = new_value
                changes_applied[instinct_name] = {
                    "old_value": old_value,
                    "change": actual_change,
                    "new_value": new_value,
                }
        self.register_action_result(
            "genetic_self_modification",
            True,
            {
                "changes_applied": changes_applied,
                "modification_type": "instinct_adjustment",
            },
        )
        return {
            "success": True,
            "message": "Genetic modification successful",
            "energy_cost": 0.5,
            "changes_applied": changes_applied,
            "cooldown_cycles": 10,
        }

    def export_genome_data(self) -> dict[str, Any]:
        return {
            "action_registry": self.action_registry.export_to_dict(),
            "instinct_profile": {
                "base_instincts": self.instinct_profile.base_instincts.copy(),
                "learned_modifiers": self.instinct_profile.learned_modifiers.copy(),
                "environmental_factors": self.instinct_profile.environmental_factors.copy(),
            },
            "instinct_action_mappings": self.instinct_action_mappings.copy(),
            "action_history": self.action_history[-1000:],
            "success_patterns": self.success_patterns.copy(),
            "export_timestamp": datetime.now().isoformat(),
        }

    def import_genome_data(self, data: dict[str, Any]) -> bool:
        try:
            if "action_registry" in data:
                self.action_registry.import_from_dict(
                    data["action_registry"], merge=False
                )
            if "instinct_profile" in data:
                profile_data: dict[str, Any] = data["instinct_profile"]
                self.instinct_profile.base_instincts = profile_data.get(
                    "base_instincts", {}
                )
                self.instinct_profile.learned_modifiers = profile_data.get(
                    "learned_modifiers", {}
                )
                self.instinct_profile.environmental_factors = profile_data.get(
                    "environmental_factors", {}
                )
            if "instinct_action_mappings" in data:
                self.instinct_action_mappings = data["instinct_action_mappings"]
            if "action_history" in data:
                self.action_history = data["action_history"]
            if "success_patterns" in data:
                self.success_patterns = data["success_patterns"]
            logger.info("Genome data imported successfully")
            return True
        except Exception as e:
            logger.error("Failed to import genome data: %s", str(e))
            return False

    # EVA helpers (best-effort, async/sync-aware)
    def record_genome_experience(
        self,
        experience_id: str | None = None,
        genome_state: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Records a genome experience using the EVAMemoryManager (best-effort).
        """
        if not self.eva_manager:
            logger.debug(
                "EVAMemoryManager not available, cannot record genome experience."
            )
            return None
        genome_state = genome_state or self.get_genome_state()
        experience_id = (
            experience_id
            or f"genome_experience_{abs(hash(str(genome_state))) & 0xFFFFFFFF}"
        )
        experience_data = {
            "genome_state": genome_state,
            "instinct_profile": genome_state.get("instinct_profile", {}),
            "action_registry_stats": genome_state.get("action_registry_stats", {}),
            "action_history": genome_state.get("action_history", []),
            "success_patterns": genome_state.get("success_patterns", {}),
            "timestamp": time.time(),
        }
        try:
            recorder = getattr(self.eva_manager, "record_experience", None)
            if not callable(recorder):
                logger.debug("EVAMemoryManager.record_experience not callable")
                return None
            res = recorder(
                entity_id=self.entity_id,
                event_type="genome_experience",
                data=experience_data,
                experience_id=experience_id,
            )
            if hasattr(res, "__await__"):
                try:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(res)
                    else:
                        loop.run_until_complete(res)
                except Exception:
                    logger.debug(
                        "Failed to schedule async EVA genome record", exc_info=True
                    )
            logger.info("Recorded genome experience: %s", experience_id)
            return experience_id
        except Exception:
            logger.exception("Failed to record genome experience")
            return None

    def recall_genome_experience(self, experience_id: str) -> dict[str, Any] | None:
        if not self.eva_manager:
            logger.debug(
                "EVAMemoryManager not available, cannot recall genome experience."
            )
            return None
        try:
            recall = getattr(self.eva_manager, "recall_experience", None)
            if not callable(recall):
                logger.debug("EVAMemoryManager.recall_experience not callable")
                return None
            res = recall(entity_id=self.entity_id, experience_id=experience_id)
            # If coroutine, best-effort scheduling; prefer synchronous return where possible
            if hasattr(res, "__await__"):
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(res)
                    logger.debug(
                        "Scheduled async recall; returning None until available"
                    )
                    return None
                else:
                    return loop.run_until_complete(res)
            return res
        except Exception:
            logger.exception("Failed to recall genome experience")
            return None


def create_default_genome(
    eva_manager: EVAMemoryManager | None, entity_id: str
) -> GenomaComportamiento:
    return GenomaComportamiento(eva_manager=eva_manager, entity_id=entity_id)


def create_enhanced_genome(
    eva_manager: EVAMemoryManager | None,
    entity_id: str,
    additional_actions: list[ActionDescriptor] | None = None,
) -> GenomaComportamiento:
    genome = GenomaComportamiento(eva_manager=eva_manager, entity_id=entity_id)
    if additional_actions:
        for action in additional_actions:
            genome.action_registry.add_new_action_type(action)
    return genome


def simulate_genome_evolution(
    genome: GenomaComportamiento,
    cycles: int = 100,
    environmental_pressures: dict[str, float] | None = None,
) -> dict[str, Any]:
    if environmental_pressures is None:
        environmental_pressures = {
            "survival_pressure": 0.3,
            "creativity_pressure": 0.2,
            "social_pressure": 0.1,
            "meta_pressure": 0.1,
            "diagnostic_pressure": 0.1,
            "emotional_pressure": 0.1,
        }
    initial_state = genome.get_genome_state()
    evolution_history: list[dict[str, Any]] = []
    for cycle in range(cycles):
        selected_action = genome.select_action(
            environmental_context=environmental_pressures,
            awakening_level=0,
            randomness=0.2,
        )
        if selected_action:
            action_descriptor = genome.action_registry.get_action(selected_action)
            success_probability = 0.5
            if action_descriptor is not None:
                success_probability = action_descriptor.success_probability_base
            if (
                selected_action in ("survive", "observe_environment")
                and environmental_pressures.get("survival_pressure", 0) > 0.5
            ):
                success_probability = min(1.0, success_probability + 0.2)
            success = random.random() < success_probability
            genome.register_action_result(selected_action, success)
            evolution_history.append(
                {
                    "cycle": cycle,
                    "action": selected_action,
                    "success": success,
                    "success_probability": success_probability,
                }
            )
        if cycle % 20 == 0 and cycle > 0:
            genome.evolve_genome(environmental_pressures)
    final_state = genome.get_genome_state()
    return {
        "initial_state": initial_state,
        "final_state": final_state,
        "evolution_history": evolution_history,
        "cycles_completed": cycles,
        "behavioral_analysis": genome.analyze_behavioral_patterns(),
        "performance_metrics": {
            "overall_success_rate": (
                (
                    sum(1 for h in evolution_history if h["success"])
                    / len(evolution_history)
                )
                if evolution_history
                else 0.0
            ),
            "action_diversity": len({h["action"] for h in evolution_history}),
            "most_used_action": (
                max(
                    {h["action"] for h in evolution_history},
                    key=lambda x: sum(
                        1 for h_rec in evolution_history if h_rec["action"] == x
                    ),
                    default=None,
                )
                if evolution_history
                else None
            ),
        },
    }
