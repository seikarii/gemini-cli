"""
Defines the `MalkuthNode`, a cognitive node representing the "Kingdom" or physical reality.

This module contains the implementation of the Malkuth Sephirot, the lowest
point of the tree, which processes perceptions to generate impulses related
to physical manifestation and anchoring to existence.

MalkuthNode - Nodo Cognitivo de Manifestación Física y Realidad (Malkuth, Reino).

Procesa percepciones para generar impulsos de concreción, anclaje y materialización.
Incluye historial de concreción, anclaje y patrones de realidad para diagnóstico y simulación avanzada.
"""

from typing import Any, cast, TYPE_CHECKING

from crisalida_lib.ADAM.config import AdamConfig
from crisalida_lib.ADAM.mente.cognitive_impulses import CognitiveImpulse, ImpulseType
from crisalida_lib.EDEN.living_symbol import MovementPattern
from crisalida_lib.EDEN.qualia_manifold import QualiaField

if TYPE_CHECKING:
    from crisalida_lib.EVA.divine_sigils import DivineSignature
else:
    DivineSignature = Any

from .cosmic_node import CosmicNode


class MalkuthNode(CosmicNode):
    """
    Malkuth (Reino) - Nodo Cognitivo de Manifestación Física y Realidad.
    Procesa percepciones para generar impulsos de concreción y anclaje a la existencia.
    """

    def __init__(
        self,
        entity_id: str,
        manifold: QualiaField,
        initial_position: tuple[float, float, float],
        config: AdamConfig = AdamConfig(),
        movement_pattern: MovementPattern = MovementPattern.STATIC,
        node_name: str = "malkuth_reality",
        mass: float = 1.0,
        influence_radius: float = 3.0,
    ):
        malkuth_divine_signature = None
        try:
            if DivineSignature is not None:
                malkuth_divine_signature = DivineSignature(glyph="מ")
        except Exception:
            malkuth_divine_signature = None

        try:
            super().__init__(
                entity_id=entity_id,
                manifold=cast(Any, manifold),
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                config=config,
                movement_pattern=movement_pattern,
                divine_signature=malkuth_divine_signature,
                mass=mass,
                influence_radius=influence_radius,
            )
        except TypeError:
            super().__init__(
                entity_id=entity_id,
                manifold=manifold,
                initial_position=initial_position,
                node_type="sephirot",
                node_name=node_name,
                config=config,
                movement_pattern=movement_pattern,
            )

        self.divine_signature = malkuth_divine_signature
        self.activation_threshold: float = 0.12
        self.memory_patterns: list[dict[str, Any]] = []
        self.concretion_history: list[dict[str, Any]] = []
        self.anchoring_history: list[dict[str, Any]] = []
        self.materialization_patterns: list[dict[str, Any]] = []
        self.reality_patterns: list[dict[str, Any]] = []

    def analyze(self, perception: dict[str, Any] | None = None) -> list[CognitiveImpulse]:
        # Get local qualia perception from the manifold (data param kept for
        # compatibility with CognitiveNode.analyze signature).
        perception = perception if perception is not None else self.perceive_local_qualia()
        if isinstance(perception, dict) and "resonance" in perception:
            resonance = perception["resonance"]
        else:
            resonance = {"coherence": 1.0, "intensity": 1.0}

        # Expose normalized fields for older code
        if isinstance(perception, dict):
            perception.setdefault("coherence", resonance.get("coherence", 1.0))
            perception.setdefault("intensity", resonance.get("intensity", 1.0))
        impulses: list[CognitiveImpulse] = []
        activation = self._calculate_activation_level(perception)
        if activation < self.activation_threshold:
            return impulses

        self._update_memory_patterns(perception)

        # Impulso de Concreción de la Realidad
        concretion_content = f"Realidad manifestándose. Claridad sensorial: {perception.get('sensory_clarity', 0.0):.2f}"
        concretion_intensity = activation * perception.get("sensory_clarity", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.LOGICAL_STRUCTURE,
                content=concretion_content,
                intensity=concretion_intensity,
                confidence=0.7,
                source_node=self.node_name,
            )
        )
        self.concretion_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "sensory_clarity": perception.get("sensory_clarity", 0.0),
                "intensity": concretion_intensity,
            }
        )

        # Impulso de Anclaje a la Existencia
        anchoring_content = f"Anclaje a la existencia. Foco cognitivo: {perception.get('cognitive_focus', 0.0):.2f}"
        anchoring_intensity = activation * perception.get("cognitive_focus", 0.0)
        impulses.append(
            CognitiveImpulse(
                impulse_type=ImpulseType.CAUSAL_INFERENCE,
                content=anchoring_content,
                intensity=anchoring_intensity,
                confidence=0.65,
                source_node=self.node_name,
            )
        )
        self.anchoring_history.append(
            {
                "timestamp": perception.get("timestamp"),
                "cognitive_focus": perception.get("cognitive_focus", 0.0),
                "intensity": anchoring_intensity,
            }
        )

        # Impulso de Materialización (si alta densidad de conciencia)
        if perception.get("consciousness_density", 0.5) > 0.7:
            materialization_content = f"Materialización física activada. Densidad de conciencia: {perception.get('consciousness_density', 0.0):.2f}"
            materialization_intensity = activation * perception.get(
                "consciousness_density", 0.0
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.STRUCTURE_ENHANCEMENT,
                    content=materialization_content,
                    intensity=materialization_intensity,
                    confidence=0.8,
                    source_node=self.node_name,
                )
            )
            self.materialization_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "consciousness_density": perception.get(
                        "consciousness_density", 0.0
                    ),
                    "intensity": materialization_intensity,
                }
            )

        # Impulso de Patrón de Realidad (si hay coherencia y claridad altas)
        if (
            perception.get("sensory_clarity", 0.0) > 0.7
            and perception.get("coherence", 0.0) > 0.7
        ):
            reality_content = f"Patrón de realidad consolidado. Coherencia: {perception.get('coherence', 0.0):.2f}, Claridad: {perception.get('sensory_clarity', 0.0):.2f}"
            reality_intensity = (
                activation
                * (
                    perception.get("coherence", 0.0)
                    + perception.get("sensory_clarity", 0.0)
                )
                / 2
            )
            impulses.append(
                CognitiveImpulse(
                    impulse_type=ImpulseType.PATTERN_RECOGNITION,
                    content=reality_content,
                    intensity=reality_intensity,
                    confidence=0.75,
                    source_node=self.node_name,
                )
            )
            self.reality_patterns.append(
                {
                    "timestamp": perception.get("timestamp"),
                    "coherence": perception.get("coherence", 0.0),
                    "sensory_clarity": perception.get("sensory_clarity", 0.0),
                    "intensity": reality_intensity,
                }
            )

        return impulses

    def _calculate_activation_level(self, perception: dict[str, Any]) -> float:
        """
        Calcula el nivel de activación del nodo Malkuth según la percepción.
        """
        sensory_clarity = perception.get("sensory_clarity", 0.5)
        cognitive_focus = perception.get("cognitive_focus", 0.5)
        consciousness_density = perception.get("consciousness_density", 0.5)
        coherence = perception.get("coherence", 0.5)
        # Activación ponderada por claridad sensorial, foco, densidad y coherencia
        activation = (
            sensory_clarity * 0.3
            + cognitive_focus * 0.25
            + consciousness_density * 0.25
            + coherence * 0.2
        )
        return max(0.0, min(1.0, activation))

    def _update_memory_patterns(self, perception: dict[str, Any]) -> None:
        """
        Actualiza patrones de memoria interna según la percepción recibida.
        """
        self.memory_patterns.append(perception)
        if len(self.memory_patterns) > 80:
            self.memory_patterns = self.memory_patterns[
                -80:
            ]  # Mantener solo los últimos 80 patrones
