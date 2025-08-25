from typing import Any
import math

class QualiaState:
    """Lightweight QualiaState that accepts multiple legacy keyword names.

    This class keeps a small compatibility layer so callers can pass
    older/alternate names like `emotional_valence`, `cognitive_complexity`,
    `consciousness_density`, `narrative_importance` and `energy_level`.
    """

    emotional: float
    complexity: float
    temporal: float
    consciousness: float
    importance: float
    energy: float

    def __init__(
        self,
        emotional: float = 0.0,
        complexity: float = 0.0,
        temporal: float = 0.0,
        consciousness: float = 0.0,
        importance: float = 0.0,
        energy: float = 0.0,
        **kwargs,
    ) -> None:
        # Map legacy/alternate kw names if provided in kwargs
        # canonical (new) attributes
        self.emotional = kwargs.get("emotional_valence", kwargs.get("emotional", emotional))
        self.complexity = kwargs.get("cognitive_complexity", kwargs.get("complexity", complexity))
        self.temporal = kwargs.get("temporal_coherence", kwargs.get("temporal", temporal))
        # 'consciousness_density' used elsewhere; prefer that name if provided
        self.consciousness = kwargs.get("consciousness_density", kwargs.get("consciousness", consciousness))
        self.importance = kwargs.get("narrative_importance", kwargs.get("importance", importance))
        self.energy = kwargs.get("energy_level", kwargs.get("energy", energy))

        # legacy/expanded attributes: preserve for compatibility with older modules
        self.arousal = kwargs.get("arousal", kwargs.get("emotional_arousal", 0.0))
        # richer dimensions commonly used in legacy QualiaState
        self.spiritual_resonance = kwargs.get("spiritual_resonance", kwargs.get("spiritual", 0.0))
        self.sensory_clarity = kwargs.get("sensory_clarity", 0.0)
        self.cognitive_focus = kwargs.get("cognitive_focus", 0.0)
        self.temporal_flow = kwargs.get("temporal_flow", 0.0)
        self.emotional_arousal = kwargs.get("emotional_arousal", self.arousal)
        self.dominant_word = kwargs.get("dominant_word", "")
        self.color = kwargs.get("color", [1.0, 1.0, 1.0])
        # optional metadata containers used by legacy
        self.history = kwargs.get("history", [])
        self.tags = kwargs.get("tags", [])

    def to_dict(self) -> dict[str, float]:
        # Return both canonical keys and legacy-named aliases for compatibility
        return {
            # canonical
            "emotional": float(self.emotional),
            "complexity": float(self.complexity),
            "temporal": float(self.temporal),
            "consciousness": float(self.consciousness),
            "importance": float(self.importance),
            "energy": float(self.energy),
            # legacy aliases
            "emotional_valence": float(getattr(self, "emotional", 0.0)),
            "arousal": float(getattr(self, "arousal", 0.0)),
            "cognitive_complexity": float(getattr(self, "complexity", 0.0)),
            "consciousness_density": float(getattr(self, "consciousness", 0.0)),
            "narrative_importance": float(getattr(self, "importance", 0.0)),
            "energy_level": float(getattr(self, "energy", 0.0)),
        }

    def get_state(self) -> dict[str, Any]:
        """Serializes the current state of QualiaState."""
        return {
            "emotional_valence": self.emotional_valence,
            "arousal": self.arousal,
            "cognitive_complexity": self.complexity,
            "consciousness_density": self.consciousness,
            "temporal_coherence": self.temporal,
            "spiritual_resonance": self.spiritual_resonance,
            "sensory_clarity": self.sensory_clarity,
            "cognitive_focus": self.cognitive_focus,
            "temporal_flow": self.temporal_flow,
            "emotional_arousal": self.emotional_arousal,
            "dominant_word": self.dominant_word,
            "color": self.color,
            "tags": self.tags.copy(),
        }

    def set_state(self, state: dict[str, Any]):
        """Deserializes and applies the state of QualiaState."""
        self.emotional_valence = state.get("emotional_valence", 0.0)
        self.arousal = state.get("arousal", 0.5)
        self.cognitive_complexity = state.get("cognitive_complexity", 0.3)
        self.consciousness_density = state.get("consciousness_density", 0.4)
        self.temporal = state.get("temporal_coherence", 1.0)
        self.spiritual_resonance = state.get("spiritual_resonance", 0.5)
        self.sensory_clarity = state.get("sensory_clarity", 0.5)
        self.cognitive_focus = state.get("cognitive_focus", 0.5)
        self.temporal_flow = state.get("temporal_flow", 0.5)
        self.emotional_arousal = state.get("emotional_arousal", 0.0)
        self.dominant_word = state.get("dominant_word", "")
        self.color = state.get("color", [1.0, 1.0, 1.0])
        self.tags = state.get("tags", [])

    def as_vector(self) -> list[float]:
        return [
            self.emotional_valence,
            self.arousal,
            self.cognitive_complexity,
            self.consciousness_density,
            self.temporal,
            self.spiritual_resonance,
            self.sensory_clarity,
            self.cognitive_focus,
            self.temporal_flow,
            self.emotional_arousal,
            self.color[0] if self.color else 0.0,
            self.color[1] if self.color else 0.0,
            self.color[2] if self.color else 0.0,
            0.0,  # Placeholder for future use
            0.0,  # Placeholder for future use
            0.0,  # Placeholder for future use
        ]

    @classmethod
    def from_vector(cls, vector: list[float]) -> "QualiaState":
        return cls(
            emotional_valence=vector[0],
            arousal=vector[1],
            cognitive_complexity=vector[2],
            consciousness_density=vector[3],
            temporal_coherence=vector[4],
            spiritual_resonance=vector[5],
            sensory_clarity=vector[6],
            cognitive_focus=vector[7],
            temporal_flow=vector[8],
            emotional_arousal=vector[9],
            color=[vector[10], vector[11], vector[12]],
            # dominant_word and tags are not part of the vector, so they will be default
        )

    def clamp_values(self):
        """Limita los valores a rangos seguros y fisiológicos."""
        self.emotional_valence = max(-1.0, min(1.0, self.emotional_valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.cognitive_complexity = max(0.0, min(1.0, self.cognitive_complexity))
        self.consciousness_density = max(0.1, min(1.0, self.consciousness_density))
        self.temporal = max(0.0, min(1.0, self.temporal))
        self.spiritual_resonance = max(0.0, min(1.0, self.spiritual_resonance))
        self.sensory_clarity = max(0.0, min(1.0, self.sensory_clarity))
        self.cognitive_focus = max(0.0, min(1.0, self.cognitive_focus))
        self.temporal_flow = max(0.0, min(1.0, self.temporal_flow))
        self.emotional_arousal = max(0.0, min(1.0, self.emotional_arousal))
        self.color = [max(0.0, min(1.0, c)) for c in self.color]

    @classmethod
    def neutral(cls) -> "QualiaState":
        """Crea un estado neutral de QualiaState."""
        return cls(
            emotional_valence=0.0,
            arousal=0.5,
            cognitive_complexity=0.5,
            consciousness_density=0.5,
            temporal_coherence=0.5,
            spiritual_resonance=0.5,
            sensory_clarity=0.5,
            cognitive_focus=0.5,
            temporal_flow=0.5,
            emotional_arousal=0.0,
            dominant_word="",
            color=[1.0, 1.0, 1.0],
        )

    def add_tag(self, tag: str) -> None:
        """Añade una etiqueta única al estado de qualia."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Elimina una etiqueta del estado de qualia."""
        if tag in self.tags:
            self.tags.remove(tag)

    def record_history(self, reason: str = "") -> None:
        """Registra el estado actual en el historial."""
        record = self.get_state()
        record["reason"] = reason
        self.history.append(record)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def calculate_harmony(self) -> float:
        """Calcula la armonía global del estado de qualia (0-1)."""
        vals = [
            abs(self.emotional_valence),
            self.arousal,
            self.cognitive_complexity,
            self.consciousness_density,
            self.temporal,
            self.spiritual_resonance,
            self.sensory_clarity,
            self.cognitive_focus,
            self.temporal_flow,
            self.emotional_arousal,
        ]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        harmony = 1.0 - math.sqrt(variance)
        return max(0.0, min(1.0, harmony))

    def calculate_complexity(self) -> float:
        """Calcula la complejidad subjetiva del estado de qualia."""
        vals = [
            self.cognitive_complexity,
            self.consciousness_density,
            self.temporal,
            self.spiritual_resonance,
            self.sensory_clarity,
        ]
        return sum(vals) / len(vals)

    def get_dominant_archetype(self) -> str:
        """Devuelve el arquetipo dominante si está en tags."""
        for tag in self.tags:
            if tag.startswith("archetype:"):
                return tag.split(":", 1)[1]
        return "unknown"

    def _capture_current_state(self) -> dict[str, float]:
        """Captura la firma actual de qualia para resonancia."""
        return {
            "emotional_valence": self.emotional_valence,
            "arousal": self.arousal,
            "cognitive_complexity": self.cognitive_complexity,
            "consciousness_density": self.consciousness_density,
            "temporal_coherence": self.temporal,
            "spiritual_resonance": self.spiritual_resonance,
            "sensory_clarity": self.sensory_clarity,
            "cognitive_focus": self.cognitive_focus,
            "temporal_flow": self.temporal_flow,
            "emotional_arousal": self.emotional_arousal,
        }

    def __str__(self) -> str:
        return (
            f"QualiaState(valence={self.emotional_valence:.2f}, arousal={self.arousal:.2f}, "
            f"complexity={self.cognitive_complexity:.2f}, density={self.consciousness_density:.2f}, "
            f"coherence={self.temporal:.2f}, resonance={self.spiritual_resonance:.2f})"
        )

    def __repr__(self) -> str:
        return f"<QualiaState: {self.get_state()}>"
