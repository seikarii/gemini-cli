from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentionMatrix:
    """Lightweight contract representing an intention matrix.

    This is intentionally minimal to serve as a testable contract and
    integration point for compilers and living symbols.
    """

    matrix_id: str
    name: str
    # content may store a matrix under 'matrix' key or other structured payloads
    content: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        # Minimal validation: must have a non-empty matrix_id and name
        return bool(self.matrix_id and self.name)

    def to_payload(self) -> dict[str, Any]:
        """Return a normalized payload shape compatible with EDEN bytecode adapter.

        The adapter expects keys like 'intention_type', 'experience' and 'payload'.
        """
        matrix = self.content.get("matrix") or self.metadata.get("matrix") or []
        return {
            "intention_type": self.metadata.get("intention_type", self.name),
            "experience": self.metadata.get("experience", {}),
            "payload": matrix,
        }

    def is_empty(self) -> bool:
        m = self.content.get("matrix") or self.metadata.get("matrix") or []
        return not any(bool(r) for r in (m or []))
