"""
Linguistic Engine (v3.1 definitive)

Edits summary:
- Hardened imports (numpy/EVA types/LivingSymbolRuntime) with graceful fallbacks.
- Replaced print() with logging and added structured debug/info messages.
- Made EVA integration async-aware and defensive (supports sync/coroutine EVAManager/divine_compiler).
- Added export/import EVA-memory JSON helpers, and safer statistics & merging logic.
- Added lightweight runtime QualiaState fallback when EVA types are not present.
- Kept public API stable: LinguisticEngine, SharedLexicon, EVALinguisticEngine.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Defensive numeric backend pattern (prefer numpy when available)
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False

# TYPE_CHECKING guarded imports to avoid circular runtime deps
if TYPE_CHECKING:
    from crisalida_lib.EDEN.living_symbol import (
        LivingSymbolRuntime,
    )  # [`crisalida_lib/EDEN/living_symbol.py`](crisalida_lib/EDEN/living_symbol.py)
    from crisalida_lib.EVA.language.sigils import (
        DimensionalAffinity,
        OntologicalCategory,
    )
    from crisalida_lib.EVA.language.sigils import (
        DimensionalAffinity,
        OntologicalCategory,
    )
    from crisalida_lib.EVA.types import (
        EVAExperience,
        QualiaState,
        RealityBytecode,
    )
else:
    # Runtime-safe imports with fallbacks if modules are missing at execution time.
    try:
        from crisalida_lib.EDEN.living_symbol import LivingSymbolRuntime  # type: ignore
    except Exception:
        LivingSymbolRuntime = None  # type: ignore

    try:
        from crisalida_lib.EVA.types import (  # type: ignore
            DimensionalAffinity,
            EVAExperience,
            OntologicalCategory,
            QualiaState,
            RealityBytecode,
        )
    except Exception:
        # Minimal runtime fallback for QualiaState and RealityBytecode used by this module.
        @dataclass
        class QualiaState:
            emotional_valence: float = 0.0
            arousal: float = 0.0
            cognitive_complexity: float = 0.0
            consciousness_density: float = 0.0
            narrative_importance: float = 0.0
            energy_level: float = 1.0
            # optional fields used by this module
            emotional_arousal: float = 0.0
            dominant_word: str = ""
            elemental_signature: dict[str, float] = field(default_factory=dict)

            def get_state(self) -> dict[str, float]:
                return {
                    "valence": float(self.emotional_valence),
                    "arousal": float(self.arousal),
                    "clarity": float(self.cognitive_complexity),
                    "density": float(self.consciousness_density),
                    "importance": float(self.narrative_importance),
                    "energy": float(self.energy_level),
                }

            def to_dict(self) -> dict[str, Any]:
                return self.get_state()

            @staticmethod
            def neutral() -> QualiaState:
                return QualiaState()

        @dataclass
        class RealityBytecode:
            bytecode_id: str
            instructions: list[Any] = field(default_factory=list)
            qualia_state: QualiaState = field(default_factory=QualiaState)
            phase: str = "default"
            timestamp: float = field(default_factory=time.time)

        EVAExperience = dict  # permissive fallback
        OntologicalCategory = Any  # type: ignore
        DimensionalAffinity = Any  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Data structures ---


@dataclass
class PhoneticWord:
    word: str
    phonemes: str
    qualia_vector: dict[str, float]
    meaning: str
    usage_count: int = 0
    last_used: float = 0.0
    creation_ts: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    origin_entity: str = ""


# --- Helpers ---


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _euclidean_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if HAS_NUMPY:
        arr = np.array(
            [float(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys], dtype=float
        )
        return float(np.linalg.norm(arr))
    # fallback
    s = 0.0
    for k in keys:
        diff = float(a.get(k, 0.0) - b.get(k, 0.0))
        s += diff * diff
    return float(s**0.5)


def _now_ts() -> float:
    return time.time()


# --- Core engine ---


class LinguisticEngine:
    """
    Motor de lenguaje emergente, fonético y meta-cognitivo para ADAM.

    Public API highlights:
      - create_new_word(qualia_state, novelty, tags)
      - find_word_for_qualia(qualia_state)
      - interpret_message(message, context)
      - evolve_vocabulary()
      - export/import pattern memory via JSON helpers
    """

    def __init__(
        self,
        entity_id: str = "adam_entity",
        shared_lexicon: SharedLexicon | None = None,
    ) -> None:
        self.entity_id = str(entity_id)
        self.vocabulary: dict[str, PhoneticWord] = {}
        self.word_history: list[tuple[str, float]] = []
        self.clusters: dict[str, list[str]] = {}
        self.meta_tags: dict[str, list[str]] = {}
        self.similarity_threshold: float = 0.1
        self.novelty_threshold: float = 0.8
        self.personal_vocabulary: dict[str, PhoneticWord] = {}
        self.comprehension_history: dict[str, list[bool]] = defaultdict(list)
        self.shared_lexicon = shared_lexicon
        self.phoneme_base = [
            "ka",
            "ta",
            "na",
            "ra",
            "sa",
            "ha",
            "ma",
            "ya",
            "wa",
            "la",
            "ki",
            "ti",
            "ni",
            "ri",
            "si",
            "hi",
            "mi",
            "yi",
            "wi",
            "li",
            "ku",
            "tu",
            "nu",
            "ru",
            "su",
            "hu",
            "mu",
            "yu",
            "wu",
            "lu",
            "ke",
            "te",
            "ne",
            "re",
            "se",
            "he",
            "me",
            "ye",
            "we",
            "le",
            "ko",
            "to",
            "no",
            "ro",
            "so",
            "ho",
            "mo",
            "yo",
            "wo",
            "lo",
        ]
        logger.info("LinguisticEngine initialized for entity=%s", self.entity_id)

    # ------------------ vocabulary creation & search ------------------

    def create_new_word(
        self, qualia_state: Any, novelty: float, tags: list[str] | None = None
    ) -> PhoneticWord | None:
        """Genera un neologismo si la experiencia es suficientemente novedosa."""
        try:
            if novelty < float(self.novelty_threshold):
                return None

            # obtain a stable hash from qualia_state
            state_hash = self._hash_qualia_state_safe(qualia_state)
            seed = (
                int(state_hash[:8], 16) if state_hash else int(_now_ts()) & 0xFFFFFFFF
            )
            rnd = random.Random(seed)

            word_length = max(
                2,
                min(6, int(getattr(qualia_state, "cognitive_complexity", 0.5) * 4) + 2),
            )
            phonemes = []
            valence = float(getattr(qualia_state, "emotional_valence", 0.0))
            for _ in range(word_length):
                if valence < -0.3:
                    preferred = ("k", "t", "r", "s")
                elif valence > 0.3:
                    preferred = ("m", "n", "l", "w", "y")
                else:
                    preferred = ("n", "r", "s", "h")
                suitable = [p for p in self.phoneme_base if p[0] in preferred]
                if not suitable:
                    suitable = self.phoneme_base
                phonemes.append(rnd.choice(suitable))

            new_word_str = "".join(phonemes)
            if new_word_str in self.vocabulary:
                logger.debug("create_new_word: collision, skipping '%s'", new_word_str)
                return None

            qualia_vec = self._safe_get_state_dict(qualia_state)
            new_word = PhoneticWord(
                word=new_word_str,
                phonemes=f"/{new_word_str}/",
                qualia_vector=dict(qualia_vec),
                meaning=f"valence={qualia_vec.get('valence', 0.0):.2f}, arousal={qualia_vec.get('arousal', 0.0):.2f}",
                usage_count=0,
                last_used=0.0,
                creation_ts=_now_ts(),
                tags=list(tags or []),
                success_rate=0.0,
                origin_entity=self.entity_id,
            )

            # register
            self.vocabulary[new_word_str] = new_word
            self.personal_vocabulary[new_word_str] = new_word
            self.word_history.append((new_word_str, new_word.creation_ts))
            for t in new_word.tags:
                self.meta_tags.setdefault(t, []).append(new_word_str)
            self._update_clusters(new_word)

            logger.info(
                "New word created: %s (entity=%s)", new_word_str, self.entity_id
            )
            if self.shared_lexicon:
                try:
                    self.shared_lexicon.add_word(new_word_str, new_word)
                except Exception:
                    logger.debug("shared_lexicon.add_word failed", exc_info=True)
            return new_word
        except Exception:
            logger.exception("create_new_word failed")
            return None

    def find_word_for_qualia(
        self, qualia_state: Any, update_usage: bool = True
    ) -> PhoneticWord | None:
        """Busca la palabra más cercana en el vocabulario para un QualiaState dado."""
        if not self.vocabulary:
            return None
        q_vec = self._safe_get_state_dict(qualia_state)
        best: PhoneticWord | None = None
        best_dist = float("inf")
        for w in self.vocabulary.values():
            try:
                dist = _euclidean_distance(w.qualia_vector, q_vec)
                if dist < best_dist:
                    best_dist = dist
                    best = w
            except Exception:
                logger.debug("distance calc failed for word=%s", w.word, exc_info=True)
        if best and best_dist < float(self.similarity_threshold):
            if update_usage:
                best.usage_count += 1
                best.last_used = _now_ts()
            return best
        return None

    def get_words_by_tag(self, tag: str) -> list[PhoneticWord]:
        """Retorna todas las palabras asociadas a un tag/meta-tag."""
        return [
            self.vocabulary[w]
            for w in self.meta_tags.get(tag, [])
            if w in self.vocabulary
        ]

    def cluster_words(self, cluster_key: str, qualia_states: list[Any]) -> None:
        """Agrupa palabras en clusters semánticos según una lista de QualiaStates."""
        cluster: list[str] = []
        for qs in qualia_states:
            word = self.find_word_for_qualia(qs, update_usage=False)
            if word:
                cluster.append(word.word)
        self.clusters[cluster_key] = cluster

    def _update_clusters(self, word: PhoneticWord) -> None:
        """Actualiza clusters internos según tags y similitud."""
        for tag in word.tags:
            self.clusters.setdefault(tag, []).append(word.word)

    # ------------------ evolution & housekeeping ------------------

    def evolve_vocabulary(self) -> None:
        """Simula evolución del vocabulario: elimina palabras poco usadas y fusiona similares."""
        now = _now_ts()
        to_remove: list[str] = []
        for key, word in list(self.vocabulary.items()):
            if word.usage_count == 0 and (now - word.creation_ts) > 3600:
                to_remove.append(key)
        for k in to_remove:
            self.vocabulary.pop(k, None)
            self.personal_vocabulary.pop(k, None)
            logger.info("Word '%s' removed due to inactivity.", k)

        # Fusionar palabras similares (cheap O(n^2) pass)
        words = list(self.vocabulary.values())
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                try:
                    dist = _euclidean_distance(
                        words[i].qualia_vector, words[j].qualia_vector
                    )
                    if dist < 0.05:
                        merged = list(set(words[i].tags + words[j].tags))
                        words[i].tags = merged
                        words[j].tags = merged
                except Exception:
                    logger.debug("fusion compare failed", exc_info=True)

    def get_vocabulary_stats(self) -> dict[str, Any]:
        """Retorna estadísticas del vocabulario y uso."""
        most_used = sorted(
            self.vocabulary.values(), key=lambda w: w.usage_count, reverse=True
        )[:5]
        return {
            "total_words": len(self.vocabulary),
            "active_clusters": len(self.clusters),
            "meta_tags": list(self.meta_tags.keys()),
            "most_used": most_used,
        }

    def get_word_history(self, last_n: int = 10) -> list[tuple[str, float]]:
        return self.word_history[-int(last_n) :]

    # ------------------ interpretation ------------------

    def interpret_message(
        self, message: str, context: QualiaState | None = None
    ) -> tuple[QualiaState | None, float]:
        """Intentar interpretar un mensaje de texto en un QualiaState agregado."""
        words = message.split()
        interpreted_states: list[QualiaState] = []
        confidence_scores: list[float] = []
        ctx = context or (
            QualiaState.neutral() if hasattr(QualiaState, "neutral") else None
        )
        for w in words:
            s, c = self._interpret_single_word(w, ctx)
            if s:
                interpreted_states.append(s)
                confidence_scores.append(c)
        if not interpreted_states:
            return None, 0.0
        combined = self._combine_qualia_states(interpreted_states)
        overall_conf = (
            float(sum(confidence_scores)) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )
        self.comprehension_history[message].append(overall_conf > 0.5)
        return combined, overall_conf

    def _interpret_single_word(
        self, word: str, context: QualiaState | None
    ) -> tuple[QualiaState | None, float]:
        """Interpreta una palabra individual con fallback strategies."""
        try:
            if word in self.personal_vocabulary:
                pw = self.personal_vocabulary[word]
                return self._qualia_from_vector(pw.qualia_vector), 0.9

            if self.shared_lexicon and hasattr(self.shared_lexicon, "get_definition"):
                shared = self.shared_lexicon.get_definition(word)
                if shared:
                    conf = min(
                        0.8,
                        float(shared.success_rate) * 0.8
                        + min(shared.usage_count / 100.0, 0.2),
                    )
                    return self._qualia_from_vector(shared.qualia_vector), float(conf)

            similar = self._find_phonetically_similar_word(word)
            if similar and similar in self.personal_vocabulary:
                base = self.personal_vocabulary[similar].qualia_vector
                q = self._qualia_from_vector(
                    {
                        "valence": base.get("valence", 0.0),
                        "arousal": base.get("arousal", 0.0),
                        "clarity": base.get("clarity", 0.0),
                    }
                )
                return self._create_semantic_variation(q), 0.3
        except Exception:
            logger.debug("interpret single word failed for '%s'", word, exc_info=True)
        return None, 0.0

    # ------------------ EVA helpers & integration ------------------

    async def _maybe_record_eva_event(self, payload: dict[str, Any]) -> None:
        """Best-effort record to EVA; supports coroutine or sync record_experience."""
        eva_mgr = getattr(self, "eva_manager", None)
        if not eva_mgr:
            return
        rec_fn = getattr(eva_mgr, "record_experience", None)
        if not callable(rec_fn):
            return
        event = {
            "entity_id": self.entity_id,
            "event_type": "linguistic_event",
            "data": payload,
            "timestamp": _now_ts(),
        }
        try:
            if inspect.iscoroutinefunction(rec_fn):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(rec_fn(**event))
                except RuntimeError:
                    await rec_fn(**event)
            else:
                # sync call
                rec_fn(**event)
        except Exception:
            logger.exception("EVA record_experience failed (non-fatal)")

    def export_memory_json(self) -> str:
        """Serialize vocabulary & EVA cache (best-effort)"""
        try:
            serializable = {
                k: getattr(v, "__dict__", v) for k, v in self.vocabulary.items()
            }
            return json.dumps(serializable, default=str)
        except Exception:
            logger.exception("export_memory_json failed")
            return "{}"

    def import_memory_json(self, payload: str) -> bool:
        """Import vocabulary JSON in permissive manner"""
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                return False
            for k, v in data.items():
                try:
                    pw = PhoneticWord(
                        word=k,
                        phonemes=v.get("phonemes", f"/{k}/"),
                        qualia_vector=v.get("qualia_vector", {}),
                        meaning=v.get("meaning", ""),
                        usage_count=int(v.get("usage_count", 0)),
                        last_used=float(v.get("last_used", 0.0)),
                        creation_ts=float(v.get("creation_ts", _now_ts())),
                        tags=list(v.get("tags", [])),
                        success_rate=float(v.get("success_rate", 0.0)),
                        origin_entity=str(v.get("origin_entity", self.entity_id)),
                    )
                    self.vocabulary[k] = pw
                    self.personal_vocabulary[k] = pw
                except Exception:
                    logger.debug("import entry failed for key=%s", k, exc_info=True)
            return True
        except Exception:
            logger.exception("import_memory_json failed")
            return False

    # ------------------ internal utilities ------------------

    def _safe_get_state_dict(self, q: Any) -> dict[str, float]:
        """Return a plain dict representing qualia; tolerant to multiple APIs."""
        try:
            if hasattr(q, "get_state"):
                st = q.get_state()
                if isinstance(st, dict):
                    return {k: float(v) for k, v in st.items()}
            # attribute-based fallback
            return {
                "valence": float(getattr(q, "emotional_valence", 0.0) or 0.0),
                "arousal": float(getattr(q, "arousal", 0.0) or 0.0),
                "clarity": float(getattr(q, "cognitive_complexity", 0.0) or 0.0),
                "density": float(getattr(q, "consciousness_density", 0.0) or 0.0),
                "importance": float(getattr(q, "narrative_importance", 0.0) or 0.0),
                "energy": float(getattr(q, "energy_level", 1.0) or 1.0),
            }
        except Exception:
            logger.debug("QualiaState read failed; returning defaults", exc_info=True)
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "clarity": 0.0,
                "density": 0.0,
                "importance": 0.0,
                "energy": 1.0,
            }

    def _hash_qualia_state_safe(self, state: Any) -> str:
        try:
            sdict = self._safe_get_state_dict(state)
            state_str = json.dumps(sdict, sort_keys=True, default=str)
            return hashlib.md5(state_str.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.md5(str(time.time()).encode()).hexdigest()

    def _qualia_from_vector(self, vect: dict[str, float]) -> QualiaState:
        """Construct a QualiaState instance (fallback-friendly)."""
        try:
            return QualiaState(
                emotional_valence=float(vect.get("valence", 0.0)),
                arousal=float(vect.get("arousal", 0.0)),
                cognitive_complexity=float(vect.get("clarity", 0.0)),
                consciousness_density=float(vect.get("density", 0.0)),
                narrative_importance=float(vect.get("importance", 0.0)),
                energy_level=float(vect.get("energy", 1.0)),
            )
        except Exception:
            # If QualiaState constructor signature differs, attempt attribute-based creation
            q = (
                QualiaState.neutral()
                if hasattr(QualiaState, "neutral")
                else QualiaState()
            )
            for k, v in vect.items():
                try:
                    setattr(q, k if hasattr(q, k) else k, v)
                except Exception:
                    continue
            return q

    def _average_qualia_attribute(
        self, states: list[QualiaState], attribute: str
    ) -> float:
        vals = [getattr(s, attribute, 0.0) for s in states if s is not None]
        return float(sum(vals) / max(1, len(vals)))

    def _combine_qualia_states(self, states: list[QualiaState]) -> QualiaState | None:
        if not states:
            return None
        avg_valence = self._average_qualia_attribute(states, "emotional_valence")
        avg_arousal = self._average_qualia_attribute(states, "arousal")
        avg_complexity = self._average_qualia_attribute(states, "cognitive_complexity")
        combined_signature: dict[str, float] = defaultdict(float)
        for s in states:
            if hasattr(s, "elemental_signature"):
                for k, v in getattr(s, "elemental_signature", {}).items():
                    combined_signature[k] += float(v) / len(states)
        dominant_word = ""
        words_present = [
            getattr(s, "dominant_word", "")
            for s in states
            if getattr(s, "dominant_word", "")
        ]
        if words_present:
            dominant_word = random.choice(words_present)
        q = self._qualia_from_vector(
            {
                "valence": avg_valence,
                "arousal": avg_arousal,
                "clarity": avg_complexity,
                "density": float(getattr(states[0], "consciousness_density", 0.0)),
                "importance": float(getattr(states[0], "narrative_importance", 0.0)),
                "energy": float(getattr(states[0], "energy_level", 1.0)),
            }
        )
        try:
            q.elemental_signature = dict(combined_signature)
            q.dominant_word = dominant_word
        except Exception:
            pass
        return q

    def _find_phonetically_similar_word(self, word: str) -> str | None:
        best_match: str | None = None
        best_score = 0.0
        for known in self.personal_vocabulary.keys():
            sim = self._phonetic_similarity(word, known)
            if sim > best_score and sim > 0.6:
                best_score = sim
                best_match = known
        return best_match

    def _phonetic_similarity(self, w1: str, w2: str) -> float:
        if not w1 or not w2:
            return 0.0
        len_sim = 1.0 - abs(len(w1) - len(w2)) / max(len(w1), len(w2))
        set1, set2 = set(w1), set(w2)
        char_sim = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
        return float((len_sim + char_sim) / 2.0)

    def _create_semantic_variation(self, base_state: QualiaState) -> QualiaState:
        variation = 0.2
        try:
            return self._qualia_from_vector(
                {
                    "valence": max(
                        -1.0,
                        min(
                            1.0,
                            getattr(base_state, "emotional_valence", 0.0)
                            + random.uniform(-variation, variation),
                        ),
                    ),
                    "arousal": max(
                        0.0,
                        getattr(base_state, "arousal", 0.0)
                        + random.uniform(-variation, variation),
                    ),
                    "clarity": max(
                        0.0,
                        getattr(base_state, "cognitive_complexity", 0.0)
                        + random.uniform(-variation, variation),
                    ),
                    "energy": max(
                        0.0,
                        getattr(base_state, "energy_level", 1.0)
                        + random.uniform(-variation, variation),
                    ),
                }
            )
        except Exception:
            logger.debug("_create_semantic_variation failed", exc_info=True)
            return base_state

    # ------------------ end class LinguisticEngine ------------------


# --- SharedLexicon ---


class SharedLexicon:
    """Léxico compartido para toda una realidad UnifiedField"""

    def __init__(self) -> None:
        self.words: dict[str, PhoneticWord] = {}
        self.word_relationships: dict[str, list[str]] = defaultdict(list)
        logger.info("SharedLexicon initialized")

    def add_word(self, word: str, definition: PhoneticWord) -> None:
        """Añade/actualiza una nueva palabra en el léxico compartido"""
        if word not in self.words:
            self.words[word] = definition
            logger.debug("SharedLexicon added word=%s", word)
        else:
            existing = self.words[word]
            # merge lightweight stats
            existing.usage_count += int(definition.usage_count or 0)
            existing.success_rate = (
                (existing.success_rate * existing.usage_count) + definition.success_rate
            ) / max(1, existing.usage_count + 1)

    def reinforce_word(self, word: str, success: bool) -> None:
        if word not in self.words:
            return
        definition = self.words[word]
        prev_count = max(0, definition.usage_count)
        definition.success_rate = (
            (definition.success_rate * prev_count) + (1.0 if success else 0.0)
        ) / max(1, prev_count + 1)
        definition.usage_count = prev_count + 1
        logger.debug(
            "SharedLexicon reinforce: %s success=%s new_rate=%.3f",
            word,
            success,
            definition.success_rate,
        )

    def get_definition(self, word: str) -> PhoneticWord | None:
        return self.words.get(word)

    def get_popular_words(self, threshold: int = 5) -> list[str]:
        return [w for w, d in self.words.items() if d.usage_count >= int(threshold)]


# --- EVA-aware engine extension ---


class EVALinguisticEngine(LinguisticEngine):
    """
    Motor lingüístico extendido para integración con EVA.
    """

    def __init__(
        self,
        entity_id: str = "eva_entity",
        shared_lexicon: SharedLexicon | None = None,
        phase: str = "default",
    ) -> None:
        super().__init__(entity_id=entity_id, shared_lexicon=shared_lexicon)
        self.eva_phase: str = phase
        self.eva_runtime = (
            LivingSymbolRuntime() if LivingSymbolRuntime is not None else None
        )
        self.eva_manager: Any | None = None
        self.eva_memory_store: dict[str, RealityBytecode] = {}
        self.eva_experience_store: dict[str, EVAExperience] = {}
        self.eva_phases: dict[str, dict[str, RealityBytecode]] = {}
        self._environment_hooks: list[Callable[..., Any]] = []
        logger.info("EVALinguisticEngine initialized (phase=%s)", self.eva_phase)

    async def eva_ingest_linguistic_experience(
        self,
        word_event: dict[str, Any],
        qualia_state: QualiaState | None = None,
        phase: str | None = None,
    ) -> str:
        """Compile & persist a linguistic experience into EVA (async-aware, best-effort)."""
        phase = phase or self.eva_phase
        qualia_state = qualia_state or (
            QualiaState.neutral() if hasattr(QualiaState, "neutral") else QualiaState()
        )
        experience_id = (
            word_event.get("experience_id")
            or f"eva_linguistic_{hash(json.dumps(word_event, default=str))}_{int(_now_ts())}"
        )
        experience_data = {
            "word_event": dict(word_event),
            "phonemes": word_event.get("phonemes"),
            "qualia_vector": word_event.get("qualia_vector"),
            "meaning": word_event.get("meaning"),
            "usage_count": word_event.get("usage_count", 0),
            "tags": word_event.get("tags", []),
            "timestamp": _now_ts(),
            "phase": phase,
        }

        # Prefer EVAManager if attached
        if self.eva_manager and hasattr(self.eva_manager, "record_experience"):
            try:
                record_fn = self.eva_manager.record_experience
                if inspect.iscoroutinefunction(record_fn):
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        record_fn(
                            entity_id=self.entity_id,
                            event_type="linguistic_experience",
                            data=experience_data,
                            qualia_state=qualia_state,
                        )
                    )
                else:
                    record_fn(
                        entity_id=self.entity_id,
                        event_type="linguistic_experience",
                        data=experience_data,
                        qualia_state=qualia_state,
                    )
                return experience_id
            except Exception:
                logger.exception(
                    "eva_manager.record_experience failed; falling back to local runtime"
                )

        # Fallback: use local runtime divine_compiler if available
        try:
            compiler = (
                getattr(self.eva_runtime, "divine_compiler", None)
                if self.eva_runtime
                else None
            )
            if compiler and hasattr(compiler, "compile_intention"):
                intention = {
                    "intention_type": "ARCHIVE_LINGUISTIC_EXPERIENCE",
                    "experience": experience_data,
                    "qualia": qualia_state,
                    "phase": phase,
                }
                if inspect.iscoroutinefunction(compiler.compile_intention):
                    try:
                        bytecode = await compiler.compile_intention(intention)
                    except RuntimeError:
                        bytecode = asyncio.run(compiler.compile_intention(intention))
                else:
                    bytecode = compiler.compile_intention(intention)
                rb = RealityBytecode(
                    bytecode_id=experience_id,
                    instructions=bytecode or [],
                    qualia_state=qualia_state,
                    phase=phase,
                    timestamp=experience_data["timestamp"],
                )
                self.eva_memory_store[experience_id] = rb
                self.eva_experience_store[experience_id] = {
                    "experience_id": rb.bytecode_id,
                    "bytecode": rb,
                }  # permissive
                self.eva_phases.setdefault(phase, {})[experience_id] = rb
                for hook in list(self._environment_hooks):
                    try:
                        hook(rb)
                    except Exception:
                        logger.debug(
                            "environment hook failed for ingestion", exc_info=True
                        )
                return experience_id
        except Exception:
            logger.exception("local compile/store failed for EVA ingestion (non-fatal)")

        # last-resort: store minimal record
        try:
            rb = RealityBytecode(
                bytecode_id=experience_id,
                instructions=[],
                qualia_state=qualia_state,
                phase=phase,
                timestamp=experience_data["timestamp"],
            )
            self.eva_memory_store[experience_id] = rb
            self.eva_experience_store[experience_id] = {
                "experience_id": rb.bytecode_id,
                "bytecode": rb,
            }
            self.eva_phases.setdefault(phase, {})[experience_id] = rb
            return experience_id
        except Exception:
            logger.exception("failed to create minimal RealityBytecode")
            return ""

    async def eva_recall_linguistic_experience(
        self, cue: str, phase: str | None = None
    ) -> dict[str, Any]:
        """Execute stored RealityBytecode (best-effort)."""
        phase = phase or self.eva_phase
        rb = self.eva_phases.get(phase, {}).get(cue) or self.eva_memory_store.get(cue)
        if not rb:
            return {"error": "No bytecode found for EVA linguistic experience"}
        manifestations = []
        quantum_field = (
            getattr(self.eva_runtime, "quantum_field", None)
            if self.eva_runtime
            else None
        )
        if quantum_field and hasattr(self.eva_runtime, "execute_instruction"):
            for instr in getattr(rb, "instructions", []) or []:
                try:
                    manifest = self.eva_runtime.execute_instruction(
                        instr, quantum_field
                    )
                    if manifest:
                        manifestations.append(manifest)
                        for hook in list(self._environment_hooks):
                            try:
                                hook(manifest)
                            except Exception:
                                logger.debug("manifest hook failed", exc_info=True)
                except Exception:
                    logger.exception("instruction execution failed (non-fatal)")
        # build permissive EVAExperience-like structure
        return {
            "experience_id": getattr(rb, "bytecode_id", cue),
            "manifestations": [
                getattr(m, "to_dict", lambda: m)() for m in manifestations
            ],
            "phase": getattr(rb, "phase", phase),
            "qualia_state": (
                getattr(rb, "qualia_state", {}).to_dict()
                if hasattr(getattr(rb, "qualia_state", {}), "to_dict")
                else {}
            ),
            "timestamp": getattr(rb, "timestamp", _now_ts()),
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        word_event: dict[str, Any],
        qualia_state: QualiaState,
    ) -> None:
        """Add an alternative phase for an existing experience (sync)."""
        try:
            compiler = (
                getattr(self.eva_runtime, "divine_compiler", None)
                if self.eva_runtime
                else None
            )
            if not compiler or not hasattr(compiler, "compile_intention"):
                raise AssertionError("divine_compiler not initialized")
            intention = {
                "intention_type": "ARCHIVE_LINGUISTIC_EXPERIENCE",
                "experience": dict(word_event),
                "qualia": qualia_state,
                "phase": phase,
            }
            if inspect.iscoroutinefunction(compiler.compile_intention):
                try:
                    bytecode = asyncio.run(compiler.compile_intention(intention))
                except Exception:
                    bytecode = []
            else:
                bytecode = compiler.compile_intention(intention)
            rb = RealityBytecode(
                bytecode_id=experience_id,
                instructions=bytecode or [],
                qualia_state=qualia_state,
                phase=phase,
                timestamp=_now_ts(),
            )
            self.eva_phases.setdefault(phase, {})[experience_id] = rb
        except Exception:
            logger.exception("add_experience_phase failed (non-fatal)")

    def set_memory_phase(self, phase: str) -> None:
        self.eva_phase = phase
        for hook in list(self._environment_hooks):
            try:
                hook({"phase_changed": phase})
            except Exception:
                logger.debug("phase hook failed", exc_info=True)

    def get_memory_phase(self) -> str:
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list[str]:
        return [p for p, exps in self.eva_phases.items() if experience_id in exps]

    def add_environment_hook(self, hook: Callable[..., Any]) -> None:
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict[str, Callable[..., Any]]:
        return {
            "eva_ingest_linguistic_experience": self.eva_ingest_linguistic_experience,
            "eva_recall_linguistic_experience": self.eva_recall_linguistic_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }
