"""EVA memory orchestrator utilities and embedder.

This micro-patch removes stray embedded content that previously made the
module unparsable. The file now begins with a concise module docstring and
imports follow.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import random
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

# Optional numpy import to avoid hard dependency at import time
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore

# Typing alias for numpy arrays; only import actual ndarray for type-checkers
if TYPE_CHECKING:
    from numpy import ndarray as NDArray  # type: ignore
else:
    NDArray = Any  # type: ignore

# Avoid heavy runtime imports that can create circular dependencies. Use
# TYPE_CHECKING for static analyzers and runtime try/except fallbacks.
if TYPE_CHECKING:
    from crisalida_lib.EVA.core_types import (
        EVAExperience,
        LivingSymbolRuntime,
        QuantumField,
        RealityBytecode,
    )
    from crisalida_lib.EVA.typequalia import QualiaState
else:
    # Runtime: attempt to import concrete types but degrade gracefully.
    try:
        from crisalida_lib.EVA.core_types import (
            EVAExperience,
            LivingSymbolRuntime,
            QuantumField,
            RealityBytecode,
        )

        # Prefer canonical QualiaState from typequalia
        from crisalida_lib.EVA.typequalia import QualiaState
    except Exception:  # pragma: no cover - optional runtime dependency
        EVAExperience = Any  # type: ignore
        LivingSymbolRuntime = Any  # type: ignore
        QuantumField = Any  # type: ignore
        RealityBytecode = Any  # type: ignore
        QualiaState = Any  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------
# Config / Constants
# -----------------------------
DEFAULT_DIM = 256
DEFAULT_EMBED_SEED = 1337
DEFAULT_MAX_REPLAY = 20_000
DEFAULT_K = 8


# -----------------------------
# Utils
# -----------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value between lo and hi."""
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def cos_sim(a: NDArray, b: NDArray) -> float:
    """Cosine similarity between two vectors (safe)."""
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# 1) Embedding (256-D)
# -----------------------------
class HashingEmbedder:
    """
    Deterministic feature-hashing embedder (L2-normalized).
    Suitable for strings, numbers, sequences and small dicts.
    """

    def __init__(
        self,
        d: int = DEFAULT_DIM,
        ngram: tuple[int, int] = (3, 5),
        use_words: bool = True,
        seed: int = DEFAULT_EMBED_SEED,
    ):
        self.d = int(d)
        self.ngram = ngram
        self.use_words = use_words
        self._rng = np.random.default_rng(seed)
        self.key = self._rng.bytes(16)

    def __call__(self, data: Any) -> NDArray:
        v = np.zeros(self.d, dtype=np.float32)
        if isinstance(data, str):
            self._add_text(v, data)
        elif (
            isinstance(data, list)
            or isinstance(data, tuple)
            or (
                np is not None
                and hasattr(np, "ndarray")
                and isinstance(data, np.ndarray)
            )
        ):
            if np is not None:
                arr = np.asarray(data, dtype=np.float32).ravel()
            else:
                # fallback to a simple Python list when numpy is unavailable
                arr = list(data)
            self._add_sequence(v, arr)
        elif isinstance(data, int) or isinstance(data, float):
            self._add_number(v, float(data))
        elif isinstance(data, dict):
            self._add_dict(v, data)
        else:
            self._add_text(v, str(data))
        n = float(np.linalg.norm(v))
        return (v / n) if n > 0.0 else v

    def _h(self, s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8"), key=self.key, digest_size=8).digest()
        return int.from_bytes(h, "big", signed=False)

    def _index_sign(self, token: str) -> tuple[int, float]:
        h = self._h(token)
        idx = h % self.d
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        return idx, sign

    def _add_text(self, v: NDArray, text: str) -> None:
        text = (text or "").lower()
        if self.use_words:
            for w in re.findall(r"[a-záéíóúñü0-9]+", text):
                idx, s = self._index_sign("w|" + w)
                v[idx] += s
        t = f"^{text}$"
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(max(0, len(t) - n + 1)):
                ng = t[i : i + n]
                idx, s = self._index_sign(f"c{n}|{ng}")
                v[idx] += 0.5 * s

    def _add_sequence(self, v: NDArray, arr: Any) -> None:
        """Add a sequence-like value into the feature-hash vector.

        This implementation keeps logic small and deterministic and avoids any
        heavy runtime dependency. It hashes the stringified entries.
        """
        for i, val in enumerate(arr):
            try:
                token = f"s{i}|{str(val)}"
                idx, sign = self._index_sign(token)
                v[idx] += sign * 0.5
            except Exception:
                # Skip problematic items but don't fail the whole embedding
                continue

    def _add_number(self, v: NDArray, num: float) -> None:
        """Add a single numeric value into the feature-hash vector."""
        try:
            token = f"n|{float(num)}"
            idx, sign = self._index_sign(token)
            v[idx] += sign
        except Exception:
            return

    def _add_dict(self, v: NDArray, d: dict) -> None:
        """Add a small dict by hashing key/value pairs."""
        try:
            for k, val in d.items():
                self._add_text(v, f"{k}:{val}")
        except Exception:
            return


def get_embedding(data: Any) -> NDArray:
    """Convenience wrapper for HashingEmbedder."""
    return HashingEmbedder(d=DEFAULT_DIM)(data)


# -----------------------------
# 2) Predictor AR(1) para PU
# -----------------------------
class ARPredictor:
    """
    Lightweight linear AR(1) predictor used to compute Prediction Utility (PU).
    """

    def __init__(
        self, d: int, lr: float = 0.01, l2: float = 1e-4, init_scale: float = 0.9
    ):
        self.d = int(d)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.A = np.eye(self.d, dtype=np.float32) * float(init_scale)

    def loss(self, x: NDArray, y: NDArray) -> float:
        e = (self.A @ x) - y
        return float(np.mean(e * e))

    def update(self, x: NDArray, y: NDArray, steps: int = 1) -> float:
        for _ in range(max(1, int(steps))):
            y_hat = self.A @ x
            e = (y_hat - y).reshape(-1, 1)
            grad = 2.0 * (e @ x.reshape(1, -1)) + 2.0 * self.l2 * self.A
            self.A -= self.lr * grad.astype(np.float32)
        return self.loss(x, y)

    def compute_PU_and_update(self, prev_x: NDArray | None, curr_y: NDArray) -> float:
        if prev_x is None:
            return 0.0
        before = self.loss(prev_x, curr_y)
        _ = self.update(prev_x, curr_y, steps=1)
        after = self.loss(prev_x, curr_y)
        if before <= 1e-9:
            return 0.0
        pu = (before - after) / before
        return clamp(pu, 0.0, 1.0)


# -----------------------------
# 3) Memoria (laberinto por entidad)
# -----------------------------
@dataclass
class Node:
    id: int
    kind: str
    embed: NDArray
    data: dict
    valence: float
    arousal: float
    salience: float = 0.1
    last_access: float = field(default_factory=time.time)
    freq: int = 0
    origin: str = "WAKE"
    edges_out: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    edges_in: dict[int, float] = field(default_factory=lambda: defaultdict(float))


class MentalLaby:
    """
    Individual mental labyrinth:
    - ingestion, recall, replay (dreams)
    - compression/pruning and fast-path policy
    """

    def __init__(
        self,
        d: int = DEFAULT_DIM,
        K: int = DEFAULT_K,
        temp_store: float = 0.7,
        tau_recency: float = 60.0,
        embedder: Callable[[Any], NDArray] | None = None,
        predictor: ARPredictor | None = None,
        max_nodes: int = 100_000,
        storage_pressure: float = 0.0,
    ):
        # Basic state
        self.d = int(d)
        self.nodes: dict[int, Node] = {}
        self.next_id = 0

        # Tunables
        self.K = int(K)
        self.temp_store = float(temp_store)
        self.tau = float(tau_recency)

        # replay buffer (deque) - kept minimal for import-time safety
        # typed for mypy: deque of (priority, node_id)
        self.replay: deque[tuple[float, int]] = deque(maxlen=DEFAULT_MAX_REPLAY)
        self.policy: list[Any] = []
        self.mode = "WAKE"

        # runtime helpers
        self.embedder = embedder if embedder is not None else HashingEmbedder(d=self.d)
        self.predictor = predictor
        self._last_observed = None
        self.max_nodes = int(max_nodes)
        self.storage_pressure = float(storage_pressure)
        self._lock = threading.RLock()

    # Class-level attribute annotations for static analyzers
    embedder: Callable[[Any], NDArray] | None
    predictor: ARPredictor | None
    _last_observed: NDArray | None

    def _call_embedder(self, data: Any) -> NDArray:
        """Call the configured embedder with a safe fallback and return an NDArray-like result."""
        from typing import cast

        if np is not None and hasattr(np, "ndarray") and isinstance(data, np.ndarray):
            return cast(NDArray, data)

        # mypy: ensure embedder_callable is seen as a callable
        embedder_callable = (
            cast(Callable[[Any], NDArray], self.embedder)
            if callable(self.embedder)
            else cast(Callable[[Any], NDArray], HashingEmbedder(d=self.d))
        )
        raw = embedder_callable(data)
        try:
            if np is not None and hasattr(raw, "astype"):
                return cast(NDArray, raw.astype(np.float32))
            if np is not None:
                return cast(NDArray, np.asarray(raw, dtype=np.float32))
        except Exception:
            # best-effort fallback: return raw as-is (callers expect NDArray-like)
            return cast(NDArray, raw)
        return cast(NDArray, raw)

    def store(
        self,
        data_or_embed: Any,
        valence: float = 0.0,
        arousal: float = 0.0,
        kind: str = "semantic",
        PU: float | None = None,
        surprise: float = 0.0,
        now: float | None = None,
    ) -> int:
        """Store a new memory or reinforce an existing one. Returns node id."""
        with self._lock:
            try:
                # Normalize embed: accept either a raw NDArray or data to be
                # embedded using the configured embedder. Use the helper
                # _call_embedder to centralize runtime guards and casting.
                if (
                    np is not None
                    and hasattr(np, "ndarray")
                    and isinstance(data_or_embed, np.ndarray)
                ):
                    embed = cast(NDArray, data_or_embed.astype(np.float32))
                    payload = {"raw": "embed"}
                else:
                    embed = self._call_embedder(data_or_embed)
                    payload = {"raw": data_or_embed}

                now = now or time.time()
                sim_list = self._similar_nodes(embed)
                novelty = 1.0 - (sim_list[0][0] if sim_list else 0.0)
                A = clamp(0.5 * abs(valence) + 0.5 * arousal)
                S = clamp(surprise)

                if PU is None and self.predictor is not None and embed is not None:
                    PU = self.predictor.compute_PU_and_update(
                        cast(NDArray, self._last_observed), cast(NDArray, embed)
                    )
                PU = clamp(PU or 0.0)

                if not sim_list or (novelty > 0.2 or S > 0.3 or PU > 0.2):
                    nid = self._add_node(embed, payload, valence, arousal, kind)
                    anchors = self._choose_anchors(sim_list, A, PU, S, now)
                    for sim_i, i in anchors:
                        R = math.exp(-(now - self.nodes[i].last_access) / self.tau)
                        w = 0.6 * sim_i + 0.2 * A + 0.15 * PU + 0.05 * R
                        self._link(nid, i, w)
                        self._link(i, nid, w * 0.6)
                    self._push_replay(nid, A, S, PU)
                else:
                    top_i = sim_list[0][1]
                    self._reinforce(top_i, valence, arousal, PU)
                    nid = top_i
                    self._push_replay(top_i, A, S, PU)

                self._last_observed = embed
                if len(self.nodes) > self.max_nodes:
                    self._budget_prune(target=int(self.max_nodes * 0.95))
                return nid
            except Exception as e:
                logger.exception("Error storing memory: %s", e)
                raise

    def recall(self, cue, max_hops=3):
        if np is not None and hasattr(np, "ndarray") and isinstance(cue, np.ndarray):
            cue_embed = cue
        else:
            cue_embed = self._call_embedder(cue)
        frontier = [(cos_sim(cue_embed, n.embed), nid) for nid, n in self.nodes.items()]
        frontier.sort(reverse=True)
        act = defaultdict(float)
        for s, nid in frontier[: self.K]:
            act[nid] += s
            self._touch(nid)
            for j, w in self.nodes[nid].edges_out.items():
                act[j] += s * w
        items = sorted(act.items(), key=lambda x: x[1], reverse=True)[: self.K]
        if not items:
            return None
        vecs = [self.nodes[i].embed * w for i, w in items]
        rec_embed = np.mean(vecs, axis=0)
        nrm = np.linalg.norm(rec_embed)
        if nrm > 0:
            rec_embed /= nrm
        # reconsolidación hebbiana ligera + snapshot narrativo
        ids = [i for i, _ in items]
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                self.nodes[i].edges_out[j] += 0.01
        self.store(rec_embed, kind="narrative", PU=None)
        return rec_embed, ids

    # ---- Sueños / Compresión ----
    def dream_cycle(
        self,
        mode="MIXED",
        steps_rem=10,
        T_rem=0.8,
        replay_k=64,
        prune_threshold=0.02,
        compress_similarity=0.985,
        gen_prob=0.15,
    ):
        """
        Ejecuta una fase de sueños:
        - NREM: replay/consolidación/compresión
        - REM: generación creativa y puentes entre clusters
        - MIXED: ambos
        """
        if mode == "NREM" or mode == "MIXED":
            self._nrem_pass(replay_k=replay_k, compress_similarity=compress_similarity)
        if mode == "REM" or mode == "MIXED":
            self._rem_pass(steps=steps_rem, T=T_rem, gen_prob=gen_prob)
        # pruning suave por presión de almacenamiento y enlaces débiles
        self._prune_links(th=prune_threshold)
        if len(self.nodes) > self.max_nodes:
            self._budget_prune(target=int(self.max_nodes * 0.95))

    # ---- Fast path ----
    def act_fast(self, cue_embed: NDArray):
        best_sim = 0.0
        best_action = None
        for pat, action, th, _util in self.policy:
            s = cos_sim(cue_embed, pat)
            if s >= th and s > best_sim:
                best_sim = s
                best_action = action
        if best_action is not None:
            self._push_replay(0, 0.1, 0.1, 0.2)  # marcador ficticio
        return best_action

    # ---- Internals ----
    def _add_node(self, embed, data, valence, arousal, kind, origin="WAKE"):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(
            nid, kind, embed.astype(np.float32), data, valence, arousal, origin=origin
        )
        return nid

    def _link(self, i, j, w):
        self.nodes[i].edges_out[j] = max(self.nodes[i].edges_out.get(j, 0.0), w)
        self.nodes[j].edges_in[i] = max(self.nodes[j].edges_in.get(i, 0.0), w)

    def _reinforce(self, nid, valence, arousal, PU):
        n = self.nodes[nid]
        A = clamp(0.5 * abs(valence) + 0.5 * arousal)
        n.salience = clamp(n.salience + 0.1 * A + 0.05 * (PU or 0.0))

    def _similar_nodes(self, embed, top=64):
        sims = [(cos_sim(embed, n.embed), nid) for nid, n in self.nodes.items()]
        sims.sort(reverse=True)
        return sims[:top]

    def _choose_anchors(self, sim_list, A, PU, S, now):
        scores = []
        for sim_i, i in sim_list:
            n = self.nodes[i]
            R = math.exp(-(now - n.last_access) / self.tau)
            SC = 1.0 / (1.0 + len(n.edges_in))
            score = 0.4 * sim_i + 0.2 * A + 0.15 * PU + 0.15 * S + 0.05 * R + 0.05 * SC
            scores.append((score, sim_i, i))
        if not scores:
            return []
        xs = np.array([s for s, _, _ in scores])
        T = max(self.temp_store, 1e-3)
        p = np.exp(xs / T)
        p = p / np.sum(p)
        idxs = np.random.choice(
            len(scores), size=min(self.K, len(scores)), replace=False, p=p
        )
        return [(scores[k][1], scores[k][2]) for k in idxs]

    def _push_replay(self, nid, A, S, PU):
        # si nid==0 puede ser marcador; priorizamos por saliencia*ganancia
        s = self.nodes[nid].salience if nid in self.nodes else 0.1
        prio = s * (A + (S or 0) + (PU or 0) + 1e-3)
        self.replay.appendleft((prio, nid))
        self.replay = deque(
            sorted(self.replay, reverse=True, key=lambda x: x[0]),
            maxlen=self.replay.maxlen,
        )

    def _touch(self, nid):
        n = self.nodes[nid]
        n.last_access = time.time()
        n.freq += 1

    # ---- NREM/REM helpers ----
    def _nrem_pass(self, replay_k=64, compress_similarity=0.985):
        seeds = [nid for _, nid in list(self.replay)[:replay_k] if nid in self.nodes]
        for nid in seeds:
            path = self._greedy_temporal_path(nid, L=6)
            for i, j in zip(path, path[1:], strict=False):
                self.nodes[i].edges_out[j] += 0.02
                self.nodes[i].salience = clamp(self.nodes[i].salience + 0.02)
        # compresión local en clusters activados
        for nid in seeds:
            cluster = self._k_hop(nid, hops=2, limit=64)
            self._compress_cluster(cluster, threshold=compress_similarity)

    def _rem_pass(self, steps=10, T=0.8, gen_prob=0.15):
        # mezcla creativa + saltos entre clusters + generación autónoma
        seeds = [nid for _, nid in list(self.replay)[: self.K] if nid in self.nodes]
        if not seeds and len(self.nodes) > 0:
            seeds = [random.choice(list(self.nodes.keys()))]
        for nid in seeds:
            walked = self._random_walk(nid, steps, T)
            if not walked:
                continue
            vec = np.zeros(self.d)
            total = 0.0
            for i in walked:
                w = self.nodes[i].salience + 0.1
                vec += w * self.nodes[i].embed
                total += w
            if total > 0:
                vec /= total
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vec /= nrm
            did = self._add_node(
                vec, {"dream_path": walked}, 0.0, 0.3, kind="dream", origin="REM"
            )
            for i in walked[-self.K :]:
                self._link(did, i, 0.2)
            self._push_replay(did, A=0.2, S=0.4, PU=0.2)

        # generación autónoma (pensamiento sin estímulo)
        if len(self.nodes) > 0 and random.random() < gen_prob:
            base = random.choice(list(self.nodes.values()))
            noise = np.random.normal(0, 0.1, size=self.d).astype(np.float32)
            vec = base.embed + noise
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vec /= nrm
            did = self._add_node(
                vec, {"dream_spark": base.id}, 0.0, 0.25, kind="dream", origin="REM"
            )
            self._push_replay(did, 0.2, 0.5, 0.2)

    def _greedy_temporal_path(self, start, L=6):
        path = [start]
        cur = start
        for _ in range(L - 1):
            outs = self.nodes[cur].edges_out
            if not outs:
                break
            nxt = max(outs.items(), key=lambda kv: kv[1])[0]
            path.append(nxt)
            cur = nxt
        return path

    def _k_hop(self, nid, hops=2, limit=128) -> list[int]:
        seen = {nid}
        frontier = [nid]
        for _ in range(hops):
            nxt = []
            for u in frontier:
                for v in list(self.nodes[u].edges_out.keys()) + list(
                    self.nodes[u].edges_in.keys()
                ):
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
                        if len(seen) >= limit:
                            return list(seen)
            frontier = nxt
        return list(seen)

    def _compress_cluster(self, ids: list[int], threshold=0.985) -> Any:
        if not ids:
            return
        ids = list(ids)
        base = ids[0]
        base_node = self.nodes[base]
        to_remove = []
        for j in ids[1:]:
            if j not in self.nodes:
                continue
            if cos_sim(base_node.embed, self.nodes[j].embed) > threshold:
                # merge j -> base
                for k, w in self.nodes[j].edges_out.items():
                    base_node.edges_out[k] = max(base_node.edges_out.get(k, 0.0), w)
                for k, w in self.nodes[j].edges_in.items():
                    base_node.edges_in[k] = max(base_node.edges_in.get(k, 0.0), w)
                base_node.salience = clamp(base_node.salience + self.nodes[j].salience)
                to_remove.append(j)
        for j in to_remove:
            del self.nodes[j]

    def _random_walk(self, nid, steps, T):
        cur = nid
        walked = [cur]
        for _ in range(steps):
            outs = self.nodes[cur].edges_out
            if not outs:
                break
            items = list(outs.items())
            logits = np.array([w for _, w in items])
            probs = np.exp(logits / max(T, 1e-3))
            probs = probs / np.sum(probs)
            cur = random.choices([j for j, _ in items], weights=probs, k=1)[0]
            walked.append(cur)
        return walked

    def _prune_links(self, th=0.02):
        for n in self.nodes.values():
            to_del = [j for j, w in n.edges_out.items() if w < th]
            for j in to_del:
                del n.edges_out[j]

    def _budget_prune(self, target: int):
        """
        Pruning por presupuesto: score de descarte bajo si (salience baja, freq baja,
        viejo, PU bajo). storage_pressure añade penalización extra.
        """
        if len(self.nodes) <= target:
            return
        now = time.time()
        scored = []
        for nid, n in self.nodes.items():
            age = now - n.last_access
            score_keep = (
                0.6 * n.salience
                + 0.2 * math.exp(-age / self.tau)
                + 0.2 * math.tanh(n.freq / 10)
            )
            score_keep -= 0.2 * self.storage_pressure  # presión reduce score
            scored.append((score_keep, nid))
        scored.sort()  # los peores primero
        to_remove = [nid for _, nid in scored[: max(0, len(self.nodes) - target)]]
        for nid in to_remove:
            # desconectar
            for j in list(self.nodes[nid].edges_out.keys()):
                if j in self.nodes and nid in self.nodes[j].edges_in:
                    del self.nodes[j].edges_in[nid]
            for i in list(self.nodes[nid].edges_in.keys()):
                if i in self.nodes and nid in self.nodes[i].edges_out:
                    del self.nodes[i].edges_out[nid]
            del self.nodes[nid]

    def search(self, query: Any, top_k: int = 5, kind: str | None = None) -> list:
        """
        Búsqueda semántica avanzada en la memoria.
        Permite filtrar por tipo de nodo y retorna los nodos más similares.
        """
        query_embed = self._call_embedder(query)
        results = []
        for nid, node in self.nodes.items():
            if kind and node.kind != kind:
                continue
            sim = cos_sim(query_embed, node.embed)
            results.append((sim, nid, node))
        results.sort(reverse=True)
        return results[:top_k]

    def trace_path(self, start_id: int, max_depth: int = 8) -> list:
        """
        Traza el camino de activación desde un nodo dado.
        Útil para explicar cadenas de memoria y razonamiento.
        """
        path = [start_id]
        cur = start_id
        for _ in range(max_depth):
            outs = self.nodes[cur].edges_out
            if not outs:
                break
            nxt = max(outs.items(), key=lambda kv: kv[1])[0]
            if nxt in path:
                break
            path.append(nxt)
            cur = nxt
        return path

    def get_salient_nodes(self, top_k: int = 10) -> list:
        """
        Retorna los nodos más salientes (importantes) para diagnóstico rápido.
        """
        return sorted(self.nodes.values(), key=lambda n: n.salience, reverse=True)[
            :top_k
        ]

    def export_snapshot(self) -> dict:
        """
        Exporta un snapshot compacto de la memoria para backup, migración o análisis.
        """
        return {
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "valence": n.valence,
                    "arousal": n.arousal,
                    "salience": n.salience,
                    "freq": n.freq,
                    "origin": n.origin,
                    "last_access": n.last_access,
                    "edges_out": dict(n.edges_out),
                    "edges_in": dict(n.edges_in),
                    "data": n.data,
                }
                for n in self.nodes.values()
            ]
        }

    def import_snapshot(self, snapshot: dict):
        """
        Importa un snapshot previamente exportado.
        """
        self.nodes.clear()
        for n in snapshot.get("nodes", []):
            node = Node(
                id=n["id"],
                kind=n["kind"],
                embed=np.zeros(self.d),  # No se serializa el embedding por defecto
                data=n["data"],
                valence=n["valence"],
                arousal=n["arousal"],
                salience=n["salience"],
                last_access=n["last_access"],
                freq=n["freq"],
                origin=n.get("origin", "WAKE"),
                edges_out=defaultdict(float, n.get("edges_out", {})),
                edges_in=defaultdict(float, n.get("edges_in", {})),
            )
            self.nodes[node.id] = node


# -----------------------------
# 4) Envoltura por Entidad + Hooks
# -----------------------------
@dataclass
class EntityMemory:
    entity_id: str
    mind: MentalLaby
    affect_bias: float = 0.0
    valence_bias: float = 0.0
    storage_budget: int = 100_000

    def ingest(
        self,
        data: Any,
        valence: float = 0.0,
        arousal: float = 0.0,
        kind: str = "semantic",
        PU: float | None = None,
        surprise: float = 0.0,
    ) -> int:
        valence = clamp(valence + self.valence_bias, -1.0, 1.0)
        arousal = clamp(arousal + self.affect_bias, 0.0, 1.0)
        self.mind.max_nodes = int(self.storage_budget)
        return self.mind.store(
            data, valence=valence, arousal=arousal, kind=kind, PU=PU, surprise=surprise
        )

    def recall(self, cue: Any):
        return self.mind.recall(cue)

    def dream(self, mode: str = "MIXED"):
        self.mind.dream_cycle(mode=mode)

    def on_cosmic_influence(self, lattice_vector: dict[str, float]) -> None:
        """
        lattice_vector puede contener pesos sephirot/qliphoth.
        Se traduce en sesgos simples de afecto/valencia para esta entidad.
        """
        seph = sum(
            v
            for k, v in lattice_vector.items()
            if k.lower()
            in {
                "keter",
                "chokmah",
                "binah",
                "chesed",
                "gevurah",
                "tiphereth",
                "netzach",
                "hod",
                "yesod",
                "malkuth",
            }
        )
        qlip = sum(
            v
            for k, v in lattice_vector.items()
            if k.lower()
            not in {
                "keter",
                "chokmah",
                "binah",
                "chesed",
                "gevurah",
                "tiphereth",
                "netzach",
                "hod",
                "yesod",
                "malkuth",
            }
        )
        self.valence_bias = clamp(self.valence_bias + 0.1 * seph, -0.5, 0.5)
        self.affect_bias = clamp(self.affect_bias + 0.1 * qlip, -0.2, 0.2)

    def on_qualia_update(self, qualia_state: dict[str, float]) -> None:
        """Integrate QualiaState-like dict into local biases."""
        v = clamp(qualia_state.get("emotional_valence", 0.0), -1.0, 1.0)
        a = clamp(qualia_state.get("consciousness_density", 0.0), 0.0, 1.0)
        self.valence_bias = 0.7 * self.valence_bias + 0.3 * v
        self.affect_bias = 0.7 * self.affect_bias + 0.3 * a


# -----------------------------
# 5) Noosfera (intercambio)
# -----------------------------
@dataclass
class QualiaCrystal:
    crystal_id: str
    entity_id: str
    content: dict
    timestamp: float
    resonance_frequency: float
    likes: int = 0
    shares: int = 0
    comments: list[dict] = field(default_factory=list)


class NoosphereBus:
    """Minimal exchange bus for QualiaCrystal objects."""

    def __init__(self):
        self.feed: list[QualiaCrystal] = []
        self._lock = threading.RLock()

    def publish(
        self, entity_id: str, content: dict[str, Any], resonance: float = 1.0
    ) -> QualiaCrystal:
        with self._lock:
            c = QualiaCrystal(
                crystal_id=f"C{len(self.feed) + 1}",
                entity_id=entity_id,
                content=content,
                timestamp=time.time(),
                resonance_frequency=float(resonance),
            )
            self.feed.append(c)
            return c

    def stream_for(self, entity_id: str, k: int = 10) -> list[QualiaCrystal]:
        with self._lock:
            return sorted(
                self.feed,
                key=lambda c: (c.resonance_frequency, c.likes, c.shares),
                reverse=True,
            )[: max(0, int(k))]


# -----------------------------
# 6) Orquestador Multi-Entidad
# -----------------------------
class UniverseMindOrchestrator:
    """
    High-level orchestrator for multiple EntityMemory instances.
    """

    def __init__(
        self,
        d: int = DEFAULT_DIM,
        default_budget: int = 100_000,
        embedder: Callable[[Any], NDArray] | None = None,
        predictor_factory: Callable[[], ARPredictor] | None = None,
    ):
        self.d = int(d)
        self.embedder = embedder if embedder is not None else HashingEmbedder(d=self.d)
        self.predictor_factory = predictor_factory or (lambda: ARPredictor(d=self.d))
        self.entities: dict[str, EntityMemory] = {}
        self.noosphere = NoosphereBus()
        self._backup_lock = threading.RLock()

    def add_entity(
        self,
        entity_id: str,
        budget: int | None = None,
        storage_pressure: float = 0.0,
        default_budget: int | None = None,
    ) -> None:
        effective_budget = int(
            budget
            if budget is not None
            else (default_budget if default_budget is not None else 100_000)
        )
        mind = MentalLaby(
            d=self.d,
            K=DEFAULT_K,
            embedder=cast(Callable[[Any], NDArray], self.embedder),
            predictor=self.predictor_factory(),
            max_nodes=effective_budget,
            storage_pressure=float(storage_pressure),
        )
        self.entities[entity_id] = EntityMemory(
            entity_id=entity_id, mind=mind, storage_budget=effective_budget
        )

    def ingest(
        self,
        entity_id: str,
        data: Any,
        valence: float = 0.0,
        arousal: float = 0.0,
        kind: str = "semantic",
        PU: float | None = None,
        surprise: float = 0.0,
    ) -> int:
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        return self.entities[entity_id].ingest(
            data, valence, arousal, kind, PU, surprise
        )

    def recall(self, entity_id: str, cue: Any):
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        return self.entities[entity_id].recall(cue)

    def dream_batch(self, entity_ids: Iterable[str] | None = None, mode: str = "MIXED"):
        ids = list(entity_ids) if entity_ids is not None else list(self.entities.keys())
        for eid in ids:
            if eid in self.entities:
                self.entities[eid].dream(mode=mode)

    def global_night(self, mode: str = "MIXED"):
        for e in self.entities.values():
            e.dream(mode=mode)

    def apply_cosmic_influence(self, lattice_field: dict[str, float]) -> None:
        for e in self.entities.values():
            e.on_cosmic_influence(lattice_field)

    def publish_state(
        self, entity_id: str, qualia_state: dict[str, float], resonance: float = 1.0
    ):
        return self.noosphere.publish(
            entity_id, {"qualia_state": qualia_state}, resonance
        )

    def consume_trending(self, entity_id: str, k: int = 10):
        crystals = self.noosphere.stream_for(entity_id, k=k)
        for c in crystals:
            payload = {
                "noosphere_crystal": c.crystal_id,
                "from": c.entity_id,
                **c.content,
            }
            if entity_id in self.entities:
                self.entities[entity_id].ingest(
                    payload,
                    valence=0.05,
                    arousal=0.2,
                    kind="social",
                    PU=0.0,
                    surprise=0.2,
                )
        return crystals

    def tick(self, schedule: dict[str, dict[str, str]]):
        for eid, plan in schedule.items():
            mode = plan.get("mode", "WAKE")
            if mode in ("NREM", "REM", "MIXED") and eid in self.entities:
                self.entities[eid].dream(mode=mode)

    def adaptive_ingest(
        self, entity_id: str, data: Any, context: dict[str, Any] | None = None
    ):
        context = context or {}
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        entity = self.entities[entity_id]
        valence = clamp(context.get("valence", 0.0) + entity.valence_bias, -1.0, 1.0)
        arousal = clamp(context.get("arousal", 0.0) + entity.affect_bias, 0.0, 1.0)
        surprise = context.get("surprise", 0.0)
        PU = context.get("PU", None)
        kind = context.get("kind", "semantic")
        return entity.ingest(
            data, valence=valence, arousal=arousal, kind=kind, PU=PU, surprise=surprise
        )

    def get_entity_stats(self, entity_id: str) -> dict[str, Any]:
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        entity = self.entities[entity_id]
        mind = entity.mind
        stats = {
            "total_nodes": len(mind.nodes),
            "avg_salience": (
                float(np.mean([n.salience for n in mind.nodes.values()]))
                if mind.nodes
                else 0.0
            ),
            "avg_freq": (
                float(np.mean([n.freq for n in mind.nodes.values()]))
                if mind.nodes
                else 0.0
            ),
            "valence_bias": entity.valence_bias,
            "affect_bias": entity.affect_bias,
            "storage_budget": entity.storage_budget,
        }
        return stats

    def backup_all(self, path: str) -> None:
        """Thread-safe backup of all entity memories to disk (pickle)."""
        with self._backup_lock:
            snapshot = {
                eid: e.mind.export_snapshot() for eid, e in self.entities.items()
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"version": 1, "snapshot": snapshot, "ts": time.time()}, f)
            logger.info("Backup written to %s", path)

    def restore_all(self, path: str) -> None:
        """Restore previously saved backup (thread-safe)."""
        with self._backup_lock:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "rb") as f:
                data = pickle.load(f)
            snapshot = data.get("snapshot", data)
            for eid, snap in snapshot.items():
                if eid in self.entities:
                    self.entities[eid].mind.import_snapshot(snap)
            logger.info("Restore completed from %s", path)

    def analyze_noosphere(self) -> dict[str, Any]:
        feed = self.noosphere.feed
        if not feed:
            return {}
        top_crystals = sorted(feed, key=lambda c: c.resonance_frequency, reverse=True)[
            :10
        ]
        entity_counts: dict[str, int] = defaultdict(int)
        for c in feed:
            entity_counts[c.entity_id] += 1
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        return {
            "top_crystals": [
                (c.crystal_id, c.resonance_frequency) for c in top_crystals
            ],
            "top_entities": top_entities,
            "total_crystals": len(feed),
        }

    def entity_search(
        self, entity_id: str, query: Any, top_k: int = 5, kind: str | None = None
    ):
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        return self.entities[entity_id].mind.search(query, top_k=top_k, kind=kind)

    def trace_entity_memory(self, entity_id: str, start_id: int, max_depth: int = 8):
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        return self.entities[entity_id].mind.trace_path(start_id, max_depth=max_depth)

    def get_salient_memories(self, entity_id: str, top_k: int = 10):
        if entity_id not in self.entities:
            raise KeyError(f"Entity '{entity_id}' not registered")
        return self.entities[entity_id].mind.get_salient_nodes(top_k=top_k)

    def optimize_storage(self) -> None:
        for eid, entity in list(self.entities.items()):
            try:
                stats = self.get_entity_stats(eid)
                usage = stats["total_nodes"] / max(1, stats["storage_budget"])
                if usage > 0.9:
                    entity.mind.storage_pressure = min(
                        1.0, entity.mind.storage_pressure + 0.1
                    )
                    entity.storage_budget = max(1000, int(entity.storage_budget * 0.95))
                    entity.mind.max_nodes = entity.storage_budget
                elif usage < 0.5:
                    entity.mind.storage_pressure = max(
                        0.0, entity.mind.storage_pressure - 0.05
                    )
                    entity.storage_budget = int(entity.storage_budget * 1.05)
                    entity.mind.max_nodes = entity.storage_budget
            except Exception:
                logger.exception("Error optimizing storage for entity %s", eid)


# -----------------------------
# 7) EVA: Extensiones Avanzadas
# -----------------------------


class EVAMemoryOrchestrator(UniverseMindOrchestrator):
    """
    EVA-specific orchestrator with RealityBytecode compilation/recall and environment hooks.
    """

    # type: ignore[attr-defined]
    _environment_hooks: list[Callable[..., Any]]

    def __init__(
        self,
        d: int = DEFAULT_DIM,
        default_budget: int = 100_000,
        embedder: Callable[[Any], NDArray] | None = None,
        predictor_factory: Callable[[], ARPredictor] | None = None,
        phase: str | None = "default",
        gpu_enabled: bool = False,
        ecs_components: dict[str, Any] | None = None,
        persistence_manager: Any = None,
    ):
        super().__init__(
            d=d,
            default_budget=default_budget,
            embedder=embedder,
            predictor_factory=predictor_factory,
        )

        # runtime EVA containers and hooks
        try:
            self.eva_runtime = LivingSymbolRuntime()
        except Exception:
            self.eva_runtime = None  # type: ignore
        self.eva_phase = phase
        self.eva_memory_store = {}  # type: dict[str, RealityBytecode]
        self.eva_experience_store = {}  # type: dict[str, EVAExperience]
        self.eva_phases = {}  # type: dict[str, dict[str, RealityBytecode]]
        self._environment_hooks: list[Callable[..., Any]] = []
        self._gpu_enabled = bool(gpu_enabled)
        self._ecs_components = ecs_components or {}
        self._persistence_manager = persistence_manager

    def add_environment_hook(self, hook: Callable[[Any], None]) -> None:
        """Register a hook for manifestation callbacks (safer wrapper)."""
        if not callable(hook):
            raise TypeError("hook must be callable")
        self._environment_hooks.append(hook)

    def eva_ingest_experience(
        self,
        entity_id: str,
        experience_data: dict,
        qualia_state: QualiaState,
        phase: str | None = None,
    ) -> str:
        """
        Compile an arbitrary experience into RealityBytecode and store it in EVA memory.
        Supports distributed persistence and benchmarking if enabled.
        """
        phase = phase or self.eva_phase
        experience_data = dict(experience_data)
        experience_data["gpu_enabled"] = self._gpu_enabled
        experience_data["ecs_components"] = list(self._ecs_components.keys())
        intention = {
            "intention_type": "ARCHIVE_ENTITY_EXPERIENCE",
            "entity_id": entity_id,
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        _eva = getattr(self, "eva_runtime", None)
        if (
            _eva is None
            or not hasattr(_eva, "divine_compiler")
            or _eva.divine_compiler is None
        ):
            bytecode = []
        else:
            # mypy cannot always infer that divine_compiler is non-None after hasattr checks;
            # cast to Any and call via a callable-check to satisfy static analysis while
            # preserving runtime behavior.
            from typing import cast

            _dc = cast(Any, _eva.divine_compiler)
            compile_fn = getattr(_dc, "compile_intention", None)
            if callable(compile_fn):
                bytecode = compile_fn(intention)
            else:
                bytecode = []
        experience_id = (
            experience_data.get("experience_id")
            or f"exp_{entity_id}_{hash(str(experience_data))}"
        )
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase or "default_phase",
            timestamp=time.time(),
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if cast(str, phase) not in self.eva_phases:
            self.eva_phases[cast(str, phase)] = {}
        self.eva_phases[cast(str, phase)][experience_id] = reality_bytecode
        if self._persistence_manager:
            self._persistence_manager.eva_ingest_experience(
                experience_data, qualia_state, phase
            )
        if self._gpu_enabled:
            # simulation_metadata may be an optional field on RealityBytecode; use setattr
            # so mypy doesn't complain about attr-defined while preserving runtime semantics.
            try:
                setattr(reality_bytecode, "simulation_metadata", {"gpu_ingest_time": time.time()})
            except Exception:
                pass
        return experience_id

    def eva_recall_experience(
        self, entity_id: str, cue: str, phase: str | None = None
    ) -> dict:
        """
        Execute the RealityBytecode of a stored experience, manifesting the simulation in QuantumField.
        Optimized for GPU/ECS if enabled.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(cast(str, phase), {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode and self._persistence_manager:
            reality_bytecode = self._persistence_manager.eva_recall_experience(
                cue, phase
            )
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA experience"}
        quantum_field = QuantumField()
        manifestations = []
        start = time.time()
        for instr in reality_bytecode.instructions:
            if self._gpu_enabled and hasattr(self.eva_runtime, "gpu_physics_engine"):
                symbol_manifest = (
                    self.eva_runtime.gpu_physics_engine.execute_instruction(
                        instr, quantum_field
                    )
                )
            else:
                symbol_manifest = self.eva_runtime.execute_instruction(
                    instr, quantum_field
                )
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA] MemoryOrchestrator environment hook failed: {e}")
        end = time.time()
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        if self._gpu_enabled:
            eva_experience.metadata = {"gpu_recall_time": end - start}
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
            "timestamp": eva_experience.timestamp,
            "gpu_enabled": self._gpu_enabled,
            "ecs_components": list(self._ecs_components.keys()),
            "benchmark": (
                eva_experience.metadata if hasattr(eva_experience, "metadata") else {}
            ),
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Add an alternative phase (timeline) for an arbitrary experience.
        """
        experience_data = dict(experience_data)
        experience_data["gpu_enabled"] = self._gpu_enabled
        experience_data["ecs_components"] = list(self._ecs_components.keys())
        intention = {
            "intention_type": "ARCHIVE_ENTITY_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        # Guard divine_compiler presence; fall back to empty bytecode when unavailable.
        if getattr(self, "eva_runtime", None) is None or not hasattr(
            self.eva_runtime, "divine_compiler"
        ):
            bytecode = []
        else:
            from typing import cast

            _dc = cast(Any, self.eva_runtime.divine_compiler)
            compile_fn = getattr(_dc, "compile_intention", None)
            if callable(compile_fn):
                bytecode = compile_fn(intention)
            else:
                bytecode = []
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase or "default_phase",
            timestamp=time.time(),
        )
        if cast(str, phase) not in self.eva_phases:
            self.eva_phases[cast(str, phase)] = {}
        self.eva_phases[cast(str, phase)][experience_id] = reality_bytecode
        if self._persistence_manager:
            self._persistence_manager.add_experience_phase(
                experience_id, phase, experience_data, qualia_state
            )

    def set_memory_phase(self, phase: str):
        """Change the active memory phase (timeline)."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Return the current memory phase."""
        return self.eva_phase or "default_phase"

    def get_experience_phases(self, experience_id: str) -> list[str]:
        """List all available phases for an experience."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def get_eva_api(self) -> dict[str, Callable]:
        """Return the EVA memory API for external integration."""
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
            "enable_gpu_optimization": self.enable_gpu_optimization,
            "register_ecs_component": self.register_ecs_component,
        }

    def enable_gpu_optimization(self, enabled: bool = True) -> None:
        """Enable or disable GPU/ECS optimization mode."""
        self._gpu_enabled = bool(enabled)

    def register_ecs_component(self, name: str, component: Any) -> None:
        """Register a named ECS component for future ingestion/manifests."""
        if not isinstance(name, str):
            raise TypeError("component name must be a string")
        self._ecs_components[name] = component
