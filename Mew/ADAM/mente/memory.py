"""
Memory System (definitive, professionalized)
===========================================

- Defensive numeric backend (numpy preferred, pure-Python fallback).
- Clear, typed public API: MentalLaby, EntityMemory, HashingEmbedder, ARPredictor.
- Async-aware EVA persistence (best-effort, schedules coroutine if needed).
- Deterministic hashing embedder, robust normalization and guards.
- Reasonable defaults and tunables for evolution and production use.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Defensive numeric backend
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False

if TYPE_CHECKING:
    from crisalida_lib.ADAM.config import AdamConfig  # type: ignore
else:
    AdamConfig = Any

from crisalida_lib.ADAM.eva_integration.eva_memory_manager import (
    EVAMemoryManager,  # type: ignore
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Utilities & fallbacks ---


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _safe_norm(vec: Any) -> float:
    if HAS_NUMPY and isinstance(vec, (np.ndarray,)):
        return float(np.linalg.norm(vec))
    # fallback: assume iterable
    try:
        s = 0.0
        for v in vec:
            s += float(v) * float(v)
        return math.sqrt(s)
    except Exception:
        return 0.0


def cos_sim(a: Any, b: Any) -> float:
    """
    Cosine similarity with numpy-accelerated path and a pure-Python fallback.
    """
    try:
        if HAS_NUMPY and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(float(np.dot(a, b)) / (na * nb))
        # pure-Python fallback
        ax = list(map(float, a))
        bx = list(map(float, b))
        dot = sum(x * y for x, y in zip(ax, bx, strict=False))
        na = math.sqrt(sum(x * x for x in ax))
        nb = math.sqrt(sum(y * y for y in bx))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(dot / (na * nb))
    except Exception:
        return 0.0


# --- Embedding & prediction helpers ---


class HashingEmbedder:
    """
    Deterministic embedding via feature hashing.

    Produces a fixed-length float vector normalized to unit norm. Uses numpy when available.
    """

    def __init__(
        self,
        d: int = 256,
        ngram: tuple = (3, 5),
        use_words: bool = True,
        seed: int = 1337,
    ):
        self.d = int(d)
        self.ngram = tuple(ngram)
        self.use_words = bool(use_words)
        self._rng = random.Random(seed)
        # deterministic 16-byte key for blake2b; keep consistent across runs unless seed changes
        self.key = self._rng.getrandbits(128).to_bytes(16, "big")

    def __call__(self, data: Any) -> np.ndarray | list[float]:
        v = [0.0] * self.d
        if isinstance(data, str):
            self._add_text(v, data)
        elif HAS_NUMPY and isinstance(data, np.ndarray):
            self._add_sequence(v, data.ravel().tolist())
        elif isinstance(data, (list, tuple)):
            self._add_sequence(v, list(data))
        elif isinstance(data, (int, float)):
            self._add_number(v, float(data))
        elif isinstance(data, dict):
            self._add_dict(v, data)
        else:
            self._add_text(v, str(data))

        # normalize
        n = math.sqrt(sum(x * x for x in v))
        if n > 0:
            v = [float(x / n) for x in v]
        if HAS_NUMPY:
            return np.array(v, dtype=np.float32)
        return v

    def _h(self, s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8"), key=self.key, digest_size=8).digest()
        return int.from_bytes(h, "big", signed=False)

    def _index_sign(self, token: str):
        h = self._h(token)
        idx = h % self.d
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        return idx, sign

    def _add_text(self, v: list[float], text: str):
        text = text.lower()
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

    def _add_sequence(self, v: list[float], arr: list[float]):
        for i, val in enumerate(arr):
            try:
                valf = float(val)
            except Exception:
                continue
            if valf == 0.0:
                continue
            idx, s = self._index_sign(f"idx|{i}")
            v[idx] += s * valf

    def _add_number(self, v: list[float], x: float):
        bucket = (
            int(math.copysign(1, x) * math.floor(math.log1p(abs(x) + 1e-12)))
            if x != 0
            else 0
        )
        idx, s = self._index_sign(f"num|{bucket}")
        v[idx] += s * (1.0 + min(1.0, abs(x)))

    def _add_dict(self, v: list[float], d: dict) -> None:
        for k, val in sorted(d.items()):
            if isinstance(val, (int, float)):
                tok = f"{k}:{round(float(val), 3)}"
                idx, s = self._index_sign("kv|" + tok)
                v[idx] += s
            else:
                tok = f"{k}:{str(val)[:64]}"
                idx, s = self._index_sign("kv|" + tok)
                v[idx] += 0.5 * s


class ARPredictor:
    """
    Lightweight AR(1) model with online update.

    Uses numpy when available; falls back to simple Python lists.
    """

    def __init__(
        self, d: int, lr: float = 0.01, l2: float = 1e-4, init_scale: float = 0.9
    ):
        self.d = int(d)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.init_scale = float(init_scale)
        if HAS_NUMPY:
            self.A = np.eye(self.d, dtype=np.float32) * float(self.init_scale)
        else:
            # fallback to diagonal representation
            self.A_diag = [float(self.init_scale)] * self.d

    def _matvec(self, A: Any, x: Any) -> list[float]:
        if HAS_NUMPY:
            return list(A @ x)
        # fallback: diagonal multiply
        return [self.A_diag[i] * x[i] for i in range(self.d)]

    def loss(self, x: Any, y: Any) -> float:
        try:
            if HAS_NUMPY:
                e = (self.A @ x) - y
                return float(np.mean(e * e))
            y_hat = self._matvec(None, x)
            e = [(yh - yy) for yh, yy in zip(y_hat, list(y), strict=False)]
            return float(sum(ei * ei for ei in e) / max(1, len(e)))
        except Exception:
            return float("inf")

    def update(self, x: Any, y: Any, steps: int = 1) -> float:
        if HAS_NUMPY:
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            for _ in range(steps):
                y_hat = self.A @ x
                e = (y_hat - y).reshape(-1, 1)
                grad = 2.0 * (e @ x.reshape(1, -1)) + 2.0 * self.l2 * self.A
                self.A -= self.lr * grad.astype(np.float32)
            return self.loss(x, y)
        # pure-Python naive update: only adjust diagonal scale
        x_list = list(map(float, x))
        y_list = list(map(float, y))
        for _ in range(steps):
            y_hat = [self.A_diag[i] * x_list[i] for i in range(self.d)]
            for i in range(self.d):
                err = y_hat[i] - y_list[i]
                self.A_diag[i] -= self.lr * (
                    2.0 * err * x_list[i] + 2.0 * self.l2 * self.A_diag[i]
                )
        return self.loss(x_list, y_list)

    def compute_PU_and_update(self, prev_x: Any | None, curr_y: Any) -> float:
        if prev_x is None:
            return 0.0
        try:
            before = self.loss(prev_x, curr_y)
            _ = self.update(prev_x, curr_y, steps=1)
            after = self.loss(prev_x, curr_y)
            if before <= 1e-9:
                return 0.0
            pu = (before - after) / before
            return clamp(pu, 0.0, 1.0)
        except Exception:
            return 0.0


# --- Core Memory Data Structures ---


@dataclass
class Node:
    id: int
    kind: str
    embed: Any  # np.ndarray or list[float]
    data: dict[str, Any]
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
    Individual memory labyrinth: store, recall, dream-cycle and compact replay.

    Designed to be resilient if numpy is not available.
    """

    def __init__(
        self,
        config: AdamConfig,
        eva_manager: EVAMemoryManager | None = None,
        entity_id: str = "adam_default",
        d: int = 256,
        K: int = 8,
        temp_store: float = 0.7,
        tau_recency: float = 60.0,
        embedder: Callable[[Any], Any] | None = None,
        predictor: ARPredictor | None = None,
        max_nodes: int = 100_000,
        storage_pressure: float = 0.0,
    ):
        self.config = config
        self.eva_manager = eva_manager
        self.entity_id = str(entity_id)
        self.d = int(d)
        self.nodes: dict[int, Node] = {}
        self.next_id = 0
        self.K = int(K)
        self.temp_store = float(temp_store)
        self.tau = float(tau_recency)
        self.replay: deque = deque(maxlen=20_000)
        self.policy: list = []
        self.mode = "WAKE"

        self.embedder = embedder if embedder is not None else HashingEmbedder(d=self.d)
        self.predictor = predictor if predictor is not None else ARPredictor(d=self.d)
        self._last_observed: Any | None = None
        self.max_nodes = int(max_nodes)
        self.storage_pressure = float(storage_pressure)

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
        """
        Store a data item or an embedding. Returns node id.
        Persists a compact event to EVA via EVAMemoryManager if available (best-effort).
        """
        try:
            if HAS_NUMPY and isinstance(data_or_embed, np.ndarray):
                embed = data_or_embed.astype(np.float32)
                payload = {"raw": "embed"}
            else:
                embed = self.embedder(data_or_embed)
                if HAS_NUMPY and isinstance(embed, np.ndarray):
                    embed = embed.astype(np.float32)
                payload = {"raw": data_or_embed}
        except Exception:
            # fallback: hash string representation to a deterministic vector
            logger.exception("embedder failed; falling back to simple hash embed")
            h = int(
                hashlib.blake2b(
                    str(data_or_embed).encode("utf-8"), digest_size=8
                ).hexdigest(),
                16,
            )
            vec = [0.0] * self.d
            vec[h % self.d] = 1.0
            embed = np.array(vec, dtype=np.float32) if HAS_NUMPY else vec
            payload = {"raw": data_or_embed}

        now = now or time.time()
        sim_list = self._similar_nodes(embed)
        novelty = 1.0 - (sim_list[0][0] if sim_list else 0.0)
        A = clamp(0.5 * abs(valence) + 0.5 * arousal)
        S = clamp(surprise)

        if PU is None:
            PU = 0.0
            try:
                PU = self.predictor.compute_PU_and_update(self._last_observed, embed)
            except Exception:
                PU = 0.0
        PU = float(clamp(PU))

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

        # EVA persistence (best-effort, async-aware)
        if self.eva_manager is not None:
            try:
                record_fn = getattr(self.eva_manager, "record_experience", None)
                if record_fn:
                    payload_event = {
                        "node_id": nid,
                        "kind": kind,
                        "valence": float(valence),
                        "arousal": float(arousal),
                        "surprise": float(surprise),
                        "PU": float(PU),
                        "data_summary": str(data_or_embed)[:200],
                        "timestamp": time.time(),
                    }
                    if inspect.iscoroutinefunction(record_fn):
                        try:
                            # schedule background task if loop present
                            import asyncio

                            asyncio.create_task(
                                record_fn(
                                    entity_id=self.entity_id,
                                    event_type="memory_store",
                                    data=payload_event,
                                )
                            )
                        except Exception:
                            # run synchronously as fallback
                            try:
                                import asyncio

                                asyncio.run(
                                    record_fn(
                                        entity_id=self.entity_id,
                                        event_type="memory_store",
                                        data=payload_event,
                                    )
                                )
                            except Exception:
                                logger.debug("EVA async record failed silently")
                    else:
                        try:
                            record_fn(
                                entity_id=self.entity_id,
                                event_type="memory_store",
                                data=payload_event,
                            )
                        except Exception:
                            logger.debug("EVA sync record failed silently")
            except Exception:
                logger.exception("EVA record_experience integration failed (non-fatal)")

        return nid

    def recall(self, cue: Any, max_hops: int = 3) -> tuple[Any, list[int]] | None:
        """
        Recall associated embeddings / node ids given a cue.
        Returns (reconstructed_embedding, [node_ids]) or None when no nodes exist.
        """
        if not self.nodes:
            return None
        cue_embed = (
            cue if HAS_NUMPY and isinstance(cue, np.ndarray) else self.embedder(cue)
        )
        frontier = [(cos_sim(cue_embed, n.embed), nid) for nid, n in self.nodes.items()]
        frontier.sort(reverse=True)
        act: dict[int, float] = defaultdict(float)
        for s, nid in frontier[: self.K]:
            act[nid] += s
            self._touch(nid)
            for j, w in list(self.nodes[nid].edges_out.items()):
                act[j] += s * w
        items = sorted(act.items(), key=lambda x: x[1], reverse=True)[: self.K]
        if not items:
            return None
        # combine weighted vectors
        if HAS_NUMPY:
            vecs = [self.nodes[i].embed * w for i, w in items]
            rec_embed = np.mean(vecs, axis=0)
            nrm = float(np.linalg.norm(rec_embed))
            if nrm > 0:
                rec_embed = rec_embed / nrm
        else:
            # fallback average
            vecs = []
            for i, w in items:
                vec = (
                    list(self.nodes[i].embed)
                    if not isinstance(self.nodes[i].embed, list)
                    else self.nodes[i].embed
                )
                vecs.append([v * w for v in vec])
            L = len(vecs)
            if L == 0:
                return None
            rec = [sum(col) / L for col in zip(*vecs, strict=False)]
            nrm = math.sqrt(sum(x * x for x in rec))
            if nrm > 0:
                rec = [x / nrm for x in rec]
            rec_embed = rec

        ids = [i for i, _ in items]
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                self.nodes[i].edges_out[j] += 0.01

        # store the reconstructed narrative embedding as a lightweight node
        try:
            self.store(rec_embed, kind="narrative", PU=None)
        except Exception:
            logger.debug("Failed to store reconstructed recall (non-fatal)")

        return rec_embed, ids

    def dream_cycle(
        self,
        mode: str = "MIXED",
        steps_rem: int = 10,
        T_rem: float = 0.8,
        replay_k: int = 64,
        prune_threshold: float = 0.02,
        compress_similarity: float = 0.985,
        gen_prob: float = 0.15,
    ):
        """
        Execute a dream cycle (NREM/REM/MIXED). Persists a compact summary to EVA (best-effort).
        """
        try:
            if mode in ("NREM", "MIXED"):
                self._nrem_pass(
                    replay_k=replay_k, compress_similarity=compress_similarity
                )
            if mode in ("REM", "MIXED"):
                self._rem_pass(steps=steps_rem, T=T_rem, gen_prob=gen_prob)
            self._prune_links(th=prune_threshold)
            if len(self.nodes) > self.max_nodes:
                self._budget_prune(target=int(self.max_nodes * 0.95))
            # EVA integration (best-effort)
            if self.eva_manager is not None:
                try:
                    record_fn = getattr(self.eva_manager, "record_experience", None)
                    if record_fn:
                        data = {
                            "mode": mode,
                            "nodes_count": len(self.nodes),
                            "replay_count": len(self.replay),
                            "timestamp": time.time(),
                        }
                        if inspect.iscoroutinefunction(record_fn):
                            import asyncio

                            try:
                                asyncio.create_task(
                                    record_fn(
                                        entity_id=self.entity_id,
                                        event_type="dream_cycle",
                                        data=data,
                                    )
                                )
                            except Exception:
                                try:
                                    asyncio.run(
                                        record_fn(
                                            entity_id=self.entity_id,
                                            event_type="dream_cycle",
                                            data=data,
                                        )
                                    )
                                except Exception:
                                    pass
                        else:
                            try:
                                record_fn(
                                    entity_id=self.entity_id,
                                    event_type="dream_cycle",
                                    data=data,
                                )
                            except Exception:
                                pass
                except Exception:
                    logger.debug("EVA dream record non-fatal failure")
        except Exception:
            logger.exception("dream_cycle failed")

    # ---- Internal methods (optimized & robust) ----
    def _add_node(
        self,
        embed: Any,
        data: dict[str, Any],
        valence: float,
        arousal: float,
        kind: str,
        origin: str = "WAKE",
    ) -> int:
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(
            nid, kind, embed, data, float(valence), float(arousal), origin=origin
        )
        return nid

    def _link(self, i: int, j: int, w: float) -> None:
        self.nodes[i].edges_out[j] = max(self.nodes[i].edges_out.get(j, 0.0), float(w))
        self.nodes[j].edges_in[i] = max(self.nodes[j].edges_in.get(i, 0.0), float(w))

    def _reinforce(
        self, nid: int, valence: float, arousal: float, PU: float | None
    ) -> None:
        n = self.nodes[nid]
        A = clamp(0.5 * abs(valence) + 0.5 * arousal)
        n.salience = clamp(n.salience + 0.1 * A + 0.05 * (PU or 0.0))

    def _similar_nodes(self, embed: Any, top: int = 64) -> list[tuple[float, int]]:
        sims = []
        for nid, n in self.nodes.items():
            try:
                s = cos_sim(embed, n.embed)
            except Exception:
                s = 0.0
            sims.append((s, nid))
        sims.sort(reverse=True)
        return sims[:top]

    def _choose_anchors(
        self,
        sim_list: list[tuple[float, int]],
        A: float,
        PU: float,
        S: float,
        now: float,
    ) -> list[tuple[float, int]]:
        scores = []
        for sim_i, i in sim_list:
            n = self.nodes[i]
            R = math.exp(-(now - n.last_access) / self.tau)
            SC = 1.0 / (1.0 + len(n.edges_in))
            score = 0.4 * sim_i + 0.2 * A + 0.15 * PU + 0.15 * S + 0.05 * R + 0.05 * SC
            scores.append((score, sim_i, i))
        if not scores:
            return []
        xs = [s for s, _, _ in scores]
        T = max(self.temp_store, 1e-3)
        # stable softmax
        max_x = max(xs)
        exps = [math.exp((x - max_x) / T) for x in xs]
        total = sum(exps) or 1.0
        p = [e / total for e in exps]
        pick_count = min(self.K, len(scores))
        idxs = random.choices(range(len(scores)), weights=p, k=pick_count)
        # ensure unique
        chosen = []
        seen = set()
        for k in idxs:
            if k in seen:
                continue
            seen.add(k)
            chosen.append((scores[k][1], scores[k][2]))
            if len(chosen) >= pick_count:
                break
        return chosen

    def _push_replay(self, nid: int, A: float, S: float, PU: float) -> None:
        s = float(self.nodes[nid].salience if nid in self.nodes else 0.1)
        prio = s * (A + (S or 0) + (PU or 0) + 1e-3)
        self.replay.appendleft((prio, nid))
        self.replay = deque(
            sorted(self.replay, reverse=True, key=lambda x: x[0]),
            maxlen=self.replay.maxlen,
        )

    def _touch(self, nid: int) -> None:
        n = self.nodes[nid]
        n.last_access = time.time()
        n.freq += 1

    def _nrem_pass(
        self, replay_k: int = 64, compress_similarity: float = 0.985
    ) -> None:
        seeds = [nid for _, nid in list(self.replay)[:replay_k] if nid in self.nodes]
        for nid in seeds:
            path = self._greedy_temporal_path(nid, L=6)
            for i, j in zip(path, path[1:], strict=False):
                self.nodes[i].edges_out[j] += 0.02
                self.nodes[i].salience = clamp(self.nodes[i].salience + 0.02)
        for nid in seeds:
            cluster = self._k_hop(nid, hops=2, limit=64)
            self._compress_cluster(cluster, threshold=compress_similarity)

    def _rem_pass(
        self, steps: int = 10, T: float = 0.8, gen_prob: float = 0.15
    ) -> None:
        seeds = [nid for _, nid in list(self.replay)[: self.K] if nid in self.nodes]
        if not seeds and len(self.nodes) > 0:
            seeds = [random.choice(list(self.nodes.keys()))]
        for nid in seeds:
            walked = self._random_walk(nid, steps, T)
            if not walked:
                continue
            if HAS_NUMPY:
                vec = np.zeros(self.d, dtype=np.float32)
            else:
                vec = [0.0] * self.d
            total = 0.0
            for i in walked:
                w = self.nodes[i].salience + 0.1
                if HAS_NUMPY:
                    vec += w * (
                        self.nodes[i].embed
                        if isinstance(self.nodes[i].embed, np.ndarray)
                        else np.array(self.nodes[i].embed)
                    )
                else:
                    vec = [
                        vv + w * ev
                        for vv, ev in zip(vec, list(self.nodes[i].embed), strict=False)
                    ]
                total += w
            if total > 0:
                if HAS_NUMPY:
                    vec /= total
                    nrm = float(np.linalg.norm(vec))
                    if nrm > 0:
                        vec /= nrm
                else:
                    vec = [x / total for x in vec]
                    nrm = math.sqrt(sum(x * x for x in vec))
                    if nrm > 0:
                        vec = [x / nrm for x in vec]
            did = self._add_node(
                vec, {"dream_path": walked}, 0.0, 0.3, kind="dream", origin="REM"
            )
            for i in walked[-self.K :]:
                self._link(did, i, 0.2)
            self._push_replay(did, A=0.2, S=0.4, PU=0.2)

        if len(self.nodes) > 0 and random.random() < gen_prob:
            base = random.choice(list(self.nodes.values()))
            noise = (
                np.random.normal(0, 0.1, size=self.d).astype(np.float32)
                if HAS_NUMPY
                else [random.gauss(0, 0.1) for _ in range(self.d)]
            )
            if HAS_NUMPY:
                vec = base.embed + noise
                nrm = float(np.linalg.norm(vec))
                if nrm > 0:
                    vec /= nrm
            else:
                vec = [be + nv for be, nv in zip(list(base.embed), noise, strict=False)]
                nrm = math.sqrt(sum(x * x for x in vec))
                if nrm > 0:
                    vec = [x / nrm for x in vec]
            did = self._add_node(
                vec, {"dream_spark": base.id}, 0.0, 0.25, kind="dream", origin="REM"
            )
            self._push_replay(did, 0.2, 0.5, 0.2)

    def _greedy_temporal_path(self, start: int, L: int = 6) -> list[int]:
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

    def _k_hop(self, nid: int, hops: int = 2, limit: int = 128) -> list[int]:
        seen = {nid}
        frontier = [nid]
        for _ in range(hops):
            nxt: list[int] = []
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

    def _compress_cluster(self, ids: list[int], threshold: float = 0.985) -> None:
        if not ids:
            return
        ids = list(ids)
        base = ids[0]
        if base not in self.nodes:
            return
        base_node = self.nodes[base]
        to_remove: list[int] = []
        for j in ids[1:]:
            if j not in self.nodes:
                continue
            try:
                if cos_sim(base_node.embed, self.nodes[j].embed) > threshold:
                    for k, w in self.nodes[j].edges_out.items():
                        base_node.edges_out[k] = max(base_node.edges_out.get(k, 0.0), w)
                    for k, w in self.nodes[j].edges_in.items():
                        base_node.edges_in[k] = max(base_node.edges_in.get(k, 0.0), w)
                    base_node.salience = clamp(
                        base_node.salience + self.nodes[j].salience
                    )
                    to_remove.append(j)
            except Exception:
                continue
        for j in to_remove:
            if j in self.nodes:
                del self.nodes[j]

    def _random_walk(self, nid: int, steps: int, T: float) -> list[int]:
        cur = nid
        walked = [cur]
        for _ in range(steps):
            outs = self.nodes[cur].edges_out
            if not outs:
                break
            items = list(outs.items())
            logits = [w for _, w in items]
            # stable softmax sampling
            max_l = max(logits) if logits else 0.0
            exps = [math.exp((l - max_l) / max(T, 1e-3)) for l in logits]
            s = sum(exps) or 1.0
            probs = [e / s for e in exps]
            choices = [j for j, _ in items]
            cur = random.choices(choices, weights=probs, k=1)[0]
            walked.append(cur)
        return walked

    def _prune_links(self, th: float = 0.02) -> None:
        for n in self.nodes.values():
            to_del = [j for j, w in list(n.edges_out.items()) if w < th]
            for j in to_del:
                if j in n.edges_out:
                    del n.edges_out[j]

    def _budget_prune(self, target: int) -> None:
        if len(self.nodes) <= target:
            return
        now = time.time()
        scored: list[tuple[float, int]] = []
        for nid, n in self.nodes.items():
            age = now - n.last_access
            score_keep = (
                (0.6 * n.salience)
                + (0.2 * math.exp(-age / self.tau))
                + (0.2 * math.tanh(n.freq / 10))
            )
            score_keep -= 0.2 * self.storage_pressure
            scored.append((score_keep, nid))
        scored.sort()
        to_remove = [nid for _, nid in scored[: max(0, len(self.nodes) - target)]]
        for nid in to_remove:
            if nid not in self.nodes:
                continue
            for j in list(self.nodes[nid].edges_out.keys()):
                if j in self.nodes and nid in self.nodes[j].edges_in:
                    del self.nodes[j].edges_in[nid]
            for i in list(self.nodes[nid].edges_in.keys()):
                if i in self.nodes and nid in self.nodes[i].edges_out:
                    del self.nodes[i].edges_out[nid]
            del self.nodes[nid]


@dataclass
class EntityMemory:
    """
    Thin wrapper per-entity exposing convenient ingest/recall/dream integration.
    """

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
        seph = sum(
            v
            for k, v in lattice_vector.items()
            if k.lower()
            in (
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
            )
        )
        qlip = sum(
            v
            for k, v in lattice_vector.items()
            if k.lower()
            not in (
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
            )
        )
        self.valence_bias = clamp(self.valence_bias + 0.1 * seph, -0.5, 0.5)
        self.affect_bias = clamp(self.affect_bias + 0.1 * qlip, -0.2, 0.2)

    def on_qualia_update(self, qualia_state: dict[str, float]) -> None:
        v = clamp(qualia_state.get("emotional_valence", 0.0), -1.0, 1.0)
        a = clamp(qualia_state.get("consciousness_density", 0.0), 0.0, 1.0)
        self.valence_bias = 0.7 * self.valence_bias + 0.3 * v
        self.affect_bias = 0.7 * self.affect_bias + 0.3 * a
