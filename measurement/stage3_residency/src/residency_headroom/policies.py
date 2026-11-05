from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CacheTransition:
    request: frozenset[int]
    before: frozenset[int]
    after: frozenset[int]
    hits: frozenset[int]
    misses: frozenset[int]
    admissions: frozenset[int]
    evictions: frozenset[int]


class AtomicCachePolicy:
    """One per-layer cache under mandatory atomic-request admission semantics."""

    name = "base"

    def __init__(self, capacity: int, num_experts: int) -> None:
        if capacity < 0:
            raise ValueError("Cache capacity cannot be negative")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if capacity > num_experts:
            raise ValueError("Cache capacity exceeds the layer's expert count")
        self.capacity = int(capacity)
        self.num_experts = int(num_experts)
        self.resident: frozenset[int] = frozenset()
        self.clock = 0

    def process(self, request: Iterable[int]) -> CacheTransition:
        requested = frozenset(int(item) for item in request)
        self._validate_request(requested)
        before = self.resident
        hits = requested & before
        misses = requested - before
        if self.capacity == 0:
            after = frozenset()
        else:
            self.clock += 1
            self._observe(requested)
            spare = self.capacity - len(requested)
            old = before - requested
            retained = self._choose_old(old, spare)
            after = requested | retained
        admissions = after - before
        evictions = before - after
        if self.capacity > 0 and admissions != misses:
            raise RuntimeError(
                f"{self.name} violated mandatory admission: admissions={admissions}, "
                f"misses={misses}"
            )
        if len(after) > self.capacity:
            raise RuntimeError(f"{self.name} exceeded cache capacity")
        if self.capacity > 0 and not requested.issubset(after):
            raise RuntimeError(f"{self.name} dropped a member of the atomic request")
        self.resident = frozenset(after)
        return CacheTransition(
            request=requested,
            before=before,
            after=self.resident,
            hits=hits,
            misses=misses,
            admissions=admissions,
            evictions=evictions,
        )

    def _validate_request(self, request: frozenset[int]) -> None:
        if not request:
            raise ValueError("Atomic requests cannot be empty")
        if min(request) < 0 or max(request) >= self.num_experts:
            raise ValueError("Atomic request contains an out-of-range expert")
        if self.capacity > 0 and len(request) > self.capacity:
            raise ValueError(
                f"Atomic request of {len(request)} experts exceeds capacity {self.capacity}"
            )

    def _observe(self, request: frozenset[int]) -> None:
        del request

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        raise NotImplementedError


class LRUPolicy(AtomicCachePolicy):
    name = "lru"

    def __init__(self, capacity: int, num_experts: int) -> None:
        super().__init__(capacity, num_experts)
        self.last_used = np.full(num_experts, -1, dtype=np.int64)

    def _observe(self, request: frozenset[int]) -> None:
        for expert in request:
            self.last_used[expert] = self.clock

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        ranked = sorted(old, key=lambda expert: (-int(self.last_used[expert]), expert))
        return frozenset(ranked[:count])


class LFUPolicy(AtomicCachePolicy):
    name = "lfu"

    def __init__(self, capacity: int, num_experts: int) -> None:
        super().__init__(capacity, num_experts)
        self.frequency = np.zeros(num_experts, dtype=np.int64)

    def _observe(self, request: frozenset[int]) -> None:
        for expert in request:
            self.frequency[expert] += 1

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        ranked = sorted(old, key=lambda expert: (-int(self.frequency[expert]), expert))
        return frozenset(ranked[:count])


class DecayedLFUPolicy(AtomicCachePolicy):
    name = "lfu_decay"

    def __init__(self, capacity: int, num_experts: int, alpha: float) -> None:
        super().__init__(capacity, num_experts)
        if not 0.0 < alpha < 1.0:
            raise ValueError("LFU-decay alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.frequency = np.zeros(num_experts, dtype=np.float64)

    def _observe(self, request: frozenset[int]) -> None:
        self.frequency *= self.alpha
        for expert in request:
            self.frequency[expert] += 1.0

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        ranked = sorted(old, key=lambda expert: (-float(self.frequency[expert]), expert))
        return frozenset(ranked[:count])


class StaticHotsetPolicy(AtomicCachePolicy):
    name = "static_hotset"

    def __init__(self, capacity: int, num_experts: int, scores: np.ndarray) -> None:
        super().__init__(capacity, num_experts)
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (num_experts,):
            raise ValueError(
                f"Static scores have shape {values.shape}; expected {(num_experts,)}"
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Static scores must be finite and nonnegative")
        self.scores = values.copy()

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        ranked = sorted(old, key=lambda expert: (-float(self.scores[expert]), expert))
        return frozenset(ranked[:count])


class RandomPolicy(AtomicCachePolicy):
    name = "random"

    def __init__(self, capacity: int, num_experts: int, seed: int, layer: int = 0) -> None:
        super().__init__(capacity, num_experts)
        self.seed = int(seed)
        self.rng = np.random.default_rng([self.seed, int(layer)])

    def _choose_old(self, old: frozenset[int], count: int) -> frozenset[int]:
        ordered = np.asarray(sorted(old), dtype=np.int64)
        if count >= len(ordered):
            return frozenset(map(int, ordered))
        if count <= 0:
            return frozenset()
        chosen = self.rng.choice(ordered, size=count, replace=False)
        return frozenset(map(int, chosen))


def make_policy(
    name: str,
    capacity: int,
    num_experts: int,
    *,
    alpha: float | None = None,
    seed: int | None = None,
    layer: int = 0,
    static_scores: np.ndarray | None = None,
) -> AtomicCachePolicy:
    if name == "lru":
        return LRUPolicy(capacity, num_experts)
    if name == "lfu":
        return LFUPolicy(capacity, num_experts)
    if name == "lfu_decay":
        if alpha is None:
            raise ValueError("lfu_decay requires alpha")
        return DecayedLFUPolicy(capacity, num_experts, alpha)
    if name == "static_hotset":
        if static_scores is None:
            raise ValueError("static_hotset requires calibration scores")
        return StaticHotsetPolicy(capacity, num_experts, static_scores)
    if name == "random":
        if seed is None:
            raise ValueError("random policy requires a fixed seed")
        return RandomPolicy(capacity, num_experts, seed, layer=layer)
    raise ValueError(f"Unknown cache policy {name!r}")
