"""Checks of the published-system eviction rules (``src/runtime/policy_baselines.py``); no GPU needed.

* HOBBIT's rule with the recency weight at 1 is LRU, decision for decision.
* MoE-Infinity's incremental cosine match equals a match computed from scratch at every event, and the
  online store grows by one request per finished sequence.
* Both rules take prefetches through ``CachePolicy.prefetch`` and keep the cache's accounting.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/runtime", "src"):
    value = str(ROOT / relative)
    if value not in sys.path:
        sys.path.append(value)

from policy import SimplePolicy  # noqa: E402
from policy_baselines import HobbitPolicy, MoEInfinityPolicy  # noqa: E402

L, E, K = 3, 12, 2


def stream(seed: int, sequences: int = 4, tokens: int = 30):
    """Skewed random requests, a different hot set per sequence (as domains change between sequences)."""

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(sequences):
        weights = rng.dirichlet(np.full(E, 0.3), size=L)
        seq = []
        for _t in range(tokens):
            for layer in range(L):
                seq.append((layer, np.sort(rng.choice(E, size=K, replace=False, p=weights[layer]))))
        out.append(seq)
    return out


def replay(policy, sequences, prefetch_rng=None):
    evictions = []
    for seq in sequences:
        policy.begin_sequence()
        for layer, request in seq:
            result = policy.step(layer, 0, request, np.ones(K))
            evictions.append(tuple(result.evicted))
            if prefetch_rng is not None and layer + 1 < L:
                guess = prefetch_rng.choice(E, size=4, replace=False)
                policy.prefetch(layer + 1, guess, 2)
                assert int(policy.resident[layer + 1].sum()) <= policy.capacity
    return evictions


class BaselinePolicyTest(unittest.TestCase):
    def test_hobbit_with_recency_weight_one_is_lru(self) -> None:
        for seed in range(3):
            for capacity in (3, 6, 9):
                seqs = stream(seed)
                lru = SimplePolicy("lru", L, E, capacity, K)
                hobbit = HobbitPolicy(L, E, capacity, K, w=1.0)
                self.assertEqual(replay(lru, seqs), replay(hobbit, seqs))
                self.assertEqual(lru.transfers, hobbit.transfers)

    def test_moe_infinity_incremental_match_equals_scratch(self) -> None:
        rng = np.random.default_rng(7)
        store = rng.integers(0, 5, size=(6, L, E)).astype(np.float64)
        policy = MoEInfinityPolicy(L, E, 6, K, store=store, online=True)
        seqs = stream(3, sequences=3, tokens=12)
        for s, seq in enumerate(seqs):
            policy.begin_sequence()
            self.assertEqual(len(policy.store), 6 + s)
            eam = np.zeros((L, E))
            for layer, request in seq:
                policy.step(layer, 0, request, np.ones(K))
                eam[layer, request] += 1.0
                flat = np.stack(policy.store).reshape(len(policy.store), -1)
                cos = flat @ eam.ravel() / (np.linalg.norm(eam) * np.maximum(np.linalg.norm(flat, axis=1), 1e-12))
                self.assertEqual(policy.match_index, int(np.argmax(cos)))
                np.testing.assert_allclose(policy.eam, eam)
        static = MoEInfinityPolicy(L, E, 6, K, store=store, online=False)
        replay(static, seqs)
        self.assertEqual(len(static.store), 6)

    def test_rules_take_prefetches(self) -> None:
        rng = np.random.default_rng(11)
        store = rng.integers(0, 5, size=(4, L, E)).astype(np.float64)
        for policy in (HobbitPolicy(L, E, 6, K, w=0.2), MoEInfinityPolicy(L, E, 6, K, store=store)):
            replay(policy, stream(5), prefetch_rng=np.random.default_rng(1))
            self.assertGreater(policy.prefetch_transfers, 0)
            self.assertEqual(policy.events, 4 * 30 * L)
            self.assertTrue(all(int(policy.resident[layer].sum()) <= 6 for layer in range(L)))


if __name__ == "__main__":
    unittest.main()
