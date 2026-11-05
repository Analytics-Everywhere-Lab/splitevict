"""Checks for the cross-model feature state (``crossmodel.py``) and the worker import order.

These need the vendored trace code (``measurement/``, or the path in ``RACE_REPOSITORY_ROOT``);
they are skipped without it.
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE = ROOT / "src"
sys.path.insert(0, str(CODE))
REPOSITORY = pathlib.Path(
    os.environ.get("RACE_REPOSITORY_ROOT", ROOT / "measurement")
).expanduser()


def repository_available() -> bool:
    return (REPOSITORY / "stage3_residency/src/residency_headroom").exists()


class _FakeModels:
    """Row-stochastic transition matrices with the ``matrix(h, l)`` interface."""

    def __init__(self, num_layers: int, num_experts: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._m = rng.random((num_layers, 6, num_experts, num_experts))
        self._m /= self._m.sum(axis=-1, keepdims=True)

    def matrix(self, horizon: int, layer: int) -> np.ndarray:
        from features2 import HORIZONS

        return self._m[layer, HORIZONS.index(horizon)]


def replay(state_type, num_layers, num_experts, top_k, steps, seed=1):
    from features3 import FeatureState3  # noqa: F401  (path check)

    rng = np.random.default_rng(seed)
    static = rng.random((num_layers, num_experts))
    state = state_type(_FakeModels(num_layers, num_experts), num_layers, num_experts, static)
    state.reset()
    state.begin_sequence()
    rows = []
    position = np.zeros(num_layers, np.int64)
    for _ in range(steps):
        layer = int(rng.integers(num_layers))
        request = rng.choice(num_experts, size=top_k, replace=False)
        gates = rng.random(top_k)
        rows.append(
            state.features(layer, request, gates, np.sort(request), int(position[layer])).copy()
        )
        state.absorb(layer, request, gates, int(position[layer]))
        position[layer] += 1
    return np.stack(rows)


@unittest.skipUnless(repository_available(), "measurement code not available")
class FeatureStateGeneralisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from run_caer import configure_repository

        configure_repository(REPOSITORY)

    def test_top_k_8_reproduces_the_frozen_feature_state(self) -> None:
        from crossmodel import feature_state_class
        from features3 import FeatureState3

        frozen = replay(FeatureState3, 3, 64, 8, 400)
        generalised = replay(feature_state_class(8), 3, 64, 8, 400)
        np.testing.assert_array_equal(generalised, frozen)

    def test_other_top_k_changes_only_the_two_rate_scale_features(self) -> None:
        from crossmodel import feature_state_class
        from features3 import NAMES, FeatureState3

        frozen = replay(FeatureState3, 2, 8, 2, 300)
        generalised = replay(feature_state_class(2), 2, 8, 2, 300)
        differs = np.any(frozen != generalised, axis=(0, 2))
        changed = {NAMES[i] for i in np.flatnonzero(differs)}
        self.assertEqual(changed, {"request_rate_shrunk", "request_rate_minus_static"})
        minus = NAMES.index("request_rate_minus_static")
        # rate - static*2 versus rate - static*8: the difference is exactly 6*static
        delta = generalised[:, minus, :] - frozen[:, minus, :]
        self.assertTrue(np.all(delta > 0))


if __name__ == "__main__":
    unittest.main()
