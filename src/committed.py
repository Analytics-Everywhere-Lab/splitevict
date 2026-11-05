from __future__ import annotations

import json

import numpy as np

from harness import ROOT

SELECTION = ROOT / "stage3_residency/stage3_ranking/results/calibration/selection.json"


def check_feature_order(numerical: bool = True) -> None:

    from features3 import NAMES, FeatureState3
    from race_stage3.features import ALL_NAMES, FeatureState

    if tuple(NAMES) != tuple(ALL_NAMES):
        mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(NAMES, ALL_NAMES)) if a != b]
        raise SystemExit(
            f"Exploration and frozen feature names diverged at {mismatch[:5]}; the "
            "committed seeds cannot be reused."
        )
    if not numerical:
        return
    from harness import load
    from race_stage3.features import static_popularity

    inputs, models = load()
    popularity = static_popularity(inputs.trace, inputs.calibration)
    layers, experts = inputs.trace.num_layers, inputs.trace.num_experts
    left = FeatureState3(models, layers, experts, popularity)
    right = FeatureState(models, layers, experts, popularity)
    requests = inputs.trace.requested_expert_ids.astype(np.int64)
    gates = inputs.trace.router_weights.astype(np.float64)
    for step in range(64):
        index = step * inputs.trace.num_layers
        request, gate = requests[index], gates[index]
        order = np.sort(request)
        a = left.features(0, request, gate, order, step)
        b = right.features(0, request, gate, order, step)
        if not np.array_equal(a, b):
            raise SystemExit(
                f"Exploration and frozen feature values diverged at step {step}; the "
                "committed seeds cannot be reused."
            )
        left.absorb(0, request, gate, step)
        right.absorb(0, request, gate, step)


def seed_weights() -> np.ndarray:

    check_feature_order()
    selection = json.loads(SELECTION.read_text())
    return np.asarray(selection["rounds"]["all"]["pooled"]["weights"], dtype=np.float64)
