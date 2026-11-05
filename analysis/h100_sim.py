"""Simulated copies for every arm of ``interleaved_h100.py`` (T5-T7), with or without next-layer prefetching.

``prefetch_sim.run_on`` covers decayed LFU and split-phase only. This module replays any arm the H100 runner
decodes, event by event as the runtime does (``prefetch_sim``'s loop): decayed LFU, HOBBIT's and MoE-Infinity's
rules (``policy_baselines``), split-phase (frozen P2/P1 rankers) and the synchronous ranker (``SplitPhase`` with
fresh features, as in ``unseen_sim.py``), prefetching with the run's own guesses.

    from h100_sim import run_arm
    per_sequence = run_arm(trace, path, "hobbit+pf", capacity=32, guesses=live, prefetch=(8, 2), params=...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_arm(base: str, capacity: int, L: int, E: int, k: int, params: dict):
    """(arm, is_simulator_class). ``params``: p2, p1, sync (bundle paths), hobbit_w, eams, online, alpha,
    wrong_rule (the rule for unrequested prefetched experts under the simple policies)."""

    if base == "lfu_decay":
        from policy import LFU_DECAY_ALPHA, SimplePolicy

        arm = SimplePolicy("lfu_decay", L, E, capacity, k, alpha=params.get("alpha", LFU_DECAY_ALPHA))
        arm.wrong_guess_rule = params.get("wrong_rule", "last")
        return arm, False
    if base == "hobbit":
        from policy_baselines import HobbitPolicy

        arm = HobbitPolicy(L, E, capacity, k, w=float(params["hobbit_w"]))
        arm.wrong_guess_rule = params.get("wrong_rule", "last")
        return arm, False
    if base == "moe_infinity":
        from policy_baselines import MoEInfinityPolicy

        arm = MoEInfinityPolicy(L, E, capacity, k, store=params["eams"], online=params.get("online", True))
        arm.wrong_guess_rule = params.get("wrong_rule", "last")
        return arm, False
    from splitphase import SplitPhase, load_bundle

    if base == "splitphase":
        p2 = load_bundle(str(params["p2"]))
        return SplitPhase(p2, capacity, "lag_req", "none", diagnose=False, mix_layer0=True, lag_bundle=p2,
                          lag_tok_bundle=load_bundle(str(params["p1"]))), True
    if base == "global_token_gbt":
        return SplitPhase(load_bundle(str(params["sync"])), capacity, "fresh", "none", diagnose=False), True
    raise ValueError(f"unknown arm base {base!r}")


def run_arm(trace, deploy, arm_name: str, capacity: int, params: dict, guesses=None, prefetch=(0, 0)) -> list[dict]:
    """Per-sequence demand copies, prefetch copies and prefetch hits of one arm replayed on ``deploy``."""

    base, _, suffix = arm_name.partition(":")[0].partition("+")
    L, E, k = int(trace.num_layers), int(trace.num_experts), int(trace.top_k)
    arm, is_sim = make_arm(base, capacity, L, E, k, params)
    m, q = prefetch if suffix == "pf" else (0, 0)
    if suffix == "pf" and guesses is None:
        raise ValueError(f"{arm_name} needs the run's guesses")
    l2o = {layer: i for i, layer in enumerate(map(int, trace.metadata["layer_indices"]))}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    out = []
    for sequence, view in deploy.iter_slices(trace):
        sid = int(sequence.source_sequence_id)
        arm.begin_sequence()
        before = (arm.transfers, arm.prefetch_transfers, arm.prefetch_hits)
        for index in range(view.start, view.stop):
            layer = l2o[int(trace.layer_index[index])]
            token = int(trace.token_id[index])
            arm.step(layer, token, requests[index], gates[index])
            if q and layer + 1 < L:
                g = guesses.get(sid, (index - view.start) // L, layer + 1, m)
                if is_sim:
                    arm.prefetch(layer + 1, token, g, q)
                else:
                    arm.prefetch(layer + 1, g, q)
        after = (arm.transfers, arm.prefetch_transfers, arm.prefetch_hits)
        out.append({"source_sequence_id": sid, "demand": after[0] - before[0], "prefetch": after[1] - before[1],
                    "hits": after[2] - before[2]})
    return out


class LiveGuesses:
    """The run's own guesses (``<run>.guesses.npz``: key ``<arm>/<sequence id>``, shaped [tokens, layers, m])."""

    def __init__(self, path: Path, arm: str):
        data = np.load(path)
        self.by_sequence = {int(key.split("/")[1]): data[key].astype(np.int64) for key in data.files
                            if key.startswith(f"{arm}/")}

    def get(self, sid: int, token: int, layer: int, m: int) -> np.ndarray:
        return self.by_sequence[sid][token, layer, :m]
