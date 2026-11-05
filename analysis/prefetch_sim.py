"""Decayed LFU and split-phase eviction with next-layer prefetching, in the simulator (T4 development).

Event by event, as the runtime does it: at layer l of token t the layer's own request is admitted
(demand misses: copies on the critical path), then up to ``q`` of the top-``m`` guessed experts of
layer l+1 that are not resident are prefetched into the slots of the policy's lowest-ranked
residents outside the guess (``policy.CachePolicy.prefetch``; split-phase ranks by the scores the
worker published for layer l+1's next event). A prefetched expert that layer l+1's request does
not use ranks last at that event. Layer 0 is never prefetched (no earlier layer in the token).

Guesses come from ``src/runtime/record_guesses.py`` (layer l+1's router on layer l's MoE input).

    python analysis/prefetch_sim.py --guesses data/guesses/olmoe_calibration_a40.npz \
        --capacity 32 --policy lfu --variant g1 --m 8 --q 2 --output prefetch_lfu_B32.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BUNDLES = ROOT / "data" / "rankers"
DEV = {"p2": BUNDLES / "olmoe_dev_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_dev_lag_tok_trees200.pkl"}
FROZEN = {"p2": BUNDLES / "olmoe_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_frozen_lag_tok_trees200.pkl"}


class Guesses:
    def __init__(self, path: Path, variant: str):
        data = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(data["meta"]))
        self.variant = variant
        self.by_sequence: dict[int, np.ndarray] = {}
        self.requests: dict[int, np.ndarray] = {}
        for key in data.files:
            if key.startswith("request/"):
                self.requests[int(key.split("/")[1])] = data[key].astype(np.int64)
        if variant == "prev":
            for sid, req in self.requests.items():
                g = np.full((req.shape[0], req.shape[1], req.shape[2]), -1, dtype=np.int64)
                g[1:, :, :] = req[:-1, :, :]  # layer l's request at the previous token
                g[:, 0, :] = -1
                self.by_sequence[sid] = g
        else:
            for key in data.files:
                if key.startswith(f"{variant}/"):
                    self.by_sequence[int(key.split("/")[1])] = data[key].astype(np.int64)

    def get(self, sid: int, token: int, layer: int, m: int) -> np.ndarray:
        return self.by_sequence[sid][token, layer, :m]


def make_arm(policy: str, capacity: int, bundles: dict, L: int, E: int, k: int, wrong: str):
    if policy in ("lfu", "lru"):
        from policy import SimplePolicy

        arm = SimplePolicy("lfu_decay" if policy == "lfu" else "lru", L, E, capacity, k)
        arm.wrong_guess_rule = wrong
        return arm, False
    from splitphase import SplitPhase, load_bundle

    p2 = load_bundle(str(bundles["p2"]))
    arm = SplitPhase(p2, capacity, "lag_req", "none", diagnose=False, mix_layer0=True, lag_bundle=p2,
                     lag_tok_bundle=load_bundle(str(bundles["p1"])))
    return arm, True


_DEV = None


def _load_dev():
    global _DEV
    if _DEV is None:
        from splitphase import load_dev

        _DEV = load_dev()
    return _DEV


def run(policy: str, capacity: int, guesses: Guesses | None, m: int, q: int, fold: str, bundle_set: str,
        wrong: str = "last", limit: int | None = None) -> dict:
    inputs, fit, deploy = _load_dev()
    if fold == "calibration":
        deploy = inputs.calibration
    elif fold == "calA":
        deploy = fit
    bundles = DEV if bundle_set == "dev" else FROZEN
    result = run_on(inputs.trace, deploy, policy, capacity, guesses, m, q, bundles, wrong, limit)
    result.update({"fold": deploy.name, "bundles": bundle_set})
    return result


def run_on(trace, deploy, policy: str, capacity: int, guesses, m: int, q: int, bundles: dict,
           wrong: str = "last", limit: int | None = None) -> dict:
    """Replay ``deploy`` (anything with ``iter_slices(trace)``) on ``trace`` with one arm."""

    L, E, k = int(trace.num_layers), int(trace.num_experts), int(trace.top_k)
    arm, is_sim = make_arm(policy, capacity, bundles, L, E, k, wrong)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    per_sequence = []
    hist_pf = np.zeros(q + 1 if guesses is not None else 1, dtype=np.int64)
    start = time.perf_counter()
    for count, (sequence, view) in enumerate(deploy.iter_slices(trace)):
        if limit is not None and count >= limit:
            break
        sid = int(sequence.source_sequence_id)
        arm.begin_sequence()
        before = (arm.transfers, arm.prefetch_transfers, arm.prefetch_hits, arm.prefetch_unused,
                  arm.prefetch_evicted_requested)
        for index in range(view.start, view.stop):
            layer = l2o[int(trace.layer_index[index])]
            token = int(trace.token_id[index])
            arm.step(layer, token, requests[index], gates[index])
            if guesses is not None and layer + 1 < L:
                t = (index - view.start) // L
                g = guesses.get(sid, t, layer + 1, m)
                if is_sim:
                    _ev, loaded = arm.prefetch(layer + 1, token, g, q)
                else:
                    _ev, loaded = arm.prefetch(layer + 1, g, q)
                hist_pf[min(int(loaded.size), hist_pf.size - 1)] += 1
        after = (arm.transfers, arm.prefetch_transfers, arm.prefetch_hits, arm.prefetch_unused,
                 arm.prefetch_evicted_requested)
        d = [a - b for a, b in zip(after, before)]
        tokens = (view.stop - view.start) // L
        per_sequence.append({"source_sequence_id": sid, "position": int(sequence.position), "tokens": tokens,
                             "demand": d[0], "prefetch": d[1], "hits": d[2], "unused": d[3], "evicted_requested": d[4]})
    tokens = sum(r["tokens"] for r in per_sequence)
    totals = {key: sum(r[key] for r in per_sequence) for key in ("demand", "prefetch", "hits", "unused", "evicted_requested")}
    return {
        "policy": policy, "capacity": capacity, "fold": getattr(deploy, "name", None),
        "bundles": {key: str(value) for key, value in bundles.items()},
        "prefetch": None if guesses is None else {"variant": guesses.variant, "m": m, "q": q, "wrong_rule": wrong,
                                                  "guess_file": str(guesses.meta.get("spec")), "guess_device": guesses.meta.get("device")},
        "sequences": len(per_sequence), "tokens": tokens, "totals": totals,
        "per_token": {key: value / max(tokens, 1) for key, value in totals.items()},
        "prefetches_per_layer_event_histogram": hist_pf.tolist(),
        "wall_seconds": time.perf_counter() - start,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guesses", type=Path)
    parser.add_argument("--variant", default="g1", choices=("g1", "g1n", "prev"))
    parser.add_argument("--policy", required=True, choices=("lfu", "lru", "splitphase"))
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--wrong", default="last", choices=("last", "policy"))
    parser.add_argument("--fold", default="calB", choices=("calA", "calB", "calibration"))
    parser.add_argument("--bundles", default="dev", choices=("dev", "frozen"))
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guesses = None if args.no_prefetch else Guesses(args.guesses, args.variant)
    result = run(args.policy, args.capacity, guesses, args.m, args.q, args.fold, args.bundles, args.wrong, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    pt = result["per_token"]
    print(f"{args.policy} B={args.capacity} pf={result['prefetch']}: demand {pt['demand']:.2f}/tok, prefetch "
          f"{pt['prefetch']:.2f}/tok, hits {pt['hits']:.2f}, unused {pt['unused']:.2f}, victims requested "
          f"{pt['evicted_requested']:.2f} [{result['wall_seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
