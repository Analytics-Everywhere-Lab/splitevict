"""T5 development: the eviction rules of HOBBIT and MoE-Infinity against decayed LFU, in the simulator.

Development data only: the original trace's calibration split, fitted on calA (sequences 0-39), measured on
calB (40-79), one cold cache carried across calB's sequences in order (as every development number).

* ``hobbit``: the recency weight w is chosen on calA (grid), then applied to calB;
* ``moe_infinity``: the store holds calA's request EAMs; each finished calB sequence is added (online);
* references: LRU, LFU, decayed LFU (alpha 0.95), and on OLMoE split-phase with the development rankers.

    python analysis/baselines_dev.py --model olmoe --capacities 32 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import splitphase  # noqa: E402  (load_dev sets up the import paths to measurement/)

PREPARED = {"qwen3": ROOT / "data/prepared/qwen3_30b_a3b",
            "mixtral": ROOT / "data/prepared/mixtral_8x7b"}
W_GRID = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))


def replay(trace, workload, arm) -> int:
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    before = arm.transfers
    for _sequence, view in workload.iter_slices(trace):
        arm.begin_sequence()
        for index in range(view.start, view.stop):
            arm.step(l2o[int(trace.layer_index[index])], int(trace.token_id[index]), requests[index], gates[index])
    return int(arm.transfers - before)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("olmoe", "qwen3", "mixtral"), required=True)
    parser.add_argument("--capacities", type=int, nargs="+", required=True)
    parser.add_argument("--splitphase", action="store_true", help="OLMoE: also split-phase (development rankers)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model != "olmoe":
        splitphase.PREPARED_ROOT = PREPARED[args.model].resolve()
    inputs, fit, deploy = splitphase.load_dev()
    trace = inputs.trace
    from policy import SimplePolicy
    from policy_baselines import HobbitPolicy, MoEInfinityPolicy, request_eams

    L, E, k = int(trace.num_layers), int(trace.num_experts), int(trace.top_k)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    store = request_eams(trace, fit, l2o)
    out = {"model": args.model, "L": L, "E": E, "k": k, "fit": fit.name, "deploy": deploy.name,
           "tokens": int(sum((v.stop - v.start) // L for _s, v in deploy.iter_slices(trace))), "capacities": {}}
    for B in args.capacities:
        start = time.perf_counter()
        res = {}
        for name in ("lru", "lfu", "lfu_decay"):
            res[name] = replay(trace, deploy, SimplePolicy(name, L, E, B, k))
        fit_w = {w: replay(trace, fit, HobbitPolicy(L, E, B, k, w=w)) for w in W_GRID}
        w_best = min(fit_w, key=lambda w: (fit_w[w], w))
        res["hobbit"] = replay(trace, deploy, HobbitPolicy(L, E, B, k, w=w_best))
        res["moe_infinity"] = replay(trace, deploy, MoEInfinityPolicy(L, E, B, k, store=store, online=True))
        res["moe_infinity_static"] = replay(trace, deploy, MoEInfinityPolicy(L, E, B, k, store=store, online=False))
        if args.splitphase and args.model == "olmoe":
            from prefetch_sim import DEV, make_arm

            arm, _ = make_arm("splitphase", B, DEV, L, E, k, "last")
            res["splitphase"] = replay(trace, deploy, arm)
        base = res["lfu_decay"]
        out["capacities"][str(B)] = {"transfers": res, "hobbit_w": w_best, "hobbit_fit_grid": fit_w,
                                     "vs_lfu_decay": {n: (v - base) / base for n, v in res.items()}}
        print(f"{args.model} B={B} [{time.perf_counter() - start:.0f}s] hobbit w={w_best}", flush=True)
        for name, v in res.items():
            print(f"  {name:20s} {v:9,d}  {100 * (v - base) / base:+6.2f}% vs decayed LFU  "
                  f"{v / out['tokens']:.2f}/token", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
