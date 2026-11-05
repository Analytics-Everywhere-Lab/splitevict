"""Exploratory (not pre-registered): the published-system rules on the T1, T3 and T4 traces, in the simulator.

The traces' evaluation splits are already used for confirmatory claims; this only places HOBBIT's and MoE-Infinity's
rules (fitted on calibration data alone: ``baselines_fit.py``, ``frozen``) next to decayed LFU and split-phase,
whose transfers on the same paths are already on record (simulator = hardware). One cold cache per path.

* ``--trace t1`` (T1, OLMoE): the ten workload paths of ``data/traces/olmoe_t1`` (as ``unseen_sim.py``);
* ``--trace t4`` (T4, OLMoE), ``t3_qwen3`` and ``t3_mixtral`` (T3): the replay spec's one path.

    python analysis/baselines_unseen.py --trace t4
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
OPT = Path(__file__).resolve().parents[1] / "measurement"

TRACES = {  # name: (model, trace root, spec or None, capacities)
    "t1": ("olmoe", ROOT / "data/traces/olmoe_t1", None, (32, 24)),
    "t4": ("olmoe", ROOT / "data/traces/olmoe_t4", ROOT / "data/specs/olmoe_t4_replay_spec.json", (32, 24)),
    "t3_qwen3": ("qwen3", ROOT / "data/traces/qwen3_30b_a3b_t3", ROOT / "data/specs/qwen3_30b_a3b_t3_replay_spec.json",
                  (64, 48)),
    "t3_mixtral": ("mixtral", ROOT / "data/traces/mixtral_8x7b_t3",
                    ROOT / "data/specs/mixtral_8x7b_t3_replay_spec.json", (4, 3)),
}


def replay_path(trace, workload, arm, l2o, requests, gates) -> int:
    before = arm.transfers
    for _sequence, view in workload.iter_slices(trace):
        arm.begin_sequence()
        for index in range(view.start, view.stop):
            arm.step(l2o[int(trace.layer_index[index])], int(trace.token_id[index]), requests[index], gates[index])
    return int(arm.transfers - before)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", choices=tuple(TRACES), required=True)
    args = parser.parse_args()
    model, root, spec_path, caps = TRACES[args.trace]
    from fresh_trace import _paths, load_fresh
    from unseen_sim import SpecPath

    if spec_path is None:
        inputs = load_fresh(OPT, root, ROOT / "data/prepared/olmoe_t1")
        trace, workloads = inputs.trace, list(inputs.workloads)
    else:
        _paths(OPT)
        from residency_headroom.trace import RoutingTrace

        trace = RoutingTrace.load(root / "routing_trace.npz", validate=True)
        spec = json.loads(spec_path.read_text())
        if spec["trace_hash"] != trace.trace_hash:
            raise SystemExit("spec and trace differ")
        workloads = [SpecPath(spec, trace)]
    from policy import SimplePolicy
    from policy_baselines import HobbitPolicy, MoEInfinityPolicy

    fit = json.loads((ROOT / f"data/baselines/{model}_frozen.json").read_text())
    store = np.load(ROOT / fit["eams"]["path"])
    L, E, k = int(trace.num_layers), int(trace.num_experts), int(trace.top_k)
    l2o = {layer: i for i, layer in enumerate(map(int, trace.metadata["layer_indices"]))}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    out = {"trace": args.trace, "model": model, "trace_hash": trace.trace_hash, "fit": fit["fit"], "capacities": {}}
    for B in caps:
        start = time.perf_counter()
        w = fit["hobbit"][str(B)]["w"]
        arms = {"lfu_decay": lambda: SimplePolicy("lfu_decay", L, E, B, k),
                "lru": lambda: SimplePolicy("lru", L, E, B, k),  # Mixtral-offloading's per-layer rule
                "hobbit": lambda: HobbitPolicy(L, E, B, k, w=w),
                "moe_infinity": lambda: MoEInfinityPolicy(L, E, B, k, store=store, online=True)}
        res = {name: {wl.name: replay_path(trace, wl, make(), l2o, requests, gates) for wl in workloads}
               for name, make in arms.items()}
        totals = {name: sum(v.values()) for name, v in res.items()}
        out["capacities"][str(B)] = {"hobbit_w": w, "paths": res, "totals": totals}
        base = totals["lfu_decay"]
        print(f"{args.trace} B={B} [{time.perf_counter() - start:.0f}s] " +
              ", ".join(f"{n} {v:,} ({100 * (v - base) / base:+.2f}%)" for n, v in totals.items()), flush=True)
    dest = ROOT / f"results/t5/baselines_unseen_{args.trace}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
