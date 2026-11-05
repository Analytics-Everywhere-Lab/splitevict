"""Simulated copies on a test trace (T1, T3): decayed LFU, the synchronous ranker, split-phase with frozen rankers.

Runs every workload path of a fresh trace (``fresh_trace.py``) with a cold policy per path, and records
per-sequence transfers for the paired bootstrap. The arms use the ``SplitPhase`` replay, which matches
the simulator exactly on calB for all three sources.

Qwen3 / Mixtral (``fresh_trace_offloaded.py``): ``--spec`` replaces ``--frozen-dir``; the one path is
the spec's sequence order (what ``interleaved_multi.py`` decodes), and the bundles are given explicitly.

    python analysis/unseen_sim.py --trace-root data/traces/olmoe_t1 --frozen-dir data/prepared/olmoe_t1 \
        --arm splitphase --capacity 32 --output results/t1/sim/sim_splitphase_B32.json
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

FROZEN = {
    "sync_bundle": ROOT / "data/rankers/olmoe_setdev_policy.pkl",
    "p2_bundle": ROOT / "data/rankers/olmoe_frozen_lag_req_trees200.pkl",
    "p1_bundle": ROOT / "data/rankers/olmoe_frozen_lag_tok_trees200.pkl",
}


class SpecPath:
    """The replay spec's sequence order as a workload: ``iter_slices`` like a frozen path."""

    def __init__(self, spec, trace):
        self.name = spec["workloads"][0]["name"]
        self.trace = trace
        by_id = {int(s["source_sequence_id"]): s for s in spec["workloads"][0]["sequences"]}
        self.items = [by_id[int(sid)] for sid in spec["calibration_sequence_ids"]]

    def iter_slices(self, trace):
        from types import SimpleNamespace

        slices = trace.sequence_slices()
        for position, item in enumerate(self.items):
            sid = int(item["source_sequence_id"])
            yield SimpleNamespace(position=position, source_sequence_id=sid, domain=item["domain"]), slices[sid]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--spec", type=Path, help="replay spec of a fresh_trace_offloaded trace (its order is the path)")
    parser.add_argument("--sync-bundle", type=Path, default=FROZEN["sync_bundle"])
    parser.add_argument("--p2-bundle", type=Path, default=FROZEN["p2_bundle"])
    parser.add_argument("--p1-bundle", type=Path, default=FROZEN["p1_bundle"])
    parser.add_argument("--arm", required=True, choices=("lfu", "sync", "splitphase"))
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--workloads", nargs="+", help="path names (default all); 'calibration' for a smoke test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from fresh_trace import _paths, load_fresh
    from splitphase import SplitPhase, load_bundle

    if args.spec is not None:
        _paths(OPT)
        from residency_headroom.trace import RoutingTrace

        trace = RoutingTrace.load(args.trace_root / "routing_trace.npz", validate=True)
        spec = json.loads(args.spec.read_text())
        if spec["trace_hash"] != trace.trace_hash:
            raise SystemExit("spec and trace differ")
        workloads = [SpecPath(spec, trace)]
    else:
        inputs = load_fresh(OPT, args.trace_root, args.frozen_dir)
        # the feature modules (features2/3, run_caer) come from src/; harness/run8 are not needed
        trace = inputs.trace
        workloads = list(inputs.workloads)
        if args.workloads:
            named = {w.name: w for w in workloads}
            named["calibration"] = inputs.calibration
            workloads = [named[n] for n in args.workloads]
    bundles = {"sync_bundle": args.sync_bundle, "p2_bundle": args.p2_bundle, "p1_bundle": args.p1_bundle}
    sync = load_bundle(str(args.sync_bundle))
    p2 = load_bundle(str(args.p2_bundle))
    p1 = load_bundle(str(args.p1_bundle))
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)

    def new_arm():
        if args.arm == "lfu":
            return SplitPhase(sync, args.capacity, "lfu", "none", diagnose=False)
        if args.arm == "sync":
            return SplitPhase(sync, args.capacity, "fresh", "none", diagnose=False)
        return SplitPhase(sync, args.capacity, "lag_req", "none", diagnose=False, mix_layer0=True,
                          lag_bundle=p2, lag_tok_bundle=p1)

    out = {"arm": args.arm, "capacity": args.capacity, "trace_hash": trace.trace_hash,
           "frozen": {k: str(v) for k, v in bundles.items()}, "paths": {}}
    start = time.perf_counter()
    for workload in workloads:
        arm = new_arm()
        rows = []
        for count, (sequence, view) in enumerate(workload.iter_slices(trace)):
            if args.limit is not None and count >= args.limit:
                break
            arm.begin_sequence()
            before = arm.transfers
            for index in range(view.start, view.stop):
                arm.step(l2o[int(trace.layer_index[index])], int(trace.token_id[index]), requests[index], gates[index])
            rows.append({"position": int(sequence.position), "source_sequence_id": int(sequence.source_sequence_id),
                         "domain": str(sequence.domain), "transfers": arm.transfers - before})
        out["paths"][workload.name] = {"transfers": int(sum(r["transfers"] for r in rows)), "sequences": rows}
        print(f"{args.arm} B={args.capacity} {workload.name}: {out['paths'][workload.name]['transfers']:,} "
              f"[{time.perf_counter() - start:.0f}s]", flush=True)
    out["transfers"] = int(sum(p["transfers"] for p in out["paths"].values()))
    out["wall_seconds"] = time.perf_counter() - start
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{args.arm} B={args.capacity} total {out['transfers']:,}", flush=True)


if __name__ == "__main__":
    main()
