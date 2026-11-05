"""Decode emulation (Section 5): replay calB through the split-phase service with emulated GPU time per layer.

Checks (1) the service's decisions equal the simulation (transfers per sequence), (2) how long
the critical path waits for worker scores, (3) the main-thread decision cost, against decayed LFU
and the synchronous ranker run through the same loop.

    python analysis/emulate_decode.py --policy splitphase --capacity 32 --layer-ms 2.6 --workers 1 \
        --p2 data/rankers/olmoe_dev_lag_req_trees200.pkl --p1 data/rankers/olmoe_dev_lag_tok_trees200.pkl \
        --output results/simulation/emulation/splitphase_B32_l2.6.json
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


def spin(ms: float) -> None:
    end = time.perf_counter() + ms / 1e3
    while time.perf_counter() < end:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=("splitphase", "lfu_decay", "sync"))
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--layer-ms", type=float, default=2.6, help="emulated GPU time before each router")
    parser.add_argument("--p2", default=str(ROOT / "data/rankers/olmoe_dev_lag_req_trees800.pkl"))
    parser.add_argument("--p1", default=str(ROOT / "data/rankers/olmoe_dev_lag_tok_trees800.pkl"))
    parser.add_argument("--sync-bundle", default=str(ROOT / "data/rankers/olmoe_setdev_dev_trees800.pkl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slow-service", action="store_true", help="worker-slowdown emulation (SPLITPHASE_SLOWDOWN)")
    parser.add_argument("--prepared-root", help="Qwen3 / Mixtral prepared root (crossmodel)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import splitphase
    from splitphase import load_dev, load_bundle
    from policy import SimplePolicy, LearnedPolicy

    if args.prepared_root:
        splitphase.PREPARED_ROOT = Path(args.prepared_root).resolve()

    inputs, _fit, deploy = load_dev()
    trace = inputs.trace
    L, E, k = trace.num_layers, trace.num_experts, trace.top_k
    service = None
    if args.policy == "splitphase":
        if args.slow_service:
            from splitphase_service_slow import SplitPhaseService, SplitPhasePolicy
        else:
            from splitphase_service import SplitPhaseService, SplitPhasePolicy

        service = SplitPhaseService(args.p2, args.p1, args.capacity, workers=args.workers,
                                    num_layers=L, num_experts=E, top_k=k)
        policy = SplitPhasePolicy(service, L, E, args.capacity, k)
    elif args.policy == "lfu_decay":
        policy = SimplePolicy("lfu_decay", L, E, args.capacity, k)
    else:
        policy = LearnedPolicy(load_bundle(args.sync_bundle), args.capacity, "global_token_gbt")

    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: i for i, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    per_seq, waits, policy_ns = [], [], []
    parts = {"decide": 0, "post": 0, "begin_token": 0}
    tokens = 0
    start = time.perf_counter()
    for count, (sequence, view) in enumerate(deploy.iter_slices(trace)):
        if args.limit is not None and count >= args.limit:
            break
        policy.begin_sequence()
        before = policy.transfers
        for base in range(view.start, view.stop, L):
            token = int(trace.token_id[base])
            if hasattr(policy, "begin_token"):
                t0 = time.perf_counter_ns()
                policy.begin_token(token)
                parts["begin_token"] += time.perf_counter_ns() - t0
            tokens += 1
            for index in range(base, base + L):
                spin(args.layer_ms)
                result = policy.step(l2o[int(trace.layer_index[index])], token, requests[index], gates[index])
                policy_ns.append(result.timings_ns["policy"])
                waits.append(result.timings_ns.get("wait", 0))
                parts["decide"] += result.timings_ns.get("decide", 0)
                parts["post"] += result.timings_ns.get("post", 0)
        per_seq.append({"source_sequence_id": int(sequence.source_sequence_id), "transfers": policy.transfers - before})
    wall = time.perf_counter() - start
    busy = service.close() if service is not None else None
    pn = np.asarray(policy_ns) / 1e6
    wn = np.asarray(waits) / 1e6
    out = {
        "policy": args.policy, "capacity": args.capacity, "workers": args.workers, "layer_ms": args.layer_ms,
        "prepared_root": args.prepared_root, "num_layers": L, "num_experts": E, "top_k": k,
        "slowdown": float(__import__("os").environ.get("SPLITPHASE_SLOWDOWN", "1")) if args.slow_service else 1.0,
        "tokens": tokens, "events": int(pn.size), "transfers": int(policy.transfers),
        "wall_seconds": wall, "ms_per_token_wall": 1e3 * wall / tokens,
        "ms_per_token_emulated_gpu": args.layer_ms * L,
        "policy_ms_per_token": float(pn.sum() / tokens),
        "policy_ms_per_event_p50_p99": [float(np.percentile(pn, 50)), float(np.percentile(pn, 99))],
        "wait_ms_per_token": float(wn.sum() / tokens),
        "events_that_waited": int((wn > 0.05).sum()),
        "worker_busy_ms_per_token": None if busy is None else [b / 1e6 / tokens for b in busy],
        "ms_per_token_parts": {k: v / 1e6 / max(tokens, 1) for k, v in parts.items()},
        "worker_profile": getattr(service, "profile", None),
        "per_sequence": per_seq,
    }
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{args.policy} B={args.capacity} layer_ms={args.layer_ms}: transfers={out['transfers']:,} "
          f"wall={out['ms_per_token_wall']:.2f} ms/tok (gpu {out['ms_per_token_emulated_gpu']:.1f}) "
          f"policy={out['policy_ms_per_token']:.3f} ms/tok wait={out['wait_ms_per_token']:.3f} ms/tok "
          f"waited_events={out['events_that_waited']} parts={out['ms_per_token_parts']}", flush=True)


if __name__ == "__main__":
    main()
