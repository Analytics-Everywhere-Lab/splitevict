"""Replay recorded routing through the deployable policies: exactness and overhead.

Two questions, answered without a GPU:

1. Does the event-by-event policy (``policy.py``) make the same decisions as the batched
   simulator?  Each learned variant's transfers per sequence are compared with the
   simulator's ``per_sequence`` records (exact equality expected); each simple policy's
   path totals are compared with the frozen ``residency_headroom`` simulator.
2. What does one decision cost?  Wall time per event is split into feature
   construction, global scoring, stage-2 reranking and bookkeeping, for each inference
   backend, on this host.  This is the per-event cost a serving loop would pay on its
   critical path before it can issue the expert transfers.

    python src/runtime/replay_overhead.py --repository-root measurement \
        --policy-bundle data/rankers/olmoe_setdev_policy.pkl --capacity 32 \
        --policies lru lfu_decay global_token_gbt caer_gbt --workloads stationary_general \
        --output replay_b32.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy import (  # noqa: E402
    LEARNED_VARIANTS,
    LFU_DECAY_ALPHA,
    SIMPLE_VARIANTS,
    PolicyBundle,
    make_policy,
)


def load_inputs(repository_root: Path, prepared_root: Path | None):
    from run_caer import configure_repository

    configure_repository(repository_root)
    if prepared_root is None:
        from harness import load

        inputs, _models = load()
        return inputs
    import crossmodel

    inputs, _models = crossmodel.load(Path(prepared_root).resolve())
    return inputs


def simple_reference(trace, workload, capacity: int, name: str, static: np.ndarray) -> int:
    from residency_headroom.simulator import simulate_policy

    kwargs: dict[str, Any] = {}
    if name == "lfu_decay":
        kwargs["alpha"] = LFU_DECAY_ALPHA
    if name == "static_hotset":
        kwargs["static_scores"] = static
    return int(simulate_policy(trace, workload, capacity, name, **kwargs).misses)


def replay(policy, trace, workload, *, limit_sequences: int | None, collect_event_times: bool):
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    l2o = {layer: ordinal for ordinal, layer in enumerate(layers)}
    requests = trace.requested_expert_ids.astype(np.int64, copy=False)
    gates = trace.router_weights.astype(np.float64, copy=False)
    tokens = trace.token_id
    layer_index = trace.layer_index
    per_sequence: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    event_times: list[int] = []
    events = 0
    decisions = 0
    stage2_rows = 0
    for count, (sequence, view) in enumerate(workload.iter_slices(trace)):
        if limit_sequences is not None and count >= limit_sequences:
            break
        policy.begin_sequence()
        before = policy.transfers
        for index in range(view.start, view.stop):
            ordinal = l2o[int(layer_index[index])]
            result = policy.step(ordinal, int(tokens[index]), requests[index], gates[index])
            events += 1
            decisions += int(result.eviction_decision)
            stage2_rows += result.stage2_rows
            for key, value in result.timings_ns.items():
                totals[key] = totals.get(key, 0) + int(value)
            if collect_event_times:
                event_times.append(int(result.timings_ns["policy"]))
        per_sequence.append(
            {
                "sequence_position": int(sequence.position),
                "source_sequence_id": int(sequence.source_sequence_id),
                "domain": str(sequence.domain),
                "expert_transfers": int(policy.transfers - before),
            }
        )
    times = np.asarray(event_times, dtype=np.float64) / 1e6 if event_times else None
    summary = {
        "events": events,
        "eviction_decisions": decisions,
        "stage2_rows_scored": stage2_rows,
        "transfers": int(sum(row["expert_transfers"] for row in per_sequence)),
        "seconds_by_component": {key: value / 1e9 for key, value in totals.items()},
        "ms_per_event_by_component": {key: value / 1e6 / max(events, 1) for key, value in totals.items()},
        "ms_per_decision_policy": totals.get("policy", 0) / 1e6 / max(decisions, 1),
        "ms_per_token_policy": totals.get("policy", 0) / 1e6 / max(events / len(layers), 1),
    }
    if times is not None:
        summary["event_policy_ms_percentiles"] = {
            f"p{p}": float(np.percentile(times, p)) for p in (50, 90, 99, 99.9)
        }
        summary["event_policy_ms_max"] = float(times.max())
    return per_sequence, summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--policies", nargs="+", default=["lru", "lfu_decay", "global_token_gbt", "caer_gbt"])
    parser.add_argument("--workloads", nargs="+", default=None, help="path names (default: all)")
    parser.add_argument("--backend", default="numba", help="numba | sklearn | flat")
    parser.add_argument("--score-scope", default="candidates", choices=("candidates", "all"))
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--reference", type=Path, help="run_caer result with per_sequence rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-event-times", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    inputs = load_inputs(args.repository_root.expanduser(), args.prepared_root)
    trace = inputs.trace
    bundle = PolicyBundle.load(args.policy_bundle)
    if str(bundle.payload["trace_hash"]) != str(trace.trace_hash):
        raise SystemExit("policy bundle was fitted on a different trace")
    workloads = list(inputs.workloads)
    if args.workloads:
        by_name = {w.name: w for w in workloads}
        missing = [n for n in args.workloads if n not in by_name]
        if missing:
            raise SystemExit(f"unknown workloads {missing}; have {sorted(by_name)}")
        workloads = [by_name[n] for n in args.workloads]

    reference_rows: dict[tuple[str, str, int], int] = {}
    if args.reference is not None:
        ref = json.loads(args.reference.read_text())
        for row in ref["per_sequence"]:
            if int(row["capacity"]) == int(args.capacity):
                reference_rows[(row["method"], row["workload"], int(row["sequence_position"]))] = int(row["expert_transfers"])

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    import sklearn

    report: dict[str, Any] = {
        "schema_version": "hardware_replay_v1",
        "label": args.label,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "numba": numba_version,
        },
        "policy_bundle": {"path": str(args.policy_bundle), "model": bundle.payload["model"],
                          "trace_hash": bundle.payload["trace_hash"], "configuration": bundle.configuration},
        "capacity": int(args.capacity),
        "backend": args.backend,
        "score_scope": args.score_scope,
        "limit_sequences": args.limit_sequences,
        "policies": {},
    }
    static = bundle.static()
    for name in args.policies:
        print(f"== {name} (B={args.capacity}, backend={args.backend})", flush=True)
        entry: dict[str, Any] = {"workloads": {}}
        for workload in workloads:
            # every path starts cold, as in the simulator: fresh cache and feature state
            policy = make_policy(
                name, num_layers=trace.num_layers, num_experts=trace.num_experts,
                capacity=int(args.capacity), top_k=trace.top_k, bundle=bundle,
                backend=args.backend, score_scope=args.score_scope, static_scores=static,
            )
            start = time.perf_counter()
            per_sequence, summary = replay(
                policy, trace, workload, limit_sequences=args.limit_sequences,
                collect_event_times=not args.no_event_times,
            )
            summary["wall_seconds"] = time.perf_counter() - start
            check: dict[str, Any] = {}
            if name in LEARNED_VARIANTS and reference_rows:
                matches = 0
                compared = 0
                mismatches = []
                for row in per_sequence:
                    key = (name, workload.name, row["sequence_position"])
                    if key in reference_rows:
                        compared += 1
                        if reference_rows[key] == row["expert_transfers"]:
                            matches += 1
                        elif len(mismatches) < 5:
                            mismatches.append({"sequence_position": row["sequence_position"],
                                               "replay": row["expert_transfers"], "simulator": reference_rows[key]})
                check = {"kind": "simulator_per_sequence", "compared": compared, "matched": matches,
                         "exact": compared > 0 and matches == compared, "first_mismatches": mismatches}
            elif name in SIMPLE_VARIANTS and args.limit_sequences is None:
                expected = simple_reference(trace, workload, int(args.capacity), name, static)
                check = {"kind": "frozen_simulator_total", "simulator": expected,
                         "replay": summary["transfers"], "exact": expected == summary["transfers"]}
            summary["check"] = check
            summary["per_sequence"] = per_sequence
            entry["workloads"][workload.name] = summary
            status = "" if not check else ("  exact" if check.get("exact") else "  MISMATCH")
            print(
                f"   {workload.name}: {summary['transfers']:,} transfers, "
                f"{summary['ms_per_decision_policy']:.3f} ms/decision, "
                f"{summary['ms_per_token_policy']:.2f} ms/token policy time, "
                f"{summary['wall_seconds']:.0f}s{status}",
                flush=True,
            )
        total_events = sum(w["events"] for w in entry["workloads"].values())
        comp: dict[str, float] = {}
        for w in entry["workloads"].values():
            for key, value in w["seconds_by_component"].items():
                comp[key] = comp.get(key, 0.0) + value
        entry["aggregate"] = {
            "events": total_events,
            "eviction_decisions": sum(w["eviction_decisions"] for w in entry["workloads"].values()),
            "transfers": sum(w["transfers"] for w in entry["workloads"].values()),
            "ms_per_event_by_component": {k: 1e3 * v / max(total_events, 1) for k, v in comp.items()},
            "ms_per_token_policy": 1e3 * comp.get("policy", 0.0) / max(total_events / trace.num_layers, 1),
            "all_checks_exact": all(w["check"].get("exact", True) for w in entry["workloads"].values() if w["check"]),
        }
        report["policies"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
