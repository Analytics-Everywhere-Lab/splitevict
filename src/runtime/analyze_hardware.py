"""Aggregate the GPU runs: speed per policy, fidelity to the simulator, break-even.

Reads every ``<policy>_B<capacity>_<workload>.json`` under ``--runs`` (one directory per
machine) and the replay validation (``results/a40/replay_validation_b32.json``: the
simulator-exact per-sequence transfer counts). Writes ``summary.json`` next to the runs
unless ``--output`` is given:

* per policy and capacity: tokens per second, wall / policy / copy milliseconds per
  token, transfers per token, routing agreement with the recorded trace, and whether the
  transfer counts equal the simulator's;
* paired, source-sequence-clustered bootstrap intervals (the paper's) for wall time
  and for transfers between policies;
* the break-even view: policy time on the critical path against the copy time the
  policy saved relative to ``--baseline`` (default ``lfu_decay``).

    python src/runtime/analyze_hardware.py --runs data/runs/a40 --output results/a40/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))


def load_runs(runs_dir: Path) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    runs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(runs_dir.glob("*_B*_*.json")):
        result = json.loads(path.read_text())
        if result.get("schema_version") != "hardware_run_v1":
            continue
        runs[(result["policy"], int(result["capacity"]))][result["workload"]] = result
    return runs


def rows_for(result_by_workload: dict[str, dict[str, Any]], key: str, capacity: int) -> dict[tuple[str, int], list]:
    out = {}
    for workload, result in result_by_workload.items():
        out[(workload, capacity)] = [
            (row["sequence_position"], row["source_sequence_id"], row["domain"], float(row[key]))
            for row in result["sequences"]
        ]
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="directory of hardware_run_v1 files")
    parser.add_argument("--validation", type=Path, default=CODE.parent / "results/a40/replay_validation_b32.json")
    parser.add_argument("--baseline", default="lfu_decay", help="heuristic the break-even is measured against")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    from paired_bootstrap import paired_bootstrap

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no hardware_run_v1 files under {args.runs}")
    validation = json.loads(args.validation.read_text()) if args.validation.exists() else None
    simulator: dict[tuple[str, int, str, int], int] = {}
    if validation is not None:
        cap = int(validation["capacity"])
        for policy, entry in validation["policies"].items():
            for workload, summary in entry["workloads"].items():
                for row in summary["per_sequence"]:
                    simulator[(policy, cap, workload, int(row["sequence_position"]))] = int(row["expert_transfers"])

    summary: dict[str, Any] = {"schema_version": "hardware_summary_v1", "runs_dir": str(args.runs), "policies": {}, "comparisons": []}
    hosts = set()
    for (policy, capacity), by_workload in sorted(runs.items()):
        totals = defaultdict(float)
        matched = compared = 0
        for workload, result in by_workload.items():
            hosts.add(json.dumps(result["host"], sort_keys=True))
            for key in ("tokens", "expert_transfers", "bytes_transferred", "wall_ms", "policy_ms", "issue_ms", "copy_ms",
                        "routing_events", "routing_matches", "token_agreements", "token_comparisons"):
                totals[key] += result["totals"][key]
            for row in result["sequences"]:
                expected = simulator.get((policy, capacity, workload, int(row["sequence_position"])))
                if expected is not None:
                    compared += 1
                    matched += int(expected == int(row["expert_transfers"]))
        tokens = max(totals["tokens"], 1)
        entry = {
            "workloads": sorted(by_workload),
            "sequences": sum(len(r["sequences"]) for r in by_workload.values()),
            "tokens": int(totals["tokens"]),
            "expert_transfers": int(totals["expert_transfers"]),
            "transfers_per_token": totals["expert_transfers"] / tokens,
            "gb_transferred": totals["bytes_transferred"] / 1e9,
            "ms_per_token_wall": totals["wall_ms"] / tokens,
            "ms_per_token_policy": totals["policy_ms"] / tokens,
            "ms_per_token_issue": totals["issue_ms"] / tokens,
            "ms_per_token_copy": totals["copy_ms"] / tokens,
            "tokens_per_second": 1e3 * tokens / max(totals["wall_ms"], 1e-9),
            "routing_match_fraction": totals["routing_matches"] / max(totals["routing_events"], 1),
            "token_agreement_fraction": totals["token_agreements"] / max(totals["token_comparisons"], 1),
            "simulator_sequences_compared": compared,
            "simulator_sequences_matched": matched,
            "transfers_equal_simulator": compared > 0 and matched == compared,
            "bandwidth_probe": next(iter(by_workload.values()))["bandwidth_probe"],
        }
        summary["policies"][f"{policy}@{capacity}"] = entry
    summary["hosts"] = [json.loads(h) for h in sorted(hosts)]

    # paired comparisons on the common workloads of each capacity
    by_capacity: dict[int, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for (policy, capacity), by_workload in runs.items():
        by_capacity[capacity][policy] = by_workload
    for capacity, policies in sorted(by_capacity.items()):
        names = sorted(policies)
        for left in names:
            for right in names:
                if left == right:
                    continue
                common = sorted(set(policies[left]) & set(policies[right]))
                if not common:
                    continue
                left_rows = {k: v for k, v in rows_for(policies[left], "wall_ms", capacity).items() if k[0] in common}
                right_rows = {k: v for k, v in rows_for(policies[right], "wall_ms", capacity).items() if k[0] in common}
                try:
                    wall = paired_bootstrap(left_rows, right_rows, common, capacity)
                except RuntimeError as exc:
                    wall = {"error": str(exc)}
                comparison = {"capacity": capacity, "slower_or_baseline": left, "candidate": right, "workloads": common,
                              "wall_time_reduction": wall}
                if left != "resident" and right != "resident":
                    left_tr = {k: v for k, v in rows_for(policies[left], "expert_transfers", capacity).items() if k[0] in common}
                    right_tr = {k: v for k, v in rows_for(policies[right], "expert_transfers", capacity).items() if k[0] in common}
                    try:
                        comparison["transfer_reduction"] = paired_bootstrap(left_tr, right_tr, common, capacity)
                    except RuntimeError as exc:
                        comparison["transfer_reduction"] = {"error": str(exc)}
                summary["comparisons"].append(comparison)

    # break-even: what the policy costs on the critical path vs the copy time it saved
    breakeven = []
    for capacity, policies in sorted(by_capacity.items()):
        base = summary["policies"].get(f"{args.baseline}@{capacity}")
        if base is None:
            continue
        for policy in sorted(policies):
            entry = summary["policies"][f"{policy}@{capacity}"]
            if policy == args.baseline or policy == "resident":
                continue
            probe = entry.get("bandwidth_probe") or {}
            ms_per_expert = probe.get("ms_per_expert")
            saved_transfers = base["transfers_per_token"] - entry["transfers_per_token"]
            breakeven.append({
                "capacity": capacity, "policy": policy, "baseline": args.baseline,
                "transfers_saved_per_token": saved_transfers,
                "copy_ms_saved_per_token_measured": base["ms_per_token_copy"] - entry["ms_per_token_copy"],
                "copy_ms_saved_per_token_at_probe_bandwidth": None if ms_per_expert is None else saved_transfers * ms_per_expert,
                "policy_ms_per_token": entry["ms_per_token_policy"],
                "baseline_policy_ms_per_token": base["ms_per_token_policy"],
                "wall_ms_per_token_delta": entry["ms_per_token_wall"] - base["ms_per_token_wall"],
                "net_faster_than_baseline": entry["ms_per_token_wall"] < base["ms_per_token_wall"],
            })
    summary["breakeven"] = breakeven

    out = args.output or (args.runs / "summary.json")
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'policy':<18}{'B':>4}{'tok/s':>9}{'wall':>8}{'policy':>8}{'copy':>7}{'xfer/tok':>10}{'route%':>8}{'tok%':>7}{'sim':>6}")
    for key, e in summary["policies"].items():
        policy, cap = key.split("@")
        print(f"{policy:<18}{cap:>4}{e['tokens_per_second']:>9.1f}{e['ms_per_token_wall']:>8.2f}{e['ms_per_token_policy']:>8.2f}"
              f"{e['ms_per_token_copy']:>7.2f}{e['transfers_per_token']:>10.2f}{100 * e['routing_match_fraction']:>8.2f}"
              f"{100 * e['token_agreement_fraction']:>7.2f}{('yes' if e['transfers_equal_simulator'] else ('n/a' if not e['simulator_sequences_compared'] else 'NO')):>6}")
    for row in breakeven:
        print(f"  break-even B={row['capacity']} {row['policy']} vs {row['baseline']}: saves {row['transfers_saved_per_token']:+.2f} transfers/token "
              f"({row['copy_ms_saved_per_token_measured']:+.2f} ms copy), costs {row['policy_ms_per_token']:.2f} ms policy; "
              f"wall {row['wall_ms_per_token_delta']:+.2f} ms/token -> {'faster' if row['net_faster_than_baseline'] else 'slower'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
