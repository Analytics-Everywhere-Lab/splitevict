"""The policy's own cost against what it buys: boosting rounds vs transfers vs milliseconds.

Joins the calibration-only tree sweep (``results/caer_trees{N}_development.json``: fitted
on calA, deployed on calB) with the replay timings of the matching exported policies
(``results/a40/policy_overhead_trees{N}.json``) and the full validation replay
(``results/a40/replay_validation_b32.json``). Writes ``results/a40/policy_overhead.json``:
for each tree count, the transfers on calB at B=24 and B=32 for ``global_token_gbt`` and
``caer_gbt``, and the measured milliseconds per layer-event and per token of each.

    python src/runtime/summarize_overhead.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HARDWARE = RESULTS / "a40"
TREES = (100, 200, 400, 800)


def main() -> None:
    rows = []
    for trees in TREES:
        dev = RESULTS / "ranker" / f"caer_trees{trees}_development.json"
        timing = HARDWARE / f"policy_overhead_trees{trees}.json"
        if not dev.exists():
            print(f"missing {dev.name}; skipped")
            continue
        result = json.loads(dev.read_text())
        if result["configuration"]["global_model"]["max_iter"] != trees or result["phase"] != "development":
            raise SystemExit(f"{dev.name} is not the {trees}-round calA -> calB run")
        aggregate = result["aggregate"]["methods"]
        row = {"trees": trees, "development": {}, "overhead": None}
        for capacity in ("24", "32"):
            row["development"][capacity] = {
                method: {
                    "transfers": aggregate[method][capacity]["expert_transfers"],
                    "oracle_gap_closed": aggregate[method][capacity]["oracle_gap_closed"],
                }
                for method in ("global_token_gbt", "caer_gbt")
            }
            row["development"][capacity]["caer_gbt"]["reduction_vs_global"] = (
                aggregate["caer_gbt"][capacity]["relative_transfer_reduction_vs_global"]
            )
        if timing.exists():
            measured = json.loads(timing.read_text())
            row["overhead"] = {
                "host": measured["host"],
                "backend": measured["backend"],
                "capacity": measured["capacity"],
                "limit_sequences": measured["limit_sequences"],
                "policies": {
                    name: {
                        "ms_per_event_by_component": entry["aggregate"]["ms_per_event_by_component"],
                        "ms_per_token_policy": entry["aggregate"]["ms_per_token_policy"],
                        "ms_per_decision_policy": next(iter(entry["workloads"].values()))["ms_per_decision_policy"],
                        "event_policy_ms_percentiles": next(iter(entry["workloads"].values())).get("event_policy_ms_percentiles"),
                    }
                    for name, entry in measured["policies"].items()
                },
            }
        rows.append(row)
    validation = HARDWARE / "replay_validation_b32.json"
    full = None
    if validation.exists():
        measured = json.loads(validation.read_text())
        full = {
            "host": measured["host"], "backend": measured["backend"], "capacity": measured["capacity"],
            "policies": {
                name: {
                    "events": entry["aggregate"]["events"],
                    "eviction_decisions": entry["aggregate"]["eviction_decisions"],
                    "transfers": entry["aggregate"]["transfers"],
                    "ms_per_event_by_component": entry["aggregate"]["ms_per_event_by_component"],
                    "ms_per_token_policy": entry["aggregate"]["ms_per_token_policy"],
                    "all_checks_exact": entry["aggregate"]["all_checks_exact"],
                }
                for name, entry in measured["policies"].items()
            },
        }
    summary = {
        "schema_version": "policy_overhead_v1",
        "status": "calibration-only tree sweep (calA -> calB) joined with replay timings; the sealed evaluation used 800 rounds",
        "tree_sweep": rows,
        "full_policy_replay_b32": full,
    }
    out = HARDWARE / "policy_overhead.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'trees':>6} {'B':>3} {'global':>9} {'caer_gbt':>9} {'vs global':>10} {'ms/event global':>16} {'ms/event caer':>14}")
    for row in rows:
        for capacity, cell in row["development"].items():
            overhead = row["overhead"]["policies"] if row["overhead"] else {}
            g = overhead.get("global_token_gbt", {}).get("ms_per_event_by_component", {}).get("policy")
            c = overhead.get("caer_gbt", {}).get("ms_per_event_by_component", {}).get("policy")
            print(f"{row['trees']:>6} {capacity:>3} {cell['global_token_gbt']['transfers']:>9,} {cell['caer_gbt']['transfers']:>9,} "
                  f"{100 * cell['caer_gbt']['reduction_vs_global']:>+9.3f}% {('%.3f' % g) if g is not None else '-':>16} {('%.3f' % c) if c is not None else '-':>14}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
