"""T5 development runs on an H100 node: the costs and spreads that T5's design and predictions need.

Reads ``data/runs/t5_dev/*.json`` (``jobs/t5_dev.sbatch``) and prints, per model and
capacity: routing / token agreement with the original trace (H100 numerics vs the trace's GPU), each arm's wall
time, timed policy cost, copies per token and residual against decayed LFU (``dev_costs.costs``), the copy time
per expert and its issue cost, the paired spread of every comparison (one pass: sequence-level), the gains, and
the copy-cost sweep against the break-even model. On the default run directory it also writes
``results/t5/dev_summary.json`` and the cost file for the a-priori predictions, ``results/t5/dev_costs.json``.

    python analysis/analyze_t5_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from dev_costs import costs  # noqa: E402

RUNS = ROOT / "data/runs/t5_dev"
PAIRS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"), ("global_token_gbt", "lfu_decay"),
         ("splitphase", "global_token_gbt"), ("moe_infinity", "lfu_decay"), ("lfu_decay+pf", "lfu_decay"),
         ("splitphase+pf", "lfu_decay+pf"), ("splitphase+pf", "hobbit+pf"), ("hobbit+pf", "lfu_decay+pf"))


def per_seq(arm: dict) -> dict[int, float]:
    return {r["source_sequence_id"]: r["wall_ms"] / r["tokens"] for r in arm["sequences"] if r.get("pass", 0) == 0}


def main() -> None:
    from paired_bootstrap import paired_bootstrap

    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    out = {}
    for path in sorted(runs.glob("*.json")):
        if path.name.endswith(".guesses.json"):
            continue
        run = json.loads(path.read_text())
        arms = run["arms"]
        c = costs(run)
        entry = {"label": run.get("label"), "capacity": run.get("capacity"), "copy_repeat": run.get("copy_repeat", 1),
                 "host": {k: run["host"].get(k) for k in ("hostname", "gpu", "torch", "transformers")},
                 "bandwidth_probes": run.get("bandwidth_probes"), "costs": c, "routing": {}, "pairs": {}}
        for name, a in arms.items():
            t = a["totals"]
            entry["routing"][name] = {"routing_match": t["routing_match_fraction"], "tokens": t["token_agreement_fraction"],
                                      "sync_ms_per_token": t.get("sync_ms", 0) / max(t["tokens"], 1)}
        for method, base in PAIRS:
            if method not in arms or base not in arms:
                continue
            key = ("dev", int(run["capacity"]))
            rows = lambda arm: [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r["wall_ms"]))
                                for r in arms[arm]["sequences"] if r.get("pass", 0) == 0]
            w = paired_bootstrap({key: rows(base)}, {key: rows(method)}, ["dev"], int(run["capacity"]))
            x, y = per_seq(arms[base]), per_seq(arms[method])
            d = np.array([x[s] - y[s] for s in x])
            entry["pairs"][f"{method} vs {base}"] = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]],
                                                     "paired_sd_ms_per_token": float(d.std(ddof=1)),
                                                     "faster": w["sequences_improved"], "units": w["paired_units"]}
        out[path.stem] = entry
        print(f"== {path.stem} ({run.get('label')}), copy x{entry['copy_repeat']}: t_copy {c['t_copy']:.3f} ms, "
              f"t_issue {1000 * c['t_issue']:.1f} us, host {entry['host']}")
        for name, a in c["arms"].items():
            r = entry["routing"][name]
            print(f"  {name:18s} wall {a['wall']:8.2f}  policy {a['policy']:6.2f}  demand {a['demand']:6.2f}  prefetch "
                  f"{a['prefetch']:6.2f}  residual {a['residual']:+6.2f}  routing {r['routing_match']:.4f}  tokens {r['tokens']:.4f}  "
                  f"sync {r['sync_ms_per_token']:.2f}")
        for pair, p in entry["pairs"].items():
            print(f"  {pair:30s} {100 * p['gain']:+6.2f}% [{100 * p['ci'][0]:+.2f}, {100 * p['ci'][1]:+.2f}] "
                  f"{p['faster']}/{p['units']}  paired sd {p['paired_sd_ms_per_token']:.2f} ms/token")
    if runs != RUNS:
        return
    # the sealed cost file for the a-priori predictions: {model: {capacity: dev_costs.costs(run)}}; the copy-cost sweep
    # runs (<model>_B<cap>_K<k>) go under "sweep"
    dev_costs = {"source": "H100 development runs (jobs/t5_dev.sbatch)", "sweep": {}}
    for name, entry in out.items():
        parts = name.split("_")
        model, cap = parts[0], parts[1].lstrip("B")
        if len(parts) > 2 and parts[2].startswith("K"):
            dev_costs["sweep"].setdefault(model, {}).setdefault(cap, {})[parts[2].lstrip("K")] = entry["costs"]
        else:
            dev_costs.setdefault(model, {})[cap] = entry["costs"]
    (ROOT / "results/t5/dev_costs.json").write_text(json.dumps(dev_costs, indent=1) + "\n")
    dest = ROOT / "results/t5/dev_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
