"""T6 analysis: Mixtral at higher copy costs on an H100 node.

Runs (``jobs/t6.sbatch``): ``data/runs/t6/mixtral/B4_K<k>.json`` for K = 1, 2, 4, 7 (four arms, one pass, every
expert copy issued K times into its slot), all on Mixtral's fresh T6 trace (examples 280-284 of each domain,
20 sequences).

The estimator, the fidelity checks and the break-even model a priori are T5's, with the helpers imported from
``analyze_t5.py``: per sequence and arm the run's one pass; every comparison the paired, sequence-clustered,
domain-stratified bootstrap (1,000 replicates); "faster" / "slower" means the 95% interval lies entirely above /
below zero. The a-priori model takes every cost from T5's sealed development cost file (``results/t5/dev_costs.json``,
entry ``mixtral/4``) with the sealed slow-link rule of ``predictions_t6.costs_at`` (t_copy and t_issue K times their
K = 1 values, policy costs and residuals as at K = 1), and from the test run only the simulator's copies per token and
the base arm's wall time. A1: every comparison predicted at least one percentage point from zero has a measured gain of
the same sign; A2: the median absolute error over all comparisons is at most one percentage point.

    python analysis/analyze_t6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.append(str(ROOT / "src"))

from analyze_t5 import SubsetPath, load_trace, model_params, predicted_gain, rows, view  # noqa: E402
from predictions_t6 import costs_at  # noqa: E402

RUNS = ROOT / "data/runs/t6/mixtral"
DEV_COSTS = ROOT / "results/t5/dev_costs.json"
BUNDLES = ROOT / "data/rankers"
INFO = {"trace": ROOT / "data/traces/mixtral_t6", "spec": ROOT / "data/specs/mixtral_t6_replay_spec.json",
        "fit": ROOT / "data/baselines/mixtral_frozen.json",
        "p2": BUNDLES / "mixtral_8x7b_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "mixtral_8x7b_frozen_lag_tok_trees200.pkl",
        "sync": BUNDLES / "mixtral_8x7b_dev_trees800.pkl"}
CAP = 4
KS = (1, 2, 4, 7)
SEALED_ARMS = ("lfu_decay", "hobbit", "splitphase", "global_token_gbt")
COMPARISONS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"),
               ("global_token_gbt", "lfu_decay"), ("splitphase", "global_token_gbt"))
# (id, K, method, base, criterion): "above" = 95% interval above zero, "below" = below zero. Directional hypotheses
# only where the sealed prediction is at least two points from zero; every comparison enters A1/A2.
HYPOTHESES = (
    ("P1", 2, "splitphase", "lfu_decay", "above"),
    ("S1", 4, "splitphase", "lfu_decay", "above"),
    ("S2", 7, "splitphase", "lfu_decay", "above"),
    ("S3", 1, "splitphase", "lfu_decay", "above"),
    ("S4a", 1, "global_token_gbt", "lfu_decay", "below"),
    ("S4b", 7, "global_token_gbt", "lfu_decay", "above"),
    *((f"S5-K{k}", k, "splitphase", "global_token_gbt", "above") for k in (1, 2, 4)),
    *((f"S6-K{k}", k, "splitphase", "hobbit", "above") for k in KS),
)


def analyse(k: int, dev: dict) -> dict | None:
    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, per_token_wall
    from h100_sim import run_arm

    path = RUNS / f"B{CAP}_K{k}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    arms = run["arms"]
    passes = int(run.get("passes", 1))
    spec = json.loads(INFO["spec"].read_text())
    trace = load_trace(INFO)
    fit = json.loads(INFO["fit"].read_text())
    out = {"model": "mixtral", "capacity": CAP, "copy_repeat": int(run.get("copy_repeat", 1)), "passes": passes,
           "host": run.get("host", {}).get("hostname"), "fidelity": {}, "comparisons": {}, "arms": {},
           "simulated_per_token": {}}
    out["fidelity"]["copy_repeat_as_sealed"] = int(run.get("copy_repeat", 1)) == k
    out["fidelity"]["passes_as_sealed"] = passes == 1
    out["fidelity"]["arms_as_sealed"] = sorted(arms) == sorted(SEALED_ARMS)
    arms = {name: a for name, a in arms.items() if name in SEALED_ARMS}
    out["fidelity"]["hobbit_w_as_sealed"] = (run.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(CAP)]["w"]
    order = [r["source_sequence_id"] for r in sorted(view(next(iter(arms.values()))["sequences"], 0),
                                                       key=lambda r: r["sequence_position"])]
    deploy = SubsetPath(order, spec)
    params = {"p2": INFO["p2"], "p1": INFO["p1"], "sync": INFO["sync"], "hobbit_w": fit["hobbit"][str(CAP)]["w"]}
    for name, a in arms.items():
        t_by = a["totals_by_pass"]
        expected = run_arm(trace, deploy, name, CAP, params)
        tokens = sum(r["tokens"] for r in view(a["sequences"], 0))
        out["simulated_per_token"][name] = {"demand": sum(e["demand"] for e in expected) / tokens, "prefetch": 0.0}
        live = sorted(view(a["sequences"], 0), key=lambda r: r["sequence_position"])
        out["fidelity"][name] = {"routing_all_events": all(t["routing_matches"] == t["routing_events"] for t in t_by),
                                 "token_agreement": [t["token_agreement_fraction"] for t in t_by],
                                 "simulator_equal_every_pass": [e["demand"] for e in expected]
                                 == [r["expert_transfers"] for r in live]}
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    sealed_costs = costs_at(dev, k)
    for method, base in COMPARISONS:
        if method not in arms or base not in arms:
            continue
        base_wall = per_token_wall(arms[base], passes)
        key = ("t6", CAP)
        w = paired_bootstrap({key: rows(view(arms[base]["sequences"], 0))},
                             {key: rows(view(arms[method]["sequences"], 0))}, ["t6"], CAP)
        est = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
               "units": w["paired_units"]}
        out["comparisons"][f"{method} vs {base}"] = {
            "a_priori": a_priori(sealed_costs, out["simulated_per_token"], method, base, base_wall),
            "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base]),
            "estimate": est, "above_zero": est["ci"][0] > 0, "below_zero": est["ci"][1] < 0}
    return out


def main() -> None:
    dev = json.loads(DEV_COSTS.read_text())["mixtral"][str(CAP)]
    result = {"runs": {}}
    for k in KS:
        entry = analyse(k, dev)
        if entry is not None:
            result["runs"][f"mixtral_B{CAP}_K{k}"] = entry
    hypotheses = {}
    for hid, k, method, base, criterion in HYPOTHESES:
        e = result["runs"].get(f"mixtral_B{CAP}_K{k}", {}).get("comparisons", {}).get(f"{method} vs {base}")
        if e is None:
            hypotheses[hid] = {"run": None}
            continue
        hypotheses[hid] = {"copy_repeat": k, "comparison": f"{method} vs {base}", "criterion": criterion,
                           "measured": e["estimate"]["gain"], "ci": e["estimate"]["ci"], "a_priori": e["a_priori"],
                           "met": e["above_zero"] if criterion == "above" else e["below_zero"]}
    result["hypotheses"] = hypotheses
    pairs = [(run, pair, e) for run, r in result["runs"].items() for pair, e in r["comparisons"].items()]
    decisive = [(run, pair, e) for run, pair, e in pairs if abs(e["a_priori"]) >= 0.01]
    errors = [abs(e["a_priori"] - e["estimate"]["gain"]) for _run, _pair, e in pairs]
    result["a_priori"] = {
        "comparisons": len(pairs), "decisive": len(decisive),
        "A1_sign_agreement_decisive": bool(decisive) and all(np.sign(e["a_priori"]) == np.sign(e["estimate"]["gain"])
                                                            for _r, _p, e in decisive),
        "sign_misses": [f"{run}: {pair}" for run, pair, e in decisive if np.sign(e["a_priori"]) != np.sign(e["estimate"]["gain"])],
        "median_abs_error": float(np.median(errors)) if errors else None,
        "A2_median_abs_error_at_most_1pt": bool(errors) and float(np.median(errors)) <= 0.01,
        "inside_interval": int(sum(e["estimate"]["ci"][0] <= e["a_priori"] <= e["estimate"]["ci"][1] for _r, _p, e in pairs))}
    dest = ROOT / "results/t6/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n")
    for run, r in result["runs"].items():
        print(f"== {run}: fidelity {json.dumps(r['fidelity'])}")
        for pair, e in r["comparisons"].items():
            v = e["estimate"]
            print(f"  {pair:30s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
                  f"{v['faster']}/{v['units']}; a priori {100 * e['a_priori']:+.2f}%; run costs {100 * e['model_run_costs']:+.2f}%")
    for hid, h in hypotheses.items():
        if h.get("run", True) is None:
            print(f"  {hid}: run missing")
            continue
        print(f"  {hid}: K={h['copy_repeat']} {h['comparison']}: {100 * h['measured']:+.2f}% "
              f"[{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] ({h['criterion']} zero) -> met {h['met']}")
    print(f"a priori: {json.dumps(result['a_priori'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
