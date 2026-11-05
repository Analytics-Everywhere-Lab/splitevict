"""T9 analysis: the decode step replayed from CUDA graphs on an H100 node, one split-phase worker.

Runs (``jobs/t9.sbatch``, ``interleaved_graphs.py``): ``data/runs/t9/olmoe/cal_B32_K<k>.json`` (calibration: three
arms, one pass, on the first 40 calibration sequences of T5's fresh OLMoE trace) and ``B32_K<k>.json`` (the three arms,
one pass, on T9's fresh trace: examples 315-324 of each domain, 40 sequences), for copy repeat K = 1, 2, 4.

The estimator is T5's (``analyze_t5`` helpers): per sequence and arm the run's one pass; every comparison the paired,
sequence-clustered, domain-stratified bootstrap (1,000 replicates); "faster" / "slower" means the 95% interval lies
entirely above / below zero. Fidelity: every arm feeds the model the same token stream in every
sequence (the graphed arithmetic can differ from the eager trace in the last bits, so routing against the trace is
reported, not required). The a-priori model (``dev_costs.a_priori``) takes every cost from the calibration run at the
same K (frozen in the job log before the fresh trace existed) and from the sealed run only each arm's measured copies per
token and the base arm's wall time. A1: every comparison predicted at least one percentage point from zero has a measured
gain of the same sign; A2: the median absolute error over all comparisons is at most one percentage point.

Reported, not tested: decayed LFU's time per token outside copies, copy issue and decisions (the policy-independent
time W0 plus the remainder) next to the eager runtime's in T7 at the same K, and each K's break-even copy times.

    python analysis/analyze_t9.py
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

from analyze_t5 import model_params, predicted_gain, rows, view  # noqa: E402

RUNS = ROOT / "data/runs/t9/olmoe"
EAGER = ROOT / "data/runs/t7/olmoe"
FIT = ROOT / "data/baselines/olmoe_frozen.json"
CAP = 32
KS = (1, 2, 4)
SEALED_ARMS = ("lfu_decay", "hobbit", "splitphase")
COMPARISONS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"))


def same_streams(arms: dict) -> bool:
    by_arm = {name: {r["source_sequence_id"]: r.get("generated_tokens") for r in view(a["sequences"], 0)}
              for name, a in arms.items()}
    first = next(iter(by_arm.values()))
    return all(v is not None for v in first.values()) and all(streams == first for streams in by_arm.values())


def outside_policy_ms(arm: dict, t_copy: float, t_issue: float) -> float:
    """Wall time per token minus copies, copy issue and the timed decisions: W0 plus the arm's remainder."""

    p = model_params(arm)
    return p["wall"] - p["demand"] * (t_copy + t_issue) - p["policy"]


def analyse(k: int, frozen: dict) -> dict | None:
    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, costs, per_token_wall

    path, cal_path = RUNS / f"B{CAP}_K{k}.json", RUNS / f"cal_B{CAP}_K{k}.json"
    if not path.exists() or not cal_path.exists():
        return None
    run, cal = json.loads(path.read_text()), json.loads(cal_path.read_text())
    dev = costs(cal)
    arms = run["arms"]
    fit = json.loads(FIT.read_text())
    out = {"copy_repeat": k, "fidelity": {}, "comparisons": {}, "arms": {},
           "costs_calibration": dev, "costs_frozen_equal": frozen[cal_path.name]["costs"] == dev,
           "breakeven_copy_ms": frozen[cal_path.name]["breakeven_copy_ms"], "graphs": run.get("graphs")}
    out["fidelity"] = {"copy_repeat_as_sealed": int(run.get("copy_repeat", 1)) == k,
                       "passes_as_sealed": int(run.get("passes", 1)) == 1,
                       "arms_as_sealed": sorted(arms) == sorted(SEALED_ARMS) and sorted(cal["arms"]) == sorted(SEALED_ARMS),
                       "hobbit_w_as_sealed": (run.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(CAP)]["w"],
                       "graphed": bool((run.get("graphs") or {}).get("graphed")),
                       "experts_in_graphs": (run.get("graphs") or {}).get("experts_in_graphs"),
                       "same_token_stream_every_arm": same_streams(arms),
                       "routing_vs_eager_trace": {name: a["totals"]["routing_match_fraction"] for name, a in arms.items()}}
    measured = {}
    for name, a in arms.items():
        p = model_params(a)
        measured[name] = {"demand": p["demand"], "prefetch": 0.0}
    out["measured_per_token"] = measured
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    for method, base in COMPARISONS:
        base_wall = per_token_wall(arms[base], 1)
        key = ("t9", CAP)
        w = paired_bootstrap({key: rows(view(arms[base]["sequences"], 0))},
                             {key: rows(view(arms[method]["sequences"], 0))}, ["t9"], CAP)
        est = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
               "units": w["paired_units"]}
        out["comparisons"][f"{method} vs {base}"] = {
            "a_priori": a_priori(dev, measured, method, base, base_wall),
            "calibration_gain": frozen[cal_path.name]["calibration_gain"][f"{method} vs {base}"],
            "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base]),
            "estimate": est, "above_zero": est["ci"][0] > 0, "below_zero": est["ci"][1] < 0}
    lfu_outside = outside_policy_ms(arms["lfu_decay"], dev["t_copy"], dev["t_issue"])
    eager = EAGER / f"B{CAP}_K{k}.json"
    eager_outside = None
    if eager.exists():
        e = json.loads(eager.read_text())["arms"]["lfu_decay"]
        t = e["totals"]
        eager_outside = outside_policy_ms(e, t["copy_ms"] / max(t["expert_transfers"], 1),
                                          t["issue_ms"] / max(t["expert_transfers"], 1))
    out["lfu_ms_per_token_outside_copies_and_decisions"] = {"graphed": lfu_outside, "eager_t7": eager_outside}
    return out


def main() -> None:
    frozen = json.loads((RUNS / "predictions_e19.json").read_text())
    result = {"host": "h100-node", "runs": {}}
    for k in KS:
        entry = analyse(k, frozen)
        if entry is not None:
            result["runs"][f"olmoe_B{CAP}_K{k}"] = entry
    hypotheses = {}
    for k in KS:
        r = result["runs"].get(f"olmoe_B{CAP}_K{k}")
        if r is None:
            continue
        for pair, e in r["comparisons"].items():
            primary = k == 4 and pair == "splitphase vs lfu_decay"
            if not primary and abs(e["calibration_gain"]) < 0.02:
                continue
            criterion = "above" if primary or e["calibration_gain"] > 0 else "below"
            hid = "P1" if primary else f"S-K{k}-{pair.replace(' ', '_')}"
            hypotheses[hid] = {"copy_repeat": k, "comparison": pair, "criterion": criterion,
                               "calibration_gain": e["calibration_gain"], "measured": e["estimate"]["gain"],
                               "ci": e["estimate"]["ci"], "a_priori": e["a_priori"],
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
    dest = ROOT / "results/t9/summary.json"
    dest.write_text(json.dumps(result, indent=1) + "\n")
    for run, r in result["runs"].items():
        print(f"== {run}: fidelity {json.dumps(r['fidelity'])}")
        print(f"   LFU ms/token outside copies and decisions: {json.dumps(r['lfu_ms_per_token_outside_copies_and_decisions'])}")
        print(f"   break-even copy ms: {json.dumps(r['breakeven_copy_ms'])}")
        for pair, e in r["comparisons"].items():
            v = e["estimate"]
            print(f"  {pair:24s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
                  f"{v['faster']}/{v['units']}; a priori {100 * e['a_priori']:+.2f}%; calibration {100 * e['calibration_gain']:+.2f}%")
    for hid, h in hypotheses.items():
        print(f"  {hid}: K={h['copy_repeat']} {h['comparison']}: {100 * h['measured']:+.2f}% "
              f"[{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] ({h['criterion']} zero) -> met {h['met']}")
    print(f"a priori: {json.dumps(result['a_priori'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
