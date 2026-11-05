"""T10 analysis: the graphed decode step of T9 with two split-phase workers, B = 32, K = 1.

Runs (``jobs/t10.sbatch``, the H100 node of T9), the decode step replayed from CUDA graphs as in T9, with
split-phase's scores computed by two worker processes (``interleaved_graphs_workers.py --splitphase-workers 2``):
``data/runs/t10/olmoe/cal_B32_K1.json`` (calibration: three arms, one pass, on the first 40 calibration
sequences of T5's fresh OLMoE trace) and ``B32_K1.json`` (the three arms, one pass, on T10's fresh trace: examples
335-344 of each domain, 40 sequences), at the H100's own copy cost (K = 1).

Estimator, a-priori model, A1 and A2 as in T9 (``analyze_t5`` helpers and ``dev_costs``). P1: split-phase vs decayed
LFU above zero. The other two comparisons are tested in the direction of the frozen calibration gain, if it is at least
two points from zero. Fidelity: T9's criteria, and the split-phase arm ran two workers, each alone on its own core, in
the calibration and in the sealed run.

Reported, not tested: split-phase's wait for its scores, its decision time and each worker's busy time per token,
next to T9's one-worker run at K = 1.

    python analysis/analyze_t10.py
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
from analyze_t9 import same_streams  # noqa: E402

RUNS = ROOT / "data/runs/t10/olmoe"
ONE_WORKER = ROOT / "data/runs/t9/olmoe/B32_K1.json"
FIT = ROOT / "data/baselines/olmoe_frozen.json"
CAP = 32
K = 1
WORKERS = 2
SEALED_ARMS = ("lfu_decay", "hobbit", "splitphase")
COMPARISONS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"))


def workers_as_sealed(run: dict) -> bool:
    """Two workers in the split-phase arm, on two distinct cores that the decode process does not use."""

    stats = (run.get("splitphase") or {}).get("splitphase") or {}
    cpus = stats.get("cpus") or []
    main = set((run.get("placement") or {}).get("main") or [])
    return (run.get("splitphase_workers") == WORKERS and stats.get("workers") == WORKERS
            and len(stats.get("worker_busy_ms_per_token") or []) == WORKERS
            and len(set(cpus)) == WORKERS and not set(cpus) & main)


def worker_view(run: dict) -> dict:
    stats = run["splitphase"]["splitphase"]
    keys = ("workers", "cpus", "wait_ms_per_token", "decide_ms_per_token", "post_ms_per_token", "worker_busy_ms_per_token")
    return {key: stats[key] for key in keys} | {"split_ms_per_token": run["arms"]["splitphase"]["totals"]["ms_per_token_wall"]}


def main() -> None:
    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, costs, per_token_wall

    frozen = json.loads((RUNS / "predictions_e24.json").read_text())
    cal_path = RUNS / f"cal_B{CAP}_K{K}.json"
    run, cal = json.loads((RUNS / f"B{CAP}_K{K}.json").read_text()), json.loads(cal_path.read_text())
    dev = costs(cal)
    arms = run["arms"]
    fit = json.loads(FIT.read_text())
    graphs = [r.get("graphs") or {} for r in (run, cal)]
    out = {"host": "h100-node", "copy_repeat": K, "costs_calibration": dev,
           "costs_frozen_equal": frozen[cal_path.name]["costs"] == dev,
           "breakeven_copy_ms": frozen[cal_path.name]["breakeven_copy_ms"], "graphs": run.get("graphs"),
           "comparisons": {}}
    out["fidelity"] = {"copy_repeat_as_sealed": int(run.get("copy_repeat", 1)) == K == int(cal.get("copy_repeat", 1)),
                       "passes_as_sealed": int(run.get("passes", 1)) == 1 == int(cal.get("passes", 1)),
                       "arms_as_sealed": sorted(arms) == sorted(SEALED_ARMS) and sorted(cal["arms"]) == sorted(SEALED_ARMS),
                       "hobbit_w_as_sealed": (run.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(CAP)]["w"],
                       "graphed": all(bool(g.get("graphed")) for g in graphs),
                       "experts_in_graphs": graphs[0].get("experts_in_graphs"),
                       "experts_in_graphs_same_every_run": graphs[0].get("experts_in_graphs") == graphs[1].get("experts_in_graphs"),
                       "two_workers_sealed_run": workers_as_sealed(run),
                       "two_workers_calibration": workers_as_sealed(cal),
                       "same_token_stream_every_arm": same_streams(arms),
                       "routing_vs_eager_trace": {name: a["totals"]["routing_match_fraction"] for name, a in arms.items()}}
    measured = {name: {"demand": model_params(a)["demand"], "prefetch": 0.0} for name, a in arms.items()}
    out["measured_per_token"] = measured
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    for method, base in COMPARISONS:
        key = ("t10", CAP)
        w = paired_bootstrap({key: rows(view(arms[base]["sequences"], 0))},
                             {key: rows(view(arms[method]["sequences"], 0))}, ["t10"], CAP)
        est = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
               "units": w["paired_units"]}
        out["comparisons"][f"{method} vs {base}"] = {
            "a_priori": a_priori(dev, measured, method, base, per_token_wall(arms[base], 1)),
            "calibration_gain": frozen[cal_path.name]["calibration_gain"][f"{method} vs {base}"],
            "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base]),
            "estimate": est, "above_zero": est["ci"][0] > 0, "below_zero": est["ci"][1] < 0}
    out["workers"] = {"sealed": worker_view(run), "calibration": worker_view(cal),
                      "t9_one_worker": worker_view(json.loads(ONE_WORKER.read_text())) if ONE_WORKER.exists() else None}
    hypotheses = {}
    for pair, e in out["comparisons"].items():
        primary = pair == "splitphase vs lfu_decay"
        if not primary and abs(e["calibration_gain"]) < 0.02:
            continue
        criterion = "above" if primary or e["calibration_gain"] > 0 else "below"
        hypotheses["P1" if primary else f"S-{pair.replace(' ', '_')}"] = {
            "comparison": pair, "criterion": criterion, "calibration_gain": e["calibration_gain"],
            "measured": e["estimate"]["gain"], "ci": e["estimate"]["ci"], "a_priori": e["a_priori"],
            "met": e["above_zero"] if criterion == "above" else e["below_zero"]}
    out["hypotheses"] = hypotheses
    pairs = list(out["comparisons"].items())
    decisive = [(pair, e) for pair, e in pairs if abs(e["a_priori"]) >= 0.01]
    errors = [abs(e["a_priori"] - e["estimate"]["gain"]) for _pair, e in pairs]
    out["a_priori"] = {
        "comparisons": len(pairs), "decisive": len(decisive),
        "A1_sign_agreement_decisive": bool(decisive) and all(np.sign(e["a_priori"]) == np.sign(e["estimate"]["gain"])
                                                            for _p, e in decisive),
        "sign_misses": [pair for pair, e in decisive if np.sign(e["a_priori"]) != np.sign(e["estimate"]["gain"])],
        "median_abs_error": float(np.median(errors)),
        "A2_median_abs_error_at_most_1pt": float(np.median(errors)) <= 0.01,
        "inside_interval": int(sum(e["estimate"]["ci"][0] <= e["a_priori"] <= e["estimate"]["ci"][1] for _p, e in pairs))}
    dest = ROOT / "results/t10/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"== olmoe_B{CAP}_K{K}, two workers: fidelity {json.dumps(out['fidelity'])}")
    print(f"   break-even copy ms: {json.dumps(out['breakeven_copy_ms'])}")
    for pair, e in out["comparisons"].items():
        v = e["estimate"]
        print(f"  {pair:24s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
              f"{v['faster']}/{v['units']}; a priori {100 * e['a_priori']:+.2f}%; calibration {100 * e['calibration_gain']:+.2f}%")
    for name, w in out["workers"].items():
        if w is not None:
            busy = ", ".join(f"{b:.2f}" for b in w["worker_busy_ms_per_token"])
            print(f"  workers ({name}): {w['workers']} on CPUs {w['cpus']}, busy [{busy}], wait {w['wait_ms_per_token']:.2f}, "
                  f"decide {w['decide_ms_per_token']:.2f}, split-phase {w['split_ms_per_token']:.2f} ms/token")
    for hid, h in hypotheses.items():
        print(f"  {hid}: {h['comparison']}: {100 * h['measured']:+.2f}% [{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] "
              f"({h['criterion']} zero) -> met {h['met']}")
    print(f"a priori: {json.dumps(out['a_priori'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
