"""T8 analysis: every demand copy read from an H100 node's NVMe drive with direct I/O.

Runs (``jobs/t8.sbatch``, one H100 node): ``data/runs/t8/olmoe/cal_B32_nvme.json`` (calibration: the four arms, one
pass, on the first 40 calibration sequences of T5's fresh OLMoE trace) and ``B32_nvme.json`` (the four arms, one pass,
on T8's fresh trace: examples 305-314 of each domain, 40 sequences).

The estimator and the fidelity checks are T5's, with the helpers imported from ``analyze_t5.py``: per sequence and arm
the run's one pass; every comparison the paired, sequence-clustered, domain-stratified bootstrap (1,000 replicates);
"faster" / "slower" means the 95% interval lies entirely above / below zero. The a-priori model
(``dev_costs.a_priori``) takes every cost from the calibration run (``dev_costs.costs``, the file frozen in the job log
before the fresh trace existed) and from the sealed run only the simulator's copies per token and the base arm's wall
time. A1: every comparison predicted at least one percentage point from zero has a measured gain of the same sign; A2:
the median absolute error over all comparisons is at most one percentage point.

    python analysis/analyze_t8.py
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

RUNS = ROOT / "data/runs/t8/olmoe"
BUNDLES = ROOT / "data/rankers"
INFO = {"trace": ROOT / "data/traces/olmoe_t8", "spec": ROOT / "data/specs/olmoe_t8_replay_spec.json",
        "fit": ROOT / "data/baselines/olmoe_frozen.json",
        "p2": BUNDLES / "olmoe_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_frozen_lag_tok_trees200.pkl",
        "sync": ROOT / "data/rankers/olmoe_setdev_policy.pkl"}
CAP = 32
SEALED_ARMS = ("lfu_decay", "hobbit", "splitphase", "global_token_gbt")
COMPARISONS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"),
               ("global_token_gbt", "lfu_decay"), ("splitphase", "global_token_gbt"))
# (id, method, base, criterion); S2's direction is the frozen calibration gain's sign, and S2 exists only if that gain
# lies at least two points from zero
HYPOTHESES = (("P1", "splitphase", "lfu_decay", "above"), ("S1", "splitphase", "hobbit", "above"),
              ("S2", "global_token_gbt", "lfu_decay", "frozen"), ("S3", "splitphase", "global_token_gbt", "above"))
CACHE_GROWTH_LIMIT_KIB = 1 << 20  # direct I/O: the page cache may not grow by 1 GiB over the arms (experts: 12.9 GB)


def nvme_fidelity(run: dict) -> dict:
    nv = run.get("nvme", {})
    probes = nv.get("probes", [])
    before, after = (probes[0].get("page_cache_kib"), probes[-1].get("page_cache_kib")) if probes else (None, None)
    growth = None if before is None or after is None else after - before
    return {"direct_io": bool(nv.get("direct_io")), "readback_equal": bool(nv.get("readback_equal")),
            "page_cache_growth_kib": growth,
            "page_cache_held": growth is not None and growth < CACHE_GROWTH_LIMIT_KIB,
            "probe_ms_mean": [p["ms_mean"] for p in probes], "repeat_ms": [[p["repeat_first_ms"], p["repeat_second_ms"]]
                                                                        for p in probes]}


def per_copy(arm: dict) -> dict:
    seqs = view(arm["sequences"], 0)
    copies = max(sum(r["expert_transfers"] for r in seqs), 1)
    return {"read_ms": sum(r.get("read_ms", 0.0) for r in seqs) / copies,
            "copy_ms": sum(r["copy_ms"] for r in seqs) / copies, "issue_ms": sum(r["issue_ms"] for r in seqs) / copies,
            "reads_equal_transfers": all(r.get("reads") == r["expert_transfers"] for r in seqs)}


def main() -> None:
    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, costs, per_token_wall
    from h100_sim import run_arm

    cal = json.loads((RUNS / "cal_B32_nvme.json").read_text())
    dev = costs(cal)
    frozen = json.loads((RUNS / "predictions_e18.json").read_text())["cal_B32_nvme.json"]
    run = json.loads((RUNS / "B32_nvme.json").read_text())
    arms = run["arms"]
    spec = json.loads(INFO["spec"].read_text())
    trace = load_trace(INFO)
    fit = json.loads(INFO["fit"].read_text())
    out = {"model": "olmoe", "capacity": CAP,
           "host": "h100-node", "fidelity": {}, "comparisons": {}, "arms": {}, "simulated_per_token": {},
           "costs_calibration": dev, "costs_frozen_equal": frozen["costs"] == dev,
           "breakeven_copy_ms": frozen["breakeven_copy_ms"], "per_copy": {}}
    out["fidelity"]["copy_repeat_as_sealed"] = int(run.get("copy_repeat", 1)) == 1
    out["fidelity"]["passes_as_sealed"] = int(run.get("passes", 1)) == 1
    out["fidelity"]["arms_as_sealed"] = sorted(arms) == sorted(SEALED_ARMS) and sorted(cal["arms"]) == sorted(SEALED_ARMS)
    out["fidelity"]["hobbit_w_as_sealed"] = (run.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(CAP)]["w"]
    out["fidelity"]["nvme_calibration"] = nvme_fidelity(cal)
    out["fidelity"]["nvme_sealed_run"] = nvme_fidelity(run)
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
        out["per_copy"][name] = per_copy(a)
        out["fidelity"][name] = {"routing_all_events": all(t["routing_matches"] == t["routing_events"] for t in t_by),
                                 "token_agreement": [t["token_agreement_fraction"] for t in t_by],
                                 "simulator_equal_every_pass": [e["demand"] for e in expected]
                                 == [r["expert_transfers"] for r in live],
                                 "reads_equal_transfers": out["per_copy"][name]["reads_equal_transfers"]}
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    for method, base in COMPARISONS:
        base_wall = per_token_wall(arms[base], 1)
        key = ("t8", CAP)
        w = paired_bootstrap({key: rows(view(arms[base]["sequences"], 0))},
                             {key: rows(view(arms[method]["sequences"], 0))}, ["t8"], CAP)
        est = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
               "units": w["paired_units"]}
        out["comparisons"][f"{method} vs {base}"] = {
            "a_priori": a_priori(dev, out["simulated_per_token"], method, base, base_wall),
            "calibration_gain": frozen["calibration_gain"][f"{method} vs {base}"],
            "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base]),
            "estimate": est, "above_zero": est["ci"][0] > 0, "below_zero": est["ci"][1] < 0}
    hypotheses = {}
    for hid, method, base, criterion in HYPOTHESES:
        e = out["comparisons"][f"{method} vs {base}"]
        if criterion == "frozen":
            if abs(e["calibration_gain"]) < 0.02:
                hypotheses[hid] = {"comparison": f"{method} vs {base}", "exists": False,
                                   "calibration_gain": e["calibration_gain"]}
                continue
            criterion = "above" if e["calibration_gain"] > 0 else "below"
        hypotheses[hid] = {"comparison": f"{method} vs {base}", "exists": True, "criterion": criterion,
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
    dest = ROOT / "results/t8/summary.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"fidelity: {json.dumps(out['fidelity'])}")
    for name, c in out["per_copy"].items():
        print(f"  {name:18s} per copy: read {c['read_ms']:.3f} ms, copy events {c['copy_ms']:.3f} ms, issue {c['issue_ms']:.3f} ms")
    for pair, e in out["comparisons"].items():
        v = e["estimate"]
        print(f"  {pair:30s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
              f"{v['faster']}/{v['units']}; a priori {100 * e['a_priori']:+.2f}%; calibration {100 * e['calibration_gain']:+.2f}%")
    for hid, h in hypotheses.items():
        if not h["exists"]:
            print(f"  {hid}: {h['comparison']}: no directional hypothesis (calibration gain {100 * h['calibration_gain']:+.2f}%)")
            continue
        print(f"  {hid}: {h['comparison']}: {100 * h['measured']:+.2f}% [{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] "
              f"({h['criterion']} zero) -> met {h['met']}")
    print(f"a priori: {json.dumps(out['a_priori'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
