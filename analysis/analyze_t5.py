"""T5 analysis: break-even predictions and the copy-cost sweep on an H100 node.

Runs (``jobs/t5.sbatch``): ``data/runs/t5/<model>/B<cap>.json`` (four arms, two passes) and the copy-cost sweep
``B<cap>_K<k>.json`` (four arms, one pass, every expert copy issued K times into its slot).

* fidelity: routing and tokens equal the model's fresh trace on every event of every arm and pass; every arm's
  per-sequence demand copies equal the simulator's (``h100_sim.run_arm`` on the fresh trace, the run's sequence order)
  in every pass; the runner's HOBBIT weight equals the sealed fit;
* the estimator: per sequence and arm, the faster of the passes (one-pass runs: that pass); every comparison is the
  paired, sequence-clustered, domain-stratified bootstrap (1,000 replicates) of ``analyze_caer.paired_bootstrap``;
  "faster" / "slower" means the 95% interval lies entirely above / below zero;
* the break-even model a priori (``dev_costs.a_priori``, the form checked post hoc on T4): every cost from the sealed
  development cost file ``results/t5/dev_costs.json`` (per-copy costs, each arm's timed policy cost and residual
  against decayed LFU), from the test run only the simulator's copies per token and the base arm's wall time (the
  denominator). Sweep points use ``sweep_costs`` (rule below). A1: every comparison predicted at least one
  point from zero has a measured gain of the same sign; A2: the median absolute error over all comparisons is at
  most one point.

    python analysis/analyze_t5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.append(str(ROOT / "src"))
OPT = Path(__file__).resolve().parents[1] / "measurement"

RUNS = ROOT / "data/runs/t5"
DEV_COSTS = ROOT / "results/t5/dev_costs.json"
BUNDLES = ROOT / "data/rankers"
FITS = ROOT / "data/baselines"
MODELS = {
    "mixtral": {"trace": ROOT / "data/traces/mixtral_t5", "spec": ROOT / "data/specs/mixtral_t5_replay_spec.json",
                "caps": (4, 3), "sweep": {}, "fit": FITS / "mixtral_frozen.json",
                "p2": BUNDLES / "mixtral_8x7b_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "mixtral_8x7b_frozen_lag_tok_trees200.pkl",
                "sync": BUNDLES / "mixtral_8x7b_dev_trees800.pkl"},
    "olmoe": {"trace": ROOT / "data/traces/olmoe_t5", "spec": ROOT / "data/specs/olmoe_t5_replay_spec.json",
              "caps": (32, 24), "sweep": {32: (2, 4, 16, 32)}, "fit": FITS / "olmoe_frozen.json",
              "p2": BUNDLES / "olmoe_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_frozen_lag_tok_trees200.pkl",
              "sync": ROOT / "data/rankers/olmoe_setdev_policy.pkl"},
    "qwen3": {"trace": ROOT / "data/traces/qwen3_t5", "spec": ROOT / "data/specs/qwen3_t5_replay_spec.json",
              "caps": (64, 48), "sweep": {64: (4, 16)}, "fit": FITS / "qwen3_frozen.json",
              "p2": BUNDLES / "qwen3_30b_a3b_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "qwen3_30b_a3b_frozen_lag_tok_trees200.pkl",
              "sync": BUNDLES / "qwen3_30b_a3b_dev_trees800.pkl"},
}
SEALED_ARMS = ("lfu_decay", "hobbit", "splitphase", "global_token_gbt")
COMPARISONS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"),
               ("global_token_gbt", "lfu_decay"), ("splitphase", "global_token_gbt"))
# (id, model, capacity, copy repeat K, method, base, criterion): "above" = 95% interval above zero, "below" = below
HYPOTHESES = (
    ("P1", "mixtral", 4, 1, "splitphase", "lfu_decay", "above"),
    ("S1", "mixtral", 3, 1, "splitphase", "lfu_decay", "above"),
    ("S2a", "mixtral", 4, 1, "splitphase", "hobbit", "above"),
    ("S2b", "mixtral", 3, 1, "splitphase", "hobbit", "above"),
    *((f"S3-{m}-{c}", m, c, 1, "global_token_gbt", "lfu_decay", "below")
      for m, c in (("mixtral", 4), ("mixtral", 3), ("olmoe", 32), ("olmoe", 24), ("qwen3", 64), ("qwen3", 48))),
    *((f"S4-{m}-{c}", m, c, 1, "splitphase", "global_token_gbt", "above")
      for m, c in (("mixtral", 4), ("mixtral", 3), ("olmoe", 32), ("olmoe", 24), ("qwen3", 64), ("qwen3", 48))),
    *((f"S5-{m}-{c}", m, c, 1, "splitphase", "lfu_decay", "below")
      for m, c in (("olmoe", 32), ("olmoe", 24), ("qwen3", 64), ("qwen3", 48))),
    *((f"S6-{m}-{c}", m, c, 1, "hobbit", "lfu_decay", "below")
      for m, c in (("olmoe", 32), ("olmoe", 24), ("qwen3", 64), ("qwen3", 48))),
    *((f"S7-split-K{k}", "olmoe", 32, k, "splitphase", "lfu_decay", "above") for k in (2, 4, 16, 32)),
    ("S7-sync-K2", "olmoe", 32, 2, "global_token_gbt", "lfu_decay", "below"),
    ("S7-sync-K4", "olmoe", 32, 4, "global_token_gbt", "lfu_decay", "below"),
    ("S7-sync-K16", "olmoe", 32, 16, "global_token_gbt", "lfu_decay", "above"),
    ("S7-sync-K32", "olmoe", 32, 32, "global_token_gbt", "lfu_decay", "above"),
    *((f"S8-K{k}", "qwen3", 64, k, "splitphase", "lfu_decay", "above") for k in (4, 16)),
)


def rows(seqs, key="wall_ms"):
    return [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r[key])) for r in seqs]


def view(seqs, which):
    """One row per sequence: pass ``which``, the faster pass ("min"), or the mean over passes ("mean")."""

    first = [r for r in seqs if r.get("pass", 0) == 0]
    if isinstance(which, int):
        return [r for r in seqs if r.get("pass", 0) == which]
    groups: dict[int, list[dict]] = {}
    for r in seqs:
        groups.setdefault(r["source_sequence_id"], []).append(r)
    out = []
    for r in first:
        g = groups[r["source_sequence_id"]]
        if which == "min":
            best = min(g, key=lambda x: x["wall_ms"])
            out.append(dict(best, sequence_position=r["sequence_position"]))
        else:
            out.append(dict(r, wall_ms=float(np.mean([x["wall_ms"] for x in g]))))
    return out


def model_params(arm: dict) -> dict:
    t = arm["totals"]
    tok = max(t["tokens"], 1)
    return {"wall": t["ms_per_token_wall"], "demand": t["expert_transfers"] / tok,
            "t_copy": t["copy_ms"] / max(t["expert_transfers"], 1), "t_issue": t["issue_ms"] / max(t["expert_transfers"], 1),
            "policy": t["ms_per_token_policy"] + t.get("ms_per_token_prefetch_policy", 0.0)}


def predicted_gain(method: dict, base: dict) -> float:
    """The break-even model with the run's own costs (a posteriori; shared copy and issue costs: the pair's mean)."""

    t_copy = (method["t_copy"] + base["t_copy"]) / 2
    t_issue = (method["t_issue"] + base["t_issue"]) / 2
    saved = (base["demand"] - method["demand"]) * (t_copy + t_issue)
    return (saved - (method["policy"] - base["policy"])) / base["wall"]


def _linear(k: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    (ka, va), (kb, vb) = a, b
    return va + (vb - va) * (k - ka) / (kb - ka)


def sweep_costs(dev: dict, model: str, cap: int, k: int) -> dict | None:
    """Development costs for a copy-cost sweep point.

    OLMoE (B=32) was measured at K=1 (``dev[olmoe][32]``), 4 and 16 (``dev[sweep][olmoe][32]``). At a measured K its
    entry is used as is. Otherwise the per-arm policy cost and residual come from the nearest measured K (K=2 from
    K=1, K=32 from K=16), and t_copy and t_issue are linear in K through the two nearest measured points (K=2
    interpolated between K=1 and 4; K=32 extrapolated from K=4 and 16). Qwen3 (B=64; no development sweep): its K=1
    entry, with t_copy and t_issue multiplied by OLMoE's ratio t(K)/t(1) at that K.
    """

    measured = {1: dev["olmoe"]["32"]}
    measured.update({int(key): v for key, v in dev["sweep"]["olmoe"]["32"].items()})

    def olmoe_at(kk: int) -> dict:
        if kk in measured:
            return measured[kk]
        known = sorted(measured)
        lower = [x for x in known if x < kk]
        upper = [x for x in known if x > kk]
        a, b = (lower[-1], upper[0]) if lower and upper else (known[-2], known[-1])
        nearest = min(known, key=lambda x: abs(x - kk))
        entry = dict(measured[nearest])
        entry["t_copy"] = _linear(kk, (a, measured[a]["t_copy"]), (b, measured[b]["t_copy"]))
        entry["t_issue"] = _linear(kk, (a, measured[a]["t_issue"]), (b, measured[b]["t_issue"]))
        return entry

    if model == "olmoe" and cap == 32:
        return olmoe_at(k)
    if model == "qwen3" and cap == 64:
        base, ref, one = dev["qwen3"]["64"], olmoe_at(k), measured[1]
        entry = dict(base)
        entry["t_copy"] = base["t_copy"] * ref["t_copy"] / one["t_copy"]
        entry["t_issue"] = base["t_issue"] * ref["t_issue"] / one["t_issue"]
        return entry
    return None


class SubsetPath:
    """The run's own sequences, in its order, as a workload (``iter_slices`` like a frozen path)."""

    def __init__(self, ids: list[int], spec: dict):
        self.name = "run_order"
        domain = {int(s["sequence_id"]): s["domain"] for s in spec["sequences"]}
        self.items = [(i, sid, domain[sid]) for i, sid in enumerate(ids)]

    def iter_slices(self, trace):
        slices = trace.sequence_slices()
        for position, sid, domain in self.items:
            yield SimpleNamespace(position=position, source_sequence_id=sid, domain=domain), slices[sid]


_TRACES: dict[str, object] = {}


def load_trace(info: dict):
    from fresh_trace import _paths

    key = str(info["trace"])
    if key not in _TRACES:
        _paths(OPT)
        from residency_headroom.trace import RoutingTrace

        _TRACES[key] = RoutingTrace.load(info["trace"] / "routing_trace.npz", validate=True)
    return _TRACES[key]


def analyse(model: str, cap: int, k: int, dev_costs: dict) -> dict | None:
    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, per_token_wall
    from h100_sim import run_arm

    info = MODELS[model]
    path = RUNS / model / (f"B{cap}.json" if k == 1 else f"B{cap}_K{k}.json")
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    arms = run["arms"]
    passes = int(run.get("passes", 1))
    spec = json.loads(info["spec"].read_text())
    trace = load_trace(info)
    fit = json.loads(info["fit"].read_text())
    out = {"model": model, "capacity": cap, "copy_repeat": int(run.get("copy_repeat", 1)), "passes": passes,
           "host": run.get("host", {}).get("hostname"), "fidelity": {}, "comparisons": {}, "arms": {},
           "simulated_per_token": {}}
    out["fidelity"]["copy_repeat_as_sealed"] = int(run.get("copy_repeat", 1)) == k
    out["fidelity"]["arms_as_sealed"] = sorted(arms) == sorted(SEALED_ARMS)
    arms = {name: a for name, a in arms.items() if name in SEALED_ARMS}  # anything else is reported, not analysed
    out["fidelity"]["hobbit_w_as_sealed"] = (run.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(cap)]["w"]
    order = [r["source_sequence_id"] for r in sorted(view(next(iter(arms.values()))["sequences"], 0),
                                                       key=lambda r: r["sequence_position"])]
    deploy = SubsetPath(order, spec)
    params = {"p2": info["p2"], "p1": info["p1"], "sync": info["sync"], "hobbit_w": fit["hobbit"][str(cap)]["w"]}
    for name, a in arms.items():
        t_by = a["totals_by_pass"]
        expected = run_arm(trace, deploy, name, cap, params)
        tokens = sum(r["tokens"] for r in view(a["sequences"], 0))
        out["simulated_per_token"][name] = {"demand": sum(e["demand"] for e in expected) / tokens, "prefetch": 0.0}
        equal = []
        for p in range(passes):
            live = sorted(view(a["sequences"], p), key=lambda r: r["sequence_position"])
            equal.append([e["demand"] for e in expected] == [r["expert_transfers"] for r in live])
        out["fidelity"][name] = {"routing_all_events": all(t["routing_matches"] == t["routing_events"] for t in t_by),
                                 "token_agreement": [t["token_agreement_fraction"] for t in t_by],
                                 "simulator_equal_every_pass": all(equal)}
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    dev = (dev_costs.get(model) or {}).get(str(cap)) if k == 1 else sweep_costs(dev_costs, model, cap, k)
    which = "min" if passes > 1 else 0
    for method, base in COMPARISONS:
        if method not in arms or base not in arms:
            continue
        base_wall = per_token_wall(arms[base], passes)
        entry = {"a_priori": None if dev is None else a_priori(dev, out["simulated_per_token"], method, base, base_wall),
                 "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base])}
        views = (which, 0, "mean") if passes > 1 else (0,)
        for v in views:
            key = ("t5", cap)
            w = paired_bootstrap({key: rows(view(arms[base]["sequences"], v))},
                                 {key: rows(view(arms[method]["sequences"], v))}, ["t5"], cap)
            entry[f"view_{v}"] = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]],
                                  "faster": w["sequences_improved"], "units": w["paired_units"]}
        main = entry[f"view_{which}"]
        entry["estimate"] = main
        entry["above_zero"] = main["ci"][0] > 0
        entry["below_zero"] = main["ci"][1] < 0
        out["comparisons"][f"{method} vs {base}"] = entry
    return out


def main() -> None:
    dev_costs = json.loads(DEV_COSTS.read_text())
    result = {"runs": {}}
    for model, info in MODELS.items():
        plan = [(cap, 1) for cap in info["caps"]] + [(cap, k) for cap, ks in info["sweep"].items() for k in ks]
        for cap, k in plan:
            entry = analyse(model, cap, k, dev_costs)
            if entry is not None:
                result["runs"][f"{model}_B{cap}" + ("" if k == 1 else f"_K{k}")] = entry
    hypotheses = {}
    for hid, model, cap, k, method, base, criterion in HYPOTHESES:
        e = result["runs"].get(f"{model}_B{cap}" + ("" if k == 1 else f"_K{k}"), {}).get("comparisons", {}).get(f"{method} vs {base}")
        if e is None:
            hypotheses[hid] = {"run": None}
            continue
        hypotheses[hid] = {"model": model, "capacity": cap, "copy_repeat": k, "comparison": f"{method} vs {base}",
                           "criterion": criterion, "measured": e["estimate"]["gain"], "ci": e["estimate"]["ci"],
                           "met": e["above_zero"] if criterion == "above" else e["below_zero"], "a_priori": e["a_priori"]}
    result["hypotheses"] = hypotheses
    pairs = [(run, pair, e) for run, r in result["runs"].items() for pair, e in r["comparisons"].items() if e["a_priori"] is not None]
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
    dest = ROOT / "results/t5/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n")
    for run, r in result["runs"].items():
        print(f"== {run}: fidelity {json.dumps(r['fidelity'])}")
        for pair, e in r["comparisons"].items():
            v = e["estimate"]
            ap = e["a_priori"]
            print(f"  {pair:30s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
                  f"{v['faster']}/{v['units']}; a priori {'--' if ap is None else f'{100 * ap:+.2f}%'}; "
                  f"run costs {100 * e['model_run_costs']:+.2f}%")
    for hid, h in hypotheses.items():
        if h.get("run", True) is None:
            print(f"  {hid}: run missing")
            continue
        print(f"  {hid}: {h['model']} B={h['capacity']} K={h['copy_repeat']} {h['comparison']}: {100 * h['measured']:+.2f}% "
              f"[{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] ({h['criterion']} zero) -> met {h['met']}")
    print(f"a priori: {json.dumps(result['a_priori'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
