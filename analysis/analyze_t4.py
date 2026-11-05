"""T4 analysis: the same next-layer prefetcher for both policies, OLMoE on an A40.

Per capacity (``data/runs/t4/B{32,24}.json``):

* fidelity: routing and tokens equal the fresh trace on every event of every arm and pass; the two prefetch
  arms computed identical guesses; every arm's per-sequence demand copies, prefetch copies and prefetch hits
  equal the simulator's (``prefetch_sim.run_on`` on the fresh trace, the spec's order, the run's own guesses)
  in every pass;
* the estimator: per sequence and arm, the faster of the passes (wall time); the comparison is the paired,
  sequence-clustered, domain-stratified bootstrap (1,000 replicates) of ``analyze_caer.paired_bootstrap``;
  pass 0 alone and the mean over passes are reported as robustness views;
* the break-even model with the run's measured parameters, for every comparison:
  predicted gain = (dDemand * (t_copy + t_issue) + dPrefetch * t_pf_issue - dPolicy) / W_base.

    python analysis/analyze_t4.py
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
OPT = Path(__file__).resolve().parents[1] / "measurement"

RUNS = ROOT / "data/runs/t4"
SPEC = ROOT / "data/specs/olmoe_t4_replay_spec.json"
TRACE = ROOT / "data/traces/olmoe_t4"
BUNDLES = {"p2": ROOT / "data/rankers/olmoe_frozen_lag_req_trees200.pkl",
           "p1": ROOT / "data/rankers/olmoe_frozen_lag_tok_trees200.pkl"}
PRIMARY = ("splitphase+pf", "lfu_decay+pf", 24)
COMPARISONS = (("splitphase+pf", "lfu_decay+pf"), ("splitphase", "lfu_decay"), ("lfu_decay+pf", "lfu_decay"),
               ("splitphase+pf", "splitphase"), ("splitphase+pf", "lfu_decay"))
# the pre-registered hypotheses: (id, method, base, capacity, predicted gain, predicted range); "met" means the
# 95% interval of the faster-pass estimator lies above zero
HYPOTHESES = (
    ("P1", "splitphase+pf", "lfu_decay+pf", 24, 0.014, (0.005, 0.025)),
    ("S1", "splitphase+pf", "lfu_decay+pf", 32, 0.005, (-0.003, 0.013)),
    ("S2a", "splitphase", "lfu_decay", 32, 0.031, (0.020, 0.045)),
    ("S2b", "splitphase", "lfu_decay", 24, 0.037, (0.025, 0.050)),
    ("S3a", "lfu_decay+pf", "lfu_decay", 32, 0.033, (0.015, 0.050)),
    ("S3b", "lfu_decay+pf", "lfu_decay", 24, 0.070, (0.045, 0.095)),
)
RESIDUAL_RANGE = (0.005, 0.020)  # S4: the model's over-prediction of each split-phase gain, in points of gain


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
    demand = t["expert_transfers"] / tok
    prefetch = t["prefetch_transfers"] / tok
    return {"wall": t["ms_per_token_wall"], "demand": demand, "prefetch": prefetch,
            "t_copy": t["copy_ms"] / max(t["expert_transfers"], 1), "t_issue": t["issue_ms"] / max(t["expert_transfers"], 1),
            "t_pf_issue": t["prefetch_issue_ms"] / max(t["prefetch_transfers"], 1) if t["prefetch_transfers"] else 0.0,
            "policy": t["ms_per_token_policy"] + t["ms_per_token_prefetch_policy"]}


def predicted_gain(method: dict, base: dict) -> float:
    """The break-even model with this run's measured costs (shared copy and issue costs: the pair's mean)."""

    t_copy = (method["t_copy"] + base["t_copy"]) / 2
    t_issue = (method["t_issue"] + base["t_issue"]) / 2
    pf_costs = [p["t_pf_issue"] for p in (method, base) if p["prefetch"]]
    t_pf = float(np.mean(pf_costs)) if pf_costs else 0.0
    saved = (base["demand"] - method["demand"]) * (t_copy + t_issue) + (base["prefetch"] - method["prefetch"]) * t_pf
    return (saved - (method["policy"] - base["policy"])) / base["wall"]


def simulate(run: dict, capacity: int, arm: str, guesses_path: Path):
    import prefetch_sim
    from fresh_trace import _paths
    from unseen_sim import SpecPath

    _paths(OPT)
    from residency_headroom.trace import RoutingTrace

    trace = RoutingTrace.load(TRACE / "routing_trace.npz", validate=True)
    spec = json.loads(SPEC.read_text())
    path = SpecPath(spec, trace)
    pf = run["arms"][arm].get("prefetch")
    guesses = None
    if pf is not None:
        data = np.load(guesses_path)

        class Live:
            variant = "live"
            meta = {"spec": str(guesses_path), "device": "live"}
            by_sequence = {int(k.split("/")[1]): data[k].astype(np.int64) for k in data.files if k.startswith(f"{arm}/")}

            def get(self, sid, token, layer, m):
                return self.by_sequence[sid][token, layer, :m]

        guesses = Live()
    policy = "lfu" if arm.startswith("lfu") else "splitphase"
    wrong = run["prefetch"]["lfu_wrong_rule"] if policy == "lfu" else "last"
    return prefetch_sim.run_on(trace, path, policy, capacity, guesses, (pf or {}).get("m", 0), (pf or {}).get("q", 0),
                               BUNDLES, wrong)["per_sequence"]


def analyse(capacity: int) -> dict | None:
    from paired_bootstrap import paired_bootstrap

    path = RUNS / f"B{capacity}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    arms = run["arms"]
    passes = int(run.get("passes", 1))
    out = {"capacity": capacity, "passes": passes, "copy_api": run.get("copy_api"), "prefetch": run["prefetch"],
           "placement": {k: run["placement"][k] for k in ("workers", "main", "excluded_siblings")} if run.get("placement") else None,
           "fidelity": {}, "comparisons": {}, "arms": {}}
    # fidelity
    for name, a in arms.items():
        t_by = a["totals_by_pass"]
        out["fidelity"][name] = {
            "routing_all_events": all(t["routing_matches"] == t["routing_events"] for t in t_by),
            "token_agreement": [t["token_agreement_fraction"] for t in t_by]}
    guess = np.load(path.with_suffix(".guesses.npz")) if path.with_suffix(".guesses.npz").exists() else None
    if guess is not None:
        a = sorted(k for k in guess.files if k.startswith("lfu_decay+pf/"))
        out["fidelity"]["guesses_identical_across_prefetch_arms"] = all(
            np.array_equal(guess[k], guess["splitphase+pf/" + k.split("/")[1]]) for k in a)
    for name in arms:
        expected = simulate(run, capacity, name, path.with_suffix(".guesses.npz"))
        per_pass = []
        for p in range(passes):
            live = view(arms[name]["sequences"], p)
            per_pass.append([e["demand"] for e in expected] == [r["expert_transfers"] for r in live]
                            and [e["prefetch"] for e in expected] == [r.get("prefetch_transfers", 0) for r in live]
                            and [e["hits"] for e in expected] == [r.get("prefetch_hits", 0) for r in live])
        out["fidelity"][name]["simulator_equal_every_pass"] = all(per_pass)
    # arms and comparisons
    params = {name: model_params(a) for name, a in arms.items()}
    out["arms"] = params
    for method, base in COMPARISONS:
        if method not in arms or base not in arms:
            continue
        entry = {"predicted_gain_model": predicted_gain(params[method], params[base])}
        for which in ("min", 0, "mean"):
            key = ("t4", capacity)
            w = paired_bootstrap({key: rows(view(arms[base]["sequences"], which))},
                                 {key: rows(view(arms[method]["sequences"], which))}, ["t4"], capacity)
            entry[f"view_{which}"] = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]],
                                      "faster": w["sequences_improved"], "units": w["paired_units"]}
        entry["criterion_met"] = entry["view_min"]["ci"][0] > 0
        out["comparisons"][f"{method} vs {base}"] = entry
    return out


def main() -> None:
    result = {"capacities": {}}
    for capacity in (32, 24):
        entry = analyse(capacity)
        if entry is not None:
            result["capacities"][str(capacity)] = entry
    m, b, c = PRIMARY
    primary = result["capacities"].get(str(c), {}).get("comparisons", {}).get(f"{m} vs {b}")
    result["primary"] = None if primary is None else {"comparison": f"{m} vs {b}", "capacity": c,
                                                      "met": primary["criterion_met"], **primary["view_min"]}
    hypotheses = {}
    for hid, method, base, cap, predicted, (lo, hi) in HYPOTHESES:
        e = result["capacities"].get(str(cap), {}).get("comparisons", {}).get(f"{method} vs {base}")
        if e is None:
            continue
        gain = e["view_min"]["gain"]
        hypotheses[hid] = {"comparison": f"{method} vs {base}", "capacity": cap, "predicted": predicted,
                           "predicted_range": [lo, hi], "measured": gain, "ci": e["view_min"]["ci"],
                           "met": e["criterion_met"], "inside_predicted_range": lo <= gain <= hi}
    residuals = {}
    for cap, entry in result["capacities"].items():
        for pair, e in entry["comparisons"].items():
            if pair in ("splitphase vs lfu_decay", "splitphase+pf vs lfu_decay+pf"):  # same prefetch condition
                over = e["predicted_gain_model"] - e["view_min"]["gain"]
                residuals[f"{pair} @B={cap}"] = {"model": e["predicted_gain_model"], "measured": e["view_min"]["gain"],
                                                 "over_prediction": over,
                                                 "inside_S4_range": RESIDUAL_RANGE[0] <= over <= RESIDUAL_RANGE[1]}
    result["hypotheses"] = hypotheses
    result["S4_model_over_prediction"] = residuals
    dest = ROOT / "results/t4/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n")
    for cap, entry in result["capacities"].items():
        print(f"== B={cap}: fidelity {json.dumps(entry['fidelity'])}")
        for pair, e in entry["comparisons"].items():
            v = e["view_min"]
            print(f"  {pair:30s} {100 * v['gain']:+6.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
                  f"{v['faster']}/{v['units']} faster; model {100 * e['predicted_gain_model']:+.2f}%; "
                  f"pass 0 {100 * e['view_0']['gain']:+.2f}%, mean {100 * e['view_mean']['gain']:+.2f}%")
    for hid, h in result["hypotheses"].items():
        print(f"  {hid}: {h['comparison']} B={h['capacity']}: {100 * h['measured']:+.2f}% [{100 * h['ci'][0]:+.2f}, "
              f"{100 * h['ci'][1]:+.2f}] predicted {100 * h['predicted']:+.1f}% -> met {h['met']}, in range "
              f"{h['inside_predicted_range']}")
    for key, r in result["S4_model_over_prediction"].items():
        print(f"  S4 {key}: model {100 * r['model']:+.2f}%, measured {100 * r['measured']:+.2f}%, over-prediction "
              f"{100 * r['over_prediction']:+.2f} points (in range {r['inside_S4_range']})")
    print(f"primary: {result['primary']}; wrote {dest}")


if __name__ == "__main__":
    main()
