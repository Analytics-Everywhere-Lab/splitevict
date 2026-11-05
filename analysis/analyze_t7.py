"""T7 analysis: choosing the policy on each host, with free-running decoding on an H100 node.

Runs (``jobs/t7.sbatch``, free-running decoding, ``interleaved_freerun.py``), per model
(OLMoE B=32, Qwen3 B=64) under ``data/runs/t7/<model>/``:

* calibration ``cal_B<cap>_K<k>.json``: decayed LFU and split-phase, one pass, the first 40 calibration sequences of
  T5's fresh trace of the model, every expert copy issued K times (K = 1, 2, 4);
* decisions ``decisions_B<cap>_K<k>.json`` (per-host selector: ``selector.py decide cal_..._K<k>.json --ks 1``) and
  ``decisions_B<cap>_extrapolated.json`` (``selector.py decide cal_..._K1.json --ks 1 2 4``), written before T7's
  fresh trace existed;
* evaluation ``B<cap>_K<k>.json``: the same two arms, one pass, on T7's fresh trace (examples 285-304 per domain);
* the job log, copied to ``data/runs/t7/t7.log``: its first "DECISIONS FROZEN" block must list every
  decision file's sha256 before the first fresh trace is generated.

The estimator is the one of T5 and T6: per sequence and arm the run's one pass; the paired, sequence-clustered,
domain-stratified bootstrap of ``analyze_caer.paired_bootstrap`` (1,000 replicates). ``selector.evaluate`` classes a
setting as a clear loss (interval entirely below zero) or a clear gain (interval entirely above one point).

    python analysis/analyze_t7.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.append(str(ROOT / "src"))

from analyze_t5 import SubsetPath, load_trace, rows, view  # noqa: E402
from selector import BASE, MARGIN, METHOD, evaluate  # noqa: E402

RUNS = ROOT / "data/runs/t7"
EXPECTED = ROOT / "results/t7/expected.json"
BUNDLES = ROOT / "data/rankers"
MODELS = {
    "olmoe": {"cap": 32, "trace": ROOT / "data/traces/olmoe_t7", "spec": ROOT / "data/specs/olmoe_t7_replay_spec.json",
              "calibration_spec": ROOT / "data/specs/olmoe_t5_replay_spec.json",
              "p2": BUNDLES / "olmoe_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_frozen_lag_tok_trees200.pkl"},
    "qwen3": {"cap": 64, "trace": ROOT / "data/traces/qwen3_t7", "spec": ROOT / "data/specs/qwen3_t7_replay_spec.json",
              "calibration_spec": ROOT / "data/specs/qwen3_t5_replay_spec.json",
              "p2": BUNDLES / "qwen3_30b_a3b_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "qwen3_30b_a3b_frozen_lag_tok_trees200.pkl"},
}
KS = (1, 2, 4)
ARMS = (BASE, METHOD)
CALIBRATION_SEQUENCES = 40
OFFSET, PER_DOMAIN = 285, 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_before_traces(log: Path, decisions: list[Path]) -> bool:
    """The job log's first "DECISIONS FROZEN" block lists every decision file's sha256, before any fresh trace."""

    if not log.exists() or not all(p.exists() for p in decisions):
        return False
    lines = log.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if "DECISIONS FROZEN" in line), None)
    trace = next((i for i, line in enumerate(lines) if "generate fresh trace" in line), len(lines))
    if start is None or start > trace:
        return False
    listed = {line.split()[0] for line in lines[start + 1:trace] if len(line.split()) == 2 and len(line.split()[0]) == 64}
    return all(sha256(p) in listed for p in decisions)


def run_checks(run: dict, k: int) -> dict:
    return {"free_running": run.get("decoding") == "free_running" and run.get("runner") == "interleaved_freerun.py",
            "copy_repeat_as_sealed": int(run.get("copy_repeat", 1)) == k,
            "passes_as_sealed": int(run.get("passes", 1)) == 1,
            "arms_as_sealed": sorted(run["arms"]) == sorted(ARMS)}


def tokens_exact(arm: dict, spec: dict) -> bool:
    """Free-running: every sequence ran to the recorded length and agreed with the recorded token at every step."""

    length = {int(s["sequence_id"]): len(s["generated_token_ids"]) for s in spec["sequences"]}
    return all(r["tokens"] == length[r["source_sequence_id"]] and r["token_agreements"] == r["token_comparisons"]
               and bool(r["first_token_agrees"]) for r in arm["sequences"])


def percentiles(arm: dict) -> dict:
    values = np.asarray([v for r in view(arm["sequences"], 0) for v in r["token_wall_ms"]], dtype=np.float64)
    return {f"p{q:g}": float(np.percentile(values, q)) for q in (50, 90, 99)} | {"mean": float(values.mean())}


def calibration(model: str, info: dict, k: int) -> dict | None:
    cap = info["cap"]
    path = RUNS / model / f"cal_B{cap}_K{k}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    spec = json.loads(info["calibration_spec"].read_text())
    ids = [int(v) for v in spec["calibration_sequence_ids"][:CALIBRATION_SEQUENCES]]
    out = {"checks": run_checks(run, k), "elapsed_seconds": run.get("elapsed_seconds"), "host": run["host"].get("hostname")}
    out["checks"]["calibration_trace"] = run.get("trace_hash") == spec["trace_hash"]
    out["checks"]["calibration_sequences"] = all(
        [r["source_sequence_id"] for r in sorted(view(a["sequences"], 0), key=lambda r: r["sequence_position"])] == ids
        for a in run["arms"].values())
    # reported, not a fidelity criterion: T5's trace was generated on another node, and the calibration only measures costs
    out["reproduces_t5_trace"] = all(tokens_exact(a, spec) for a in run["arms"].values())
    decisions = RUNS / model / f"decisions_B{cap}_K{k}.json"
    if decisions.exists():
        d = json.loads(decisions.read_text())
        out["decision"] = d["decisions"]["1"]
        out["checks"]["decision_reads_this_calibration"] = d["calibration_sha256"] == sha256(path)
        out["checks"]["decision_margin_as_sealed"] = d["margin"] == MARGIN
    return out


def evaluation(model: str, info: dict, k: int) -> dict | None:
    from paired_bootstrap import paired_bootstrap
    from h100_sim import run_arm

    cap = info["cap"]
    path = RUNS / model / f"B{cap}_K{k}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    spec = json.loads(info["spec"].read_text())
    arms = run["arms"]
    out = {"checks": run_checks(run, k), "host": run["host"].get("hostname"), "elapsed_seconds": run.get("elapsed_seconds")}
    out["checks"]["fresh_trace"] = run.get("trace_hash") == spec["trace_hash"]
    out["checks"]["prompt_offset"] = int(spec["trace_generation"].get("prompt_offset", -1)) == OFFSET
    out["checks"]["sequences"] = len(view(arms[BASE]["sequences"], 0)) == 4 * PER_DOMAIN
    order = [r["source_sequence_id"] for r in sorted(view(arms[BASE]["sequences"], 0), key=lambda r: r["sequence_position"])]
    deploy = SubsetPath(order, spec)
    trace = load_trace(info)
    params = {"p2": info["p2"], "p1": info["p1"]}
    for name, a in arms.items():
        expected = run_arm(trace, deploy, name, cap, params)
        live = sorted(view(a["sequences"], 0), key=lambda r: r["sequence_position"])
        out["checks"][f"{name}_routing_all_events"] = all(t["routing_matches"] == t["routing_events"] for t in a["totals_by_pass"])
        out["checks"][f"{name}_tokens_exact"] = tokens_exact(a, spec)
        out["checks"][f"{name}_simulator_equal"] = [e["demand"] for e in expected] == [r["expert_transfers"] for r in live]
    key = ("t7", cap)
    w = paired_bootstrap({key: rows(view(arms[BASE]["sequences"], 0))}, {key: rows(view(arms[METHOD]["sequences"], 0))},
                         ["t7"], cap)
    out["measured"] = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
                       "units": w["paired_units"]}
    out["arms"] = {}
    for name, a in arms.items():
        t = a["totals"]
        tok = max(t["tokens"], 1)
        out["arms"][name] = {"wall": t["ms_per_token_wall"], "copies_per_token": t["expert_transfers"] / tok,
                             "timed_copy_issue_per_copy": (t["copy_ms"] + t["issue_ms"]) / max(t["expert_transfers"], 1),
                             "policy": t["ms_per_token_policy"], "token_wall": percentiles(a),
                             "decode_mhz": float(np.mean([m for r in a["sequences"] for m in r.get("token_mhz", [])] or [np.nan]))}
    out["worker"] = run.get("splitphase")
    return out


def exposed_copy_time(evaluations: dict) -> dict:
    """Per arm, the wall time each extra copy-repeat adds per copy, as a share of the timed extra copy time (K vs 1)."""

    out = {}
    for model in MODELS:
        one = evaluations.get(f"{model}_K1")
        for k in KS[1:]:
            other = evaluations.get(f"{model}_K{k}")
            if one is None or other is None:
                continue
            for name in ARMS:
                a, b = one["arms"][name], other["arms"][name]
                extra_timed = b["copies_per_token"] * (b["timed_copy_issue_per_copy"] - a["timed_copy_issue_per_copy"])
                out[f"{model}_K{k}_{name}"] = (b["wall"] - a["wall"]) / extra_timed if extra_timed else None
    return out


def main() -> None:
    expected = json.loads(EXPECTED.read_text())["settings"]
    result = {"calibration": {}, "evaluation": {}, "extrapolated_checks": {},
              "hypotheses": {}}
    per_host, extrapolated, measured = {}, {}, {}
    extrapolated_gain, per_host_gain = {}, {}
    for model, info in MODELS.items():
        cap = info["cap"]
        xpath = RUNS / model / f"decisions_B{cap}_extrapolated.json"
        x = json.loads(xpath.read_text()) if xpath.exists() else None
        if x is not None:
            result["extrapolated_checks"][model] = {
                "reads_the_K1_calibration": x["calibration_sha256"] == sha256(RUNS / model / f"cal_B{cap}_K1.json"),
                "margin_as_sealed": x["margin"] == MARGIN}
        for k in KS:
            key = f"{model}_K{k}"
            c = calibration(model, info, k)
            e = evaluation(model, info, k)
            result["calibration"][key], result["evaluation"][key] = c, e
            if c is not None and "decision" in c:
                per_host[key] = c["decision"]["policy"]
                per_host_gain[key] = c["decision"]["predicted_gain"]
            if x is not None:
                extrapolated[key] = x["decisions"][str(k)]["policy"]
                extrapolated_gain[key] = x["decisions"][str(k)]["predicted_gain"]
            if e is not None:
                measured[key] = e["measured"]
    result["selector_per_host"] = evaluate(per_host, measured)
    result["selector_extrapolated"] = evaluate(extrapolated, measured)
    sel = result["selector_per_host"]
    h = result["hypotheses"]
    h["P1"] = {"met": bool(sel["avoids_every_clear_loss"] and sel["keeps_every_clear_gain"] and sel["clear_losses"] >= 1
                           and sel["clear_gains"] >= 1 and len(measured) == 6 and set(per_host) == set(measured)),
               "clear_losses": sel["clear_losses"], "clear_gains": sel["clear_gains"], "settings_measured": len(measured),
               "settings_decided": len(per_host)}
    # S1: per-host decisions equal the sealed expectations (from T5's runs) wherever the expected gain is at least a
    # point from the margin
    s1 = {}
    for model, info in MODELS.items():
        for k in KS:
            exp = expected[f"{model}_B{info['cap']}"]["per_host"][str(k)]
            key = f"{model}_K{k}"
            if exp is not None and abs(exp["predicted_gain"] - MARGIN) >= 0.01 and key in per_host:
                s1[key] = per_host[key] == exp["policy"]
    h["S1"] = {"met": bool(s1) and all(s1.values()) and len(s1) == 5, "settings": s1}
    r = sel["regret_sum"]
    h["S2"] = {"met": r["selector"] < r["always_splitphase"] and r["selector"] < r["always_lfu"], "regret_sum": r}
    over = {key: extrapolated_gain[key] - measured[key]["gain"] for key in ("qwen3_K2", "qwen3_K4")
            if key in extrapolated_gain and key in measured}
    h["S3"] = {"met": len(over) == 2 and all(v > 0 for v in over.values()), "over_prediction": over}
    pairs = [(per_host_gain[key], measured[key]["gain"]) for key in measured if key in per_host_gain]
    decisive = [(p, m) for p, m in pairs if abs(p) >= 0.01]
    errors = [abs(p - m) for p, m in pairs]
    h["S4"] = {"sign_agreement_decisive": bool(decisive) and all(np.sign(p) == np.sign(m) for p, m in decisive),
               "decisive": len(decisive), "median_abs_error": float(np.median(errors)) if errors else None}
    h["S4"]["met"] = bool(h["S4"]["sign_agreement_decisive"] and len(pairs) == 6 and h["S4"]["median_abs_error"] <= 0.01)
    result["extrapolation_errors"] = {key: extrapolated_gain[key] - measured[key]["gain"] for key in measured if key in extrapolated_gain}
    result["exposed_copy_time"] = exposed_copy_time(result["evaluation"])
    complete = all(result["calibration"].get(f"{m}_K{k}") is not None and result["evaluation"].get(f"{m}_K{k}") is not None
                   for m in MODELS for k in KS) and len(result["extrapolated_checks"]) == len(MODELS)
    decision_files = [RUNS / m / f"decisions_B{info['cap']}_{tag}.json" for m, info in MODELS.items()
                      for tag in (*(f"K{k}" for k in KS), "extrapolated")]
    result["decisions_frozen_before_traces"] = frozen_before_traces(RUNS / "t7.log", decision_files)
    result["fidelity_all"] = complete and result["decisions_frozen_before_traces"] and all(
        all(c["checks"].values()) for c in result["calibration"].values()) and all(
        all(e["checks"].values()) for e in result["evaluation"].values()) and all(
        all(v.values()) for v in result["extrapolated_checks"].values())
    dest = ROOT / "results/t7/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1, default=str) + "\n")
    for key in measured:
        m = measured[key]
        print(f"{key}: split-phase vs LFU {100 * m['gain']:+.2f}% [{100 * m['ci'][0]:+.2f}, {100 * m['ci'][1]:+.2f}] "
              f"({m['faster']}/{m['units']}); per host {per_host.get(key)} ({100 * per_host_gain.get(key, np.nan):+.2f}%); "
              f"extrapolated {extrapolated.get(key)} ({100 * extrapolated_gain.get(key, np.nan):+.2f}%)")
    for hid, v in h.items():
        print(f"{hid}: {json.dumps(v)}")
    print(f"fidelity all: {result['fidelity_all']}; wrote {dest}")


if __name__ == "__main__":
    main()
