"""T1 analysis: OLMoE on an A40, unseen prompts, each policy in its own pass.

GPU: per-policy totals on the fresh mixed_interleaved path; paired, source-sequence-clustered,
domain-stratified bootstrap (analyze_caer.paired_bootstrap) of wall time and transfers; drift check;
critical-path cost; fidelity (routing/token agreement; transfers equal the unseen simulator per
sequence). Simulator: retention and paired transfer intervals on the ten fresh paths.
Writes results/t1/summary.json.

    python analysis/analyze_t1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os
RUNS = Path(os.environ.get("T1_RUNS", ROOT / "data/runs/t1"))
DRIFT = ROOT / "data/runs/t1_drift"
SIM = ROOT / "results/t1/sim"
PATH = os.environ.get("T1_PATH", "mixed_interleaved")


def rows(result, key):
    return [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r[key])) for r in result["sequences"]]


def sim_rows(sim, path):
    return [(r["position"], r["source_sequence_id"], r["domain"], float(r["transfers"])) for r in sim["paths"][path]["sequences"]]


def main() -> None:
    from paired_bootstrap import paired_bootstrap

    out = {}
    gpu = {}
    for f in sorted(RUNS.glob(f"*_{PATH}.json")):
        r = json.loads(f.read_text())
        gpu[(r["policy"], int(r["capacity"]))] = r
    drift = {}
    for f in sorted(DRIFT.glob(f"*_{PATH}.json")) if DRIFT.exists() else []:
        r = json.loads(f.read_text())
        drift[(r["policy"], int(r["capacity"]))] = r
    sims = {}
    for f in sorted(SIM.glob("sim_*.json")):
        s = json.loads(f.read_text())
        sims[(s["arm"], int(s["capacity"]))] = s
    simname = {"lfu_decay": "lfu", "global_token_gbt": "sync", "splitphase": "splitphase"}

    table = {}
    for (policy, cap), r in sorted(gpu.items()):
        t = r["totals"]
        entry = {k: t[k] for k in ("tokens", "expert_transfers", "ms_per_token_wall", "ms_per_token_policy",
                                   "ms_per_token_issue", "ms_per_token_copy", "transfers_per_token",
                                   "tokens_per_second", "routing_match_fraction", "token_agreement_fraction")}
        if "splitphase" in r:
            entry["splitphase"] = {k: r["splitphase"][k] for k in ("wait_ms_per_token", "decide_ms_per_token",
                                                                    "post_ms_per_token", "worker_busy_ms_per_token")}
        sim = sims.get((simname.get(policy, ""), cap))
        if sim is not None and PATH in sim["paths"]:
            expected = {row["source_sequence_id"]: row["transfers"] for row in sim["paths"][PATH]["sequences"]}
            got = {row["source_sequence_id"]: row["expert_transfers"] for row in r["sequences"]}
            entry["transfers_equal_simulator"] = expected == got
            entry["simulator_mismatches"] = sum(1 for k in got if expected.get(k) != got[k])
        table[f"{policy}@{cap}"] = entry
    out["gpu"] = table

    comparisons = []
    for cap in (32, 24):
        for base, cand in (("lfu_decay", "splitphase"), ("lfu_decay", "global_token_gbt"), ("global_token_gbt", "splitphase")):
            if (base, cap) not in gpu or (cand, cap) not in gpu:
                continue
            left = {(PATH, cap): rows(gpu[(base, cap)], "wall_ms")}
            right = {(PATH, cap): rows(gpu[(cand, cap)], "wall_ms")}
            wall = paired_bootstrap(left, right, [PATH], cap)
            lt = {(PATH, cap): rows(gpu[(base, cap)], "expert_transfers")}
            rt = {(PATH, cap): rows(gpu[(cand, cap)], "expert_transfers")}
            tr = paired_bootstrap(lt, rt, [PATH], cap)
            comparisons.append({"capacity": cap, "baseline": base, "candidate": cand,
                                "wall_time_reduction": wall, "transfer_reduction": tr})
    out["gpu_comparisons"] = comparisons

    if ("lfu_decay", 32) in drift and ("lfu_decay", 32) in gpu:
        a = gpu[("lfu_decay", 32)]["totals"]["ms_per_token_wall"]
        b = drift[("lfu_decay", 32)]["totals"]["ms_per_token_wall"]
        out["drift_check"] = {"first_ms_per_token": a, "second_ms_per_token": b, "relative": (b - a) / a,
                              "within_1_percent": abs(b - a) / a < 0.01}

    sim_out = {}
    for cap in (32, 24):
        if not all((arm, cap) in sims for arm in ("lfu", "sync", "splitphase")):
            continue
        lfu, sync, sp = (sims[(a, cap)] for a in ("lfu", "sync", "splitphase"))
        paths = sorted(lfu["paths"])
        retention = (lfu["transfers"] - sp["transfers"]) / (lfu["transfers"] - sync["transfers"])
        left = {(p, cap): sim_rows(lfu, p) for p in paths}
        right = {(p, cap): sim_rows(sp, p) for p in paths}
        sim_out[str(cap)] = {"lfu": lfu["transfers"], "sync": sync["transfers"], "splitphase": sp["transfers"],
                             "retention": retention,
                             "splitphase_vs_lfu": paired_bootstrap(left, right, paths, cap),
                             "sync_vs_lfu": paired_bootstrap(left, {(p, cap): sim_rows(sync, p) for p in paths}, paths, cap)}
    out["simulator"] = sim_out

    verdict = {}
    for c in comparisons:
        if c["baseline"] == "lfu_decay" and c["candidate"] == "splitphase":
            key = "primary_B32" if c["capacity"] == 32 else "secondary_B24"
            w = c["wall_time_reduction"]
            verdict[key] = {"improvement": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "met": w["ci_low"] > 0}
        if c["baseline"] == "lfu_decay" and c["candidate"] == "global_token_gbt":
            w = c["wall_time_reduction"]
            verdict[f"sync_slower_B{c['capacity']}"] = {"improvement": w["improvement"], "ci": [w["ci_low"], w["ci_high"]],
                                                        "met": w["ci_high"] < 0}
    for cap, s in sim_out.items():
        verdict[f"sim_retention_B{cap}"] = {"retention": s["retention"],
                                            "met": s["retention"] >= (0.85 if cap == "32" else 0.80)
                                            and s["splitphase_vs_lfu"]["ci_low"] > 0}
    for cap in (32, 24):
        sp, lfu = table.get(f"splitphase@{cap}"), table.get(f"lfu_decay@{cap}")
        if sp and lfu:
            verdict[f"critical_path_B{cap}"] = {
                "splitphase_policy_ms_per_token": sp["ms_per_token_policy"], "lfu_policy_ms_per_token": lfu["ms_per_token_policy"],
                "ratio": sp["ms_per_token_policy"] / lfu["ms_per_token_policy"],
                "wait_ms_per_token": sp.get("splitphase", {}).get("wait_ms_per_token"),
                "met": sp["ms_per_token_policy"] <= 2 * lfu["ms_per_token_policy"]
                and (sp.get("splitphase", {}).get("wait_ms_per_token") or 0) <= 0.5}
    out["verdict"] = verdict
    dest = Path(os.environ.get("T1_DEST", ROOT / "results/t1/summary.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k in ("verdict", "drift_check")}, indent=1))
    for k, e in table.items():
        print(f"{k:22s} wall {e['ms_per_token_wall']:.2f} policy {e['ms_per_token_policy']:.2f} copy {e['ms_per_token_copy']:.2f} "
              f"xfer/tok {e['transfers_per_token']:.2f} route {100 * e['routing_match_fraction']:.2f}% sim_equal {e.get('transfers_equal_simulator')}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
