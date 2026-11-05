"""T3 analysis: Qwen3-30B-A3B on an A40, Mixtral-8x7B on an A100 host with a PCIe gen3 link.

For each model and capacity: per-policy totals of the interleaved GPU run; paired, sequence-clustered,
domain-stratified bootstrap (analyze_caer.paired_bootstrap, 1,000 replicates) of per-sequence decode
wall time for split-phase vs decayed LFU, split-phase vs the synchronous ranker, and the synchronous
ranker vs decayed LFU; transfers equal the unseen simulator per sequence; the critical path; the
simulator's retention on the fresh trace. Every criterion is evaluated exactly as sealed.

    python analysis/analyze_t3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SIM = ROOT / "results/t3/sim"
MODELS = {
    "qwen3_30b_a3b": {"runs": ROOT / "data/runs/t3_qwen3_30b_a3b", "caps": (64, 48)},
    "mixtral_8x7b": {"runs": ROOT / "data/runs/t3_mixtral_8x7b", "caps": (4, 3)},
}
SIM_ARM = {"lfu_decay": "lfu", "global_token_gbt": "sync", "splitphase": "splitphase"}


def rows(seqs, key):
    return [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r[key])) for r in seqs]


def compare(base, cand, cap, key="wall_ms"):
    from paired_bootstrap import paired_bootstrap

    k = ("interleaved", cap)
    return paired_bootstrap({k: rows(base["sequences"], key)}, {k: rows(cand["sequences"], key)}, ["interleaved"], cap)


def main() -> None:
    out = {"models": {}}
    for model, info in MODELS.items():
        entry = {}
        for cap in info["caps"]:
            path = info["runs"] / f"B{cap}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            pol = run["policies"]
            lfu, sync, sp = pol["lfu_decay"], pol["global_token_gbt"], pol["splitphase"]
            res = {
                "totals": {name: {k: p["totals"][k] for k in ("ms_per_token_wall", "ms_per_token_policy", "ms_per_token_copy",
                                                              "ms_per_copy", "transfers_per_token", "routing_match_fraction",
                                                              "token_agreement_fraction", "tokens")} for name, p in pol.items()},
                "splitphase": run.get("splitphase"),
                "bandwidth_probe": run.get("bandwidth_probe"),
                "split_vs_lfu": compare(lfu, sp, cap),
                "split_vs_sync": compare(sync, sp, cap),
                "sync_vs_lfu": compare(lfu, sync, cap),
                "transfers_split_vs_lfu": compare(lfu, sp, cap, "expert_transfers"),
            }
            sim_equal = {}
            for name, arm in SIM_ARM.items():
                f = SIM / f"sim_{model}_{arm}_B{cap}.json"
                if f.exists():
                    s = json.loads(f.read_text())
                    seqs = next(iter(s["paths"].values()))["sequences"]
                    sim_equal[name] = [x["transfers"] for x in seqs] == [x["expert_transfers"] for x in pol[name]["sequences"]]
            res["transfers_equal_simulator"] = sim_equal
            routing = min(p["totals"]["routing_match_fraction"] for p in pol.values())
            res["fidelity_met"] = routing >= 0.999 and len(sim_equal) == 3 and all(sim_equal.values())
            wait = (run.get("splitphase") or {}).get("wait_ms_per_token")
            limit = 0.005 * lfu["totals"]["ms_per_token_wall"]
            res["critical_path"] = {"split_policy_ms_per_token": sp["totals"]["ms_per_token_policy"],
                                    "lfu_policy_ms_per_token": lfu["totals"]["ms_per_token_policy"],
                                    "sync_policy_ms_per_token": sync["totals"]["ms_per_token_policy"],
                                    "wait_ms_per_token": wait, "wait_limit_ms_per_token": limit,
                                    "met": wait is not None and wait <= limit}  # F3
            res["criteria"] = {
                "sync_slower_than_lfu": res["sync_vs_lfu"]["ci_high"] < 0,        # Q1 / M1
                "split_faster_than_sync": res["split_vs_sync"]["ci_low"] > 0,     # Q2 / M2
                "split_faster_than_lfu": res["split_vs_lfu"]["ci_low"] > 0,       # Q3 / M3 (secondary)
                "fidelity": res["fidelity_met"],                                  # F1 + F2
                "critical_path": res["critical_path"]["met"],                     # F3
            }
            workers = info["runs"] / "workers.json"
            res["workers"] = json.loads(workers.read_text())["workers"] if workers.exists() else None
            entry[str(cap)] = res
        sims = {}
        for cap in info["caps"]:
            arms = {a: SIM / f"sim_{model}_{a}_B{cap}.json" for a in ("lfu", "sync", "splitphase")}
            if all(p.exists() for p in arms.values()):
                t = {a: json.loads(p.read_text())["transfers"] for a, p in arms.items()}
                retention = (t["lfu"] - t["splitphase"]) / max(t["lfu"] - t["sync"], 1)
                sims[str(cap)] = {**t, "retention": retention,
                                  "met": retention >= (0.85 if model == "qwen3_30b_a3b" else 0.50)}  # S1
        entry["simulator"] = sims
        out["models"][model] = entry
    dest = ROOT / "results/t3/summary.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for model, entry in out["models"].items():
        for cap, res in entry.items():
            if cap == "simulator":
                continue
            w = {k: res[k] for k in ("split_vs_lfu", "split_vs_sync", "sync_vs_lfu")}
            print(f"{model} B={cap}: " + "; ".join(f"{k} {100 * v['improvement']:+.1f}% [{100 * v['ci_low']:.1f}, {100 * v['ci_high']:.1f}]"
                                                   for k, v in w.items())
                  + f"; criteria {res['criteria']}; workers {res['workers']}")
        print(f"{model} simulator: {entry['simulator']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
