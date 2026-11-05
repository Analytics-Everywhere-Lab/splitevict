"""Exploratory table: published-system eviction rules against decayed LFU and split-phase on the T1, T3, T4 traces.

Decayed LFU and the rules come from ``baselines_unseen.py``: LRU (Mixtral-offloading's per-layer rule), HOBBIT's and
MoE-Infinity's, fitted on calibration data only. Split-phase transfers are the ones on record for the same paths
(simulator; equal to hardware per sequence): T1's ``results/t1/sim/sim_*`` files (ten paths), the simulator entries
of ``results/t3/summary.json``, and T4's run files (pass 0). Prints LaTeX rows for ``tab:rules`` and writes
``results/t5/rules_table.json``.

    python analysis/report_rules.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/t5"


def split_transfers(trace: str, cap: int) -> int | None:
    if trace == "t1":
        return json.loads((ROOT / f"results/t1/sim/sim_splitphase_B{cap}.json").read_text())["transfers"]
    if trace == "t4":
        path = ROOT / f"data/runs/t4/B{cap}.json"
        if not path.exists():
            return None
        seqs = json.loads(path.read_text())["arms"]["splitphase"]["sequences"]
        return sum(r["expert_transfers"] for r in seqs if r.get("pass", 0) == 0)
    model = {"t3_qwen3": "qwen3_30b_a3b", "t3_mixtral": "mixtral_8x7b"}[trace]
    return json.loads((ROOT / "results/t3/summary.json").read_text())["models"][model]["simulator"][str(cap)]["splitphase"]


def main() -> None:
    labels = {"t1": "OLMoE (T1, 10 paths)", "t4": "OLMoE (T4)", "t3_qwen3": "Qwen3 (T3)", "t3_mixtral": "Mixtral (T3)"}
    out, rows = {}, []
    for trace in ("t1", "t4", "t3_qwen3", "t3_mixtral"):
        data = json.loads((RES / f"baselines_unseen_{trace}.json").read_text())
        for cap, entry in data["capacities"].items():
            t = entry["totals"]
            lfu, hob, moe, lru = t["lfu_decay"], t["hobbit"], t["moe_infinity"], t["lru"]
            sp = split_transfers(trace, int(cap))
            rec = {"lfu_decay": lfu, "lru": lru, "hobbit": hob, "moe_infinity": moe, "splitphase": sp,
                   "hobbit_w": entry["hobbit_w"], "lru_vs_lfu": (lru - lfu) / lfu,
                   "hobbit_vs_lfu": (hob - lfu) / lfu, "moe_infinity_vs_lfu": (moe - lfu) / lfu,
                   "split_vs_lfu": None if sp is None else (sp - lfu) / lfu,
                   "split_vs_hobbit": None if sp is None else (sp - hob) / hob}
            out[f"{trace}_B{cap}"] = rec
            f = lambda v: "--" if v is None else f"${100 * v:+.1f}$"
            rows.append(f"    {labels[trace]} & {cap} & {f(rec['lru_vs_lfu'])} & {f(rec['hobbit_vs_lfu'])} & {f(rec['moe_infinity_vs_lfu'])} & "
                        f"{f(rec['split_vs_lfu'])} & {f(rec['split_vs_hobbit'])} \\\\")
    print("\n".join(rows))
    (RES / "rules_table.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {RES / 'rules_table.json'}")


if __name__ == "__main__":
    main()
