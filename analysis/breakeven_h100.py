"""Break-even copy time per model and policy on the H100 node (T5), from development costs (post-hoc summary).

The a-priori model of ``dev_costs.a_priori`` with the gain set to zero: a policy m beats decayed LFU when one expert
copy takes longer than

    t*_copy = [(pol_m - pol_lfu) + (res_m - res_lfu)] / (D_lfu - D_m) - t_issue,

with pol and res (critical-path policy time and residual, ms per token) and t_issue from the development run of the
same model and capacity (``results/t5/dev_costs.json``), and D the copies per token the simulator gives on the
sealed run's fresh trace (``results/t5/summary.json``). These are the inputs of the sealed predictions; the
values are a restatement of them, computed after the run, and decide nothing.

    python analysis/breakeven_h100.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"mixtral": ("4", "3"), "olmoe": ("32", "24"), "qwen3": ("64", "48")}
POLICIES = {"split": "splitphase", "sync": "global_token_gbt", "hobbit": "hobbit"}


def main() -> None:
    dev = json.loads((ROOT / "results/t5/dev_costs.json").read_text())
    t5 = json.loads((ROOT / "results/t5/summary.json").read_text())
    out = {}
    for model, caps in RUNS.items():
        for cap in caps:
            d, run = dev[model][cap], t5["runs"][f"{model}_B{cap}"]
            sim, lfu = run["simulated_per_token"], d["arms"]["lfu_decay"]
            row = {"copy_ms_sealed_run": run["arms"]["lfu_decay"]["t_copy"], "t_issue_ms_dev": d["t_issue"]}
            for short, name in POLICIES.items():
                arm = d["arms"][name]
                extra = (arm["policy"] - lfu["policy"]) + (arm["residual"] - lfu["residual"])
                saved = sim["lfu_decay"]["demand"] - sim[name]["demand"]
                row[short] = {"extra_ms_per_token": extra, "copies_saved_per_token": saved,
                              "breakeven_copy_ms": extra / saved - d["t_issue"]}
            out[f"{model}_B{cap}"] = row
            print(f"{model} B={cap}: copy {row['copy_ms_sealed_run']:.2f} ms; break-even "
                  + ", ".join(f"{s} {row[s]['breakeven_copy_ms']:.2f} ms" for s in POLICIES))
    dest = ROOT / "results/t5/breakeven_h100.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
