"""Break-even copy time per model from the T1 (OLMoE) and T3 runs (post-hoc; not part of any sealed criterion).

The break-even model applied to each run's own measured parameters:
W(policy) = R + critical_path + n_transfers * (t_copy + t_issue). R cancels in a difference, so a
policy beats decayed LFU when the copies it saves cost more than its extra critical path:
break-even t_copy = (pol - pol_lfu) / (n_lfu - n) - t_issue. t_copy and t_issue are decayed LFU's
measured values on the same run; the prediction is compared with the measured wall-time difference.
OLMoE uses the T1 run on an A40 (B=32, separate passes); the others the interleaved T3 runs.

    python analysis/breakeven_crossmodel.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    ("qwen3_30b_a3b", 64): ROOT / "data/runs/t3_qwen3_30b_a3b/B64.json",
    ("qwen3_30b_a3b", 48): ROOT / "data/runs/t3_qwen3_30b_a3b/B48.json",
    ("mixtral_8x7b", 4): ROOT / "data/runs/t3_mixtral_8x7b/B4.json",
}
NAMES = {"sync": "global_token_gbt", "split": "splitphase"}


def load(model: str, cap: int, path: Path) -> dict:
    """Per-policy totals (ms per token, transfers per token) of one run."""
    if path is None:  # OLMoE: the T1 run on an A40
        gpu = json.loads((ROOT / "results/t1/summary.json").read_text())["gpu"]
        return {name: gpu[f"{name}@{cap}"] for name in ("lfu_decay", *NAMES.values())}
    return {name: p["totals"] for name, p in json.loads(path.read_text())["policies"].items()}


def main() -> None:
    out = {}
    for (model, cap), path in {("olmoe", 32): None, **RUNS}.items():
        pol = load(model, cap, path)
        lfu = pol["lfu_decay"]
        t_copy = lfu["ms_per_token_copy"] / lfu["transfers_per_token"]
        t_issue = lfu["ms_per_token_issue"] / lfu["transfers_per_token"]
        row = {"t_copy_ms": t_copy, "t_issue_ms": t_issue, "lfu_wall_ms": lfu["ms_per_token_wall"]}
        for short, name in NAMES.items():
            t = pol[name]
            d_pol = t["ms_per_token_policy"] - lfu["ms_per_token_policy"]
            d_n = lfu["transfers_per_token"] - t["transfers_per_token"]
            predicted = (d_n * (t_copy + t_issue) - d_pol) / lfu["ms_per_token_wall"]
            measured = (lfu["ms_per_token_wall"] - t["ms_per_token_wall"]) / lfu["ms_per_token_wall"]
            row[short] = {"extra_critical_path_ms": d_pol, "transfers_saved_per_token": d_n,
                          "breakeven_copy_ms": d_pol / d_n - t_issue,
                          "predicted_gain": predicted, "measured_gain": measured,
                          # ms per token the model does not explain (positive: slower than predicted)
                          "unexplained_ms_per_token": (predicted - measured) * lfu["ms_per_token_wall"]}
        out[f"{model}_B{cap}"] = row
        print(f"{model} B={cap}: copy {t_copy:.3f} ms (+{t_issue:.3f} issue); "
              + "; ".join(f"{s} break-even {row[s]['breakeven_copy_ms']:.2f} ms, predicted {100 * row[s]['predicted_gain']:+.1f}% "
                          f"measured {100 * row[s]['measured_gain']:+.1f}%" for s in NAMES))
    dest = ROOT / "results/t3/sim/breakeven.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
