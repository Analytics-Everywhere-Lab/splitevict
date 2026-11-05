"""T6 predictions, fixed before the run: Mixtral (B=4) on an H100 node at higher copy costs, from development costs.

The break-even model a priori (``dev_costs.a_priori``), as in T5: every cost comes from T5's development cost file
(``results/t5/dev_costs.json``, entry ``mixtral/4``: per-copy costs, each arm's timed policy cost and residual
against decayed LFU). The copy-cost rule for K > 1: t_copy and t_issue are K times
their K = 1 values (a 352 MB expert copy is bandwidth-bound and every repeat is issued); the policy costs and
residuals stay at their K = 1 values. Here, copies per token come from T5's Mixtral B=4 run (its simulator counts),
and each arm's wall time at K is that run's wall time at K = 1 plus its extra copy cost. T6's analysis
(``analyze_t6.py``) instead uses the T6 run's own simulator counts and base-arm wall times.

    python analysis/predictions_t6.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 2, 4, 7)
ARMS = ("lfu_decay", "hobbit", "splitphase", "global_token_gbt")
PAIRS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"),
         ("global_token_gbt", "lfu_decay"), ("splitphase", "global_token_gbt"))


def costs_at(dev: dict, k: int) -> dict:
    """The sealed sweep rule: copy and issue cost linear in K, policy and residual as at K = 1."""
    entry = dict(dev)
    entry["t_copy"] = dev["t_copy"] * k
    entry["t_issue"] = dev["t_issue"] * k
    return entry


def gain(dev: dict, copies: dict, walls: dict, m: str, b: str, k: int) -> float:
    c = costs_at(dev, k)
    a, base = c["arms"][m], c["arms"][b]
    extra_per_copy = (c["t_copy"] + c["t_issue"]) - (dev["t_copy"] + dev["t_issue"])
    wall_b = walls[b] + copies[b] * extra_per_copy
    delta = (copies[m] - copies[b]) * (c["t_copy"] + c["t_issue"]) + (a["policy"] - base["policy"]) \
        + (a["residual"] - base["residual"])
    return -delta / wall_b


def main() -> None:
    dev = json.loads((ROOT / "results/t5/dev_costs.json").read_text())["mixtral"]["4"]
    run = json.loads((ROOT / "results/t5/summary.json").read_text())["runs"]["mixtral_B4"]
    copies = {a: run["simulated_per_token"][a]["demand"] for a in ARMS}
    walls = {a: run["arms"][a]["wall"] for a in ARMS}
    out = {"source": "T5 development costs (mixtral/4), T5 Mixtral B=4 copies and wall times", "copy_ms": {}, "gains": {}}
    for k in KS:
        out["copy_ms"][str(k)] = k * dev["t_copy"]
        out["gains"][str(k)] = {f"{m} vs {b}": gain(dev, copies, walls, m, b, k) for m, b in PAIRS}
        be = {m: ((dev["arms"][m]["policy"] - dev["arms"]["lfu_decay"]["policy"])
                  + (dev["arms"][m]["residual"] - dev["arms"]["lfu_decay"]["residual"]))
              / (copies["lfu_decay"] - copies[m]) - dev["t_issue"] for m in ("splitphase", "global_token_gbt", "hobbit")}
        out["breakeven_copy_ms_K1_issue"] = be
        print(f"K={k}: copy {k * dev['t_copy']:.1f} ms  " + "  ".join(
            f"{p} {100 * g:+.2f}%" for p, g in out["gains"][str(k)].items()))
    print("break-even copy time at K=1 issue cost:", {m: round(v, 2) for m, v in be.items()})
    dest = ROOT / "results/t6/predictions.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
