"""Helpers for Figure 6: wall-time gain over decayed LFU against the cost of one expert copy, on one H100 (T5).

Points: measured gains with 95% intervals (main runs, K = 1, and the copy-cost sweep, K > 1) for SplitEvict, the
synchronous ranker and HOBBIT's rule. Lines: the break-even model a priori, as sealed: every cost from the development
runs (per-copy cost, each arm's timed policy cost and residual against decayed LFU, K = 1 entry) and, from the sealed
run, only its simulated copies per token and decayed LFU's wall time, extended in the copy cost c:
gain(c) = -[(D_m - D_b) c + (pol_m - pol_b) + (res_m - res_b)] / (W_b + D_b (c - c_1)).
``figures.py`` draws the figure from these helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, GREEN, VERMILLION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV_COSTS = ROOT / "results/t5/dev_costs.json"
ARMS = (("splitphase", "SplitEvict", GREEN, "o"), ("global_token_gbt", "synchronous ranker", VERMILLION, "s"),
        ("hobbit", "HOBBIT's rule", BLUE, "^"))
YLIM = (-32.0, 22.0)


def points(model: str, cap: int, ks: tuple[int, ...]):
    """({arm: [(cost of one copy incl. issue, gain, ci low, ci high), ...]}, per-token copies, LFU wall at K = 1)."""

    out = {arm: [] for arm, *_ in ARMS}
    copies = None
    summary = json.loads((ROOT / "results/t5/summary.json").read_text())
    for k in ks:
        key = f"{model}_B{cap}" + ("" if k == 1 else f"_K{k}")
        run = summary["runs"].get(key)
        if run is None:
            continue
        lfu = run["arms"]["lfu_decay"]
        if k == 1:
            copies = {a: v["demand"] for a, v in run["simulated_per_token"].items()}
        for arm, *_ in ARMS:
            e = run["comparisons"].get(f"{arm} vs lfu_decay")
            if e:
                out[arm].append((lfu["t_copy"] + lfu["t_issue"], e["estimate"]["gain"], e["estimate"]["ci"][0],
                                 e["estimate"]["ci"][1]))
    # decayed LFU's wall time per token, the faster pass of each sequence (report_t5.py)
    wall = json.loads((ROOT / "results/t5/report.json").read_text())["runs"][f"{model}_B{cap}"]["lfu_wall"]
    return out, copies, wall


def curve(costs: dict, copies: dict, wall: float, arm: str, grid: np.ndarray) -> np.ndarray:
    a, b = costs["arms"][arm], costs["arms"]["lfu_decay"]
    c1 = costs["t_copy"] + costs["t_issue"]
    saved = (copies["lfu_decay"] - copies[arm]) * grid
    extra = (a["policy"] - b["policy"]) + (a["residual"] - b["residual"])
    return (saved - extra) / (wall + copies["lfu_decay"] * (grid - c1))
