"""T8-T11 frozen predictions from a calibration run, written before any fresh trace exists.

For every comparison of a calibration run (the output of a T8-T11 runner on the first 40 calibration sequences of T5's
OLMoE trace): the development costs of ``dev_costs.costs`` (per-copy t_copy and t_issue, each arm's policy
cost and residual against decayed LFU), the gain the break-even model gives with the calibration run's own copies per
token and base wall time (which is the calibration run's measured gain, since the residuals are fitted to it), and each
policy's break-even copy time against decayed LFU,

    t*_copy = [(pol_m - pol_lfu) + (res_m - res_lfu)] / (D_lfu - D_m) - t_issue.

The sealed a-priori prediction for the fresh trace is ``dev_costs.a_priori`` with these costs and the fresh trace's
copies per token and base wall time (computed in the analysis); the calibration gains decide which directional
hypotheses exist (a prediction at least two points from zero).

    python analysis/predictions_calibration.py CAL_RUN.json [CAL_RUN.json ...] --output predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dev_costs import a_priori, costs

PAIRS = (("splitphase", "lfu_decay"), ("hobbit", "lfu_decay"), ("splitphase", "hobbit"),
         ("global_token_gbt", "lfu_decay"), ("splitphase", "global_token_gbt"))


def predictions(run: dict) -> dict:
    dev = costs(run)
    arms = dev["arms"]
    per_token = {name: {"demand": a["demand"], "prefetch": a["prefetch"]} for name, a in arms.items()}
    lfu = arms["lfu_decay"]
    out = {"copy_repeat": int(run.get("copy_repeat", 1)), "costs": dev, "calibration_gain": {}, "breakeven_copy_ms": {}}
    for method, base in PAIRS:
        if method in arms and base in arms:
            out["calibration_gain"][f"{method} vs {base}"] = a_priori(dev, per_token, method, base, arms[base]["wall"])
    for name, a in arms.items():
        saved = lfu["demand"] - a["demand"]
        if name != "lfu_decay" and saved > 0:
            extra = (a["policy"] - lfu["policy"]) + (a["residual"] - lfu["residual"])
            out["breakeven_copy_ms"][name] = extra / saved - dev["t_issue"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for path in args.runs:
        entry = predictions(json.loads(path.read_text()))
        out[path.name] = entry
        c = entry["costs"]
        print(f"{path.name}: K={entry['copy_repeat']}, copy {c['t_copy']:.3f} ms + issue {c['t_issue']:.3f} ms per copy")
        for pair, gain in entry["calibration_gain"].items():
            print(f"  {pair:30s} {100 * gain:+7.2f}%")
        for name, ms in entry["breakeven_copy_ms"].items():
            print(f"  break-even {name:20s} {ms:7.3f} ms per copy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
