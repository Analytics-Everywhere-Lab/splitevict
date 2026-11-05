"""T11 frozen predictions per setting and the directional hypotheses they decide, written before any fresh trace exists.

For each setting (``native``: experts copied from pinned host memory; ``nvme``: every demand copy read from the drive),
``predictions_calibration.predictions`` on that setting's calibration run: the development costs of
``dev_costs.costs`` (with two passes, each sequence's faster pass for the wall times), the calibration gain of every
comparison and each policy's break-even copy time against decayed LFU. The hypotheses it fixes:

* nvme: P1 split-phase vs decayed LFU, S1 split-phase vs HOBBIT's rule, S3 split-phase vs the synchronous ranker, all
  "above zero"; S2 the synchronous ranker vs decayed LFU in the direction of its calibration gain, only if that gain
  lies at least two points from zero (otherwise no directional hypothesis).
* native: N1-N4, the same four comparisons, each in the direction of its calibration gain, only if that gain lies at
  least one point from zero (otherwise no directional hypothesis).

    python analysis/predictions_t11.py --native CAL_NATIVE.json --nvme CAL_NVME.json --output predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictions_calibration import predictions

# (id, method, base, rule): "above", or the distance from zero the calibration gain needs for a directional hypothesis
HYPOTHESES = {
    "nvme": (("P1", "splitphase", "lfu_decay", "above"), ("S1", "splitphase", "hobbit", "above"),
             ("S2", "global_token_gbt", "lfu_decay", 0.02), ("S3", "splitphase", "global_token_gbt", "above")),
    "native": (("N1", "splitphase", "lfu_decay", 0.01), ("N2", "splitphase", "hobbit", 0.01),
               ("N3", "global_token_gbt", "lfu_decay", 0.01), ("N4", "splitphase", "global_token_gbt", 0.01)),
}


def hypotheses(setting: str, gains: dict) -> dict:
    out = {}
    for hid, method, base, rule in HYPOTHESES[setting]:
        pair = f"{method} vs {base}"
        gain = gains[pair]
        if rule == "above":
            criterion = "above"
        else:
            criterion = None if abs(gain) < rule else ("above" if gain > 0 else "below")
        out[hid] = {"comparison": pair, "criterion": criterion, "calibration_gain": gain}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--nvme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for setting, path in (("native", args.native), ("nvme", args.nvme)):
        entry = predictions(json.loads(path.read_text()))
        entry["calibration_run"] = path.name
        entry["hypotheses"] = hypotheses(setting, entry["calibration_gain"])
        out[setting] = entry
        c = entry["costs"]
        print(f"{setting} ({path.name}): copy {c['t_copy']:.3f} ms + issue {c['t_issue']:.3f} ms per copy")
        for pair, gain in entry["calibration_gain"].items():
            print(f"  {pair:30s} {100 * gain:+7.2f}%")
        for name, ms in entry["breakeven_copy_ms"].items():
            print(f"  break-even {name:20s} {ms:7.3f} ms per copy")
        for hid, h in entry["hypotheses"].items():
            print(f"  {hid}: {h['comparison']}: " + (f"{h['criterion']} zero" if h["criterion"] else
                                                     f"no directional hypothesis ({100 * h['calibration_gain']:+.2f}%)"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
