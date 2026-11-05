"""T2 analysis: decayed LFU and split-phase interleaved per sequence.

Data: the 80 calibration sequences of T1's fresh trace, on an A40. Per capacity: paired bootstrap of wall time
and transfers, transfers equal the simulator per sequence, the copy-time ratio of the two policies, and the
effect of decoding first or second in a pair. Writes results/t2/summary.json.

    python analysis/analyze_t2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNS = ROOT / "data/runs/t2"
SIM = ROOT / "results/t2/sim"


def rows(seqs, key):
    return [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r[key])) for r in seqs]


def main() -> None:
    from paired_bootstrap import paired_bootstrap

    out = {"capacities": {}}
    for B in (32, 24):
        path = RUNS / f"B{B}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        lfu, sp = r["policies"]["lfu_decay"], r["policies"]["splitphase"]
        key = ("interleaved", B)
        wall = paired_bootstrap({key: rows(lfu["sequences"], "wall_ms")}, {key: rows(sp["sequences"], "wall_ms")}, ["interleaved"], B)
        xfer = paired_bootstrap({key: rows(lfu["sequences"], "expert_transfers")}, {key: rows(sp["sequences"], "expert_transfers")}, ["interleaved"], B)
        sim = {}
        for arm, name in (("lfu", "lfu_decay"), ("splitphase", "splitphase")):
            f = SIM / f"sim_{arm}_B{B}.json"
            if f.exists():
                s = json.loads(f.read_text())["paths"]["calibration_mixed"]["sequences"]
                expected = [x["transfers"] for x in s]
                got = [x["expert_transfers"] for x in r["policies"][name]["sequences"]]
                sim[name] = expected == got
        # first-vs-second position effect (alternation check)
        def mean_by_slot(seqs, slot):
            vals = [x["wall_ms"] / x["tokens"] for x in seqs if x["order_slot"] == slot]
            return sum(vals) / len(vals)
        copy_ratio = sp["totals"]["ms_per_copy"] / lfu["totals"]["ms_per_copy"]
        out["capacities"][str(B)] = {
            "lfu_decay": {k: lfu["totals"][k] for k in ("ms_per_token_wall", "ms_per_token_policy", "ms_per_token_copy", "ms_per_copy", "transfers_per_token", "routing_match_fraction", "token_agreement_fraction", "tokens")},
            "splitphase": {k: sp["totals"][k] for k in ("ms_per_token_wall", "ms_per_token_policy", "ms_per_token_copy", "ms_per_copy", "transfers_per_token", "routing_match_fraction", "token_agreement_fraction", "tokens")},
            "splitphase_wait_ms_per_token": r["splitphase"]["wait_ms_per_token"],
            "worker_busy_ms_per_token": r["splitphase"]["worker_busy_ms_per_token"],
            "bandwidth_probes": r["bandwidth_probes"],
            "wall_time_reduction": wall, "transfer_reduction": xfer,
            "copy_time_ratio_splitphase_over_lfu": copy_ratio,
            "transfers_equal_simulator": sim,
            "position_effect_ms_per_token": {"lfu_first": mean_by_slot(lfu["sequences"], 0), "lfu_second": mean_by_slot(lfu["sequences"], 1),
                                             "split_first": mean_by_slot(sp["sequences"], 0), "split_second": mean_by_slot(sp["sequences"], 1)},
            "criterion_met": wall["ci_low"] > 0,
            "copy_regime_matched": abs(copy_ratio - 1) < 0.02,
        }
    dest = ROOT / "results/t2/summary.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for B, c in out["capacities"].items():
        w = c["wall_time_reduction"]
        print(f"B={B}: LFU {c['lfu_decay']['ms_per_token_wall']:.2f} vs split {c['splitphase']['ms_per_token_wall']:.2f} ms/token; "
              f"reduction {100 * w['improvement']:+.2f}% [{100 * w['ci_low']:.2f}, {100 * w['ci_high']:.2f}], "
              f"{w['sequences_improved']}/{w['paired_units']} faster; copy ratio {c['copy_time_ratio_splitphase_over_lfu']:.3f}; "
              f"sim equal {c['transfers_equal_simulator']}; met {c['criterion_met']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
