"""T7 expected decisions: what the on-node selector should decide, from T5's runs on the same prompts.

T7 calibrates on the node itself: decayed LFU and split-phase, one pass, on the first 40 calibration sequences of
T5's fresh trace of the model (ten per domain; outside the frozen rankers' fit and disjoint from T7's prompts), at
each copy repeat K = 1, 2, 4. Two selectors read these runs, both ``selector.decide``:

* per host (primary): the calibration run at the setting's own K, decided at that K (``--ks 1``). Its prediction is
  the calibration's measured difference.
* extrapolated (secondary): the K = 1 calibration run only, extrapolated to K = 2 and 4 (``--ks 1 2 4``).

T5 decoded the same prompts on the same node type: its main runs at K = 1 (cut to the 40 sequences and their first
pass), and its copy-cost sweep at K = 2 and 4 for OLMoE (on the same 40 sequences) and at K = 4 for Qwen3 (on the first
20 of them). Those runs give the expected decisions; Qwen3 at K = 2 has no T5 run, so no per-host expectation.

    python analysis/predictions_t7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dev_costs import costs  # noqa: E402
from selector import BASE, MARGIN, METHOD, decide  # noqa: E402

SETTINGS = (("olmoe", 32), ("qwen3", 64))
KS = (1, 2, 4)
CALIBRATION_SEQUENCES = 40
T5_RUNS = ROOT / "data/runs/t5"


def calibration_ids(model: str) -> list[int]:
    spec = json.loads((ROOT / f"data/specs/{model}_t5_replay_spec.json").read_text())
    return [int(v) for v in spec["calibration_sequence_ids"][:CALIBRATION_SEQUENCES]]


def first_pass_subset(run: dict, ids: set[int]) -> dict:
    """The run cut to those of ``ids`` it decoded and to its first pass, with the totals ``dev_costs.costs`` reads
    rebuilt from the rows."""

    out = {"passes": 1, "arms": {}}
    for name in (BASE, METHOD):
        rows = [r for r in run["arms"][name]["sequences"] if r.get("pass", 0) == 0 and r["source_sequence_id"] in ids]
        tokens = sum(r["tokens"] for r in rows)
        out["arms"][name] = {"sequences": rows, "totals": {
            "tokens": tokens, "expert_transfers": sum(r["expert_transfers"] for r in rows),
            "copy_ms": sum(r["copy_ms"] for r in rows), "issue_ms": sum(r["issue_ms"] for r in rows),
            "ms_per_token_policy": sum(r["policy_ms"] for r in rows) / tokens}}
    return out


def summary(entry: dict) -> dict:
    return {"t_copy": entry["t_copy"], "t_issue": entry["t_issue"], "arms": {n: entry["arms"][n] for n in (BASE, METHOD)}}


def main() -> None:
    out = {"source": "T5 runs on T7's calibration prompts (first 40 calibration ids of T5's traces)",
           "margin": MARGIN, "settings": {}}
    for model, cap in SETTINGS:
        ids = set(calibration_ids(model))
        main_run = json.loads((T5_RUNS / f"{model}/B{cap}.json").read_text())
        k1 = costs(first_pass_subset(main_run, ids))
        per_host = {}
        for k in KS:
            path = T5_RUNS / f"{model}/B{cap}.json" if k == 1 else T5_RUNS / f"{model}/B{cap}_K{k}.json"
            if not path.exists():
                per_host[str(k)] = None
                continue
            run = main_run if k == 1 else json.loads(path.read_text())
            sub = first_pass_subset(run, ids)
            entry = costs(sub)
            decision = decide(entry, [1])["1"]
            decision["copy_ms"] = entry["t_copy"]
            decision["source"] = f"{path.relative_to(ROOT)}, {len(sub['arms'][BASE]['sequences'])} of the 40 sequences"
            per_host[str(k)] = decision
        extrapolated = decide(k1, KS)
        out["settings"][f"{model}_B{cap}"] = {"calibration_ids": sorted(ids), "k1_calibration": summary(k1),
                                              "per_host": per_host, "extrapolated": extrapolated}
        for k in map(str, KS):
            p, x = per_host[k], extrapolated[k]
            host = "no T5 run" if p is None else f"{100 * p['predicted_gain']:+.2f}% -> {p['policy']} ({p['source']})"
            print(f"{model} B={cap} K={k}: per host {host}; extrapolated {100 * x['predicted_gain']:+.2f}% -> {x['policy']}")
    dest = ROOT / "results/t7/expected.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
