"""The break-even model (Equations 1-2): cost parameters from development runs, and a-priori predictions (T5).

From one development run (an ``interleaved_prefetch.py`` / ``interleaved_h100.py`` output), per arm: the timed
critical-path policy cost (ms/token, decisions plus prefetch decisions), the measured wall time (ms/token; the
faster pass per sequence when the run has two), and the arm's *residual*: the part of its wall-time difference
from decayed LFU that the timed costs do not explain,

    residual(arm) = (W_arm - W_lfu) - [(D_arm - D_lfu)(t_copy + t_issue) + (P_arm - P_lfu) t_pf + (pol_arm - pol_lfu)],

with D, P the demand and prefetch copies per token and t_copy, t_issue, t_pf the run's per-copy costs. The
a-priori prediction for a comparison on new data uses these development costs and only the new data's
simulated copies per token (and the new base arm's wall time as the denominator):

    gain(m vs b) = -[(D_m - D_b)(t_copy + t_issue) + (P_m - P_b) t_pf + (pol_m - pol_b) + (res_m - res_b)] / W_b.

    python analysis/dev_costs.py RUN.json [RUN.json ...] --output dev_costs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def per_token_wall(arm: dict, passes: int) -> float:
    rows = arm["sequences"]
    if passes > 1:
        best: dict[int, dict] = {}
        for r in rows:
            sid = r["source_sequence_id"]
            if sid not in best or r["wall_ms"] < best[sid]["wall_ms"]:
                best[sid] = r
        rows = list(best.values())
    return sum(r["wall_ms"] for r in rows) / sum(r["tokens"] for r in rows)


def costs(run: dict, base: str = "lfu_decay") -> dict:
    passes = int(run.get("passes", 1))
    arms = {}
    copies = issue_ms = copy_ms = pf_copies = pf_issue_ms = 0.0
    for name, a in run["arms"].items():
        t = a["totals"]  # totals of pass 0
        tok = max(t["tokens"], 1)
        arms[name] = {"wall": per_token_wall(a, passes), "demand": t["expert_transfers"] / tok,
                      "prefetch": t.get("prefetch_transfers", 0) / tok,
                      "policy": t["ms_per_token_policy"] + t.get("ms_per_token_prefetch_policy", 0.0)}
        copies += t["expert_transfers"]
        copy_ms += t["copy_ms"]
        issue_ms += t["issue_ms"]
        pf_copies += t.get("prefetch_transfers", 0)
        pf_issue_ms += t.get("prefetch_issue_ms", 0.0)
    t_copy, t_issue = copy_ms / max(copies, 1), issue_ms / max(copies, 1)
    t_pf = pf_issue_ms / pf_copies if pf_copies else 0.0
    b = arms[base]
    for name, a in arms.items():
        explained = ((a["demand"] - b["demand"]) * (t_copy + t_issue) + (a["prefetch"] - b["prefetch"]) * t_pf
                     + (a["policy"] - b["policy"]))
        a["residual"] = (a["wall"] - b["wall"]) - explained
    return {"t_copy": t_copy, "t_issue": t_issue, "t_pf_issue": t_pf, "passes": passes, "arms": arms,
            "label": run.get("label"), "capacity": run.get("capacity")}


def a_priori(dev: dict, new_per_token: dict, method: str, base: str, base_wall: float) -> float | None:
    """Predicted gain of ``method`` over ``base``: development costs, new copies per token, new base wall."""

    if method not in dev["arms"] or base not in dev["arms"]:
        return None
    m, b = dev["arms"][method], dev["arms"][base]
    nm, nb = new_per_token[method], new_per_token[base]
    delta = ((nm["demand"] - nb["demand"]) * (dev["t_copy"] + dev["t_issue"])
             + (nm["prefetch"] - nb["prefetch"]) * dev["t_pf_issue"]
             + (m["policy"] - b["policy"]) + (m["residual"] - b["residual"]))
    return -delta / base_wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for path in args.runs:
        run = json.loads(path.read_text())
        entry = costs(run)
        out[str(path)] = entry
        print(f"{path}: t_copy {entry['t_copy']:.3f} ms, t_issue {entry['t_issue'] * 1000:.1f} us, "
              f"t_pf_issue {entry['t_pf_issue'] * 1000:.1f} us")
        for name, a in entry["arms"].items():
            print(f"  {name:24s} wall {a['wall']:7.2f}  policy {a['policy']:6.2f}  demand {a['demand']:6.2f}  "
                  f"prefetch {a['prefetch']:6.2f}  residual {a['residual']:+6.2f} ms/token")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
