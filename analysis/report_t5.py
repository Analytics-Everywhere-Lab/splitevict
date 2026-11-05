"""T5's table rows and the per-setting gains that ``selector.py t5`` reads (post-hoc presentation).

Reads ``results/t5/summary.json`` (written by ``analyze_t5.py``) and the run files; prints the LaTeX rows of the
H100 table and writes ``results/t5/report.json``. Nothing here decides a criterion.

    python analysis/report_t5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev_costs import per_token_wall  # noqa: E402

LABELS = {"mixtral": "Mixtral", "olmoe": "OLMoE", "qwen3": "Qwen3"}


def pct(x: float, digits: int = 1) -> str:
    s = f"{100 * x:+.{digits}f}"
    if float(s) == 0:
        s = f"{0:.{digits}f}"
    return f"${s}$"


def cell(e: dict) -> str:
    v = e["estimate"]
    lo, hi = (f"{100 * c:.1f}" for c in v["ci"])
    lo = lo if not lo.startswith("-") else f"$-${lo[1:]}"
    hi = hi if not hi.startswith("-") else f"$-${hi[1:]}"
    return f"{pct(v['gain'])} [{lo}, {hi}]"


def main() -> None:
    summary = json.loads((ROOT / "results/t5/summary.json").read_text())
    out, rows = {"runs": {}, "a_priori": summary["a_priori"]}, []
    for model, caps in (("mixtral", (4, 3)), ("olmoe", (32, 24)), ("qwen3", (64, 48))):
        for cap in caps:
            r = summary["runs"][f"{model}_B{cap}"]
            data = json.loads((ROOT / f"data/runs/t5/{model}/B{cap}.json").read_text())
            lfu_wall = per_token_wall(data["arms"]["lfu_decay"], int(data["passes"]))
            c = r["comparisons"]
            copy_ms = r["arms"]["lfu_decay"]["t_copy"]
            saved = r["simulated_per_token"]["lfu_decay"]["demand"] - r["simulated_per_token"]["splitphase"]["demand"]
            out["runs"][f"{model}_B{cap}"] = {
                "copy_ms": copy_ms, "lfu_wall": lfu_wall, "saved_copies_per_token": saved,
                "lfu_copies_per_token": r["simulated_per_token"]["lfu_decay"]["demand"],
                **{k: {"gain": e["estimate"]["gain"], "ci": e["estimate"]["ci"], "a_priori": e["a_priori"]} for k, e in c.items()}}
            ap = c["splitphase vs lfu_decay"]["a_priori"]
            rows.append(f"    {LABELS[model]}, {cap} & {copy_ms:.2f} & {lfu_wall:.1f} & {cell(c['splitphase vs lfu_decay'])} "
                        f"({pct(ap)}) & {cell(c['splitphase vs hobbit'])} & {cell(c['global_token_gbt vs lfu_decay'])} \\\\")
    for model, cap, ks in (("olmoe", 32, (2, 4, 16, 32)), ("qwen3", 64, (4, 16))):
        for k in ks:
            r = summary["runs"][f"{model}_B{cap}_K{k}"]
            out["runs"][f"{model}_B{cap}_K{k}"] = {
                "copy_ms": r["arms"]["lfu_decay"]["t_copy"],
                **{p: {"gain": e["estimate"]["gain"], "ci": e["estimate"]["ci"], "a_priori": e["a_priori"]}
                   for p, e in r["comparisons"].items()}}
    print("\n".join(rows))
    for key, v in out["runs"].items():
        if "_K" in key:
            print(f"{key}: copy {v['copy_ms']:.2f} ms; split {100 * v['splitphase vs lfu_decay']['gain']:+.2f}% "
                  f"(a priori {100 * v['splitphase vs lfu_decay']['a_priori']:+.2f}%), sync "
                  f"{100 * v['global_token_gbt vs lfu_decay']['gain']:+.2f}%, hobbit {100 * v['hobbit vs lfu_decay']['gain']:+.2f}%")
    errors = [abs(e["a_priori"] - e["gain"]) for v in out["runs"].values() for k, e in v.items() if isinstance(e, dict)]
    out["max_abs_error"] = max(errors)
    print(f"a priori: {summary['a_priori']}; max abs error {100 * max(errors):.2f} points")
    dest = ROOT / "results/t5/report.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
