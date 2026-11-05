"""Where split-phase's extra time per token against decayed LFU goes, in the H100 runs of T5 and T7 (exploratory).

Per arm and token, every run row already records the wall time, the time inside the offloaded MoE forwards (``moe_ms``),
and inside them the CPU's wait for the GPU at each router read (``sync_ms``), the timed decisions (``policy_ms``) and the
slot bookkeeping and copy issue (``issue_ms``); the copy time on the GPU (``copy_ms``, CUDA events); and, with
``--monitor``, the decode thread's core clock per token (``token_mhz``). Split-phase minus decayed LFU, per token:

* non-MoE time (attention, norms, head: the decode thread's launch work outside the MoE layers) and the MoE forward's
  untimed rest: a slower decode thread (a shared socket's clock or cache) would show here;
* the copy saving the break-even model counts, D_saved x (t_copy + t_issue) with decayed LFU's per-copy costs, against
  the saving that reaches the wall clock through the router reads and the copy issue, -(d sync + d issue). The
  difference is copy time the GPU spent while the CPU was still launching kernels: hidden copies save no time.

    python analysis/remainder_split.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/runs"
FILES = [("T5", "t5/olmoe/B32.json"), ("T5", "t5/olmoe/B32_K2.json"), ("T5", "t5/olmoe/B32_K4.json"),
         ("T5", "t5/olmoe/B32_K16.json"), ("T5", "t5/olmoe/B32_K32.json"), ("T5", "t5/olmoe/B24.json"),
         ("T5", "t5/qwen3/B64.json"), ("T5", "t5/qwen3/B64_K4.json"), ("T5", "t5/qwen3/B64_K16.json"),
         ("T5", "t5/qwen3/B48.json"), ("T5", "t5/mixtral/B4.json"), ("T5", "t5/mixtral/B3.json"),
         ("T7", "t7/olmoe/B32_K1.json"), ("T7", "t7/olmoe/B32_K2.json"), ("T7", "t7/olmoe/B32_K4.json"),
         ("T7", "t7/qwen3/B64_K1.json"), ("T7", "t7/qwen3/B64_K2.json"), ("T7", "t7/qwen3/B64_K4.json")]


def per_token(rows: list[dict]) -> dict:
    tokens = sum(r["tokens"] for r in rows)

    def f(key):
        return sum(r.get(key, 0.0) for r in rows) / tokens

    mhz = [m for r in rows for m in r.get("token_mhz", []) if m > 0]
    out = {key: f(key) for key in ("wall_ms", "moe_ms", "sync_ms", "policy_ms", "issue_ms", "copy_ms", "expert_transfers")}
    out["non_moe_ms"] = out["wall_ms"] - out["moe_ms"]
    out["moe_rest_ms"] = out["moe_ms"] - out["sync_ms"] - out["policy_ms"] - out["issue_ms"]
    out["core_mhz_median"] = statistics.median(mhz) if mhz else None
    out["core_mhz_mean"] = statistics.fmean(mhz) if mhz else None
    out["tokens"] = tokens
    return out


def main() -> None:
    result = []
    for test, name in FILES:
        path = RUNS / name
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        arms = {arm: per_token([r for r in a["sequences"] if r.get("pass", 0) == 0])
                for arm, a in run["arms"].items() if arm in ("lfu_decay", "splitphase")}
        lfu, split = arms["lfu_decay"], arms["splitphase"]
        d = {key: split[key] - lfu[key] for key in lfu if isinstance(lfu[key], float) and key not in ("core_mhz_median",)}
        saved = -d["expert_transfers"]
        per_copy = (lfu["copy_ms"] + lfu["issue_ms"]) / lfu["expert_transfers"]
        counted = saved * per_copy
        visible = -(d["sync_ms"] + d["issue_ms"])
        entry = {"test": test, "run": name, "copy_repeat": int(run.get("copy_repeat", 1)), "arms": arms, "split_minus_lfu": d,
                 "copies_saved_per_token": saved, "lfu_ms_per_copy": per_copy, "model_copy_saving_ms": counted,
                 "visible_copy_saving_ms": visible, "hidden_copy_saving_ms": counted - visible,
                 "visible_share": visible / counted if counted else None}
        result.append(entry)
        print(f"{test} {name:28s} K={entry['copy_repeat']:2d}: wall {d['wall_ms']:+.3f} ms/tok = policy {d['policy_ms']:+.3f}"
              f" - visible saving {visible:.3f} (model counts {counted:.3f}, share {entry['visible_share']:.2f})"
              f" + non-MoE {d['non_moe_ms']:+.3f} + MoE rest {d['moe_rest_ms']:+.3f}; core MHz {lfu['core_mhz_median']} / "
              f"{split['core_mhz_median']}")
    dest = ROOT / "results/t5/remainder_split.json"
    dest.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
