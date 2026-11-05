"""T4 development pass (calibration data): timing, noise, and simulator equality of the four arms.

For every run file (``src/runtime/interleaved_prefetch.py`` output) it reports per arm the wall-time
decomposition, the paired wall-time comparisons (paired, sequence-clustered, domain-stratified bootstrap, 1,000
replicates: ``paired_bootstrap``), the per-sequence spread of the paired differences, and whether each arm's
per-sequence demand and prefetch copies equal the simulator's on the same sequences in the same order with the
guesses the run used. The runs are ``data/runs/t4_dev``: ``B32_q2`` and ``B24_q2`` from the first development
pass and ``B32`` and ``B24`` from the last, with two passes and the alternative LFU prefetch settings. On these
default runs it writes ``results/t4/dev_summary.json``.

    python analysis/analyze_t4_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "runtime"))
sys.path.append(str(ROOT / "src"))

RUNS = [ROOT / "data/runs/t4_dev" / f"{name}.json" for name in ("B32_q2", "B24_q2", "B32", "B24")]
PAIRS = (("splitphase", "lfu_decay"), ("splitphase+pf", "lfu_decay+pf"), ("lfu_decay+pf", "lfu_decay"),
         ("splitphase+pf", "splitphase"), ("splitphase+pf", "lfu_decay"))


def rows(seqs, key):
    return [(r["sequence_position"], r["source_sequence_id"], r["domain"], float(r[key])) for r in seqs]


def by_pass(seqs, which):
    """One row per sequence: pass ``which`` (an int), or the pass with the smaller wall time ("min")."""

    if which != "min":
        return [r for r in seqs if r.get("pass", 0) == which]
    best: dict[int, dict] = {}
    for r in seqs:
        sid = r["source_sequence_id"]
        if sid not in best or r["wall_ms"] < best[sid]["wall_ms"]:
            best[sid] = r
    first = [r for r in seqs if r.get("pass", 0) == 0]
    return [dict(best[r["source_sequence_id"]], sequence_position=r["sequence_position"]) for r in first]


def per_token(seqs, key):
    return np.array([r[key] / max(r["tokens"], 1) for r in seqs])


class LiveGuesses:
    """The guesses one arm used (the run's ``.guesses.npz``), in prefetch_sim.Guesses' interface."""

    def __init__(self, path: Path, arm: str, m: int):
        data = np.load(path)
        self.by_sequence = {int(k.split("/")[1]): data[k].astype(np.int64) for k in data.files if k.startswith(f"{arm}/")}
        self.variant = "live"
        self.meta = {"spec": str(path), "device": "live"}
        self.m = m

    def get(self, sid, token, layer, m):
        return self.by_sequence[sid][token, layer, :m]


def simulate(arm: str, run: dict, capacity: int, guesses_path: Path) -> list[dict] | None:
    """The arm's per-sequence demand and prefetch copies in the simulator, same order, same guesses."""

    import prefetch_sim

    inputs, _fit, _deploy = prefetch_sim._load_dev()
    order = [r["source_sequence_id"] for r in run["arms"][arm]["sequences"] if r.get("pass", 0) == 0]
    cal = inputs.calibration
    from residency_headroom.workloads import Workload, WorkloadSequence  # as collect2.sub builds them

    by_id = {int(s.source_sequence_id): s for s in cal.sequences}
    if any(sid not in by_id for sid in order):
        return None
    workload = Workload("t4_dev_order", cal.regime, tuple(
        WorkloadSequence(by_id[sid].source_sequence_id, i, by_id[sid].segment_index, by_id[sid].segment_label,
                         by_id[sid].domain) for i, sid in enumerate(order)))
    pf = dict(run["prefetch"])
    pf.update(run["arms"][arm].get("prefetch") or {})
    with_pf = run["arms"][arm].get("prefetch") is not None or arm.endswith("+pf")
    guesses = LiveGuesses(guesses_path, arm, pf["m"]) if with_pf else None
    policy = "lfu" if arm.startswith("lfu") else "splitphase"
    wrong = pf["lfu_wrong_rule"] if policy == "lfu" else "last"
    bundles = "dev" if "dev_lag" in str(run["bundles"]["p2"]["path"] if run["bundles"]["p2"] else "dev_lag") else "frozen"
    original = prefetch_sim._DEV
    prefetch_sim._DEV = (inputs, workload, workload)
    try:
        result = prefetch_sim.run(policy, capacity, guesses, pf["m"], pf["q"], "calB", bundles, wrong)
    finally:
        prefetch_sim._DEV = original
    return result["per_sequence"]


def main() -> None:
    from paired_bootstrap import paired_bootstrap

    paths = [Path(arg) for arg in sys.argv[1:]] or RUNS
    report = {}
    for path in paths:
        run = json.loads(path.read_text())
        B = int(run["capacity"])
        arms = run["arms"]
        entry = {"capacity": B, "prefetch": run["prefetch"], "placement": None if run["placement"] is None else
                 {k: run["placement"][k] for k in ("workers", "excluded_siblings")}, "arms": {}, "pairs": {},
                 "bandwidth_probes": run["bandwidth_probes"], "splitphase": run["splitphase"]}
        for name, a in arms.items():
            t = a["totals"]
            seqs = by_pass(a["sequences"], 0)
            entry["arms"][name] = {
                "wall": t["ms_per_token_wall"], "policy": t["ms_per_token_policy"],
                "prefetch_policy": t["ms_per_token_prefetch_policy"], "issue": t["ms_per_token_issue"],
                "prefetch_issue": t["ms_per_token_prefetch_issue"], "copy": t["ms_per_token_copy"],
                "prefetch_copy": t["ms_per_token_prefetch_copy"], "stall": t["ms_per_token_prefetch_stall"],
                "demand_per_token": t["transfers_per_token"], "prefetch_per_token": t["prefetches_per_token"],
                "hits_per_token": t.get("prefetch_hits", 0) / max(t["tokens"], 1),
                "ms_per_copy": t["ms_per_copy"], "routing": t["routing_match_fraction"],
                "guess_match": t["guess_match_fraction"], "tokens": t["tokens"],
                "noncopy": float(np.mean(per_token(seqs, "wall_ms") - per_token(seqs, "copy_ms"))),
                "noncopy_p90": float(np.percentile(per_token(seqs, "wall_ms") - per_token(seqs, "copy_ms"), 90)),
            }
        pairs = [pair for pair in PAIRS if pair[0] in arms and pair[1] in arms]
        pairs += [(name, name.split("+")[0]) for name in arms if ":" in name and name.split("+")[0] in arms]
        passes = int(run.get("passes", 1))
        views = list(range(passes)) + (["min"] if passes > 1 else [])
        for method, base in pairs:
            if method in arms and base in arms:
                for view in views:
                    key = ("dev", B)
                    a = by_pass(arms[base]["sequences"], view)
                    b = by_pass(arms[method]["sequences"], view)
                    w = paired_bootstrap({key: rows(a, "wall_ms")}, {key: rows(b, "wall_ms")}, ["dev"], B)
                    d = per_token(a, "wall_ms") - per_token(b, "wall_ms")
                    label = f"{method} vs {base}" + ("" if passes == 1 else f" [pass {view}]")
                    entry["pairs"][label] = {
                        "gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]], "faster": w["sequences_improved"],
                        "units": w["paired_units"], "paired_sd_ms_per_token": float(d.std(ddof=1)),
                        "mean_diff_ms_per_token": float(d.mean())}
        guesses_path = path.with_suffix(".guesses.npz")
        sim = {}
        for name in arms:
            try:
                expected = simulate(name, run, B, guesses_path)
            except Exception as exc:  # report, do not hide
                sim[name] = f"simulator failed: {type(exc).__name__}: {exc}"
                continue
            if expected is None:
                sim[name] = "sequences not in the calibration split"
                continue
            checks = []
            for view in range(passes):
                live = by_pass(arms[name]["sequences"], view)
                checks.append({
                    "demand_equal": [e["demand"] for e in expected] == [r["expert_transfers"] for r in live],
                    "prefetch_equal": [e["prefetch"] for e in expected] == [r.get("prefetch_transfers", 0) for r in live],
                    "hits_equal": [e["hits"] for e in expected] == [r.get("prefetch_hits", 0) for r in live],
                    "sequences": len(live)})
            sim[name] = checks[0] if passes == 1 else {"all_passes_equal": all(all(v for k, v in c.items() if k != "sequences")
                                                                                for c in checks), "per_pass": checks}
        entry["simulator_equal"] = sim
        report[path.stem] = entry
        print(f"== {path.name}: B={B}, prefetch {run['prefetch']}")
        for name, a in entry["arms"].items():
            print(f"  {name:14s} wall {a['wall']:6.2f} | policy {a['policy']:.2f}+{a['prefetch_policy']:.2f} issue "
                  f"{a['issue']:.2f}+{a['prefetch_issue']:.2f} copy {a['copy']:5.2f} pf-copy {a['prefetch_copy']:5.2f} "
                  f"stall {a['stall']:.2f} | demand {a['demand_per_token']:5.2f} pf {a['prefetch_per_token']:5.2f} "
                  f"hits {a['hits_per_token']:5.2f} | noncopy {a['noncopy']:.2f} (p90 {a['noncopy_p90']:.2f}) "
                  f"| routing {100 * a['routing']:.1f}% guesses {a['guess_match']}")
        for pair, p in entry["pairs"].items():
            print(f"  {pair:30s} {100 * p['gain']:+6.2f}% [{100 * p['ci'][0]:+.2f}, {100 * p['ci'][1]:+.2f}] "
                  f"{p['faster']}/{p['units']} faster, paired sd {p['paired_sd_ms_per_token']:.2f} ms/token")
        print(f"  simulator equal: {sim}")
    if paths == RUNS:
        out = ROOT / "results/t4/dev_summary.json"
        out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
