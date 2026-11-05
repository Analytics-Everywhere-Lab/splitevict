"""T11 analysis: the desktop, experts copied from host memory or read from its NVMe drive.

Runs (``jobs/t11.sh``, the desktop: RTX 4090 on PCIe gen4 x16, Intel i9-13900K), in
``data/runs/t11/olmoe/``, per setting (``native``: every demand copy from pinned host memory;
``nvme``: every demand copy read from the local NVMe drive with direct I/O):

* ``cal_B32_<setting>.json``: calibration, the four arms, two passes, on the first 40 calibration sequences of T5's
  fresh OLMoE trace;
* ``B32_<setting>.json``: the four arms, two passes, on T11's fresh trace (examples 325-334 of each domain, 40
  sequences);
* ``dev_costs_*.json`` and ``predictions_*.json``: the calibration costs, gains and directional hypotheses of both
  settings, hashed into the job log before the fresh trace existed;
* per run: the GPU's processes at the end (``.gpu_end``) and every 30 s (``.gpu_apps``, first line the runner's pid),
  the drive's /proc/diskstats line before and after (``.diskstats``) and, for nvme, the expert files' bytes in the
  page cache after the run (``.fincore``).

The estimator and the fidelity checks are T5's and T8's, imported from ``analyze_t5.py`` and
``analyze_t8.py``: per sequence and arm the faster of the two passes (``view_min``; the first pass, ``view_0``, is
reported and decides nothing); every comparison the paired, sequence-clustered, domain-stratified bootstrap of the
ratio of totals (1,000 replicates); "above" / "below zero" means the 95% interval lies entirely above / below zero.
The a-priori model (``dev_costs.a_priori``) takes every cost from the setting's calibration run and, from the sealed
run, only the simulator's copies per token and the base arm's wall time (``view_min``). Per setting, A1: every
comparison predicted at least one point from zero has a measured gain of the same sign; A2: the median absolute error
over the five comparisons is at most one point.

The host's page cache may not grow by 1 GiB over the arms of an nvme run (as in T8). Other users share the desktop, so
a growth of 1 GiB or more is decided by the expert files' own bytes in the page cache after the run, which must then be
under 1 GiB.

    python analysis/analyze_t11.py
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

from analyze_t5 import SubsetPath, load_trace, model_params, predicted_gain, rows, view  # noqa: E402
from analyze_t8 import CACHE_GROWTH_LIMIT_KIB, COMPARISONS, SEALED_ARMS, nvme_fidelity  # noqa: E402

RUNS = ROOT / "data/runs/t11/olmoe"
BUNDLES = ROOT / "data/rankers"
INFO = {"trace": ROOT / "data/traces/olmoe_t11", "spec": ROOT / "data/specs/olmoe_t11_replay_spec.json",
        "fit": ROOT / "data/baselines/olmoe_frozen.json",
        "p2": BUNDLES / "olmoe_frozen_lag_req_trees200.pkl", "p1": BUNDLES / "olmoe_frozen_lag_tok_trees200.pkl",
        "sync": ROOT / "data/rankers/olmoe_setdev_policy.pkl"}
CAP = 32
PASSES = 2
SETTINGS = ("native", "nvme")


def run_side(runs: Path, name: str, arm_read_bytes: int | None) -> dict:
    """The GPU's other processes at the end and in the 30-s samples; the drive's I/O over the whole run (model load,
    staging, probes, warm-up, the arms' reads, other users) next to the arms' own reads; the expert files' bytes in the
    page cache after the run."""

    end = [line for line in (runs / f"{name}.gpu_end").read_text().splitlines() if line.strip()]
    samples = (runs / f"{name}.gpu_apps").read_text().splitlines()
    runner = samples[0].split()[1]
    others = sorted({line.split(",")[1].strip() for line in samples[1:]
                     if line.count(",") >= 3 and line.split(",")[1].strip() != runner})
    disk = [line.split() for line in (runs / f"{name}.diskstats").read_text().splitlines()]
    read_bytes = (int(disk[1][5]) - int(disk[0][5])) * 512
    fincore = runs / f"{name}.fincore"
    out = {"gpu_free_at_end": not end, "gpu_at_end": end, "other_gpu_pids_in_samples": others,
           "gpu_samples": len(samples) - 1, "drive_read_bytes": read_bytes,
           "drive_written_bytes": (int(disk[1][9]) - int(disk[0][9])) * 512,
           "arm_read_bytes": arm_read_bytes,
           "expert_files_cached_bytes": int(fincore.read_text().strip()) if fincore.exists() else None}
    return out


def analyse(setting: str, runs: Path, info: dict, frozen_all: dict) -> dict:
    """One setting: fidelity, the comparisons (view_min, view_0), the hypotheses frozen for it, and A1/A2."""

    from paired_bootstrap import paired_bootstrap
    from dev_costs import a_priori, costs, per_token_wall
    from h100_sim import run_arm

    cal_name, run_name = f"cal_B{CAP}_{setting}", f"B{CAP}_{setting}"
    cal = json.loads((runs / f"{cal_name}.json").read_text())
    run = json.loads((runs / f"{run_name}.json").read_text())
    dev = costs(cal)
    frozen = frozen_all[setting]
    arms = run["arms"]
    spec = json.loads(info["spec"].read_text())
    trace = load_trace(info)
    fit = json.loads(info["fit"].read_text())
    out = {"setting": setting, "fidelity": {}, "comparisons": {}, "arms": {}, "simulated_per_token": {}, "per_copy": {},
           "costs_calibration": dev, "costs_frozen_equal": frozen["costs"] == dev,
           "breakeven_copy_ms": frozen["breakeven_copy_ms"], "bandwidth_probes": run.get("bandwidth_probes"),
           "placement": run.get("placement"), "decode_mhz_median": {}, "calibration_routing_match": {}}
    fid = out["fidelity"]
    both = (cal, run)
    fid["copy_repeat_as_sealed"] = all(int(r.get("copy_repeat", 1)) == 1 for r in both)
    fid["passes_as_sealed"] = all(int(r.get("passes", 1)) == PASSES for r in both)
    fid["arms_as_sealed"] = all(sorted(r["arms"]) == sorted(SEALED_ARMS) for r in both)
    fid["hobbit_w_as_sealed"] = all((r.get("baselines") or {}).get("hobbit_w") == fit["hobbit"][str(CAP)]["w"] for r in both)
    fid["placement_as_sealed"] = all(str((r.get("placement") or {}).get("rule", "")).startswith("t11") for r in both)
    fid["copy_path_as_sealed"] = all(("nvme" in r) == (setting == "nvme") for r in both)
    for label, r, name in (("calibration", cal, cal_name), ("sealed_run", run, run_name)):
        ours = None
        if setting == "nvme":
            ours = sum(r["nvme"]["arm_reads"].values()) * int(r["nvme"]["stride"])
        side = run_side(runs, name, ours)
        if setting == "nvme":
            nv = nvme_fidelity(r)
            growth_ok = nv["page_cache_held"]
            files = side["expert_files_cached_bytes"]
            nv["page_cache_held"] = growth_ok or (files is not None and files < CACHE_GROWTH_LIMIT_KIB * 1024)
            nv["decided_by"] = "host page-cache growth" if growth_ok else "expert files' bytes in the page cache"
            side["nvme"] = nv
        fid[label] = side
    order = [r["source_sequence_id"] for r in sorted(view(next(iter(arms.values()))["sequences"], 0),
                                                       key=lambda r: r["sequence_position"])]
    deploy = SubsetPath(order, spec)
    params = {"p2": info["p2"], "p1": info["p1"], "sync": info["sync"], "hobbit_w": fit["hobbit"][str(CAP)]["w"]}
    for name, a in arms.items():
        t_by = a["totals_by_pass"]
        expected = run_arm(trace, deploy, name, CAP, params)
        tokens = sum(r["tokens"] for r in view(a["sequences"], 0))
        out["simulated_per_token"][name] = {"demand": sum(e["demand"] for e in expected) / tokens, "prefetch": 0.0}
        equal = [[e["demand"] for e in expected]
                 == [r["expert_transfers"] for r in sorted(view(a["sequences"], p), key=lambda r: r["sequence_position"])]
                 for p in range(PASSES)]
        seqs = view(a["sequences"], 0)
        copies = max(sum(r["expert_transfers"] for r in seqs), 1)
        out["per_copy"][name] = {"copy_ms": sum(r["copy_ms"] for r in seqs) / copies,
                                 "issue_ms": sum(r["issue_ms"] for r in seqs) / copies,
                                 "read_ms": sum(r.get("read_ms", 0.0) for r in seqs) / copies}
        mhz = [m for r in seqs for m in r.get("token_mhz", []) if m > 0]
        out["decode_mhz_median"][name] = float(np.median(mhz)) if mhz else None
        out["calibration_routing_match"][name] = cal["arms"][name]["totals"]["routing_match_fraction"]
        fid[name] = {"routing_all_events": all(t["routing_matches"] == t["routing_events"] for t in t_by),
                     "token_agreement": [t["token_agreement_fraction"] for t in t_by],
                     "simulator_equal_every_pass": all(equal)}
        if setting == "nvme":
            fid[name]["reads_equal_transfers"] = all(r.get("reads") == r["expert_transfers"] for r in a["sequences"])
    arm_ok = all(fid[n]["routing_all_events"] and all(x == 1.0 for x in fid[n]["token_agreement"])
                 and fid[n]["simulator_equal_every_pass"] and fid[n].get("reads_equal_transfers", True) for n in arms)
    side_ok = all(fid[label]["gpu_free_at_end"] and (setting != "nvme" or (
        fid[label]["nvme"]["direct_io"] and fid[label]["nvme"]["readback_equal"] and fid[label]["nvme"]["page_cache_held"]))
        for label in ("calibration", "sealed_run"))
    sealed_ok = all(fid[k] for k in ("copy_repeat_as_sealed", "passes_as_sealed", "arms_as_sealed", "hobbit_w_as_sealed",
                                      "placement_as_sealed", "copy_path_as_sealed"))
    out["fidelity_holds"] = bool(arm_ok and side_ok and sealed_ok)
    out["arms"] = {name: model_params(a) for name, a in arms.items()}
    for method, base in COMPARISONS:
        pair = f"{method} vs {base}"
        base_wall = per_token_wall(arms[base], PASSES)
        entry = {"a_priori": a_priori(dev, out["simulated_per_token"], method, base, base_wall),
                 "calibration_gain": frozen["calibration_gain"][pair],
                 "model_run_costs": predicted_gain(out["arms"][method], out["arms"][base])}
        for v in ("min", 0):
            key = ("t11", CAP)
            w = paired_bootstrap({key: rows(view(arms[base]["sequences"], v))},
                                 {key: rows(view(arms[method]["sequences"], v))}, ["t11"], CAP)
            entry[f"view_{v}"] = {"gain": w["improvement"], "ci": [w["ci_low"], w["ci_high"]],
                                  "faster": w["sequences_improved"], "units": w["paired_units"]}
        entry["estimate"] = entry["view_min"]
        entry["above_zero"] = entry["estimate"]["ci"][0] > 0
        entry["below_zero"] = entry["estimate"]["ci"][1] < 0
        out["comparisons"][pair] = entry
    hypotheses = {}
    for hid, h in frozen["hypotheses"].items():
        e = out["comparisons"][h["comparison"]]
        if h["criterion"] is None:
            hypotheses[hid] = {"comparison": h["comparison"], "exists": False, "calibration_gain": h["calibration_gain"]}
            continue
        hypotheses[hid] = {"comparison": h["comparison"], "exists": True, "criterion": h["criterion"],
                           "measured": e["estimate"]["gain"], "ci": e["estimate"]["ci"], "a_priori": e["a_priori"],
                           "met": e["above_zero"] if h["criterion"] == "above" else e["below_zero"]}
    out["hypotheses"] = hypotheses
    pairs = list(out["comparisons"].items())
    decisive = [(pair, e) for pair, e in pairs if abs(e["a_priori"]) >= 0.01]
    errors = [abs(e["a_priori"] - e["estimate"]["gain"]) for _pair, e in pairs]
    out["a_priori"] = {
        "comparisons": len(pairs), "decisive": len(decisive),
        "A1_sign_agreement_decisive": bool(decisive) and all(np.sign(e["a_priori"]) == np.sign(e["estimate"]["gain"])
                                                            for _p, e in decisive),
        "sign_misses": [pair for pair, e in decisive if np.sign(e["a_priori"]) != np.sign(e["estimate"]["gain"])],
        "median_abs_error": float(np.median(errors)),
        "A2_median_abs_error_at_most_1pt": float(np.median(errors)) <= 0.01,
        "inside_interval": int(sum(e["estimate"]["ci"][0] <= e["a_priori"] <= e["estimate"]["ci"][1] for _p, e in pairs))}
    return out


def report(out: dict) -> None:
    print(f"== {out['setting']}: fidelity holds {out['fidelity_holds']}")
    print(f"  fidelity: {json.dumps(out['fidelity'])}")
    for name, c in out["per_copy"].items():
        print(f"  {name:18s} per copy: copy events {c['copy_ms']:.3f} ms (read {c['read_ms']:.3f}), issue "
              f"{c['issue_ms']:.3f} ms; decode core median {out['decode_mhz_median'][name]} MHz")
    for pair, e in out["comparisons"].items():
        v, v0 = e["estimate"], e["view_0"]
        print(f"  {pair:30s} {100 * v['gain']:+7.2f}% [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}] "
              f"{v['faster']}/{v['units']}; first pass {100 * v0['gain']:+.2f}%; a priori {100 * e['a_priori']:+.2f}%; "
              f"calibration {100 * e['calibration_gain']:+.2f}%")
    for hid, h in out["hypotheses"].items():
        if not h["exists"]:
            print(f"  {hid}: {h['comparison']}: no directional hypothesis (calibration gain {100 * h['calibration_gain']:+.2f}%)")
            continue
        print(f"  {hid}: {h['comparison']}: {100 * h['measured']:+.2f}% [{100 * h['ci'][0]:+.2f}, {100 * h['ci'][1]:+.2f}] "
              f"({h['criterion']} zero) -> met {h['met']}")
    print(f"  a priori: {json.dumps(out['a_priori'])}")


def main() -> None:
    frozen = json.loads((RUNS / "predictions_e23.json").read_text())
    out = {"model": "olmoe", "capacity": CAP, "host": "desktop",
           "settings": {s: analyse(s, RUNS, INFO, frozen) for s in SETTINGS}}
    dest = ROOT / "results/t11/summary.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for s in SETTINGS:
        report(out["settings"][s])
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
