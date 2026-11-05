"""Cost-aware policy selection: split-phase eviction only where the break-even model, calibrated on the host, says it pays.

A deployment cannot know the copy counts of traffic it has not served yet. The selector therefore uses one
*calibration run* on the target host (an ``interleaved_h100.py`` output with the arms ``lfu_decay`` and
``splitphase``, on prompts outside the rankers' fit and outside any test) and nothing else:

1. ``dev_costs.costs`` turns the run into the break-even model's terms: the host's per-copy costs (t_copy, t_issue),
   and per arm the copies per token, the timed policy cost per token, the wall time per token and the residual
   (the wall-time difference from decayed LFU that the timed costs do not explain).
2. For a copy cost K times the calibrated one, ``predictions_t6.gain`` predicts split-phase's gain over decayed LFU:
   copy and issue time K times their calibrated values, policy cost and residual unchanged (T6's rule), copies
   per token and wall times from the calibration run itself.
3. Split-phase is turned on when the prediction is at least MARGIN, one percentage point (T5's threshold
   for a decisive prediction); otherwise the host keeps decayed LFU.

At the calibrated copy cost (K = 1) the prediction equals the calibration run's own measured difference, because the
residual closes the model on that run. At other copy costs the model extrapolates from one run. With the residual
defined on the calibration run, the timed and untimed costs cancel and the prediction reduces to

    G(K) = [(W_lfu - W_split) + (D_lfu - D_split)(K - 1) c] / [W_lfu + D_lfu (K - 1) c],   c = t_copy + t_issue:

the calibration's measured difference plus the copies split-phase saves, priced at the extra copy time, over decayed
LFU's wall time at K.

``evaluate`` scores decisions against a later measurement: a *clear loss* is a setting whose measured 95% interval
for split-phase against decayed LFU lies entirely below zero, a *clear gain* one whose interval lies entirely above
MARGIN. The selector should be off at every clear loss and on at every clear gain; between the two, either choice
costs at most about one point, and no claim is made. Regret is the measured gain given up against the faster of the two
policies (point estimates).

    python analysis/selector.py decide CAL.json --ks 1 2 4 --output decisions.json
    python analysis/selector.py t5             # retrospective table on T5 (development data)
    python analysis/selector.py t6-decisions   # T6's decisions, sealed before the T6 run
    python analysis/selector.py t6-evaluate    # after analyze_t6.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dev_costs import costs  # noqa: E402
from predictions_t6 import gain  # noqa: E402

MARGIN = 0.01
METHOD, BASE = "splitphase", "lfu_decay"


def predicted_gain(entry: dict, k: int) -> float:
    """Split-phase vs decayed LFU at K times the calibrated copy cost, from one ``dev_costs.costs`` entry."""

    copies = {name: entry["arms"][name]["demand"] for name in (METHOD, BASE)}
    walls = {name: entry["arms"][name]["wall"] for name in (METHOD, BASE)}
    return gain(entry, copies, walls, METHOD, BASE, k)


def decide(entry: dict, ks, margin: float = MARGIN) -> dict:
    out = {}
    for k in ks:
        g = predicted_gain(entry, int(k))
        out[str(k)] = {"copy_ms": entry["t_copy"] * int(k), "predicted_gain": g,
                       "policy": METHOD if g >= margin else BASE}
    return out


def calibration_costs(path: Path) -> dict:
    run = json.loads(path.read_text())
    missing = {METHOD, BASE} - set(run["arms"])
    if missing:
        raise SystemExit(f"calibration run {path} lacks arms {sorted(missing)}")
    return costs(run)


def evaluate(decisions: dict, measured: dict, margin: float = MARGIN) -> dict:
    """``decisions``: setting -> chosen policy; ``measured``: setting -> {"gain", "ci"} of split-phase vs decayed LFU."""

    settings = {}
    for key, policy in decisions.items():
        m = measured.get(key)
        if m is None:
            settings[key] = {"policy": policy, "measured": None}
            continue
        g, (lo, hi) = m["gain"], m["ci"]
        on = policy == METHOD
        kind = "clear_loss" if hi < 0 else "clear_gain" if lo > margin else "no_claim"
        settings[key] = {"policy": policy, "measured": g, "ci": [lo, hi], "kind": kind,
                         "correct": (not on) if kind == "clear_loss" else on if kind == "clear_gain" else None,
                         "realized": g if on else 0.0, "regret": max(g, 0.0) - (g if on else 0.0)}
    scored = [s for s in settings.values() if s.get("measured") is not None]
    losses = [s for s in scored if s["kind"] == "clear_loss"]
    gains = [s for s in scored if s["kind"] == "clear_gain"]
    return {"margin": margin, "settings": settings,
            "avoids_every_clear_loss": all(s["correct"] for s in losses),
            "keeps_every_clear_gain": all(s["correct"] for s in gains),
            "clear_losses": len(losses), "clear_gains": len(gains),
            "regret_sum": {"selector": sum(s["regret"] for s in scored),
                           "always_splitphase": sum(max(-s["measured"], 0.0) for s in scored),
                           "always_lfu": sum(max(s["measured"], 0.0) for s in scored)},
            "worst_realized": min((s["realized"] for s in scored), default=None)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- T5, retrospective (development data only) --------------------------------------------------------------------

T5_RESULTS = ROOT / "results/t5"
T5_SETTINGS = (  # (report key, model, capacity, K, development entry: (path into dev_costs.json), K relative to it)
    ("mixtral_B4", "mixtral", 4, 1, ("mixtral", "4"), 1), ("mixtral_B3", "mixtral", 3, 1, ("mixtral", "3"), 1),
    ("olmoe_B32", "olmoe", 32, 1, ("olmoe", "32"), 1), ("olmoe_B24", "olmoe", 24, 1, ("olmoe", "24"), 1),
    ("qwen3_B64", "qwen3", 64, 1, ("qwen3", "64"), 1), ("qwen3_B48", "qwen3", 48, 1, ("qwen3", "48"), 1),
    ("olmoe_B32_K2", "olmoe", 32, 2, ("olmoe", "32"), 2),
    ("olmoe_B32_K4", "olmoe", 32, 4, ("sweep", "olmoe", "32", "4"), 1),
    ("olmoe_B32_K16", "olmoe", 32, 16, ("sweep", "olmoe", "32", "16"), 1),
    ("olmoe_B32_K32", "olmoe", 32, 32, ("olmoe", "32"), 32),
    ("qwen3_B64_K4", "qwen3", 64, 4, ("qwen3", "64"), 4), ("qwen3_B64_K16", "qwen3", 64, 16, ("qwen3", "64"), 16),
)


def t5_retro() -> dict:
    """The selector on T5, calibrated on T5's development runs (original calibration prompts, run before the test):
    every input existed before the fresh traces did. The rule itself was written after T5."""

    dev = json.loads((T5_RESULTS / "dev_costs.json").read_text())
    report = json.loads((T5_RESULTS / "report.json").read_text())["runs"]
    decisions, measured, rows = {}, {}, []
    for key, model, cap, k, where, rel in T5_SETTINGS:
        entry = dev
        for part in where:
            entry = entry[part]
        g = predicted_gain(entry, rel)
        policy = METHOD if g >= MARGIN else BASE
        m = report[key][f"{METHOD} vs {BASE}"]
        decisions[key], measured[key] = policy, {"gain": m["gain"], "ci": m["ci"]}
        rows.append({"setting": key, "model": model, "capacity": cap, "copy_repeat": k,
                     "calibration": "development run at this K" if rel == 1 else f"development run at K=1, scaled x{rel}",
                     "predicted_gain": g, "policy": policy})
    result = {"source": "T5 development cost file (dev_costs.json) -> decisions; T5 report (report.json) -> "
                        "measured", "dev_costs_sha256": sha256(T5_RESULTS / "dev_costs.json"),
              "report_sha256": sha256(T5_RESULTS / "report.json"), "rows": rows, "evaluation": evaluate(decisions, measured)}
    print(f"{'setting':15s} {'calibration':36s} {'predicted':>9s}  {'policy':10s} {'measured':>8s}  interval          regret")
    for row in rows:
        s = result["evaluation"]["settings"][row["setting"]]
        print(f"{row['setting']:15s} {row['calibration']:36s} {100 * row['predicted_gain']:+8.2f}%  {row['policy']:10s} "
              f"{100 * s['measured']:+7.2f}%  [{100 * s['ci'][0]:+.2f}, {100 * s['ci'][1]:+.2f}]  {100 * s['regret']:.2f}")
    ev = result["evaluation"]
    print(f"clear losses {ev['clear_losses']} (all avoided: {ev['avoids_every_clear_loss']}), clear gains "
          f"{ev['clear_gains']} (all kept: {ev['keeps_every_clear_gain']}); worst realized {100 * ev['worst_realized']:+.2f}%; "
          "regret sums (points): " + ", ".join(f"{n} {100 * v:.2f}" for n, v in ev["regret_sum"].items()))
    dest = T5_RESULTS / "selector_retro.json"
    dest.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", dest)
    return result


# ---- T6 selector decisions, sealed before the T6 run ---------------------------------------------------------------

T6_CALIBRATION = ROOT / "data/runs/t5/mixtral/B4.json"
T6_KS = (1, 2, 4, 7)


def t6_decisions() -> dict:
    """Calibration: T5's Mixtral B=4 run (same node type, prompts outside the frozen rankers' fit and disjoint from
    T6's). Decisions for T6's copy repeats K = 1, 2, 4, 7."""

    entry = calibration_costs(T6_CALIBRATION)
    out = {"calibration": str(T6_CALIBRATION.relative_to(ROOT)), "calibration_sha256": sha256(T6_CALIBRATION),
           "margin": MARGIN, "t_copy": entry["t_copy"], "t_issue": entry["t_issue"],
           "arms": {n: entry["arms"][n] for n in (BASE, METHOD)}, "decisions": decide(entry, T6_KS)}
    for k, d in out["decisions"].items():
        print(f"K={k}: copy {d['copy_ms']:.1f} ms, predicted {100 * d['predicted_gain']:+.2f}% -> {d['policy']}")
    dest = ROOT / "results/t6/selector_decisions.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)
    return out


def t6_evaluate() -> dict:
    """The sealed T6 decisions against T6's measured gains (``results/t6/summary.json``, ``analyze_t6.py``)."""

    sealed = json.loads((ROOT / "results/t6/selector_decisions.json").read_text())
    summary = json.loads((ROOT / "results/t6/summary.json").read_text())["runs"]
    decisions = {k: d["policy"] for k, d in sealed["decisions"].items()}
    measured = {}
    for k in decisions:
        run = summary.get(f"mixtral_B4_K{k}")
        if run is not None:
            measured[k] = run["comparisons"][f"{METHOD} vs {BASE}"]["estimate"]
    result = evaluate(decisions, measured)
    dest = ROOT / "results/t6/selector.json"
    dest.write_text(json.dumps(result, indent=1) + "\n")
    for k, s in result["settings"].items():
        if s.get("measured") is None:
            print(f"K={k}: {s['policy']}, run missing")
            continue
        print(f"K={k}: {s['policy']}; measured {100 * s['measured']:+.2f}% [{100 * s['ci'][0]:+.2f}, "
              f"{100 * s['ci'][1]:+.2f}] ({s['kind']}), regret {100 * s['regret']:.2f} points")
    print(f"avoids every clear loss: {result['avoids_every_clear_loss']} ({result['clear_losses']}); keeps every clear "
          f"gain: {result['keeps_every_clear_gain']} ({result['clear_gains']}); wrote {dest}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decide", help="decisions from one calibration run")
    d.add_argument("calibration", type=Path)
    d.add_argument("--ks", type=int, nargs="+", required=True)
    d.add_argument("--margin", type=float, default=MARGIN)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--label", default="")
    sub.add_parser("t5")
    sub.add_parser("t6-decisions")
    sub.add_parser("t6-evaluate")
    args = parser.parse_args()
    if args.command == "t5":
        t5_retro()
    elif args.command == "t6-decisions":
        t6_decisions()
    elif args.command == "t6-evaluate":
        t6_evaluate()
    else:
        if args.output.exists():
            raise SystemExit(f"REFUSED: {args.output} exists; decisions are made once and never recomputed")
        entry = calibration_costs(args.calibration)
        out = {"label": args.label, "calibration": str(args.calibration), "calibration_sha256": sha256(args.calibration),
               "margin": args.margin, "t_copy": entry["t_copy"], "t_issue": entry["t_issue"],
               "arms": {n: entry["arms"][n] for n in (BASE, METHOD)}, "decisions": decide(entry, args.ks, args.margin)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + "\n")
        for k, v in out["decisions"].items():
            print(f"{args.label} K={k}: copy {v['copy_ms']:.3f} ms, predicted {100 * v['predicted_gain']:+.2f}% -> {v['policy']}")
        print(f"decisions sha256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
