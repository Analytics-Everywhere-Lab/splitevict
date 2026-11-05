"""The paper's Figures 2-6, written to figures/.

- fig_bottleneck.pdf (Figure 2): each A40 policy's wall time per token as stacked bars: the time with all experts
  resident, eviction decisions, expert copies, and copy issue plus the rest (``results/a40/summary.json``).
- fig_stale.pdf (Figure 3): the share of the synchronous ranker's saving kept by the stale rankings, with and without
  training on stale features, and by decayed LFU's candidates reordered near the cutoff, at B = 32 and 24.
- fig_regret_by_offset.pdf (Figure 4): where the stale and fresh rankers pick different victims, how often the stale
  choice is worse or better, by how many ranks above the fresh cutoff the stale victim sits (B = 32 and 24).
  Labels mark the two ends: "about even" next to the cutoff, "stale worse" far above it.
- fig_all_runs.pdf (Figure 5): every comparison of SplitEvict with decayed LFU on unseen prompts (T1-T11), gain
  against the copy time per expert, on all four GPUs.
- fig_h100_sweep.pdf (Figure 6): the data and model curves of ``fig_h100_sweep.py``, plus shading where the
  break-even model predicts that SplitEvict beats decayed LFU (right of its break-even point) and diamonds at the
  break-even points of SplitEvict and the synchronous ranker.

    python analysis/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_h100_sweep as sweep  # noqa: E402
from figstyle import BLUE, COLUMN_IN, GRAY, GREEN, INK, SKY, TEXT_IN, VERMILLION, save, use  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
NAME = "SplitEvict"
SPLIT = "splitphase vs lfu_decay"


def load(path: str):
    return json.loads((ROOT / path).read_text())


def bottleneck() -> None:
    pol = load("results/a40/summary.json")["policies"]
    rows = (("lfu_decay@32", "Decayed LFU"), ("lru@32", "LRU"), ("global_token_gbt@32", "Learned ranker"),
            ("caer_gbt@32", "+ second-stage\nreranker"))
    base = pol["resident@64"]["ms_per_token_wall"]
    parts = (("all experts resident", "#D9D9D9"), ("eviction decisions", VERMILLION),
             ("expert copies (number per token)", BLUE), ("copy issue and rest", SKY))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.408))
    y = np.arange(len(rows))[::-1]
    for yi, (key, label) in zip(y, rows):
        e = pol[key]
        seg = (base, e["ms_per_token_policy"], e["ms_per_token_copy"],
               e["ms_per_token_wall"] - base - e["ms_per_token_policy"] - e["ms_per_token_copy"])
        left = 0.0
        for (name, color), w in zip(parts, seg):
            ax.barh(yi, w, left=left, height=0.56, color=color, edgecolor="white", linewidth=0.4)
            left += w
        ax.text(left + 0.6, yi, f"{e['ms_per_token_wall']:.1f} ms", va="center", ha="left", fontsize=6, color=INK)
        ax.text(base + e["ms_per_token_policy"] + e["ms_per_token_copy"] / 2, yi, f"{e['transfers_per_token']:.1f}",
                va="center", ha="center", fontsize=5.6, color="white")
    lfu, gl = pol["lfu_decay@32"], pol["global_token_gbt@32"]
    # the verdict: the learned ranker's extra decision time against the copy time it saves
    extra = gl["ms_per_token_policy"] - lfu["ms_per_token_policy"]
    saved = lfu["ms_per_token_copy"] - gl["ms_per_token_copy"]
    ax.text(base + 0.3, (y[1] + y[2]) / 2, f"+{extra:.1f} ms deciding, −{saved:.1f} ms copying",
            fontsize=6, color=VERMILLION, ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlim(0, 68)
    ax.set_ylim(-0.45, len(rows) - 0.55)
    ax.set_xlabel("wall time per token (ms)")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=c, label=n) for n, c in parts]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=5.8, bbox_to_anchor=(-0.02, 0.98), handlelength=1.0,
              columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, OUT / "fig_bottleneck.pdf")
    print(f"bottleneck: resident {base:.2f} ms; learned ranker +{extra:.2f} ms deciding, saves {saved:.2f} ms copying")


def stale_rows() -> list[tuple[str, float, float, bool]]:
    """Share of the synchronous ranker's saving kept (%): (variant, B = 32, B = 24, trained on stale features)."""
    stale = {(r["arm"], r["B"]): 100 * r["retention"] for r in load("results/simulation/staleness_summary.json")}
    trained = {(r["arm"], r["B"]): 100 * r["retention"] for r in load("results/simulation/refresh_summary.json")}
    variants = (("Stale ranking", stale, "lag_req"), ("trained on stale features", trained, "lag_req_native"),
                ("Stale ranking, previous token", stale, "lag_tok"),
                ("trained on stale features", trained, "lag_tok_native"),
                ("Decayed LFU + learned order near cutoff", stale, "lfu+fresh"),
                ("Decayed LFU + true order near cutoff", stale, "lfu+oracle"))
    return [(name, src[(arm, 32)], src[(arm, 24)], src is trained) for name, src, arm in variants]


def stale() -> None:
    rows = stale_rows()
    labels = {"Stale ranking": "stale ranking", "Stale ranking, previous token": "stale, previous token",
              "Decayed LFU + learned order near cutoff": "HALP-style: LFU picks,\nranker orders",
              "Decayed LFU + true order near cutoff": "LFU picks,\ntrue order"}
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.54))
    y = np.arange(len(rows))[::-1].astype(float)
    y[-2:] -= 0.35  # a gap before the HALP-style rows
    h = 0.36
    for yi, (name, b32, b24, sub) in zip(y, rows):
        halp = name.startswith("Decayed LFU")
        c32, c24 = (GRAY, "#C4C4C4") if halp else (GREEN, "#7FCFB8")
        ax.barh(yi + h / 2, b32, height=h, color=c32, edgecolor="white", linewidth=0.3)
        ax.barh(yi - h / 2, b24, height=h, color=c24, edgecolor="white", linewidth=0.3)
        for off, v, col in ((h / 2, b32, "white"), (-h / 2, b24, INK)):
            ax.text(v - 1.2, yi + off, f"{v:.1f}", va="center", ha="right", fontsize=5.4, color=col)
    names = ["+ trained on stale\nfeatures" if sub else labels.get(name, name) for name, *_, sub in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.8, linespacing=0.95)
    ax.axvline(100, color=INK, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(99, y[0] + 0.62, "synchronous\nranker = 100%", ha="right", va="bottom", fontsize=5.8, color=INK,
            linespacing=0.95)
    ax.text(50, (y[-2] + y[-3]) / 2 + 0.02, "decayed LFU's candidates lose the coarse order", ha="center", va="center",
            fontsize=5.8, color=GRAY)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.25)
    ax.set_xlabel("share of the synchronous ranker's saving kept (%)")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=GREEN, label="$B{=}32$"), Patch(color="#7FCFB8", label="$B{=}24$")]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=6, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout(pad=0.3)
    save(fig, OUT / "fig_stale.pdf")
    print("stale:", [(r[0], round(r[1], 1), round(r[2], 1)) for r in rows])


def regret() -> None:
    buckets = ["-1", "-2", "-3..-4", "-5..-8", "<=-9"]
    labels = ["1", "2", "3–4", "5–8", "9+"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.62), sharey=True)
    for ax, B in zip(axes, (32, 24)):
        r = json.load(open(ROOT / f"results/simulation/regret/lag_req_regret_B{B}.json"))["regret"]
        worse, better = [], []
        for b in buckets:
            w, bt, t = (r.get(f"{b}|{o}", 0) for o in ("lag_worse", "lag_better", "tie"))
            n = max(w + bt + t, 1)
            worse.append(100 * w / n)
            better.append(100 * bt / n)
        x = list(range(len(buckets)))
        ax.bar([i - 0.19 for i in x], worse, 0.36, color=VERMILLION, lw=0)
        ax.bar([i + 0.19 for i in x], better, 0.36, color=BLUE, lw=0)
        # the verdict at both ends: even next to the cutoff, worse far above it
        ax.text(0, max(worse[0], better[0]) + 4, "about\neven", ha="center", va="bottom", fontsize=6, color=INK,
                linespacing=0.95)
        ax.text(4, max(worse[4], better[4]) + 4, "stale\nworse", ha="center", va="bottom", fontsize=6, color=INK,
                linespacing=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(buckets) - 0.4)
        ax.set_title(f"$B\\!=\\!{B}$")
    axes[0].set_ylim(0, 86)
    axes[0].set_yticks([0, 20, 40, 60])
    axes[0].set_ylabel("disagreements (%)")
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("stale victim's ranks above the fresh cutoff", fontsize=7, y=0.01)
    fig.legend([Patch(color=VERMILLION), Patch(color=BLUE)], ["stale choice worse", "stale choice better"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0), handlelength=1.0)
    fig.tight_layout(pad=0.25, w_pad=0.8, rect=(0, 0, 1, 0.9))
    save(fig, OUT / "fig_regret_by_offset.pdf")


def interval(e: dict) -> tuple[float, float, float]:
    """(gain, low, high) in %, from either form of a paired comparison with its 95% interval."""
    if "improvement" in e:
        return 100 * e["improvement"], 100 * e["ci_low"], 100 * e["ci_high"]
    return 100 * e["gain"], 100 * e["ci"][0], 100 * e["ci"][1]


def all_runs_rows() -> list[dict]:
    """Every comparison of SplitEvict with decayed LFU on unseen prompts, in the order of the appendix's table of all
    runs, then the H100 runs at native copy cost. Copy: decayed LFU's ms per expert (the selector runs: the
    calibration run's at the same K)."""
    rows = []

    def add(gpu: str, model: str, cap: int, copy: float, e: dict, k: int = 1) -> None:
        gain, lo, hi = interval(e)
        rows.append(dict(gpu=gpu, model=model, B=cap, K=k, copy=copy, gain=gain, lo=lo, hi=hi,
                         prefetch="prefetch" in model, graphs="graphs" in model, two="two workers" in model))

    def run(gpu: str, model: str, cap: int, r: dict, k: int = 1) -> None:
        add(gpu, model, cap, r["arms"]["lfu_decay"]["t_copy"], r["comparisons"][SPLIT]["estimate"], k)

    t1 = load("results/t1/summary.json")
    for cap in (32, 24):
        lfu = t1["gpu"][f"lfu_decay@{cap}"]
        e = next(c["wall_time_reduction"] for c in t1["gpu_comparisons"]
                 if (c["capacity"], c["candidate"], c["baseline"]) == (cap, "splitphase", "lfu_decay"))
        add("A40", "OLMoE", cap, lfu["ms_per_token_copy"] / lfu["transfers_per_token"], e)
    t2 = load("results/t2/summary.json")["capacities"]
    for cap in (32, 24):
        add("A40", "OLMoE", cap, t2[str(cap)]["lfu_decay"]["ms_per_copy"], t2[str(cap)]["wall_time_reduction"])
    t4 = load("results/t4/summary.json")["capacities"]  # the faster of two passes per sequence
    for model, pf in (("OLMoE", ""), ("OLMoE, prefetch", "+pf")):
        for cap in (32, 24):
            c = t4[str(cap)]
            add("A40", model, cap, c["arms"][f"lfu_decay{pf}"]["t_copy"],
                c["comparisons"][f"splitphase{pf} vs lfu_decay{pf}"]["view_min"])
    t3 = load("results/t3/summary.json")["models"]
    for gpu, model, name, cap in (("A40", "Qwen3", "qwen3_30b_a3b", 64), ("A40", "Qwen3", "qwen3_30b_a3b", 48),
                                  ("A100", "Mixtral", "mixtral_8x7b", 4)):
        c = t3[name][str(cap)]
        add(gpu, model, cap, c["totals"]["lfu_decay"]["ms_per_copy"], c["split_vs_lfu"])
    t6 = load("results/t6/summary.json")["runs"]
    for k in (1, 2, 4, 7):
        run("H100", "Mixtral", 4, t6[f"mixtral_B4_K{k}"], k)
    t7 = load("results/t7/summary.json")
    for model, name, cap in (("OLMoE", "olmoe", 32), ("Qwen3", "qwen3", 64)):
        for k in (1, 2, 4):
            key = f"{name}_K{k}"
            add("H100", model, cap, t7["calibration"][key]["decision"]["copy_ms"], t7["evaluation"][key]["measured"], k)
    run("H100", "OLMoE, NVMe", 32, load("results/t8/summary.json"))
    t9 = load("results/t9/summary.json")["runs"]
    for k in (1, 2, 4):
        run("H100", "OLMoE, graphs", 32, t9[f"olmoe_B32_K{k}"], k)
    run("H100", "OLMoE, graphs, two workers", 32, load("results/t10/summary.json"))
    run("H100", "OLMoE, graphs, two workers", 24, load("results/t10/summary_b24.json"))
    t11 = load("results/t11/summary.json")["settings"]
    run("RTX 4090", "OLMoE", 32, t11["native"])
    run("RTX 4090", "OLMoE, NVMe", 32, t11["nvme"])
    t5 = load("results/t5/summary.json")["runs"]
    for model, name, cap in (("Mixtral", "mixtral", 4), ("Mixtral", "mixtral", 3), ("OLMoE", "olmoe", 32),
                             ("OLMoE", "olmoe", 24), ("Qwen3", "qwen3", 64), ("Qwen3", "qwen3", 48)):
        run("H100", model, cap, t5[f"{name}_B{cap}"])
    return rows


def all_runs() -> None:
    rows = all_runs_rows()
    marker = {"A40": "o", "A100": "s", "H100": "^", "RTX 4090": "D"}
    fig, ax = plt.subplots(figsize=(TEXT_IN, 1.9))
    ylim = (-10, 16)
    ax.axhspan(0, ylim[1], color=GREEN, alpha=0.07, lw=0, zorder=0)
    ax.text(58, ylim[1] - 0.8, f"{NAME} faster", ha="right", va="top", fontsize=6.5, color=INK)
    ax.text(0.95, ylim[0] + 0.4, "decayed LFU faster", ha="left", va="bottom", fontsize=6.5, color=GRAY)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    n_win = n_loss = 0
    seen = {}
    for r in rows:
        key = round(r["copy"], 2)
        seen[key] = seen.get(key, 0) + 1
        x = r["copy"] * (1 + 0.013 * (seen[key] // 2) * (-1) ** seen[key])  # a small spread for equal copy times
        color = GREEN if r["lo"] > 0 else VERMILLION if r["hi"] < 0 else GRAY
        n_win += r["lo"] > 0
        n_loss += r["hi"] < 0
        mk = ("P" if r["two"] else "v") if r["graphs"] else marker[r["gpu"]]
        face = "white" if r["prefetch"] else color
        ax.errorbar(x, r["gain"], yerr=[[r["gain"] - r["lo"]], [r["hi"] - r["gain"]]], fmt=mk,
                    color=color, mfc=face, mec=color, ms=4.0 if r["graphs"] else 3.8, mew=0.8, elinewidth=0.8,
                    capthick=0.8, zorder=3)
    notes = ((0.158, -8.7, "H100, OLMoE and Qwen3\nat native copy cost", "left"),
             (0.405, 4.6, "Qwen3, A40", "right"),
             (0.53, 8.7, "desktop", "center"),
             (4.12, 14.6, "desktop, experts on NVMe", "center"),
             (2.85, 11.6, "H100, experts on NVMe", "right"),
             (46.26, 2.7, "A100 Mixtral,\nPCIe gen3", "center"),
             (1.25, 7.6, "H100, OLMoE and Qwen3\ncopies repeated 2, 4 times", "left"),
             (0.235, 8.6, "H100, CUDA graphs,\ntwo workers", "center"),
             (16.5, 9.4, "H100 Mixtral, copies\nrepeated 1 to 7 times", "center"))
    for x, yv, s, ha in notes:
        ax.text(x, yv, s, fontsize=5.8, color=INK, ha=ha, va="center", linespacing=0.95)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 62)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20, 50]))
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(*ylim)
    ax.set_yticks([-10, -5, 0, 5, 10, 15])
    ax.set_xlabel("copy time per expert (ms), each platform's own or repeated $K$ times on the H100")
    ax.set_ylabel("gain over decayed LFU (%)")
    gh = [Line2D([], [], marker=m, ls="none", color=GRAY, mfc=GRAY, ms=3.8, label=g) for g, m in marker.items()]
    gh += [Line2D([], [], marker="v", ls="none", color=GRAY, mfc=GRAY, ms=4.0, label="H100, CUDA graphs"),
           Line2D([], [], marker="P", ls="none", color=GRAY, mfc=GRAY, ms=4.0, label="... and two workers")]
    ch = [Line2D([], [], marker="o", ls="none", color=c, ms=3.8, label=l)
          for c, l in ((GREEN, "interval above zero"), (GRAY, "inconclusive"), (VERMILLION, "interval below zero"))]
    hh = [Line2D([], [], marker="o", ls="none", color=GRAY, mfc="white", ms=3.8, label="both policies prefetch")]
    ax.legend(handles=gh + ch + hh, loc="lower right", ncol=3, fontsize=6, bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout(pad=0.3)
    save(fig, OUT / "fig_all_runs.pdf")
    print(f"all runs: {len(rows)} comparisons, {n_win} with interval above zero, {n_loss} below zero")


def zero_crossing(grid: np.ndarray, gain: np.ndarray) -> float | None:
    """Copy time where the model's gain first turns from negative to positive (log-linear interpolation)."""
    s = np.sign(gain)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if not len(idx):
        return None
    i = idx[0]
    f = gain[i] / (gain[i] - gain[i + 1])
    return float(np.exp(np.log(grid[i]) + f * (np.log(grid[i + 1]) - np.log(grid[i]))))


def h100_sweep() -> None:
    dev_costs = json.loads(sweep.DEV_COSTS.read_text())
    panels = (("olmoe", 32, (1, 2, 4, 16, 32), "OLMoE, $B\\!=\\!32$"), ("qwen3", 64, (1, 4, 16), "Qwen3, $B\\!=\\!64$"),
              ("mixtral", 4, (1,), "Mixtral, $B\\!=\\!4$"))
    ylim = sweep.YLIM
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 1.95), sharey=True)
    grid = np.geomspace(0.12, 12.0, 600)
    for ax, (model, cap, ks, title) in zip(axes, panels):
        pts, copies, wall = sweep.points(model, cap, ks)
        costs = dev_costs[model][str(cap)]
        curves = {arm: 100 * sweep.curve(costs, copies, wall, arm, grid) for arm, *_ in sweep.ARMS}
        be_split = zero_crossing(grid, curves["splitphase"])
        be_sync = zero_crossing(grid, curves["global_token_gbt"])
        # the verdict: where the break-even model says SplitEvict beats decayed LFU
        if be_split is not None:
            ax.axvspan(be_split, grid[-1], color=GREEN, alpha=0.07, lw=0, zorder=0)
            ax.text(grid[-1] / 1.08, ylim[0] + 1.5, f"{NAME}\nfaster", ha="right", va="bottom", fontsize=6, color=INK,
                    linespacing=0.95)
        native = pts["splitphase"][0][0]
        ax.axvline(native, color=GRAY, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
        ax.text(native * 1.07, ylim[1] - 1.2, "H100\nnative", color=GRAY, fontsize=6, va="top", ha="left")
        for arm, label, color, marker in sweep.ARMS:
            ax.plot(grid, curves[arm], color=color, lw=1.1, alpha=0.85, zorder=2)
            if pts[arm]:
                x, g, lo, hi = (np.array(v) for v in zip(*pts[arm]))
                shown = (100 * g >= ylim[0]) & (100 * g <= ylim[1])
                ax.errorbar(x[shown], 100 * g[shown], yerr=[100 * (g - lo)[shown], 100 * (hi - g)[shown]], fmt=marker,
                            color=color, ms=3.8, mec="white", mew=0.5, elinewidth=0.8, capthick=0.8, label=label,
                            zorder=3)
                for xi, gi in zip(x[~shown], g[~shown]):  # off-scale points: an arrow at the edge with the value
                    ax.annotate(f"{100 * gi:.0f}%".replace("-", "−"), xy=(xi, ylim[0] + 0.5),
                                xytext=(xi, ylim[0] + 7), color=color, ha="center", fontsize=6,
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, mutation_scale=6))
        ax.axhline(0, color=INK, lw=0.6, zorder=1.5)
        # break-even points on the zero line (diamonds), SplitEvict and the synchronous ranker
        for be, color in ((be_split, GREEN), (be_sync, VERMILLION)):
            if be is not None:
                ax.plot([be], [0], marker="D", ms=4.2, color=color, mec="white", mew=0.6, zorder=4, ls="none")
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])
        ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
        ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(*ylim)
        ax.set_yticks([-30, -20, -10, 0, 10, 20])
        ax.set_xlabel("copy plus issue time per expert (ms)")
        ax.set_title(title)
    axes[0].set_ylabel("gain over decayed LFU (%)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    extra = [Line2D([], [], color=GRAY, lw=1.1),
             Line2D([], [], marker="D", ms=4.2, color=GRAY, mec="white", mew=0.6, ls="none")]
    axes[2].legend(handles + extra, labels + ["break-even model", "break-even point"], loc="upper left")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, OUT / "fig_h100_sweep.pdf")


if __name__ == "__main__":
    use()
    bottleneck()
    stale()
    regret()
    all_runs()
    h100_sweep()
    print("wrote fig_bottleneck, fig_stale, fig_regret_by_offset, fig_all_runs, fig_h100_sweep (.pdf and .png)")
