"""The paper's appendix Figures A.1 and B.1, written to figures/.

- fig_staleness.pdf (Figure A.1): the learned saving kept when the ranking is refreshed every d tokens, with and
  without rescoring the experts used since the refresh. It marks d = 1 as what SplitEvict does (a refresh after every
  router call), and notes that without rescoring the saving is nearly gone by d = 16.
- fig_crossmodel.pdf (Figure B.1): the saving kept by a stale ranking with the current token and with a stale token
  on Mixtral, OLMoE and Qwen3. It points to Mixtral's stale-token bars, where a stale token removes the saving.

    python analysis/figures_appendix.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, INK, ORANGE, SKY, save, use  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
SUPP_IN = 6.5  # base width (in) of these figures; changing it changes their printed text size


def staleness() -> None:
    rows = json.load(open(ROOT / "results/simulation/refresh_summary.json"))
    get = {(r["arm"], r["B"]): 100 * r["retention"] for r in rows}
    fig, ax = plt.subplots(figsize=(0.7 * SUPP_IN, 2.3))
    d = [1, 2, 4, 8, 16]
    for B, c in ((32, BLUE), (24, ORANGE)):
        touch = [get[("lag_req", B)]] + [get[(f"lag_req_d{k}_touch", B)] for k in d[1:]]
        none = [get[("lag_req", B)]] + [get[(f"lag_req_d{k}", B)] for k in d[1:]]
        ax.plot(d, touch, "-o", color=c, mec="white", mew=0.5)
        ax.plot(d, none, "--", color=c, lw=1.0)
    top = max(get[("lag_req", 32)], get[("lag_req", 24)])
    # the verdict: SplitEvict sits at d = 1, and every extra token between refreshes loses saving
    ax.annotate("SplitEvict:\nrefresh after every router call", xy=(1.0, top), xytext=(1.55, 101),
                ha="left", va="top", fontsize=6, color=INK, linespacing=0.95,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=3))
    ax.text(16, 22, "not rescored:\nsaving nearly\ngone by $d\\!=\\!16$", ha="right", va="bottom", fontsize=6,
            color=INK, linespacing=0.95)
    ax.set_xscale("log", base=2)
    ax.set_xticks(d)
    ax.set_xticklabels([str(k) for k in d])
    ax.minorticks_off()
    ax.set_xlim(0.85, 19)
    ax.set_ylim(0, 102)
    ax.set_xlabel("ranking refreshed every $d$ tokens")
    ax.set_ylabel("learned saving kept (%)")
    handles = [Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color=ORANGE, lw=1.2),
               Line2D([], [], color=INK, lw=1.2, marker="o", ms=3.2), Line2D([], [], color=INK, lw=1.0, ls="--")]
    ax.legend(handles, ["$B\\!=\\!32$", "$B\\!=\\!24$", "experts used since the refresh rescored", "not rescored"],
              loc="lower left")
    fig.tight_layout(pad=0.3)
    save(fig, OUT / "fig_staleness.pdf")


def crossmodel() -> None:
    rows = json.load(open(ROOT / "results/simulation/crossmodel_summary.json"))
    staleness = {(r["arm"], r["B"]): r["retention"] for r in json.load(open(ROOT / "results/simulation/staleness_summary.json"))}
    groups = [("Mixtral\n$E\\!=\\!8$, $B\\!=\\!3$", "mixtral_8x7b", 3), ("Mixtral\n$E\\!=\\!8$, $B\\!=\\!4$", "mixtral_8x7b", 4),
              ("OLMoE\n$E\\!=\\!64$, $B\\!=\\!24$", None, 24), ("OLMoE\n$E\\!=\\!64$, $B\\!=\\!32$", None, 32),
              ("Qwen3\n$E\\!=\\!128$, $B\\!=\\!48$", "qwen3_30b_a3b", 48), ("Qwen3\n$E\\!=\\!128$, $B\\!=\\!64$", "qwen3_30b_a3b", 64)]
    req, tok = [], []
    for _, m, B in groups:
        if m is None:
            req.append(100 * staleness[("lag_req", B)])
            tok.append(100 * staleness[("lag_tok", B)])
        else:
            get = {(r["arm"]): r["retention"] for r in rows if r["model"] == m and r["B"] == B}
            req.append(100 * get["lag_req"])
            tok.append(100 * get["lag_tok"])
    fig, ax = plt.subplots(figsize=(0.85 * SUPP_IN, 2.4))
    x = range(len(groups))
    for vals, off, c in ((req, -0.19, BLUE), (tok, 0.19, SKY)):
        ax.bar([i + off for i in x], vals, 0.36, color=c, lw=0)
        for i, v in zip(x, vals):  # value labels: above a positive bar, below a negative one
            ax.text(i + off, v + (2 if v >= 0 else -2), f"{v:.0f}".replace("-", "−"), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6, color=INK)
    # the verdict: only Mixtral, with 8 experts per layer, loses the saving when the token is stale too
    ax.text(1.5, -13, "a stale token removes the saving", ha="left", va="center", fontsize=6, color=INK)
    for target in ((1.37, min(tok[1], 0) - 0.5), (0.40, tok[0] / 2)):
        ax.annotate("", xy=target, xytext=(1.47, -13),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-25, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("learned saving kept (%)")
    ax.legend([Patch(color=BLUE), Patch(color=SKY)],
              ["stale ranking (previous requests, current token)",
               "stale ranking, stale token (previous requests and token)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0)
    fig.tight_layout(pad=0.3)
    save(fig, OUT / "fig_crossmodel.pdf")


if __name__ == "__main__":
    use()
    staleness()
    crossmodel()
    print("wrote fig_staleness.pdf, fig_crossmodel.pdf")
