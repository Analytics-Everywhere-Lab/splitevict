"""Shared style for the paper's data figures (matplotlib), matched to Figure 1.

Every figure is drawn at the size it prints (no scaling in LaTeX), so the fonts print at the sizes set here:
Arial for text (as in Figure 1) at 6.5-7.5 pt and Computer Modern for maths symbols ($B$, $E$, $d$), as in the
paper's own text; both embedded as TrueType (no Type 3 fonts). White
background, flat marks, left and bottom axes only, a faint horizontal grid behind the data. Colours are
colour-blind safe (Okabe-Ito) and keep one meaning across figures: green = split-phase eviction, vermillion =
synchronous ranker, as in Figure 1.

    from figstyle import use, save, COLUMN_IN, TEXT_IN, GREEN, VERMILLION, BLUE
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMN_IN = 3.25  # MLSys \columnwidth (in)
TEXT_IN = 6.75  # MLSys \textwidth (in)

GREEN = "#009E73"  # split-phase eviction
VERMILLION = "#D55E00"  # synchronous ranker / a worse outcome
BLUE = "#0072B2"  # a baseline or a second series
SKY = "#56B4E9"  # a lighter partner of BLUE
ORANGE = "#E69F00"
INK = "#1A1A1A"
GRAY = "#7A7A7A"
GRID = "#E8E8E8"


def use() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,  # symbols in the paper's maths font
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titleweight": "normal", "axes.titlepad": 3,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.2, "lines.markersize": 3.6, "errorbar.capsize": 1.6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
    })


def save(fig, path: Path) -> None:
    """Vector PDF at the figure's own size, plus a 300 dpi PNG preview."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
