"""One visual system for every figure in the repo (light theme, CVD-checked palette)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import FIG_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
GRAY_BAR = "#b9b8b0"

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, NEUTRAL, RED])
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def save(fig, name: str) -> str:
    # Tests point this at a temp dir so they never overwrite the published figures.
    fig_dir = Path(os.environ.get("CATCAFE_FIG_DIR", FIG_DIR))
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{name}.png"


apply_theme()
