"""Shared matplotlib style: validated categorical palette, thin marks, recessive chrome."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
DE_EMPHASIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6,
    "legend.frameon": False, "legend.fontsize": 9, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hbar(ax, labels, values, highlight=None, fmt="{:.3f}"):
    """Ranked horizontal bars, one series colour, optional emphasis on one bar."""
    colors = [SERIES[0] if (highlight is None or lab == highlight) else DE_EMPHASIS for lab in labels]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  " + fmt.format(v), va="center",
                fontsize=9, color=INK2)
    ax.set_xlim(0, max(values) * 1.15)
    return bars
