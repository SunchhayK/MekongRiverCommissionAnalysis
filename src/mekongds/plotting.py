"""Shared matplotlib style + small helpers for all notebooks."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

STAGE_COLORS = {
    "flood": "#b2182b",
    "alarm": "#ef8a62",
    "normal": "#2166ac",
    "obs": "#222222",
    "fc": "#d95f02",
    "band": "#9ecae1",
}

PALETTE = [
    "#2166ac", "#d95f02", "#1b7837", "#762a83", "#e08214",
    "#4393c3", "#c51b7d", "#5aae61", "#9970ab", "#b35806",
    "#92c5de", "#de77ae",
]


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "font.family": "sans-serif",
    })


def add_threshold_lines(ax, alarm, flood, label=True):
    ax.axhline(alarm, color=STAGE_COLORS["alarm"], lw=1.0, ls="--")
    ax.axhline(flood, color=STAGE_COLORS["flood"], lw=1.0, ls="--")
    if label:
        ax.text(
            ax.get_xlim()[1], alarm, " alarm", color=STAGE_COLORS["alarm"],
            va="center", fontsize=8,
        )
        ax.text(
            ax.get_xlim()[1], flood, " flood", color=STAGE_COLORS["flood"],
            va="center", fontsize=8,
        )


def savefig(fig, name):
    from .data import figures_dir

    fig.savefig(figures_dir() / f"{name}.png", bbox_inches="tight")
