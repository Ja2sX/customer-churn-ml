"""Shared figure styling for the report and the training artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt

NAVY = "#1F4E79"
TERRACOTTA = "#C65D3B"
TEAL = "#1F7A6B"
SLATE = "#52606D"
GRID = "#E6E8EB"
INK = "#1C2833"

MODEL_COLORS = {
    "logistic_regression": NAVY,
    "random_forest": TEAL,
    "xgboost": TERRACOTTA,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def style_axes(ax, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
