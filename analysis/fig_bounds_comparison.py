"""
analysis/fig_bounds_comparison.py
----------------------------------
Single strip plot: feasible set interval vs baseline interval for DD,
one row per scenario sorted by true metric.
Consistent scenarios only (baseline only valid there).

Each row shows:
  - Grey bar   = baseline interval
  - Teal bar   = feasible set interval  (always narrower)
  - Black dot  = true DD
  - Dark diamond = FS mean

Run:
    python analysis/fig_bounds_comparison.py
    python analysis/fig_bounds_comparison.py \\
        --results results/experiment_results.csv \\
        --figures figures/paper
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines

FS_COLOR   = "#1D9E75"
BL_COLOR   = "#888780"
GT_COLOR   = "#2C2C2A"
MEAN_COLOR = "#085041"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, 
    "figure.dpi": 130, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def plot_bounds_comparison(df: pd.DataFrame, save_path=None):
    cons = (df[df.label == "consistent"]
            .dropna(subset=["DD_lo", "DD_bl_lo"])
            .sort_values("DD_gt")
            .reset_index(drop=True))

    n   = len(cons)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.32)))

    for i, row in cons.iterrows():
        y = float(i)

        # Baseline (grey, tall, behind)
        ax.barh(y, row.DD_bl_hi - row.DD_bl_lo, left=row.DD_bl_lo,
                height=0.65, color=BL_COLOR, alpha=0.28, zorder=2)
        ax.plot([row.DD_bl_lo, row.DD_bl_lo], [y-0.32, y+0.32],
                color=BL_COLOR, lw=1.2, alpha=0.55, zorder=3)
        ax.plot([row.DD_bl_hi, row.DD_bl_hi], [y-0.32, y+0.32],
                color=BL_COLOR, lw=1.2, alpha=0.55, zorder=3)

        # Feasible set (teal, narrower, in front)
        ax.barh(y, row.DD_hi - row.DD_lo, left=row.DD_lo,
                height=0.38, color=FS_COLOR, alpha=0.80, zorder=4)

        # FS mean (dark diamond)
        ax.scatter(row.DD_mean, y, color=MEAN_COLOR,
                   s=45, marker="D", zorder=6, linewidths=0)

        # True DD (black circle)
        ax.scatter(row.DD_gt, y, color=GT_COLOR,
                   s=55, zorder=7, linewidths=0)

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.25)
    ax.set_yticks([])
    ax.set_xlabel("Demographic Disparity (DD)", fontsize=11)
    ax.set_ylabel(f"Scenario  (n={n}, sorted by true DD ↑)", fontsize=10)
    ax.set_title(
        "Feasible set bounds vs Fréchet baseline bounds\n"
        "Consistent scenarios only  ·  Feasible set is always tighter",
        fontsize=12, fontweight="bold")
    ax.spines["left"].set_visible(False)

    # Stats box
    fs_med = cons["DD_width"].median()
    bl_med = cons["DD_bl_width"].median()
    red    = (1 - fs_med / bl_med) * 100
    fs_cov = cons["DD_covered"].mean() * 100
    bl_cov = cons["DD_bl_covered"].mean() * 100
    ax.text(0.015, 0.985,
            f"n = {n}  (consistent scenarios)\n\n"
            f"Median width\n"
            f"  Feasible set   {fs_med:.3f}\n"
            f"  Baseline       {bl_med:.3f}\n"
            f"  Reduction      {red:.0f}%\n\n"
            f"Coverage of true DD\n"
            f"  Feasible set   {fs_cov:.0f}%\n"
            f"  Baseline       {bl_cov:.0f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec="#bbb", alpha=0.97))

    ax.legend(handles=[
        mpatches.Patch(facecolor=BL_COLOR, alpha=0.4,  label="Baseline interval"),
        mpatches.Patch(facecolor=FS_COLOR, alpha=0.85, label="Feasible set interval"),
        mlines.Line2D([0],[0], marker="o", color=GT_COLOR,   ms=7, lw=0, label="True DD"),
        mlines.Line2D([0],[0], marker="D", color=MEAN_COLOR, ms=6, lw=0, label="FS mean"),
    ], fontsize=9, loc="lower right", framealpha=0.97, edgecolor="#ccc")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_results.csv")
    parser.add_argument("--figures", default="figures/paper")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    plot_bounds_comparison(
        df, save_path=os.path.join(args.figures, "fig_bounds_comparison.png"))
