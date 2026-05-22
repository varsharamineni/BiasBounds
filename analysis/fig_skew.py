"""
analysis/fig_skew_examples.py
-------------------------------
Two-panel figure from existing results CSV.
Finds one low-skew and one high-skew scenario, reconstructs the
approximate distribution using a truncated normal, and plots:
  - KDE histogram of the feasible DI distribution
  - True DI (orange), FS mean (dark green), Interval midpoint (purple dashed)

Also prints exact caption values.

Run:
    cd fairness_experiments
    python analysis/fig_skew_examples.py
    python analysis/fig_skew_examples.py \\
        --results results/experiment_results_big.csv \\
        --figures figures
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import truncnorm

GT_COLOR   = "#D85A30"
MEAN_COLOR = "#085041"
MID_COLOR  = "#534AB7"
FS_COLOR   = "steelblue"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _approx_dist(mean, std, lo, hi, n=2000):
    if not all(np.isfinite([mean, std, lo, hi])) or std < 1e-9:
        return None
    a = (lo - mean) / std
    b = (hi - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std,
                         size=n, random_state=42)


def find_examples(df):
    import numpy as np
    df = df.copy()
    df["DI_mid"]      = (df["DI_lo"] + df["DI_hi"]) / 2
    df["err_mean"]    = (df["DI_mean"] - df["DI_gt"]).abs()
    df["err_mid"]     = (df["DI_mid"]  - df["DI_gt"]).abs()
    df["improvement"] = df["err_mid"] - df["err_mean"]
    df["skew_abs"]    = df["DI_skew"].abs()

    low = df[
        (df["skew_abs"] < 0.1) &
        df["DI_std"].notna() &
        (df["DI_std"] > 0.01) &
        (df["DI_width"] > 0.1)
    ].nsmallest(5, "err_mean").iloc[0]

    # Want high skew, moderate width, midpoint clearly visible in the centre
    df["DI_mid"]     = (df["DI_lo"] + df["DI_hi"]) / 2
    df["mid_frac"]   = (df["DI_mid"] - df["DI_lo"]) / df["DI_width"].replace(0, np.nan)
    high = df[
        (df["DI_skew"] > 0.8) &
        (df["DI_skew"] < 4) &
        (df["DI_width"] < 2.2) &
        (df["DI_width"] > 0.5) &
        (df["DI_lo"] > 0.1) &
        (df["mid_frac"] > 0.3) &
        (df["mid_frac"] < 0.7) &
        df["DI_std"].notna() &
        (df["improvement"] > 0.2)
    ].nlargest(1, "improvement").iloc[0]

    return low, high


def plot(results_path, save_path=None):
    df = pd.read_csv(results_path)
    print(f"  Loaded {len(df):,} scenarios")

    low, high = find_examples(df)

    print()
    print("=" * 60)
    print("  CAPTION VALUES")
    print("=" * 60)
    for name, row in [("LOW SKEW", low), ("HIGH SKEW", high)]:
        mid = (row["DI_lo"] + row["DI_hi"]) / 2
        print(f"\n  {name}:")
        print(f"    skewness = {row['DI_skew']:.3f}")
        print(f"    true DI  = {row['DI_gt']:.3f}")
        print(f"    FS mean  = {row['DI_mean']:.3f}   |err| = {abs(row['DI_mean']-row['DI_gt']):.3f}")
        print(f"    midpoint = {mid:.3f}   |err| = {abs(mid-row['DI_gt']):.3f}")
        print(f"    interval = [{row['DI_lo']:.3f}, {row['DI_hi']:.3f}]")
    print()

    fig, axes = plt.subplots(2, 1, figsize=(5, 11))
    plt.subplots_adjust(wspace=0.38)

    for ax, (row, title) in zip(axes, [(low, "Low skew"), (high, "High skew")]):
        mid  = (row["DI_lo"] + row["DI_hi"]) / 2
        true = row["DI_gt"]
        mean = row["DI_mean"]
        sk   = row["DI_skew"]
        vals = _approx_dist(mean, row["DI_std"], row["DI_lo"], row["DI_hi"])

        if vals is not None:
            sns.histplot(vals, ax=ax, color=FS_COLOR, alpha=0.55,
                         stat="probability", bins=30, 
                         line_kws={"lw": 2, "color": FS_COLOR})

        ax.axvline(true, color=GT_COLOR,   lw=2.3, zorder=5,
                   label=f"True DI = {true:.3f}")
        ax.axvline(mean, color=MEAN_COLOR, lw=2,   zorder=4,
                   label=f"FS mean = {mean:.3f}  (|err| = {abs(mean-true):.3f})")
        ax.axvline(mid,  color=MID_COLOR,  lw=2,   zorder=4, ls="--",
                   label=f"Midpoint = {mid:.3f}  (|err| = {abs(mid-true):.3f})")

        ax.set_xlabel("Disparate Impact (DI)", fontsize=11)
        ax.set_ylabel("Probability" if ax is axes[0] else "", fontsize=11)
        ax.set_title(f"{title} ",
                     fontsize=11, fontweight="bold")

    fig.legend(handles=[
        plt.Rectangle((0,0),1,1, fc=FS_COLOR, alpha=0.6,
                       label="Feasible set distribution"),
        mlines.Line2D([0],[0], color=GT_COLOR,   lw=2.3, label="True DI"),
        mlines.Line2D([0],[0], color=MEAN_COLOR, lw=2,   label="FS mean"),
        mlines.Line2D([0],[0], color=MID_COLOR,  lw=2, ls="--",
                      label="Interval midpoint"),
    ], fontsize=10, loc="lower center", ncol=4, frameon=False,
       bbox_to_anchor=(0.5, -0.05))
    plt.subplots_adjust(bottom=0.03)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_results_big.csv")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    plot(args.results,
         save_path=os.path.join(args.figures, "fig_skew_examples.pdf"))