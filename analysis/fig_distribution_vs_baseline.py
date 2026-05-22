from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

FS_COLOR   = "#1D9E75"
BL_COLOR   = "#AAAAAA"
GT_COLOR   = "#D85A30"
MEAN_COLOR = "#085041"


def _get_data(sc_indices, num_grid=80):
    from core.scenarios.scenario_a import generate_scenario_a
    from core.joint_solver import solve_feasible_set
    from core.metrics import distributions_over_feasible_set, compute_all_metrics
    from core.baseline import get_baseline_bounds

    scs = generate_scenario_a(n_ground_truths=30, alphas=[0.0], seed=42)
    out = []
    for idx in sc_indices:
        sc     = scs[idx]
        joints = solve_feasible_set(sc.p_az, sc.p_xz, num=num_grid)
        dists  = distributions_over_feasible_set(joints, sc.classifier)
        j_str  = {f"p{x}{z}{a}": v for (x,z,a),v in sc.joint_gt.items()}
        gt     = compute_all_metrics(j_str, sc.classifier)
        bl     = get_baseline_bounds(sc.p_az, sc.p_xz, sc.classifier)
        d      = dists["DD"].values
        out.append(dict(
            vals  = d,
            true  = gt.DD,
            mean  = d.mean(),
            bl_lo = bl["DD_lo"],
            bl_hi = bl["DD_hi"],
        ))
    return out


def plot(save_path=None, num_grid=80):
    sc_indices = [20, 4, 12, 22]

    print("  Running feasible set solver...")
    data = _get_data(sc_indices, num_grid)
    print("  Done.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    for i, (ax, d) in enumerate(zip(axes.flat, data)):
        vals  = d["vals"]
        true  = d["true"]
        mean  = d["mean"]
        bl_lo = d["bl_lo"]
        bl_hi = d["bl_hi"]

        # Baseline span
        ax.axvspan(bl_lo, bl_hi, alpha=0.12, color=BL_COLOR, zorder=0)
        ax.axvline(bl_lo, color=BL_COLOR, lw=1.2, ls="--", alpha=0.7, zorder=1)
        ax.axvline(bl_hi, color=BL_COLOR, lw=1.2, ls="--", alpha=0.7, zorder=1)

        # KDE histogram
        sns.histplot(vals, ax=ax, color=FS_COLOR, alpha=0.65,
                     stat="density", bins=35, kde=True,
                     line_kws={"lw": 2.2, "color": FS_COLOR}, zorder=2)

        # FS mean and True DD lines
        ax.axvline(mean, color=MEAN_COLOR, lw=2,   zorder=4)
        ax.axvline(true, color=GT_COLOR,   lw=2.2, zorder=5)

        ax.set_xlabel("DD", fontsize=11)
        ax.set_ylabel("Density" if i % 2 == 0 else "", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Baseline width label below the plot
        ax.text(0.5, -0.18,
                f"Baseline  [{bl_lo:+.2f}, {bl_hi:+.2f}]",
                transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#777")

    # Shared legend at bottom
    fig.legend(handles=[
        mpatches.Patch(facecolor=FS_COLOR, alpha=0.7,
                       label="Feasible set distribution"),
        mpatches.Patch(facecolor=BL_COLOR, alpha=0.3,
                       label="Baseline interval"),
        mlines.Line2D([0],[0], color=MEAN_COLOR, lw=2,   label="FS mean"),
        mlines.Line2D([0],[0], color=GT_COLOR,   lw=2.2, label="True DD"),
    ], fontsize=9.5, loc="lower center", ncol=4,
       frameon=False, bbox_to_anchor=(0.5, -0.02))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures",  default="figures/paper")
    parser.add_argument("--num_grid", type=int, default=80)
    args = parser.parse_args()
    plot(save_path=os.path.join(args.figures, "fig_distribution_vs_baseline.png"),
         num_grid=args.num_grid)
