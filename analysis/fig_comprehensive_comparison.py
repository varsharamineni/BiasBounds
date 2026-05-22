"""
analysis/fig_comprehensive_comparison.py
-----------------------------------------
2x2 figure, each panel one scenario. Each panel shows:

  Top half:  horizontal bars — Kallus bounds (grey), tight Frechet (blue),
             feasible set interval (teal), all with midpoints/mean marked
  Bottom half: KDE histogram of feasible set distribution,
               with true DD, FS mean, Kallus midpoint, tight midpoint marked

Run:
    cd fairness_experiments
    python analysis/fig_comprehensive_comparison.py
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

KALLUS_COLOR = "#888780"
TIGHT_COLOR  = "#534AB7"
FS_COLOR     = "#1D9E75"
GT_COLOR     = "#D85A30"
MEAN_COLOR   = "#085041"
TIGHT_MID    = "#7F77DD"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 150,
    "savefig.facecolor": "white",
})


def _tight_bounds(p_az, p_xz, clf):
    pz = {z: p_az[(0,z)] + p_az[(1,z)] for z in [0,1]}
    results = {}
    for a in [0,1]:
        p_a = sum(p_az[(a,z)] for z in [0,1])
        lo_a, hi_a = 0.0, 0.0
        for z in [0,1]:
            paz = p_az[(a,z)] / pz[z]
            pxz = p_xz[(1,z)] / pz[z]
            p1_lo = max(0, pxz + paz - 1) / paz
            p1_hi = min(pxz, paz) / paz
            py0 = clf[(0,z)][1]; py1 = clf[(1,z)][1]
            c = py1 - py0
            py_lo = py0 + c * (p1_lo if c >= 0 else p1_hi)
            py_hi = py0 + c * (p1_hi if c >= 0 else p1_lo)
            w = p_az[(a,z)] / p_a
            lo_a += py_lo * w; hi_a += py_hi * w
        results[a] = (lo_a, hi_a)
    return results[1][0] - results[0][1], results[1][1] - results[0][0]


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
        t_lo, t_hi = _tight_bounds(sc.p_az, sc.p_xz, sc.classifier)
        d = dists["DD"].values
        out.append(dict(
            vals     = d,
            true     = gt.DD,
            fs_mean  = d.mean(),
            fs_lo    = d.min(), fs_hi = d.max(),
            bl_lo    = bl["DD_lo"], bl_hi = bl["DD_hi"],
            t_lo     = t_lo, t_hi = t_hi,
        ))
    return out


def plot(save_path=None, num_grid=80):
    sc_indices = [20, 4, 12, 22]

    print("  Running feasible set solver...")
    data = _get_data(sc_indices, num_grid)
    print("  Done.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    for ax, d in zip(axes.flat, data):
        vals    = d["vals"]
        true    = d["true"]
        fs_mean = d["fs_mean"]
        fs_lo   = d["fs_lo"];  fs_hi  = d["fs_hi"]
        bl_lo   = d["bl_lo"];  bl_hi  = d["bl_hi"]
        t_lo    = d["t_lo"];   t_hi   = d["t_hi"]
        bl_mid  = (bl_lo + bl_hi) / 2
        t_mid   = (t_lo + t_hi) / 2

        # Split axis: top = interval bars, bottom = histogram
        # Use inset axes for the interval bars at the top
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # ── Bottom: KDE histogram ──────────────────────────────────────────
        # Set x range to encompass all three intervals
        xmin = min(bl_lo, vals.min()) - 0.05
        xmax = max(bl_hi, vals.max()) + 0.05

        sns.histplot(vals, ax=ax, color=FS_COLOR, alpha=0.55,
                     stat="density", bins=35, kde=True,
                     line_kws={"lw": 2, "color": FS_COLOR}, zorder=2)

        # True DD
        ax.axvline(true,    color=GT_COLOR,   lw=2.2, zorder=5)
        # FS mean
        ax.axvline(fs_mean, color=MEAN_COLOR, lw=2,   zorder=4, ls="-")
        # Kallus midpoint
        ax.axvline(bl_mid,  color=KALLUS_COLOR, lw=1.8, ls="--", zorder=3)
        # Tight midpoint (≈ FS mean, shown for completeness)
        ax.axvline(t_mid,   color=TIGHT_COLOR,  lw=1.8, ls=":",  zorder=3)

        ax.set_xlabel("DD", fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xmin, xmax)

        # ── Top inset: interval bars ───────────────────────────────────────
        ax_top = inset_axes(ax, width="100%", height="30%",
                            loc="upper center",
                            bbox_to_anchor=ax.get_position(),
                            bbox_transform=fig.transFigure,
                            borderpad=0)
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_ylim(-0.5, 2.5)
        ax_top.axis("off")

        bar_specs = [
            (2.0, bl_lo, bl_hi, bl_mid, KALLUS_COLOR, 0.30, "Kallus"),
            (1.0, t_lo,  t_hi,  t_mid,  TIGHT_COLOR,  0.35, "Tight Fréchet"),
            (0.0, fs_lo, fs_hi, fs_mean, FS_COLOR,     0.40, "Feasible set"),
        ]
        for y, lo, hi, mid, color, alpha, lbl in bar_specs:
            ax_top.barh(y, hi - lo, left=lo, height=0.45,
                        color=color, alpha=alpha)
            ax_top.axvline(lo,  ymin=(y-0.22)/3, ymax=(y+0.22)/3,
                           color=color, lw=1.5, alpha=0.8)
            ax_top.axvline(hi,  ymin=(y-0.22)/3, ymax=(y+0.22)/3,
                           color=color, lw=1.5, alpha=0.8)
            # Midpoint/mean marker
            ax_top.scatter(mid, y, color=color, s=50, zorder=5,
                           marker="|", linewidths=2.5)
            # Label on left
            ax_top.text(xmin - 0.01*(xmax-xmin), y, lbl,
                        ha="right", va="center", fontsize=7.5,
                        color=color, fontweight="bold")

        # True value line through both panels
        ax_top.axvline(true, color=GT_COLOR, lw=2.2, alpha=0.9)

    # Shared legend
    fig.legend(handles=[
        mpatches.Patch(facecolor=KALLUS_COLOR, alpha=0.4, label="Kallus bounds"),
        mpatches.Patch(facecolor=TIGHT_COLOR,  alpha=0.45, label="Tight Fréchet bounds"),
        mpatches.Patch(facecolor=FS_COLOR,     alpha=0.6, label="Feasible set interval + distribution"),
        mlines.Line2D([0],[0], color=GT_COLOR,     lw=2.2, label="True DD"),
        mlines.Line2D([0],[0], color=MEAN_COLOR,   lw=2,   label="FS mean"),
        mlines.Line2D([0],[0], color=KALLUS_COLOR, lw=1.8, ls="--", label="Kallus midpoint"),
        mlines.Line2D([0],[0], color=TIGHT_COLOR,  lw=1.8, ls=":",  label="Tight midpoint"),
    ], fontsize=8.5, loc="lower center", ncol=4, frameon=False,
       bbox_to_anchor=(0.5, -0.03))

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
    plot(save_path=os.path.join(args.figures, "fig_comprehensive_comparison.png"),
         num_grid=args.num_grid)
