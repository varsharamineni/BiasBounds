"""
analysis/fig_mean_vs_midpoint.py
---------------------------------
Two-part figure:

  Left:  Scatter of DI interval midpoint error vs FS mean error across all
         scenarios. Points below diagonal = FS mean is better.
         Coloured by skewness of the feasible distribution.

  Right: 4 example panels (2x2) — high-skew DI distributions showing
         that the FS mean (weighted centroid of distribution) is a better
         estimate than the naive interval midpoint.

Run:
    cd fairness_experiments
    python analysis/fig_mean_vs_midpoint.py
"""

from __future__ import annotations
import sys, os, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

GT_COLOR   = "#D85A30"
MEAN_COLOR = "#085041"
MID_COLOR  = "#534AB7"
FS_COLOR   = "#1D9E75"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 150,
    "savefig.facecolor": "white",
})


def _build_data(num_grid=80):
    from core.scenarios.scenario_a import generate_scenario_a
    from core.joint_solver import solve_feasible_set
    from core.metrics import distributions_over_feasible_set, compute_all_metrics
    from core.baseline import get_baseline_bounds
    import scipy.stats as sps

    def tight_bounds_dd(p_az, p_xz, clf):
        pz = {z: p_az[(0,z)] + p_az[(1,z)] for z in [0,1]}
        results = {}
        for a in [0,1]:
            p_a = sum(p_az[(a,z)] for z in [0,1])
            lo_a, hi_a = 0.0, 0.0
            for z in [0,1]:
                paz = p_az[(a,z)] / pz[z]; pxz = p_xz[(1,z)] / pz[z]
                p1_lo = max(0, pxz + paz - 1) / paz
                p1_hi = min(pxz, paz) / paz
                py0 = clf[(0,z)][1]; py1 = clf[(1,z)][1]
                c = py1 - py0
                py_lo = py0 + c*(p1_lo if c>=0 else p1_hi)
                py_hi = py0 + c*(p1_hi if c>=0 else p1_lo)
                w = p_az[(a,z)] / p_a
                lo_a += py_lo*w; hi_a += py_hi*w
            results[a] = (lo_a, hi_a)
        return results[1][0]-results[0][1], results[1][1]-results[0][0]

    scs = generate_scenario_a(n_ground_truths=200, alphas=[0.0], seed=42)
    records = []
    print("  Running solver on 200 scenarios...")
    for i, sc in enumerate(scs):
        joints = solve_feasible_set(sc.p_az, sc.p_xz, num=num_grid)
        if len(joints) < 5: continue
        dists  = distributions_over_feasible_set(joints, sc.classifier)
        j_str  = {f"p{x}{z}{a}": v for (x,z,a),v in sc.joint_gt.items()}
        gt     = compute_all_metrics(j_str, sc.classifier)
        bl     = get_baseline_bounds(sc.p_az, sc.p_xz, sc.classifier)
        t_lo, t_hi = tight_bounds_dd(sc.p_az, sc.p_xz, sc.classifier)
        dd = dists["DD"].values
        di = dists["DI"].values
        records.append({
            "sc_idx": i,
            "dd_true": gt.DD, "dd_fs_mean": float(np.mean(dd)),
            "dd_kallus_mid": (bl["DD_lo"]+bl["DD_hi"])/2,
            "dd_tight_mid": (t_lo+t_hi)/2,
            "dd_lo": float(np.min(dd)), "dd_hi": float(np.max(dd)),
            "t_lo": t_lo, "t_hi": t_hi,
            "bl_lo": bl["DD_lo"], "bl_hi": bl["DD_hi"],
            "dd_width": float(np.max(dd)-np.min(dd)),
            "bl_width": bl["DD_hi"]-bl["DD_lo"],
            "di_true": gt.DI, "di_fs_mean": float(np.mean(di)),
            "di_mid": (float(np.min(di))+float(np.max(di)))/2,
            "di_skew": float(sps.skew(di)),
            "di_lo": float(np.min(di)), "di_hi": float(np.max(di)),
            "dd_vals": dd, "di_vals": di,
        })
    print(f"  Done — {len(records)} scenarios")
    return records


def plot(save_path=None, num_grid=80):
    # Load or compute
    cache = "/tmp/comp_data.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        records = saved["records"]
        print("  Loaded cached data.")
    else:
        records = _build_data(num_grid)
        with open(cache, "wb") as f:
            pickle.dump({"records": records}, f)

    df = pd.DataFrame([{k:v for k,v in r.items() if not k.endswith("_vals")}
                       for r in records])
    df["di_err_mean"] = (df["di_fs_mean"] - df["di_true"]).abs()
    df["di_err_mid"]  = (df["di_mid"]     - df["di_true"]).abs()
    df["dd_err_mean"] = (df["dd_fs_mean"] - df["dd_true"]).abs()
    df["dd_err_kal"]  = (df["dd_kallus_mid"] - df["dd_true"]).abs()

    # Pick 4 high-skew examples for the right panels
    df_sorted = df.sort_values("di_skew", ascending=False)
    ex_indices = df_sorted.head(4).index.tolist()

    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(2, 4, wspace=0.42, hspace=0.55)
    ax_scatter = fig.add_subplot(gs[:, :2])   # full left column
    ex_axes    = [fig.add_subplot(gs[r, 2+c])
                  for r in range(2) for c in range(2)]

    # ── Left: scatter FS mean err vs interval midpoint err ────────────────
    skews  = df["di_skew"].values
    norm   = mcolors.Normalize(vmin=0, vmax=float(np.max(skews)))
    cmap   = cm.get_cmap("YlOrRd")
    colors = [cmap(norm(s)) for s in skews]

    sc_plot = ax_scatter.scatter(
        df["di_err_mid"], df["di_err_mean"],
        c=skews, cmap="YlOrRd", vmin=0, vmax=float(np.max(skews)),
        s=30, alpha=0.75, edgecolors="none")

    # Diagonal: equal performance
    mx = max(df["di_err_mid"].max(), df["di_err_mean"].max()) * 1.05
    ax_scatter.plot([0, mx], [0, mx], color="#ccc", lw=1.2, ls="--",
                    label="Equal performance")
    ax_scatter.fill_between([0, mx], [0, 0], [0, mx],
                            color=MEAN_COLOR, alpha=0.05)
    ax_scatter.fill_between([0, mx], [0, mx], [mx, mx],
                            color=MID_COLOR, alpha=0.05)

    # Region labels
    ax_scatter.text(0.72, 0.15, "FS mean\nbetter",
                    transform=ax_scatter.transAxes,
                    fontsize=9, color=MEAN_COLOR, ha="center")
    ax_scatter.text(0.15, 0.72, "Midpoint\nbetter",
                    transform=ax_scatter.transAxes,
                    fontsize=9, color=MID_COLOR, ha="center")

    pct_better = (df["di_err_mean"] < df["di_err_mid"]).mean() * 100
    ax_scatter.text(0.97, 0.97,
                    f"FS mean better\nin {pct_better:.0f}% of scenarios",
                    transform=ax_scatter.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#ccc", alpha=0.9))

    cb = plt.colorbar(sc_plot, ax=ax_scatter, shrink=0.7)
    cb.set_label("Skewness of DI distribution", fontsize=9)

    ax_scatter.set_xlabel("|Interval midpoint − true DI|", fontsize=10)
    ax_scatter.set_ylabel("|FS mean − true DI|", fontsize=10)
    ax_scatter.set_xlim(0, mx); ax_scatter.set_ylim(0, mx)
    ax_scatter.set_aspect("equal")

    # ── Right: 4 example distributions ────────────────────────────────────
    for ax, row_idx in zip(ex_axes, ex_indices):
        r    = records[row_idx]
        di   = r["di_vals"]
        true = r["di_true"]
        mean = r["di_fs_mean"]
        mid  = r["di_mid"]
        sk   = r["di_skew"]

        sns.histplot(di, ax=ax, color=FS_COLOR, alpha=0.55,
                     stat="density", bins=35, kde=True,
                     line_kws={"lw": 1.8, "color": FS_COLOR})

        ax.axvline(true, color=GT_COLOR,   lw=2.2, label="True DI")
        ax.axvline(mean, color=MEAN_COLOR, lw=2,   label="FS mean")
        ax.axvline(mid,  color=MID_COLOR,  lw=1.8, ls="--",
                   label="Interval midpoint")

        ax.set_xlabel("DI", fontsize=9)
        ax.set_ylabel("Density" if ax in [ex_axes[0], ex_axes[2]] else "",
                      fontsize=9)
        ax.set_title(f"skew = {sk:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)

        if ax is ex_axes[0]:
            ax.legend(fontsize=7.5, frameon=False)

    plt.suptitle(
        "FS mean vs interval midpoint as point estimate for DI\n"
        "When the distribution is skewed, the FS mean is a better estimate than the midpoint of [lo, hi]",
        fontsize=11, fontweight="bold", y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"  Saved: {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures",  default="figures/paper")
    parser.add_argument("--num_grid", type=int, default=80)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    if args.no_cache and os.path.exists("/tmp/comp_data.pkl"):
        os.remove("/tmp/comp_data.pkl")

    plot(save_path=os.path.join(args.figures, "fig_mean_vs_midpoint.png"),
         num_grid=args.num_grid)
