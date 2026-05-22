"""
analysis/paper_plots.py
------------------------
Generates exactly three figures for the paper:

  Fig 1 — Simulated study: distribution of (FS mean − true) for DI and DD
           across all scenarios. Already provided by user, included here
           for one-command convenience.

  Fig 2 — Real-world data: for each dataset (adult, compas, german),
           KDE histogram of feasible DI and DD distributions, with
           true metric, FS mean, and baseline interval marked.

  Fig 3 — Bounds comparison: FS interval vs Kallus baseline vs tight
           Fréchet bounds for DD on consistent scenarios, shown as a
           strip plot sorted by true DD.

Run:
    cd ~/bounding-fairness/fairness_experiments

    python analysis/paper_plots.py \
        --simulated results/experiment_results_25k.csv \
        --real_a    results/real_v2_a.csv \
        --real_b    results/real_v2_b.csv \
        --figures   figures/paper
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import truncnorm

FS_COLOR     = "#1D9E75"
BL_COLOR     = "#AAAAAA"
TIGHT_COLOR  = "#534AB7"
GT_COLOR     = "#D85A30"
MEAN_COLOR   = "#085041"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

LABEL_ORDER = ["consistent", "low", "medium", "high"]
LC = {"consistent": "#1D9E75", "low": "#5DCAA5",
      "medium":     "#534AB7", "high": "#D85A30"}
LABEL_MAP = {"consistent": "Consistent", "low": "Low",
             "medium": "Medium",          "high": "High"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    print(f"  Saved: {path}")


# =============================================================================
# Figure 1 — Simulated study accuracy  (provided by user, reproduced here)
# =============================================================================

def fig1_accuracy(df: pd.DataFrame, save_path: str):
    """
    Simple publication-style figure:
    - Two panels: DI and DD
    - Probability histogram of FS mean − true
    - One color per metric (clean separation)
    - Minimal annotations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    COLORS = {
        "DI": "steelblue",   # muted blue
        "DD": "#1b7764"    # muted orange
    }

    for ax, metric, label in zip(
        axes,
        ["DI", "DD"],
        ["Disparate Impact (DI)", "Demographic Disparity (DD)"]
    ):
        mean_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else f"{metric}_fs_mean"
        gt_col = f"{metric}_gt"

        err = (df[mean_col] - df[gt_col]).dropna()

        lo, hi = np.nanpercentile(err, [1, 99])
        bins = np.linspace(lo, hi, 30)

        sns.histplot(
            err,
            bins=bins,
            stat="probability",
            ax=ax,
            color=COLORS[metric],
            alpha=0.70
        )

        # zero reference
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        # stats box
        ax.text(
            0.97, 0.97,
            f"mean = {err.mean():+.3f}\nstd  = {err.std():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        )

        ax.set_xlabel("FS mean − true")
        ax.set_ylabel("Probability")
        ax.set_title(label)

    plt.tight_layout()
    _save(fig, save_path)
    return fig
# =============================================================================
# Figure 2 — Real-world data
# =============================================================================

def _approx_distribution(mean, std, lo, hi, n=800):
    """Truncated normal approximation from summary stats."""
    if not all(np.isfinite([mean, std, lo, hi])) or std < 1e-9:
        return None
    a = (lo - mean) / std
    b = (hi - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=42)


def _get_real_row(df, dataset_name):
    col = "dataset" if "dataset" in df.columns else "label"
    rows = df[df[col] == dataset_name]
    return rows.iloc[0] if len(rows) else None


def fig2_real_world(da: pd.DataFrame, db: pd.DataFrame, save_path: str):
    """
    One row per dataset, two panels per row (DD left, DI right).
    KDE histogram of feasible set distribution, baseline shaded band,
    true metric and FS mean as vertical lines.
    """
    name_col = "dataset" if "dataset" in da.columns else "label"
    datasets = da[name_col].unique()
    n_ds     = len(datasets)

    fig, axes = plt.subplots(n_ds, 2,
                             figsize=(12, 3.2 * n_ds),
                             squeeze=False)
    plt.subplots_adjust(hspace=0.55, wspace=0.35)

    for row_i, ds in enumerate(datasets):
        row_a = _get_real_row(da, ds)

        for col_i, metric in enumerate(["DD", "DI"]):
            ax = axes[row_i][col_i]
            if row_a is None:
                ax.axis("off")
                continue

            def flt(key):
                try:
                    v = float(row_a.get(key, np.nan))
                    return v if np.isfinite(v) else np.nan
                except Exception:
                    return np.nan

            mean = flt(f"{metric}_fs_mean")
            std  = flt(f"{metric}_fs_std")
            lo   = flt(f"{metric}_fs_lo")
            hi   = flt(f"{metric}_fs_hi")
            true = flt(f"{metric}_gt")
            bl_lo = flt(f"{metric}_baseline_lo")
            bl_hi = flt(f"{metric}_baseline_hi")

            vals = _approx_distribution(mean, std, lo, hi)
            if vals is not None:
                sns.histplot(vals, ax=ax, color=FS_COLOR, alpha=0.60,
                             stat="frequency", bins=30, 
                             line_kws={"lw": 2, "color": FS_COLOR})

            # Baseline band (DD only)
            if np.isfinite(bl_lo) and np.isfinite(bl_hi):
                ax.axvspan(bl_lo, bl_hi, alpha=0.12, color=BL_COLOR)
                ax.axvline(bl_lo, color=BL_COLOR, lw=1.2, ls="--", alpha=0.7)
                ax.axvline(bl_hi, color=BL_COLOR, lw=1.2, ls="--", alpha=0.7)
            elif metric == "DI":
                ax.text(0.97, 0.05, "No baseline\n(ratio metric)",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color=BL_COLOR)

            if np.isfinite(mean):
                ax.axvline(mean, color=MEAN_COLOR, lw=2,
                           label=f"FS mean = {mean:.3f}")
            if np.isfinite(true):
                ax.axvline(true, color=GT_COLOR, lw=2.2,
                           label=f"True = {true:.3f}")

            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel("Density" if col_i == 0 else "", fontsize=10)
            if row_i == 0:
                ax.set_title(
                    "Demographic Disparity (DD)" if metric == "DD"
                    else "Disparate Impact (DI)", fontsize=10)
            ax.legend(fontsize=8, frameon=False)

        # Dataset label on left
        axes[row_i][0].annotate(
            ds.capitalize(), xy=(-0.22, 0.5),
            xycoords="axes fraction", fontsize=11,
            fontweight="bold", va="center", ha="right",
            rotation=90)

    # Shared legend
    fig.legend(handles=[
        mpatches.Patch(facecolor=FS_COLOR, alpha=0.65,
                       label="Feasible set distribution"),
        mpatches.Patch(facecolor=BL_COLOR, alpha=0.25,
                       label="Baseline interval (DD only)"),
        mlines.Line2D([0],[0], color=MEAN_COLOR, lw=2, label="FS mean"),
        mlines.Line2D([0],[0], color=GT_COLOR, lw=2.2, label="True metric"),
    ], fontsize=9, loc="lower center", ncol=4, frameon=False,
       bbox_to_anchor=(0.5, -0.02))

    _save(fig, save_path)
    return fig


# =============================================================================
# Figure 3 — Bounds comparison (strip plot)
# =============================================================================

def _tight_bounds_dd(p_az, p_xz, clf):
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
            lo_a += py_lo * p_az[(a,z)] / p_a
            hi_a += py_hi * p_az[(a,z)] / p_a
        results[a] = (lo_a, hi_a)
    return results[1][0] - results[0][1], results[1][1] - results[0][0]


def fig3_bounds_comparison(save_path: str, n_gt: int = 30, num_grid: int = 60):
    """
    Strip plot: one row per consistent scenario sorted by true DD.
    Shows Kallus (grey), tight Fréchet (purple), FS (teal) intervals
    with true DD and FS mean marked.
    Annotated with median widths and coverage.
    """
    from core.scenarios.scenario_a import generate_scenario_a
    from core.joint_solver import solve_feasible_set
    from core.metrics import distributions_over_feasible_set, compute_all_metrics
    from core.baseline import get_baseline_bounds

    print(f"  Running solver on {n_gt} consistent scenarios...")
    scs  = generate_scenario_a(n_ground_truths=n_gt, alphas=[0.0], seed=42)
    rows = []
    for sc in scs:
        joints = solve_feasible_set(sc.p_az, sc.p_xz, num=num_grid)
        if len(joints) < 5: continue
        dists  = distributions_over_feasible_set(joints, sc.classifier)
        j_str  = {f"p{x}{z}{a}": v for (x,z,a),v in sc.joint_gt.items()}
        gt     = compute_all_metrics(j_str, sc.classifier)
        bl     = get_baseline_bounds(sc.p_az, sc.p_xz, sc.classifier)
        t_lo, t_hi = _tight_bounds_dd(sc.p_az, sc.p_xz, sc.classifier)
        d = dists["DD"].values
        rows.append(dict(
            true=gt.DD,
            fs_lo=float(np.min(d)), fs_hi=float(np.max(d)),
            fs_mean=float(np.mean(d)),
            t_lo=t_lo, t_hi=t_hi,
            bl_lo=bl["DD_lo"], bl_hi=bl["DD_hi"],
        ))
    print("  Done.")

    df = pd.DataFrame(rows).sort_values("true").reset_index(drop=True)
    n  = len(df)

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.30)))

    for i, row in df.iterrows():
        y = float(i)

        # Kallus (grey, widest, behind)
        ax.barh(y, row.bl_hi - row.bl_lo, left=row.bl_lo,
                height=0.60, color=BL_COLOR, alpha=0.25, zorder=1)
        ax.plot([row.bl_lo, row.bl_lo], [y-.30, y+.30],
                color=BL_COLOR, lw=1.2, alpha=0.6, zorder=2)
        ax.plot([row.bl_hi, row.bl_hi], [y-.30, y+.30],
                color=BL_COLOR, lw=1.2, alpha=0.6, zorder=2)

        # Tight Fréchet (purple, middle)
        ax.barh(y, row.t_hi - row.t_lo, left=row.t_lo,
                height=0.42, color=TIGHT_COLOR, alpha=0.30, zorder=3)
        ax.plot([row.t_lo, row.t_lo], [y-.21, y+.21],
                color=TIGHT_COLOR, lw=1.2, alpha=0.8, zorder=4)
        ax.plot([row.t_hi, row.t_hi], [y-.21, y+.21],
                color=TIGHT_COLOR, lw=1.2, alpha=0.8, zorder=4)

        # Feasible set (teal, narrowest, front)
        ax.barh(y, row.fs_hi - row.fs_lo, left=row.fs_lo,
                height=0.26, color=FS_COLOR, alpha=0.80, zorder=5)

        # FS mean (dark diamond)
        ax.scatter(row.fs_mean, y, color=MEAN_COLOR,
                   s=35, marker="D", zorder=7, linewidths=0)

        # True DD (black circle)
        ax.scatter(row.true, y, color=GT_COLOR,
                   s=45, zorder=8, linewidths=0)

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.25)
    ax.set_yticks([])
    ax.set_xlabel("Demographic Disparity (DD)", fontsize=11)
    ax.set_ylabel(f"Scenario  (n = {n}, sorted by true DD ↑)", fontsize=10)
    ax.spines["left"].set_visible(False)

    # Stats box
    bl_med = (df.bl_hi - df.bl_lo).median()
    t_med  = (df.t_hi  - df.t_lo).median()
    fs_med = (df.fs_hi - df.fs_lo).median()
    bl_cov = ((df.bl_lo <= df.true) & (df.true <= df.bl_hi)).mean() * 100
    t_cov  = ((df.t_lo  <= df.true) & (df.true <= df.t_hi)).mean()  * 100
    fs_cov = ((df.fs_lo <= df.true) & (df.true <= df.fs_hi)).mean() * 100

    ax.text(0.015, 0.985,
            f"Median width\n"
            f"  Kallus           {bl_med:.3f}\n"
            f"  Tight Fréchet    {t_med:.3f}  ({(1-t_med/bl_med)*100:.0f}% narrower)\n"
            f"  Feasible set     {fs_med:.3f}  ({(1-fs_med/bl_med)*100:.0f}% narrower)\n\n"
            f"Coverage of true DD\n"
            f"  Kallus           {bl_cov:.0f}%\n"
            f"  Tight Fréchet    {t_cov:.0f}%\n"
            f"  Feasible set     {fs_cov:.0f}%",
            transform=ax.transAxes, va="top", fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec="#bbb", alpha=0.97))

    ax.legend(handles=[
        mpatches.Patch(facecolor=BL_COLOR,    alpha=0.35, label="Kallus bounds"),
        mpatches.Patch(facecolor=TIGHT_COLOR, alpha=0.40, label="Tight Fréchet bounds"),
        mpatches.Patch(facecolor=FS_COLOR,    alpha=0.85, label="Feasible set interval"),
        mlines.Line2D([0],[0], marker="o", color=GT_COLOR,   ms=7, lw=0, label="True DD"),
        mlines.Line2D([0],[0], marker="D", color=MEAN_COLOR, ms=6, lw=0, label="FS mean"),
    ], fontsize=8.5, loc="lower right", framealpha=0.97, edgecolor="#ccc")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Run all three
# =============================================================================

def run_all(sim_path, real_a_path, real_b_path, fig_dir,
            bounds_n_gt=50, bounds_num_grid=60):
    os.makedirs(fig_dir, exist_ok=True)

    # Fig 1 — simulated accuracy
    print("\n[Fig 1] Simulated study accuracy")
    df = pd.read_csv(sim_path)
    print(f"  Loaded {len(df):,} simulated scenarios")
    fig1_accuracy(df, save_path=os.path.join(fig_dir, "fig1_accuracy.pdf"))

    # Fig 2 — real world
    if os.path.exists(real_a_path) and os.path.exists(real_b_path):
        print("\n[Fig 2] Real-world data")
        da = pd.read_csv(real_a_path)
        db = pd.read_csv(real_b_path)
        print(f"  Loaded {len(da)} Scenario A rows, {len(db)} Scenario B rows")
        fig2_real_world(da, db,
                        save_path=os.path.join(fig_dir, "fig2_real_world.pdf"))
    else:
        print(f"\n[Fig 2] Real data not found — skipping.")
        print(f"  Run: experiments/v2/run_experiment_v2.py --real ...")

    # Fig 3 — bounds comparison
    print("\n[Fig 3] Bounds comparison")
    fig3_bounds_comparison(
        save_path=os.path.join(fig_dir, "fig3_bounds_comparison.pdf"),
        n_gt=bounds_n_gt, num_grid=bounds_num_grid)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated",  default="results/experiment_results_25k.csv",
                        help="Simulated results CSV (25k scenarios)")
    parser.add_argument("--real_a",     default="results/real_v2_a.csv")
    parser.add_argument("--real_b",     default="results/real_v2_b.csv")
    parser.add_argument("--figures",    default="figures/paper")
    parser.add_argument("--bounds_n_gt",    type=int, default=50,
                        help="Scenarios for bounds strip plot")
    parser.add_argument("--bounds_grid",    type=int, default=60)
    args = parser.parse_args()

    run_all(args.simulated, args.real_a, args.real_b, args.figures,
            bounds_n_gt=args.bounds_n_gt,
            bounds_num_grid=args.bounds_grid)
