"""
analysis/fig_real_datasets.py
-------------------------------
Single publication figure:

One figure (3×1 layout):
  - Adult (DI only)
  - COMPAS (DI only)
  - German (DI only)

Each panel shows:
  - KDE histogram of feasible-set DI distribution
  - True DI (orange vertical line)
  - Feasible-set mean (green vertical line)

Two modes:
  A) Precomputed CSV (fast)
  B) Raw CSV (exact feasible set recomputation — preferred)

Output:
  figures/paper/fig_real_di_only.png
"""

from __future__ import annotations
import os, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
FS_COLOR   = "steelblue"
GT_COLOR   = "#D85A30"
MEAN_COLOR = "#085041"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

DATASET_ORDER = ["adult", "compas", "german"]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    print(f"Saved: {path}")


def _approx_dist(mean, std, lo, hi, n=1000):
    """Truncated normal approximation of feasible-set distribution."""
    from scipy.stats import truncnorm

    if not all(np.isfinite([mean, std, lo, hi])) or std < 1e-9:
        return None

    a = (lo - mean) / std
    b = (hi - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=42)


# ---------------------------------------------------------------------
# Loading (exact recomputation mode)
# ---------------------------------------------------------------------
def _load_exact(data_dir):
    from data.loader import load_real_data, DATASET_CONFIGS
    from core.joint_solver import solve_feasible_set
    from core.metrics import distributions_over_feasible_set, compute_all_metrics

    results = {}

    for name in DATASET_ORDER:
        cfg = DATASET_CONFIGS.get(name, {})
        path = os.path.join(data_dir, f"{name}.csv")

        if not cfg or not os.path.exists(path):
            print(f"Skipping {name}")
            continue

        print(f"Loading {name}...")

        scs, _ = load_real_data(path=path, name=name, scenario="A", **cfg)
        sc = scs[0]

        joints = solve_feasible_set(sc.p_az, sc.p_xz, num=80)
        dists  = distributions_over_feasible_set(joints, sc.classifier)

        j_str = {f"p{x}{z}{a}": v for (x, z, a), v in sc.joint_gt.items()}
        gt = compute_all_metrics(j_str, sc.classifier)

        results[name] = {
            "di_vals": dists["DI"].values,
            "di_true": gt.DI,
            "di_mean": float(np.mean(dists["DI"].values)),
        }

    return results


# ---------------------------------------------------------------------
# Loading (precomputed CSV mode)
# ---------------------------------------------------------------------
def _load_from_csv(real_a):
    import pandas as pd

    df = pd.read_csv(real_a)
    name_col = "dataset" if "dataset" in df.columns else "label"

    results = {}

    for name in DATASET_ORDER:
        row = df[df[name_col] == name]
        if row.empty:
            continue

        r = row.iloc[0]

        def f(k):
            try:
                v = float(r.get(k, np.nan))
                return v if np.isfinite(v) else np.nan
            except:
                return np.nan

        di_vals = _approx_dist(
            f("DI_fs_mean"),
            f("DI_fs_std"),
            f("DI_fs_lo"),
            f("DI_fs_hi"),
        )

        results[name] = {
            "di_vals": di_vals,
            "di_true": f("DI_gt"),
            "di_mean": f("DI_fs_mean"),
        }

    return results


# ---------------------------------------------------------------------
# Plot (DI only, 3×1)
# ---------------------------------------------------------------------
def plot_di_only(results, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))

    for ax, name in zip(axes, DATASET_ORDER):
        data = results[name]

        vals = data.get("di_vals", None)
        true = data.get("di_true", np.nan)
        mean = data.get("di_mean", np.nan)

        if vals is not None and len(vals) > 0:
            sns.histplot(
                vals,
                ax=ax,
                color=FS_COLOR,
                alpha=0.6,
                stat="probability",
                bins=30,
                zorder=2
            )

        if np.isfinite(mean):
            ax.axvline(mean, color=MEAN_COLOR, lw=2, zorder=4, label="FS mean")

        if np.isfinite(true):
            ax.axvline(true, color=GT_COLOR, lw=2.3, zorder=5, label="True value")

        # Show legend only on the top subplot
        if (np.isfinite(mean) or np.isfinite(true)) and ax is axes[0]:
            ax.legend(loc="upper right", frameon=False)

        ax.set_title(f"{name.title()}")
        ax.set_ylabel("Probability")

    axes[-1].set_xlabel("Disparate Impact (DI)")

    plt.tight_layout()
    _save(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run(results, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, "fig_real_di_only.pdf")
    plot_di_only(results, save_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None,
                        help="Raw CSVs (exact mode)")
    parser.add_argument("--real_a", default="results/real_results.csv",
                        help="Precomputed results CSV")
    parser.add_argument("--figures", default="figures/paper")

    args = parser.parse_args()

    if args.data_dir and os.path.isdir(args.data_dir):
        print("Using exact recomputation mode...")
        results = _load_exact(args.data_dir)
    else:
        print("Using precomputed CSV mode...")
        results = _load_from_csv(args.real_a)

    if not results:
        print("No data found.")
    else:
        run(results, args.figures)