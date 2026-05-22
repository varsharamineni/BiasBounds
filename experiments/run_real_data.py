"""
experiments/run_real_data.py
-----------------------------
Run feasible set experiment on real datasets.

Simply reads each CSV, computes p(a,z) and p(x,z) from the data,
trains a classifier, then runs the feasible set solver.

Usage:
    cd fairness_experiments
    python experiments/run_real_data.py --data_dir data/real
    python experiments/run_real_data.py --data_dir /full/path/to/csvs
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.joint_solver import solve_feasible_set
from core.metrics import distributions_over_feasible_set, compute_all_metrics
from core.baseline import get_baseline_bounds, get_tight_frechet_bounds


# ---------------------------------------------------------------------------
# Dataset configs — edit if your column names differ
# ---------------------------------------------------------------------------

DATASETS = {
    
    "adult": dict(a_col="gender",   z_col="marital-status",
                  x_col="capital-gain", y_col="income"),
    "compas": dict(a_col="race",  z_col="score_text",  
                   x_col="priors_count", y_col="two_year_recid"),
    "german": dict(a_col="sex", z_col="housing",
                   x_col="employment-since", y_col="class-label"),
}





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarise(s: pd.Series) -> pd.Series:
    """Binary already → as-is. Multi-class → majority=0, rest=1."""
    if set(s.dropna().unique()).issubset({0, 1}):
        return s.astype(int)
    majority = s.value_counts().index[0]
    return (s != majority).astype(int)


def empirical_joint(df, col1, col2):
    n = len(df)
    return {(v1, v2): int(((df[col1]==v1)&(df[col2]==v2)).sum()) / n
            for v1 in [0,1] for v2 in [0,1]}


def empirical_joint3(df, c1, c2, c3):
    n = len(df)
    return {(v1,v2,v3): int(((df[c1]==v1)&(df[c2]==v2)&(df[c3]==v3)).sum())/n
            for v1 in [0,1] for v2 in [0,1] for v3 in [0,1]}


def fit_classifier(train, x_col, z_col, y_col):
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(train[[x_col, z_col]], train[y_col])
    result = {}
    for x in [0, 1]:
        for z in [0, 1]:
            p1 = float(clf.predict_proba(pd.DataFrame([[x, z]],
                       columns=[x_col, z_col]))[0, 1])
            result[(x, z)] = {0: 1-p1, 1: p1}
    return result


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------

def run_one(name, path, a_col, z_col, x_col, y_col,
            num_grid=80, seed=42):
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    df = df[[a_col, z_col, x_col, y_col]].dropna()

    for col in [a_col, z_col, x_col, y_col]:
        df[col] = binarise(df[col])

    print(f"\n  [{name}]  n={len(df)}"
          f"  P(a=1)={df[a_col].mean():.3f}"
          f"  P(z=1)={df[z_col].mean():.3f}"
          f"  P(x=1)={df[x_col].mean():.3f}"
          f"  P(y=1)={df[y_col].mean():.3f}")

    # Split: 50% train, 25% Dataset A, 25% Dataset B
    train_df, rest = train_test_split(df, test_size=0.5, random_state=seed,
                                      stratify=df[[a_col, y_col]])
    df_a, df_b    = train_test_split(rest, test_size=0.5, random_state=seed+1,
                                     stratify=rest[[a_col, y_col]])

    # Marginals from their respective splits
    p_az = empirical_joint(df_a, a_col, z_col)
    p_xz = empirical_joint(df_b, x_col, z_col)

    # Classifier trained on train split
    clf = fit_classifier(train_df, x_col, z_col, y_col)

    # Ground truth from FULL dataset
    gt_joint = empirical_joint3(df, x_col, z_col, a_col)
    j_str    = {f"p{x}{z}{a}": v for (x,z,a),v in gt_joint.items()}
    gt       = compute_all_metrics(j_str, clf)

    # Feasible set
    joints = solve_feasible_set(p_az, p_xz, num=num_grid)
    if not joints:
        print(f"  [{name}] No feasible joints.")
        return None
    dists = distributions_over_feasible_set(joints, clf)

    # Bounds
    bl    = get_baseline_bounds(p_az, p_xz, clf)
    tight = get_tight_frechet_bounds(p_az, p_xz, clf)

    row = {"dataset": name, "n_feasible": len(joints)}

    for metric in ["DD", "DI"]:
        d      = dists[metric].values
        gt_val = getattr(gt, metric)
        row.update({
            f"{metric}_gt":         round(float(gt_val), 6),
            f"{metric}_fs_mean":    round(float(np.mean(d)), 6),
            f"{metric}_fs_median":  round(float(np.median(d)), 6),
            f"{metric}_fs_std":     round(float(np.std(d)), 6),
            f"{metric}_fs_lo":      round(float(np.min(d)), 6),
            f"{metric}_fs_hi":      round(float(np.max(d)), 6),
            f"{metric}_fs_width":   round(float(np.max(d)-np.min(d)), 6),
            f"{metric}_fs_skew":    round(float(sp_skew(d)), 6),
            f"{metric}_fs_n":       len(d),
            f"{metric}_fs_covered": int(np.min(d) <= gt_val <= np.max(d)),
            f"{metric}_fs_err":     round(float(abs(np.mean(d)-gt_val)), 6),
        })

    row.update({
        "DD_baseline_lo":      round(bl["DD_lo"], 6),
        "DD_baseline_hi":      round(bl["DD_hi"], 6),
        "DD_baseline_width":   round(bl["DD_hi"]-bl["DD_lo"], 6),
        "DD_baseline_covered": int(bl["DD_lo"] <= gt.DD <= bl["DD_hi"]),
        "tight_DD_lo":         round(tight["tight_DD_lo"], 6),
        "tight_DD_hi":         round(tight["tight_DD_hi"], 6),
        "tight_DD_width":      round(tight["tight_DD_width"], 6),
        "tight_DD_covered":    int(tight["tight_DD_lo"] <= gt.DD <= tight["tight_DD_hi"]),
    })

    print(f"    DD: true={gt.DD:.3f}  FS=[{row['DD_fs_lo']:.3f},{row['DD_fs_hi']:.3f}]  mean={row['DD_fs_mean']:.3f}")
    print(f"    DI: true={gt.DI:.3f}  FS=[{row['DI_fs_lo']:.3f},{row['DI_fs_hi']:.3f}]  mean={row['DI_fs_mean']:.3f}")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/real")
    parser.add_argument("--num_grid", type=int, default=80)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   default="results/real_results.csv")
    args = parser.parse_args()

    rows = []
    for name, cfg in DATASETS.items():
        path = os.path.join(args.data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  Skipping {name} — not found at {path}")
            continue
        row = run_one(name, path, num_grid=args.num_grid,
                      seed=args.seed, **cfg)
        if row:
            rows.append(row)

    if rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\n  Saved → {args.output}  ({len(rows)} datasets)")
    else:
        print("\n  No results. Check --data_dir and CSV names.")