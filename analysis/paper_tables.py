"""
analysis/paper_tables.py
--------------------------
Produces all analysis tables for the paper.

Table 1 — General stats: FS mean accuracy
  Mean/std of (FS mean − true) and interval width for DD and DI
  across all scenarios and broken down by inconsistency level.

Table 2 — Coverage
  % of scenarios where true metric falls inside FS interval,
  Kallus interval, and tight Fréchet interval.

Table 3 — Inconsistency impact (KL bins)
  Mean |error| and width across KL divergence bins.

Table 4 — Bounds comparison (consistent scenarios only)
  Median width: Kallus vs tight Fréchet vs FS.
  Midpoint |error| for each.

Table 5 — FS mean vs interval midpoint (DI, skewed distribution)
  Overall and for high-skew scenarios.

Run:
    cd fairness_experiments
    python analysis/paper_tables.py
    python analysis/paper_tables.py --results results/experiment_results_25k.csv
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

KL_BINS   = [0, 0.01, 0.05, 0.1, 0.3, 0.5, np.inf]
KL_LABELS = ["0–0.01", "0.01–0.05", "0.05–0.1",
              "0.1–0.3",  "0.3–0.5",  "0.5+"]
LABEL_ORDER = ["consistent", "low", "medium", "high"]


def _print_table(title, df_table):
    print()
    print("=" * 75)
    print(f"  {title}")
    print("=" * 75)
    print(df_table.to_string())
    print()


def _save_table(df_table, path, title):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df_table.to_csv(path)
    print(f"  Saved: {path}")


# =============================================================================
# Table 1 — General FS accuracy stats
# =============================================================================

def table1_general_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in LABEL_ORDER + ["all"]:
        sub = df if level == "all" else df[df.label == level]
        n   = len(sub)
        for metric in ["DD", "DI"]:
            signed  = sub[f"{metric}_mean"] - sub[f"{metric}_gt"]
            width   = sub[f"{metric}_width"]
            rows.append({
                "Level":           level.capitalize() if level != "all" else "All",
                "Metric":          metric,
                "n":               n,
                "Mean error":      round(float(signed.mean()), 4),
                "Std error":       round(float(signed.std()),  4),
                "|error| mean":    round(float(signed.abs().mean()), 4),
                "Interval width":  round(float(width.mean()), 4),
                "Coverage %":      round(float(sub[f"{metric}_covered"].mean()*100), 1),
            })
    return pd.DataFrame(rows).set_index(["Level", "Metric"])


# =============================================================================
# Table 2 — Coverage: FS vs Kallus vs tight Fréchet
# =============================================================================

def table2_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cons = df[df.label == "consistent"]
    for level in LABEL_ORDER + ["all"]:
        sub = df if level == "all" else df[df.label == level]
        n   = len(sub)
        row = {"Level": level.capitalize() if level != "all" else "All", "n": n}

        # FS coverage
        row["FS DD cov %"]    = round(sub["DD_covered"].mean()*100, 1)
        row["FS DI cov %"]    = round(sub["DI_covered"].mean()*100, 1)

        # Tight Fréchet (available for all scenarios)
        if "tight_DD_covered" in df.columns:
            row["Tight DD cov %"] = round(sub["tight_DD_covered"].mean()*100, 1)
        else:
            row["Tight DD cov %"] = "—"

        # Kallus (consistent only)
        if level == "consistent":
            row["Kallus DD cov %"] = round(cons["DD_bl_covered"].mean()*100, 1)
        elif level == "all":
            row["Kallus DD cov %"] = round(cons["DD_bl_covered"].mean()*100, 1) \
                                     if "DD_bl_covered" in df.columns else "—"
        else:
            row["Kallus DD cov %"] = "n/a"

        rows.append(row)
    return pd.DataFrame(rows).set_index("Level")


# =============================================================================
# Table 3 — Inconsistency impact (KL bins)
# =============================================================================

def table3_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kl_bin"] = pd.cut(df["kl_mean"], bins=KL_BINS, labels=KL_LABELS)
    grp  = df.groupby("kl_bin", observed=True)
    rows = []
    for label in KL_LABELS:
        sub = df[df["kl_bin"] == label]
        if len(sub) == 0:
            continue
        rows.append({
            "KL bin":          label,
            "n":               len(sub),
            "DD |err| mean":   round(float(sub["DD_mean_err"].mean()),  4),
            "DI |err| mean":   round(float(sub["DI_mean_err"].mean()),  4),
            "DD width mean":   round(float(sub["DD_width"].mean()),     4),
            "DI width mean":   round(float(sub["DI_width"].mean()),     4),
            "DD cov %":        round(float(sub["DD_covered"].mean()*100), 1),
            "DI cov %":        round(float(sub["DI_covered"].mean()*100), 1),
        })
    return pd.DataFrame(rows).set_index("KL bin")


# =============================================================================
# Table 4 — Bounds comparison (consistent scenarios)
# =============================================================================

def table4_bounds(df: pd.DataFrame) -> pd.DataFrame:
    cons = df[df.label == "consistent"].copy()
    n    = len(cons)

    rows = []

    # Kallus
    bl_w  = cons["DD_bl_width"].dropna()
    bl_mid_err = ((cons["DD_bl_lo"]+cons["DD_bl_hi"])/2 - cons["DD_gt"]).abs()
    rows.append({
        "Method":          "Kallus baseline",
        "n":               len(bl_w),
        "Median width":    round(float(bl_w.median()), 4),
        "Mean width":      round(float(bl_w.mean()),   4),
        "Midpoint |err|":  round(float(bl_mid_err.mean()), 4),
        "Coverage %":      round(float(cons["DD_bl_covered"].mean()*100), 1),
        "Width reduction": "—",
    })

    # Tight Fréchet
    if "tight_DD_width" in cons.columns:
        t_w = cons["tight_DD_width"].dropna()
        t_mid_err = cons["tight_DD_err"].dropna()
        bl_med = bl_w.median()
        rows.append({
            "Method":          "Tight Fréchet",
            "n":               len(t_w),
            "Median width":    round(float(t_w.median()), 4),
            "Mean width":      round(float(t_w.mean()),   4),
            "Midpoint |err|":  round(float(t_mid_err.mean()), 4),
            "Coverage %":      round(float(cons["tight_DD_covered"].mean()*100), 1),
            "Width reduction": f"{(1-t_w.median()/bl_med)*100:.0f}%",
        })

    # Feasible set
    fs_w = cons["DD_width"]
    fs_mid_err = ((cons["DD_lo"]+cons["DD_hi"])/2 - cons["DD_gt"]).abs()
    bl_med = bl_w.median()
    rows.append({
        "Method":          "Feasible set",
        "n":               n,
        "Median width":    round(float(fs_w.median()), 4),
        "Mean width":      round(float(fs_w.mean()),   4),
        "Midpoint |err|":  round(float(fs_mid_err.mean()), 4),
        "Coverage %":      round(float(cons["DD_covered"].mean()*100), 1),
        "Width reduction": f"{(1-fs_w.median()/bl_med)*100:.0f}%",
    })

    return pd.DataFrame(rows).set_index("Method")


# =============================================================================
# Table 5 — FS mean vs interval midpoint (DI, skewness matters)
# =============================================================================

def table5_mean_vs_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    import scipy.stats as sps

    df = df.copy()
    df["DI_mid"] = (df["DI_lo"] + df["DI_hi"]) / 2
    df["err_mean"] = (df["DI_mean"] - df["DI_gt"]).abs()
    df["err_mid"]  = (df["DI_mid"]  - df["DI_gt"]).abs()
    df["skew_abs"] = df["DI_skew"].abs()

    rows = []
    for label, mask in [
        ("All scenarios",           np.ones(len(df), dtype=bool)),
        ("Low skew (|s|<0.3)",      df["skew_abs"] < 0.3),
        ("High skew (|s|>0.3)",     df["skew_abs"] > 0.3),
        ("Very high skew (|s|>0.5)",df["skew_abs"] > 0.5),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "Subset":            label,
            "n":                 len(sub),
            "Mean skewness":     round(float(sub["DI_skew"].mean()), 3),
            "FS mean |err|":     round(float(sub["err_mean"].mean()), 4),
            "Midpoint |err|":    round(float(sub["err_mid"].mean()),  4),
            "FS mean better %":  round(float((sub["err_mean"]<sub["err_mid"]).mean()*100), 1),
        })
    return pd.DataFrame(rows).set_index("Subset")


# =============================================================================
# Run all
# =============================================================================


# =============================================================================
# Table 6 — Detailed estimate vs true for each method
# =============================================================================

def table6_estimate_vs_true(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric (DD, DI) x level x method:
      signed error (bias), |error| mean, std
    Methods: FS mean, FS median, tight midpoint (DD), Kallus midpoint (DD, consistent)
    """
    rows = []
    for level in LABEL_ORDER + ["all"]:
        sub  = df if level == "all" else df[df.label == level]
        cons = df[df.label == "consistent"]
        n    = len(sub)
        lbl  = level.capitalize() if level != "all" else "All"

        for metric in ["DD", "DI"]:
            gt = sub[f"{metric}_gt"]

            # FS mean
            err = sub[f"{metric}_mean"] - gt
            rows.append({"Level": lbl, "Metric": metric, "Method": "FS mean",
                         "n": n,
                         "Signed error": round(float(err.mean()), 4),
                         "|Error| mean": round(float(err.abs().mean()), 4),
                         "Std":          round(float(err.std()), 4)})

            # FS median
            err = sub[f"{metric}_median"] - gt
            rows.append({"Level": lbl, "Metric": metric, "Method": "FS median",
                         "n": n,
                         "Signed error": round(float(err.mean()), 4),
                         "|Error| mean": round(float(err.abs().mean()), 4),
                         "Std":          round(float(err.std()), 4)})

            # Tight midpoint (DD only)
            if metric == "DD" and "tight_DD_lo" in df.columns:
                mid = (sub["tight_DD_lo"] + sub["tight_DD_hi"]) / 2
                err = mid - gt
                rows.append({"Level": lbl, "Metric": metric, "Method": "Tight midpoint",
                             "n": n,
                             "Signed error": round(float(err.mean()), 4),
                             "|Error| mean": round(float(err.abs().mean()), 4),
                             "Std":          round(float(err.std()), 4)})

            # Kallus midpoint (DD only, consistent)
            if metric == "DD":
                kal_sub = cons if level in ("consistent", "all") else sub
                bl_valid = kal_sub[["DD_bl_lo","DD_bl_hi"]].dropna()
                if len(bl_valid) > 0:
                    mid = (kal_sub["DD_bl_lo"] + kal_sub["DD_bl_hi"]) / 2
                    err = (mid - kal_sub["DD_gt"]).dropna()
                    rows.append({"Level": lbl, "Metric": metric,
                                 "Method": "Kallus midpoint",
                                 "n": len(err),
                                 "Signed error": round(float(err.mean()), 4),
                                 "|Error| mean": round(float(err.abs().mean()), 4),
                                 "Std":          round(float(err.std()), 4)})

    return pd.DataFrame(rows).set_index(["Level", "Metric", "Method"])



# =============================================================================
# Paper text stats — formatted numbers for direct use in writing
# =============================================================================

def print_paper_numbers(df: pd.DataFrame):
    """
    Print all key numbers formatted for direct insertion into paper text.
    """
    print()
    print("=" * 75)
    print("  NUMBERS FOR PAPER TEXT")
    print("=" * 75)

    n = len(df)
    kl_min = df["kl_mean"].min()
    kl_max = df["kl_mean"].max()

    print(f"\nDataset: n = {n:,} scenarios")
    print(f"KL divergence range: {kl_min:.3f} to {kl_max:.3f}")

    print()
    print("--- Overall accuracy (FS mean vs true) ---")
    for metric in ["DD", "DI"]:
        signed = df[f"{metric}_mean"] - df[f"{metric}_gt"]
        width  = df[f"{metric}_width"]
        cov    = df[f"{metric}_covered"].mean() * 100
        print(f"  {metric}:")
        print(f"    Signed error:  mean = {signed.mean():+.3f}  (±{signed.std():.3f})")
        print(f"    |Error|:       mean = {signed.abs().mean():.3f}  median = {signed.abs().median():.3f}")
        print(f"    Interval width: mean = {width.mean():.3f}  (±{width.std():.3f})")
        print(f"    Coverage:      {cov:.1f}%")

    print()
    print("--- KL bin breakdown: signed error and width ---")
    df2 = df.copy()
    df2["kl_bin"] = pd.cut(df2["kl_mean"], bins=KL_BINS, labels=KL_LABELS)
    for metric in ["DD", "DI"]:
        print(f"  {metric}:")
        print(f"  {'Bin':<14} {'n':>5}  {'Signed err':>12}  {'|err|':>8}  {'Width':>8}  {'Cov%':>6}")
        for lbl in KL_LABELS:
            sub = df2[df2["kl_bin"] == lbl]
            if len(sub) == 0:
                continue
            signed = sub[f"{metric}_mean"] - sub[f"{metric}_gt"]
            width  = sub[f"{metric}_width"]
            cov    = sub[f"{metric}_covered"].mean() * 100
            print(f"  {lbl:<14} {len(sub):>5}  {signed.mean():>+12.4f}  "
                  f"{signed.abs().mean():>8.4f}  {width.mean():>8.4f}  {cov:>6.1f}%")
        print()

    print()
    print("--- Bounds comparison (consistent scenarios) ---")
    cons = df[df.label == "consistent"]
    nc   = len(cons)
    print(f"  n = {nc} consistent scenarios")
    for method, lo_col, hi_col, cov_col in [
        ("Kallus",        "DD_bl_lo",     "DD_bl_hi",     "DD_bl_covered"),
        ("Tight Fréchet", "tight_DD_lo",  "tight_DD_hi",  "tight_DD_covered"),
        ("Feasible set",  "DD_lo",        "DD_hi",        "DD_covered"),
    ]:
        if lo_col not in cons.columns:
            continue
        w   = (cons[hi_col] - cons[lo_col]).dropna()
        cov = cons[cov_col].mean() * 100
        mid_err = ((cons[lo_col]+cons[hi_col])/2 - cons["DD_gt"]).abs().dropna()
        print(f"  {method:<16}  width median={w.median():.3f}  mean={w.mean():.3f}  "
              f"midpt|err|={mid_err.mean():.3f}  cov={cov:.1f}%")

    print()
    print("--- DI mean vs midpoint (skewness) ---")
    df2["DI_mid"] = (df2["DI_lo"] + df2["DI_hi"]) / 2
    df2["err_mean"] = (df2["DI_mean"] - df2["DI_gt"]).abs()
    df2["err_mid"]  = (df2["DI_mid"]  - df2["DI_gt"]).abs()
    df2["skew_abs"] = df2["DI_skew"].abs()
    for label, mask in [
        ("All",              np.ones(len(df2), bool)),
        ("|skew| > 0.3",     df2["skew_abs"] > 0.3),
        ("|skew| > 0.5",     df2["skew_abs"] > 0.5),
    ]:
        sub = df2[mask]
        better = (sub["err_mean"] < sub["err_mid"]).mean() * 100
        print(f"  {label:<16}  n={len(sub):>4}  "
              f"FS mean |err|={sub['err_mean'].mean():.3f}  "
              f"midpt |err|={sub['err_mid'].mean():.3f}  "
              f"FS mean better={better:.0f}%")
    print()


def run_all(results_path: str, out_dir: str = "results"):
    print(f"\nLoading: {results_path}")
    df = pd.read_csv(results_path)
    print(f"  {len(df):,} scenarios  —  {df.label.value_counts().to_dict()}")

    t1 = table1_general_stats(df)
    t2 = table2_coverage(df)
    t3 = table3_inconsistency(df)
    t4 = table4_bounds(df)
    t5 = table5_mean_vs_midpoint(df)

    _print_table("Table 1: FS accuracy — signed error and width by level", t1)
    _print_table("Table 2: Coverage by inconsistency level", t2)
    _print_table("Table 3: Impact of marginal inconsistency (KL bins)", t3)
    _print_table("Table 4: Bounds comparison (DD, consistent scenarios)", t4)
    _print_table("Table 5: FS mean vs interval midpoint for DI", t5)

    os.makedirs(out_dir, exist_ok=True)
    _save_table(t1, os.path.join(out_dir, "table1_general_stats.csv"),    "Table 1")
    _save_table(t2, os.path.join(out_dir, "table2_coverage.csv"),         "Table 2")
    _save_table(t3, os.path.join(out_dir, "table3_inconsistency.csv"),    "Table 3")
    _save_table(t4, os.path.join(out_dir, "table4_bounds.csv"),           "Table 4")
    _save_table(t5, os.path.join(out_dir, "table5_mean_vs_midpoint.csv"), "Table 5")

    print_paper_numbers(df)

    t6 = table6_estimate_vs_true(df)
    _print_table("Table 6: Detailed estimate vs true — all methods", t6)
    _save_table(t6, os.path.join(out_dir, "table6_estimate_vs_true.csv"), "Table 6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_results.csv")
    parser.add_argument("--out",     default="results")
    args = parser.parse_args()
    run_all(args.results, args.out)
