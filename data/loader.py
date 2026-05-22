"""
data/loader.py
--------------
Load a real dataset CSV and produce ScenarioA / ScenarioB objects
that plug directly into run_experiment_v2.py / run_one_a / run_one_b.

Your CSV format (all columns already binary-encoded integers):
    race, marital-status, relationship, gender, workclass, capital-gain, income
    0,    0,              0,            0,      1,         1,            0

    - income (or equivalent) is y — the true outcome, binary 0/1
    - multi-class columns (e.g. race with values 0,1,2,3) are binarised:
      most common value → 0, everything else → 1

What the loader does
--------------------
1.  Load CSV, binarise any non-binary columns.
2.  Split into three non-overlapping parts:
      train       (train_frac)       — fit the classifier
      dataset_a   (dataset_a_frac)   — compute p(a, z)
      dataset_b   (remainder)        — compute p(x, z) or p(y, z)
3.  Fit logistic regression classifier on the train split:
      Scenario A: p(ŷ | x, z)  features = [x_col, z_col]
      Scenario B: p(ŷ | z)     features = [z_col]
4.  Compute empirical marginals from the respective splits.
5.  Compute ground-truth joint from the FULL dataset.
6.  Return ScenarioA and/or ScenarioB objects.

Usage
-----
    from data.loader import load_real_data

    scenarios_a, scenarios_b = load_real_data(
        path    = "data/real/adult.csv",
        a_col   = "race",
        z_col   = "marital-status",
        x_col   = "workclass",
        y_col   = "income",
    )

    # Feed directly into the v2 experiment runner
    from experiments.v2.run_experiment_v2 import run_one_a, run_one_b
    row_a = run_one_a(scenarios_a[0], num_grid=40)
    row_b = run_one_b(scenarios_b[0], num_grid=40)

Dataset configs
---------------
Adult:
    a_col = "race"             binarised: White=0, non-White=1
    z_col = "marital-status"
    x_col = "workclass"
    y_col = "income"

COMPAS:
    a_col = "race"
    z_col = "relationship"
    x_col = "capital-gain"
    y_col = "income"

German:
    a_col = "gender"
    z_col = "marital-status"
    x_col = "workclass"
    y_col = "income"
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.scenarios.scenario_a import ScenarioA
from core.scenarios.scenario_b import ScenarioB


# ---------------------------------------------------------------------------
# Binarise
# ---------------------------------------------------------------------------

def _binarise(series: pd.Series) -> pd.Series:
    """
    If already 0/1: return as-is.
    Multi-class: most common value → 0, all others → 1.
    """
    vals = set(series.dropna().unique())
    if vals.issubset({0, 1}):
        return series.astype(int)
    majority = series.value_counts().index[0]
    return (series != majority).astype(int)


# ---------------------------------------------------------------------------
# Empirical marginals
# ---------------------------------------------------------------------------

def _joint_az(df, a_col, z_col):
    n = len(df)
    return {(a, z): int(((df[a_col]==a)&(df[z_col]==z)).sum()) / n
            for a in [0,1] for z in [0,1]}

def _joint_xz(df, x_col, z_col):
    n = len(df)
    return {(x, z): int(((df[x_col]==x)&(df[z_col]==z)).sum()) / n
            for x in [0,1] for z in [0,1]}

def _joint_yz(df, y_col, z_col):
    n = len(df)
    return {(y, z): int(((df[y_col]==y)&(df[z_col]==z)).sum()) / n
            for y in [0,1] for z in [0,1]}

def _joint_xza(df, x_col, z_col, a_col):
    n = len(df)
    return {(x, z, a): int(((df[x_col]==x)&(df[z_col]==z)&(df[a_col]==a)).sum()) / n
            for x in [0,1] for z in [0,1] for a in [0,1]}

def _joint_yza(df, y_col, z_col, a_col):
    n = len(df)
    return {(y, z, a): int(((df[y_col]==y)&(df[z_col]==z)&(df[a_col]==a)).sum()) / n
            for y in [0,1] for z in [0,1] for a in [0,1]}


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def _fit_classifier_xz(train, x_col, z_col, y_col):
    """p(ŷ|x,z) → {(x,z): {0:p, 1:p}}"""
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(train[[x_col, z_col]], train[y_col])
    result = {}
    for x in [0, 1]:
        for z in [0, 1]:
            row = pd.DataFrame([[x, z]], columns=[x_col, z_col])
            p1 = float(clf.predict_proba(row)[0, 1])
            result[(x, z)] = {0: 1.0 - p1, 1: p1}
    return result


def _fit_classifier_z(train, z_col, y_col):
    """p(ŷ|z) → {z: {0:p, 1:p}}"""
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(train[[z_col]], train[y_col])
    result = {}
    for z in [0, 1]:
        row = pd.DataFrame([[z]], columns=[z_col])
        p1 = float(clf.predict_proba(row)[0, 1])
        result[z] = {0: 1.0 - p1, 1: p1}
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_real_data(
    path:            str,
    a_col:           str,
    z_col:           str,
    x_col:           str,
    y_col:           str,
    scenario:        str   = "both",   # "A", "B", or "both"
    train_frac:      float = 0.50,
    dataset_a_frac:  float = 0.25,
    # dataset_b = remaining 25%
    seed:            int   = 42,
    name:            str   = "",
) -> tuple[list[ScenarioA], list[ScenarioB]]:
    """
    Returns (scenarios_a, scenarios_b) — each a list with one element.
    alpha is set to 0.0 and baseline_valid=True because real data
    has a single consistent P(Z) across both splits.
    """
    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(path)
    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    df = df[[a_col, z_col, x_col, y_col]].dropna()

    # Binarise every column
    for col in [a_col, z_col, x_col, y_col]:
        df[col] = _binarise(df[col])

    label = name or os.path.splitext(os.path.basename(path))[0]
    print(f"\n  [{label}]  n={len(df)}")
    print(f"    a={a_col}  z={z_col}  x={x_col}  y={y_col}")
    print(f"    P(a=1)={df[a_col].mean():.3f}  "
          f"P(z=1)={df[z_col].mean():.3f}  "
          f"P(x=1)={df[x_col].mean():.3f}  "
          f"P(y=1)={df[y_col].mean():.3f}")

    # ── Split ─────────────────────────────────────────────────────────────
    dataset_b_frac = 1.0 - train_frac - dataset_a_frac
    assert dataset_b_frac > 0.05, (
        f"dataset_b_frac={dataset_b_frac:.2f} too small. "
        "Reduce train_frac or dataset_a_frac.")

    train_df, rest = train_test_split(
        df, test_size=1.0 - train_frac,
        random_state=seed, stratify=df[[a_col, y_col]])

    a_share = dataset_a_frac / (dataset_a_frac + dataset_b_frac)
    df_a, df_b = train_test_split(
        rest, test_size=1.0 - a_share,
        random_state=seed + 1, stratify=rest[[a_col, y_col]])

    print(f"    Split → train={len(train_df)}  "
          f"DatasetA={len(df_a)}  DatasetB={len(df_b)}")

    # ── Ground truth joints (full dataset) ────────────────────────────────
    gt_xza = _joint_xza(df, x_col, z_col, a_col)
    gt_yza = _joint_yza(df, y_col, z_col, a_col)

    # ── Marginals (from their respective splits) ───────────────────────────
    p_az = _joint_az(df_a, a_col, z_col)   # Dataset A
    p_xz = _joint_xz(df_b, x_col, z_col)  # Dataset B — Scenario A
    p_yz = _joint_yz(df_b, y_col, z_col)  # Dataset B — Scenario B

    # ── Classifiers (trained on train split) ──────────────────────────────
    clf_xz = _fit_classifier_xz(train_df, x_col, z_col, y_col)
    clf_z  = _fit_classifier_z(train_df, z_col, y_col)

    # ── Assemble scenario objects ──────────────────────────────────────────
    scenarios_a, scenarios_b = [], []

    if scenario in ("A", "both"):
        scenarios_a.append(ScenarioA(
            scenario_id    = 0,
            gt_id          = 0,
            alpha          = 0.0,
            label          = label,
            joint_gt       = gt_xza,
            classifier     = clf_xz,
            p_az           = p_az,
            p_xz           = p_xz,
            baseline_valid = True,
        ))

    if scenario in ("B", "both"):
        scenarios_b.append(ScenarioB(
            scenario_id    = 0,
            gt_id          = 0,
            alpha          = 0.0,
            label          = label,
            joint_gt       = gt_yza,
            classifier     = clf_z,
            p_az           = p_az,
            p_yz           = p_yz,
            baseline_valid = True,
        ))

    return scenarios_a, scenarios_b


# ---------------------------------------------------------------------------
# Load all three standard datasets
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "adult": dict(a_col="gender",   z_col="marital-status",
                  x_col="capital-gain", y_col="income"),
    "compas": dict(a_col="race",  z_col="score_text",  
                   x_col="priors_count", y_col="two_year_recid"),
    "german": dict(a_col="sex", z_col="housing",
                   x_col="employment-since", y_col="class-label"),
}


def load_all_datasets(
    data_dir: str = "data/real",
    seed:     int = 42,
) -> dict[str, tuple[list[ScenarioA], list[ScenarioB]]]:
    """
    Load all datasets in data_dir that have a matching DATASET_CONFIGS entry.
    Returns {name: (scenarios_a, scenarios_b)}.
    Override DATASET_CONFIGS before calling this if your column names differ.
    """
    results = {}
    for name, cfg in DATASET_CONFIGS.items():
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  Skipping {name} — not found at {path}")
            continue
        try:
            scs_a, scs_b = load_real_data(
                path=path, name=name, seed=seed, **cfg)
            results[name] = (scs_a, scs_b)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    return results