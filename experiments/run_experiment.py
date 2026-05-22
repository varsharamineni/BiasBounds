"""
experiments/run_experiment.py
------------------------------
Run the full comparison experiment.

Feasible set method : applied to ALL scenarios (consistent + inconsistent).
Baseline method     : applied to CONSISTENT scenarios only, where its
                      shared-P(Z) assumption holds.

The comparison is therefore:
  - Feasible set across the full inconsistency range
  - Baseline as a tight-bounds reference on consistent data
"""

from __future__ import annotations
import sys, os, argparse, time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_generation import generate_scenarios, ground_truth_p_yhat_given_a
from core.joint_solver import solve_feasible_set
from core.baseline import get_baseline_bounds, get_tight_frechet_bounds
from core.metrics import compute_metrics, compute_all_metrics, distributions_over_feasible_set, ALL_METRICS, BASELINE_METRICS


def evaluate_scenario(sc, num_grid: int = 40) -> dict | None:
    try:
        # ── Ground truth ─────────────────────────────────────────────────
        py_a_gt = ground_truth_p_yhat_given_a(sc.joint_gt, sc.classifier)
        # Use compute_all_metrics with full joint to get ground truth for ALL metrics
        j_str = {f"p{x}{z}{a}": v for (x,z,a),v in sc.joint_gt.items()}
        gt = compute_all_metrics(j_str, sc.classifier)

        # ── Feasible set method (always) ─────────────────────────────────
        joints = solve_feasible_set(sc.p_az, sc.p_xz, num=num_grid)
        n_feasible = len(joints)
        if n_feasible == 0:
            return None

        dists = distributions_over_feasible_set(joints, sc.classifier)

        # ── Baseline (consistent scenarios only) ─────────────────────────
        bl = get_baseline_bounds(sc.p_az, sc.p_xz, sc.classifier) \
             if sc.baseline_valid else None

        # ── Build result row ─────────────────────────────────────────────
        row = {
            "scenario_id":    sc.scenario_id,
            "label":          sc.label,
            "alpha":          sc.alpha_az,
            "kl_az":          sc.kl_az,
            "kl_xz":          sc.kl_xz,
            "kl_mean":        sc.kl_mean,
            "pz_discrepancy": sc.pz_discrepancy,
            "n_feasible":     n_feasible,
            "baseline_valid": int(sc.baseline_valid),
        }

        for metric in ALL_METRICS:
            d      = dists[metric]
            gt_val = getattr(gt, metric)

            row[f"{metric}_gt"]          = gt_val
            row[f"{metric}_mean"]        = d.mean
            row[f"{metric}_median"]      = d.median
            row[f"{metric}_std"]         = d.std
            row[f"{metric}_lo"]          = d.lo
            row[f"{metric}_hi"]          = d.hi
            row[f"{metric}_q10"]         = d.q10
            row[f"{metric}_q25"]         = d.q25
            row[f"{metric}_q75"]         = d.q75
            row[f"{metric}_q90"]         = d.q90
            row[f"{metric}_skew"]        = d.skew
            row[f"{metric}_width"]       = d.hi - d.lo
            row[f"{metric}_iqr"]         = d.q75 - d.q25
            row[f"{metric}_n"]           = d.n

            if np.isfinite(gt_val):
                row[f"{metric}_mean_err"]   = abs(d.mean - gt_val)
                row[f"{metric}_median_err"] = abs(d.median - gt_val)
                row[f"{metric}_covered"]    = int(d.lo <= gt_val <= d.hi)
            else:
                row[f"{metric}_mean_err"]   = np.nan
                row[f"{metric}_median_err"] = np.nan
                row[f"{metric}_covered"]    = np.nan

            # Baseline columns — NaN for inconsistent scenarios
            if bl is not None and metric == "DD":
                bl_lo = bl[f"{metric}_lo"] if f"{metric}_lo" in bl else np.nan
                bl_hi = bl[f"{metric}_hi"] if f"{metric}_hi" in bl else np.nan
                bl_width = bl_hi - bl_lo if np.isfinite(bl_hi) else np.inf
                row[f"{metric}_bl_lo"]    = bl_lo
                row[f"{metric}_bl_hi"]    = bl_hi
                row[f"{metric}_bl_width"] = bl_width
                if np.isfinite(gt_val):
                    row[f"{metric}_bl_covered"] = int(bl_lo <= gt_val <= bl_hi)
                else:
                    row[f"{metric}_bl_covered"] = np.nan
                if np.isfinite(bl_width) and bl_width > 1e-9:
                    row[f"{metric}_width_reduction"] = \
                        1.0 - (d.hi - d.lo) / bl_width
                else:
                    row[f"{metric}_width_reduction"] = np.nan
            else:
                for k in ["bl_lo","bl_hi","bl_width","bl_covered","width_reduction"]:
                    row[f"{metric}_{k}"] = np.nan

        # Tight Fréchet bounds on DD (always computable — uses p(x,z) + p(ŷ|x,z))
        try:
            tight = get_tight_frechet_bounds(sc.p_az, sc.p_xz, sc.classifier)
            gt_dd = row.get("DD_gt", np.nan)
            row["tight_DD_lo"]      = tight["tight_DD_lo"]
            row["tight_DD_hi"]      = tight["tight_DD_hi"]
            row["tight_DD_width"]   = tight["tight_DD_width"]
            row["tight_DD_covered"] = int(
                tight["tight_DD_lo"] <= gt_dd <= tight["tight_DD_hi"])                 if np.isfinite(gt_dd) else np.nan
            row["tight_DD_err"] = abs(
                (tight["tight_DD_lo"]+tight["tight_DD_hi"])/2 - gt_dd)                 if np.isfinite(gt_dd) else np.nan
        except Exception:
            for k in ["tight_DD_lo","tight_DD_hi","tight_DD_width",
                      "tight_DD_covered","tight_DD_err"]:
                row[k] = np.nan

        return row

    except Exception:
        return None


def run_experiment(
    n_gt: int = 200,
    num_grid: int = 40,
    seed: int = 42,
    alphas: list[float] | None = None,
    output_path: str = "results/experiment_results.csv",
) -> pd.DataFrame:

    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    print(f"\n{'='*60}")
    print(f"  Fairness Estimation Experiment")
    print(f"  Feasible set : all scenarios")
    print(f"  Baseline LP  : consistent scenarios only")
    print(f"{'='*60}")
    print(f"  Ground truths  : {n_gt}")
    print(f"  Alpha levels   : {alphas}")
    print(f"  Total scenarios: {n_gt * len(alphas)}")
    print(f"  Grid resolution: {num_grid}")
    print(f"{'='*60}\n")

    scenarios = generate_scenarios(n_ground_truths=n_gt, alphas=alphas, seed=seed)

    rows, skipped = [], 0
    t0 = time.time()

    for sc in tqdm(scenarios, desc="Scenarios", ncols=80):
        result = evaluate_scenario(sc, num_grid=num_grid)
        if result is not None:
            rows.append(result)
        else:
            skipped += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Done in {time.time()-t0:.1f}s | "
          f"succeeded={len(rows)} skipped={skipped}")
    print(f"  Saved → {output_path}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gt",     type=int, default=200)
    parser.add_argument("--num_grid", type=int, default=40)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str,
                        default="results/experiment_results.csv")
    args = parser.parse_args()
    df = run_experiment(n_gt=args.n_gt, num_grid=args.num_grid,
                        seed=args.seed, output_path=args.output)
