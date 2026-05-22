"""
core/baseline.py
----------------
Baseline from Coston et al. "Assessing Algorithmic Fairness with Unobserved
Protected Class Using Data Combination" (Management Science, 2022).

The method bounds fairness metrics of the form:

    E[f(ŷ, Y) | event E, A=1]  −  E[f(ŷ, Y) | event E, A=0]

i.e. DIFFERENCES of conditional expectations. This family includes:
  - Demographic Disparity (DD):  f = 1[ŷ=1],  E = true
  - TPR disparity:               f = 1[ŷ=1],  E = {Y=1}
  - FPR disparity:               f = 1[ŷ=1],  E = {Y=0}
  - TNR/FNR/accuracy parity:     analogous

It does NOT include ratio metrics like Disparate Impact
  DI = P(ŷ=1|A=1) / P(ŷ=1|A=0)
because the ratio of two linear functions of w_z is NOT linear,
so the LP approach does not apply. DI is therefore not computed
by this baseline.

Implementation (matching get_dem_disp_obs_outcomes)
----------------------------------------------------
  p_yz[i] = P(ŷ=1 | Z_i)   per observation
  p_az[i] = P(A=1 | Z_i)   per observation
  w_i     = P(A=1 | Z_i, ŷ_i=1)   UNKNOWN

  Fréchet bounds:
    a_[i] = max((p_az[i] + p_yz[i] - 1) / p_yz[i], 0)
    b_[i] = min(p_az[i] / p_yz[i], 1)

  LP:
    min  Σ_i  w_i · sign · ŷ_i      [sign=-1 for max, +1 for min]
    s.t. a_[i] ≤ w_i ≤ b_[i]

  DD from optimal weights:
    DD = mean(w·ŷ) / mean(p_az)  −  mean((1-w)·ŷ) / (1 - mean(p_az))
"""

from __future__ import annotations
import numpy as np

try:
    import gurobipy as gp
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

_MAX_N_OBS = 900   # 2*n vars total; free Gurobi licence cap ~2000 vars


# ---------------------------------------------------------------------------
# Fréchet bounds (exact from paper)
# ---------------------------------------------------------------------------

def lower_FH(p_yz: np.ndarray, p_az: np.ndarray) -> np.ndarray:
    """max((p_az + p_yz - 1) / p_yz, 0)"""
    safe = np.where(p_yz > 1e-12, p_yz, 1e-12)
    return np.maximum((p_az + p_yz - 1.0) / safe, 0.0)


def upper_FH(p_yz: np.ndarray, p_az: np.ndarray) -> np.ndarray:
    """min(p_az / p_yz, 1)"""
    safe = np.where(p_yz > 1e-12, p_yz, 1e-12)
    return np.minimum(p_az / safe, 1.0)


# ---------------------------------------------------------------------------
# DD from weights (matches compute_bounds_dem_disp_given_weights_p_az_haty)
# ---------------------------------------------------------------------------

def _dd_from_weights(w: np.ndarray, y: np.ndarray,
                     p_az: np.ndarray) -> float:
    p_a1 = float(np.mean(p_az))
    p_a0 = 1.0 - p_a1
    if p_a1 < 1e-12 or p_a0 < 1e-12:
        return np.nan
    return float(np.mean(w * y) / p_a1 - np.mean((1.0 - w) * y) / p_a0)


# ---------------------------------------------------------------------------
# LP (matches get_dem_disp_obs_outcomes, 2-class version)
# ---------------------------------------------------------------------------

def _lp_bound(y, p_az, a_, b_, direction="max", quiet=True):
    n    = len(y)
    m    = gp.Model()
    if quiet:
        m.setParam("OutputFlag", 0)
    w_y1 = [m.addVar(lb=float(a_[i]), ub=float(b_[i])) for i in range(n)]
    w_y0 = [m.addVar(lb=0., ub=1.) for _ in range(n)]   # declared, not in obj
    m.update()
    sign = -1.0 if direction == "max" else 1.0
    obj  = gp.LinExpr()
    for i in range(n):
        obj += w_y1[i] * sign * float(y[i])
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        w = np.array([w_y1[i].X for i in range(n)])
        return _dd_from_weights(w, y, p_az)
    return np.nan


# ---------------------------------------------------------------------------
# Analytic solution for binary Z (exact, no sampling noise)
# ---------------------------------------------------------------------------

def _analytic_dd(p_a1_z, p_yhat1_z, pz):
    """
    For binary Z, DD is linear in w_z with positive coefficients,
    so the optimum is at w_z = upper_FH_z (max) or lower_FH_z (min).
    """
    p_a1 = sum(p_a1_z[z] * pz[z] for z in [0, 1])
    p_a0 = 1.0 - p_a1

    fh_lo = {}
    fh_hi = {}
    for z in [0, 1]:
        pyz = p_yhat1_z[z]
        paz = p_a1_z[z]
        fh_lo[z] = float(max((paz + pyz - 1) / pyz, 0.0)) if pyz > 1e-12 else 0.0
        fh_hi[z] = float(min(paz / pyz, 1.0))             if pyz > 1e-12 else 1.0

    def dd(ws):
        r1 = sum(ws[z] * p_yhat1_z[z] * pz[z] for z in [0, 1]) / p_a1
        r0 = sum((1 - ws[z]) * p_yhat1_z[z] * pz[z] for z in [0, 1]) / p_a0
        return r1 - r0

    return {
        "DD_lo":  float(dd(fh_lo)),
        "DD_hi":  float(dd(fh_hi)),
        "FH_lo":  fh_lo,
        "FH_hi":  fh_hi,
    }


# ---------------------------------------------------------------------------
# Public entry point  —  returns DD bounds only
# ---------------------------------------------------------------------------

def get_baseline_bounds(
    p_az: dict,
    p_xz: dict,
    classifier: dict,
    n_samples: int = 800,
    seed: int = 0,
    quiet: bool = True,
) -> dict:
    """
    Compute Fréchet + LP bounds on Demographic Disparity (DD).

    DI (Disparate Impact) is NOT computed because it is a ratio metric
    and therefore not linear in w_z — the LP approach does not apply.

    Parameters
    ----------
    p_az       : {(a,z): prob}
    p_xz       : {(x,z): prob}
    classifier : {(x,z): {0:p, 1:p}}
    n_samples  : pseudo-observation count for sample-based LP
    seed       : RNG seed

    Returns
    -------
    dict with keys:
      DD_lo, DD_hi   — tight Fréchet bounds on demographic disparity
      FH_lo, FH_hi   — Fréchet bounds on w_z per Z value
      p_yhat1_z      — P(ŷ=1|z) for z ∈ {0,1}
      p_a1_z         — P(A=1|z) for z ∈ {0,1}
    """
    # Derive per-Z quantities
    pz = {z: p_az[(0, z)] + p_az[(1, z)] for z in [0, 1]}

    p_a1_z = {
        z: p_az[(1, z)] / pz[z] if pz[z] > 1e-12 else 0.5
        for z in [0, 1]
    }
    p_x_given_z = {}
    for z in [0, 1]:
        tot = p_xz[(0, z)] + p_xz[(1, z)]
        for x in [0, 1]:
            p_x_given_z[(x, z)] = p_xz[(x, z)] / tot if tot > 1e-12 else 0.5

    # P(ŷ=1|z) = Σ_x P(ŷ=1|x,z)·P(x|z)   [p_yz in paper notation]
    p_yhat1_z = {
        z: sum(classifier[(x, z)][1] * p_x_given_z[(x, z)] for x in [0, 1])
        for z in [0, 1]
    }

    # Try LP (sample-based, matching paper structure)
    if GUROBI_AVAILABLE:
        n = min(n_samples, _MAX_N_OBS)
        rng     = np.random.default_rng(seed)
        Z_obs   = rng.choice([0, 1], size=n, p=[pz[0], pz[1]])
        y_obs   = np.array([rng.binomial(1, p_yhat1_z[z]) for z in Z_obs],
                            dtype=float)
        paz_obs = np.array([p_a1_z[z]    for z in Z_obs])
        pyz_obs = np.array([p_yhat1_z[z] for z in Z_obs])

        a_ = lower_FH(pyz_obs, paz_obs)
        b_ = upper_FH(pyz_obs, paz_obs)

        try:
            dd_hi = _lp_bound(y_obs, paz_obs, a_, b_, "max", quiet)
            dd_lo = _lp_bound(y_obs, paz_obs, a_, b_, "min", quiet)
            analytic = _analytic_dd(p_a1_z, p_yhat1_z, pz)
            return {
                "DD_lo":     float(dd_lo),
                "DD_hi":     float(dd_hi),
                "FH_lo":     analytic["FH_lo"],
                "FH_hi":     analytic["FH_hi"],
                "p_yhat1_z": p_yhat1_z,
                "p_a1_z":    p_a1_z,
            }
        except Exception:
            pass

    # Analytic fallback
    result = _analytic_dd(p_a1_z, p_yhat1_z, pz)
    result.update({"p_yhat1_z": p_yhat1_z, "p_a1_z": p_a1_z})
    return result


# ---------------------------------------------------------------------------
# Scenario B baseline  (classifier is p(ŷ|z), not p(ŷ|x,z))
# ---------------------------------------------------------------------------

def get_baseline_bounds_b(
    p_az: dict,
    classifier_z: dict,   # {z: {0:p, 1:p}}
    n_samples: int = 800,
    seed: int = 0,
    quiet: bool = True,
) -> dict:
    """
    Fréchet + LP bounds on DD for Scenario B.

    classifier_z : {z: {0: P(ŷ=0|z), 1: P(ŷ=1|z)}}
    p_yz is not needed — P(ŷ=1|z) comes directly from the classifier.
    """
    pz = {z: p_az[(0, z)] + p_az[(1, z)] for z in [0, 1]}

    p_a1_z = {
        z: p_az[(1, z)] / pz[z] if pz[z] > 1e-12 else 0.5
        for z in [0, 1]
    }
    p_yhat1_z = {z: classifier_z[z][1] for z in [0, 1]}

    if GUROBI_AVAILABLE:
        n = min(n_samples, _MAX_N_OBS)
        rng     = np.random.default_rng(seed)
        Z_obs   = rng.choice([0, 1], size=n, p=[pz[0], pz[1]])
        y_obs   = np.array([rng.binomial(1, p_yhat1_z[z]) for z in Z_obs],
                            dtype=float)
        paz_obs = np.array([p_a1_z[z]    for z in Z_obs])
        pyz_obs = np.array([p_yhat1_z[z] for z in Z_obs])

        a_ = lower_FH(pyz_obs, paz_obs)
        b_ = upper_FH(pyz_obs, paz_obs)

        try:
            dd_hi = _lp_bound(y_obs, paz_obs, a_, b_, "max", quiet)
            dd_lo = _lp_bound(y_obs, paz_obs, a_, b_, "min", quiet)
            analytic = _analytic_dd(p_a1_z, p_yhat1_z, pz)
            return {
                "DD_lo": float(dd_lo), "DD_hi": float(dd_hi),
                "FH_lo": analytic["FH_lo"], "FH_hi": analytic["FH_hi"],
                "p_yhat1_z": p_yhat1_z, "p_a1_z": p_a1_z,
            }
        except Exception:
            pass

    result = _analytic_dd(p_a1_z, p_yhat1_z, pz)
    result.update({"p_yhat1_z": p_yhat1_z, "p_a1_z": p_a1_z})
    return result


# ---------------------------------------------------------------------------
# Tight Fréchet bounds  (uses p(x,z) and p(ŷ|x,z), not just p(ŷ|z))
# ---------------------------------------------------------------------------

def get_tight_frechet_bounds(
    p_az: dict,
    p_xz: dict,
    classifier: dict,
) -> dict:
    """
    Tighter closed-form bounds on DD using p(a,z), p(x,z), p(ŷ|x,z).

    These apply Fréchet at the (x,z) level rather than the z level.
    With binary x, these are equivalent to the feasible set interval
    endpoints (the feasible set distribution adds shape beyond the bounds).

    Returns dict with keys: tight_DD_lo, tight_DD_hi, tight_DD_width
    """
    pz = {z: p_az[(0,z)] + p_az[(1,z)] for z in [0,1]}
    results = {}
    for a in [0,1]:
        p_a = sum(p_az[(a,z)] for z in [0,1])
        lo_a, hi_a = 0.0, 0.0
        for z in [0,1]:
            paz = p_az[(a,z)] / pz[z] if pz[z] > 1e-12 else 0.5
            tot = p_xz[(0,z)] + p_xz[(1,z)]
            pxz = (p_xz[(1,z)] / tot) if tot > 1e-12 else 0.5

            # Fréchet bounds on P(x=1|a,z)
            p1_lo = max(0.0, pxz + paz - 1.0) / paz if paz > 1e-12 else 0.0
            p1_hi = min(pxz, paz) / paz if paz > 1e-12 else 1.0

            py0 = classifier[(0,z)][1]
            py1 = classifier[(1,z)][1]
            c = py1 - py0
            if c >= 0:
                py_lo = py0 + c * p1_lo
                py_hi = py0 + c * p1_hi
            else:
                py_lo = py0 + c * p1_hi
                py_hi = py0 + c * p1_lo

            w = p_az[(a,z)] / p_a if p_a > 1e-12 else 0.0
            lo_a += py_lo * w
            hi_a += py_hi * w
        results[a] = (lo_a, hi_a)

    dd_lo = results[1][0] - results[0][1]
    dd_hi = results[1][1] - results[0][0]
    return {
        "tight_DD_lo":    float(dd_lo),
        "tight_DD_hi":    float(dd_hi),
        "tight_DD_width": float(dd_hi - dd_lo),
    }
