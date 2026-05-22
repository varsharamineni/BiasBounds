"""
core/joint_solver.py
--------------------
Feasible set solver for the joint distribution p(x, z, a) given two
separate marginal datasets that share a common variable z.

Variables (matching the paper)
-------------------------------
  x : internal feature / predictive attribute  (binary 0/1)
      Observed only in Dataset B alongside z.
  z : common / linking variable                (binary 0/1)
      Observed in BOTH datasets — this is the bridge.
  a : protected attribute                      (binary 0/1)
      Observed only in Dataset A alongside z.

Observed marginals
------------------
  p_az : p(a, z)  — Dataset A  (external: protected attr × common)
  p_xz : p(x, z)  — Dataset B  (internal: feature × common)

Goal
----
  Recover the feasible set of joint distributions p(x, z, a) that are
  consistent with both observed marginals simultaneously. Because z appears
  in both, its marginal p(z) must match across datasets — this is the key
  identifying constraint.

Classifier
----------
  p(ŷ | x, z) — provided separately. After recovering p(x, z, a) we
  marginalise over x and z to obtain p(ŷ | a), which is what fairness
  metrics are computed from.

Joint key convention
--------------------
  Keys are strings 'pXZA' where X, Z, A ∈ {0, 1}.
  e.g.  'p011'  means  P(x=0, z=1, a=1)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_marginal(d: dict, name: str, keys: list, tol: float = 1e-6) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")
    total = sum(d[k] for k in keys)
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name} sums to {total:.6f}, not 1")
    if any(d[k] < -1e-9 for k in keys):
        raise ValueError(f"{name} has negative entries")


def _validate_classifier(clf: dict, tol: float = 1e-6) -> None:
    """
    clf keys: (x, z) -> {0: p(ŷ=0|x,z), 1: p(ŷ=1|x,z)}
    Each row must sum to 1.
    """
    for (x, z), row in clf.items():
        total = row[0] + row[1]
        if abs(total - 1.0) > tol:
            raise ValueError(f"Classifier row (x={x},z={z}) sums to {total:.4f}")


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_feasible_set(
    p_az: dict,
    p_xz: dict,
    num: int = 50,
    tol: float = 1e-9,
) -> list[dict]:
    """
    Enumerate feasible joint distributions p(x, z, a) consistent with
    the two observed marginals p(a, z) and p(x, z).

    The key constraint is that z is shared:
      ∑_a p(a, z) = p(z)  must equal  ∑_x p(x, z) = p(z)

    We derive p(x | z) from p_xz, then use it with p_az to construct
    all valid trivariate joints via two free parameters (c, k).

    Parameters
    ----------
    p_az : dict  Keys (a, z) ∈ {0,1}²
    p_xz : dict  Keys (x, z) ∈ {0,1}²
    num  : int   Grid resolution — higher finds more distributions, slower.
    tol  : float Numerical tolerance.

    Returns
    -------
    list of dicts with string keys 'pXZA', values are probabilities.
    e.g. 'p011' = P(x=0, z=1, a=1)
    """
    az_keys = [(a, z) for a in [0, 1] for z in [0, 1]]
    xz_keys = [(x, z) for x in [0, 1] for z in [0, 1]]
    _validate_marginal(p_az, "p_az", az_keys, tol=1e-5)
    _validate_marginal(p_xz, "p_xz", xz_keys, tol=1e-5)

    # Check z-marginal consistency between datasets
    for z in [0, 1]:
        pz_from_az = p_az[(0, z)] + p_az[(1, z)]
        pz_from_xz = p_xz[(0, z)] + p_xz[(1, z)]
        if abs(pz_from_az - pz_from_xz) > 1e-4:
            # Datasets have inconsistent z-marginals — still attempt solve
            # (this is the "inconsistent marginals" scenario from the paper)
            pass

    # Derive p(x | z) from p_xz  — this is what pb encodes
    p_x_given_z = {}
    for z in [0, 1]:
        total = p_xz[(0, z)] + p_xz[(1, z)]
        if total < tol:
            raise ValueError(f"p_xz marginal for z={z} is zero")
        for x in [0, 1]:
            p_x_given_z[(x, z)] = p_xz[(x, z)] / total

    # Notation: p_az entries
    # p(a=0,z=0), p(a=0,z=1), p(a=1,z=0), p(a=1,z=1)
    pa0z0 = p_az[(0, 0)]
    pa0z1 = p_az[(0, 1)]
    pa1z0 = p_az[(1, 0)]
    pa1z1 = p_az[(1, 1)]

    # p(z=0), p(z=1) from Dataset A
    pz0 = pa0z0 + pa1z0
    pz1 = pa0z1 + pa1z1

    # p(x=0 | z=0), p(x=0 | z=1)
    px0_given_z0 = p_x_given_z[(0, 0)]
    px0_given_z1 = p_x_given_z[(0, 1)]

    # Free parameters:
    #   c = P(x=0, z=0, a=0)    ranges over valid bounds
    #   k = P(x=0, z=1, a=0)    ranges over valid bounds
    #
    # All other 6 entries are determined from c, k via:
    #   P(x=0, z=0, a=1) = p(x=0|z=0)*p(z=0) - c  =  px0_given_z0 * pz0 - c
    #   P(x=1, z=0, a=0) = pa0z0 - c
    #   P(x=1, z=0, a=1) = pa1z0 - (px0_given_z0*pz0 - c)
    #   P(x=0, z=1, a=1) = px0_given_z1*pz1 - k
    #   P(x=1, z=1, a=0) = pa0z1 - k
    #   P(x=1, z=1, a=1) = pa1z1 - (px0_given_z1*pz1 - k)

    c_lo = max(0.0, pa0z0 - 1.0, px0_given_z0 * pz0 - 1.0)
    c_hi = min(1.0, pa0z0, px0_given_z0 * pz0)
    k_lo = max(0.0, pa0z1 - 1.0, px0_given_z1 * pz1 - 1.0)
    k_hi = min(1.0, pa0z1, px0_given_z1 * pz1)

    if c_lo > c_hi + tol or k_lo > k_hi + tol:
        return []

    cs = np.linspace(max(0.0, c_lo), min(1.0, c_hi), num)
    ks = np.linspace(max(0.0, k_lo), min(1.0, k_hi), num)

    valid = []
    for c in cs:
        for k in ks:
            # p{x}{z}{a}
            p000 = c                              # P(x=0, z=0, a=0)
            p001 = px0_given_z0 * pz0 - c        # P(x=0, z=0, a=1)
            p100 = pa0z0 - c                      # P(x=1, z=0, a=0)
            p101 = pa1z0 - p001                   # P(x=1, z=0, a=1)
            p010 = k                              # P(x=0, z=1, a=0)
            p011 = px0_given_z1 * pz1 - k        # P(x=0, z=1, a=1)
            p110 = pa0z1 - k                      # P(x=1, z=1, a=0)
            p111 = pa1z1 - p011                   # P(x=1, z=1, a=1)

            joint = {
                "p000": p000, "p001": p001,
                "p010": p010, "p011": p011,
                "p100": p100, "p101": p101,
                "p110": p110, "p111": p111,
            }

            vals = np.array(list(joint.values()))
            if np.any(vals < -tol):
                continue
            if abs(vals.sum() - 1.0) > tol:
                continue

            # Verify p(x|z) constraint is satisfied
            ok = True
            for z in [0, 1]:
                denom = sum(joint[f"p{x_}{z}{a_}"] for x_ in [0, 1] for a_ in [0, 1])
                if denom < tol:
                    ok = False
                    break
                for x in [0, 1]:
                    numer = sum(joint[f"p{x}{z}{a_}"] for a_ in [0, 1])
                    if abs(numer / denom - p_x_given_z[(x, z)]) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Verify p(a,z) constraint
            paz_check = {
                (0, 0): joint["p000"] + joint["p100"],   # P(a=0,z=0) = sum over x
                (0, 1): joint["p010"] + joint["p110"],   # P(a=0,z=1)
                (1, 0): joint["p001"] + joint["p101"],   # P(a=1,z=0)
                (1, 1): joint["p011"] + joint["p111"],   # P(a=1,z=1)
            }
            if any(abs(paz_check[key] - p_az[key]) > tol for key in az_keys):
                continue

            valid.append(joint)

    return valid


# ---------------------------------------------------------------------------
# Classifier application
# ---------------------------------------------------------------------------

def apply_classifier(
    joint: dict,
    classifier: dict,
) -> dict:
    """
    Given a feasible joint p(x, z, a) and classifier p(ŷ | x, z),
    compute the induced distribution p(ŷ, a) by marginalising over x and z.

    Parameters
    ----------
    joint : dict  Keys 'pXZA' — a single feasible joint from solve_feasible_set.
    classifier : dict
        Keys (x, z) -> {0: P(ŷ=0|x,z), 1: P(ŷ=1|x,z)}
        e.g. {(0,0): {0: 0.7, 1: 0.3}, (0,1): {0:0.4, 1:0.6}, ...}

    Returns
    -------
    dict  Keys (yhat, a) ∈ {0,1}², values = P(ŷ=yhat, a=a)
    """
    p_yhat_a = {(y, a): 0.0 for y in [0, 1] for a in [0, 1]}

    for x in [0, 1]:
        for z in [0, 1]:
            for a in [0, 1]:
                p_xza = joint[f"p{x}{z}{a}"]
                for y in [0, 1]:
                    p_yhat_a[(y, a)] += p_xza * classifier[(x, z)][y]

    return p_yhat_a


def compute_p_yhat_given_a(
    joint: dict,
    classifier: dict,
) -> dict:
    """
    Compute P(ŷ=1 | a) for each group a ∈ {0, 1}.

    Returns
    -------
    dict  {0: P(ŷ=1|a=0), 1: P(ŷ=1|a=1)}
    """
    p_yhat_a = apply_classifier(joint, classifier)

    result = {}
    for a in [0, 1]:
        p_a = p_yhat_a[(0, a)] + p_yhat_a[(1, a)]
        result[a] = p_yhat_a[(1, a)] / p_a if p_a > 1e-12 else np.nan

    return result


# ---------------------------------------------------------------------------
# Baseline: bounds using only the two marginals, no joint recovery
# ---------------------------------------------------------------------------

def baseline_bounds(p_az: dict, classifier: dict) -> dict:
    """
    Compute Fréchet-style bounds on fairness metrics using only p(a,z)
    and the classifier p(ŷ|x,z), without recovering the full joint.

    The baseline can compute P(ŷ=1|z) by averaging the classifier over x
    using only the z-marginal (ignoring the x-dependence), giving a wide
    interval because the x-distribution within each z group is unknown.

    Returns
    -------
    dict with keys 'DI', 'DD' each mapping to (lo, hi) tuples.
    """
    # p(z) from p_az
    pz = {z: p_az[(0, z)] + p_az[(1, z)] for z in [0, 1]}
    # p(a|z)
    p_a_given_z = {}
    for z in [0, 1]:
        total = pz[z]
        for a in [0, 1]:
            p_a_given_z[(a, z)] = p_az[(a, z)] / total if total > 1e-12 else 0.5

    # Without knowing p(x|z), p(ŷ=1|z) can range over all convex combinations
    # of p(ŷ=1|x=0,z) and p(ŷ=1|x=1,z):
    #   p(ŷ=1|z) ∈ [min(clf(0,z), clf(1,z)),  max(clf(0,z), clf(1,z))]
    clf_y1 = {(x, z): classifier[(x, z)][1] for x in [0, 1] for z in [0, 1]}

    p_y1_given_z_lo = {z: min(clf_y1[(0, z)], clf_y1[(1, z)]) for z in [0, 1]}
    p_y1_given_z_hi = {z: max(clf_y1[(0, z)], clf_y1[(1, z)]) for z in [0, 1]}

    # P(ŷ=1|a) = ∑_z P(ŷ=1|z) * P(z|a)
    # P(z|a) = p(a,z) / p(a)
    p_a = {a: sum(p_az[(a, z)] for z in [0, 1]) for a in [0, 1]}

    def p_y1_given_a_range(a):
        p_z_given_a = {z: p_az[(a, z)] / p_a[a] if p_a[a] > 1e-12 else 0.5
                       for z in [0, 1]}
        lo = sum(p_y1_given_z_lo[z] * p_z_given_a[z] for z in [0, 1])
        hi = sum(p_y1_given_z_hi[z] * p_z_given_a[z] for z in [0, 1])
        return lo, hi

    a0_lo, a0_hi = p_y1_given_a_range(0)
    a1_lo, a1_hi = p_y1_given_a_range(1)

    di_lo = a1_lo / a0_hi if a0_hi > 1e-12 else 0.0
    di_hi = a1_hi / a0_lo if a0_lo > 1e-12 else np.inf
    dd_lo = a1_lo - a0_hi
    dd_hi = a1_hi - a0_lo

    return {
        "DI": (max(0.0, di_lo), di_hi),
        "DD": (dd_lo, dd_hi),
        "a0_lo": a0_lo, "a0_hi": a0_hi,
        "a1_lo": a1_lo, "a1_hi": a1_hi,
    }
