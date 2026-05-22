"""
core/metrics.py
---------------
Compute fairness metrics from a feasible joint p(x,z,a) + classifier p(ŷ|x,z).

Metrics available to BOTH methods
-----------------------------------
  DD  : Demographic Disparity     = P(ŷ=1|a=1) - P(ŷ=1|a=0)
  DI  : Disparate Impact          = P(ŷ=1|a=1) / P(ŷ=1|a=0)
  DP  : Demographic Parity gap    = |DD|

Metrics available ONLY to the feasible set method
--------------------------------------------------
These require the joint p(x,z,a), which the baseline cannot recover.

  DD_z0, DD_z1 : DD conditioned on Z=0 and Z=1 separately
                 (subgroup fairness within each proxy stratum)
  DI_z0, DI_z1 : DI conditioned on Z

  DD_x0, DD_x1 : DD conditioned on X=0 and X=1
                 (fairness among low- vs high-feature individuals)

  xz_gap       : max over (x,z) of |P(ŷ=1|a=1,x,z) - P(ŷ=1|a=0,x,z)|
                 (worst-case subgroup disparity)

  calibration_gap : |P(x=1|ŷ=1,a=1) - P(x=1|ŷ=1,a=0)|
                    (whether the feature distribution among positives
                     differs by group — proxy for predictive parity)

The baseline can only compute DD, DI, DP because it only bounds
w_z = P(A=1|Z,ŷ=1), which gives P(ŷ=1|A) but nothing conditional
on X or on joint (X,Z) strata.
"""

from __future__ import annotations
import numpy as np
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class MetricValue(NamedTuple):
    # Available to both methods
    DI:   float
    DD:   float
    DP:   float
    # Feasible set only — conditional on Z stratum
    DD_z0: float
    DD_z1: float
    DI_z0: float
    DI_z1: float
    # Feasible set only — conditional on X stratum
    DD_x0: float
    DD_x1: float
    # Feasible set only — worst-case subgroup and calibration
    xz_gap:          float
    calibration_gap: float


class MetricDistribution(NamedTuple):
    metric: str
    values: np.ndarray
    mean:   float
    median: float
    std:    float
    lo:     float
    hi:     float
    q10:    float
    q25:    float
    q75:    float
    q90:    float
    skew:   float
    n:      int


# ---------------------------------------------------------------------------
# Per-joint computation
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else np.nan


def compute_metrics(p_yhat_given_a: dict) -> "MetricValue":
    """
    Compute DD, DI, DP from p_yhat_given_a = {0: P(ŷ=1|a=0), 1: P(ŷ=1|a=1)}.
    Feasible-set-only metrics set to NaN (need the full joint).
    Used for ground truth and baseline comparisons.
    """
    r0 = p_yhat_given_a[0]
    r1 = p_yhat_given_a[1]
    DD = r1 - r0 if not (np.isnan(r0) or np.isnan(r1)) else np.nan
    DI = _safe_div(r1, r0)
    DP = abs(DD) if np.isfinite(DD) else np.nan
    nan = np.nan
    return MetricValue(DI=DI, DD=DD, DP=DP,
                       DD_z0=nan, DD_z1=nan, DI_z0=nan, DI_z1=nan,
                       DD_x0=nan, DD_x1=nan,
                       xz_gap=nan, calibration_gap=nan)


def compute_all_metrics(joint: dict, classifier: dict) -> MetricValue:
    """
    Compute all metrics from a feasible joint p{x}{z}{a} + classifier.

    This is only available to the feasible set method because it requires
    the full joint p(x,z,a).
    """
    # ── Build full joint p(x,z,a,ŷ) ─────────────────────────────────────
    p_xzay = {}
    for x in [0, 1]:
        for z in [0, 1]:
            for a in [0, 1]:
                p_xza = joint[f"p{x}{z}{a}"]
                for yh in [0, 1]:
                    p_xzay[(x, z, a, yh)] = p_xza * classifier[(x, z)][yh]

    def marginal(*keep_dims):
        """Sum out all dims not in keep_dims. keep_dims: subset of (x,z,a,yh)."""
        dim_map = {0: 'x', 1: 'z', 2: 'a', 3: 'yh'}
        result = {}
        for (x, z, a, yh), v in p_xzay.items():
            key = tuple((x, z, a, yh)[d] for d in keep_dims)
            result[key] = result.get(key, 0.0) + v
        return result

    # ── Marginals ─────────────────────────────────────────────────────────
    p_a    = marginal(2)                 # {a: P(a)}
    p_yh_a = marginal(3, 2)             # {(yh,a): P(yh,a)}
    p_z_a  = marginal(1, 2)             # {(z,a): P(z,a)}
    p_x_a  = marginal(0, 2)             # {(x,a): P(x,a)}
    p_yh_z_a = marginal(3, 1, 2)        # {(yh,z,a): P(yh,z,a)}
    p_yh_x_a = marginal(3, 0, 2)        # {(yh,x,a): P(yh,x,a)}
    p_yh_xz_a = marginal(3, 0, 1, 2)   # {(yh,x,z,a): P(yh,x,z,a)}
    p_x_yh_a = marginal(0, 3, 2)        # {(x,yh,a): P(x,yh,a)} for calibration

    def p_yh1_given_a(a_val):
        pa = p_a.get((a_val,), p_a.get(a_val, 0.0))
        num = p_yh_a.get((1, a_val), 0.0)
        return _safe_div(num, pa)

    def p_yh1_given_za(z_val, a_val):
        denom = p_z_a.get((z_val, a_val), 0.0)
        num   = p_yh_z_a.get((1, z_val, a_val), 0.0)
        return _safe_div(num, denom)

    def p_yh1_given_xa(x_val, a_val):
        denom = p_x_a.get((x_val, a_val), 0.0)
        num   = p_yh_x_a.get((1, x_val, a_val), 0.0)
        return _safe_div(num, denom)

    def p_yh1_given_xza(x_val, z_val, a_val):
        denom = p_xzay.get((x_val, z_val, a_val, 0), 0.0) + \
                p_xzay.get((x_val, z_val, a_val, 1), 0.0)
        num   = p_xzay.get((x_val, z_val, a_val, 1), 0.0)
        return _safe_div(num, denom)

    # ── Marginal fairness metrics (also available to baseline) ────────────
    r0  = p_yh1_given_a(0)
    r1  = p_yh1_given_a(1)
    DD  = r1 - r0 if all(np.isfinite([r0, r1])) else np.nan
    DI  = _safe_div(r1, r0)
    DP  = abs(DD) if np.isfinite(DD) else np.nan

    # ── Subgroup DD conditioned on Z (feasible set only) ─────────────────
    def dd_given_z(z_val):
        r0z = p_yh1_given_za(z_val, 0)
        r1z = p_yh1_given_za(z_val, 1)
        return r1z - r0z if all(np.isfinite([r0z, r1z])) else np.nan

    def di_given_z(z_val):
        r0z = p_yh1_given_za(z_val, 0)
        r1z = p_yh1_given_za(z_val, 1)
        return _safe_div(r1z, r0z)

    DD_z0 = dd_given_z(0)
    DD_z1 = dd_given_z(1)
    DI_z0 = di_given_z(0)
    DI_z1 = di_given_z(1)

    # ── Subgroup DD conditioned on X (feasible set only) ─────────────────
    def dd_given_x(x_val):
        r0x = p_yh1_given_xa(x_val, 0)
        r1x = p_yh1_given_xa(x_val, 1)
        return r1x - r0x if all(np.isfinite([r0x, r1x])) else np.nan

    DD_x0 = dd_given_x(0)
    DD_x1 = dd_given_x(1)

    # ── Worst-case subgroup disparity over all (x,z) cells ────────────────
    xz_gaps = []
    for x_val in [0, 1]:
        for z_val in [0, 1]:
            r0xz = p_yh1_given_xza(x_val, z_val, 0)
            r1xz = p_yh1_given_xza(x_val, z_val, 1)
            if all(np.isfinite([r0xz, r1xz])):
                xz_gaps.append(abs(r1xz - r0xz))
    xz_gap = max(xz_gaps) if xz_gaps else np.nan

    # ── Calibration gap: P(x=1|ŷ=1,a) difference ────────────────────────
    # Whether high-feature individuals among predicted-positives differs by group
    def p_x1_given_yh1_a(a_val):
        denom = p_yh_a.get((1, a_val), 0.0)
        num   = p_x_yh_a.get((1, 1, a_val), 0.0)
        return _safe_div(num, denom)

    cal0 = p_x1_given_yh1_a(0)
    cal1 = p_x1_given_yh1_a(1)
    calibration_gap = abs(cal1 - cal0) if all(np.isfinite([cal0, cal1])) else np.nan

    return MetricValue(
        DI=DI, DD=DD, DP=DP,
        DD_z0=DD_z0, DD_z1=DD_z1,
        DI_z0=DI_z0, DI_z1=DI_z1,
        DD_x0=DD_x0, DD_x1=DD_x1,
        xz_gap=xz_gap,
        calibration_gap=calibration_gap,
    )


# ---------------------------------------------------------------------------
# Distribution over feasible set
# ---------------------------------------------------------------------------

def _skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / sigma) ** 3))


def metric_distribution(values: np.ndarray, metric: str) -> MetricDistribution:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return MetricDistribution(metric, v, *([np.nan]*8), 0)
    return MetricDistribution(
        metric=metric, values=v,
        mean=float(v.mean()), median=float(np.median(v)),
        std=float(v.std()), lo=float(v.min()), hi=float(v.max()),
        q10=float(np.percentile(v, 10)), q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)), q90=float(np.percentile(v, 90)),
        skew=_skewness(v), n=len(v),
    )


# All metric field names (matches MetricValue fields)
ALL_METRICS = [
    "DI", "DD", "DP",
    "DD_z0", "DD_z1", "DI_z0", "DI_z1",
    "DD_x0", "DD_x1",
    "xz_gap", "calibration_gap",
]

# Which metrics the baseline can also compute
BASELINE_METRICS = ["DI", "DD", "DP"]

# Which metrics require the full joint (feasible set only)
JOINT_ONLY_METRICS = [m for m in ALL_METRICS if m not in BASELINE_METRICS]


def distributions_over_feasible_set(
    joints: list[dict],
    classifier: dict,
) -> dict[str, MetricDistribution]:
    """
    Compute all metrics for every feasible joint, return distributions.
    """
    collected = {m: [] for m in ALL_METRICS}

    for j in joints:
        mv = compute_all_metrics(j, classifier)
        for m in ALL_METRICS:
            collected[m].append(getattr(mv, m))

    return {
        m: metric_distribution(np.array(collected[m], dtype=float), m)
        for m in ALL_METRICS
    }
