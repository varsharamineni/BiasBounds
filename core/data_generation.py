"""
core/data_generation.py
-----------------------
Generate simulation scenarios matching the paper setup.

Inconsistency model
-------------------
Both marginals p(a,z) and p(x,z) are perturbed independently by
mixing with Dirichlet noise at level alpha. This breaks P(Z) consistency
across the two datasets, reflecting real-world data collection mismatch.

Baseline applicability
----------------------
The LP+Fréchet baseline assumes a shared P(Z) across datasets. It is
therefore only valid on CONSISTENT scenarios (alpha=0). For all other
scenarios, only the feasible set method is applied.

Variables
---------
  x  : internal predictive feature   (binary)
  z  : common / linking variable     (binary)
  a  : protected attribute           (binary)
  ŷ  : classifier prediction         (binary)  from p(ŷ|x,z)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Ground truth joint  p(x, z, a)
# ---------------------------------------------------------------------------

def sample_ground_truth_joint(rng: np.random.Generator) -> dict:
    """Sample p(x,z,a) from Dirichlet(1) prior. Keys: (x,z,a)."""
    probs = rng.dirichlet(np.ones(8))
    keys = [(x, z, a) for x in [0,1] for z in [0,1] for a in [0,1]]
    return dict(zip(keys, probs))


def joint_to_p_az(joint: dict) -> dict:
    """Marginalise out x → p(a,z). Keys: (a,z)."""
    return {
        (a, z): sum(joint[(x, z, a)] for x in [0,1])
        for a in [0,1] for z in [0,1]
    }


def joint_to_p_xz(joint: dict) -> dict:
    """Marginalise out a → p(x,z). Keys: (x,z)."""
    return {
        (x, z): sum(joint[(x, z, a)] for a in [0,1])
        for x in [0,1] for z in [0,1]
    }


# ---------------------------------------------------------------------------
# Classifier  p(ŷ | x, z)
# ---------------------------------------------------------------------------

def sample_classifier(rng: np.random.Generator) -> dict:
    """
    Sample p(ŷ|x,z) independently per context (x,z).
    Returns {(x,z): {0: P(ŷ=0|x,z), 1: P(ŷ=1|x,z)}}.
    """
    clf = {}
    for x in [0,1]:
        for z in [0,1]:
            p1 = rng.beta(1, 1)
            clf[(x, z)] = {0: 1 - p1, 1: p1}
    return clf


def ground_truth_p_yhat_given_a(joint: dict, classifier: dict) -> dict:
    """
    Compute true P(ŷ=1|a) by applying classifier to ground truth joint.
    Returns {0: P(ŷ=1|a=0), 1: P(ŷ=1|a=1)}.
    """
    p_yhat_a = {(y, a): 0.0 for y in [0,1] for a in [0,1]}
    for x in [0,1]:
        for z in [0,1]:
            for a in [0,1]:
                p_xza = joint[(x, z, a)]
                for y in [0,1]:
                    p_yhat_a[(y, a)] += p_xza * classifier[(x, z)][y]
    result = {}
    for a in [0,1]:
        p_a = p_yhat_a[(0, a)] + p_yhat_a[(1, a)]
        result[a] = p_yhat_a[(1, a)] / p_a if p_a > 1e-12 else np.nan
    return result


# ---------------------------------------------------------------------------
# Perturbation — perturb the full joint marginals independently
# ---------------------------------------------------------------------------

def perturb_marginal(marginal: dict, alpha: float,
                     rng: np.random.Generator) -> dict:
    """
    Mix marginal with Dirichlet noise at level alpha.
    alpha=0 → unchanged; alpha=1 → fully random.
    Perturbs the full joint (including P(Z)), reflecting real-world
    mismatch where datasets were collected at different times or places.
    """
    keys = list(marginal.keys())
    noise = dict(zip(keys, rng.dirichlet(np.ones(len(keys)))))
    perturbed = {k: (1 - alpha) * marginal[k] + alpha * noise[k]
                 for k in keys}
    total = sum(perturbed.values())
    return {k: v / total for k, v in perturbed.items()}


def kl_divergence(p: dict, q: dict) -> float:
    kl = 0.0
    for k in p:
        if p[k] > 1e-12 and q.get(k, 0) > 1e-12:
            kl += p[k] * np.log(p[k] / q[k])
    return float(kl)


def pz_discrepancy(p_az: dict, p_xz: dict) -> float:
    """
    Measure how much P(Z) disagrees between the two datasets.
    = |P(Z=0 from p_az) - P(Z=0 from p_xz)|
    """
    pz0_az = p_az[(0,0)] + p_az[(1,0)]
    pz0_xz = p_xz[(0,0)] + p_xz[(1,0)]
    return abs(pz0_az - pz0_xz)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

class Scenario:
    def __init__(self, scenario_id, joint_gt, p_az, p_xz, classifier,
                 alpha_az, alpha_xz, kl_az, kl_xz, label):
        self.scenario_id   = scenario_id
        self.joint_gt      = joint_gt
        self.p_az          = p_az
        self.p_xz          = p_xz
        self.classifier    = classifier
        self.alpha_az      = alpha_az
        self.alpha_xz      = alpha_xz
        self.kl_az         = kl_az
        self.kl_xz         = kl_xz
        self.kl_mean       = (kl_az + kl_xz) / 2
        self.label         = label
        self.pz_discrepancy = pz_discrepancy(p_az, p_xz)
        # Baseline is only valid when datasets share P(Z)
        self.baseline_valid = (label == "consistent")

    def __repr__(self):
        return (f"Scenario(id={self.scenario_id}, label={self.label!r}, "
                f"kl={self.kl_mean:.3f}, "
                f"pz_disc={self.pz_discrepancy:.3f})")


def _label(alpha: float) -> str:
    if alpha == 0.0:   return "consistent"
    elif alpha <= 0.2: return "low"
    elif alpha <= 0.5: return "medium"
    else:              return "high"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_scenarios(
    n_ground_truths: int = 100,
    alphas: list[float] | None = None,
    seed: int = 42,
) -> list[Scenario]:
    """
    Generate scenarios with full marginal perturbation.

    For each ground truth:
      alpha=0   → exact marginals, baseline valid
      alpha>0   → both p(a,z) and p(x,z) perturbed independently,
                  breaking P(Z) consistency. Feasible set method only.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    rng = np.random.default_rng(seed)
    scenarios = []
    sid = 0

    for _ in range(n_ground_truths):
        joint_gt   = sample_ground_truth_joint(rng)
        p_az_true  = joint_to_p_az(joint_gt)
        p_xz_true  = joint_to_p_xz(joint_gt)
        classifier = sample_classifier(rng)

        for alpha in alphas:
            if alpha == 0.0:
                p_az_obs = dict(p_az_true)
                p_xz_obs = dict(p_xz_true)
                kl_az = kl_xz = 0.0
            else:
                p_az_obs = perturb_marginal(p_az_true, alpha, rng)
                p_xz_obs = perturb_marginal(p_xz_true, alpha, rng)
                kl_az = kl_divergence(p_az_obs, p_az_true)
                kl_xz = kl_divergence(p_xz_obs, p_xz_true)

            scenarios.append(Scenario(
                scenario_id=sid,
                joint_gt=joint_gt,
                p_az=p_az_obs,
                p_xz=p_xz_obs,
                classifier=classifier,
                alpha_az=alpha,
                alpha_xz=alpha,
                kl_az=kl_az,
                kl_xz=kl_xz,
                label=_label(alpha),
            ))
            sid += 1

    return scenarios
