"""
core/scenarios/scenario_a.py
-----------------------------
Scenario A — internal feature as second dataset.

Observed:
  Dataset A : p(a, z)          protected attribute × proxy
  Dataset B : p(x, z)          internal feature × proxy
  Classifier: p(ŷ | x, z)      known separately

Feasible joint: p(x, z, a)

Metrics available from feasible set:
  DD, DI, DP                           (from P(ŷ=1|a))
  DD|Z=0, DD|Z=1, DI|Z=0, DI|Z=1     (subgroup by proxy)
  DD|X=0, DD|X=1                       (subgroup by feature)
  calibration_gap

Metrics available from baseline (Fréchet LP, differences only):
  DD

EO/TPR disparity NOT available — requires true label y,
which is not in either dataset for Scenario A.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ScenarioA:
    scenario_id:    int
    gt_id:          int
    alpha:          float
    label:          str          # consistent / low / medium / high
    # ground truth
    joint_gt:       dict         # {(x,z,a): prob}
    classifier:     dict         # {(x,z): {0:p, 1:p}}
    # observed marginals (possibly perturbed)
    p_az:           dict         # {(a,z): prob}
    p_xz:           dict         # {(x,z): prob}
    baseline_valid: bool         # True only when alpha == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dirichlet_joint(rng: np.random.Generator) -> dict:
    probs = rng.dirichlet(np.ones(8))
    keys  = [(x, z, a) for x in [0,1] for z in [0,1] for a in [0,1]]
    return dict(zip(keys, probs))


def _marginalise_az(joint: dict) -> dict:
    return {(a, z): sum(joint[(x, z, a)] for x in [0,1])
            for a in [0,1] for z in [0,1]}


def _marginalise_xz(joint: dict) -> dict:
    return {(x, z): sum(joint[(x, z, a)] for a in [0,1])
            for x in [0,1] for z in [0,1]}


def _sample_classifier(rng: np.random.Generator) -> dict:
    return {(x, z): {0: 1 - (p1 := rng.beta(1, 1)), 1: p1}
            for x in [0,1] for z in [0,1]}


def _perturb(marginal: dict, alpha: float, rng: np.random.Generator) -> dict:
    """Mix marginal with Dirichlet noise at level alpha."""
    keys  = list(marginal.keys())
    vals  = np.array([marginal[k] for k in keys])
    noise = rng.dirichlet(np.ones(len(keys)))
    mixed = (1 - alpha) * vals + alpha * noise
    return dict(zip(keys, mixed / mixed.sum()))


def _alpha_to_label(alpha: float) -> str:
    if alpha == 0.0:               return "consistent"
    if alpha <= 0.2:               return "low"
    if alpha <= 0.5:               return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_scenario_a(
    n_ground_truths: int = 50,
    alphas: list[float] | None = None,
    seed: int = 42,
) -> list[ScenarioA]:
    """
    Generate ScenarioA instances.

    For each ground truth we produce one scenario per alpha value.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    rng       = np.random.default_rng(seed)
    scenarios = []
    sc_id     = 0

    for gt_id in range(n_ground_truths):
        joint_gt   = _dirichlet_joint(rng)
        classifier = _sample_classifier(rng)
        p_az_true  = _marginalise_az(joint_gt)
        p_xz_true  = _marginalise_xz(joint_gt)

        for alpha in alphas:
            if alpha == 0.0:
                p_az = dict(p_az_true)
                p_xz = dict(p_xz_true)
            else:
                p_az = _perturb(p_az_true, alpha, rng)
                p_xz = _perturb(p_xz_true, alpha, rng)

            scenarios.append(ScenarioA(
                scenario_id    = sc_id,
                gt_id          = gt_id,
                alpha          = alpha,
                label          = _alpha_to_label(alpha),
                joint_gt       = joint_gt,
                classifier     = classifier,
                p_az           = p_az,
                p_xz           = p_xz,
                baseline_valid = (alpha == 0.0),
            ))
            sc_id += 1

    return scenarios
