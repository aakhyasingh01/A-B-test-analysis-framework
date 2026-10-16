"""Generate synthetic experiment data with a known ground-truth effect.

Synthetic data with a *known* true effect is valuable for two reasons:
    1. It lets the framework run end-to-end with zero external data.
    2. It lets us validate the stats — if we inject a 2% lift, a correctly
       powered test should usually detect it, and the eval relies on that.
"""
from __future__ import annotations

import numpy as np


def simulate_conversion_experiment(
    control_rate: float = 0.10,
    treatment_rate: float = 0.12,
    n_per_arm: int = 5000,
    seed: int | None = 42,
) -> dict:
    """Binary conversion outcomes for two arms with a known true lift."""
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, control_rate, n_per_arm)
    treatment = rng.binomial(1, treatment_rate, n_per_arm)
    return {
        "control_conversions": int(control.sum()),
        "control_n": n_per_arm,
        "treatment_conversions": int(treatment.sum()),
        "treatment_n": n_per_arm,
        "true_control_rate": control_rate,
        "true_treatment_rate": treatment_rate,
    }


def simulate_revenue_experiment(
    control_mean: float = 25.0,
    treatment_mean: float = 26.5,
    std_dev: float = 15.0,
    n_per_arm: int = 4000,
    seed: int | None = 42,
) -> dict:
    """Continuous (revenue-per-user) outcomes for two arms."""
    rng = np.random.default_rng(seed)
    # Clip at zero — revenue can't be negative — which also adds realistic skew.
    control = np.clip(rng.normal(control_mean, std_dev, n_per_arm), 0, None)
    treatment = np.clip(rng.normal(treatment_mean, std_dev, n_per_arm), 0, None)
    return {
        "control": control.tolist(),
        "treatment": treatment.tolist(),
        "true_control_mean": control_mean,
        "true_treatment_mean": treatment_mean,
    }
