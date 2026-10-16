"""Power analysis and sample-size planning.

Running a test without planning the sample size first is the single most common
A/B testing mistake — it leads to underpowered tests (can't detect real effects)
or peeking (inflated false positives). These functions answer the two questions
you should ask *before* launching an experiment:

    1. How many users per arm do I need to detect an effect of size X?
    2. Given the sample I have, what effect can I actually detect?
"""
from __future__ import annotations

import math

from scipy import stats


def sample_size_for_proportion(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = True,
) -> int:
    """Required sample size *per arm* for a two-proportion test.

    Args:
        baseline_rate: current conversion rate (control), e.g. 0.10 for 10%.
        min_detectable_effect: smallest effect worth detecting (MDE). If
            ``relative`` is True this is a fraction of the baseline (0.05 = a 5%
            relative lift); otherwise it is an absolute change in rate.
        alpha: significance level (two-sided).
        power: desired statistical power (1 - beta).
        relative: interpret ``min_detectable_effect`` as relative vs absolute.

    Returns:
        Users required per arm (rounded up).
    """
    p1 = baseline_rate
    if relative:
        p2 = p1 * (1 + min_detectable_effect)
    else:
        p2 = p1 + min_detectable_effect

    if not (0 < p2 < 1):
        raise ValueError(f"resulting treatment rate {p2:.4f} is out of (0,1)")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Standard two-proportion sample-size formula.
    pooled = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    n = numerator / denominator
    return math.ceil(n)


def sample_size_for_mean(
    std_dev: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size *per arm* for a two-sample mean comparison.

    Args:
        std_dev: expected standard deviation of the metric.
        min_detectable_effect: smallest absolute difference in means to detect.
    """
    if min_detectable_effect <= 0:
        raise ValueError("min_detectable_effect must be positive")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * std_dev / min_detectable_effect) ** 2
    return math.ceil(n)


def achieved_power_proportion(
    baseline_rate: float,
    treatment_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Post-hoc: what power did a test of this size actually have?"""
    p1, p2 = baseline_rate, treatment_rate
    se = math.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    if se == 0:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    effect_z = abs(p2 - p1) / se
    # Power = P(reject H0 | H1 true), two-sided approximation.
    return float(stats.norm.cdf(effect_z - z_alpha) + stats.norm.cdf(-effect_z - z_alpha))
