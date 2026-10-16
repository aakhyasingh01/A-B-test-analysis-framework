"""Core statistical functions for A/B test analysis.

Everything here is implemented from the underlying formulas (with SciPy only for
distribution functions) so the statistics are transparent and auditable rather
than hidden behind a black-box call. Each function documents the test it runs
and the assumptions behind it.

Two families of metric are supported:
    - **Proportion / conversion-rate** metrics (binary outcome per user) ->
      two-proportion z-test.
    - **Continuous** metrics (e.g. revenue per user, time on page) ->
      Welch's two-sample t-test (does not assume equal variances).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """Outcome of a significance test, with everything needed to decide."""
    metric_type: str          # "proportion" or "continuous"
    control_estimate: float    # control mean / rate
    treatment_estimate: float  # treatment mean / rate
    absolute_effect: float     # treatment - control
    relative_effect: float     # (treatment - control) / control
    statistic: float           # z or t statistic
    p_value: float
    ci_low: float              # CI on the absolute effect
    ci_high: float
    alpha: float
    significant: bool

    def summary(self) -> str:
        lines = [
            f"Metric type:        {self.metric_type}",
            f"Control estimate:   {self.control_estimate:.4f}",
            f"Treatment estimate: {self.treatment_estimate:.4f}",
            f"Absolute effect:    {self.absolute_effect:+.4f}",
            f"Relative effect:    {self.relative_effect:+.2%}",
            f"Test statistic:     {self.statistic:.4f}",
            f"p-value:            {self.p_value:.4f}",
            f"{int((1-self.alpha)*100)}% CI (abs):      "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
            f"Significant:        {'YES' if self.significant else 'NO'} "
            f"(alpha={self.alpha})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Proportion / conversion-rate test
# ---------------------------------------------------------------------------
def two_proportion_z_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> TestResult:
    """Two-proportion z-test for comparing conversion rates.

    Uses the *pooled* proportion for the standard error of the test statistic
    (the correct form under H0: p_control == p_treatment), and the *unpooled*
    standard error for the confidence interval on the observed difference.
    """
    if not (0 < control_n and 0 < treatment_n):
        raise ValueError("sample sizes must be positive")

    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    diff = p_t - p_c

    # Pooled SE for the hypothesis test (assumes equal proportions under H0).
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = diff / se_pool if se_pool > 0 else 0.0

    if two_sided:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1 - stats.norm.cdf(z)

    # Unpooled SE for the CI around the observed difference.
    se_unpooled = math.sqrt(
        p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    return TestResult(
        metric_type="proportion",
        control_estimate=p_c,
        treatment_estimate=p_t,
        absolute_effect=diff,
        relative_effect=diff / p_c if p_c else float("nan"),
        statistic=z,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# Continuous metric test (Welch's t-test)
# ---------------------------------------------------------------------------
def welch_t_test(
    control: list[float],
    treatment: list[float],
    alpha: float = 0.05,
    two_sided: bool = True,
) -> TestResult:
    """Welch's two-sample t-test for a continuous metric.

    Welch's variant does NOT assume equal variances, which is the safer default
    for real experiment data where the treatment can change spread as well as
    mean.
    """
    import numpy as np

    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if len(c) < 2 or len(t) < 2:
        raise ValueError("each group needs at least 2 observations")

    mean_c, mean_t = c.mean(), t.mean()
    var_c, var_t = c.var(ddof=1), t.var(ddof=1)
    n_c, n_t = len(c), len(t)
    diff = mean_t - mean_c

    se = math.sqrt(var_c / n_c + var_t / n_t)
    t_stat = diff / se if se > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom.
    df = (var_c / n_c + var_t / n_t) ** 2 / (
        (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    )

    if two_sided:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        p_value = 1 - stats.t.cdf(t_stat, df)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    return TestResult(
        metric_type="continuous",
        control_estimate=mean_c,
        treatment_estimate=mean_t,
        absolute_effect=diff,
        relative_effect=diff / mean_c if mean_c else float("nan"),
        statistic=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant=p_value < alpha,
    )
