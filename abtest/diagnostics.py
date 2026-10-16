"""Pre-analysis diagnostics — catch the pitfalls that invalidate A/B tests.

Interviewers love to probe whether you check these *before* trusting a p-value.
Each function returns a (passed, message) pair.
"""
from __future__ import annotations

from scipy import stats


def sample_ratio_mismatch(
    control_n: int,
    treatment_n: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.001,
) -> tuple[bool, str]:
    """Sample Ratio Mismatch (SRM) check.

    If users were meant to be split 50/50 but the observed split differs more
    than chance allows, the randomization or logging is broken and the whole
    experiment is suspect — regardless of how good the p-value looks. Tested
    with a chi-square goodness-of-fit at a strict alpha.
    """
    total = control_n + treatment_n
    expected_c = total * expected_ratio
    expected_t = total * (1 - expected_ratio)
    chi2 = ((control_n - expected_c) ** 2 / expected_c
            + (treatment_n - expected_t) ** 2 / expected_t)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    if p < alpha:
        return False, (
            f"SRM DETECTED (p={p:.5f}): observed split "
            f"{control_n}/{treatment_n} deviates from expected "
            f"{expected_ratio:.0%}/{1-expected_ratio:.0%}. "
            "Investigate randomization/logging before trusting results."
        )
    return True, f"No sample ratio mismatch (p={p:.3f})."


def peeking_warning(num_looks: int, alpha: float = 0.05) -> tuple[bool, str]:
    """Warn about inflated false-positive rate from repeated 'peeking'.

    Checking significance repeatedly and stopping at the first 'win' inflates the
    real false-positive rate well above alpha. This returns an approximate
    inflated rate to make the danger concrete.
    """
    if num_looks <= 1:
        return True, "Single analysis at a fixed sample size — no peeking inflation."
    # Rough upper-bound approximation of inflated type-I error.
    inflated = 1 - (1 - alpha) ** num_looks
    return False, (
        f"PEEKING RISK: with {num_looks} looks, the effective false-positive "
        f"rate is ~{inflated:.1%} (nominal alpha={alpha}). Use a fixed sample "
        "size or a sequential-testing correction (e.g. alpha spending)."
    )
