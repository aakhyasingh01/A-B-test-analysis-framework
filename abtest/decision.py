"""Ship / no-ship decision logic.

Statistical significance alone is not a decision. A result can be significant but
too small to matter, or a point estimate can look good while its confidence
interval still includes "no effect" or even harm. This module turns a TestResult
plus a business threshold into a clear, defensible recommendation — the part an
interviewer cares about most, because it connects stats to a business call.
"""
from __future__ import annotations

from dataclasses import dataclass

from .stats import TestResult


@dataclass
class Decision:
    recommendation: str   # "SHIP", "DO NOT SHIP", or "INCONCLUSIVE"
    rationale: str


def make_decision(
    result: TestResult,
    practical_threshold: float = 0.0,
    higher_is_better: bool = True,
) -> Decision:
    """Combine statistical significance with practical significance.

    Args:
        result: output of a significance test.
        practical_threshold: the minimum *absolute* effect that is worth
            shipping (accounts for engineering cost, risk, etc.). An effect can
            be statistically real yet below this bar.
        higher_is_better: whether an increase in the metric is the goal.
    """
    effect = result.absolute_effect if higher_is_better else -result.absolute_effect
    ci_low = result.ci_low if higher_is_better else -result.ci_high

    if not result.significant:
        return Decision(
            "INCONCLUSIVE",
            "Result is not statistically significant "
            f"(p={result.p_value:.4f} >= alpha={result.alpha}). "
            "We cannot distinguish the observed effect from noise. "
            "Consider a larger sample or a longer run before deciding.",
        )

    if effect < practical_threshold:
        return Decision(
            "DO NOT SHIP",
            f"Statistically significant but the effect ({effect:+.4f}) is below "
            f"the practical threshold ({practical_threshold:+.4f}). "
            "The change is real but too small to justify shipping.",
        )

    if ci_low <= 0:
        return Decision(
            "DO NOT SHIP",
            "Significant and above threshold, but the confidence interval still "
            f"includes non-positive effects (CI lower bound {ci_low:+.4f}). "
            "There is meaningful risk the true effect is zero or harmful.",
        )

    return Decision(
        "SHIP",
        f"Statistically significant (p={result.p_value:.4f}) and practically "
        f"meaningful (effect {effect:+.4f} exceeds threshold "
        f"{practical_threshold:+.4f}), with a confidence interval that excludes "
        "no-effect. Recommend shipping.",
    )
