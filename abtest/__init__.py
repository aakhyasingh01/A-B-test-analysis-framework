"""A statistically rigorous A/B test analysis framework."""
from .decision import Decision, make_decision
from .diagnostics import peeking_warning, sample_ratio_mismatch
from .power import (
    achieved_power_proportion,
    sample_size_for_mean,
    sample_size_for_proportion,
)
from .stats import TestResult, two_proportion_z_test, welch_t_test

__all__ = [
    "TestResult",
    "two_proportion_z_test",
    "welch_t_test",
    "sample_size_for_proportion",
    "sample_size_for_mean",
    "achieved_power_proportion",
    "make_decision",
    "Decision",
    "sample_ratio_mismatch",
    "peeking_warning",
]
