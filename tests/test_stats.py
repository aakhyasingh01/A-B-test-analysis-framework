"""Tests for the statistics, including cross-checks against SciPy references."""
import numpy as np
from scipy import stats as scipy_stats

from abtest import (
    make_decision,
    peeking_warning,
    sample_ratio_mismatch,
    sample_size_for_proportion,
    two_proportion_z_test,
    welch_t_test,
)


def test_welch_matches_scipy():
    """Our Welch t-test should match scipy.stats.ttest_ind(equal_var=False)."""
    rng = np.random.default_rng(0)
    c = rng.normal(10, 3, 500)
    t = rng.normal(11, 4, 450)
    result = welch_t_test(c.tolist(), t.tolist())
    ref = scipy_stats.ttest_ind(t, c, equal_var=False)  # note arg order: t - c
    assert abs(result.statistic - ref.statistic) < 1e-6
    assert abs(result.p_value - ref.pvalue) < 1e-6


def test_proportion_detects_real_effect():
    """A large, clear lift should come back significant."""
    result = two_proportion_z_test(1000, 10000, 1200, 10000)
    assert result.significant
    assert result.absolute_effect > 0
    assert result.p_value < 0.05


def test_proportion_no_effect_not_significant():
    """Identical rates should not be significant."""
    result = two_proportion_z_test(1000, 10000, 1005, 10000)
    assert not result.significant


def test_confidence_interval_brackets_effect():
    result = two_proportion_z_test(1000, 10000, 1200, 10000)
    assert result.ci_low < result.absolute_effect < result.ci_high


def test_sample_size_increases_for_smaller_effect():
    big_effect = sample_size_for_proportion(0.10, 0.20)
    small_effect = sample_size_for_proportion(0.10, 0.05)
    assert small_effect > big_effect  # smaller MDE needs more users


def test_srm_flags_broken_split():
    ok, _ = sample_ratio_mismatch(5000, 5000)
    assert ok
    bad_ok, _ = sample_ratio_mismatch(5000, 6000)
    assert not bad_ok


def test_peeking_warns_on_multiple_looks():
    ok, _ = peeking_warning(1)
    assert ok
    bad_ok, _ = peeking_warning(10)
    assert not bad_ok


def test_decision_inconclusive_when_not_significant():
    result = two_proportion_z_test(1000, 10000, 1005, 10000)
    decision = make_decision(result, practical_threshold=0.0)
    assert decision.recommendation == "INCONCLUSIVE"


def test_decision_ship_when_significant_and_meaningful():
    result = two_proportion_z_test(1000, 10000, 1400, 10000)
    decision = make_decision(result, practical_threshold=0.01)
    assert decision.recommendation == "SHIP"


def test_decision_no_ship_when_below_threshold():
    # Significant but tiny effect, with a high practical bar.
    result = two_proportion_z_test(10000, 1000000, 10200, 1000000)
    decision = make_decision(result, practical_threshold=0.05)
    assert decision.recommendation == "DO NOT SHIP"
