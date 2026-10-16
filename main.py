"""End-to-end A/B test analysis walkthrough.

Runs the full workflow the way you would for a real experiment:
    1. PLAN   — compute the sample size needed to detect the target effect.
    2. CHECK  — run pre-analysis diagnostics (SRM, peeking).
    3. TEST   — run the appropriate significance test.
    4. DECIDE — apply the ship / no-ship rubric.

Usage:
    python main.py              # conversion-rate example
    python main.py --revenue    # continuous (revenue) example
"""
from __future__ import annotations

import sys

from abtest import (
    make_decision,
    peeking_warning,
    sample_ratio_mismatch,
    sample_size_for_mean,
    sample_size_for_proportion,
    two_proportion_z_test,
    welch_t_test,
)
from abtest.simulate import (
    simulate_conversion_experiment,
    simulate_revenue_experiment,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_conversion_example() -> None:
    _section("A/B TEST: CONVERSION RATE")

    baseline = 0.10
    mde_relative = 0.15  # want to detect a 15% relative lift (10% -> 11.5%)

    print("\n[1] PLAN — sample size")
    n = sample_size_for_proportion(baseline, mde_relative, power=0.80)
    print(f"  To detect a {mde_relative:.0%} relative lift on a {baseline:.0%} "
          f"baseline at 80% power: {n:,} users per arm.")

    print("\n[2] DATA — simulate experiment (true lift injected)")
    data = simulate_conversion_experiment(
        control_rate=baseline, treatment_rate=0.12, n_per_arm=max(n, 5000)
    )
    print(f"  Control:   {data['control_conversions']:,}/{data['control_n']:,}")
    print(f"  Treatment: {data['treatment_conversions']:,}/{data['treatment_n']:,}")

    print("\n[3] CHECK — pre-analysis diagnostics")
    ok, msg = sample_ratio_mismatch(data["control_n"], data["treatment_n"])
    print(f"  SRM:     {msg}")
    ok, msg = peeking_warning(num_looks=1)
    print(f"  Peeking: {msg}")

    print("\n[4] TEST — two-proportion z-test")
    result = two_proportion_z_test(
        data["control_conversions"], data["control_n"],
        data["treatment_conversions"], data["treatment_n"],
    )
    for line in result.summary().splitlines():
        print(f"  {line}")

    print("\n[5] DECIDE — ship / no-ship")
    # Only worth shipping if we gain at least 1 percentage point absolute.
    decision = make_decision(result, practical_threshold=0.01)
    print(f"  Recommendation: {decision.recommendation}")
    print(f"  Rationale:      {decision.rationale}")


def run_revenue_example() -> None:
    _section("A/B TEST: REVENUE PER USER (CONTINUOUS)")

    print("\n[1] PLAN — sample size")
    n = sample_size_for_mean(std_dev=15.0, min_detectable_effect=1.0, power=0.80)
    print(f"  To detect a $1.00 mean difference (sd=$15) at 80% power: "
          f"{n:,} users per arm.")

    print("\n[2] DATA — simulate experiment")
    data = simulate_revenue_experiment(n_per_arm=max(n, 4000))

    print("\n[3] TEST — Welch's t-test")
    result = welch_t_test(data["control"], data["treatment"])
    for line in result.summary().splitlines():
        print(f"  {line}")

    print("\n[4] DECIDE")
    decision = make_decision(result, practical_threshold=0.50)
    print(f"  Recommendation: {decision.recommendation}")
    print(f"  Rationale:      {decision.rationale}")


if __name__ == "__main__":
    if "--revenue" in sys.argv[1:]:
        run_revenue_example()
    else:
        run_conversion_example()
    print()
