# A/B Test Analysis Framework

A statistically rigorous, well-tested framework for **designing and analyzing
A/B tests** — from sample-size planning through significance testing to a clear
ship / no-ship decision. Written in Python with NumPy and SciPy.

The emphasis is on doing experiments *correctly*: planning power up front,
checking for the pitfalls that silently invalidate results, and separating
statistical significance from the business decision.

---

## Why this exists

Most A/B test mistakes aren't in the final test — they're in everything around
it. This framework encodes the full, disciplined workflow:

| Stage | What it does | Why it matters |
|---|---|---|
| **Plan** | Sample-size & power analysis | Underpowered tests can't detect real effects; this sizes the test *before* launch. |
| **Check** | SRM and peeking diagnostics | A broken 50/50 split or repeated "peeking" invalidates results no matter how good the p-value looks. |
| **Test** | Two-proportion z-test & Welch's t-test | The right test for conversion-rate vs. continuous metrics; Welch doesn't assume equal variances. |
| **Decide** | Ship / no-ship rubric | Combines statistical *and* practical significance + CI direction into a defensible call. |

---

## Quickstart

```bash
git clone https://github.com/aakhyasingh01/A-B-test-analysis-framework.git
cd A-B-test-analysis-framework
pip install -r requirements.txt

python main.py             # conversion-rate walkthrough
python main.py --revenue   # continuous-metric (revenue) walkthrough
python -m pytest -q        # run the test suite
```

### Example output (conversion walkthrough)

```
[1] PLAN — sample size
  To detect a 15% relative lift on a 10% baseline at 80% power: N users per arm.
[3] CHECK — pre-analysis diagnostics
  SRM:     No sample ratio mismatch (p=1.000).
[4] TEST — two-proportion z-test
  Relative effect:    +XX.XX%
  p-value:            0.00XX
  Significant:        YES (alpha=0.05)
[5] DECIDE — ship / no-ship
  Recommendation: SHIP
```

---

## What's inside

```
abtest/
├── stats.py         # two-proportion z-test, Welch's t-test, CIs
├── power.py         # sample-size & power analysis (proportion + mean)
├── decision.py      # ship / no-ship rubric (stat + practical significance)
├── diagnostics.py   # sample-ratio-mismatch (SRM) + peeking warnings
└── simulate.py      # synthetic experiment data with known ground truth
main.py              # end-to-end PLAN -> CHECK -> TEST -> DECIDE walkthrough
tests/test_stats.py  # unit tests incl. cross-check vs. scipy reference
```

---

## Statistical notes

- **Two-proportion z-test** uses the *pooled* standard error for the test
  statistic (correct under H0) and the *unpooled* SE for the confidence interval
  on the observed difference.
- **Welch's t-test** uses the Welch–Satterthwaite degrees of freedom and does not
  assume equal variances — the safer default for real experiment data. A unit
  test cross-checks it against `scipy.stats.ttest_ind(equal_var=False)`.
- **Power / sample size** uses the standard normal-approximation formulas for a
  two-sided test at a configurable alpha and power.
- **Decision logic** requires *all three* of: statistical significance, an effect
  above a practical threshold, and a confidence interval that excludes no-effect.

---

## Possible extensions

- Sequential testing / alpha-spending (so early looks are statistically valid).
- CUPED variance reduction using pre-experiment covariates.
- Bayesian A/B testing (posterior on the lift, probability-to-beat-control).
- Multiple-comparison correction for multi-metric or multi-variant tests.

---

## License

MIT — see `LICENSE`.
