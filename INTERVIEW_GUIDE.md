# Interview Study Guide — A/B Test Analysis Framework

Prep for talking about this project. **Read the code and run the tests
alongside this** — the goal is genuine understanding. A/B testing questions go
deep fast, and the interviewer will follow up.

> Honest framing: *"A project I built to practice and demonstrate A/B testing
> end-to-end — power analysis, the right significance tests, the diagnostics
> that catch bad experiments, and a ship/no-ship decision rubric."* You studied
> the statistics in your MBA; this is where you applied them in code.

---

## 30-second summary

> "It's a framework that walks an experiment through four stages: plan the
> sample size with power analysis, check for pitfalls like sample-ratio-mismatch
> and peeking, run the correct significance test — a two-proportion z-test for
> conversion or Welch's t-test for continuous metrics — then apply a decision
> rubric that combines statistical and practical significance. It's tested,
> including a cross-check of my t-test against SciPy."

---

## Core concepts you MUST be able to explain

### p-value
The probability of seeing an effect at least this extreme **if there were truly
no difference** (H0 true). A small p-value means the data is surprising under
"no effect." It is **not** the probability that H0 is true.

### Statistical power
P(detect an effect | the effect is real). Conventionally 80%. Low power → you
miss real wins. Power rises with sample size and effect size, falls with
variance.

### Why plan sample size *before* the test
If you don't fix N in advance, you're tempted to stop when you see a win —
"peeking" — which inflates the false-positive rate well above your alpha. Sizing
up front keeps the test honest.

### Confidence interval
A 95% CI is a range that would contain the true effect in 95% of repeated
experiments. It's often more useful than the p-value because it shows the
*magnitude and uncertainty* — a CI that includes zero means "could be no effect."

### Type I vs Type II error
Type I = false positive (ship something that doesn't work), rate = alpha.
Type II = false negative (miss a real win), rate = beta = 1 − power.

### Pooled vs unpooled standard error (a detail they may probe)
For the **z-test statistic** we pool the two proportions, because under H0 they're
assumed equal — that's the correct variance estimate for the hypothesis test. For
the **confidence interval** we use the unpooled SE, because we're estimating the
actual observed difference, not assuming equality.

---

## Likely questions and answers

### "Which test did you use and why?"
Two-proportion z-test for binary/conversion metrics; Welch's t-test for
continuous metrics (revenue, time). Welch over Student's because it doesn't
assume equal variances between arms, which rarely holds in practice.

### "What's sample ratio mismatch and why check it?"
If you intended a 50/50 split but observe, say, 50/55, a chi-square test tells you
whether that deviation is beyond chance. SRM usually means broken randomization
or logging — and if assignment is broken, **no** downstream result is
trustworthy. It's a kill-switch check before you even look at the metric.

### "What's wrong with peeking, and how do you fix it?"
Checking repeatedly and stopping at the first significant result inflates the
real false-positive rate (10 looks at alpha=0.05 → ~40% effective FP rate). Fixes:
fix the sample size in advance, or use sequential methods with alpha-spending.

### "Significant but tiny effect — do you ship?"
Not automatically. My decision rubric requires significance **and** an effect
above a practical threshold **and** a CI that excludes no-effect. A real but
trivial lift may not justify engineering cost/risk.

### "How do you know your stats are correct?"
A unit test cross-checks my Welch t-test against
`scipy.stats.ttest_ind(equal_var=False)` to within 1e-6, and other tests verify
the framework detects a known injected effect and stays quiet when there's none.

### "How would you extend this?"
Sequential testing (valid early stopping), CUPED (variance reduction using
pre-experiment data), Bayesian A/B testing (probability-to-beat-control), and
multiple-comparison correction for multi-metric tests. Name these — they signal
you know what production experimentation platforms do.

---

## Do before the interview

1. `pip install -r requirements.txt` then run `python main.py` and
   `python main.py --revenue`.
2. Run `python -m pytest -q` and read the cross-check test.
3. Change the injected effect in `abtest/simulate.py` to zero and watch the
   decision flip to INCONCLUSIVE — understand *why*.
4. Be ready to define p-value, power, and CI in one sentence each, from memory.

## Honesty guardrails
- You studied this in your MBA and built this to apply it — say exactly that.
- Don't claim you ran A/B tests in production at work (you haven't).
- Do claim: you understand experiment design and can analyze a test correctly.
