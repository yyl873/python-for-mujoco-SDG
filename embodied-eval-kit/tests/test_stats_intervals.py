"""Intervals: closed forms, defining equations, and simulated coverage.

Three independent ways to be sure an interval is right, none of which re-implements the
same formula:

1. closed forms that hold exactly for extreme counts (``k == 0`` / ``k == n``);
2. the defining equation of the interval, evaluated at its own bounds;
3. Monte-Carlo coverage -- the property the interval is *for*.  An interval that
   contains the truth 99% of the time at the 95% level is not conservative, it is wrong.
"""

from __future__ import annotations

import random

import pytest

from eeval.stats import (
    binom_cdf,
    binom_sf,
    clopper_pearson,
    newcombe_diff_ci,
    proportion,
    wilson,
    wilson_interval,
)


# ----------------------------------------------------------------------------------
# closed forms
# ----------------------------------------------------------------------------------
def test_clopper_pearson_extremes_have_closed_forms():
    alpha = 0.05
    # k == 0: the upper bound is the p that makes (1 - p)^n = alpha / 2
    assert clopper_pearson(0, 12)[1] == pytest.approx(1 - (alpha / 2) ** (1 / 12), abs=1e-9)
    # k == n: the lower bound is the p that makes p^n = alpha / 2
    assert clopper_pearson(40, 40)[0] == pytest.approx((alpha / 2) ** (1 / 40), abs=1e-9)
    assert clopper_pearson(0, 12)[0] == 0.0
    assert clopper_pearson(12, 12)[1] == 1.0


def test_clopper_pearson_is_symmetric_around_a_half():
    for k, n in [(3, 10), (7, 12), (20, 40), (1, 5)]:
        lo, hi = clopper_pearson(k, n)
        lo_m, hi_m = clopper_pearson(n - k, n)
        assert lo == pytest.approx(1 - hi_m, abs=1e-9)
        assert hi == pytest.approx(1 - lo_m, abs=1e-9)


# ----------------------------------------------------------------------------------
# defining equations
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("k,n", [(0, 12), (1, 12), (5, 12), (20, 40), (31, 34), (40, 40)])
def test_clopper_pearson_bounds_solve_the_binomial_tail(k, n):
    """At the returned bounds, the tail probability is exactly alpha / 2."""
    lo, hi = clopper_pearson(k, n, alpha=0.05)
    if k > 0:
        assert binom_sf(k, n, lo) == pytest.approx(0.025, abs=1e-7)
    if k < n:
        assert binom_cdf(k, n, hi) == pytest.approx(0.025, abs=1e-7)


@pytest.mark.parametrize("k,n", [(1, 12), (5, 12), (20, 40), (31, 34), (40, 40)])
def test_wilson_bounds_satisfy_the_score_equation(k, n):
    """Wilson's interval is defined by inverting the score statistic, so at each bound
    ``(p_hat - bound) / sqrt(bound (1 - bound) / n) == +/- 1.96``."""
    point = k / n
    lo, hi = wilson(k, n)
    z = 1.96
    if lo > 0:
        assert (point - lo) / ((lo * (1 - lo) / n) ** 0.5) == pytest.approx(z, abs=1e-6)
    if hi < 1:
        assert (hi - point) / ((hi * (1 - hi) / n) ** 0.5) == pytest.approx(z, abs=1e-6)


def test_wilson_never_reports_false_certainty():
    """40/40 is not 100% -- the old Wald interval would say it was."""
    lo, hi = wilson(40, 40)
    assert hi == 1.0 and 0.90 < lo < 0.92
    assert wilson(0, 76)[0] == 0.0 and 0.0 < wilson(0, 76)[1] < 0.06


def test_wilson_is_symmetric_and_monotone():
    for n in (12, 40):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            lo_m, hi_m = wilson(n - k, n)
            assert lo == pytest.approx(1 - hi_m, abs=1e-9)
            assert hi == pytest.approx(1 - lo_m, abs=1e-9)
            assert lo <= k / n <= hi
    widths = [wilson(5, n)[1] - wilson(5, n)[0] for n in (10, 40, 400)]
    assert widths[0] > widths[1] > widths[2]


def test_proportion_bundle_is_consistent():
    est = proportion(31, 34)
    assert est.point == pytest.approx(31 / 34)
    assert est.wilson == wilson_interval(31, 34)
    data = est.as_dict()
    assert data["point"] == 0.9118 and data["n"] == 34
    assert "Wilson" in str(est)


def test_zero_trials_never_raises():
    assert wilson(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    assert proportion(0, 0).point is None


# ----------------------------------------------------------------------------------
# coverage simulations (seeded, so deterministic)
# ----------------------------------------------------------------------------------
def test_wilson_coverage_is_close_to_nominal():
    rng = random.Random(20240117)
    p_true, n, reps, hits = 0.30, 20, 4000, 0
    for _ in range(reps):
        k = sum(1 for _ in range(n) if rng.random() < p_true)
        lo, hi = wilson(k, n)
        hits += lo <= p_true <= hi
    coverage = hits / reps
    assert 0.93 <= coverage <= 0.985, coverage


def test_clopper_pearson_coverage_is_at_least_nominal():
    rng = random.Random(20240118)
    p_true, n, reps, hits = 0.20, 15, 3000, 0
    for _ in range(reps):
        k = sum(1 for _ in range(n) if rng.random() < p_true)
        lo, hi = clopper_pearson(k, n)
        hits += lo <= p_true <= hi
    coverage = hits / reps
    assert 0.94 <= coverage <= 0.995, coverage


def test_newcombe_diff_interval_covers_the_true_difference():
    rng = random.Random(20240119)
    p1, p2, n, reps, hits = 0.75, 0.25, 12, 3000, 0
    true_diff = p1 - p2
    for _ in range(reps):
        k1 = sum(1 for _ in range(n) if rng.random() < p1)
        k2 = sum(1 for _ in range(n) if rng.random() < p2)
        lo, hi = newcombe_diff_ci(k1, n, k2, n)
        hits += lo <= true_diff <= hi
    coverage = hits / reps
    assert 0.92 <= coverage <= 0.99, coverage


def test_newcombe_contains_the_point_difference_and_respects_bounds():
    lo, hi = newcombe_diff_ci(9, 12, 1, 12)
    assert lo <= 9 / 12 - 1 / 12 <= hi
    assert -1.0 <= lo <= hi <= 1.0
    # identical samples give an interval straddling zero
    lo0, hi0 = newcombe_diff_ci(6, 12, 6, 12)
    assert lo0 < 0.0 < hi0