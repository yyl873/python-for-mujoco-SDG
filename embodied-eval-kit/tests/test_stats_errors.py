"""Bad input must fail loudly and early -- a silently wrong denominator is worse than a crash."""

from __future__ import annotations

import pytest

from eeval.stats import (
    benjamini_hochberg,
    binom_cdf,
    binom_pmf,
    binom_two_sided,
    clopper_pearson,
    fisher_exact_2x2,
    mcnemar_exact,
    newcombe_diff_ci,
    proportion,
    wilson_interval,
)


@pytest.mark.parametrize("k,n", [(5, 4), (-1, 4), (2.5, 4)])
def test_count_validation(k, n):
    with pytest.raises((TypeError, ValueError)):
        binom_two_sided(k, n)
    with pytest.raises((TypeError, ValueError)):
        proportion(k, n)


def test_negative_trials_rejected():
    with pytest.raises(ValueError):
        wilson_interval(0, -3)


def test_probability_validation():
    with pytest.raises(ValueError):
        binom_pmf(1, 4, 1.5)
    with pytest.raises(ValueError):
        binom_cdf(1, 4, -0.1)
    with pytest.raises(ValueError):
        binom_two_sided(1, 4, 0.0)


def test_alpha_validation():
    with pytest.raises(ValueError):
        wilson_interval(1, 4, alpha=0.0)
    with pytest.raises(ValueError):
        clopper_pearson(1, 4, alpha=1.0)


@pytest.mark.parametrize("table", [(-1, 2, 3, 4), (1, 2, 3, 4.0)])
def test_fisher_rejects_bad_cells(table):
    with pytest.raises((TypeError, ValueError)):
        fisher_exact_2x2(*table)


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_newcombe_rejects_bad_counts():
    with pytest.raises(ValueError):
        newcombe_diff_ci(5, 4, 1, 4)


def test_benjamini_hochberg_rejects_impossible_p_values():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.2, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.2], q=1.5)


def test_out_of_range_counts_are_clamped_only_where_documented():
    """``wilson`` clamps (it is a descriptive helper); the exact test refuses."""
    assert wilson_interval(7, 4)[0] == wilson_interval(4, 4)[0]
    with pytest.raises(ValueError):
        clopper_pearson(7, 4)