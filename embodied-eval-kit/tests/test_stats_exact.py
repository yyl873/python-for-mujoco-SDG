"""The float implementations are checked against exact rational arithmetic.

These are the tests that make the statistics claim real: if ``binom_two_sided`` were a
copy-paste of a remembered formula, or if the log-space sum dropped a term in the
tail, the exact reference below would disagree immediately.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from reference import (
    exact_binom_two_sided,
    exact_fisher_p,
    exact_mcnemar,
)

from eeval.stats import binom_pmf, binom_two_sided, fisher_exact_2x2, mcnemar_exact

BINOM_CASES = [
    (0, 10, Fraction(1, 2)),
    (5, 10, Fraction(1, 2)),
    (7, 10, Fraction(1, 2)),
    (10, 10, Fraction(1, 2)),
    (3, 9, Fraction(1, 3)),
    (11, 20, Fraction(2, 5)),
    (1, 12, Fraction(1, 2)),
    (2, 12, Fraction(5, 12)),
    (31, 34, Fraction(1, 2)),
    (0, 76, Fraction(1, 2)),
]


@pytest.mark.parametrize("k,n,p", BINOM_CASES)
def test_binom_two_sided_matches_exact_rational(k, n, p):
    exact = float(exact_binom_two_sided(k, n, p))
    assert binom_two_sided(k, n, float(p)) == pytest.approx(exact, abs=1e-10)


@pytest.mark.parametrize("k,n,p", BINOM_CASES)
def test_binom_pmf_matches_exact_rational(k, n, p):
    assert binom_pmf(k, n, float(p)) == pytest.approx(float(_pmf(k, n, p)), rel=1e-12)


def _pmf(k, n, p):
    from fractions import Fraction as F
    from math import comb

    return comb(n, k) * p**k * (1 - p) ** (n - k)


@pytest.mark.parametrize(
    "table",
    [
        (1, 11, 9, 3),
        (1, 11, 8, 4),
        (4, 8, 9, 3),
        (0, 12, 5, 7),
        (12, 0, 0, 12),
        (3, 3, 3, 3),
        (6, 1, 2, 9),
        (0, 0, 3, 4),
    ],
)
def test_fisher_matches_exact_rational(table):
    a, b, c, d = table
    assert fisher_exact_2x2(a, b, c, d) == pytest.approx(float(exact_fisher_p(a, b, c, d)), abs=1e-12)


@pytest.mark.parametrize("b,c", [(8, 1), (1, 8), (5, 5), (13, 2), (0, 7), (40, 12), (0, 0)])
def test_mcnemar_matches_exact_rational(b, c):
    assert mcnemar_exact(b, c) == pytest.approx(float(exact_mcnemar(b, c)), abs=1e-12)


def test_fisher_is_symmetric_under_transposition():
    """Swapping rows (or columns) describes the same table, so p must not move."""
    base = fisher_exact_2x2(1, 11, 9, 3)
    assert fisher_exact_2x2(9, 3, 1, 11) == pytest.approx(base)  # rows swapped
    assert fisher_exact_2x2(1, 9, 11, 3) == pytest.approx(base)  # columns swapped
    assert fisher_exact_2x2(11, 1, 3, 9) == pytest.approx(base)  # both


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(8, 1) == pytest.approx(mcnemar_exact(1, 8))


def test_tails_do_not_underflow_to_zero():
    """A log-space sum must survive a one-in-a-million tail (naive float code does not)."""
    assert 0.0 < binom_two_sided(40, 40, 0.5) < 1e-9
    assert 0.0 < fisher_exact_2x2(0, 30, 30, 0) < 1e-12
    assert binom_pmf(200, 400, 0.5) > 0.0