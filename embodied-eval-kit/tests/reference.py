"""Independent reference implementations, used to check the float code.

The point of this file is that it shares *no* arithmetic with :mod:`eeval.stats`.
Everything here is computed with exact rationals (``fractions.Fraction``) and integer
binomial coefficients, so a test that compares the two cannot be a tautology the way
"run the same formula twice" would be.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


def exact_binom_pmf(k: int, n: int, p: Fraction) -> Fraction:
    if k < 0 or k > n:
        return Fraction(0)
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def exact_binom_two_sided(k: int, n: int, p: Fraction) -> Fraction:
    """Lehmann ordering: every outcome at most as likely as the observed one."""
    pmf = [exact_binom_pmf(i, n, p) for i in range(n + 1)]
    observed = pmf[k]
    return sum((value for value in pmf if value <= observed), Fraction(0))


def exact_fisher_p(a: int, b: int, c: int, d: int) -> Fraction:
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    denominator = comb(n, col1)

    def prob(x: int) -> Fraction:
        return Fraction(comb(row1, x) * comb(row2, col1 - x), denominator)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return sum((prob(x) for x in range(lo, hi + 1) if prob(x) <= observed), Fraction(0))


def exact_mcnemar(b: int, c: int) -> Fraction:
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum((comb(n, i) for i in range(min(b, c) + 1)), 0)
    return min(Fraction(1), Fraction(2 * tail, 2**n))


def exact_clopper_pearson_lower(k: int, n: int, alpha: Fraction) -> Fraction:
    """Smallest p with ``P(X >= k | p) >= alpha/2``: the exact lower bound."""
    target = alpha / 2
    for denominator in (2000, 20000, 200000):
        for numerator in range(1, denominator):
            p = Fraction(numerator, denominator)
            tail = sum((exact_binom_pmf(i, n, p) for i in range(k, n + 1)), Fraction(0))
            if tail >= target:
                return p
    return Fraction(0)