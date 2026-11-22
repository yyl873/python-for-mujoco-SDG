"""Exact small-sample statistics, written from scratch on the standard library.

Every number this module produces ends up in a report that decides whether a robot
policy is shipped or thrown away, so the conventions deliberately follow the exact
tests in R (``binom.test``, ``fisher.test``, ``mcnemar.test``, ``prop.test``) instead
of the normal approximations found in most analysis notebooks.

Nothing outside ``math`` and ``statistics`` is imported.  That is a hard requirement:
the kit has to run on a robot computer or a bare CI runner with no scientific stack.

Conventions
-----------
* ``k`` is the number of successes, ``n`` the number of trials, so the point estimate
  is ``k / n``.
* Two-sided intervals and tests use ``alpha = 0.05`` by default (95%).
* Probabilities are kept in log space (``lgamma`` of the binomial coefficient) and
  summed with ``math.fsum`` before exponentiating; naive ``comb(n, k) * p ** k``
  overflows around ``n = 1000`` and loses all precision in the far tail.
* Exact tests are used even when ``n`` is large.  They cost ``O(n)`` per call, which
  is irrelevant next to the minutes a robotic rollout takes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "BHRResult",
    "FisherResult",
    "Proportion",
    "TwoProportionDiff",
    "benjamini_hochberg",
    "binom_cdf",
    "binom_pmf",
    "binom_sf",
    "binom_two_sided",
    "clopper_pearson",
    "fisher_exact_2x2",
    "fisher_test_2x2",
    "log_binom_coef",
    "mcnemar_exact",
    "newcombe_diff_ci",
    "paired_odds_ratio_interval",
    "proportion",
    "two_proportion_diff",
    "wilson",
    "wilson_interval",
]

# 1.96 is the textbook z for 95%; kept as the default of ``wilson`` so that results
# computed by earlier versions of this file stay comparable.
_Z95 = 1.96


def _z(alpha: float) -> float:
    """Two-sided normal quantile: ``P(|Z| <= z) = 1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _r(value: float | None, ndigits: int) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _check_counts(k: int, n: int) -> None:
    for name, value in (("k", k), ("n", n)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")


# --------------------------------------------------------------------------------------
# binomial building blocks
# --------------------------------------------------------------------------------------
def log_binom_coef(n: int, k: int) -> float:
    """``log C(n, k)`` via ``lgamma``; ``-inf`` outside ``0 <= k <= n``."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_binom_pmf(i: int, n: int, p: float) -> float:
    return log_binom_coef(n, i) + i * math.log(p) + (n - i) * math.log1p(-p)


def binom_pmf(k: int, n: int, p: float) -> float:
    """``P(X = k)`` for ``X ~ Binomial(n, p)``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(_log_binom_pmf(k, n, p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)``.  Sums whichever tail is shorter to limit cancellation."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    if k < n * p:
        return _clip(math.fsum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k + 1)))
    return _clip(1.0 - math.fsum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k + 1, n + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` -- the upper tail, computed with the same tail choice as the CDF."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if k > n * p:
        return _clip(math.fsum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k, n + 1)))
    return _clip(1.0 - binom_cdf(k - 1, n, p))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (Lehmann's minimum-likelihood-ratio ordering).

    This is what R's ``binom.test()`` reports: the total probability of every outcome
    at most as likely as the one observed.  It answers "if the instruction were ignored,
    how often would the landing side look this lopsided by luck alone?".
    """
    _check_counts(k, n)
    if n <= 0:
        return 1.0
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    pmf = [binom_pmf(i, n, p) for i in range(n + 1)]
    observed = pmf[k]
    total = math.fsum(value for value in pmf if value <= observed * (1 + 1e-7))
    return _clip(min(1.0, total))


# --------------------------------------------------------------------------------------
# intervals for a single proportion
# --------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval.

    Unlike Wald, this never returns the fake precision ``[1.0, 1.0]`` for ``k == n``:
    40 successes out of 40 gives a lower bound of 0.912, which is the honest statement.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return (0.0, 1.0)
    k = min(max(k, 0), n)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (_clip(centre - half), _clip(centre + half))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson interval with the normal quantile derived from ``alpha``."""
    return wilson(k, n, _z(alpha))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval, guaranteed coverage at the cost of width.

    Obtained by inverting the binomial tails with a bisection on ``p``:

    * lower bound solves ``P(X >= k | p) = alpha / 2``
    * upper bound solves ``P(X <= k | p) = alpha / 2``

    Both tail probabilities are monotone in ``p``, so bisection converges to machine
    precision in a bounded number of steps without needing the incomplete beta function.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return (0.0, 1.0)
    _check_counts(k, n)
    lower = 0.0 if k == 0 else _solve(binom_sf, k, n, alpha / 2.0, increasing=True)
    upper = 1.0 if k == n else _solve(binom_cdf, k, n, alpha / 2.0, increasing=False)
    return (_clip(lower), _clip(upper))


def _solve(fn, k: int, n: int, target: float, increasing: bool, iterations: int = 200) -> float:
    """Invert a monotone ``fn(k, n, p)`` for ``p`` with bisection on ``[0, 1]``.

    ``increasing`` says which way the tail probability moves with ``p``:
    ``P(X >= k)`` grows with ``p`` while ``P(X <= k)`` shrinks, so the two bounds of
    the Clopper-Pearson interval need opposite bracketing.
    """
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        above = fn(k, n, mid) > target
        if increasing:
            # fn grows with p, so once it overshoots the root sits to the left
            hi, lo = (mid, lo) if above else (hi, mid)
        else:
            lo, hi = (mid, hi) if above else (lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Proportion:
    """A success count with the intervals a report should quote."""

    k: int
    n: int
    wilson: tuple[float, float]
    clopper_pearson: tuple[float, float]

    @property
    def point(self) -> float | None:
        return self.k / self.n if self.n else None

    def as_dict(self, ndigits: int = 4) -> dict:
        return {
            "k": self.k,
            "n": self.n,
            "point": _r(self.point, ndigits),
            "wilson": [_r(v, ndigits) for v in self.wilson],
            "clopper_pearson": [_r(v, ndigits) for v in self.clopper_pearson],
        }

    def __str__(self) -> str:
        point = "n/a" if self.point is None else f"{self.point:.1%}"
        return (
            f"{self.k}/{self.n} = {point} "
            f"(Wilson {self.wilson[0]:.3f}-{self.wilson[1]:.3f}, "
            f"exact {self.clopper_pearson[0]:.3f}-{self.clopper_pearson[1]:.3f})"
        )


def proportion(k: int, n: int, alpha: float = 0.05) -> Proportion:
    """Bundle ``k / n`` with its Wilson and exact intervals."""
    _check_counts(k, n)
    return Proportion(
        k=k,
        n=n,
        wilson=wilson_interval(k, n, alpha),
        clopper_pearson=clopper_pearson(k, n, alpha),
    )


# --------------------------------------------------------------------------------------
# comparing two proportions
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FisherResult:
    """Two-sided Fisher exact test on a 2x2 table ``[[a, b], [c, d]]``."""

    p_value: float
    odds_ratio: float
    table: tuple[tuple[int, int], tuple[int, int]]

    def as_dict(self, ndigits: int = 6) -> dict:
        return {
            "p_value": _r(self.p_value, ndigits),
            "odds_ratio": None if math.isinf(self.odds_ratio) else _r(self.odds_ratio, ndigits),
            "table": [list(row) for row in self.table],
        }


def fisher_test_2x2(a: int, b: int, c: int, d: int) -> FisherResult:
    """Fisher's exact test on ``[[a, b], [c, d]]`` (rows = groups, columns = outcomes).

    Conditional on the margins the table is hypergeometric, so the exact null
    distribution needs no approximation::

        P(X = x) = C(r1, x) * C(r2, c1 - x) / C(n, c1)

    The two-sided p-value sums every table whose probability is at most the observed
    one (the convention of R's ``fisher.test``).  This is the test behind the real-robot
    A/B in this project: 1/12 vs 9/12 grasp success, p = 0.0028.
    """
    for name, value in (("a", a), ("b", b), ("c", c), ("d", d)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    table = ((a, b), (c, d))
    denom_odds = b * c
    odds_ratio = math.inf if denom_odds == 0 else (a * d) / denom_odds

    if n == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col1 == n:
        return FisherResult(p_value=1.0, odds_ratio=odds_ratio, table=table)

    def log_prob(x: int) -> float:
        return (
            log_binom_coef(row1, x)
            + log_binom_coef(row2, col1 - x)
            - log_binom_coef(n, col1)
        )

    lo, hi = max(0, col1 - row2), min(row1, col1)
    observed = math.exp(log_prob(a))
    tail = math.fsum(
        math.exp(log_prob(x))
        for x in range(lo, hi + 1)
        if math.exp(log_prob(x)) <= observed * (1 + 1e-7)
    )
    return FisherResult(p_value=_clip(min(1.0, tail)), odds_ratio=odds_ratio, table=table)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher p-value only (thin wrapper kept for a stable public API)."""
    return fisher_test_2x2(a, b, c, d).p_value


def newcombe_diff_ci(
    k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Newcombe's hybrid score interval for ``p1 - p2`` (independent samples, method 10).

    The Wald interval for a difference of proportions degenerates whenever a cell is 0,
    which is exactly the regime a 12-trial robot A/B lives in.  Newcombe's method
    combines the two Wilson intervals::

        lower = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        upper = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    Reference: Newcombe (1998), *Statistics in Medicine* 17:857-872, method 10.
    """
    _check_counts(k1, n1)
    _check_counts(k2, n2)
    if n1 == 0 or n2 == 0:
        return (-1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, alpha)
    l2, u2 = wilson_interval(k2, n2, alpha)
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


@dataclass(frozen=True)
class TwoProportionDiff:
    """Difference of two independent proportions with its interval and exact test."""

    k1: int
    n1: int
    k2: int
    n2: int
    diff: float
    ci: tuple[float, float]
    p_value: float
    fisher: FisherResult

    def as_dict(self, ndigits: int = 6) -> dict:
        return {
            "group1": {"k": self.k1, "n": self.n1},
            "group2": {"k": self.k2, "n": self.n2},
            "diff": _r(self.diff, ndigits),
            "ci_newcombe": [_r(v, ndigits) for v in self.ci],
            "p_value": _r(self.p_value, ndigits),
            "odds_ratio": None
            if math.isinf(self.fisher.odds_ratio)
            else _r(self.fisher.odds_ratio, ndigits),
        }

    def __str__(self) -> str:
        return (
            f"{self.k1}/{self.n1} vs {self.k2}/{self.n2}: diff {self.diff:+.1%} "
            f"(95% CI {self.ci[0]:+.1%} to {self.ci[1]:+.1%}), Fisher p = {self.p_value:.4f}"
        )


def two_proportion_diff(
    k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05
) -> TwoProportionDiff:
    """Compare two independent success rates the way a robot A/B report should."""
    _check_counts(k1, n1)
    _check_counts(k2, n2)
    fisher = fisher_test_2x2(k1, n1 - k1, k2, n2 - k2)
    return TwoProportionDiff(
        k1=k1,
        n1=n1,
        k2=k2,
        n2=n2,
        diff=(k1 / n1 - k2 / n2) if n1 and n2 else 0.0,
        ci=newcombe_diff_ci(k1, n1, k2, n2, alpha),
        p_value=fisher.p_value,
        fisher=fisher,
    )


# --------------------------------------------------------------------------------------
# paired binary data
# --------------------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test for paired binary outcomes.

    Only the discordant pairs matter: ``b`` pairs where model 1 succeeded and model 2
    failed, ``c`` the reverse.  Under the null the ``b + c`` discordant pairs are a fair
    coin flip, so the p-value is an exact binomial test at ``p = 0.5``.

    Pairing is what makes a paired A/B sensitive: scene difficulty cancels out instead of
    drowning the effect in between-scene variance.
    """
    for name, value in (("b", b), ("c", c)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    discordant = b + c
    if discordant == 0:
        return 1.0
    return _clip(min(1.0, 2.0 * binom_cdf(min(b, c), discordant, 0.5)))


def paired_odds_ratio_interval(b: int, c: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval for the odds ratio of matched pairs, from the discordant count.

    For paired binary data the conditional likelihood of ``b`` given ``b + c`` is
    ``Binomial(b + c, psi / (1 + psi))``, so an exact Clopper-Pearson interval on that
    proportion maps monotonically onto an interval for ``psi``.  (The *difference* of
    paired proportions additionally needs the marginals; report McNemar's p-value for
    the test and this odds ratio for the effect size rather than inventing a difference CI.)
    """
    discordant = b + c
    if discordant == 0:
        return (0.0, math.inf)
    lo, hi = clopper_pearson(b, discordant, alpha)

    def to_odds(prob: float) -> float:
        if prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return math.inf
        return prob / (1.0 - prob)

    return (to_odds(lo), to_odds(hi))


# --------------------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BHRResult:
    """Benjamini-Hochberg step-up result."""

    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    threshold: float

    def as_dict(self, ndigits: int = 6) -> dict:
        return {
            "adjusted": [_r(v, ndigits) for v in self.adjusted],
            "rejected": list(self.rejected),
            "threshold": self.threshold,
        }


def benjamini_hochberg(p_values, q: float = 0.05) -> BHRResult:
    """Benjamini-Hochberg FDR control, for sweeps with many comparisons.

    Ablating four knobs over three seeds is twelve tests; quoting the smallest raw
    p-value out of twelve is how projects lie to themselves.  This returns adjusted
    p-values in the original order plus which hypotheses survive at ``q``.
    """
    values = list(p_values)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q!r}")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {value!r}")
    m = len(values)
    if m == 0:
        return BHRResult(adjusted=(), rejected=(), threshold=q)

    order = sorted(range(m), key=lambda i: values[i])
    adjusted_raw = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running_min = min(running_min, values[index] * m / (rank + 1))
        adjusted_raw[index] = min(1.0, running_min)
    adjusted = tuple(adjusted_raw)
    return BHRResult(adjusted=adjusted, rejected=tuple(a <= q for a in adjusted), threshold=q)
