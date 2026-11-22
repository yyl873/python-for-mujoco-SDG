#!/usr/bin/env python3
"""Refresh ``tests/fixtures/scipy_reference.json`` -- the numbers scipy produces.

The kit is deliberately dependency-free, so it cannot call scipy at test time.  Instead
the reference values are generated **once**, on a machine that has scipy, and committed:
CI then asserts the hand-written implementations against scipy's numbers without
installing scipy.  This script both writes the fixture and fails if the two disagree, so
a refresh can never silently bake in a regression.

    # needs an interpreter with scipy + numpy (here: the system python)
    env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tests/tools/make_scipy_reference.py

Coverage of the cross-check, and what cannot be cross-checked:

==========================  =========================================================
function                    scipy counterpart
==========================  =========================================================
``wilson``                  ``scipy.stats.binomtest(...).proportion_ci("wilson")``
``clopper_pearson``         ``binomial_ci`` / ``proportion_ci("exact")``
``binom_two_sided``         ``scipy.stats.binomtest(...).pvalue``
``fisher_test_2x2``         ``scipy.stats.fisher_exact`` (p-value and odds ratio)
``mcnemar_exact``           ``binomtest(min(b, c), b + c, 0.5).pvalue`` -- scipy's
                            two-sided exact binomial is already the doubled tail
==========================  =========================================================

Not cross-checked, because scipy (1.8) has no equivalent: ``newcombe_diff_ci`` (verified
by Monte-Carlo coverage instead), ``paired_odds_ratio_interval`` (built from the
Clopper-Pearson numbers above) and ``benjamini_hochberg`` (hand-checked against the
1995 paper's worked example in ``tests/test_stats_paired.py``).
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "scipy_reference.json"
sys.path.insert(0, str(ROOT))

import scipy  # noqa: E402
import scipy.stats as stats  # noqa: E402

from eeval.stats import (  # noqa: E402
    binom_two_sided,
    clopper_pearson,
    fisher_test_2x2,
    mcnemar_exact,
    wilson_interval,
)

TOL = 1e-9

# (k, n) pairs chosen to cover the extremes (0/12, 40/40), an odd n (3/7, 31/34), and a
# large n (120/300) where the naive pmf formula overflows.
WILSON_CASES = [(0, 12), (1, 12), (5, 12), (9, 12), (20, 40), (31, 34), (40, 40), (0, 76),
                (3, 7), (120, 300)]
BINOM_CASES = [(k, n, p) for (k, n) in WILSON_CASES for p in (0.5, 1 / 3)]
FISHER_CASES = [
    (1, 11, 9, 3),     # the real-robot A/B in this project
    (1, 11, 8, 4),     # the same-session control
    (0, 12, 8, 4),
    (12, 0, 0, 12),    # perfect separation: odds ratio is infinite
    (3, 3, 3, 3),      # no effect at all
    (6, 1, 2, 9),
    (0, 0, 3, 4),      # an empty row
    (5, 7, 11, 13),    # a wider, less extreme table
]
MCNEMAR_CASES = [(8, 1), (1, 8), (5, 5), (13, 2), (0, 7), (40, 12), (0, 0)]


def build() -> dict:
    mismatches: list[str] = []

    def check(name: str, mine: float, theirs: float, ctx: dict) -> None:
        if math.isinf(mine) and math.isinf(theirs):
            return
        if abs(mine - theirs) > TOL:
            mismatches.append(f"{name} {ctx}: mine={mine!r} scipy={theirs!r}")

    wilson_cases, cp_cases, binom_cases, fisher_cases, mcnemar_cases = [], [], [], [], []

    for k, n in WILSON_CASES:
        reference = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
        mine = wilson_interval(k, n)
        check("wilson.low", mine[0], reference.low, {"k": k, "n": n})
        check("wilson.high", mine[1], reference.high, {"k": k, "n": n})
        wilson_cases.append({"k": k, "n": n, "low": reference.low, "high": reference.high})

        exact = stats.binomtest(k, n).proportion_ci(0.95, method="exact")
        mine = clopper_pearson(k, n)
        check("cp.low", mine[0], exact.low, {"k": k, "n": n})
        check("cp.high", mine[1], exact.high, {"k": k, "n": n})
        cp_cases.append({"k": k, "n": n, "low": exact.low, "high": exact.high})

    for k, n, p in BINOM_CASES:
        reference = float(stats.binomtest(k, n, p).pvalue)
        mine = binom_two_sided(k, n, p)
        check("binom", mine, reference, {"k": k, "n": n, "p": round(p, 6)})
        binom_cases.append({"k": k, "n": n, "p": p, "p_value": reference})

    for a, b, c, d in FISHER_CASES:
        odds, p_value = stats.fisher_exact([[a, b], [c, d]])
        mine = fisher_test_2x2(a, b, c, d)
        check("fisher.p", mine.p_value, float(p_value), {"table": [a, b, c, d]})
        check("fisher.or", mine.odds_ratio, float(odds), {"table": [a, b, c, d]})
        fisher_cases.append({
            "table": [a, b, c, d],
            "p_value": float(p_value),
            "odds_ratio": float(odds),
        })

    for b, c in MCNEMAR_CASES:
        discordant = b + c
        # scipy's two-sided exact binomial test already sums both tails, which is exactly
        # what McNemar's exact test is on the discordant pairs -- no factor of two here.
        reference = 1.0 if discordant == 0 else float(
            stats.binomtest(min(b, c), discordant, 0.5).pvalue)
        mine = mcnemar_exact(b, c)
        check("mcnemar", mine, reference, {"b": b, "c": c})
        mcnemar_cases.append({"b": b, "c": c, "p_value": reference})

    if mismatches:
        print(f"✗ {len(mismatches)} 处与 scipy 不一致：", file=sys.stderr)
        for line in mismatches[:20]:
            print("  " + line, file=sys.stderr)
        raise SystemExit(1)

    return {
        "generated_by": "tests/tools/make_scipy_reference.py",
        "generator": "scipy.stats",
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "generated_on": date.today().isoformat(),
        "tolerance": TOL,
        "note": (
            "Reference values produced by scipy, committed so the dependency-free test "
            "suite can assert against them. Run the generator to refresh; it refuses to "
            "write if eeval and scipy differ by more than `tolerance`."
        ),
        "wilson": wilson_cases,
        "clopper_pearson": cp_cases,
        "binom_two_sided": binom_cases,
        "fisher_2x2": fisher_cases,
        "mcnemar": mcnemar_cases,
    }


def main() -> int:
    payload = build()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    counts = {key: len(payload[key]) for key in
              ("wilson", "clopper_pearson", "binom_two_sided", "fisher_2x2", "mcnemar")}
    print(f"✓ {FIXTURE.relative_to(ROOT)}  scipy {scipy.__version__}  python "
          f"{platform.python_version()}   cases={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())