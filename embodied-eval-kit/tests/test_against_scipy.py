"""Agreement with scipy, asserted without needing scipy installed.

The kit has no runtime dependencies, so it cannot call scipy while testing. Instead the
reference values are generated once by ``tests/tools/make_scipy_reference.py`` (which
refuses to write if the two implementations disagree) and committed under
``tests/fixtures/``. Every CI run therefore checks the hand-written statistics against
scipy's numbers on a machine where scipy is not even installed.

A second, live test runs only where scipy exists; it re-derives the same values instead of
trusting the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeval.stats import (
    binom_two_sided,
    clopper_pearson,
    fisher_test_2x2,
    mcnemar_exact,
    wilson_interval,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scipy_reference.json"


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_records_its_provenance(reference):
    """Without provenance the fixture is just a list of numbers someone could have typed."""
    assert reference["generator"] == "scipy.stats"
    assert reference["scipy_version"]
    assert reference["generated_on"]
    assert reference["tolerance"] <= 1e-8
    assert (Path(__file__).parent / "tools" / "make_scipy_reference.py").exists()
    for key in ("wilson", "clopper_pearson", "binom_two_sided", "fisher_2x2", "mcnemar"):
        assert reference[key], f"{key} 没有对照用例"


def test_wilson_matches_scipy(reference):
    for case in reference["wilson"]:
        low, high = wilson_interval(case["k"], case["n"])
        assert low == pytest.approx(case["low"], abs=reference["tolerance"]), case
        assert high == pytest.approx(case["high"], abs=reference["tolerance"]), case


def test_clopper_pearson_matches_scipy(reference):
    for case in reference["clopper_pearson"]:
        low, high = clopper_pearson(case["k"], case["n"])
        assert low == pytest.approx(case["low"], abs=reference["tolerance"]), case
        assert high == pytest.approx(case["high"], abs=reference["tolerance"]), case


def test_binomial_test_matches_scipy(reference):
    for case in reference["binom_two_sided"]:
        value = binom_two_sided(case["k"], case["n"], case["p"])
        assert value == pytest.approx(case["p_value"], abs=reference["tolerance"]), case


def test_fisher_matches_scipy(reference):
    for case in reference["fisher_2x2"]:
        result = fisher_test_2x2(*case["table"])
        assert result.p_value == pytest.approx(case["p_value"], abs=reference["tolerance"]), case
        if case["odds_ratio"] == case["odds_ratio"]:  # NaN guard for infinite ratios
            assert result.odds_ratio == pytest.approx(case["odds_ratio"],
                                                      rel=1e-9), case


def test_mcnemar_matches_scipy(reference):
    for case in reference["mcnemar"]:
        value = mcnemar_exact(case["b"], case["c"])
        assert value == pytest.approx(case["p_value"], abs=reference["tolerance"]), case


def test_fixture_covers_the_extremes(reference):
    """A cross-check that only tests 5/10 proves nothing about the tails."""
    wilson_pairs = {(case["k"], case["n"]) for case in reference["wilson"]}
    assert (0, 12) in wilson_pairs and (40, 40) in wilson_pairs
    tables = {tuple(case["table"]) for case in reference["fisher_2x2"]}
    assert (1, 11, 9, 3) in tables and (12, 0, 0, 12) in tables


# ---------------------------------------------------------------------------
# live cross-check (skipped wherever scipy is absent -- CI, robot machines)
# ---------------------------------------------------------------------------
def test_live_scipy_agreement(reference):
    stats = pytest.importorskip("scipy.stats")
    for case in reference["wilson"]:
        mine = wilson_interval(case["k"], case["n"])
        theirs = stats.binomtest(case["k"], case["n"]).proportion_ci(0.95, method="wilson")
        assert mine[0] == pytest.approx(theirs.low, abs=1e-9)
        assert mine[1] == pytest.approx(theirs.high, abs=1e-9)
    for a, b, c, d in ((1, 11, 9, 3), (0, 12, 8, 4), (6, 1, 2, 9)):
        assert fisher_test_2x2(a, b, c, d).p_value == pytest.approx(
            float(stats.fisher_exact([[a, b], [c, d]])[1]), abs=1e-9)